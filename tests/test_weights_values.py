"""`backtest_weights`: the bookkeeping that the mutation runs showed unchecked: borrow cost, yearly returns, the argument rules for capital, and the settings written to the result and the ledger.

WW1 borrow        a constant short pays borrow_bp times its size, as a number and as a panel of rates
WW2 yearly        the yearly figures chain the bars of each calendar year
WW3 arguments     capital must be positive; a minimum trade value or a lot size without capital is refused
WW4 settings      spec and ledger configuration: spread and borrow (number or panel), funding on only when asked for and present, execution settings by number or by security
"""
from __future__ import annotations
import sys, tempfile, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from test_argument_checks import _raises
from test_synthetic import make_panel


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _setup(n_days=300, n_stocks=24, seed=3, funding=False):
    p, _ = make_panel(n_days=n_days, n_stocks=n_stocks, seed=seed)
    if funding:
        p = q.Panel(close=p.close, eligible=p.eligible, volume=p.volume, mkt_cap=p.mkt_cap, funding=pd.DataFrame(0.0001, index=p.dates, columns=p.tickers), market="T", entry_lag=1)
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    W.iloc[70:, 0] = -0.5
    W.iloc[70:, 1] = 0.5
    return p, W


def test_borrow_and_yearly():
    p, W = _setup()
    kw = dict(benchmark=None, check_universe=False)
    base = _quiet(lambda: q.backtest_weights(p, W, **kw))
    r = _quiet(lambda: q.backtest_weights(p, W, borrow_bp=100.0, **kw))
    cut = len(p.dates) - (p.entry_lag + 1)
    short = np.maximum(-r.holdings, 0.0)
    want_bp = (short[:cut].sum(axis=1) * 100.0 / 1e4 / p.periods_per_year).mean() * p.periods_per_year * 1e4
    assert abs(r.metrics["borrow_annual_bp"] - want_bp) < 1e-9 and want_bp > 20
    diff = (base.net_returns - r.net_returns).to_numpy()
    assert np.allclose(diff, short[:cut].sum(axis=1) * 100.0 / 1e4 / p.periods_per_year, atol=1e-15)
    rate = pd.DataFrame(100.0, index=p.dates, columns=p.tickers); rate.iloc[:, 0] = 300.0; rate.iloc[5, 3] = np.nan
    r2 = _quiet(lambda: q.backtest_weights(p, W, borrow_bp=rate, **kw))
    d2 = (base.net_returns - r2.net_returns).to_numpy()
    assert np.allclose(d2, short[:cut, 0] * 300.0 / 1e4 / p.periods_per_year + short[:cut, 1:].sum(axis=1) * 100.0 / 1e4 / p.periods_per_year, atol=1e-15)
    for y, v in r.yearly.items():
        g = r.net_returns[r.net_returns.index.year == int(y)]
        assert abs(v - ((1 + g).prod() - 1)) < 1e-12
    assert len(r.yearly) >= 2
    print("WW1/2 borrow cost equals borrow_bp times the short size (number and panel of rates); yearly returns chain the bars of the year  PASS")


def test_capital_arguments():
    p, W = _setup()
    kw = dict(benchmark=None, check_universe=False)
    price = p.close
    _raises(lambda: q.backtest_weights(p, W, capital=0.0, price=price, **kw), "positive")
    _raises(lambda: q.backtest_weights(p, W, capital=-5.0, price=price, **kw), "positive")
    _raises(lambda: q.backtest_weights(p, W, min_trade_value=0.5, **kw), "need capital")
    _raises(lambda: q.backtest_weights(p, W, lot=10.0, **kw), "need capital")
    _raises(lambda: q.backtest_weights(p, W, capital=1e6, **kw), "needs price")
    _quiet(lambda: q.backtest_weights(p, W, capital=1e6, price=price, **kw))
    _quiet(lambda: q.backtest_weights(p, W, min_trade_value=0.0, lot=1.0, **kw))                 # the do-nothing settings need no capital
    print("WW3 capital must be positive and priced; a minimum trade value or a lot size alone is refused, the neutral values are not  PASS")


def test_settings():
    p, W = _setup(funding=True)
    p0, _ = _setup(funding=False)
    kw = dict(benchmark=None, check_universe=False)
    with tempfile.TemporaryDirectory(prefix="ww4-") as d:
        led = q.Ledger(d)
        r1 = _quiet(lambda: q.backtest_weights(p, W, spread_bp=7.0, borrow_bp=50.0, ledger=led, family="a", name="a", **kw))
        spread = pd.DataFrame(5.0, index=p.dates, columns=p.tickers)
        r2 = _quiet(lambda: q.backtest_weights(p, W, spread_bp=spread, borrow_bp=pd.DataFrame(10.0, index=p.dates, columns=p.tickers), funding=False, ledger=led, family="b", name="b", **kw))
        r3 = _quiet(lambda: q.backtest_weights(p0, W, funding=True, ledger=led, family="c", name="c", **kw))
        price = p.close
        r4 = _quiet(lambda: q.backtest_weights(p, W, capital=1e6, price=price, lot=1.0, min_trade_value=0.0, freeze_days=5, freeze_return=-0.1, cap_gross=True, ledger=led, family="d", name="d", **kw))
        mtv = pd.Series(10.0, index=p.tickers)
        r5 = _quiet(lambda: q.backtest_weights(p, W, capital=1e6, price=price, lot=pd.Series(10.0, index=p.tickers), min_trade_value=mtv, ledger=led, family="e", name="e", **kw))
        cfg = {k: led.trials(k)[0]["config"] for k in "abcde"}
    assert r1.spec["spread"] == "7.0bp round trip" and r1.spec["borrow_bp"] == 50.0 and r2.spec["spread"] == "panel" and r2.spec["borrow_bp"] == "panel"
    assert (r1.spec["funding"], r2.spec["funding"], r3.spec["funding"]) == (True, False, False)
    assert cfg["a"]["spread_bp"] == 7.0 and isinstance(cfg["b"]["spread_bp"], str) and cfg["a"]["funding"] is True and cfg["b"]["funding"] is False and cfg["c"]["funding"] is False
    assert cfg["d"]["capital"] == 1e6 and cfg["d"]["min_trade_value"] == 0.0 and cfg["d"]["lot"] == 1.0 and cfg["d"]["freeze_days"] == 5 and cfg["d"]["freeze_return"] == -0.1 and cfg["d"]["cap_gross"] is True
    assert "capital" not in cfg["a"] and "cap_gross" not in cfg["a"] and "freeze_days" not in cfg["a"]
    assert isinstance(cfg["e"]["min_trade_value"], str) and isinstance(cfg["e"]["lot"], str)
    assert r4.spec["execution"]["min_trade_value"] == 0.0 and r5.spec["execution"]["min_trade_value"] == "by security" and r1.spec["execution"] is None
    print("WW4 spec and ledger configuration record spread, borrow, funding and the execution settings as given  PASS")


if __name__ == "__main__":
    test_borrow_and_yearly(); test_capital_arguments(); test_settings()
    print("weights value tests: all passed")
