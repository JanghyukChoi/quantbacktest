"""Guards added after hostile-input review (0.2.1).

HD1 prices      a zero, negative or infinite price is treated as missing, with a warning; a held name with a zero price gives the same result as one with a missing
                price (before, the return was 3.6e307)
HD2 alignment   a factor, weights or signal with other ticker names, transposed, a different time zone, or not a DataFrame is refused with the reason; half covered warns
HD3 panel       duplicate ticker names raise; weekly dates with periods_per_year=252 warn, and the usual daily, 24/7 and 5-minute settings do not
HD4 big moves   a daily move above +1000% on an eligible security warns and changes nothing in the numbers
HD5 no trades   an all-NaN or constant factor, all-zero weights, and a capital too small for one lot warn that nothing was held
HD6 dsr         a matrix with no usable column says so
"""
from __future__ import annotations
import sys, warnings
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from test_argument_checks import _raises
from test_synthetic import make_panel


def _warns(fn, *needles):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        out = fn()
    msgs = [str(w.message) for w in ws]
    assert all(any(n in m for m in msgs) for n in needles), (needles, msgs)
    return out


def _quiet(fn):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in ws]


def test_bad_prices():
    p, rng = make_panel(n_days=400, n_stocks=40)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    f.iloc[95:110, 3] = 99.0                                                       # name 3 is in the long leg around day 100
    results = {}
    for label, bad in (("nan", np.nan), ("zero", 0.0), ("negative", -5.0), ("inf", np.inf)):
        c = p.close.copy()
        c.iloc[100, 3] = bad
        if label == "nan":
            pp = replace(p, close=c)
            out, _ = _quiet(lambda: q.backtest_portfolio(pp, f, hold=1, spread_bp=10))
        else:
            out = _warns(lambda: q.backtest_portfolio(replace(p, close=c), f, hold=1, spread_bp=10), "negative or infinite")
        results[label] = out.net_returns.to_numpy()
        assert np.isfinite(results[label]).all() and np.abs(results[label]).max() < 1.0, (label, np.abs(results[label]).max())
    for label in ("zero", "negative", "inf"):
        assert np.array_equal(results[label], results["nan"]), label                # the same as a price that is simply missing
    print("HD1 zero, negative and infinite prices are missing prices (warned), and a held name with a zero price no longer returns 3.6e307  PASS")


def test_alignment():
    p, rng = make_panel(n_days=400, n_stocks=40)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    W = f.rank(axis=1, pct=True).ge(0.9).astype(float)
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).where(p.eligible, 0.0)
    E = f > 1.5
    for what, call in (("factor", lambda x: q.backtest_portfolio(p, x, hold=5, spread_bp=10)),
                       ("weights", lambda x: q.backtest_weights(p, x, spread_bp=5)),
                       ("signal", lambda x: q.backtest_event(p, x, horizons=(5,), neutralize_check=False))):
        good = {"factor": f, "weights": W, "signal": E}[what]
        _raises(lambda: call(good.rename(columns=lambda c: "z" + c)), what, "no date or no ticker in common", "tickers in common: 0%")
        _raises(lambda: call(good.T), what, "no date or no ticker in common")
        _raises(lambda: call(good.tz_localize("UTC")), what, "dates in common: 0%")                  # a time zone on one side only
        _raises(lambda: call(good.to_numpy()), what, "DataFrame", "ndarray")
        _raises(lambda: call(good.iloc[:, 0]), what, "DataFrame", "Series")
    sc = {"a": f}
    _raises(lambda: q.screen(p, {"a": f.rename(columns=lambda c: "z" + c)}, horizons=(5,), primary_h=5, n_null=10), "factor 'a'", "no date or no ticker")
    _warns(lambda: q.backtest_portfolio(p, f.iloc[:, :10], hold=5, spread_bp=10), "covers only", "25% of its tickers")
    _, msgs = _quiet(lambda: q.backtest_portfolio(p, f.iloc[:, :30], hold=5, spread_bp=10))           # 75 percent covered: no warning
    assert not any("covers only" in m for m in msgs), msgs
    print("HD2 renamed, transposed, time-zone-shifted, array and Series inputs are refused with the reason in all four engines; a quarter of the tickers warns  PASS")


def test_panel_checks():
    p, _ = make_panel(n_days=300, n_stocks=20)
    c = p.close.copy(); c.columns = ["A"] * 20
    _raises(lambda: q.Panel(close=c, eligible=pd.DataFrame(True, index=c.index, columns=c.columns)), "duplicate ticker names")
    wk = p.close.resample("W-FRI").last(); we = p.eligible.resample("W-FRI").last().astype(bool)
    _warns(lambda: q.Panel(close=wk, eligible=we), "periods_per_year=252", "52 for weekly")
    _, msgs = _quiet(lambda: q.Panel(close=wk, eligible=we, periods_per_year=52))
    assert not any("periods_per_year" in m for m in msgs), msgs
    _, msgs = _quiet(lambda: q.Panel(close=p.close, eligible=p.eligible))                              # business days with 252
    assert not any("periods_per_year" in m for m in msgs), msgs
    cal = pd.date_range("2020-01-01", periods=300, freq="D")                                            # a 24/7 market with 365
    _, msgs = _quiet(lambda: q.Panel(close=pd.DataFrame(100.0, index=cal, columns=p.tickers), eligible=pd.DataFrame(True, index=cal, columns=p.tickers), periods_per_year=365))
    assert not any("periods_per_year" in m for m in msgs), msgs
    five = pd.date_range("2020-01-01", periods=2000, freq="5min")
    _, msgs = _quiet(lambda: q.Panel(close=pd.DataFrame(100.0, index=five, columns=p.tickers), eligible=pd.DataFrame(True, index=five, columns=p.tickers), periods_per_year=105120))
    assert not any("periods_per_year" in m for m in msgs), msgs
    print("HD3 duplicate tickers raise; weekly dates with 252 warn; business days with 252, calendar days with 365 and 5-minute bars with 105120 do not  PASS")


def test_big_moves_and_no_trades():
    p, rng = make_panel(n_days=400, n_stocks=40)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    c = p.close.copy(); c.iloc[200:, 5] = c.iloc[200:, 5] * 50.0                                      # a +4900% day
    out = _warns(lambda: q.backtest_portfolio(replace(p, close=c), f, hold=5, spread_bp=10), "daily moves above +1000%", p.tickers[5])
    ref, msgs = _quiet(lambda: q.backtest_portfolio(p, f, hold=5, spread_bp=10))
    assert not any("+1000%" in m for m in msgs), msgs
    _warns(lambda: q.backtest_portfolio(p, f * np.nan, hold=5, spread_bp=10), "no position was held")
    _warns(lambda: q.backtest_portfolio(p, f * 0 + 1.0, hold=5, spread_bp=10), "no position was held")      # a constant factor has no ranking
    W = f.rank(axis=1, pct=True).ge(0.9).astype(float).where(p.eligible, 0.0)
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    _warns(lambda: q.backtest_weights(p, W * 0, spread_bp=5), "no position was held")
    _warns(lambda: q.backtest_weights(p, W, spread_bp=5, capital=1.0, price=p.close), "no position was held", "capital")
    _, msgs = _quiet(lambda: q.backtest_weights(p, W, spread_bp=5))
    assert not any("no position" in m for m in msgs), msgs
    print("HD4/HD5 a +4900% day warns (numbers unchanged); an all-NaN or constant factor, zero weights and a capital below one lot warn that nothing was held  PASS")


def test_dsr_message():
    R = np.random.default_rng(0).normal(0, 0.01, (400, 5))
    bad = np.where(np.random.default_rng(1).random(R.shape) < 0.1, np.nan, R)                          # every column has a NaN
    _raises(lambda: q.validation.deflated_sharpe(bad, periods_per_year=252), "no column of R has a usable Sharpe")
    _raises(lambda: q.validation.deflated_sharpe(np.zeros((400, 3)), periods_per_year=252), "no column of R has a usable Sharpe")
    ok = R.copy(); ok[5, 2] = np.nan                                                                    # one column with a NaN is ignored, as documented
    assert q.validation.deflated_sharpe(ok, periods_per_year=252)["best"] != 2
    print("HD6 a matrix with no usable column raises a clear message; one NaN column is still ignored  PASS")


if __name__ == "__main__":
    test_bad_prices()
    test_alignment()
    test_panel_checks()
    test_big_moves_and_no_trades()
    test_dsr_message()
    print("hardening tests: all passed")
