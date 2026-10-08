"""Delisting inside `screen` and `backtest_event`, and four guards found by independent review of the specification.

DG1 forward     a holding period that runs into a delisting keeps its loss (hand-computed), with and without `delist_return`; untouched when
                nothing is delisted
DG2 event       the trades that ended in a delisting are in the statistics: `delist_return` moves the mean trade
DG3 guards      an intraday backtest refuses entry_lag 0 (also through latency_sweep), a scalar NaN cost raises, inject_delistings checks its
                arguments at once, and a panel shorter than a year leaves out the 12-1 momentum control (with a warning) instead of dropping
                every date
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
from pitbacktest.core.controls import build_controls
from pitbacktest.crypto import intraday as ib
from pitbacktest.equity import inject_delistings
from test_argument_checks import _raises
from test_synthetic import make_panel


def _with_dead(n_dead=1, n_days=900, n_stocks=100, crash=0.2, first=300, spacing=40):
    p, rng = make_panel(n_days=n_days, n_stocks=n_stocks)
    close = p.close.copy()
    da = pd.DataFrame(False, index=p.dates, columns=p.tickers)
    dead = list(p.tickers[:n_dead])
    for j, c in enumerate(dead):
        k = first + spacing * j
        close.loc[p.dates[k]:, c] = close.loc[p.dates[k - 1], c] * crash          # the last bars trade at a crashed price ...
        close.loc[p.dates[k + 1]:, c] = np.nan                                     # ... and then the security is gone
        da.loc[p.dates[k], c] = True
    return replace(p, close=close, delist_after=da), dead, first, spacing


def test_forward_keeps_the_delisting_loss():
    p, dead, k, _ = _with_dead()
    c, h = dead[0], 10
    last, px = p.close[c].iloc[k], p.close[c]
    for dr in (None, -0.5):
        fw = p.forward(h, dr)[c]
        cross = (last * (1.0 + (dr or 0.0))) / px.iloc[295 + 1] - 1.0                # t=295: the window 296 -> 306 runs past the last bar (300)
        assert abs(fw.iloc[295] - cross) < 1e-6, (dr, fw.iloc[295], cross)
        assert abs(fw.iloc[289] - (px.iloc[300] / px.iloc[290] - 1.0)) < 1e-6          # t=289: the window ends on the last bar, no delisting factor yet
        assert fw.iloc[320] == 0.0                                                    # after the delisting the security sits at a flat price
    base, _ = make_panel(n_days=500, n_stocks=60)
    assert np.array_equal(base.forward(5).to_numpy(), (base.close.shift(-6) / base.close.shift(-1) - 1.0).astype(np.float32).to_numpy(), equal_nan=True)
    gap = base.close.copy(); gap.iloc[100:103, 0] = np.nan                            # a halt that is not a delisting still gives NaN
    assert np.isnan(replace(base, close=gap).forward(5).iloc[95, 0])
    print("DG1 a window that runs into a delisting keeps its loss (-80.2%, -90.1% with delist_return -50%); no delisting: unchanged; a halt: NaN  PASS")


def test_event_and_screen_count_the_delisted_trades():
    p, dead, _, _ = _with_dead(n_dead=30, n_stocks=100, first=250, spacing=15)
    sig = pd.DataFrame(True, index=p.dates, columns=p.tickers)                        # fires on everything, every day: plenty of trades
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = q.backtest_event(p, sig, horizons=(10,), neutralize_check=False)
        r9 = q.backtest_event(p, sig, horizons=(10,), neutralize_check=False, delist_return=-0.9)
        f = -p.close.pct_change(7)
        s9 = q.screen(p, {"a": f}, horizons=(5,), primary_h=5, n_null=10, neutralize_all=False, delist_return=-0.9)
    a, b = r0.per_horizon[10]["mean_bp"], r9.per_horizon[10]["mean_bp"]
    assert b < a - 5.0, (a, b)                                                        # the extra loss reaches the average trade
    assert len(s9.summary) == 1
    print(f"DG2 the trades that ended in a delisting are in the event statistics: the mean trade is {a:.0f} bp, {b:.0f} bp with a -90% delisting return  PASS")


def test_guards():
    p = replace(make_panel(n_days=400, n_stocks=40)[0], entry_lag=0)
    f = -p.close.pct_change(1)
    _raises(lambda: ib.backtest_intraday(p, f), "entry_lag >= 1")
    ok = replace(p, entry_lag=1); ok.meta = {"bar_minutes": 5}
    _raises(lambda: ib.latency_sweep(ok, f, lags=(0, 1)), "entry_lag >= 1")           # the sweep cannot go round the rule
    w = pd.DataFrame(0.0, index=ok.dates, columns=ok.tickers); w.iloc[100:, :5] = 0.2
    _raises(lambda: q.backtest_weights(ok, w, spread_bp=float("nan")), "spread_bp", "finite")
    _raises(lambda: q.backtest_portfolio(ok, f, spread_bp=float("nan")), "spread_bp", "finite")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q.backtest_weights(ok, w, spread_bp=pd.DataFrame(np.where(np.random.default_rng(0).random(ok.close.shape) < 0.3, np.nan, 5.0),
                                                         index=ok.dates, columns=ok.tickers).to_numpy(), benchmark=None)      # NaN inside a panel stays allowed
    base, _ = make_panel(n_days=500, n_stocks=60)
    _raises(lambda: inject_delistings(base, hazard="bogus", annual_rate=0.0001), "hazard")                # even when no name would be drawn
    _raises(lambda: inject_delistings(replace(base, volume=None), hazard="illiquid"), "volume")
    _raises(lambda: inject_delistings(base, annual_rate=1.5), "annual_rate")
    short, _ = make_panel(n_days=200, n_stocks=60)
    with warnings.catch_warnings(record=True) as w_:
        warnings.simplefilter("always")
        c = build_controls(short, include_chars=False)
    assert "mom252_21" not in c and any("12-1 momentum control" in str(x.message) for x in w_)
    assert all(np.isfinite(v.to_numpy()[100:150]).any() for v in c.values())                              # the rest still carry information
    longp, _ = make_panel(n_days=700, n_stocks=60)
    with warnings.catch_warnings(record=True) as w_:
        warnings.simplefilter("always")
        assert "mom252_21" in build_controls(longp, include_chars=False) and not any("12-1" in str(x.message) for x in w_)
    print("DG3 intraday entry_lag 0 (also via latency_sweep), a scalar NaN cost, a bad hazard or rate, and a too-short panel for the momentum control are handled  PASS")


if __name__ == "__main__":
    test_forward_keeps_the_delisting_loss()
    test_event_and_screen_count_the_delisted_trades()
    test_guards()
    print("delisting and guard tests: all passed")
