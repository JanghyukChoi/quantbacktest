"""An account that is wiped out stays wiped out.

RU1 weights   a 30x long that falls 50 percent in one name: the return that day is -100 percent, every later return is 0, the drawdown is -100 percent, the
              date is named, a warning is raised; before the ruin day nothing changes
RU2 portfolio the same for `backtest_portfolio` (weighting by signal gives a gross above 1 only through hold, so the ruin is built with a huge cost)
RU3 no ruin   a result that never loses 100 percent has `ruined` False and is identical with and without the guard (a hand-computed series)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.portfolio import stop_at_ruin


def _flat_panel(n_days=300, n=12, drop_day=150, drop=-0.5):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    close = pd.DataFrame(100.0, index=dates, columns=[f"A{i}" for i in range(n)])
    close.iloc[drop_day:, 0] = 100.0 * (1 + drop)                       # the first name falls once and stays there
    return q.Panel(close=close, eligible=pd.DataFrame(True, index=dates, columns=close.columns), market="TEST")


def test_weights_ruin():
    p = _flat_panel()
    w = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    w.iloc[100:, 0] = 30.0
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = q.backtest_weights(p, w, spread_bp=0, benchmark=None)
    net = r.net_returns.to_numpy()
    i = 148                                                             # row d earns close(d+1) -> close(d+2): the drop at row 150 is earned by row 148
    assert abs(net[i] + 1.0) < 1e-12 and (net[i + 1:] == 0.0).all() and (net[:i] == 0.0).all(), (net[i - 1:i + 3])
    assert r.metrics["ruined"] is True and r.metrics["ruin_date"] == str(p.dates[i].date())
    assert abs(r.metrics["MDD"] + 1.0) < 1e-12, r.metrics["MDD"]
    assert any("lost 100%" in str(x.message) for x in ws), [str(x.message) for x in ws]
    assert abs(r.metrics["CAGR"] + 1.0) < 1e-9 or np.isfinite(r.metrics["CAGR"])      # a finite number, not NaN from a negative base
    print(f"RU1 a 30x long that falls 50% is ruined on {r.metrics['ruin_date']}: -100%, then 0, MDD -100%, a warning  PASS")


def test_portfolio_ruin():
    p = _flat_panel(n_days=400, n=40)
    f = pd.DataFrame(np.random.default_rng(0).standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = q.backtest_portfolio(p, f, long_q=0.5, short_q=0.5, hold=1, spread_bp=1e6)          # cost 100x the traded weight each day
    net = r.net_returns.to_numpy()
    first = int(np.flatnonzero(net <= -1.0)[0])
    assert r.metrics["ruined"] and (net[first + 1:] == 0.0).all() and abs(net[first] + 1.0) < 1e-12
    assert any("lost 100%" in str(x.message) for x in ws)
    print("RU2 backtest_portfolio stops at the ruin day as well  PASS")


def test_no_ruin_is_unchanged():
    r0 = np.array([0.01, -0.02, 0.0, 0.03, -0.5, 0.2])
    out, i = stop_at_ruin(r0, len(r0))
    assert i is None and out is r0                                      # nothing to do: the very same array
    r1 = np.array([0.01, -1.0, 0.5, 0.5])
    out, i = stop_at_ruin(r1, 4)
    assert i == 1 and out.tolist() == [0.01, -1.0, 0.0, 0.0] and r1.tolist() == [0.01, -1.0, 0.5, 0.5]    # input untouched
    out, i = stop_at_ruin(np.array([0.1, -3.0, 0.2]), 2)
    assert i == 1 and out.tolist() == [0.1, -1.0, 0.0]                  # beyond the cut the value is not looked at; within it -300% becomes -100%
    out, i = stop_at_ruin(np.array([0.1, 0.1, -5.0]), 2)
    assert i is None                                                    # a ruin in the rows cut away (outside the sample) is not counted
    p = _flat_panel()
    w = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); w.iloc[100:, 0] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # no warning when nothing is lost beyond 100%
        r = q.backtest_weights(p, w, spread_bp=0, benchmark=None)
    assert r.metrics["ruined"] is False and r.metrics["ruin_date"] is None
    assert abs(r.metrics["MDD"] + 0.5) < 1e-12                          # the plain 50% fall of an unleveraged long
    print("RU3 no ruin: the same array comes back, the input is never changed, no warning, MDD -50% for an unleveraged long  PASS")


if __name__ == "__main__":
    test_weights_ruin()
    test_portfolio_ruin()
    test_no_ruin_is_unchanged()
    print("ruin tests: all passed")
