"""Tests written because an automatic mutation run (local/robustness/automut.py) broke the code in a small way and no test noticed.

Each one pins a behaviour the other tests happened to leave free: which neighbours the plateau looks at, what an all-zero grid says, which factor columns the regression keeps, which
way the top and bottom contributors are sorted, what "worse than this" means when nothing can be worse, and what a third of the bars is.

MK1 plateau       the neighbours of the best cell are exactly the eight around it
MK2 plateau       a grid whose best score is zero has no ratio and no share, and says why, without a numerical warning
MK3 decompose     a constant factor and a factor with too few observations are left out, the usable one stays
MK4 attribution   top contributors are the largest, bottom the smallest
MK5 drawdowns     if the history never falls, no resampled history is worse than it
MK6 regimes       calm and stormy are each a third of the bars that have a volatility
MK7 event shift   the default smallest shift is twice the holding period
MK8 win interval  a win rate of 0 or 1 lies inside its Wilson interval (found by the property tests)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import robustness as rb


def test_plateau_neighbours():
    T = np.zeros((7, 7))
    T[3, 3] = 1.0
    T[2:5, 2:5] = 0.8
    T[3, 3] = 1.0
    r = rb.parameter_plateau(pd.DataFrame(T))
    assert r["best_at"] == (3, 3) and r["n_neighbours"] == 8
    assert abs(r["neighbour_mean"] - 0.8) < 1e-12 and abs(r["neighbour_ratio"] - 0.8) < 1e-12 and r["verdict"] == "plateau"
    T2 = np.zeros((7, 7)); T2[3, 3] = 1.0; T2[3, 5] = 0.9; T2[1, 3] = 0.9        # two cells away: not neighbours
    r2 = rb.parameter_plateau(pd.DataFrame(T2))
    assert r2["neighbour_mean"] == 0.0 and r2["verdict"] == "spike"
    print("MK1 the plateau looks at the eight cells around the best one and no further  PASS")


def test_plateau_zero_best():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = rb.parameter_plateau(pd.DataFrame(np.zeros((4, 4))))
        s = rb.parameter_plateau(pd.Series([0.0, -0.2, -0.1, -0.3]))
    for x in (r, s):
        assert np.isnan(x["neighbour_ratio"]) and np.isnan(x["share_within"]) and x["verdict"] == "undetermined" and x["reason"] == "no setting has a positive Sharpe ratio", x
    print("MK2 an all-zero or all-negative grid is undetermined and says why, without a warning  PASS")


def test_decompose_column_filter():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=300)
    good = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    y = 0.5 * good + rng.normal(0, 0.005, 300)
    F = pd.DataFrame({"good": good, "flat": 0.001, "sparse": np.where(np.arange(300) < 40, rng.normal(0, 0.01, 300), np.nan)}, index=idx)
    d = rb.decompose(y, F)
    assert list(d["factors"]) == ["good"], list(d["factors"])
    print("MK3 a constant factor and a nearly empty factor are dropped, the real one stays  PASS")


def test_attribution_order():
    rng = np.random.default_rng(5)
    T, N = 400, 14
    idx = pd.bdate_range("2020-01-01", periods=T)
    cols = [f"S{i:02d}" for i in range(N)]
    ret = rng.normal(0, 0.002, (T, N))
    ret[:, 0] += 0.004                                                   # the winner
    ret[:, 1] += 0.002
    ret[:, 2] -= 0.003                                                   # the loser
    close = pd.DataFrame(100 * np.exp(ret.cumsum(0)), index=idx, columns=cols)
    p = q.Panel(close=close, eligible=pd.DataFrame(True, index=idx, columns=cols), volume=None, mkt_cap=None, market="T")
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    W.iloc[:, [0, 1, 2]] = [0.3, 0.3, 0.3]
    W.iloc[:30] = 0.0
    r = q.backtest_weights(p, W, spread_bp=0.0, benchmark=None, check_universe=False, check_shortable=False)
    at = rb.attribution(p, r, n_top=3)
    top, bottom = list(at["names"]["top"].items()), list(at["names"]["bottom"].items())
    assert [k for k, _ in top][:2] == ["S00", "S01"] and top[0][1] > top[1][1] > 0, top
    assert bottom[0][0] == "S02" and bottom[0][1] < 0 and bottom[0][1] <= bottom[1][1] <= bottom[2][1], bottom
    print("MK4 top contributors are sorted largest first and bottom contributors smallest first  PASS")


def test_drawdown_never_falls():
    x = pd.Series(np.full(300, 0.001), index=pd.bdate_range("2020-01-01", periods=300))
    d = rb.drawdown_distribution(x, n=200, seed=1)
    assert d["realized"] == 0.0 and d["p_worse"] == 0.0 and d["p95"] == 0.0
    print("MK5 a history that never falls has no resampled history worse than it  PASS")


def test_regime_thirds():
    rng = np.random.default_rng(9)
    n = 600
    idx = pd.bdate_range("2019-01-01", periods=n)
    scale = np.where((np.arange(n) // 100) % 2 == 0, 0.004, 0.02)
    b = pd.Series(rng.normal(0.0002, 1, n) * scale, index=idx)
    s = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    t = rb.regimes(s, b)["table"]
    valid = n - 59                                                       # the first 59 bars have no 60-bar volatility yet
    calm = int(t.loc[[i for i in t.index if i.startswith("calm")][0], "bars"])
    storm = int(t.loc[[i for i in t.index if i.startswith("stormy")][0], "bars"])
    assert abs(calm - valid / 3) <= 0.06 * valid and abs(storm - valid / 3) <= 0.06 * valid, (calm, storm, valid)
    print("MK6 calm and stormy are each about a third of the bars with a volatility  PASS")


def test_event_shift_default_minimum():
    from test_synthetic import make_panel
    p, _ = make_panel(n_days=120, n_stocks=15, seed=2)
    sig = pd.DataFrame(np.random.default_rng(2).random(p.close.shape) < 0.15, index=p.dates, columns=p.tickers) & p.eligible
    rb.signal_permutation(p, sig, 5, n=20)                                                  # horizon 5: smallest shift is 21 bars, room enough
    try:
        rb.signal_permutation(p, sig, 28, n=20)                                             # horizon 28: smallest shift is 56 bars, 120 - 112 < 10
    except ValueError as e:
        assert "too few" in str(e), e
    else:
        raise AssertionError("a holding period of 28 on 120 bars must be refused: the default shift is twice the holding period")
    print("MK7 the default smallest shift is twice the holding period  PASS")


def test_win_interval_contains_the_estimate():
    """Found by the property tests (seed 0, 80 bars all positive): the Wilson interval's upper end was 0.9999999999999999 for a win rate of exactly 1."""
    from pitbacktest.review import _wilson
    idx = pd.bdate_range("2020-01-01", periods=80)
    up = pd.Series(0.001 + np.arange(80) * 1e-9, index=idx)
    for x, rate in ((up, 1.0), (-up, 0.0)):
        m = rb.mean_tests(x)
        lo, hi = m["win_ci"]
        assert m["win_rate"] == rate and 0.0 <= lo <= rate <= hi <= 1.0, (lo, hi)
    assert _wilson(10, 10)[1] == 1.0 and _wilson(0, 10)[0] == 0.0 and all(np.isnan(_wilson(0, 0)))
    print("MK8 a win rate of 0 or 1 sits inside its own interval, to the last digit  PASS")


if __name__ == "__main__":
    for fn in (test_plateau_neighbours, test_plateau_zero_best, test_decompose_column_filter, test_attribution_order, test_drawdown_never_falls, test_regime_thirds, test_event_shift_default_minimum, test_win_interval_contains_the_estimate):
        fn()
    print("mutant-kill tests: all passed")
