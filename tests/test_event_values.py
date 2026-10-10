"""The arithmetic inside `backtest_event`, on arrays small enough to work out by hand or to recompute in a few lines.

An automatic mutation run changed percent scalings, percentile levels, pool thresholds and a regression's inclusion rule in event.py without any test failing: the other tests check that
events behave sensibly, not what the numbers are.

EV1 trade statistics   win rate, base rate, lift, mean, median, average win and loss, payoff, quartiles and the skew warning of a hand-made set of trades; the 100-trade floor and the pool of 20
EV2 daily excess       the days that enter (a firing name and a big enough pool, the default pool being half the median universe up to 30) and the value of each
EV3 per horizon        excess, its Newey-West t, the number of days and the net figure of a full run; the default horizons
EV4 dummy regression   the raw coefficient and its t equal a plain difference of group means per day, with the day filters (30 names, 3 firing and 3 not)
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
from pitbacktest.core.controls import build_controls
from pitbacktest.event import _daily_excess, _trade_stats
from test_synthetic import make_panel


def test_trade_stats():
    rng = np.random.default_rng(1)
    D, N = 40, 30
    cum = rng.normal(0.001, 0.03, (D, N))
    elig = np.ones((D, N), bool)
    fire = np.zeros((D, N), bool)
    for i in range(D):
        fire[i, rng.choice(N, 4, replace=False)] = True                              # 160 trades
    cost = 0.002
    st = _trade_stats(fire, cum, elig, cost)
    t = (cum - cost)[fire]
    w, l = t[t > 0], t[t < 0]
    assert st["n_trades"] == 160 and abs(st["win_rate"] - (t > 0).mean() * 100) < 1e-9
    base = np.mean([(cum[i] - cost > 0).mean() for i in range(D) for _ in range(4)])
    assert abs(st["base_rate"] - base * 100) < 1e-9 and abs(st["lift_pp"] - (st["win_rate"] - st["base_rate"])) < 1e-9
    assert abs(st["mean_bp"] - t.mean() * 1e4) < 1e-9 and abs(st["median_bp"] - np.median(t) * 1e4) < 1e-9
    assert abs(st["avg_win_bp"] - w.mean() * 1e4) < 1e-9 and abs(st["avg_loss_bp"] - l.mean() * 1e4) < 1e-9 and abs(st["payoff"] - w.mean() / abs(l.mean())) < 1e-12
    assert abs(st["p25_bp"] - np.percentile(t, 25) * 1e4) < 1e-9 and abs(st["p75_bp"] - np.percentile(t, 75) * 1e4) < 1e-9
    assert st["skew_warning"] == bool(t.mean() > 0 > np.median(t))
    # a pool of 19 gives no base rate, a pool of 20 does
    f2 = np.zeros((D, 19), bool); f2[:, :3] = True; f2 = np.tile(f2, (2, 1))[:D + 40]
    c19 = rng.normal(0, 0.03, (len(f2), 19)); e19 = np.ones_like(f2)
    s19 = _trade_stats(f2, c19, e19, 0.0)
    assert s19 is not None and np.isnan(s19["base_rate"]) and np.isnan(s19["lift_pp"])
    f20 = np.zeros((D + 40, 20), bool); f20[:, :3] = True
    s20 = _trade_stats(f20, rng.normal(0, 0.03, (D + 40, 20)), np.ones_like(f20), 0.0)
    assert np.isfinite(s20["base_rate"])
    # 99 trades: nothing; 100: statistics
    f99 = np.zeros((33, 30), bool); f99[:, :3] = True
    assert _trade_stats(f99, rng.normal(0, 0.03, (33, 30)), np.ones_like(f99), 0.0) is None
    f100 = np.zeros((34, 30), bool); f100[:33, :3] = True; f100[33, 0] = True
    assert _trade_stats(f100, rng.normal(0, 0.03, (34, 30)), np.ones_like(f100), 0.0)["n_trades"] == 100
    print("EV1 trade statistics equal the definitions; a pool of 19 has no base rate, 20 has; 99 trades give nothing, 100 do  PASS")


def test_daily_excess():
    rng = np.random.default_rng(3)

    def build(universe, pools, fires):
        D = len(pools)
        cum = rng.normal(0, 0.02, (D, universe))
        elig = np.ones((D, universe), bool)
        for i, pl in enumerate(pools):
            cum[i, pl:] = np.nan
        fire = np.zeros((D, universe), bool)
        for i, k in enumerate(fires):
            fire[i, :k] = True
        return fire, cum, elig

    def expect(fire, cum, days):
        return np.array([cum[i][fire[i] & np.isfinite(cum[i])].mean() - cum[i][np.isfinite(cum[i])].mean() for i in days])

    # 40 names: the default pool is half of 40 = 20; a pool of 20 enters, 19 does not
    f, c, e = build(40, [40, 20, 19, 25, 30], [2, 2, 2, 0, 1])
    out = _daily_excess(f, c, e)
    assert np.allclose(out, expect(f, c, [0, 1, 4])), out
    # 100 names: half is 50 but the cap is 30; 30 enters, 29 does not
    f, c, e = build(100, [100, 30, 29, 60], [3, 3, 3, 3])
    assert np.allclose(_daily_excess(f, c, e), expect(f, c, [0, 1, 3]))
    # 12 names: half is 6; 6 enters, 5 does not
    f, c, e = build(12, [12, 6, 5], [2, 2, 2])
    assert np.allclose(_daily_excess(f, c, e), expect(f, c, [0, 1]))
    # 6 names: half is 3 but the floor is 5
    f, c, e = build(6, [6, 5, 4], [2, 2, 2])
    assert np.allclose(_daily_excess(f, c, e), expect(f, c, [0, 1]))
    # explicit settings
    f, c, e = build(40, [40, 40, 40], [1, 2, 3])
    assert len(_daily_excess(f, c, e, min_fire=2)) == 2 and len(_daily_excess(f, c, e, min_pool=41)) == 0 and len(_daily_excess(f, c, e, min_pool=40)) == 3
    print("EV2 daily excess: the default pool floor is half the median universe, at least 5 and at most 30, and each day's value is the firing mean minus the pool mean  PASS")


def _setup():
    p, _ = make_panel(n_days=520, n_stocks=40, seed=4)
    rng = np.random.default_rng(11)
    sig = pd.DataFrame(rng.random(p.close.shape) < 0.04, index=p.dates, columns=p.tickers) & p.eligible
    return p, sig


def test_per_horizon_and_defaults():
    p, sig = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = q.backtest_event(p, sig, horizons=(2, 5), cost_bp=15.0, neutralize_check=False)
        d = q.backtest_event(p, sig, neutralize_check=False)
    for h, st in r.per_horizon.items():
        cum = p.forward(h).to_numpy(np.float64)
        ex = _daily_excess((sig & p.eligible).to_numpy(), cum, p.eligible.to_numpy())
        mu, _, t, n = rb.newey_west_t(pd.Series(ex), max(h, 21))
        assert abs(st["excess_bp"] - mu * 1e4) < 1e-9 and abs(st["t"] - t) < 1e-9 and st["n_days"] == len(ex) == n and abs(st["net_bp"] - (mu * 1e4 - 15.0)) < 1e-9
    from pitbacktest.core.estimators import decile_profile
    bh = r.spec["best_horizon"]
    want = decile_profile(sig.astype(float), p.forward(bh), p.eligible, lag=max(bh, 21))
    assert all(np.allclose(np.asarray(r.spec["decile"][k], float), np.asarray(want[k], float), equal_nan=True) for k in want if k in r.spec["decile"]) and len(r.spec["decile"]) > 0
    assert d.spec["horizons"] == [1, 2, 5, 10, 21] and d.spec["cost_bp"] == 20.0
    print("EV3 per-horizon excess, t, day count and net figure equal a recomputation; the default horizons and cost are 1, 2, 5, 10, 21 and 20 bp  PASS")


def test_dummy_regression():
    p, sig = _setup()
    sig = sig.copy()
    for i, k in zip(range(300, 315), [0, 1, 2, 3, 4] * 3):                       # days when all but k eligible names fire: the 3-and-3 rule is tested at its edge
        names = np.flatnonzero(p.eligible.iloc[i].to_numpy())
        sig.iloc[i, :] = False
        sig.iloc[i, names[k:]] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = q.backtest_event(p, sig, horizons=(5,), cost_bp=10.0)
    fire = (sig & p.eligible).to_numpy()
    cum = p.forward(5).to_numpy(np.float64)
    ctrl = [c.values.astype(np.float64) for c in build_controls(p).values()]
    ev = p.eligible.to_numpy()
    b = np.full(len(cum), np.nan)
    for i in range(len(cum)):
        pool = ev[i] & np.isfinite(cum[i])
        for c in ctrl:
            pool = pool & np.isfinite(c[i])
        n = int(pool.sum())
        d = fire[i][pool]
        if n < 30 or d.sum() < 3 or d.sum() > n - 3:
            continue
        y = cum[i][pool]
        b[i] = y[d].mean() - y[~d].mean()                                          # OLS on a constant and a dummy: the difference of the group means
    m0, _, t0, _ = rb.newey_west_t(pd.Series(b), 21)
    got = r.neutralized[5]
    assert np.isfinite(m0) and abs(got["raw_bp"] - m0 * 1e4) < 1e-6 and abs(got["raw_t"] - t0) < 1e-6, (got["raw_bp"], m0 * 1e4, got["raw_t"], t0)
    assert abs(got["survival_pct"] - got["neutral_bp"] / got["raw_bp"] * 100) < 1e-6
    print("EV4 the dummy coefficient and its t equal the per-day difference of group means with the 30-name, 3-and-3 filters  PASS")


if __name__ == "__main__":
    test_trade_stats(); test_daily_excess(); test_per_horizon_and_defaults(); test_dummy_regression()
    print("event value tests: all passed")
