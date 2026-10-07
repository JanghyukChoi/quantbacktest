"""backtest_weights: equivalence with the engine, an independent loop implementation, known answers for each cost,
guard rails, and the shape of the capacity curve."""
from __future__ import annotations
import sys, tempfile
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.weights import ImpactModel, backtest_weights, capacity_curve
from test_reconcile import _panel


def _wpanel(seed=1, T=300, N=30, lag=1, vol_level=2e6):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=T)
    cols = [f"S{i:02d}" for i in range(N)]
    close = pd.DataFrame(40 * np.exp(np.cumsum(rng.normal(0, 0.015, (T, N)), axis=0)), index=dates, columns=cols)
    vol = pd.DataFrame(rng.lognormal(np.log(vol_level / 40), 0.3, (T, N)), index=dates, columns=cols)
    el = pd.DataFrame(True, index=dates, columns=cols)
    el.iloc[:40] = False
    return q.Panel(close=close, eligible=el, volume=vol, market="TEST", entry_lag=lag), rng


def _random_weights(p, rng, k=6):
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    for d in range(45, len(p.dates), 3):
        names = rng.choice(p.tickers, size=k, replace=False)
        w = rng.normal(0, 1, k); w = w / np.abs(w).sum()
        W.loc[p.dates[d], names] = w
    return W.replace(0.0, np.nan).ffill().fillna(0.0)          # rebalance every 3 days, hold in between


# -------------------------------------------------------------------------------- equivalence with the engine
def test_equals_backtest_portfolio_on_its_own_holdings():
    """Exact wherever no name is long and short at once. With overlapping tranches a name can sit in both legs; the engine
    charges the two legs as separate sleeves, `backtest_weights` charges the net trade, so it can only be cheaper."""
    worst, netting = 0.0, 0.0
    for kw, lag in ((dict(), 1), (dict(), 0), (dict(delist=True, funding=True), 1)):
        p, f = _panel(seed=12, **kw)
        p = replace(p, entry_lag=lag)
        for args in (dict(long_q=0.2, short_q=0.2, hold=1), dict(long_q=0.3, short_q=None, hold=3, weighting="rank")):
            r = q.backtest_portfolio(p, f, spread_bp=10.0, benchmark="equal", grid=False, delist_return=-0.3, **args)
            w = backtest_weights(p, pd.DataFrame(r.holdings, index=p.dates, columns=p.tickers), spread_bp=10.0,
                                 benchmark="equal", delist_return=-0.3)               # the universe check stays on: no false positive
            worst = max(worst, float(np.max(np.abs(r.net_returns.to_numpy() - w.net_returns.to_numpy()))))
            assert abs(r.metrics["Sharpe"] - w.metrics["Sharpe"]) < 1e-9
            assert np.allclose(r.benchmark_returns, w.benchmark_returns, atol=1e-15)
        r = q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5, spread_bp=10.0, benchmark=None, grid=False, delist_return=-0.3)
        w = backtest_weights(p, pd.DataFrame(r.holdings, index=p.dates, columns=p.tickers), spread_bp=10.0, benchmark=None,
                             delist_return=-0.3, check_universe=False)
        d = w.net_returns.to_numpy() - r.net_returns.to_numpy()
        assert d.min() > -1e-15, d.min()                                               # netting never costs more
        netting = max(netting, float(d.max()))
        assert (np.abs(d) > 1e-15).mean() < 0.15                                       # and it matters on few days
    assert worst < 1e-12, worst
    print(f"W1 the engine's own holdings: exact for long-only and one-day long-short (max |diff| {worst:.0e}); with overlapping "
          f"tranches netting only lowers cost (by at most {netting * 1e4:.2f} bp a day)  PASS")


# ---------------------------------------------------------------------------- independent loop implementation
def _ref(p, H, spread_rt, borrow_bp, y, aum, vw=20, aw=30, cap_bp=100.0):
    T, N = H.shape
    close = p.close.to_numpy(float); volm = p.volume.to_numpy(float); lag = p.entry_lag
    ret = np.full((T, N), np.nan)
    ret[1:] = close[1:] / close[:-1] - 1
    out = np.zeros(T)
    for d in range(T):
        e = d + lag + 1
        g = sum(H[d, j] * (0.0 if np.isnan(ret[e, j]) else ret[e, j]) for j in range(N)) if e < T else 0.0
        if d == 0:
            out[d] = g
            continue
        tr = np.abs(H[d] - H[d - 1])
        sp = tr.sum() * spread_rt / 2 / 1e4
        bo = sum(max(-H[d, j], 0.0) for j in range(N)) * borrow_bp / 1e4 / p.periods_per_year
        im = 0.0
        for j in range(N):
            if tr[j] == 0:
                continue
            win = ret[max(0, d - vw + 1):d + 1, j]
            ok = win[~np.isnan(win)]
            sig = np.std(ok, ddof=1) if len(ok) >= vw // 2 and len(ok) > 1 else np.nan
            dv = close[max(0, d - aw + 1):d + 1, j] * volm[max(0, d - aw + 1):d + 1, j]
            adv = dv.mean() if (d - aw + 1 >= 0 and not np.isnan(dv).any()) else np.nan
            unit = cap_bp / 1e4
            if np.isfinite(sig) and np.isfinite(adv) and adv > 0:
                unit = min(unit, y * sig * np.sqrt(tr[j] * aum / adv))
            im += tr[j] * unit
        out[d] = g - sp - bo - im
    return out[: T - (lag + 1)]


def test_matches_independent_loops_with_all_costs():
    worst = 0.0
    for seed, lag in ((3, 1), (4, 0)):
        p, rng = _wpanel(seed=seed, T=160, N=14, lag=lag, vol_level=3e5)
        W = _random_weights(p, rng, k=5)
        r = backtest_weights(p, W, spread_bp=8.0, borrow_bp=300.0, impact=ImpactModel(aum=2e7, y=1.3), benchmark=None, funding=False)
        ref = _ref(p, W.to_numpy(), 8.0, 300.0, 1.3, 2e7)
        worst = max(worst, float(np.max(np.abs(r.net_returns.to_numpy() - ref))))
        assert r.metrics["impact_annual_bp"] > 0 and r.metrics["borrow_annual_bp"] > 0
    assert worst < 1e-12, worst
    print(f"W2 spread + borrow + impact against plain loops, entry lag 1 and 0   max |diff| {worst:.0e}  PASS")


# --------------------------------------------------------------------------------------------- known answers
def test_impact_known_answer_and_sqrt_scaling():
    p, rng = _wpanel(seed=5, T=120, N=12, vol_level=5e6)
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    d = 60
    W.iloc[d:, 0] = 0.10                                       # one purchase of 10% of capital in S00 on day 60, then held
    m = ImpactModel(aum=1e8, y=1.0, max_cost_bp=1e9)                  # cap out of the way, so the formula itself is tested
    r = backtest_weights(p, W, impact=m, benchmark=None)
    sigma = p.ret1().iloc[d - 19:d + 1, 0].std(ddof=1)
    adv = (p.close.iloc[d - 29:d + 1, 0] * p.volume.iloc[d - 29:d + 1, 0]).mean()
    unit = 1.0 * sigma * np.sqrt(0.10 * 1e8 / adv)
    assert 0.0005 < unit < 0.05, unit                                 # a real price impact, far from both zero and the cap
    expected = 0.10 * unit
    got = r.metrics["impact_annual_bp"] / 1e4 / p.periods_per_year * (len(p.dates) - 2)    # total over the sample
    assert abs(got - expected) / expected < 1e-9, (got, expected)
    capped = backtest_weights(p, W, impact=ImpactModel(aum=1e8, y=1.0, max_cost_bp=unit * 1e4 / 2), benchmark=None)
    assert abs(capped.metrics["impact_annual_bp"] / 1e4 / p.periods_per_year * (len(p.dates) - 2) - 0.10 * unit / 2) < 1e-12   # the cap binds
    # scaling: with the cap out of the way, 4x the AUM means 2x the cost
    big = ImpactModel(aum=1e8, y=1.0, max_cost_bp=1e9)
    c1 = backtest_weights(p, W, impact=big, benchmark=None).metrics["impact_annual_bp"]
    c4 = backtest_weights(p, W, impact=replace(big, aum=4e8), benchmark=None).metrics["impact_annual_bp"]
    c2y = backtest_weights(p, W, impact=replace(big, y=2.0), benchmark=None).metrics["impact_annual_bp"]
    assert abs(c4 / c1 - 2.0) < 1e-9 and abs(c2y / c1 - 2.0) < 1e-9, (c4 / c1, c2y / c1)
    assert backtest_weights(p, W, impact=replace(big, aum=0.0), benchmark=None).metrics["impact_annual_bp"] == 0.0
    assert backtest_weights(p, W, benchmark=None).metrics["impact_annual_bp"] == 0.0
    print(f"W3 impact = |trade| * Y * sigma * sqrt(participation) (hand value {expected * 1e4:.2f} bp of capital, unit cost {unit * 1e4:.0f} bp per unit traded); "
          f"4x AUM -> 2.000x cost, 2x Y -> 2.000x cost, zero AUM -> zero  PASS")


def test_borrow_known_answer_and_untradable_cap():
    p, rng = _wpanel(seed=6, T=100, N=12)
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    W.iloc[50:, 0], W.iloc[50:, 1] = 0.5, -0.5                    # 50% long, 50% short, held
    r = backtest_weights(p, W, borrow_bp=250.0, benchmark=None)
    per_day = 0.5 * 250.0 / 1e4 / 252
    assert abs(r.metrics["borrow_annual_bp"] / 1e4 / 252 * (len(p.dates) - 2) - per_day * (len(p.dates) - 2 - 50)) < 1e-12
    only_long = W.clip(lower=0)
    assert backtest_weights(p, only_long, borrow_bp=250.0, benchmark=None).metrics["borrow_annual_bp"] == 0.0
    # a name with no volume history is charged exactly the cap per unit traded
    pv = replace(p, volume=p.volume.assign(S00=np.nan))
    W2 = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); W2.iloc[60:, 0] = 0.1
    c = backtest_weights(pv, W2, impact=ImpactModel(aum=1e6, max_cost_bp=100.0), benchmark=None)
    assert abs(c.metrics["impact_annual_bp"] / 1e4 / 252 * (len(p.dates) - 2) - 0.1 * 0.01) < 1e-12
    print("W4 borrow = short value x fee / periods per year; long-only pays none; an untradable name pays exactly the cap  PASS")


# ------------------------------------------------------------------------------------------- guard rails
def test_guard_rails():
    p, rng = _wpanel(seed=7, T=100, N=12)
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    W.iloc[10, 0] = 0.1                                           # day 10 is before the first eligible day (40)
    try:
        backtest_weights(p, W, benchmark=None)
    except ValueError as e:
        assert "not eligible" in str(e)
    else:
        raise AssertionError("opening a position outside the universe must raise")
    W2 = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); W2.iloc[50:, 0] = 0.1
    el = p.eligible.copy(); el.iloc[70:, 0] = False               # the name leaves the universe while held
    kept = backtest_weights(replace(p, eligible=el), W2, benchmark=None)
    assert kept.holdings[80, 0] == 0.1
    W3 = W2.copy(); W3.iloc[80:, 0] = 0.2
    try:
        backtest_weights(replace(p, eligible=el), W3, benchmark=None)
    except ValueError:
        pass
    else:
        raise AssertionError("increasing a position outside the universe must raise")
    Winf = W2.copy(); Winf.iloc[60, 1] = np.inf
    try:
        backtest_weights(p, Winf, benchmark=None)
    except ValueError as e:
        assert "inf" in str(e)
    else:
        raise AssertionError("inf must raise")
    Wn = W2.copy(); Wn.iloc[60, 2] = np.nan
    assert backtest_weights(p, Wn, benchmark=None).metrics["nan_weights_treated_as_zero"] >= 1
    try:
        backtest_weights(replace(p, volume=None), W2, impact=ImpactModel(aum=1e6), benchmark=None)
    except ValueError as e:
        assert "volume" in str(e)
    else:
        raise AssertionError("impact without volume must raise")
    print("W5 opening or raising a position outside the universe raises; holding one that left is allowed; inf and missing volume raise; NaN counted  PASS")


# -------------------------------------------------------------------------------------------- capacity
def test_capacity_curve_shape_and_ledger():
    p, rng = _wpanel(seed=8, T=300, N=30, vol_level=2e6)
    W = _random_weights(p, rng, k=8)
    curve = capacity_curve(p, W, aums=[1e5, 4e5, 1.6e6, 6.4e6], y_values=(0.5, 1.0, 2.0), spread_bp=4.0, max_cost_bp=1e9)
    c1 = curve[curve.y == 1.0].reset_index(drop=True)
    assert (np.diff(c1["impact_annual_bp"]) > 0).all() and (np.diff(c1["sharpe"]) <= 1e-12).all()
    assert (np.diff(c1["participation_p99"]) > 0).all()
    assert np.allclose(c1["impact_annual_bp"].iloc[1:].to_numpy() / c1["impact_annual_bp"].iloc[:-1].to_numpy(), 2.0, rtol=1e-9)  # 4x AUM = 2x cost
    mid = curve[curve.aum == 4e5].set_index("y")["impact_annual_bp"]
    assert abs(mid[2.0] / mid[1.0] - 2.0) < 1e-9 and abs(mid[1.0] / mid[0.5] - 2.0) < 1e-9          # linear in Y
    capped = capacity_curve(p, W, aums=[4e5], y_values=(1.0, 2.0), spread_bp=4.0).set_index("y")["impact_annual_bp"]
    assert capped[2.0] <= 2.0 * capped[1.0] + 1e-9 and capped[2.0] < 2.0 * mid[1.0] - 1.0         # a binding cap makes cost grow more slowly
    with tempfile.TemporaryDirectory(prefix="weightstest-") as d:
        led = q.Ledger(d)
        backtest_weights(p, W, spread_bp=4.0, benchmark=None, ledger=led, family="w", name="a")
        backtest_weights(p, W, spread_bp=4.0, benchmark=None, ledger=led, family="w", name="a")
        backtest_weights(p, W, spread_bp=8.0, benchmark=None, ledger=led, family="w", name="a")
        assert led.n_trials("w") == 2
    print("W6 capacity: costs and participation rise with AUM, Sharpe does not rise, cost is linear in Y; the ledger counts it  PASS")


if __name__ == "__main__":
    test_equals_backtest_portfolio_on_its_own_holdings()
    test_matches_independent_loops_with_all_costs()
    test_impact_known_answer_and_sqrt_scaling()
    test_borrow_known_answer_and_untradable_cap()
    test_guard_rails()
    test_capacity_curve_shape_and_ledger()
    print("weights tests: all passed")
