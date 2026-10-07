"""Reconciliation against an independent implementation, and calibration under the null.

The engine is vectorised numpy. Here the same portfolio is computed again with plain Python loops, written from the
documented rules and sharing no code with `quantbt.portfolio`, and the two must agree to rounding error. A second
group of tests checks that the overfitting statistics are calibrated: on pure noise they must reject about as often
as they claim to.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import quantbt as q
from quantbt import validation as v
from quantbt.portfolio import metrics


def _panel(seed=3, T=260, N=40, delist=False, funding=False):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=T)
    cols = [f"S{i:02d}" for i in range(N)]
    close = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0)), index=dates, columns=cols)
    el = pd.DataFrame(rng.random((T, N)) > 0.15, index=dates, columns=cols)
    kw = {}
    if delist:
        da = pd.DataFrame(False, index=dates, columns=cols)
        for j, d in ((3, 150), (11, 200), (20, 90)):          # three names whose last real bar is day d
            close.iloc[d + 1:, j] = np.nan
            da.iloc[d, j] = True
            el.iloc[d + 1:, j] = False
        kw["delist_after"] = da
    if funding:
        kw["funding"] = pd.DataFrame(rng.normal(0.0002, 0.0003, (T, N)), index=dates, columns=cols)
    fac = pd.DataFrame(rng.normal(size=(T, N)), index=dates, columns=cols)
    return q.Panel(close=close, eligible=el, market="TEST", entry_lag=1, **kw), fac


def reference_net(panel, fac, *, long_q, short_q, hold, round_trip_bp, delist_return=None):
    """Plain loops. Rules, from the documentation:
    signal on day d is ranked among eligible names with a finite factor; the top `long_q` share is the long leg and the
    bottom `short_q` share the short leg, equal weight within a leg; the position held on day d is the mean of the legs
    of the last `hold` signal days (days with no signal count as an empty tranche); it is entered at close(d+lag) and
    earns close(d+lag) -> close(d+lag+1); each leg pays `round_trip_bp / 2` per unit of weight traded."""
    T, N = panel.close.shape
    close = panel.close.to_numpy(float)
    el = panel.eligible.to_numpy(bool)
    f = fac.reindex(index=panel.dates, columns=panel.tickers).to_numpy(float)
    lag = panel.entry_lag
    ret = np.zeros((T, N))
    for t in range(1, T):
        for j in range(N):
            a, b = close[t - 1, j], close[t, j]
            ret[t, j] = 0.0 if (math.isnan(a) or math.isnan(b)) else b / a - 1
    if delist_return is not None and panel.delist_after is not None:
        da = panel.delist_after.to_numpy(bool)
        for t in range(1, T):
            for j in range(N):
                if da[t - 1, j]:
                    ret[t, j] = delist_return
    fund = np.zeros((T, N))
    if panel.funding is not None:
        fund = np.nan_to_num(panel.funding.to_numpy(float), nan=0.0)

    def leg_weights(d, top):
        idx = [j for j in range(N) if el[d, j] and not math.isnan(f[d, j])]
        w = np.zeros(N)
        if not idx:
            return w
        order = sorted(idx, key=lambda j: f[d, j])
        n = len(order)
        pct = {j: (r + 1) / n for r, j in enumerate(order)}      # no ties in continuous random data
        q_ = long_q if top else short_q
        pick = [j for j in idx if (pct[j] >= 1 - q_ if top else pct[j] <= q_)]
        for j in pick:
            w[j] = 1.0 / len(pick)
        return w

    sig_l = [leg_weights(d, True) for d in range(T)]
    sig_s = [leg_weights(d, False) for d in range(T)] if short_q else [np.zeros(N)] * T

    def held(sig, d):
        lo = max(0, d - hold + 1)
        return sum(sig[t] for t in range(lo, d + 1)) / (d - lo + 1)

    HL = [held(sig_l, d) for d in range(T)]
    HS = [held(sig_s, d) for d in range(T)] if short_q else [np.zeros(N)] * T
    net = np.zeros(T)
    for d in range(T):
        e = d + lag + 1
        gross = fcost = 0.0
        if e < T:
            gross = float(HL[d] @ ret[e] - HS[d] @ ret[e])
            fcost = float(HL[d] @ fund[e] - HS[d] @ fund[e])
        cost = 0.0
        if d >= 1:
            cost = (np.abs(HL[d] - HL[d - 1]).sum() + np.abs(HS[d] - HS[d - 1]).sum()) * round_trip_bp / 1e4 * 0.5
        net[d] = gross - cost - fcost
    return net[: T - (lag + 1)]


def _check(panel, fac, **kw):
    rt = kw.pop("round_trip_bp")
    r = q.backtest_portfolio(panel, fac, spread_bp=rt, benchmark=None, grid=False, **kw)
    ref = reference_net(panel, fac, round_trip_bp=rt, **{k: kw[k] for k in ("long_q", "short_q", "hold")},
                        delist_return=kw.get("delist_return"))
    got = r.net_returns.to_numpy()
    assert len(got) == len(ref), (len(got), len(ref))
    err = float(np.max(np.abs(got - ref)))
    assert err < 1e-12, f"max abs difference {err:.2e}"
    return err


def test_long_short_matches_loops():
    p, f = _panel()
    e = _check(p, f, long_q=0.2, short_q=0.2, hold=5, round_trip_bp=10.0)
    e2 = _check(p, f, long_q=0.1, short_q=0.1, hold=1, round_trip_bp=0.0)
    print(f"R1 long-short, hold 5 / hold 1   max |diff| {max(e, e2):.1e}  PASS")


def test_long_only_matches_loops():
    p, f = _panel(seed=4)
    e = _check(p, f, long_q=0.3, short_q=None, hold=3, round_trip_bp=6.0)
    print(f"R2 long-only                      max |diff| {e:.1e}  PASS")


def test_delisting_and_funding_match_loops():
    p, f = _panel(seed=5, delist=True, funding=True)
    e = _check(p, f, long_q=0.25, short_q=0.25, hold=4, round_trip_bp=8.0, delist_return=-0.4)
    print(f"R3 delisting return + funding     max |diff| {e:.1e}  PASS")


def test_metrics_match_textbook_definitions():
    rng = np.random.default_rng(9)
    r = rng.normal(0.0004, 0.012, 600)
    dates = pd.bdate_range("2020-01-01", periods=600)
    m = metrics(r, dates, 252)
    n = len(r)
    cagr = float(np.prod(1 + r) ** (252 / n) - 1)
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252))
    eq = np.cumprod(1 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    downside = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))          # standard downside deviation, target 0
    sortino = float(r.mean() / downside * math.sqrt(252))
    assert abs(m["CAGR"] - cagr) < 1e-9, (m["CAGR"], cagr)
    assert abs(m["Sharpe"] - sharpe) < 1e-6, (m["Sharpe"], sharpe)
    assert abs(m["MDD"] - mdd) < 1e-12, (m["MDD"], mdd)
    assert abs(m["Sortino"] - sortino) < 1e-6, f"Sortino {m['Sortino']:.4f} vs textbook {sortino:.4f}"
    print(f"R4 CAGR / Sharpe / MDD / Sortino  match textbook definitions  PASS")


def test_cost_units_scalar_is_round_trip_panel_is_one_way():
    """The same name means two units. Pin both so nobody changes one silently."""
    p, f = _panel(seed=6)
    a = q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=2, spread_bp=20.0, benchmark=None, grid=False)
    b = q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=2, benchmark=None, grid=False,
                             spread_bp=pd.DataFrame(10.0, index=p.dates, columns=p.tickers).to_numpy())
    d = float(np.max(np.abs(a.net_returns.to_numpy() - b.net_returns.to_numpy())))
    assert d < 1e-12, d
    print(f"R5 scalar 20 bp (round trip) == panel of 10 bp (one way)   max |diff| {d:.1e}  PASS")


def test_dsr_is_not_anti_conservative_on_noise():
    """Best of 20 pure-noise strategies: DSR above 0.95 should happen in 5% of datasets or fewer. It is conservative
    (0 of 400 here), which is the safe direction; the power side is tested in test_validation.py."""
    rng = np.random.default_rng(11)
    hits = sum(v.deflated_sharpe(rng.normal(0, 0.01, (750, 20)))["dsr"] > 0.95 for _ in range(400))
    rate = hits / 400
    assert rate < 0.09, f"false-positive rate {rate:.3f}"
    print(f"R6 DSR false-positive rate on noise {rate:.3f} (target <= 0.05, bound 0.09)  PASS")


def test_permutation_p_values_are_not_anti_conservative():
    def search(r):                                   # best of 4 trend rules on a random walk of returns
        best = -9.0
        for k in (5, 10, 20, 40):
            sig = np.sign(np.convolve(r, np.ones(k) / k, mode="full")[: len(r)])
            pos = np.r_[0.0, sig[:-1]]
            x = pos * r
            best = max(best, float(x.mean() / (x.std() + 1e-12)))
        return best

    ps = [v.permutation_test(np.random.default_rng(500 + k).normal(0, 0.01, 800), search, n=99, seed=k)["p"] for k in range(40)]
    rate = sum(p <= 0.05 for p in ps) / len(ps)
    assert rate <= 0.15, f"rejects {rate:.2f} of pure-noise datasets at 5%"
    print(f"R7 permutation test rejects {rate:.2f} of noise datasets at p<=0.05 (expect ~0.05, bound 0.15)  PASS")


if __name__ == "__main__":
    test_long_short_matches_loops()
    test_long_only_matches_loops()
    test_delisting_and_funding_match_loops()
    test_metrics_match_textbook_definitions()
    test_cost_units_scalar_is_round_trip_panel_is_one_way()
    test_dsr_is_not_anti_conservative_on_noise()
    test_permutation_p_values_are_not_anti_conservative()
    print("reconcile tests: all passed")
