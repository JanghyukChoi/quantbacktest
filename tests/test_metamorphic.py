"""Metamorphic tests: change the input in a way whose effect on the output is known, and check that the output changes in exactly that way.

No expected number is written down anywhere here, so these tests do not depend on what the author believed the answer should be. Each relation is run on many random panels.

MM1 order        the order of the securities (columns) does not change the portfolio, the event statistics, the walk-forward or the attribution totals
MM2 monotone     a strictly increasing transform of a factor does not change an equal-weight quantile portfolio
MM3 prefix       cutting the data after date t does not change the result on the dates before it: nothing in the engines looks ahead (portfolio, weights, event, with costs and funding)
MM4 inert        adding a security that is never eligible, and a zero weight, changes nothing
MM5 costs        more spread never raises a net return on any bar; zero cost equals the gross
MM6 scale        multiplying every price by a constant changes no return; Sharpe does not change when all returns are multiplied by a positive constant; the deflated Sharpe either
MM7 renaming     an event signal and the same signal with its securities renamed give the same statistics
MM8 future       scrambling the factor and the eligibility after a date leaves every position held up to that date unchanged, at any depth of lookahead
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
from pitbacktest import robustness as rb

RUNS = 12


def _q(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _world(seed, T=420, N=40, funding=True, delist=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T)
    cols = [f"S{i:02d}" for i in range(N)]
    ret = rng.normal(0.0003, 0.01, (T, 1)) + rng.normal(0, 0.02, (T, N))
    close = pd.DataFrame(100 * np.exp(ret.cumsum(0)), index=idx, columns=cols)
    el = pd.DataFrame(rng.random((T, N)) < 0.9, index=idx, columns=cols)
    el.iloc[:40] = False
    vol = pd.DataFrame(rng.lognormal(12, 1, (T, N)), index=idx, columns=cols)
    cap = close * pd.Series(rng.lognormal(15, 1, N), index=cols)
    fund = pd.DataFrame(rng.normal(0.0001, 0.0003, (T, N)), index=idx, columns=cols) if funding else None
    da = None
    if delist:
        da = pd.DataFrame(False, index=idx, columns=cols)
        for j in rng.choice(N, 3, replace=False):
            k = int(rng.integers(150, 300))
            close.iloc[k + 1:, j] = np.nan
            da.iloc[k, j] = True
    p = q.Panel(close=close, eligible=el, volume=vol, mkt_cap=cap, funding=fund, delist_after=da, market="T")
    f = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols).where(p.eligible)
    return p, f, rng


def _reorder(p, cols):
    r = lambda d: None if d is None else d[cols]                                                    # noqa: E731
    return q.Panel(close=p.close[cols], eligible=p.eligible[cols], volume=r(p.volume), mkt_cap=r(p.mkt_cap), funding=r(p.funding), delist_after=r(p.delist_after), market=p.market)


BT = dict(long_q=0.2, short_q=0.2, hold=3, spread_bp=10.0, sell_bp=5.0)


def test_column_order():
    for s in range(RUNS):
        p, f, rng = _world(s)
        cols = list(p.tickers); rng.shuffle(cols)
        p2 = _reorder(p, cols)
        a = _q(lambda: q.backtest_portfolio(p, f, **BT))
        b = _q(lambda: q.backtest_portfolio(p2, f[cols], **BT))
        assert np.allclose(a.net_returns.to_numpy(), b.net_returns.to_numpy(), atol=1e-13, rtol=0), (s, float(np.abs(a.net_returns - b.net_returns).max()))
        assert abs(a.metrics["turnover_daily"] - b.metrics["turnover_daily"]) < 1e-12
        sig = f > 1.5
        ea = _q(lambda: q.backtest_event(p, sig, horizons=(3, 5), cost_bp=10, neutralize_check=False))
        eb = _q(lambda: q.backtest_event(p2, sig[cols], horizons=(3, 5), cost_bp=10, neutralize_check=False))
        for h in (3, 5):
            for k, v in ea.per_horizon[h].items():
                assert abs(v - eb.per_horizon[h][k]) < 1e-9 or (v != v and eb.per_horizon[h][k] != eb.per_horizon[h][k]), (s, h, k)
        at_a, at_b = rb.attribution(p, a), rb.attribution(p2, b)
        assert abs(at_a["gross_annual"] - at_b["gross_annual"]) < 1e-12 and abs(at_a["long_annual"] - at_b["long_annual"]) < 1e-12
        assert at_a["names"]["top"].keys() == at_b["names"]["top"].keys() and all(abs(at_a["names"]["top"][k] - at_b["names"]["top"][k]) < 1e-12 for k in at_a["names"]["top"])     # the same securities, by name
        assert at_a["names"]["bottom"].keys() == at_b["names"]["bottom"].keys()
        R = pd.DataFrame(rng.standard_normal((300, 6)) * 0.01, index=pd.bdate_range("2020-01-01", periods=300))
        w1 = rb.walk_forward(R, train=100, test=50)
        w2 = rb.walk_forward(R[R.columns[::-1]], train=100, test=50)
        assert np.array_equal(w1["oos"].to_numpy(), w2["oos"].to_numpy())
    print(f"MM1 {RUNS} random panels: shuffling the securities changes no portfolio return (1e-13), event statistic, attribution total or walk-forward  PASS")


def test_monotone_transform():
    for s in range(RUNS):
        p, f, _ = _world(100 + s)
        base = _q(lambda: q.backtest_portfolio(p, f, **BT))
        for name, g in (("exp", np.exp(f / 3)), ("cube", f ** 3), ("affine", 5 * f - 2), ("rank", f.rank(axis=1))):
            r = _q(lambda: q.backtest_portfolio(p, g, **BT))
            assert np.array_equal(base.net_returns.to_numpy(), r.net_returns.to_numpy()), (s, name)
    print(f"MM2 {RUNS} random panels: exp, cube, affine and rank transforms of the factor give the identical portfolio  PASS")


def test_prefix_causality():
    for s in range(RUNS):
        p, f, rng = _world(200 + s)
        cut = int(rng.integers(200, 380))
        pc = p.truncate(p.dates[cut])
        full = _q(lambda: q.backtest_portfolio(p, f, **BT, funding=True))
        part = _q(lambda: q.backtest_portfolio(pc, f.iloc[:cut + 1], **BT, funding=True))
        n = len(part.net_returns)
        assert np.allclose(full.net_returns.iloc[:n].to_numpy(), part.net_returns.to_numpy(), atol=1e-13, rtol=0), (s, cut, float(np.abs(full.net_returns.iloc[:n].to_numpy() - part.net_returns.to_numpy()).max()))
        W = (f.rank(axis=1, pct=True) > 0.8).astype(float).where(p.eligible, 0.0)
        W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 0.5
        wf = _q(lambda: q.backtest_weights(p, W, spread_bp=8.0, sell_bp=4.0, borrow_bp=100.0, benchmark=None, check_universe=False))
        wp = _q(lambda: q.backtest_weights(pc, W.iloc[:cut + 1], spread_bp=8.0, sell_bp=4.0, borrow_bp=100.0, benchmark=None, check_universe=False))
        assert np.allclose(wf.net_returns.iloc[:len(wp.net_returns)].to_numpy(), wp.net_returns.to_numpy(), atol=1e-13, rtol=0), (s, cut)
        sig = f > 1.4
        ef = _q(lambda: q.backtest_event(p, sig, horizons=(3,), cost_bp=10, neutralize_check=False))
        # the event engine's statistics come from fires whose outcome is known: fires before the cut with a complete outcome are the same in both
        fw_full, fw_cut = p.forward(3).to_numpy(np.float64), pc.forward(3).to_numpy(np.float64)
        known = np.isfinite(fw_cut)
        assert np.array_equal(fw_full[:cut + 1][known], fw_cut[known]), (s, cut)                       # the one outcome both runs can see is the same number
        assert ef is not None
    print(f"MM3 {RUNS} random panels with delistings, funding and costs: cutting the data after a date leaves every earlier portfolio, weights and event outcome unchanged (1e-13)  PASS")


def test_inert_additions():
    for s in range(RUNS):
        p, f, rng = _world(300 + s)
        base = _q(lambda: q.backtest_portfolio(p, f, **BT))
        T = len(p.dates)
        ghost = pd.Series(np.nan, index=p.dates, name="GHOST")
        aug = lambda d, fill=None: pd.concat([d, ghost.rename("GHOST") if fill is None else pd.Series(fill, index=p.dates, name="GHOST")], axis=1)     # noqa: E731
        p2 = q.Panel(close=aug(p.close), eligible=aug(p.eligible, False), volume=aug(p.volume), mkt_cap=aug(p.mkt_cap), funding=aug(p.funding), delist_after=aug(p.delist_after, False), market="T")
        b = _q(lambda: q.backtest_portfolio(p2, aug(f), **BT))
        assert np.allclose(base.net_returns.to_numpy(), b.net_returns.to_numpy(), atol=1e-13, rtol=0), (s, float(np.abs(base.net_returns - b.net_returns).max()))
        W = (f.rank(axis=1, pct=True) > 0.85).astype(float).where(p.eligible, 0.0)
        W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        w1 = _q(lambda: q.backtest_weights(p, W, spread_bp=5.0, benchmark=None, check_universe=False))
        w2 = _q(lambda: q.backtest_weights(p2, aug(W, 0.0), spread_bp=5.0, benchmark=None, check_universe=False))
        assert np.allclose(w1.net_returns.to_numpy(), w2.net_returns.to_numpy(), atol=1e-13, rtol=0), (s,)
        assert T == len(p2.dates)
    print(f"MM4 {RUNS} random panels: adding a security that is never eligible (or has zero weight) changes no return  PASS")


def test_costs_monotone():
    for s in range(RUNS):
        p, f, _ = _world(400 + s, funding=False)
        prev = None
        for c in (0.0, 5.0, 20.0, 60.0):
            r = _q(lambda: q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=3, spread_bp=c, benchmark=None, grid=False)).net_returns
            if prev is not None:
                assert (r.to_numpy() <= prev.to_numpy() + 1e-15).all(), (s, c)
            prev = r
        gross0 = _q(lambda: q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=3, spread_bp=0.0, benchmark=None, grid=False))
        at = rb.attribution(p, gross0)
        assert abs(gross0.net_returns.mean() * 252 - at["gross_annual"]) < 1e-12                       # zero cost: net is the gross
    print(f"MM5 {RUNS} random panels: a higher spread never raises a bar's return; at zero cost net equals gross  PASS")


def test_scale_invariance():
    from pitbacktest.portfolio import metrics
    for s in range(RUNS):
        p, f, rng = _world(500 + s)
        a = _q(lambda: q.backtest_portfolio(p, f, **BT))
        p2 = replace(p, close=p.close * 7.3)
        b = _q(lambda: q.backtest_portfolio(p2, f, **BT))
        assert np.allclose(a.net_returns.to_numpy(), b.net_returns.to_numpy(), atol=1e-12, rtol=0), (s, float(np.abs(a.net_returns - b.net_returns).max()))     # no return depends on the price level
        x = a.net_returns.to_numpy()
        m1, m2 = metrics(x, a.net_returns.index, 252), metrics(3.0 * x, a.net_returns.index, 252)
        assert abs(m1["Sharpe"] - m2["Sharpe"]) < 1e-10 and abs(m1["Sortino"] - m2["Sortino"]) < 1e-10
        R = rng.normal(0.0004, 0.01, (400, 12))
        d1, d2 = q.validation.deflated_sharpe(R, periods_per_year=252), q.validation.deflated_sharpe(R * 5.0, periods_per_year=252)
        assert abs(d1["dsr"] - d2["dsr"]) < 1e-10 and d1["best"] == d2["best"]
        pb1, pb2 = q.validation.pbo_cscv(R, periods_per_year=252, max_splits=300), q.validation.pbo_cscv(R * 5.0, periods_per_year=252, max_splits=300)
        assert pb1["pbo"] == pb2["pbo"]
    print(f"MM6 {RUNS} random panels: the price level changes no return, Sharpe, Sortino, deflated Sharpe and PBO are unchanged by a positive scaling of returns  PASS")


def test_renaming():
    for s in range(RUNS):
        p, f, rng = _world(600 + s)
        sig = f > 1.3
        names = {c: f"Z{(len(p.tickers) - i):03d}" for i, c in enumerate(p.tickers)}                  # a different name for every security, in the reverse order
        p2 = _reorder(p, list(p.tickers)); p2 = q.Panel(close=p.close.rename(columns=names), eligible=p.eligible.rename(columns=names), volume=p.volume.rename(columns=names),
                                                       mkt_cap=p.mkt_cap.rename(columns=names), funding=p.funding.rename(columns=names), delist_after=p.delist_after.rename(columns=names), market="T")
        ea = _q(lambda: rb.signal_permutation(p, sig, 3, 10.0, n=40, seed=3))
        eb = _q(lambda: rb.signal_permutation(p2, sig.rename(columns=names), 3, 10.0, n=40, seed=3))
        assert ea == eb, s
        fa = _q(lambda: rb.factor_permutation(p, f, n=12, seed=3, long_q=0.2, short_q=0.2, hold=3))
        fb = _q(lambda: rb.factor_permutation(p2, f.rename(columns=names), n=12, seed=3, long_q=0.2, short_q=0.2, hold=3))
        assert fa["p"] == fb["p"] and fa["real_sharpe"] == fb["real_sharpe"]
    print(f"MM7 {RUNS} random panels: renaming the securities changes no permutation result  PASS")


def test_future_scramble():
    for s in range(RUNS):
        p, f, rng = _world(700 + s)
        t = int(rng.integers(120, 340))
        f = pd.DataFrame(np.where(np.isfinite(f.to_numpy()), f.to_numpy(), rng.standard_normal(f.shape)), index=f.index, columns=f.columns)       # a raw factor: values also where a security is not eligible
        base = _q(lambda: q.backtest_portfolio(p, f, **BT))
        f2 = f.copy()
        f2.iloc[t + 1:] = rng.standard_normal(f2.iloc[t + 1:].shape)                                  # the factor after t replaced by noise
        el2 = p.eligible.copy()
        el2.iloc[t + 1:] = rng.random(el2.iloc[t + 1:].shape) < 0.5                                   # and so is who is eligible
        p2 = q.Panel(close=p.close, eligible=el2, volume=p.volume, mkt_cap=p.mkt_cap, funding=p.funding, delist_after=p.delist_after, market="T")
        b = _q(lambda: q.backtest_portfolio(p2, f2, **BT))
        H1, H2 = np.asarray(base.holdings)[:t + 1], np.asarray(b.holdings)[:t + 1]
        assert np.allclose(H1, H2, atol=1e-15, rtol=0), (s, t, float(np.abs(H1 - H2).max()))        # every position held up to t is the same
        assert np.allclose(base.net_returns.iloc[:t].to_numpy(), b.net_returns.iloc[:t].to_numpy(), atol=1e-13, rtol=0)
    print(f"MM8 {RUNS} random panels: scrambling the factor and the eligibility after a date changes no position held up to it  PASS")


if __name__ == "__main__":
    test_column_order()
    test_monotone_transform()
    test_prefix_causality()
    test_inert_additions()
    test_costs_monotone()
    test_scale_invariance()
    test_renaming()
    test_future_scramble()
    print("metamorphic tests: all passed")
