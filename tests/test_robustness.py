"""The robustness battery against answers worked out by hand or by a plain loop.

RB1 market        capture ratios, hit rates, excess CAGR, information ratio and relative drawdown of constructed series
RB2 mean tests    the exact sign test, the Wilson interval, the Sharpe t, and the Newey-West t against statsmodels when it is installed
RB3 subperiods    yearly returns and the halves equal a direct computation
RB4 walk-forward  equals a plain-loop reference fold by fold; an always-better setting is always chosen; the embargo moves the test start; noise does not transfer
RB5 plateau       a hill, a spike and a corner cell
RB6 many tests    BH and Holm equal the textbook example and statsmodels
RB7 permutations  a planted factor and a planted event reach the smallest possible p, noise does not, and the null's expected win rate equals the exact expectation
RB11 calibration  information-free persistent signals are called significant about as often as the nominal 5 percent with the new nulls, and much more often with the old ones
RB8 decomposition a strategy that is a style factor loads 1 on it, a market strategy plus a constant has beta 1 and that alpha
RB9 attribution   legs, names, thirds and groups of a hand-built book equal direct sums; gross minus each cost equals net in both engines
RB10 review fixes decompose adds up on the rows it used, zero bars stay out of the win rate, a NaN column can be chosen, unknown styles are NaN, unclassified positions have a row
"""
from __future__ import annotations
import math, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import robustness as rb
from test_argument_checks import _raises
from test_synthetic import make_panel


def _q(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def test_market_relative():
    idx = pd.bdate_range("2020-01-01", periods=240)
    m = pd.Series(np.tile([0.01, -0.01], 120), index=idx)
    s = m + 0.0005                                                                   # the same days, 5 bp better every day
    r = rb.market_relative(s, m)
    assert abs(r["up_capture"] - 1.05) < 1e-12 and abs(r["down_capture"] - 0.95) < 1e-12, r             # (0.01+0.0005)/0.01 and (-0.01+0.0005)/-0.01
    assert abs(r["beta"] - 1.0) < 1e-12 and abs(r["correlation"] - 1.0) < 1e-12
    assert r["hit_rate_month"] == 1.0 and r["hit_rate_year"] == 1.0                                       # better in every month and year
    assert abs(r["excess_cagr"] - ((np.prod(1 + s) ** (252 / 240) - 1) - (np.prod(1 + m) ** (252 / 240) - 1))) < 1e-12
    assert r["tracking_error"] < 1e-12 and np.isnan(r["information_ratio"])                              # a constant difference has no spread to divide by
    rng = np.random.default_rng(0)
    noisy = m + rng.normal(0.0004, 0.003, 240)
    r2 = rb.market_relative(noisy, m)
    d = (noisy - m).to_numpy()
    assert abs(r2["information_ratio"] - d.mean() / d.std(ddof=1) * math.sqrt(252)) < 1e-12 and abs(r2["tracking_error"] - d.std(ddof=1) * math.sqrt(252)) < 1e-12
    # relative drawdown: the benchmark doubles up then halves, the strategy sits in cash: its ratio to the benchmark first falls from 1 to 1/1.21
    b3 = pd.Series([0.0] * 17 + [0.10, 0.10, -0.50], index=idx[:20])
    r3 = rb.market_relative(pd.Series(0.0, index=idx[:20]), b3)
    assert abs(r3["max_relative_drawdown"] - (1 / 1.21 - 1)) < 1e-12, r3["max_relative_drawdown"]
    b4 = pd.Series([0.10] + [0.0] * 19, index=idx[:20])                                # the benchmark gains on the very first bar: the start (ratio 1) is the peak
    r4 = rb.market_relative(pd.Series(0.0, index=idx[:20]), b4)
    assert abs(r4["max_relative_drawdown"] - (1 / 1.10 - 1)) < 1e-12, r4["max_relative_drawdown"]
    _raises(lambda: rb.market_relative(s.iloc[:10], m.iloc[:10]), "at least 20")
    _raises(lambda: rb.market_relative(s.to_numpy(), m), "Series")
    print("RB1 capture 1.05 / 0.95, hit rate 1.0, excess CAGR, information ratio and relative drawdown 1/1.21-1 equal hand computations  PASS")


def test_mean_tests():
    idx = pd.bdate_range("2020-01-01", periods=100)
    x = pd.Series(np.r_[np.full(60, 0.01), np.full(40, -0.01)], index=idx)           # 60 wins, 40 losses
    t = rb.mean_tests(x)
    assert abs(t["sign_p"] - 0.056887) < 2e-6, t["sign_p"]                           # exact two-sided binomial(100, .5) for 60 wins
    lo, hi = t["win_ci"]
    assert abs(lo - 0.5020) < 5e-4 and abs(hi - 0.6906) < 5e-4, (lo, hi)             # Wilson interval of 0.6 at n = 100
    big = pd.Series(np.r_[np.full(1600, 0.01), np.full(1400, -0.01)], index=pd.bdate_range("2000-01-01", periods=3000))     # 3000 bars: 2.0 ** n would overflow
    pb = rb.mean_tests(big)["sign_p"]
    z = (1600 - 1500) / math.sqrt(3000 * 0.25)
    assert abs(pb - math.erfc(z / math.sqrt(2))) < 5e-3 and 0.0 < pb < 0.05, pb                                       # close to the normal approximation 0.0143
    assert rb.mean_tests(pd.Series(np.tile([0.01, -0.01], 600), index=pd.bdate_range("2000-01-01", periods=1200)))["sign_p"] == 1.0
    rng = np.random.default_rng(1)
    y = pd.Series(rng.normal(0.001, 0.01, 500), index=pd.bdate_range("2020-01-01", periods=500))
    ty = rb.mean_tests(y)
    assert abs(ty["sharpe_t"] - y.mean() / y.std(ddof=1) * math.sqrt(500)) < 1e-12 and abs(ty["sharpe"] - y.mean() / y.std(ddof=1) * math.sqrt(252)) < 1e-12
    try:
        import statsmodels.api as sm
        res = sm.OLS(y.to_numpy(), np.ones(500)).fit(cov_type="HAC", cov_kwds={"maxlags": ty["lag"], "use_correction": False})
        assert abs(ty["nw_t"] - float(res.tvalues[0])) < 1e-9, (ty["nw_t"], res.tvalues[0])
        note = "equals statsmodels HAC"
    except ImportError:
        note = "statsmodels not installed: Newey-West t not cross-checked"
    assert 0.0 <= ty["nw_p"] <= 1.0 and abs(ty["nw_p"] - math.erfc(abs(ty["nw_t"]) / math.sqrt(2))) < 1e-15
    _raises(lambda: rb.mean_tests(y.iloc[:10]), "at least 30")
    print(f"RB2 sign test p 0.056887, Wilson interval (0.502, 0.6906), Sharpe t = Sharpe x sqrt(n): equal; Newey-West t {note}  PASS")


def test_subperiods():
    idx = pd.bdate_range("2019-01-01", periods=3 * 261)
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    r = rb.subperiods(s)
    start = 1.0
    for y, g in s.groupby(s.index.year):
        eq = (1 + g).cumprod() * start
        assert abs(r["years"].loc[y, "ret"] - (eq.iloc[-1] / start - 1)) < 1e-12
        assert abs(r["years"].loc[y, "max_dd"] - (eq / np.maximum(eq.cummax(), start) - 1).min()) < 1e-12
        start = float(eq.iloc[-1])
    h = len(s) // 2
    sh = lambda a: a.mean() / a.std(ddof=1) * math.sqrt(252)            # noqa: E731
    assert abs(r["halves"][0] - sh(s.iloc[:h])) < 1e-12 and abs(r["halves"][1] - sh(s.iloc[h:])) < 1e-12
    assert r["share_positive"] == float((r["years"]["ret"] > 0).mean()) and r["worst_year"] == r["years"]["ret"].min()
    print("RB3 yearly returns, drawdowns from the year before's end, and the two halves' Sharpe equal a direct computation  PASS")


def _wf_reference(R, train, test, embargo, expanding=True):
    """Plain loops: the same procedure written independently."""
    T, K = R.shape
    out, chosen, k0 = [], [], train
    sh = lambda a: a.mean() / a.std(ddof=1) * math.sqrt(252) if a.std(ddof=1) > 1e-12 else float("nan")   # noqa: E731
    while k0 + embargo + test <= T:
        lo = 0 if expanding else k0 - train
        scores = [sh(R[lo:k0, j]) for j in range(K)]
        j = int(np.nanargmax(scores))
        chosen.append(j)
        out.extend(R[k0 + embargo:k0 + embargo + test, j].tolist())
        k0 += test
    return np.array(out), chosen


def test_walk_forward():
    rng = np.random.default_rng(3)
    T = 700
    idx = pd.bdate_range("2019-01-01", periods=T)
    a = np.where(np.arange(T) < 350, 0.003, -0.003) + rng.normal(0, 0.01, T)           # great in the first half, bad in the second
    b = 0.0004 + rng.normal(0, 0.01, T)
    c = rng.normal(0, 0.01, T)
    R = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)
    for expanding, emb in ((True, 0), (True, 5), (False, 0)):
        wf = rb.walk_forward(R, train=200, test=50, embargo=emb, expanding=expanding)
        ref, chosen = _wf_reference(R.to_numpy(), 200, 50, emb, expanding)
        assert np.array_equal(wf["oos"].to_numpy(), ref), (expanding, emb)
        assert [list(R.columns).index(x) for x in wf["folds"]["chosen"]] == chosen
        assert wf["n_folds"] == len(chosen)
    wf = rb.walk_forward(R, train=200, test=50, embargo=5)
    assert (wf["folds"]["test_start"] == [R.index[200 + 5 + 50 * i] for i in range(wf["n_folds"])]).all()          # the test starts after the embargo
    assert (wf["folds"]["train_end"] == [R.index[200 - 1 + 50 * i] for i in range(wf["n_folds"])]).all()
    # an expanding window remembers the good half for a long time (that is its nature); a rolling window of 100 bars drops A soon after it stops working
    strong = pd.DataFrame({"A": np.where(np.arange(T) < 350, 0.005, -0.005) + rng.normal(0, 0.01, T), "B": 0.0004 + rng.normal(0, 0.01, T), "C": rng.normal(0, 0.01, T)}, index=idx)
    roll = rb.walk_forward(strong, train=100, test=50, expanding=False)
    f = roll["folds"]
    assert (f.loc[f["test_end"] < strong.index[350], "chosen"] == "A").all(), f[["test_start", "chosen"]]
    assert (f.loc[f["test_start"] > strong.index[350 + 100], "chosen"] != "A").all(), f[["test_start", "chosen"]]
    always = pd.DataFrame({"good": 0.003 + rng.normal(0, 0.01, T), "zero": rng.normal(0, 0.01, T), "bad": -0.002 + rng.normal(0, 0.01, T)}, index=idx)
    w2 = rb.walk_forward(always, train=150, test=50)
    assert (w2["folds"]["chosen"] == "good").mean() > 0.9 and w2["oos_sharpe"] > 1.5 and w2["efficiency"] > 0.5, w2["chosen_counts"]
    noise = pd.DataFrame(rng.normal(0, 0.01, (T, 30)), index=idx)
    w3 = rb.walk_forward(noise, train=200, test=50)
    assert w3["mean_is_sharpe"] > 1.0 and w3["oos_sharpe"] < w3["mean_is_sharpe"] - 1.0 and w3["efficiency"] < 0.6, w3     # picking the best of 30 noise columns does not transfer
    _raises(lambda: rb.walk_forward(R, train=2, test=50), "at least 10")
    _raises(lambda: rb.walk_forward(R, train=650, test=100), "needs")
    _raises(lambda: rb.walk_forward(R.iloc[:, :1], train=200, test=50), "at least two columns")
    print(f"RB4 walk-forward equals a plain loop in three settings; the embargo moves the test start; an always-better setting is chosen {(w2['folds']['chosen'] == 'good').mean():.0%} of the time; "
          f"the best of 30 noise columns has in-sample Sharpe {w3['mean_is_sharpe']:.2f} and out-of-sample {w3['oos_sharpe']:.2f}  PASS")


def test_plateau():
    g = np.full((5, 5), 0.2); g[2, 2] = 1.0
    hill = g.copy(); hill[1:4, 1:4] = 0.9; hill[2, 2] = 1.0
    t = pd.DataFrame(hill, index=range(5), columns=list("abcde"))
    r = rb.parameter_plateau(t)
    assert r["best_at"] == (2, "c") and r["n_neighbours"] == 8 and abs(r["neighbour_ratio"] - 0.9) < 1e-12 and r["verdict"] == "plateau" and r["share_within"] == 1.0
    sp = g.copy(); sp[1:4, 1:4] = 0.0; sp[2, 2] = 1.0
    r2 = rb.parameter_plateau(pd.DataFrame(sp))
    assert r2["neighbour_ratio"] == 0.0 and r2["verdict"] == "spike" and r2["share_within"] == 0.0
    mid = g.copy(); mid[1:4, 1:4] = 0.5; mid[2, 2] = 1.0
    assert rb.parameter_plateau(pd.DataFrame(mid))["verdict"] == "hill"
    corner = np.zeros((4, 4)); corner[0, 0] = 2.0; corner[0, 1] = 1.0; corner[1, 0] = 1.0; corner[1, 1] = 0.0
    r4 = rb.parameter_plateau(pd.DataFrame(corner))
    assert r4["best_at"] == (0, 0) and r4["n_neighbours"] == 3 and abs(r4["neighbour_ratio"] - (2.0 / 3.0) / 2.0) < 1e-12        # three neighbours: 1, 1, 0
    s1 = rb.parameter_plateau(pd.Series([0.1, 0.5, 1.0, 0.6, 0.2], index=[5, 10, 20, 40, 80]))
    assert s1["best_at"][0] == 20 and s1["n_neighbours"] == 2 and abs(s1["neighbour_ratio"] - 0.55) < 1e-12
    _raises(lambda: rb.parameter_plateau(pd.DataFrame([[1.0, 2.0]])), "at least 3")
    print("RB5 a hill (ratio 0.9), a spike (0), a corner cell with three neighbours and a one-parameter series are classified as worked out  PASS")


def test_multiple_testing():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(rb.bh_fdr(p), [0.02, 0.04, 0.04, 0.02])                        # the textbook Benjamini-Hochberg example
    assert np.allclose(rb.holm(p), [0.03, 0.06, 0.06, 0.02])
    rng = np.random.default_rng(4)
    pv = rng.random(200) ** 2
    try:
        from statsmodels.stats.multitest import multipletests
        assert np.allclose(rb.bh_fdr(pv), multipletests(pv, method="fdr_bh")[1], atol=1e-12)
        assert np.allclose(rb.holm(pv), multipletests(pv, method="holm")[1], atol=1e-12)
        note = "and statsmodels"
    except ImportError:
        note = "(statsmodels not installed)"
    assert (rb.bh_fdr(pv) >= pv - 1e-15).all() and (rb.holm(pv) >= rb.bh_fdr(pv) - 1e-15).all() and rb.bh_fdr([1.0, 1.0]).tolist() == [1.0, 1.0]
    _raises(lambda: rb.bh_fdr([0.1, 1.2]), "[0, 1]")
    _raises(lambda: rb.holm([]), "non-empty")
    print(f"RB6 Benjamini-Hochberg and Holm equal the textbook example {note}; adjusted values never fall below the raw ones  PASS")


def test_permutations():
    p, rng = make_panel(n_days=700, n_stocks=60)
    fwd = p.close.pct_change().shift(-2)
    z = (fwd.rank(axis=1, pct=True) - .5) * math.sqrt(12)
    noise = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    planted = (0.15 * z + noise).where(p.eligible)
    kw = dict(long_q=0.2, short_q=0.2, hold=1, spread_bp=0.0)
    r1 = _q(lambda: rb.factor_permutation(p, planted, n=30, seed=1, **kw))
    assert abs(r1["p"] - 1 / 31) < 1e-12 and r1["real_sharpe"] > r1["null_p99"], r1                  # no shuffle did as well as the planted factor
    pn = [_q(lambda: rb.factor_permutation(p, noise.where(p.eligible).add(0.0) * (1 + 0 * i), n=30, seed=10 + i, **kw))["p"] for i in range(1)]
    r2 = _q(lambda: rb.factor_permutation(p, noise.where(p.eligible), n=30, seed=2, **kw))
    assert r2["p"] > 0.05 and abs(r2["null_mean"]) < 0.6 and r2["null_std"] > 0.2, r2                # noise is not distinguished from its own shuffles
    _raises(lambda: rb.factor_permutation(p, planted, n=5), "at least 10")
    # a factor with no information but slowly moving scores, run with a 20 bp spread: with costs in the null it looked significant (p 0.02) because the shuffles lose to the costs
    slow = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers).ewm(alpha=0.03).mean().where(p.eligible)
    sl = _q(lambda: rb.factor_permutation(p, slow, n=40, seed=1, long_q=0.2, short_q=0.2, hold=5, spread_bp=20.0))
    assert sl["p"] > 0.05, sl["p"]
    gross = _q(lambda: q.backtest_portfolio(p, slow, long_q=0.2, short_q=0.2, hold=5, spread_bp=0.0, benchmark=None, grid=False, funding=False)).metrics["Sharpe"]
    assert sl["real_sharpe"] == gross                                                                          # the real figure is the gross one, whatever spread was passed
    # events: a signal that cheats (fires on names that go up) reaches the smallest p; the null's expected win rate equals the exact expectation
    fw5 = p.forward(5).to_numpy(np.float64)
    cheat = pd.DataFrame(fw5 > np.nanpercentile(fw5, 90), index=p.dates, columns=p.tickers) & p.eligible
    c = rb.signal_permutation(p, cheat, 5, cost_bp=10.0, n=100, seed=0)
    assert c["p_win"] == 1 / 101 and c["p_mean"] == 1 / 101 and c["real_win_rate"] > 0.95
    rand_sig = pd.DataFrame(rng.random(p.close.shape) < 0.03, index=p.dates, columns=p.tickers) & p.eligible
    ev = rb.signal_permutation(p, rand_sig, 5, cost_bp=10.0, n=200, seed=3, method="random_picks")
    usable = p.eligible.to_numpy(bool) & np.isfinite(fw5)
    fire = rand_sig.to_numpy(bool) & usable
    exp_win = sum(fire[i].sum() * ((fw5[i][usable[i]] - 0.001) > 0).mean() for i in range(len(fire)) if fire[i].any()) / fire.sum()
    se = math.sqrt(exp_win * (1 - exp_win) / fire.sum())
    assert abs(ev["null_win_rate_mean"] - exp_win) < 3 * se / math.sqrt(1) and ev["p_win"] > 0.05, (ev["null_win_rate_mean"], exp_win, se)
    assert ev["n_fires"] == int(fire.sum())
    _raises(lambda: rb.signal_permutation(p, rand_sig & False, 5), "at least 30")
    _raises(lambda: rb.signal_permutation(p, rand_sig, 0), "horizon")
    print(f"RB7 a planted factor and a cheating event signal reach the smallest p (1/31, 1/101); noise p {r2['p']:.2f}; the random null's win rate {ev['null_win_rate_mean']:.4f} equals the exact expectation {exp_win:.4f}  PASS")


def test_decomposition():
    p, rng = make_panel(n_days=900, n_stocks=80)
    st = _q(lambda: rb.style_factor_returns(p))
    assert {"market", "size", "reversal", "lowvol", "illiquidity"} <= set(st.columns) and "momentum" in st.columns
    sz = _q(lambda: q.backtest_portfolio(p, -p.mkt_cap.where(p.mkt_cap > 0), long_q=0.3, short_q=0.3, hold=5, spread_bp=0.0, benchmark=None, grid=False, funding=False)).net_returns
    assert float(np.abs(sz - st["size"]).max()) < 1e-15                                # the style factor is that backtest
    d = rb.decompose(sz + 0.0, st)
    assert abs(d["factors"]["size"]["beta"] - 1.0) < 1e-6 and d["r2"] > 0.999 and abs(d["alpha_annual"]) < 1e-6, d
    mk = st["market"].dropna()
    fw = p.forward(1).where(p.eligible).mean(axis=1)
    assert float(np.abs(mk - fw.reindex(mk.index)).max()) < 1e-15                      # the market is the equal-weight mean of the same one-bar return
    d2 = rb.decompose(mk + 0.0002, st[["market"]].reindex(mk.index))
    assert abs(d2["factors"]["market"]["beta"] - 1.0) < 1e-9 and abs(d2["alpha_annual"] - 0.0002 * 252) < 1e-9, d2
    assert abs(d2["factors"]["market"]["contribution_annual"] - mk.mean() * 252) < 1e-9
    _raises(lambda: rb.decompose(mk, pd.DataFrame({"x": np.zeros(len(mk))}, index=mk.index)), "no usable factor")
    print(f"RB8 a size strategy loads {d['factors']['size']['beta']:.6f} on size with R2 {d['r2']:.6f}; a market strategy plus 2 bp has beta 1 and alpha 5.04 percent  PASS")


def test_attribution():
    T, N = 300, 12
    idx = pd.bdate_range("2020-01-01", periods=T)
    names = [f"S{i:02d}" for i in range(N)]
    close = pd.DataFrame(100.0, index=idx, columns=names)
    close["S00"] = 100.0 * 1.01 ** np.arange(T)                    # A: +1 percent every bar
    close["S01"] = 100.0 * 0.995 ** np.arange(T)                   # B: -0.5 percent every bar
    elig = pd.DataFrame(True, index=idx, columns=names); elig.iloc[:60] = False
    cap = pd.DataFrame(np.tile(1e5 * (1 + 0.01 * np.arange(N)), (T, 1)), index=idx, columns=names)
    cap["S00"], cap["S01"] = 1e9, 1e3                              # A the largest, B the smallest
    vol = pd.DataFrame(1e4, index=idx, columns=names); vol["S00"] = 1e6; vol["S01"] = 1.0         # A the most liquid, B the least
    p = q.Panel(close=close, eligible=elig, mkt_cap=cap, volume=vol, market="T")
    W = pd.DataFrame(0.0, index=idx, columns=names); W.iloc[60:, 0] = 1.0; W.iloc[60:, 1] = -1.0
    r = _q(lambda: q.backtest_weights(p, W, spread_bp=10.0, benchmark=None, check_universe=False))
    n = len(r.net_returns)
    years = n / 252
    active = n - 60
    at = rb.attribution(p, r, groups=pd.Series({"S00": "tech", "S01": "energy"}), n_top=3)
    assert abs(at["long_annual"] - active * 0.01 / years) < 1e-9 and abs(at["short_annual"] - active * 0.005 / years) < 1e-9, at
    assert abs(at["long_annual"] + at["short_annual"] - at["gross_annual"]) < 1e-12
    assert list(at["names"]["top"])[:2] == ["S00", "S01"] and at["names"]["n_positive"] == 2 and at["names"]["n_held"] == 2 and at["names"]["top_share"] == 1.0
    assert abs(at["names"]["best_days_share"] - 5 / active) < 1e-9, at["names"]["best_days_share"]          # every active bar earns the same, so the best five are 5/active of the total
    for key, big, small in (("by_size", "largest third", "smallest third"), ("by_liquidity", "most liquid third", "least liquid third")):
        assert abs(at[key][big]["contribution_annual"] - at["long_annual"]) < 1e-9 and abs(at[key][small]["contribution_annual"] - at["short_annual"]) < 1e-9, key
        assert abs(at[key][big]["exposure_share"] - 0.5) < 1e-12 and at[key]["middle third"]["contribution_annual"] == 0.0
    assert abs(at["by_group"]["tech"]["contribution_annual"] - at["long_annual"]) < 1e-9 and abs(at["by_group"]["energy"]["contribution_annual"] - at["short_annual"]) < 1e-9
    assert at["by_group"]["(no group)"]["contribution_annual"] == 0.0
    wf = at["waterfall"]
    assert abs(wf["residual_bp"]) < 1e-6 and abs(dict(wf["costs"])["spread"] - 10.0 / n * 252) < 1e-9, wf       # one entry of 2 units at 5 bp each, spread over the sample
    assert abs(wf["gross_bp"] - sum(c for _, c in wf["costs"]) - wf["net_bp"]) < 1e-6
    # the portfolio engine reports its costs together; the identity holds there too, with impact and funding
    p2, rng = make_panel(n_days=700, n_stocks=60)
    f = pd.DataFrame(rng.standard_normal(p2.close.shape), index=p2.dates, columns=p2.tickers)
    for kw in (dict(spread_bp=20.0), dict(spread_bp=10.0, sell_bp=15.0, impact=q.ImpactModel(aum=2e7))):
        rp = _q(lambda: q.backtest_portfolio(p2, f, long_q=0.2, short_q=0.2, hold=5, benchmark=None, **kw))
        a2 = rb.attribution(p2, rp)
        assert abs(a2["waterfall"]["residual_bp"]) < 1e-6, (kw, a2["waterfall"])
        assert abs(a2["long_annual"] + a2["short_annual"] - a2["gross_annual"]) < 1e-12
    # funding and borrow cost something in these books: the waterfall must carry them or its residual is not zero
    fund = pd.DataFrame(0.0, index=idx, columns=names); fund["S00"] = 0.0004; fund["S01"] = -0.0002
    pf = q.Panel(close=close, eligible=elig, funding=fund, market="T")
    rf = _q(lambda: q.backtest_weights(pf, W, spread_bp=10.0, borrow_bp=300.0, benchmark=None, check_universe=False))
    wf_ = rb.attribution(pf, rf)["waterfall"]
    assert abs(wf_["residual_bp"]) < 1e-6 and {"short borrow", "funding", "spread"} <= {lb for lb, _ in wf_["costs"]} and abs(dict(wf_["costs"])["funding"]) > 100 and dict(wf_["costs"])["short borrow"] > 100, wf_
    fq = pd.DataFrame(rng.normal(0, 0.0003, p2.close.shape), index=p2.dates, columns=p2.tickers)
    p3 = q.Panel(close=p2.close, eligible=p2.eligible, funding=fq, mkt_cap=p2.mkt_cap, market="T")
    rp3 = _q(lambda: q.backtest_portfolio(p3, f, long_q=0.2, short_q=0.2, hold=5, spread_bp=10.0, benchmark=None))
    w3 = rb.attribution(p3, rp3)["waterfall"]
    assert abs(w3["residual_bp"]) < 1e-6 and any(lb == "funding" and abs(v) > 1 for lb, v in w3["costs"]), w3
    _raises(lambda: rb.attribution(p, r, groups=["a"]), "Series")
    _raises(lambda: rb.attribution(p, r, n_top=0), "at least 1")
    import dataclasses
    _raises(lambda: rb.attribution(p, dataclasses.replace(r, holdings=None)), "no holdings")
    print(f"RB9 long {at['long_annual']:.4f} + short {at['short_annual']:.4f} = gross a year, thirds and groups match, gross minus costs equals net to 1e-6 bp in both engines  PASS")


def test_review_fixes():
    rng = np.random.default_rng(11)
    # decompose: alpha plus the contributions is the mean of the rows the regression used (it was not, when the factor had more rows than the strategy)
    idx = pd.bdate_range("2012-01-02", periods=2500)
    f1 = pd.Series(rng.normal(0.0008, 0.01, 2500), index=idx, name="mom")
    f2 = pd.Series(rng.normal(-0.0003, 0.012, 2500), index=idx, name="vol"); f2.iloc[:900] = np.nan
    y = (0.5 * f1 + 0.3 * f2.fillna(0) + rng.normal(0.0002, 0.004, 2500)).iloc[-600:]
    d = rb.decompose(y, pd.concat([f1, f2], axis=1))
    used = pd.concat([y.rename("y"), f1, f2], axis=1).dropna()
    assert abs(d["alpha_annual"] + sum(v["contribution_annual"] for v in d["factors"].values()) - used["y"].mean() * 252) < 1e-9, d
    d1 = rb.decompose(y, f1.to_frame())
    assert abs(d1["alpha_annual"] + d1["factors"]["mom"]["contribution_annual"] - y.mean() * 252) < 1e-9
    # mean_tests: a bar with exactly zero return is no trade
    z = pd.Series(rng.normal(0.0008, 0.01, 700), index=pd.bdate_range("2019-01-01", periods=700)); z.iloc[::7] = 0.0
    mt = rb.mean_tests(z)
    nz = z[z != 0]
    assert mt["n_zero"] == 100 and abs(mt["win_rate"] - (nz > 0).mean()) < 1e-15 and abs(mt["sign_p"] - rb._binom_two_sided(int((nz > 0).sum()), len(nz))) < 1e-15
    assert mt["win_ci"][0] < mt["win_rate"] < mt["win_ci"][1]
    # walk-forward: a column with leading NaN bars is scored on the bars it has and can be chosen; the in-sample figure is a Sharpe whatever `select` is
    T = 500
    ix = pd.bdate_range("2019-01-01", periods=T)
    R = pd.DataFrame({"a": rng.normal(0, 0.01, T), "b": rng.normal(0.0004, 0.01, T), "c": 0.004 + rng.normal(0, 0.01, T)}, index=ix)
    R.iloc[:30, 2] = np.nan
    wf = rb.walk_forward(R, train=100, test=50)
    assert wf["chosen_counts"].get("c", 0) >= 0.8 * wf["n_folds"], wf["chosen_counts"]
    wf2 = rb.walk_forward(R, train=100, test=50, select=lambda col: float(np.mean(col)) * 1e4)
    f0 = wf2["folds"].iloc[0]
    tr = R.iloc[:100][f0["chosen"]].dropna().to_numpy()
    assert abs(f0["is_sharpe"] - tr.mean() / tr.std(ddof=1) * math.sqrt(252)) < 1e-9 and abs(f0["is_score"] - tr.mean() * 1e4) < 1e-9 and wf2["mean_is_sharpe"] < 30    # in Sharpe units
    # style factors: unknown before the characteristic exists
    p, _ = make_panel(n_days=700, n_stocks=60)
    st = _q(lambda: rb.style_factor_returns(p))
    assert st["momentum"].loc[:p.dates[250]].isna().all() and st["momentum"].loc[p.dates[300]:].notna().all()          # the 12-1 month characteristic needs 252 bars
    assert st["reversal"].first_valid_index() >= p.dates[60] and st["momentum"].first_valid_index() > st["reversal"].first_valid_index()
    # attribution: positions held on a bar where the name was not eligible are their own row, and the rows add up to the whole
    pe = make_panel(n_days=700, n_stocks=60)[0]
    el = pd.DataFrame(np.random.default_rng(4).random(pe.close.shape) < 0.7, index=pe.dates, columns=pe.tickers)
    pp = q.Panel(close=pe.close, eligible=el | (pe.eligible & False), mkt_cap=pe.mkt_cap, volume=pe.volume, market="T")
    fct = -pp.close.pct_change(5)
    res = _q(lambda: q.backtest_portfolio(pp, fct, long_q=0.2, short_q=0.2, hold=5, spread_bp=0.0, benchmark=None, grid=False))
    at = rb.attribution(pp, res)
    assert "not classified" in at["by_size"] and abs(sum(v["contribution_annual"] for v in at["by_size"].values()) - at["gross_annual"]) < 1e-9
    assert abs(sum(v["exposure_share"] for v in at["by_size"].values()) - 1.0) < 1e-9
    # plateau: a one-parameter table has a one-element best_at and says why it cannot tell
    pl = rb.parameter_plateau(pd.Series([-0.5, -0.2, -0.9, -1.0], index=[3, 5, 10, 20]))
    assert pl["best_at"] == (5,) and pl["verdict"] == "undetermined" and pl["reason"] == "no setting has a positive Sharpe ratio"
    assert rb.parameter_plateau(pd.Series([0.1, 0.9, 0.8, 0.2], index=[3, 5, 10, 20]))["reason"] == ""
    print("RB10 decompose adds up on its own rows, zero bars stay out of the win rate, a NaN column can be chosen, unknown styles are NaN, unclassified positions have a row  PASS")


def _iid_panel(seed, T=500, N=50, drift_sd=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T)
    cols = [f"S{i:02d}" for i in range(N)]
    r = rng.normal(0, 0.012, (T, N)) + rng.normal(0, drift_sd / 252, N)               # a true drift of its own for each security
    c = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=cols)
    el = pd.DataFrame(True, index=idx, columns=cols); el.iloc[:20] = False
    return q.Panel(close=c, eligible=el, volume=pd.DataFrame(1e6, index=idx, columns=cols), market="T"), rng


def test_shift_null_exact_and_calibrated():
    # the time-shifted null against a plain loop over every admissible shift: its mean win rate is the average of the win rates of all shifts
    p, rng = make_panel(n_days=400, n_stocks=30)
    sig = pd.DataFrame(rng.random(p.close.shape) < 0.04, index=p.dates, columns=p.tickers) & p.eligible
    fw = p.forward(5).to_numpy(np.float64)
    usable = p.eligible.to_numpy(bool) & np.isfinite(fw)
    f0 = sig.to_numpy(bool)
    T = len(fw); lo = max(2 * 5, 21)
    wins = []
    for sft in range(lo, T - lo + 1):
        rolled = np.roll(f0, sft, axis=0) & usable
        v = fw[rolled] - 0.001
        if len(v):
            wins.append((v > 0).mean())
    r = rb.signal_permutation(p, sig, 5, 10.0, n=400, seed=5)
    sd = float(np.std(wins, ddof=1))
    assert abs(r["null_win_rate_mean"] - np.mean(wins)) < 4 * sd / math.sqrt(400), (r["null_win_rate_mean"], np.mean(wins), sd)
    assert r["method"] == "shift" and rb.signal_permutation(p, sig, 5, 10.0, n=400, seed=5) == r
    cheat = pd.DataFrame(fw > np.nanpercentile(fw, 90), index=p.dates, columns=p.tickers) & p.eligible
    assert rb.signal_permutation(p, cheat, 5, 10.0, n=100, seed=0)["p_win"] == 1 / 101                   # moving the cheating signal in time destroys it
    _raises(lambda: rb.signal_permutation(p, sig, 5, method="x"), "method")
    _raises(lambda: rb.signal_permutation(p, sig & False, 5, n=20), "fires")


def test_calibration():
    # (a) an event signal with no information whose fires persist: each security fires on one 60-bar spell. 60 panels with iid returns.
    K = 60
    rej = {"shift": 0, "random_picks": 0}
    for s in range(K):
        p, rng = _iid_panel(s)
        sig = pd.DataFrame(False, index=p.dates, columns=p.tickers)
        for j in range(len(p.tickers)):
            a = int(rng.integers(30, 430)); sig.iloc[a:a + 60, j] = True
        for m in rej:
            rej[m] += rb.signal_permutation(p, sig, 5, 0.0, n=200, seed=s, method=m)["p_win"] <= 0.05
    sh, rp = rej["shift"] / K, rej["random_picks"] / K
    assert sh <= 0.12 and rp >= 0.10, (sh, rp)                                  # about 5 percent (a binomial of 60 at 0.05 reaches 0.12 about 2 percent of the time) against the old null's 13
    # (b) a factor with fixed random scores on securities whose true drifts differ (sd 25 percent a year): no information in it
    K2 = 60
    rej2 = {"labels": 0, "daily": 0}
    for s in range(K2):
        p, rng = _iid_panel(100 + s, N=40, drift_sd=0.25)
        f = pd.DataFrame(np.tile(rng.standard_normal(40), (len(p.dates), 1)), index=p.dates, columns=p.tickers).where(p.eligible)
        for m in rej2:
            rej2[m] += _q(lambda: rb.factor_permutation(p, f, n=29, seed=s, method=m, long_q=0.3, short_q=0.3, hold=5))["p"] <= 0.05
    la, da = rej2["labels"] / K2, rej2["daily"] / K2
    p0, rng0 = _iid_panel(100, N=40, drift_sd=0.25)                                                           # the default is the relabelled null, not the daily shuffle
    f0 = pd.DataFrame(np.tile(rng0.standard_normal(40), (len(p0.dates), 1)), index=p0.dates, columns=p0.tickers).where(p0.eligible)
    assert _q(lambda: rb.factor_permutation(p0, f0, n=29, seed=0, long_q=0.3, short_q=0.3, hold=5))["method"] == "labels"
    assert rb.signal_permutation(p0, pd.DataFrame(rng0.random(p0.close.shape) < 0.03, index=p0.dates, columns=p0.tickers) & p0.eligible, 5, n=20)["method"] == "shift"
    assert la <= 0.13 and da >= 0.15, (la, da)                                  # the relabelled null is near its nominal 5 percent (60 panels: 0.13 is 2 in 100), the daily shuffle is far above it
    print(f"RB11 information-free persistent signals called significant at the nominal 5 percent: time-shifted events {sh:.0%} (random picks {rp:.0%}), relabelled factors {la:.0%} (daily shuffle {da:.0%})  PASS")


if __name__ == "__main__":
    test_market_relative()
    test_mean_tests()
    test_subperiods()
    test_walk_forward()
    test_plateau()
    test_multiple_testing()
    test_permutations()
    test_decomposition()
    test_attribution()
    test_review_fixes()
    test_shift_null_exact_and_calibrated()
    test_calibration()
    print("robustness tests: all passed")
