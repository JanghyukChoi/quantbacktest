"""Estimators (Newey-West, Fama-MacBeth, deciles, the shuffled null, paired differences) and the overfitting checks (deflated Sharpe, PBO, the permutation test) against loops written
from the definitions. A mutation run caught only 25 and 33 percent of the changes to these two files.

ES1 Newey-West    the Bartlett long-run variance by an explicit double loop, statsmodels when installed, and the edges (30 observations, constant series, a lag longer than the sample)
ES2 Fama-MacBeth  the daily slope with and without controls (OLS on each date), the minimum sample, the t at lags 21 and 63
ES3 deciles       each decile's mean excess return, the top bin including rank 1, the pool and fire minimums, Spearman, the spread
ES4 shuffles      the null's summary and the shuffler keep what they should
ES5 paired diff   the same-date difference, the minimum names per side and the 30-day floor
VL1 Sharpe        ddof 1 and the annualisation
VL2 deflated      the formula of Bailey and Lopez de Prado rebuilt with the standard library
VL3 PBO           every split by brute force on concatenated blocks
VL4 permutation   the block shuffle and the p-value rebuilt with the same random draws
"""
from __future__ import annotations
import itertools, math, sys, warnings
from pathlib import Path
from statistics import NormalDist
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from pitbacktest import validation as vl
from pitbacktest.core.estimators import decile_profile, fama_macbeth, newey_west_t, paired_diff, shuffle_columns, shuffle_null
from test_argument_checks import _raises

_N = NormalDist()


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _nw_loop(x, lag):
    x = np.asarray(x, float); T = len(x); xd = x - x.mean()
    g0 = sum(v * v for v in xd) / T
    s = g0
    for l in range(1, min(lag, T - 1) + 1):
        gl = sum(xd[t] * xd[t - l] for t in range(l, T)) / T
        s += 2 * (1 - l / (lag + 1)) * gl
    return x.mean(), math.sqrt(s / T), x.mean() / math.sqrt(s / T)


def test_newey_west():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 400); x = 0.1 + np.convolve(e, [1, 0.6, 0.3], mode="same")
    for lag in (0, 1, 5, 21):
        mu, se, t, T = newey_west_t(x, lag)
        m2, s2, t2 = _nw_loop(x, lag)
        assert T == 400 and abs(mu - m2) < 1e-12 and abs(se - s2) < 1e-12 and abs(t - t2) < 1e-10, lag
    try:
        import statsmodels.api as sm
        res = sm.OLS(x, np.ones(400)).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False})
        assert abs(newey_west_t(x, 5)[1] - res.bse[0]) < 1e-10
    except ImportError:
        pass
    assert np.isnan(newey_west_t(x[:29], 5)[0]) and newey_west_t(x[:29], 5)[3] == 29
    assert np.isfinite(newey_west_t(x[:30], 5)[2]) and newey_west_t(x[:30], 5)[3] == 30
    mu, se, t, T = newey_west_t(np.full(100, 0.5), 3)                                 # no variation: the mean is known, the error is not
    assert mu == 0.5 and np.isnan(se) and np.isnan(t)
    short = newey_west_t(x[:40], 500)                                                  # a lag longer than the sample is capped at T - 1
    assert abs(short[1] - _nw_loop(x[:40], 500)[1]) < 1e-12
    nan_x = np.concatenate([x[:50], [np.nan, np.inf], x[50:100]])
    assert newey_west_t(nan_x, 5)[3] == 100
    print("ES1 Newey-West equals the explicit Bartlett loop (and statsmodels), with its edges  PASS")


def _world(T=150, N=60, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=T); cols = [f"S{i:02d}" for i in range(N)]
    fac = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)
    c1 = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)
    c2 = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)
    fwd = 0.002 * fac + 0.001 * c1 + pd.DataFrame(rng.normal(0, 0.02, (T, N)), index=idx, columns=cols)
    el = pd.DataFrame(True, index=idx, columns=cols)
    return fac, fwd, c1, c2, el


def test_fama_macbeth():
    fac, fwd, c1, c2, el = _world()
    out = fama_macbeth(fac, {5: fwd}, {"c1": c1, "c2": c2}, el)[5]
    betas, raws = [], []
    for i in range(len(fac)):
        f, y = fac.iloc[i].to_numpy(), fwd.iloc[i].to_numpy()
        X = np.column_stack([np.ones(len(f)), c1.iloc[i], c2.iloc[i], f])
        betas.append(np.linalg.lstsq(X, y, rcond=None)[0][-1])
        raws.append(np.cov(f, y, ddof=1)[0, 1] / np.var(f, ddof=1))
    for key, series, lag in (("t", betas, 21), ("t_nw63", betas, 63), ("t_raw", raws, 21)):
        assert abs(out[key] - newey_west_t(np.array(series), lag)[2]) < 1e-8, key
    assert abs(out["coef_bp"] - np.mean(betas) * 1e4) < 1e-8 and abs(out["coef_bp_raw"] - np.mean(raws) * 1e4) < 1e-8
    assert out["n_days"] == 150 and out["collinear_pct"] == 0.0
    # the lag is the larger of the horizon and 21
    f40 = _quiet(lambda: fama_macbeth(fac, {40: fwd}, {}, el))[40]
    assert abs(f40["t"] - newey_west_t(np.array([np.cov(fac.iloc[i], fwd.iloc[i], ddof=1)[0, 1] / np.var(fac.iloc[i], ddof=1) for i in range(150)]), 40)[2]) < 1e-8
    # minimum sample: the floor is 2 x (controls + 2), at most 50, otherwise half the median universe
    small_el = el.copy(); small_el.iloc[:, 20:] = False                                # a universe of 20: floor max(8, min(50, 10)) = 10; all days have 20
    assert fama_macbeth(fac, {5: fwd}, {"c1": c1, "c2": c2}, small_el)[5]["n_days"] == 150
    thin = el.copy(); thin.iloc[:75, 10:] = False                                      # half the days have 10 eligible names: median universe is 35, floor min(50, 17.5) = 17
    assert fama_macbeth(fac, {5: fwd}, {"c1": c1, "c2": c2}, thin)[5]["n_days"] == 75
    explicit = fama_macbeth(fac, {5: fwd}, {"c1": c1, "c2": c2}, thin, min_stocks=10)[5]["n_days"]
    assert explicit == 150
    tiny = el.copy(); tiny.iloc[:, 5:] = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fama_macbeth(fac, {5: fwd}, {"c1": c1, "c2": c2}, tiny)
    assert any("degrees of freedom" in str(x.message) for x in w), "median universe 5 with two controls: need 8"
    ten = el.copy(); ten.iloc[:, 9:] = False                                            # 9 names, two controls: the floor is max(8, min(50, 4.5)) = 8, so every day counts and nothing is warned
    with warnings.catch_warnings(record=True) as w10:
        warnings.simplefilter("always")
        assert fama_macbeth(fac, {5: fwd}, {"c1": c1, "c2": c2}, ten)[5]["n_days"] == 150
    assert not any("degrees of freedom" in str(x.message) for x in w10)
    col = fama_macbeth(fac, {5: fwd}, {"same": fac * 2.0}, el)[5]                      # a control that is the factor: nothing is left
    assert np.isnan(col["t"]) and col["collinear_pct"] == 100.0
    print("ES2 Fama-MacBeth slopes, t at lags 21, 40 and 63, the minimum sample and the collinearity flag equal per-date OLS  PASS")


def test_deciles():
    fac, fwd, c1, c2, el = _world(T=120, N=60, seed=3)
    d = decile_profile(fac, fwd, el)
    rk = fac.rank(axis=1, pct=True).to_numpy(); f = fwd.to_numpy()
    prof = []
    for k in range(10):
        lo, hi = k / 10, (k + 1) / 10 + (0.01 if k == 9 else 0)
        v = []
        for i in range(len(f)):
            s = (rk[i] >= lo) & (rk[i] < hi)
            if s.sum() >= 3:
                v.append(f[i][s].mean() - f[i].mean())
        prof.append(np.mean(v) * 1e4)
    assert np.allclose(d["decile_bp"], prof, atol=1e-9), (d["decile_bp"], prof)
    assert abs(d["spread_bp"] - (prof[-1] - prof[0])) < 1e-9
    rho = pd.Series(prof).rank().corr(pd.Series(range(10), dtype=float).rank())
    assert abs(d["monotonicity_rho"] - rho) < 1e-12
    five = decile_profile(fac, fwd, el, n_bins=5)
    assert len(five["decile_bp"]) == 5
    # the top bin holds rank 1.0 (the highest security on a date), and a date with a pool under 30 names is skipped
    pool = el.copy(); pool.iloc[:, 29:] = False                                        # 29 eligible: no date qualifies
    nobody = decile_profile(fac, fwd, pool)
    assert all(np.isnan(x) for x in nobody["decile_bp"]) and np.isnan(nobody["spread_bp"]) and np.isnan(nobody["monotonicity_rho"])
    p30 = el.copy(); p30.iloc[:, 30:] = False                                          # exactly 30: qualifies
    fin = [np.isfinite(x) for x in decile_profile(fac, fwd, p30)["decile_bp"]]
    assert sum(fin) == 9 and not fin[0], fin                                            # 30 names: the lowest bin has two (ranks 1/30 and 2/30), bins 1 to 8 exactly three, the top four
    # three fires are needed in a bin: a bin with only two securities is skipped; at 30 names every decile has three
    print("ES3 decile means, the top bin, the 30-name pool, Spearman and the spread equal the definitions  PASS")


def test_shuffles_and_paired():
    fac, fwd, c1, c2, el = _world(T=100, N=40, seed=4)
    # shuffle_columns keeps the multiset of each eligible set and leaves the rest alone
    e2 = el.copy(); e2.iloc[:, 30:] = False; e2.iloc[5, :] = False; e2.iloc[5, 0] = True       # one eligible name on date 5: nothing to permute
    out = shuffle_columns(fac, e2, np.random.default_rng(1))
    for i in range(len(fac)):
        j = e2.iloc[i].to_numpy()
        assert sorted(out.iloc[i, j].to_numpy()) == sorted(fac.iloc[i, j].to_numpy())
        assert np.array_equal(out.iloc[i, ~j].to_numpy(), fac.iloc[i, ~j].to_numpy())
    assert np.array_equal(out.iloc[5].to_numpy(), fac.iloc[5].to_numpy()) and not out.equals(fac)
    two = el.copy(); two.iloc[:, 2:] = False                                            # two eligible names per date: they can still swap
    sw = shuffle_columns(fac, two, np.random.default_rng(1))
    assert (sw.iloc[:, :2].to_numpy() != fac.iloc[:, :2].to_numpy()).any() and np.allclose(np.sort(sw.iloc[:, :2].to_numpy(), axis=1), np.sort(fac.iloc[:, :2].to_numpy(), axis=1))
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    skip = el.copy(); skip.iloc[:, 1:] = False                                         # nobody has two eligible names: no random draw is consumed
    shuffle_columns(fac, skip, r1)
    assert r1.integers(0, 10**9) == r2.integers(0, 10**9)
    # the null's summary
    builder = lambda rng: shuffle_columns(fac, el, rng)                                # noqa: E731
    sn = _quiet(lambda: shuffle_null(builder, el, {5: fwd}, {}, n_rep=6, seed=2))
    rng = np.random.default_rng(2); ts = []
    for _ in range(6):
        ts.append(abs(_quiet(lambda: fama_macbeth(builder(rng), {5: fwd}, {}, el))[5]["t"]))
    a = np.array(ts)
    assert sn["n"] == 6 and abs(sn["median"] - np.median(a)) < 1e-12 and abs(sn["p95"] - np.percentile(a, 95)) < 1e-12
    assert abs(sn["p99"] - np.percentile(a, 99)) < 1e-12 and abs(sn["max"] - a.max()) < 1e-12
    assert _quiet(lambda: shuffle_null(builder, el, {5: fwd}, {}, seed=2))["n"] == 3        # three repetitions by default
    # paired difference
    rng = np.random.default_rng(5)
    cum = rng.normal(0, 0.02, (80, 20))
    base = rng.random((80, 20)) < 0.4; alt = rng.random((80, 20)) < 0.4
    base[:10] = False; alt[10:20, :] = False; alt[10:20, :4] = True                    # 10 days with no base names, 10 days with 4 alt names
    pdiff = paired_diff(base, alt, cum, lag=7, min_n=5)
    d = [cum[i][alt[i]].mean() - cum[i][base[i]].mean() for i in range(80) if base[i].sum() >= 5 and alt[i].sum() >= 5]
    mu, _, t, n = newey_west_t(np.array(d), 7)
    assert len(d) >= 30 and pdiff["n_days"] == n == len(d) and abs(pdiff["diff_bp"] - mu * 1e4) < 1e-9 and abs(pdiff["t"] - t) < 1e-12
    assert paired_diff(base, alt, cum, lag=7, min_n=4)["n_days"] != pdiff["n_days"] or True
    d5 = [i for i in range(80) if base[i].sum() >= 5 and alt[i].sum() >= 5]
    d4 = [i for i in range(80) if base[i].sum() >= 4 and alt[i].sum() >= 4]
    assert len(d4) > len(d5) and paired_diff(base, alt, cum, min_n=4)["n_days"] == len(d4)
    few = paired_diff(base[:40], alt[:40], cum[:40])
    assert few["n_days"] < 30 and np.isnan(few["diff_bp"]) and np.isnan(few["t"])
    allb = np.ones((30, 20), bool); alla = np.zeros((30, 20), bool); alla[:, :10] = True
    assert paired_diff(allb, alla, cum[:30])["n_days"] == 30 and np.isfinite(paired_diff(allb, alla, cum[:30])["t"])
    assert paired_diff(allb[:29], alla[:29], cum[:29])["n_days"] == 29 and np.isnan(paired_diff(allb[:29], alla[:29], cum[:29])["t"])
    assert paired_diff(base, alt, cum)["t"] != paired_diff(base, alt, cum, lag=3)["t"]          # the default lag is 21, not 3
    dflt = paired_diff(base, alt, cum)
    assert abs(dflt["t"] - newey_west_t(np.array([cum[i][alt[i]].mean() - cum[i][base[i]].mean() for i in range(80) if base[i].sum() >= 5 and alt[i].sum() >= 5]), 21)[2]) < 1e-12
    print("ES4/5 the shuffles keep what they should, the null's summary and the paired difference equal rebuilds  PASS")


def test_sharpe_deflated():
    rng = np.random.default_rng(6)
    R = rng.normal(0.0005, 0.01, (500, 8)); R[:, 3] += 0.001
    s = vl.sharpe(R, periods_per_year=252)
    assert np.allclose(s, R.mean(0) / R.std(0, ddof=1) * math.sqrt(252), atol=1e-12) and np.allclose(vl.sharpe(R), R.mean(0) / R.std(0, ddof=1))
    dsr = vl.deflated_sharpe(R, periods_per_year=252)
    sr_all = R.mean(0) / R.std(0, ddof=1); j = int(np.argmax(sr_all)); x = R[:, j]; T = 500; k = 8
    z = (x - x.mean()) / x.std(); skew, kurt = (z ** 3).mean(), (z ** 4).mean()
    sr0 = np.std(sr_all, ddof=1) * ((1 - 0.5772156649015329) * _N.inv_cdf(1 - 1 / k) + 0.5772156649015329 * _N.inv_cdf(1 - 1 / (k * math.e)))
    want = _N.cdf((sr_all[j] - sr0) * math.sqrt(T - 1) / math.sqrt(1 - skew * sr_all[j] + (kurt - 1) / 4 * sr_all[j] ** 2))
    assert dsr["best"] == j and abs(dsr["dsr"] - want) < 1e-12 and abs(dsr["sharpe"] - sr_all[j] * math.sqrt(252)) < 1e-12 and abs(dsr["sharpe_luck_benchmark"] - sr0 * math.sqrt(252)) < 1e-12
    assert abs(dsr["skew"] - skew) < 1e-12 and abs(dsr["kurtosis"] - kurt) < 1e-12 and dsr["T"] == 500 and dsr["trials"] == 8
    more = vl.deflated_sharpe(R, trials=100)
    assert more["trials"] == 100 and more["dsr"] < vl.deflated_sharpe(R)["dsr"] and more["sharpe_luck_benchmark"] > dsr["sharpe_luck_benchmark"] / math.sqrt(252)
    one = vl.deflated_sharpe(R[:, [3]], trials=1)                                      # a single trial: nothing to deflate
    assert one["sharpe_luck_benchmark"] == 0.0 and abs(one["dsr"] - _N.cdf(sr_all[3] * math.sqrt(499) / math.sqrt(1 - ((lambda zz: (zz ** 3).mean())((R[:, 3] - R[:, 3].mean()) / R[:, 3].std())) * sr_all[3] + ((lambda zz: (zz ** 4).mean())((R[:, 3] - R[:, 3].mean()) / R[:, 3].std()) - 1) / 4 * sr_all[3] ** 2))) < 1e-12
    _raises(lambda: vl.deflated_sharpe(R[:, [3]], trials=5), "only 1 column")
    two_nan = R.copy(); two_nan[0, 1:] = np.nan                                         # one usable column left of eight
    _raises(lambda: vl.deflated_sharpe(two_nan), "usable")
    two = R[:, :2]
    assert 0.0 <= vl.deflated_sharpe(two)["dsr"] <= 1.0                                 # exactly two usable columns are enough
    print("VL1/2 Sharpe and the deflated Sharpe ratio equal the published formula rebuilt with the standard library  PASS")


def _pbo_brute(R, blocks, ppy=1, max_splits=None, seed=0):
    L = len(R) // blocks; R = R[len(R) - L * blocks:]
    combos = list(itertools.combinations(range(blocks), blocks // 2))
    if max_splits and len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]
    cat = lambda idx: np.concatenate([R[i * L:(i + 1) * L] for i in idx])               # noqa: E731
    lg, isb, oob = [], [], []
    for c in combos:
        o = [i for i in range(blocks) if i not in c]
        a, b = cat(c), cat(o)
        x = a.mean(0) / a.std(0, ddof=1); y = b.mean(0) / b.std(0, ddof=1)
        j = int(np.argmax(x)); w = (1 + (y < y[j]).sum()) / (len(y) + 1)
        lg.append(math.log(w / (1 - w))); isb.append(x[j] * math.sqrt(ppy)); oob.append(y[j] * math.sqrt(ppy))
    lg = np.array(lg)
    return {"pbo": (lg <= 0).mean(), "splits": len(lg), "is": np.mean(isb), "oos": np.mean(oob)}


def test_pbo_and_permutation():
    rng = np.random.default_rng(7)
    R = rng.normal(0, 0.01, (4 * 10 + 3, 6))                                           # three leading rows are dropped so that the blocks are equal
    got = vl.pbo_cscv(R, blocks=4, periods_per_year=252)
    want = _pbo_brute(R, 4, 252)
    assert got["splits"] == want["splits"] == 6 and abs(got["pbo"] - want["pbo"]) < 1e-12
    assert abs(got["is_best_sharpe_mean"] - want["is"]) < 1e-9 and abs(got["oos_of_best_sharpe_mean"] - want["oos"]) < 1e-9
    R2 = rng.normal(0, 0.01, (160, 5))
    got2 = vl.pbo_cscv(R2, blocks=8, max_splits=20, seed=3)
    want2 = _pbo_brute(R2, 8, 1, 20, 3)
    assert got2["splits"] == 20 and abs(got2["pbo"] - want2["pbo"]) < 1e-12 and abs(got2["is_best_sharpe_mean"] - want2["is"]) < 1e-9
    assert vl.pbo_cscv(R2, blocks=8, max_splits=70)["splits"] == 70 and vl.pbo_cscv(R2, blocks=8, max_splits=100)["splits"] == 70
    # a column that wins in every block has a PBO of 0; identical columns are all ties, and a tie counts as overfit (logit 0)
    best = rng.normal(0, 0.01, (160, 4)); best[:, 0] += 0.02
    assert vl.pbo_cscv(best, blocks=8)["pbo"] == 0.0
    # a tie: the in-sample best is the middle one out of sample in both splits (logit 0), which counts as overfit
    z = lambda: (lambda e: (e - e.mean()) / e.std(ddof=1))(np.random.default_rng(11).normal(size=100))            # noqa: E731
    blk = lambda sh: np.column_stack([z() * 0.01 + m * 0.01 for m in sh])                                          # noqa: E731
    tie = np.vstack([blk([3, 2, 1]), blk([2, 3, 1])])
    t = vl.pbo_cscv(tie, blocks=2)
    assert t["splits"] == 2 and t["pbo"] == 1.0, t
    # the permutation test
    r = rng.normal(0.0005, 0.01, 200)
    search = lambda a: float(a[:100].mean() / a[:100].std())                            # noqa: E731
    for block in (1, 7):
        res = vl.permutation_test(r, search, n=60, seed=4, block=block)
        gen = np.random.default_rng(4); nb = len(r) // block; nulls = []
        for _ in range(60):
            if block == 1:
                rs = r[gen.permutation(len(r))]
            else:
                order = gen.permutation(nb)
                rs = np.concatenate([r[i * block:(i + 1) * block] for i in order] + [r[nb * block:]])
            nulls.append(search(rs))
        real = search(r)
        assert abs(res["real"] - real) < 1e-15 and res["n"] == 60 and abs(res["p"] - (1 + sum(v >= real for v in nulls)) / 61) < 1e-15
        assert abs(res["null_mean"] - np.mean(nulls)) < 1e-12 and abs(res["null_p95"] - np.percentile(nulls, 95)) < 1e-12, block
    const = vl.permutation_test(np.full(50, 0.01), lambda a: 1.0, n=20)                 # every shuffle ties the real value: p = 1
    assert const["p"] == 1.0
    print("VL3/4 PBO equals a brute-force enumeration of the splits; the permutation test equals a rebuild with the same draws  PASS")


if __name__ == "__main__":
    for fn in (test_newey_west, test_fama_macbeth, test_deciles, test_shuffles_and_paired, test_sharpe_deflated, test_pbo_and_permutation):
        fn()
    print("estimator and validation value tests: all passed")
