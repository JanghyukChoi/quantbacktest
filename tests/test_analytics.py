"""Alpha and beta, information coefficients and bootstrap intervals: known answers, error rates and cross-checks.

Part 1 compares with independent implementations (statsmodels, scipy) when they are installed; they are not
dependencies of the library, so those tests are skipped without them. Parts 2 and 3 need only numpy and pandas.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import analytics as an
from test_reconcile import _panel

try:
    import statsmodels.api as sm
    from scipy.stats import spearmanr
    HAVE_REF = True
except ImportError:
    HAVE_REF = False


def _ar1(rng, T, phi, sd):
    e = np.zeros(T)
    eps = rng.normal(0, sd, T)
    for t in range(1, T):
        e[t] = phi * e[t - 1] + eps[t]
    return e


# ------------------------------------------------------------------------------- 1. against independent code
def test_alpha_beta_matches_statsmodels_hac():
    if not HAVE_REF:
        print("A1 alpha_beta vs statsmodels   SKIPPED (statsmodels/scipy not installed)")
        return
    worst_c = worst_t = 0.0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        T, k = int(rng.integers(120, 1200)), int(rng.integers(1, 4))
        idx = pd.bdate_range("2015-01-01", periods=T)
        F = pd.DataFrame(rng.normal(0, 0.01, (T, k)), index=idx, columns=[f"f{i}" for i in range(k)])
        y = pd.Series(0.0003 + F.to_numpy() @ rng.normal(0.5, 0.5, k) + _ar1(rng, T, 0.4, 0.01), index=idx)
        y.iloc[rng.integers(0, T, 3)] = np.nan
        r = an.alpha_beta(y, F, lag=[None, 0, 1, 5, 12][seed % 5])
        d = pd.concat([y.rename("y"), F], axis=1).dropna()
        m = sm.OLS(d["y"], sm.add_constant(d[F.columns])).fit(
            cov_type="HAC", cov_kwds={"maxlags": r["lag"], "use_correction": False, "kernel": "bartlett"})
        b = np.array([r["alpha"]] + [r["betas"][c]["beta"] for c in F.columns])
        t = np.array([r["alpha_t"]] + [r["betas"][c]["t"] for c in F.columns])
        worst_c = max(worst_c, float(np.max(np.abs(b - m.params.to_numpy()))))
        worst_t = max(worst_t, float(np.max(np.abs(t - m.tvalues.to_numpy()))))
    assert worst_c < 1e-12 and worst_t < 1e-9, (worst_c, worst_t)
    print(f"A1 alpha_beta vs statsmodels HAC, 40 designs   max |coef| {worst_c:.0e}  max |t| {worst_t:.0e}  PASS")


def test_ic_matches_scipy_spearman():
    if not HAVE_REF:
        print("A2 IC vs scipy                 SKIPPED (statsmodels/scipy not installed)")
        return
    rng = np.random.default_rng(7)
    T, N = 120, 70
    idx = pd.bdate_range("2020-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    f = pd.DataFrame(np.round(rng.normal(size=(T, N)), 1), index=idx, columns=cols)            # rounding makes ties
    y = pd.DataFrame(np.round(rng.normal(size=(T, N)) + 0.3 * f.to_numpy(), 2), index=idx, columns=cols)
    f.iloc[rng.integers(0, T, 250), rng.integers(0, N, 250)] = np.nan
    el = pd.DataFrame(rng.random((T, N)) > 0.2, index=idx, columns=cols)
    ic = an.information_coefficient(f, y, el, min_obs=20)
    worst = 0.0
    for i in range(T):
        ok = el.iloc[i].to_numpy() & f.iloc[i].notna().to_numpy() & y.iloc[i].notna().to_numpy()
        if ok.sum() < 20:
            assert np.isnan(ic.iloc[i])
            continue
        worst = max(worst, abs(ic.iloc[i] - spearmanr(f.iloc[i][ok], y.iloc[i][ok]).statistic))
    assert worst < 1e-12, worst
    print(f"A2 IC vs scipy spearmanr (ties, NaN, eligibility)   max |diff| {worst:.0e}  PASS")


# --------------------------------------------------------------------------------------- 2. alpha and beta
def test_alpha_beta_known_answer_and_invariances():
    rng = np.random.default_rng(1)
    T = 3000
    idx = pd.bdate_range("2005-01-01", periods=T)
    f = pd.Series(rng.normal(0, 0.01, T), index=idx, name="mkt")
    y = pd.Series(0.0004 + 1.3 * f.to_numpy() + rng.normal(0, 0.005, T), index=idx)
    r = an.alpha_beta(y, f)
    se_a = abs(r["alpha"]) / abs(r["alpha_t"])
    assert abs(r["alpha"] - 0.0004) < 3 * se_a and abs(r["betas"]["mkt"]["beta"] - 1.3) < 0.02, r
    assert r["r2"] > 0.8
    r1000 = an.alpha_beta(y * 1000, f)                                  # scaling y scales alpha, not the t-statistic
    assert abs(r1000["alpha"] - 1000 * r["alpha"]) < 1e-9 and abs(r1000["alpha_t"] - r["alpha_t"]) < 1e-9
    rs = an.alpha_beta(y, f + 0.02)                                     # a constant shift moves alpha by beta * c only
    assert abs(rs["betas"]["mkt"]["beta"] - r["betas"]["mkt"]["beta"]) < 1e-9
    assert abs(rs["alpha"] - (r["alpha"] - r["betas"]["mkt"]["beta"] * 0.02)) < 1e-9
    assert abs(r["alpha_annual"] - r["alpha"] * 252) < 1e-12
    print(f"A3 known answer (alpha 4bp/day, beta 1.3) and invariances   alpha {r['alpha']*1e4:.2f}bp t {r['alpha_t']:.1f}  "
          f"beta {r['betas']['mkt']['beta']:.3f}  PASS")


def test_alpha_beta_size_with_autocorrelated_errors():
    """Alpha is truly zero and the errors are AR(1). HAC t-statistics should reject near 5%; ordinary ones reject more."""
    rng = np.random.default_rng(2)
    T, reps = 500, 800
    idx = pd.bdate_range("2015-01-01", periods=T)
    rej_hac = rej_ols = 0
    for _ in range(reps):
        f = pd.Series(rng.normal(0, 0.01, T), index=idx, name="m")
        y = pd.Series(_ar1(rng, T, 0.5, 0.01), index=idx)
        r = an.alpha_beta(y, f)
        rej_hac += abs(r["alpha_t"]) > 1.96
        rej_ols += abs(an.alpha_beta(y, f, lag=0)["alpha_t"]) > 1.96
    h, o = rej_hac / reps, rej_ols / reps
    assert 0.03 < h < 0.12, f"HAC rejection rate {h:.3f}"      # known finite-sample over-rejection; measured 0.094
    assert o > h + 0.05, (o, h)
    print(f"A4 size at 5% with AR(1) 0.5 errors, T=500   HAC {h:.3f}   no correction {o:.3f}  PASS")


def test_alpha_beta_errors_and_alignment():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(3)
    f = pd.Series(rng.normal(0, 0.01, 200), index=idx, name="m")
    y = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    for bad, msg in ((pd.Series(0.01, index=idx, name="m"), "constant"),
                     (pd.DataFrame({"a": f, "b": 2 * f}), "collinear"),
                     (f.iloc[:30], "usable observations")):
        try:
            an.alpha_beta(y, bad)
        except ValueError as e:
            assert msg in str(e), (msg, str(e))
        else:
            raise AssertionError(f"no error for {msg}")
    y2 = y.copy(); y2.iloc[5] = np.inf; y2.iloc[9] = np.nan
    r = an.alpha_beta(y2, f.iloc[3:])                       # different date ranges, inf and NaN
    assert r["n"] == len(set(f.index[3:]) - {y.index[5], y.index[9]}), r["n"]
    print("A5 constant, collinear and short inputs raise; inf, NaN and mismatched dates are dropped  PASS")


def test_portfolio_equal_to_benchmark_has_beta_one():
    p, f = _panel(seed=8)
    r = q.backtest_portfolio(p, f, long_q=1.0, short_q=None, hold=1, spread_bp=0.0, benchmark="equal", grid=False)
    ab = r.alpha_beta()
    b = ab["betas"]["benchmark"]["beta"]
    assert abs(b - 1.0) < 1e-9 and abs(ab["alpha"]) < 1e-12 and ab["r2"] > 1 - 1e-9, ab
    short = q.backtest_portfolio(p, f, long_q=0.3, short_q=0.3, hold=3, spread_bp=0.0, benchmark="equal", grid=False)
    assert abs(short.alpha_beta()["betas"]["benchmark"]["beta"]) < 0.4              # dollar neutral: small market exposure
    try:
        q.backtest_portfolio(p, f, benchmark=None, grid=False).alpha_beta()
    except ValueError:
        pass
    else:
        raise AssertionError("alpha_beta without benchmark should explain itself")
    print(f"A6 PortfolioResult.alpha_beta: holding the whole benchmark gives beta {b:.6f}, alpha 0, R2 1  PASS")


# --------------------------------------------------------------------------------- 3. IC and forward returns
def _ref_forward(panel, h, delist_return):
    T, N = panel.close.shape
    c = panel.close.to_numpy(float)
    ret = np.zeros((T, N))
    for t in range(1, T):
        for j in range(N):
            a, b = c[t - 1, j], c[t, j]
            ret[t, j] = 0.0 if (math.isnan(a) or math.isnan(b)) else b / a - 1
    if delist_return is not None and panel.delist_after is not None:
        da = panel.delist_after.to_numpy(bool)
        for t in range(1, T):
            for j in range(N):
                if da[t - 1, j]:
                    ret[t, j] = delist_return
    out = np.full((T, N), np.nan)
    for t in range(T):
        if t + panel.entry_lag + h < T:
            g = np.ones(N)
            for k in range(1, h + 1):
                g = g * (1 + ret[t + panel.entry_lag + k])
            out[t] = g - 1
    return out


def test_forward_returns_match_loops_including_delisting():
    p, _ = _panel(seed=5, delist=True)
    worst = 0.0
    for h in (1, 5, 20):
        for dr in (None, -0.5):
            got = an.forward_returns(p, h, delist_return=dr).to_numpy()
            ref = _ref_forward(p, h, dr)
            assert np.array_equal(np.isnan(got), np.isnan(ref))
            worst = max(worst, float(np.nanmax(np.abs(got - ref))))
    assert worst < 1e-12, worst
    f_drop = an.forward_returns(p, 5, delist_return=None).to_numpy()
    f_keep = an.forward_returns(p, 5, delist_return=-0.5).to_numpy()
    assert np.nansum(f_keep < f_drop - 1e-12) > 0                  # the delisting loss reaches the windows that contain it
    print(f"A7 forward returns vs loops (h 1/5/20, with and without a delisting return)   max |diff| {worst:.0e}  PASS")


def test_ic_known_answers_and_size():
    p, _ = _panel(seed=6)
    fwd = an.forward_returns(p, 1)
    perfect = an.information_coefficient(fwd, fwd, p.eligible).dropna()
    assert (perfect > 1 - 1e-12).all()
    anti = an.information_coefficient(-fwd, fwd, p.eligible).dropna()
    assert (anti < -1 + 1e-12).all()
    rng = np.random.default_rng(4)
    rej = 0
    for k in range(60):
        noise = pd.DataFrame(rng.normal(size=p.close.shape), index=p.dates, columns=p.tickers)
        rep = q.analytics.ic_report(p, noise, horizons=(5,))[5]
        rej += abs(rep["t_nw"]) > 1.96
    rate = rej / 60
    assert rate < 0.2, rate
    skilled = pd.DataFrame(fwd.to_numpy() + rng.normal(0, 0.1, p.close.shape), index=p.dates, columns=p.tickers)
    rep = q.analytics.ic_report(p, skilled, horizons=(1,))[1]
    assert rep["mean_ic"] > 0.1 and rep["t_nw"] > 5 and rep["hit_rate"] > 0.9, rep
    print(f"A8 IC: perfect +1, reversed -1, noise rejected {rate:.2f} of the time at |t|>1.96, a real signal is found  PASS")


# -------------------------------------------------------------------------------------------- 4. bootstrap
def test_bootstrap_sharpe_matches_theory_and_covers():
    rng = np.random.default_rng(5)
    T, reps, ppy = 500, 200, 252
    mu, sd = 0.0004, 0.01
    true_sr = mu / sd * math.sqrt(ppy)
    cover, ses = 0, []
    for k in range(reps):
        x = pd.Series(rng.normal(mu, sd, T))
        ci = an.sharpe_ci(x, periods_per_year=ppy, n=400, seed=k)
        cover += ci["lo"] <= true_sr <= ci["hi"]
        ses.append(ci["se"])
    cov = cover / reps
    se_theory = math.sqrt((1 + (mu / sd) ** 2 / 2) / T) * math.sqrt(ppy)          # iid normal returns (Lo 2002)
    ratio = float(np.mean(ses)) / se_theory
    assert cov >= 0.90, f"coverage {cov:.3f}"
    assert 0.85 < ratio < 1.20, ratio
    print(f"B1 Sharpe bootstrap on iid returns: 95% interval covers the truth {cov:.3f}; bootstrap se / theory se = {ratio:.2f}  PASS")


def test_bootstrap_respects_autocorrelation():
    rng = np.random.default_rng(6)
    x = pd.Series(_ar1(rng, 3000, 0.3, 0.01) + 0.0003)
    s1 = an.sharpe_ci(x, n=600, mean_block=1, seed=1)["se"]
    s10 = an.sharpe_ci(x, n=600, mean_block=10, seed=1)["se"]
    assert 1.1 < s10 / s1 < 1.7, (s1, s10)             # theory for the mean: sqrt((1+rho)/(1-rho)) = 1.36
    print(f"B2 with autocorrelation 0.3 the block bootstrap widens the se by {s10 / s1:.2f}x versus iid resampling (theory ~1.36)  PASS")


def test_paired_difference():
    rng = np.random.default_rng(8)
    T = 1500
    idx = pd.bdate_range("2012-01-01", periods=T)
    a = pd.Series(rng.normal(0.0004, 0.01, T), index=idx)
    same = an.sharpe_diff_ci(a, a, n=300)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0 and same["p_boot"] == 1.0
    b = a + rng.normal(0, 0.001, T) + 0.0002                         # almost the same strategy plus a small edge
    d = an.sharpe_diff_ci(a, b, n=800, seed=2)
    se_a, se_b = an.sharpe_ci(a, n=800)["se"], an.sharpe_ci(b, n=800)["se"]
    assert d["se"] < 0.5 * math.sqrt(se_a ** 2 + se_b ** 2), (d["se"], se_a, se_b)    # pairing is what makes it detectable
    assert d["lo"] > 0 and d["p_boot"] < 0.05, d
    assert an.sharpe_diff_ci(a, b, n=300, seed=3) == an.sharpe_diff_ci(a, b, n=300, seed=3)
    try:
        an.sharpe_ci(a.iloc[:10])
    except ValueError:
        pass
    else:
        raise AssertionError("short series should raise")
    print(f"B3 paired difference: identical series give exactly 0; se {d['se']:.2f} vs {math.sqrt(se_a**2 + se_b**2):.2f} unpaired  PASS")


if __name__ == "__main__":
    test_alpha_beta_matches_statsmodels_hac()
    test_ic_matches_scipy_spearman()
    test_alpha_beta_known_answer_and_invariances()
    test_alpha_beta_size_with_autocorrelated_errors()
    test_alpha_beta_errors_and_alignment()
    test_portfolio_equal_to_benchmark_has_beta_one()
    test_forward_returns_match_loops_including_delisting()
    test_ic_known_answers_and_size()
    test_bootstrap_sharpe_matches_theory_and_covers()
    test_bootstrap_respects_autocorrelation()
    test_paired_difference()
    print("analytics tests: all passed")
