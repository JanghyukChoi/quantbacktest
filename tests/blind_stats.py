# Reference implementation written from the specification alone by an assistant that had not seen the library (docs/blind_review/README.md); unchanged except for this line.
# tests/test_blind_reference.py compares it with the library.
"""Implementation of the statistics spec (area B). numpy/pandas only (math for erf)."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------------------
# normal helpers
# ----------------------------------------------------------------------------
def _ncdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _nppf(p):
    """Inverse normal cdf: Acklam's rational approximation + one Halley step."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= 1 - plow:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    for _ in range(2):  # Halley refinement
        e = _ncdf(x) - p
        u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
        x = x - u / (1 + x * u / 2)
    return x


def _two_sided_p(t):
    if not np.isfinite(t):
        return float("nan") if np.isnan(t) else 0.0
    return math.erfc(abs(t) / math.sqrt(2.0))


def _plugin_lag(T):
    return int(4 * (T / 100.0) ** (2.0 / 9.0))


def _clean_1d(x):
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def _skew_kurt(x):
    """Population (biased) skewness and Pearson kurtosis (normal = 3)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    m = x.mean()
    d = x - m
    m2 = (d ** 2).mean()
    if m2 == 0:
        return float("nan"), float("nan")
    return float((d ** 3).mean() / m2 ** 1.5), float((d ** 4).mean() / m2 ** 2)


# ----------------------------------------------------------------------------
# Newey-West
# ----------------------------------------------------------------------------
def newey_west_t(series, lag):
    """Newey-West corrected t for overlapping returns. Returns (mean, se, t, T)."""
    x = _clean_1d(series)
    T = len(x)
    if T == 0:
        return float("nan"), float("nan"), float("nan"), 0
    lag = int(lag)
    if lag < 0:
        raise ValueError("lag must be >= 0")
    mu = x.mean()
    u = x - mu
    s = float(np.dot(u, u)) / T
    for l in range(1, min(lag, T - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float(np.dot(u[l:], u[:-l])) / T
    se = math.sqrt(max(s, 0.0) / T)
    t = mu / se if se > 0 else (float("nan") if mu == 0 else math.copysign(math.inf, mu))
    return float(mu), float(se), float(t), int(T)


# ----------------------------------------------------------------------------
# Sharpe, deflated Sharpe, PBO
# ----------------------------------------------------------------------------
def _as_2d(R):
    a = np.asarray(R, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return a


def sharpe(R, periods_per_year=1):
    """Sharpe ratio of each column (risk-free 0), annualised by sqrt(periods_per_year)."""
    a = _as_2d(R)
    mu = a.mean(axis=0)
    sd = a.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mu / sd, np.nan)
    return out * math.sqrt(periods_per_year)


def deflated_sharpe(R, trials=None, periods_per_year=1):
    a = _as_2d(R)
    T, N = a.shape
    sr = sharpe(a, 1)  # per bar
    if np.all(np.isnan(sr)):
        raise ValueError("no column has a defined Sharpe ratio")
    best = int(np.nanargmax(sr))
    sr_best = float(sr[best])
    n_trials = int(trials) if trials is not None else N
    if N >= 2 and n_trials >= 2:
        sd_sr = float(np.nanstd(sr, ddof=1))
        z1 = _nppf(1.0 - 1.0 / n_trials)
        z2 = _nppf(1.0 - 1.0 / (n_trials * math.e))
        bench = sd_sr * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    else:
        bench = 0.0
    skew, kurt = _skew_kurt(a[:, best])
    denom_sq = 1.0 - skew * sr_best + (kurt - 1.0) / 4.0 * sr_best ** 2
    if denom_sq > 0 and T > 1:
        z = (sr_best - bench) * math.sqrt(T - 1) / math.sqrt(denom_sq)
        dsr = _ncdf(z)
    else:
        dsr = float("nan")
    k = math.sqrt(periods_per_year)
    return {
        "best": best,
        "sharpe": sr_best * k,
        "sharpe_luck_benchmark": bench * k,
        "trials": n_trials,
        "skew": skew,
        "kurtosis": kurt,
        "T": int(T),
        "dsr": float(dsr),
    }


def pbo_cscv(R, blocks=16, periods_per_year=1, max_splits=None, seed=0):
    a = _as_2d(R)
    T, N = a.shape
    S = int(blocks)
    if S < 2 or S % 2:
        raise ValueError("blocks must be an even number >= 2")
    if T < S * 2:
        raise ValueError("too few rows for this number of blocks")
    L = T // S
    a = a[T - L * S:]  # drop the remainder at the START (keep the most recent rows)
    blk = a.reshape(S, L, N)
    half = S // 2
    total = math.comb(S, half)
    if max_splits is not None and max_splits < total:
        rng = np.random.default_rng(seed)
        seen = set()
        combos = []
        while len(combos) < max_splits:
            c = tuple(sorted(rng.choice(S, half, replace=False).tolist()))
            if c not in seen:
                seen.add(c)
                combos.append(c)
    else:
        combos = itertools.combinations(range(S), half)
    k = math.sqrt(periods_per_year)
    n_le = 0
    n_done = 0
    is_best, oos_best = [], []
    allidx = np.arange(S)
    for c in combos:
        c = np.array(c)
        mask = np.zeros(S, bool)
        mask[c] = True
        ins = blk[mask].reshape(-1, N)
        oos = blk[~mask].reshape(-1, N)
        s_in = sharpe(ins, 1)
        s_out = sharpe(oos, 1)
        if np.all(np.isnan(s_in)):
            continue
        nstar = int(np.nanargmax(s_in))
        so = np.where(np.isnan(s_out), -np.inf, s_out)
        # rank among N, 1 = worst, ties share the average rank
        less = np.sum(so < so[nstar])
        eq = np.sum(so == so[nstar])
        rank = less + (eq + 1) / 2.0
        omega = rank / (N + 1.0)
        if omega <= 0.5:  # logit <= 0
            n_le += 1
        n_done += 1
        is_best.append(s_in[nstar] * k)
        oos_best.append(s_out[nstar] * k)
    return {
        "pbo": n_le / n_done if n_done else float("nan"),
        "splits": n_done,
        "is_best_sharpe_mean": float(np.nanmean(is_best)) if is_best else float("nan"),
        "oos_of_best_sharpe_mean": float(np.nanmean(oos_best)) if oos_best else float("nan"),
    }


# ----------------------------------------------------------------------------
# sign test, Wilson, multiple testing
# ----------------------------------------------------------------------------
def binom_two_sided(k, n):
    """Exact two-sided sign-test p against 0.5: min(1, 2*P(X <= min(k, n-k)))."""
    k = int(k)
    n = int(n)
    if n <= 0:
        return 1.0
    m = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * tail))


def wilson(k, n):
    n = int(n)
    if n <= 0:
        return float("nan"), float("nan")
    z = _nppf(0.975)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    ps = p[order]
    adj = ps * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def holm(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    ps = p[order]
    adj = ps * (m - np.arange(m))
    adj = np.maximum.accumulate(adj)
    adj = np.minimum(adj, 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


# ----------------------------------------------------------------------------
# mean tests
# ----------------------------------------------------------------------------
def mean_tests(net, periods_per_year=252, lag=None):
    x = _clean_1d(net)
    T = len(x)
    if lag is None:
        lag = _plugin_lag(T)
    mean, se, t, _ = newey_west_t(x, lag)
    sd = x.std(ddof=1) if T > 1 else float("nan")
    sh = mean / sd * math.sqrt(periods_per_year) if sd and sd > 0 else float("nan")
    sharpe_t = sh * math.sqrt(T / periods_per_year) if np.isfinite(sh) else float("nan")
    nz = x[x != 0]
    n_zero = int(T - len(nz))
    k = int((nz > 0).sum())
    m = len(nz)
    skew, kurt = _skew_kurt(x)
    return {
        "mean_daily": mean,
        "nw_t": t,
        "nw_p": _two_sided_p(t),
        "lag": int(lag),
        "sharpe_t": sharpe_t,
        "win_rate": k / m if m else float("nan"),
        "sign_p": binom_two_sided(k, m) if m else float("nan"),
        "n_zero": n_zero,
        "win_ci": wilson(k, m),
        "skew": skew,
        "kurtosis": kurt,
    }


# ----------------------------------------------------------------------------
# risk profile
# ----------------------------------------------------------------------------
def _cvar(x, var):
    below = x[x <= var]
    return float(below.mean()) if len(below) else float(var)


def _drawdowns(r):
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return eq, peak, eq / peak - 1.0


def _longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def risk_profile(net, periods_per_year=252, ewma_lambda=0.94):
    s = net if isinstance(net, pd.Series) else pd.Series(np.asarray(net, dtype=float))
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    r = s.to_numpy(dtype=float)
    n = len(r)
    out = {"n": n}
    if n == 0:
        return out
    eq, peak, dd = _drawdowns(r)
    out["total_return"] = float(eq[-1] - 1.0)
    q = lambda arr, p: float(np.quantile(arr, p))
    out["var_95"] = q(r, 0.05)
    out["var_99"] = q(r, 0.01)
    out["cvar_95"] = _cvar(r, out["var_95"])
    out["cvar_99"] = _cvar(r, out["var_99"])
    # filtered historical simulation
    lam = float(ewma_lambda)
    s2 = np.empty(n)
    s2[0] = r.var(ddof=1) if n > 1 else r[0] ** 2
    for t in range(1, n):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    s2_now = lam * s2[-1] + (1 - lam) * r[-1] ** 2
    vol_now = math.sqrt(s2_now)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = r / np.sqrt(s2)
    z = z[np.isfinite(z)]
    out["vol_now"] = vol_now
    if len(z):
        out["var_95_now"] = q(z, 0.05) * vol_now
        out["var_99_now"] = q(z, 0.01) * vol_now
    else:
        out["var_95_now"] = out["var_99_now"] = float("nan")
    sd = r.std(ddof=1) if n > 1 else float("nan")
    out["var_95_normal"] = float(r.mean() + _nppf(0.05) * sd)
    q5 = out["var_95"]
    out["tail_ratio"] = q(r, 0.95) / abs(q5) if q5 != 0 else float("nan")
    out["best_bar"] = float(r.max())
    out["worst_bar"] = float(r.min())
    if isinstance(s.index, pd.DatetimeIndex):
        m = (1.0 + s).groupby(s.index.to_period("M")).prod() - 1.0
        out["best_month"] = float(m.max())
        out["worst_month"] = float(m.min())
    out["skew"], out["kurtosis"] = _skew_kurt(r)
    under = eq < peak
    out["max_dd_bars"] = _longest_run(under)
    out["share_underwater"] = float(under.mean())
    ulcer = math.sqrt(float(np.mean(dd ** 2)))
    out["ulcer_index"] = ulcer
    cagr = eq[-1] ** (periods_per_year / n) - 1.0 if eq[-1] > 0 else -1.0
    out["ulcer_performance"] = cagr / ulcer if ulcer > 0 else float("nan")
    losses = -r[r < 0].sum()
    tot = r.sum()
    if losses > 0:
        out["gain_to_pain"] = float(tot / losses)
    else:
        out["gain_to_pain"] = float("inf") if tot > 0 else float("nan")
    return out


# ----------------------------------------------------------------------------
# market relative
# ----------------------------------------------------------------------------
def _align(a, b):
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        df = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    else:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        if len(aa) != len(bb):
            raise ValueError("series must have the same length when no index is given")
        df = pd.DataFrame({"a": aa, "b": bb})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _cagr(r, ppy):
    eq = float(np.prod(1.0 + r))
    return eq ** (ppy / len(r)) - 1.0 if eq > 0 else -1.0


def market_relative(net, bench, periods_per_year=252):
    df = _align(net, bench)
    s = df["a"].to_numpy()
    b = df["b"].to_numpy()
    n = len(df)
    ppy = periods_per_year
    out = {"n": n}
    scagr = _cagr(s, ppy)
    bcagr = _cagr(b, ppy)
    out["strategy_cagr"] = scagr
    out["benchmark_cagr"] = bcagr
    out["excess_cagr"] = scagr - bcagr
    if isinstance(df.index, pd.DatetimeIndex):
        for key, freq in (("hit_rate_month", "M"), ("hit_rate_year", "Y")):
            g = (1.0 + df).groupby(df.index.to_period(freq)).prod() - 1.0
            out[key] = float((g["a"] > g["b"]).mean())
    else:
        out["hit_rate_month"] = out["hit_rate_year"] = float("nan")
    up = b > 0
    dn = b < 0
    out["up_capture"] = float(s[up].mean() / b[up].mean()) if up.any() else float("nan")
    out["down_capture"] = float(s[dn].mean() / b[dn].mean()) if dn.any() else float("nan")
    vb = b.var(ddof=1)
    cov = np.cov(s, b, ddof=1)[0, 1]
    out["beta"] = float(cov / vb) if vb > 0 else float("nan")
    ss = s.std(ddof=1)
    sb = b.std(ddof=1)
    out["correlation"] = float(cov / (ss * sb)) if ss > 0 and sb > 0 else float("nan")
    d = s - b
    sd = d.std(ddof=1)
    out["tracking_error"] = float(sd * math.sqrt(ppy))
    out["information_ratio"] = float(d.mean() / sd * math.sqrt(ppy)) if sd > 0 else float("nan")
    ratio = np.concatenate([[1.0], np.cumprod(1.0 + s) / np.cumprod(1.0 + b)])
    peak = np.maximum.accumulate(ratio)
    out["max_relative_drawdown"] = float((ratio / peak - 1.0).min())
    return out


# ----------------------------------------------------------------------------
# alpha / beta with Newey-West standard errors
# ----------------------------------------------------------------------------
def alpha_beta(returns, factors, periods_per_year=252, lag=None, min_obs=60):
    if isinstance(factors, pd.Series):
        fdf = factors.to_frame(name=factors.name if factors.name is not None else "factor")
    elif isinstance(factors, pd.DataFrame):
        fdf = factors
    else:
        f = np.asarray(factors, dtype=float)
        if f.ndim == 1:
            f = f[:, None]
        fdf = pd.DataFrame(f, columns=[f"factor{i}" if f.shape[1] > 1 else "factor"
                                       for i in range(f.shape[1])])
    if isinstance(returns, pd.Series):
        y = returns.rename("__y__")
    else:
        y = pd.Series(np.asarray(returns, dtype=float), name="__y__")
    if isinstance(returns, pd.Series):
        df = pd.concat([y, fdf], axis=1, join="inner")
    else:
        if len(y) != len(fdf):
            raise ValueError("returns and factors must have the same length")
        df = pd.concat([y.reset_index(drop=True), fdf.reset_index(drop=True)], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    names = list(fdf.columns)
    k = len(names) + 1
    T = len(df)
    if T < max(min_obs, 3 * k):
        raise ValueError(f"too few observations: {T}")
    yv = df["__y__"].to_numpy(dtype=float)
    F = df[names].to_numpy(dtype=float)
    for j, nm in enumerate(names):
        if np.ptp(F[:, j]) == 0:
            raise ValueError(f"regressor {nm!r} is constant")
    X = np.column_stack([np.ones(T), F])
    if np.linalg.matrix_rank(X) < k:
        raise ValueError("regressors are collinear")
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ yv)
    u = yv - X @ beta
    if lag is None:
        lag = _plugin_lag(T)
    lag = int(lag)
    Z = X * u[:, None]
    S = Z.T @ Z
    for l in range(1, min(lag, T - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        G = Z[l:].T @ Z[:-l]
        S = S + w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    sst = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float((u ** 2).sum()) / sst if sst > 0 else float("nan")
    alpha = float(beta[0])
    alpha_annual = alpha * periods_per_year
    resid_vol = float(u.std(ddof=1) * math.sqrt(periods_per_year))
    return {
        "alpha": alpha,
        "alpha_annual": alpha_annual,
        "alpha_t": float(tvals[0]),
        "alpha_p": _two_sided_p(float(tvals[0])),
        "betas": {nm: {"beta": float(beta[i + 1]), "t": float(tvals[i + 1])}
                  for i, nm in enumerate(names)},
        "r2": r2,
        "n": int(T),
        "lag": lag,
        "resid_vol_annual": resid_vol,
        "appraisal": alpha_annual / resid_vol if resid_vol > 0 else float("nan"),
    }
