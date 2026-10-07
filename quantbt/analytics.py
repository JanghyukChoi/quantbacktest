"""Reading a result: alpha and beta, information coefficients, and bootstrap intervals for the Sharpe ratio.

Three questions a reviewer asks first, each with the defaults that avoid the usual mistakes.

  alpha_beta                Is the return alpha, or exposure to something I could have bought? (OLS with Newey-West errors)
  information_coefficient   Does the signal rank next-period returns? (daily rank correlation; delisted names kept)
  sharpe_ci / sharpe_diff_ci  How precise is a Sharpe ratio, and is the difference between two of them real?
                            (stationary block bootstrap, so autocorrelation in returns is respected)

None of these proves a strategy works. They say how much the data can tell apart.

References
  Newey, West (1987, 1994): heteroskedasticity and autocorrelation consistent covariance, plug-in lag rule.
  Politis, Romano (1994): the stationary bootstrap.
  Lo (2002): the Sharpe ratio's sampling error is large and depends on autocorrelation.
"""
from __future__ import annotations

import math
import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()


# --------------------------------------------------------------------------------------------------- alpha and beta
def _align(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    df = pd.concat([y.rename("__y__"), X], axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    return df["__y__"], df.drop(columns="__y__")


def alpha_beta(returns: pd.Series, factors: pd.Series | pd.DataFrame, *, periods_per_year: int = 252,
               lag: int | None = None, min_obs: int = 60) -> dict:
    """Regress `returns` on `factors` (for example the market) with an intercept.

    Standard errors are Newey-West (Bartlett kernel, no small-sample correction), with the lag from the
    Newey-West (1994) plug-in rule `int(4 * (T / 100) ** (2 / 9))` unless `lag` is given. The t-statistics are
    asymptotic, compared with a normal distribution. **They are still too optimistic in finite samples**: with
    AR(1) errors of 0.5 and 500 observations, a true alpha of zero is rejected at |t| > 1.96 in about 9% of datasets
    instead of 5% (the uncorrected figure is 26%; `tests/test_analytics.py` A4). Read |t| below about 2.5 as no evidence. Dates are aligned on the intersection and rows with a missing
    value are dropped; at least `min_obs` rows (and 3 per coefficient) must remain.

    `alpha_annual` is the per-period intercept times `periods_per_year` (arithmetic, not compounded).
    `appraisal` is `alpha_annual` over the annualised residual volatility.
    Net returns include costs and funding while a benchmark does not, so a small negative alpha is expected for a
    strategy with no edge.
    """
    y = pd.Series(returns).astype(float)
    F = factors.to_frame(factors.name or "factor") if isinstance(factors, pd.Series) else factors.astype(float)
    if F.shape[1] == 0:
        raise ValueError("no factor columns")
    y, F = _align(y, F)
    T, k = len(y), F.shape[1] + 1
    if T < max(min_obs, 3 * k):
        raise ValueError(f"only {T} usable observations for {k} coefficients (need {max(min_obs, 3 * k)})")
    if (F.std(ddof=0) <= 1e-12 * np.maximum(1.0, F.abs().mean())).any():      # relative: identical floats can have std ~1e-18
        raise ValueError("a factor column is constant")
    X = np.column_stack([np.ones(T), F.to_numpy(float)])
    if np.linalg.matrix_rank(X) < k:
        raise ValueError("factor columns are collinear")
    Y = y.to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ coef
    L = int(4 * (T / 100) ** (2 / 9)) if lag is None else int(lag)
    L = max(0, min(L, T - 1))
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1.0 - l / (L + 1.0)) * (G + G.T)
    bread = np.linalg.inv(X.T @ X)
    V = bread @ S @ bread
    se = np.sqrt(np.diag(V))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = coef / se                                            # a perfect fit has zero error: t is inf or nan
    sse = float(e @ e)
    sst = float(((Y - Y.mean()) ** 2).sum())
    resid_vol = float(e.std(ddof=k) * math.sqrt(periods_per_year))
    ann = coef[0] * periods_per_year

    def p(tv: float) -> float:
        return float(2 * (1 - _N.cdf(abs(tv)))) if np.isfinite(tv) else float("nan")

    return {"alpha": float(coef[0]), "alpha_annual": float(ann), "alpha_t": float(t[0]), "alpha_p": p(t[0]),
            "betas": {c: {"beta": float(coef[i + 1]), "t": float(t[i + 1])} for i, c in enumerate(F.columns)},
            "r2": float(1 - sse / sst) if sst > 0 else float("nan"), "n": T, "lag": L,
            "resid_vol_annual": resid_vol, "appraisal": float(ann / resid_vol) if resid_vol > 0 else float("nan")}


# ----------------------------------------------------------------------------------------- information coefficient
def forward_returns(panel, h: int, *, delist_return: float | None = None) -> pd.DataFrame:
    """Compounded return from close(t + lag) to close(t + lag + h), for a signal at the close of t.

    A name with no price on a day earns 0 that day (it is treated as cash after its last bar), the same rule as
    `backtest_portfolio`; with `delist_return` set, the day after a delisting flagged in `panel.delist_after` earns
    that return instead. Dropping such names (a NaN forward return) would silently remove the losers from the test.
    Windows that run past the end of the sample are NaN."""
    ret = np.nan_to_num(panel.ret1().to_numpy(float), nan=0.0)
    if delist_return is not None and panel.delist_after is not None:
        da = panel.delist_after.reindex(index=panel.dates, columns=panel.tickers).fillna(False).to_numpy(bool)
        nxt = np.zeros(ret.shape, dtype=bool)
        nxt[1:] = da[:-1]
        ret = np.where(nxt, float(delist_return), ret)
    T = ret.shape[0]
    g = np.ones(ret.shape)
    for k in range(1, h + 1):
        s = panel.entry_lag + k
        sh = np.full(ret.shape, np.nan)
        if s < T:
            sh[: T - s] = 1.0 + ret[s:]
        g = g * sh
    return pd.DataFrame(g - 1.0, index=panel.dates, columns=panel.tickers)


def information_coefficient(factor: pd.DataFrame, forward: pd.DataFrame, eligible: pd.DataFrame, *,
                            min_obs: int = 20) -> pd.Series:
    """Daily cross-sectional Spearman correlation between the factor and the forward return, among eligible names
    with both values. Days with fewer than `min_obs` names, or no variation, are NaN."""
    f = factor.reindex(index=forward.index, columns=forward.columns)
    el = eligible.reindex(index=forward.index, columns=forward.columns).fillna(False).astype(bool)
    ok = el & f.notna() & forward.notna()
    rf = f.where(ok).rank(axis=1)
    ry = forward.where(ok).rank(axis=1)
    n = ok.sum(axis=1).to_numpy(float)
    a, b = rf.to_numpy(float), ry.to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN rows are expected (days with no signal)
        a = a - np.nanmean(a, axis=1, keepdims=True)
        b = b - np.nanmean(b, axis=1, keepdims=True)
        num = np.nansum(a * b, axis=1)
        den = np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
        ic = np.where((n >= min_obs) & (den > 0), num / den, np.nan)
    return pd.Series(ic, index=forward.index, name="ic")


def ic_report(panel, factor: pd.DataFrame, horizons=(1, 5, 20), *, delist_return: float | None = None,
              min_obs: int = 20) -> dict:
    """IC summary per horizon: mean IC, its standard deviation, ICIR (mean over standard deviation), the share of
    positive days, and a Newey-West t-statistic of the mean with lag = horizon (the forward windows overlap)."""
    from .core.estimators import newey_west_t
    out = {}
    for h in horizons:
        ic = information_coefficient(factor, forward_returns(panel, h, delist_return=delist_return),
                                     panel.eligible, min_obs=min_obs).dropna()
        if len(ic) < 30:
            out[h] = {"n_days": int(len(ic)), "mean_ic": float("nan"), "std_ic": float("nan"), "icir": float("nan"),
                      "hit_rate": float("nan"), "t_nw": float("nan")}
            continue
        mu, _, t, T = newey_west_t(ic.to_numpy(), lag=max(1, h))
        sd = float(ic.std(ddof=1))
        out[h] = {"n_days": int(T), "mean_ic": float(mu), "std_ic": sd, "icir": float(mu / sd) if sd > 0 else float("nan"),
                  "hit_rate": float((ic > 0).mean()), "t_nw": float(t)}
    return out


# --------------------------------------------------------------------------------------------- bootstrap intervals
def _stationary_indices(T: int, n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """(n, T) resampling indices for the stationary bootstrap: blocks of geometric length (mean `mean_block`),
    wrapping around the end of the sample."""
    p = 1.0 / max(1.0, mean_block)
    start = rng.integers(0, T, size=(n, T))
    new = rng.random((n, T)) < p
    idx = np.empty((n, T), dtype=np.int64)
    idx[:, 0] = start[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(new[:, t], start[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def _sharpe_rows(R: np.ndarray, ppy: int) -> np.ndarray:
    return R.mean(axis=1) / R.std(axis=1, ddof=1) * math.sqrt(ppy)


def _default_block(T: int) -> float:
    return float(max(1, round(T ** (1 / 3))))


def sharpe_ci(returns: pd.Series, *, periods_per_year: int = 252, n: int = 2000, mean_block: float | None = None,
              level: float = 0.95, seed: int = 0) -> dict:
    """Percentile interval for the annualised Sharpe ratio from a stationary block bootstrap.

    `mean_block` defaults to `round(T ** (1/3))`, a common rule of thumb; use a larger one if returns are strongly
    autocorrelated. Intervals from short samples tend to be too narrow, so treat a lower bound close to zero with
    caution. Sharpe uses risk-free 0 and the sample standard deviation (ddof=1)."""
    x = pd.Series(returns).dropna().to_numpy(float)
    T = len(x)
    if T < 30:
        raise ValueError(f"{T} observations is too few for a bootstrap")
    b = _default_block(T) if mean_block is None else float(mean_block)
    rng = np.random.default_rng(seed)
    sr = []
    for c in _chunks(n):
        sr.append(_sharpe_rows(x[_stationary_indices(T, c, b, rng)], periods_per_year))
    sr = np.concatenate(sr)
    a = (1 - level) / 2
    return {"sharpe": float(x.mean() / x.std(ddof=1) * math.sqrt(periods_per_year)),
            "lo": float(np.quantile(sr, a)), "hi": float(np.quantile(sr, 1 - a)), "se": float(sr.std(ddof=1)),
            "n_boot": int(n), "mean_block": b, "level": level, "T": T}


def sharpe_diff_ci(a: pd.Series, b: pd.Series, *, periods_per_year: int = 252, n: int = 2000,
                   mean_block: float | None = None, level: float = 0.95, seed: int = 0) -> dict:
    """Sharpe(b) - Sharpe(a) with a **paired** stationary bootstrap: the same resampled dates are used for both
    series, so the common movement of the two cancels, which is what makes a small difference detectable.
    `p_boot` is twice the smaller tail of the bootstrap distribution around zero (a two-sided percentile p-value,
    approximate). Series are aligned on their common dates."""
    df = pd.concat([pd.Series(a).rename("a"), pd.Series(b).rename("b")], axis=1, join="inner").dropna()
    T = len(df)
    if T < 30:
        raise ValueError(f"{T} common observations is too few for a bootstrap")
    xa, xb = df["a"].to_numpy(float), df["b"].to_numpy(float)
    bl = _default_block(T) if mean_block is None else float(mean_block)
    rng = np.random.default_rng(seed)
    d = []
    for c in _chunks(n):
        idx = _stationary_indices(T, c, bl, rng)
        d.append(_sharpe_rows(xb[idx], periods_per_year) - _sharpe_rows(xa[idx], periods_per_year))
    d = np.concatenate(d)
    al = (1 - level) / 2
    est = float((xb.mean() / xb.std(ddof=1) - xa.mean() / xa.std(ddof=1)) * math.sqrt(periods_per_year))
    p = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
    return {"diff": est, "lo": float(np.quantile(d, al)), "hi": float(np.quantile(d, 1 - al)), "se": float(d.std(ddof=1)),
            "p_boot": p, "n_boot": int(n), "mean_block": bl, "level": level, "T": T}


def _chunks(n: int, size: int = 400):
    left = n
    while left > 0:
        c = min(size, left)
        yield c
        left -= c
