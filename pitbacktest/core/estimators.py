"""Estimators: Fama-MacBeth, Newey-West, deciles and the shuffled null.

Overlapping returns
    Computing an h-day holding return every day makes h-1 days overlap. The independent observations are far fewer, so without a
    correction t is inflated several times over. Every t is computed with Newey-West.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

EPS = 1e-12


def newey_west_t(series: np.ndarray, lag: int) -> tuple[float, float, float, int]:
    """Newey-West corrected t for overlapping returns. Returns (mean, se, t, T)."""
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 30:
        return float("nan"), float("nan"), float("nan"), T
    mu = x.mean()
    xd = x - mu
    s = (xd @ xd) / T
    for lg in range(1, min(lag, T - 1) + 1):
        w = 1.0 - lg / (lag + 1.0)
        s += 2.0 * w * (xd[lg:] @ xd[:-lg]) / T
    if s <= 0:
        return float(mu), float("nan"), float("nan"), T
    se = np.sqrt(s / T)
    return float(mu), float(se), float(mu / se), T


def fama_macbeth(factor: pd.DataFrame, fwd: dict[int, pd.DataFrame],
                 controls: dict[str, pd.DataFrame], eligible: pd.DataFrame,
                 *, min_stocks: int | None = None) -> dict:
    """FM coefficient series of one factor x several horizons -> NW t.

    The factor is orthogonalised against the space of the controls by QR before the coefficient is taken.
    Both the t with controls and the t without (t_raw) are produced, but the caller is made to **report only the one after controls**.
    """
    fv = factor.values.astype(np.float64)
    ev = eligible.values
    cvs = [c.values.astype(np.float64) for c in controls.values()]
    hs = sorted(fwd)

    # A fixed minimum sample throws away every date in a small universe (for example 20 names).
    # The least a regression needs is (controls + intercept + factor), so twice that is the floor,
    # but never more than half the median universe size.
    if min_stocks is None:
        need = 2 * (len(cvs) + 2)
        med = float(np.median(ev.sum(axis=1)))
        min_stocks = int(max(need, min(50, med * 0.5)))
        if med < need:
            warnings.warn(
                f"The median universe has {med:.0f} names but there are {len(cvs)} controls, "
                f"so the regression has too few degrees of freedom (at least {need} names needed). "
                f"Use fewer controls or more names.", stacklevel=2)
    rv = {h: fwd[h].values.astype(np.float64) for h in hs}
    n = fv.shape[0]
    betas = {h: np.full(n, np.nan) for h in hs}
    raws = {h: np.full(n, np.nan) for h in hs}
    n_eligible_days = 0      # days with a large enough sample
    n_collinear = 0          # of those, days thrown away because the residual was 0 (perfectly collinear with the controls)

    for i in range(n):
        f = fv[i]
        ok = ev[i] & np.isfinite(f)
        for c in cvs:
            ok &= np.isfinite(c[i])
        if ok.sum() < min_stocks:
            continue
        fo = f[ok]
        Z = np.column_stack([np.ones(ok.sum())] + [c[i][ok] for c in cvs])
        try:
            Q, _ = np.linalg.qr(Z)
        except np.linalg.LinAlgError:
            continue
        fp = fo - Q @ (Q.T @ fo)
        den = fp @ fp
        foc = fo - fo.mean()
        den_r = foc @ foc
        n_eligible_days += 1
        if den <= EPS or den_r <= EPS:
            n_collinear += 1
            continue
        for h in hs:
            r = rv[h][i]
            okh = ok & np.isfinite(r)
            if okh.sum() < min_stocks:
                continue
            if okh.sum() != ok.sum():
                fo2 = f[okh]
                Z2 = np.column_stack([np.ones(okh.sum())] + [c[i][okh] for c in cvs])
                try:
                    Q2, _ = np.linalg.qr(Z2)
                except np.linalg.LinAlgError:
                    continue
                fp2 = fo2 - Q2 @ (Q2.T @ fo2)
                d2 = fp2 @ fp2
                if d2 <= EPS:
                    continue
                rr = r[okh]
                betas[h][i] = (fp2 @ (rr - Q2 @ (Q2.T @ rr))) / d2
                f2c = fo2 - fo2.mean()
                raws[h][i] = (f2c @ (rr - rr.mean())) / (f2c @ f2c)
            else:
                rr = r[ok]
                betas[h][i] = (fp @ (rr - Q @ (Q.T @ rr))) / den
                raws[h][i] = (foc @ (rr - rr.mean())) / den_r

    out = {}
    for h in hs:
        mu, se, t, T = newey_west_t(betas[h], lag=max(h, 21))
        _, _, t63, _ = newey_west_t(betas[h], lag=max(h, 63))
        mu_r, _, t_r, _ = newey_west_t(raws[h], lag=max(h, 21))
        out[h] = {"coef_bp": mu * 1e4 if np.isfinite(mu) else np.nan, "t": t,
                  "t_nw63": t63, "n_days": T,
                  "coef_bp_raw": mu_r * 1e4 if np.isfinite(mu_r) else np.nan, "t_raw": t_r}

    # Perfect-collinearity diagnosis: if the factor is effectively the same as the controls the residual is 0 and every date is thrown away.
    # A silent NaN would hide the cause, so say it out loud.
    if n_eligible_days and n_collinear / n_eligible_days > 0.5:
        warnings.warn(
            f"The factor is almost perfectly collinear with the controls "
            f"(residual about 0 on {n_collinear}/{n_eligible_days} days). "
            f"No information is left after the controls, so t is NaN. "
            f"Change the factor definition or drop that control.",
            stacklevel=2)
    for h in hs:
        out[h]["collinear_pct"] = (100.0 * n_collinear / n_eligible_days
                                   if n_eligible_days else np.nan)
    return out


def decile_profile(factor: pd.DataFrame, fwd_h: pd.DataFrame, eligible: pd.DataFrame,
                   *, lag: int = 21, n_bins: int = 10) -> dict:
    """Decile profile. Without monotonicity only the extremes differ, which is hard to call a signal."""
    rk = factor.where(eligible).rank(axis=1, pct=True, na_option="keep")
    c = fwd_h.values.astype(np.float64)
    ev = eligible.values
    prof = []
    for d in range(n_bins):
        lo, hi = d / n_bins, (d + 1) / n_bins + (0.01 if d == n_bins - 1 else 0)
        sel = ((rk >= lo) & (rk < hi) & eligible).values
        v = []
        for i in range(c.shape[0]):
            s = sel[i] & np.isfinite(c[i])
            pool = ev[i] & np.isfinite(c[i])
            if s.sum() >= 3 and pool.sum() >= 30:
                v.append(c[i][s].mean() - c[i][pool].mean())
        prof.append(float(np.mean(v) * 1e4) if v else np.nan)
    # Spearman correlation = Pearson correlation of the ranks. method="spearman" would need scipy, so it is not used
    # (the core dependencies of this package are pandas and numpy only).
    rho = pd.Series(prof).rank().corr(pd.Series(range(n_bins), dtype=float).rank())
    hi_lo = [x for x in (prof[-1], prof[0]) if np.isfinite(x)]
    return {"decile_bp": prof, "monotonicity_rho": float(rho) if pd.notna(rho) else np.nan,
            "spread_bp": float(prof[-1] - prof[0]) if len(hi_lo) == 2 else np.nan}


def shuffle_null(factor_builder, eligible: pd.DataFrame, fwd: dict[int, pd.DataFrame],
                 controls: dict[str, pd.DataFrame], *, n_rep: int = 3,
                 seed: int = 0) -> dict:
    """Shuffled null: **measures** the multiple-testing threshold.

    Only the security order is randomly permuted within each date. The date structure and the factor distribution are kept and
    only the security-factor link is destroyed, so the |t| distribution that comes out is the 'size of chance'.
    Theoretical corrections such as Bonferroni ignore the correlation between tests and are far too conservative.

    factor_builder: (rng) -> DataFrame  -- a callback that rebuilds the factor from the shuffled source
    """
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(n_rep):
        f = factor_builder(rng)
        r = fama_macbeth(f, fwd, controls, eligible)
        ts.extend(abs(v["t"]) for v in r.values() if np.isfinite(v["t"]))
    a = np.array(ts)
    if not len(a):
        return {"n": 0, "p95": np.nan, "max": np.nan}
    return {"n": int(len(a)), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def shuffle_columns(df: pd.DataFrame, eligible: pd.DataFrame, rng) -> pd.DataFrame:
    """Permute values only among the eligible securities of each date. The standard shuffler of shuffle_null."""
    a = df.values.copy()
    ev = eligible.values
    for i in range(a.shape[0]):
        j = np.where(ev[i])[0]
        if len(j) > 1:
            a[i, j] = a[i, rng.permutation(j)]
    return pd.DataFrame(a, index=df.index, columns=df.columns)


def paired_diff(base_mask: np.ndarray, alt_mask: np.ndarray, cum: np.ndarray,
                *, lag: int = 21, min_n: int = 5) -> dict:
    """Paired comparison: the pure contribution of adding a condition on the same dates.

    Adding a condition almost always improves the headline number. To see whether it is a real improvement it must be compared
    **on the same dates and the same sample**, in pairs.
    """
    d = []
    for i in range(cum.shape[0]):
        a = base_mask[i] & np.isfinite(cum[i])
        b = alt_mask[i] & np.isfinite(cum[i])
        if a.sum() >= min_n and b.sum() >= min_n:
            d.append(cum[i][b].mean() - cum[i][a].mean())
    if len(d) < 30:
        return {"diff_bp": np.nan, "t": np.nan, "n_days": len(d)}
    mu, _, t, n = newey_west_t(np.array(d), lag=lag)
    return {"diff_bp": float(mu * 1e4), "t": float(t), "n_days": n}
