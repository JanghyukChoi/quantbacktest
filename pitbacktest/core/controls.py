"""Controls: 7 price-based and 5 firm characteristics.

Why firm characteristics are the default
    Testing with price controls alone makes you **mistake someone else's alpha for your own**. Without controlling for
    profitability and value, the apparent alpha often shrinks a lot or flips sign
    (T6 in tests/test_synthetic.py feeds a control in as the factor and reproduces this disappearance).
    Price controls (mom20/mom120) alone cannot capture standard momentum (12-1 months), profitability or value.
    So chars are **included automatically** when present, and a warning is raised when they are not.

Normalisation
    Everything is a per-date percentile rank -> mean 0, variance 1. Robust to outliers and lets a coefficient be read as 'bp per 1 SD'.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

EPS = 1e-12
PRICE_CONTROLS = ("rev1", "rev5", "mom21", "mom63", "mom252_21", "logsize", "turnover", "vol21")
CHAR_NAMES = ("log_size", "log_bm", "momentum", "roa", "asset_growth")


def xs_rank(v: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Per-date percentile rank [0,1] among the eligible. A cross-sectional operation, so the market-wide component drops out by itself."""
    return v.where(eligible).rank(axis=1, pct=True, na_option="keep")


def xs_norm(v: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Rank -> mean 0, variance 1 scale."""
    return ((xs_rank(v, eligible) - 0.5) * np.sqrt(12.0)).astype(np.float32)


def build_controls(panel, *, include_chars: bool = True) -> dict[str, pd.DataFrame]:
    """The bundle of controls. Firm characteristics are included when panel.chars exists."""
    close, el = panel.close, panel.eligible
    ret = close.pct_change()
    out: dict[str, pd.DataFrame] = {
        # rev1 is required: the order flow of mean-reversion traders is a proxy for 'the day's fall'.
        # The 1-day reversal is the strongest short-term effect, so leaving it out makes you rediscover reversal and call it a signal.
        "rev1": ret,
        "rev5": close.pct_change(5),
        "mom21": close.pct_change(21),
        "mom63": close.pct_change(63),
        # standard momentum 12-1: mom21/63 alone cannot capture it
        "mom252_21": close.pct_change(getattr(panel, "periods_per_year", 252)) - close.pct_change(21),
        "vol21": ret.rolling(21, min_periods=21).std(),
    }
    if panel.mkt_cap is not None:
        out["logsize"] = np.log(panel.mkt_cap.replace(0, np.nan))
        adv = panel.adv(20)
        if adv is not None:
            out["turnover"] = adv / (panel.mkt_cap + EPS)

    if include_chars:
        if not panel.chars:
            warnings.warn(
                "Firm characteristics (chars) are missing. With price controls alone the ROA and book-to-market exposures are not captured, so "
                "'someone else's alpha' can be mistaken for the factor's own. "
                "Fill chars through an adapter, or state include_chars=False explicitly.",
                stacklevel=2,
            )
        for k, v in panel.chars.items():
            out[f"char_{k}"] = v

    return {k: xs_norm(v, el).astype(np.float32) for k, v in out.items() if v is not None}


def neutralize(factor: pd.DataFrame, controls: dict[str, pd.DataFrame],
               eligible: pd.DataFrame, *, min_obs: int = 50) -> pd.DataFrame:
    """Per-date cross-sectional regression residual: orthogonalised against the controls.

    For continuous factors. Event (0/1) signals use the dummy regression in event.py.
    A day whose residual is below 1e-5 of the factor's size (the controls explain the factor almost completely) stays NaN.
    """
    fv = factor.values.astype(np.float64)
    ev = eligible.values
    cvs = [c.values.astype(np.float64) for c in controls.values()]
    out = np.full(fv.shape, np.nan)
    for i in range(fv.shape[0]):
        ok = ev[i] & np.isfinite(fv[i])
        for c in cvs:
            ok &= np.isfinite(c[i])
        if ok.sum() < min_obs:
            continue
        y = fv[i][ok]
        X = np.column_stack([np.ones(ok.sum())] + [c[i][ok] for c in cvs])
        try:
            Q, _ = np.linalg.qr(X)
        except np.linalg.LinAlgError:
            continue
        res = y - Q @ (Q.T @ y)
        # A factor inside the span of the controls leaves only rounding noise (about 1e-8 of its size in float32).
        # Left in, a later rank-normalisation would blow that noise up to unit scale and the result would depend on the
        # BLAS build. No information is left, so the day stays NaN.
        if not (res.std() > 1e-5 * y.std()):
            continue
        out[i][ok] = res
    return pd.DataFrame(out, index=factor.index, columns=factor.columns)


def build_chars_from_financials(close: pd.DataFrame, mkt_cap: pd.DataFrame,
                                book_equity: pd.DataFrame | None = None,
                                net_income: pd.DataFrame | None = None,
                                total_assets: pd.DataFrame | None = None) -> dict:
    """Five firm characteristics: standard Fama-French style definitions.

    Quarterly financials are passed already forward-filled to daily (the adapter's job).
    """
    chars = {"log_size": np.log(mkt_cap.clip(lower=1)),
             "momentum": close.pct_change(252) - close.pct_change(21)}
    if book_equity is not None:
        chars["log_bm"] = np.log((book_equity / mkt_cap.replace(0, np.nan)).clip(lower=1e-6))
    if net_income is not None and total_assets is not None:
        chars["roa"] = net_income / total_assets.replace(0, np.nan)
    if total_assets is not None:
        chars["asset_growth"] = total_assets.pct_change(252)
    return chars
