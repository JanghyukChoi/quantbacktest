"""Trading costs and capacity for crypto perpetuals.

There is no public order-book history in the archive, so costs here are *estimates*, and they are labelled as such.
- Fees: a flat taker fee (default 5 bp, the standard retail tier). Change it to your tier.
- Spread: a flat assumption plus a thin-contract penalty. A daily high-low estimator (Corwin-Schultz) was tried first
  and rejected because it is invalid for crypto volatility (see `liquidity_cost_bp`).
- Capacity: instead of pretending to model market impact without data, `participation_report` shows how large the
  trades are relative to each contract's turnover for a given account size.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..core.costs import corwin_schultz


def liquidity_cost_bp(panel, *, taker_fee_bp: float = 5.0, half_spread_bp: float = 2.0, window: int = 30,
                      thin_usd: float = 1e8, thin_extra_bp: float = 5.0,
                      spread_estimator: str = "fixed") -> pd.DataFrame:
    """(date x ticker) one-way cost in bp = taker fee + half spread (+ a penalty for thin contracts).

    spread_estimator
      "fixed"           a flat half spread (`half_spread_bp`). The default. It is an assumption, not a measurement, so
                        use `cost_sensitivity` or a grid of values instead of trusting one number.
      "corwin_schultz"  half the Corwin-Schultz estimate from daily highs and lows. **Do not use it for crypto.**
                        On Binance perpetuals it gave a median spread of about 1.5% and about 38 bp one way for
                        BTC, orders of magnitude above any real quote (the study in `studies/crypto_cross_section`
                        documents this in Amendment 1). I did not establish why; daily crypto ranges are dominated
                        by jumps and volatility clustering, which the estimator's assumptions do not cover. It is
                        kept only to reproduce the preregistered run.

    Contracts whose trailing turnover is below `thin_usd` pay `thin_extra_bp` more, a blunt stand-in for impact.
    Uses only data up to day t."""
    adv = panel.adv(window)
    extra = (adv < thin_usd).astype(float) * thin_extra_bp if adv is not None else 0.0
    if spread_estimator == "corwin_schultz":
        warnings.warn("Corwin-Schultz overstates crypto spreads by orders of magnitude; use only to reproduce old runs.",
                      stacklevel=2)
        cs = corwin_schultz(panel.high, panel.low)
        half = (cs.rolling(window, min_periods=10).median() * 0.5 * 1e4).clip(lower=1.0)
    elif spread_estimator == "fixed":
        half = half_spread_bp
    else:
        raise ValueError(spread_estimator)
    return (taker_fee_bp + half + extra).astype(np.float64)


def participation_report(panel, weights: np.ndarray, aum_usd: float, window: int = 30) -> dict:
    """Trade size as a share of trailing daily turnover, for an account of `aum_usd`.

    weights  (date x ticker) target weights, e.g. the `holdings` of a portfolio. Returns percentiles of
    |delta weight| * aum / ADV over all trades, and the account size at which the 95th percentile would reach 1%
    of a contract's daily turnover (a common ceiling for not moving the market)."""
    adv = panel.adv(window).values
    dw = np.abs(np.diff(weights, axis=0))
    part = dw * aum_usd / np.where(adv[1:] > 0, adv[1:], np.nan)
    part = part[(dw > 1e-9) & np.isfinite(part)]
    if part.size == 0:
        return {"aum_usd": aum_usd, "n_trades": 0}
    p50, p95, p99 = (float(np.percentile(part, q)) for q in (50, 95, 99))
    return {"aum_usd": aum_usd, "n_trades": int(part.size), "participation_p50": p50, "participation_p95": p95,
            "participation_p99": p99, "aum_at_1pct_p95": float(aum_usd * 0.01 / p95) if p95 > 0 else np.inf}
