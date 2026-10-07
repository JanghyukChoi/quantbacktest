"""Liquidity-aware trading costs and capacity for crypto perpetuals.

There is no public order-book history in the archive, so costs here are *estimates*, and they are labelled as such.
- Fees: a flat taker fee (default 5 bp, the standard retail tier). Change it to your tier.
- Spread: Corwin-Schultz from daily highs and lows, smoothed with a trailing median and floored. It is noisy on
  daily data and tends to understate the cost of thin contracts, so a liquidity floor is added.
- Capacity: instead of pretending to model market impact without data, `participation_report` shows how large the
  trades are relative to each contract's turnover for a given account size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.costs import corwin_schultz


def liquidity_cost_bp(panel, *, taker_fee_bp: float = 5.0, window: int = 30, floor_bp: float = 1.0,
                      thin_usd: float = 2e7, thin_extra_bp: float = 5.0) -> pd.DataFrame:
    """(date x ticker) one-way cost in bp = fee + half the estimated spread, with a liquidity floor.

    Uses only data up to day t. Contracts below `thin_usd` trailing turnover pay `thin_extra_bp` more, a blunt
    penalty that stands in for impact we cannot measure."""
    cs = corwin_schultz(panel.high, panel.low)                       # ratio
    half_bp = (cs.rolling(window, min_periods=10).median() * 0.5 * 1e4).clip(lower=floor_bp)
    adv = panel.adv(window)
    extra = (adv < thin_usd).astype(float) * thin_extra_bp if adv is not None else 0.0
    return (taker_fee_bp + half_bp + extra).astype(np.float64)


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
