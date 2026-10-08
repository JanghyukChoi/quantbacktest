"""Trading costs: no flat assumptions, per-security measurement as the rule.

Why a flat assumption is dangerous
    Spreads differ by security, so assuming some round-trip bp for everything understates the cost of a high-turnover
    strategy more the higher its turnover, and can even flip the sign of the return. With high daily turnover a small
    difference in the one-way cost accumulates a lot over a year.

Estimators
    Roll (1984)       autocovariance of trade prices. The most reliable when intraday bars exist.
    Corwin-Schultz    high-low based. **Can return 0 for ETFs and low-volatility names, so be careful**.
    Traded-value regression  interpolates securities with no measurement as log(bp) = a + b*log(traded value).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def roll_spread(prices: np.ndarray) -> float:
    """Roll (1984) effective spread (in price units). NaN if the autocovariance is positive."""
    p = np.asarray(prices, dtype=np.float64)
    p = p[np.isfinite(p)]
    if len(p) < 30:
        return np.nan
    d = np.diff(p)
    c = np.cov(d[1:], d[:-1])[0, 1]
    return 2 * np.sqrt(-c) if c < 0 else np.nan


def corwin_schultz(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Corwin-Schultz (2012) high-low spread estimate (as a ratio).

    Warning: for assets with a narrow high-low range (ETFs, large low-volatility stocks) the estimate goes negative and is clipped to 0.
          If the result is all 0, do not use this estimator; move to Roll or the traded-value regression.
    """
    h, l = np.log(high), np.log(low)
    beta = (h - l) ** 2 + (h.shift(1) - l.shift(1)) ** 2
    h2 = np.maximum(high, high.shift(1))          # elementwise: pd.concat(axis=1) would double the columns
    l2 = np.minimum(low, low.shift(1))
    gamma = (np.log(h2) - np.log(l2)) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    return (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0, upper=0.2)


def fit_spread_model(dvol_m: np.ndarray, spread_bp: np.ndarray) -> tuple[float, float]:
    """Regression log(one-way bp) = a + b*log(traded value in M). For interpolating securities with no measurement."""
    m = np.isfinite(dvol_m) & np.isfinite(spread_bp) & (dvol_m > 0) & (spread_bp > 0)
    if m.sum() < 20:
        return np.nan, np.nan
    b, a = np.polyfit(np.log(dvol_m[m]), np.log(spread_bp[m]), 1)
    return float(a), float(b)


def spread_panel(adv: pd.DataFrame, *, a: float, b: float,
                 measured: dict[str, float] | None = None,
                 lo: float = 0.05, hi: float = 50.0) -> pd.DataFrame:
    """(date x ticker) panel of one-way spreads (bp). Measured values override it when present."""
    dv_m = (adv / 1e6).clip(lower=0.1)
    est = np.exp(a + b * np.log(dv_m))
    if measured:
        for k, v in measured.items():
            if k in est.columns and np.isfinite(v):
                est[k] = v
    return est.clip(lower=lo, upper=hi)


def apply_turnover_cost(holdings: np.ndarray, spread_bp: np.ndarray | float) -> np.ndarray:
    """Cost proportional to turnover. Returns the daily cost (in return units).

    holdings   (date x ticker) weights. Sum = 1 (long) or one per leg for long-short.
    spread_bp  scalar (flat) or a (date x ticker) measured panel. **A panel is recommended**.
    """
    d = np.abs(np.diff(holdings, axis=0))
    cost = np.zeros(holdings.shape[0])
    if np.isscalar(spread_bp):
        cost[1:] = d.sum(axis=1) * (spread_bp / 1e4) * 0.5
    else:
        s = np.nan_to_num(np.asarray(spread_bp), nan=float(np.nanmedian(spread_bp)))
        cost[1:] = (d * s[1:] / 1e4).sum(axis=1)
    return cost


def rate_schedule(dates: pd.DatetimeIndex, entries, name: str = "rate") -> pd.Series:
    """A rate that changes on known effective dates, as a value for every date: each date takes the latest entry whose
    effective date is on or before it. `entries` is a list of (effective_date, value). A date before the first entry raises,
    because the rate is then unknown and a made-up rate (or 0) would hide that."""
    if not len(entries):
        raise ValueError(f"{name}: the schedule is empty")
    s = pd.Series({pd.Timestamp(d): float(v) for d, v in entries}).sort_index()
    out = s.reindex(dates, method="ffill")
    if out.isna().any():
        raise ValueError(f"{name}: no rate is known before {s.index[0].date()}, but the panel starts on {dates[0].date()} "
                         f"(add an earlier entry or start the panel later)")
    return out


def side_cost_input(value, dates: pd.DatetimeIndex, tickers: pd.Index, name: str):
    """Validate a one-way cost for one side of a trade (bp) and bring it to a form `apply_side_cost` can use: a float, a
    (dates x 1) array (a Series indexed by date: a rate that changes over time) or a (dates x tickers) array. Unlike `spread_bp`, a
    missing value raises instead of being replaced by a median: a rate you do not know is not a rate of zero."""
    if value is None:
        return 0.0
    if isinstance(value, pd.DataFrame):
        a = value.reindex(index=dates, columns=tickers).to_numpy(float)
    elif isinstance(value, pd.Series):
        if not isinstance(value.index, pd.DatetimeIndex):
            raise ValueError(f"{name}: a Series must be indexed by date")
        a = value.sort_index().reindex(dates, method="ffill").to_numpy(float).reshape(-1, 1)
    else:
        a = np.asarray(value, dtype=float)
        if a.ndim == 0:
            if not np.isfinite(a) or a < 0:
                raise ValueError(f"{name} must be a finite number, not negative, got {value!r}")
            return float(a)
        if a.shape != (len(dates), len(tickers)):
            raise ValueError(f"{name}: an array must have shape (dates, tickers) = {(len(dates), len(tickers))}, got {a.shape}")
    if np.isnan(a).any():
        raise ValueError(f"{name}: the value is unknown for some dates or securities (NaN, or a date before the first entry); "
                         f"a missing cost is an error, not zero")
    if np.isinf(a).any() or (a < 0).any():
        raise ValueError(f"{name} must be finite and not negative (a negative cost would pay you for trading)")
    return a


def apply_side_cost(holdings: np.ndarray, buy_bp, sell_bp) -> np.ndarray:
    """Daily cost of trades that depends on the side: `buy_bp` per unit of weight bought and `sell_bp` per unit sold (one-way,
    bp, on top of the spread). A weight going up is a buy and a weight going down is a sell, so opening a short is a sell and
    covering it a buy. The first row, a build from cash, is not charged (as for the spread). `buy_bp` and `sell_bp` are floats or
    arrays from `side_cost_input`."""
    d = np.diff(holdings, axis=0)
    out = np.zeros(holdings.shape[0])
    b = buy_bp[1:] if isinstance(buy_bp, np.ndarray) else buy_bp
    s = sell_bp[1:] if isinstance(sell_bp, np.ndarray) else sell_bp
    out[1:] = (np.maximum(d, 0.0) * b + np.maximum(-d, 0.0) * s).sum(axis=1) / 1e4
    return out


def turnover(holdings: np.ndarray) -> np.ndarray:
    """Daily one-way turnover."""
    t = np.zeros(holdings.shape[0])
    t[1:] = 0.5 * np.abs(np.diff(holdings, axis=0)).sum(axis=1)
    return t
