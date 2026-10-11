# Reference implementation written from the specification alone by an assistant that had not seen the library (docs/blind_review/README.md); unchanged except for this line.
# tests/test_blind_reference.py compares it with the library.
"""Blind implementation of the simple long-short factor portfolio (spec A)."""
import warnings

import numpy as np
import pandas as pd


def _legs(eligible, factor, long_q, short_q, weighting):
    """Per-signal-day normalised leg weights (magnitudes, each leg sums to 1)."""
    fac = factor.astype(float)
    valid = eligible.astype(bool) & fac.notna()
    f = fac.where(valid)
    r = f.rank(axis=1, pct=True)  # percentile among eligible & non-NaN, ties averaged

    def build(mask, kind, q):
        mask = mask & valid
        if weighting == "equal":
            w = mask.astype(float)
        elif weighting == "signal":
            w = f.abs().where(mask, 0.0)
        elif weighting == "rank":
            if kind == "long":
                w = (r - (1.0 - q)).where(mask, 0.0)
            else:
                w = (q - r).where(mask, 0.0)
        else:
            raise ValueError("weighting must be equal | signal | rank")
        w = w.fillna(0.0)
        s = w.sum(axis=1)
        w = w.div(s.where(s > 0), axis=0).fillna(0.0)  # empty / zero-sum leg -> all zeros
        return w.to_numpy(float)

    L = build(r >= (1.0 - long_q), "long", long_q)
    S = None
    if short_q is not None:
        S = build(r <= short_q, "short", short_q)
    return L, S


def _held(Lg, hold):
    """Average of the most recent `hold` signal-day legs (fewer at the start)."""
    T = Lg.shape[0]
    acc = np.zeros_like(Lg)
    for k in range(hold):
        if k >= T:
            break
        acc[k:] += Lg[: T - k]
    cnt = np.minimum(np.arange(T) + 1, hold).astype(float)[:, None]
    return acc / cnt


def _check(long_q, short_q, hold, entry_lag, weighting, factor):
    if not (0 < long_q <= 1):
        raise ValueError("long_q in (0,1]")
    if short_q is not None:
        if not (0 < short_q <= 1):
            raise ValueError("short_q must be > 0")
        if long_q + short_q > 1:
            raise ValueError("long_q + short_q above 1")
    if int(hold) != hold or hold < 1:
        raise ValueError("hold must be an int >= 1")
    if entry_lag < 0 or int(entry_lag) != entry_lag:
        raise ValueError("entry_lag must be >= 0")
    if weighting not in ("equal", "signal", "rank"):
        raise ValueError("weighting")
    if all(dt == bool for dt in factor.dtypes):
        raise ValueError("boolean factor refused")


def _weights(close, eligible, factor, long_q, short_q, hold, entry_lag, weighting):
    _check(long_q, short_q, hold, entry_lag, weighting, factor)
    elig = eligible.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    fac = factor.reindex(index=close.index, columns=close.columns)
    L, S = _legs(elig, fac, long_q, short_q, weighting)
    WL = _held(L, int(hold))
    WS = _held(S, int(hold)) if S is not None else None
    return WL, WS


def portfolio_net_returns(close, eligible, factor, long_q, short_q, hold, spread_bp,
                          entry_lag=1, weighting="equal"):
    spread_bp = float(spread_bp)
    if not np.isfinite(spread_bp) or spread_bp < 0:
        raise ValueError("spread_bp")
    T = len(close)
    n = T - entry_lag - 1
    if n <= 0:
        return pd.Series(dtype=float, index=close.index[:0])
    WL, WS = _weights(close, eligible, factor, long_q, short_q, hold, entry_lag, weighting)
    c = close.astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.full_like(c, np.nan)
        ret[1:] = c[1:] / c[:-1] - 1.0
    ret = np.where(np.isnan(ret), 0.0, ret)
    fwd = ret[entry_lag + 1: entry_lag + 1 + n]  # row d -> ret over close(d+lag)->close(d+lag+1)
    gross = (WL[:n] * fwd).sum(axis=1)
    if WS is not None:
        gross = gross - (WS[:n] * fwd).sum(axis=1)
    cost = np.zeros(n)
    trade = np.abs(np.diff(WL[:n], axis=0)).sum(axis=1)
    if WS is not None:
        trade = trade + np.abs(np.diff(WS[:n], axis=0)).sum(axis=1)
    cost[1:] = trade * spread_bp / 2.0 / 1e4  # first row: built from cash, no cost
    return pd.Series(gross - cost, index=close.index[:n], dtype=float)


def turnover_daily(close, eligible, factor, long_q, short_q, hold, entry_lag=1, weighting="equal"):
    T = len(close)
    n = T - entry_lag - 1
    if n <= 1:
        return float("nan")
    WL, WS = _weights(close, eligible, factor, long_q, short_q, hold, entry_lag, weighting)
    trade = np.abs(np.diff(WL[:n], axis=0)).sum(axis=1)
    if WS is not None:
        trade = trade + np.abs(np.diff(WS[:n], axis=0)).sum(axis=1)
    return float(trade.mean())  # mean over rows 1..n-1 (row 0 is the build from cash)


def stop_at_ruin(net, cut):
    out = np.array(net, dtype=float, copy=True)
    k = min(int(cut), len(out))
    idx = np.flatnonzero(out[:k] <= -1.0)
    if len(idx) == 0:
        return out, None
    i = int(idx[0])
    out[i] = -1.0
    out[i + 1:] = 0.0
    return out, i


def metrics_from_returns(net, ann=252):
    r = np.asarray(net, dtype=float)
    r, _ = stop_at_ruin(r, len(r))
    n = len(r)
    nan = float("nan")
    res = dict(CAGR=nan, MDD=nan, Sharpe=nan, Sortino=nan, Calmar=nan, vol=nan, pos_days=nan)
    if n == 0:
        return res
    mean = r.mean()
    std = r.std(ddof=1) if n > 1 else nan
    res["vol"] = float(std * np.sqrt(ann)) if np.isfinite(std) else nan
    res["pos_days"] = float((r > 0).mean())
    dd = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    res["Sortino"] = float(mean / dd * np.sqrt(ann)) if dd > 1e-12 else nan
    if n < ann // 2:
        warnings.warn(f"only {n} observations (< ann//2 = {ann // 2}): CAGR, MDD, Sharpe are NaN")
        return res
    cum = np.cumprod(1.0 + r)
    with np.errstate(invalid="ignore"):
        cagr = cum[-1] ** (ann / n) - 1.0
    peak = np.maximum(np.maximum.accumulate(cum), 1.0)
    mdd = float(np.min(cum / peak - 1.0))
    res["CAGR"] = float(cagr)
    res["MDD"] = mdd
    res["Sharpe"] = float(mean / std * np.sqrt(ann)) if (np.isfinite(std) and std > 1e-12) else nan
    res["Calmar"] = float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else nan
    return res
