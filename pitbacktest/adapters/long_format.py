"""Bring your own point-in-time data: a long table (one row per security and day) into a Panel, with strict checks.

This is the door for CRSP (WRDS), Sharadar, Norgate, Polygon or any other source that keeps delisted securities. The
framework cannot create survivorship-free data; it can refuse to accept data that is ambiguous.

Required columns (names are configurable): a **permanent security id** (not the ticker), a date and a close.
Optional: volume, open/high/low, market cap, a point-in-time `in_universe` flag (for example S&P 500 membership on that
date), a `delist_return` on the last row of a security, and a ticker column (only used to detect ticker reuse).

Checks: no duplicate (id, date), positive closes, sorted dates. Delisting is detected as the last bar of a security that
ended well before the panel end. If a delisting return is given it is compounded into the last close (the CRSP
convention: the return on the last day is (1 + r_last) * (1 + r_delist) - 1), so the engine sees a real loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.panel import Panel


def panel_from_long(df: pd.DataFrame, *, id_col: str = "id", date_col: str = "date", close_col: str = "close",
                    volume_col: str | None = None, open_col: str | None = None, high_col: str | None = None,
                    low_col: str | None = None, mktcap_col: str | None = None, in_universe_col: str | None = None,
                    delist_return_col: str | None = None, ticker_col: str | None = None,
                    min_age_days: int = 60, adv_window: int = 30, min_adv: float | None = None,
                    ended_gap_days: int = 30, entry_lag: int = 1, market: str = "US",
                    periods_per_year: int = 252) -> Panel:
    need = [id_col, date_col, close_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"missing columns: {miss}")
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    if d.duplicated([id_col, date_col]).any():
        raise ValueError("duplicate (id, date) rows: the id must identify one security on one date")
    if (d[close_col] <= 0).any():
        raise ValueError("non-positive closes: clean them or drop the rows")
    d = d.sort_values([id_col, date_col])
    meta: dict = {"rows": int(len(d)), "securities": int(d[id_col].nunique())}

    last_row = d.groupby(id_col).tail(1).set_index(id_col)
    end = d[date_col].max()
    ended = last_row[date_col] < end - pd.Timedelta(days=ended_gap_days)
    meta["ended_before_end"] = int(ended.sum())
    meta["ended_share"] = float(ended.mean())

    n_dl = 0
    if delist_return_col and delist_return_col in d.columns:
        idx = d.groupby(id_col).tail(1).index
        dr = d.loc[idx, delist_return_col]
        hit = idx[dr.notna().values & ended.reindex(d.loc[idx, id_col]).values]
        d.loc[hit, close_col] = d.loc[hit, close_col] * (1.0 + d.loc[hit, delist_return_col])
        n_dl = len(hit)
    meta["delist_returns_applied"] = n_dl

    def mat(c: str) -> pd.DataFrame:
        return d.pivot(index=date_col, columns=id_col, values=c).sort_index()

    close = mat(close_col)
    vol = mat(volume_col) if volume_col else None
    age = close.notna().cumsum()
    if in_universe_col:
        ok = mat(in_universe_col).fillna(False).astype(bool) & close.notna()
    else:
        ok = close.notna() & (age >= min_age_days)
        if min_adv is not None and vol is not None:
            ok &= (close * vol).rolling(adv_window, min_periods=adv_window).median() >= min_adv
    da = pd.DataFrame(False, index=close.index, columns=close.columns)
    for s in ended[ended].index:
        da.loc[last_row.loc[s, date_col], s] = True
    if ticker_col and ticker_col in d.columns:
        t = d.groupby(ticker_col)[id_col].nunique()
        meta["reused_tickers"] = sorted(t[t > 1].index.astype(str).tolist())      # one ticker, several securities
    p = Panel(close=close, eligible=ok, open=mat(open_col) if open_col else None, high=mat(high_col) if high_col else None,
              low=mat(low_col) if low_col else None, volume=vol, mkt_cap=mat(mktcap_col) if mktcap_col else None,
              market=market, periods_per_year=periods_per_year, entry_lag=entry_lag, delist_after=da)
    p.meta = meta
    return p
