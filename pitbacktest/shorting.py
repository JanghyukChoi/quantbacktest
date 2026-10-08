"""Which securities can be sold short on which dates.

A backtest that shorts a security on a day when short selling is banned, or when there is nothing to borrow, earns money that
was never available. `Panel.shortable` is the (date x ticker) bool frame the engines read; this module builds it from ban periods.

The library does not ship a calendar of bans. Dates, and which securities were exempt, differ by market and change by decision, and
a wrong table would be worse than none, so you pass the periods you have checked against the regulator's notice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def shortable_from_bans(dates: pd.DatetimeIndex, tickers: pd.Index, bans, exempt=None) -> pd.DataFrame:
    """(dates x tickers) bool: True where short selling is allowed.

    bans    list of (start, end) pairs, both inclusive; `end=None` means the ban has not ended. Dates may be strings.
            Periods may overlap. Outside every period everything is shortable.
    exempt  securities that may still be sold short during a ban (for example market-maker or index-constituent exceptions):
            None (nobody), a list or set of tickers (exempt in every ban period), or a (dates x tickers) bool frame (True =
            exempt that day; anything not in the frame is not exempt).

    A partial ban (only some securities may be shorted again) is a ban for the others: pass the securities that may be shorted as
    `exempt`. If you do not know them, do not pretend: treating the whole period as a full ban (no exempt) is the cautious choice
    and the result is a lower bound on what a short-selling strategy could have done."""
    dates = pd.DatetimeIndex(dates)
    banned = pd.Series(False, index=dates)
    for item in bans:
        if len(item) != 2:
            raise ValueError(f"each ban must be a (start, end) pair, got {item!r}")
        start, end = pd.Timestamp(item[0]), (None if item[1] is None else pd.Timestamp(item[1]))
        if end is not None and end < start:
            raise ValueError(f"a ban ends before it starts: {item!r}")
        banned |= (dates >= start) & (True if end is None else dates <= end)
    out = pd.DataFrame(True, index=dates, columns=tickers)
    if not banned.any():
        return out
    if exempt is None:
        allowed = pd.DataFrame(False, index=dates, columns=tickers)
    elif isinstance(exempt, pd.DataFrame):
        allowed = exempt.reindex(index=dates, columns=tickers).fillna(False).astype(bool)
    else:
        allowed = pd.DataFrame(False, index=dates, columns=tickers)
        allowed[[t for t in tickers if t in set(exempt)]] = True
    return pd.DataFrame(np.where(banned.to_numpy()[:, None], allowed.to_numpy(bool), True), index=dates, columns=tickers)
