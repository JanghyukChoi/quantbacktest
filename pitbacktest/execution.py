"""Whether a trade can be done, and at what size, at the price the backtest uses.

A daily backtest that trades every position change at the close assumes three things that are often false: that the security
was open (a halted name cannot be traded), that the price was not locked at a daily limit (a name closed at the upper limit has a
queue of buyers and no sellers, so you cannot buy it), and that you can hold any fraction of a share. This module states those
assumptions as data the engines read.

    can_buy, can_sell   `Panel` fields (bool, date x ticker). A trade that increases a position needs `can_buy` on the execution
                        day, one that decreases it needs `can_sell`. A blocked trade does not happen and the position stays as it was.
    tradability()       builds them from prices and volume: no price or no volume means halted; a daily move at the limit means locked.
    at_prices()         a copy of the panel that trades at another price (the open instead of the close).
    realize()           the engine's position path under those rules, plus whole-lot sizes and a minimum trade value.

Nothing here is specific to one market. Limits and lot sizes differ by market and change by rule, so they are arguments, not a table
inside the library.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .core.costs import rate_schedule
from .core.panel import Panel


def tradability(panel: Panel, *, limit=None, tol: float = 0.01, halt_on_zero_volume: bool = True,
                max_gap_days: int | None = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(can_buy, can_sell) built from the panel's prices and volume.

    After a security's flagged delisting (`delist_after`) both trades are allowed, so that a position in it can be closed.
    Halted: no close that day, or (if the panel has `volume` and `halt_on_zero_volume`) no volume. A security with a close but a
    missing volume is also treated as halted, because an unknown is not permission. A panel without `volume` is judged by prices alone.
    limit   the daily price limit as a fraction (0.30 for 30%), a list of (effective_date, fraction) when it changed, or None for a
            market without one. A day whose move is at least `limit - tol` up cannot be bought (a locked-up name has no sellers) and a
            day at least `limit - tol` down cannot be sold. The tolerance is there because tick rounding keeps the limit move slightly
            under the limit. This is **conservative**: it also blocks days that reached the limit but traded freely before it.
    tol     in fractions, 0 <= tol < limit.
    max_gap_days  after this many consecutive days **without any price**, a position is treated as settled and can be closed either way
            (None: never). An exchange settles a delisted contract and a vendor gap or a ticker that came back years later (a contract
            that was absent for years and then listed again, in the Binance cache) is not a position you can hold through. A name with a price
            but no volume (a Korean trading suspension) is not covered: that one really is frozen, for as long as the data says."""
    c = panel.close
    ok = c.notna()
    if halt_on_zero_volume and panel.volume is not None:
        ok = ok & (panel.volume.fillna(0.0) > 0)
    can_buy, can_sell = ok.copy(), ok.copy()
    if panel.delist_after is not None and bool(np.asarray(panel.delist_after.values).any()):
        # After a flagged delisting there is no price for ever. The engines already settle such a position (`delist_return`, or the last
        # price), so closing it is allowed in **both** directions (a long is sold, a short is bought back): refusing the exit would freeze a
        # dead position in the book for the rest of the sample, and a frozen short is exactly what the first real-data run showed.
        da = panel.delist_after.reindex(index=c.index, columns=c.columns).fillna(False).astype(bool)
        gone = (da.astype(int).cumsum() - da.astype(int)) > 0
        can_buy, can_sell = can_buy | gone, can_sell | gone
    if max_gap_days is not None:
        isn = c.isna().to_numpy()
        run = np.zeros(c.shape[1], dtype=int)
        long_gap = np.zeros(c.shape, dtype=bool)
        for t in range(c.shape[0]):
            run = np.where(isn[t], run + 1, 0)
            long_gap[t] = run > max_gap_days
        settled = pd.DataFrame(long_gap, index=c.index, columns=c.columns)
        can_buy, can_sell = can_buy | settled, can_sell | settled
    if limit is not None:
        lim = (rate_schedule(panel.dates, limit, "limit").to_numpy(float) if isinstance(limit, (list, tuple))
               else np.full(len(panel.dates), float(limit)))
        if not (np.all(lim > 0) and np.all(lim < 1)):
            raise ValueError(f"limit must be a fraction between 0 and 1 (0.30 for 30%), got {limit!r}")
        if not (0 <= tol < lim.min()):
            raise ValueError(f"tol must be at least 0 and below the limit, got {tol!r}")
        ret = c.pct_change(fill_method=None).to_numpy(float)
        thr = (lim - tol)[:, None]
        with np.errstate(invalid="ignore"):
            up, down = ret >= thr, ret <= -thr
        can_buy = can_buy & ~pd.DataFrame(up, index=c.index, columns=c.columns)
        can_sell = can_sell & ~pd.DataFrame(down, index=c.index, columns=c.columns)
    return can_buy, can_sell


def at_prices(panel: Panel, price="open") -> Panel:
    """A copy of `panel` whose prices are another series, so that a position is entered and marked at it. `price` is the name of a panel
    field (`"open"`, `"high"`, `"low"`) or a (date x ticker) frame. With the open, a signal from the close of day d is entered at the open
    of d+lag and held to the open of d+lag+1. Only `close` changes: eligibility, volume and the rest were built from the original
    closes, and `screen` and `backtest_event` still compute their controls from whatever `close` now holds."""
    px = getattr(panel, price) if isinstance(price, str) else price
    if px is None:
        raise ValueError(f"the panel has no {price!r}")
    if not isinstance(px, pd.DataFrame):
        raise ValueError("price must be a field name or a (date x ticker) frame")
    return replace(panel, close=px.reindex(index=panel.dates, columns=panel.tickers))


def exec_masks(panel: Panel):
    """(can_buy, can_sell) as arrays indexed by the **signal date**, i.e. read on the execution day d + entry_lag. None when the panel
    has neither field. A missing frame counts as always possible."""
    if panel.can_buy is None and panel.can_sell is None:
        return None, None
    lag = panel.entry_lag

    def one(f):
        if f is None:
            return np.ones(panel.close.shape, dtype=bool)
        return f.shift(-lag, fill_value=True).to_numpy(bool)
    return one(panel.can_buy), one(panel.can_sell)


def round_to_lots(weights: np.ndarray, capital: float, price: np.ndarray, lot) -> np.ndarray:
    """The weights you can actually hold: whole lots at the given prices. Shares = weight x capital / price, rounded to a multiple of
    `lot`. `price` must be the **real** price level (not a back-adjusted series, whose level is arbitrary). A name with no price keeps weight NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = weights * capital / price
        lots = np.round(shares / lot) * lot
        return lots * price / capital


def realize(target: np.ndarray, can_buy: np.ndarray | None = None, can_sell: np.ndarray | None = None, *,
            capital: float | None = None, price: np.ndarray | None = None, lot=1.0, min_trade_value: float = 0.0):
    """The positions that result from asking for `target` row by row (row 0 is a build from cash).

    Each day, per name: round the target to whole lots (when `capital` is given; `price` is read on the **same row** as the target, so pass the price of the day the
    trade is done: `backtest_weights` shifts it by `entry_lag`); skip the trade if its value is under `min_trade_value` (a number, or one per security);
    skip it if it increases the position where `can_buy` is False, or decreases it where `can_sell` is False. A skipped trade leaves the
    position as it was the day before. A name with no price on a day (NaN) keeps its position.
    The weight is what the engines keep constant, so a day without a trade keeps the previous **weight**, not the previous share count: after the price moves, the
    shares it implies need not be a multiple of the lot (a halted name does not move, so there it is). Returns (positions, stats). `mean_stuck_weight` is the average over days of the weight held where the target said otherwise because a trade was
    blocked; `longest_freeze_days` the longest run of consecutive days a wanted trade in one name was blocked. A position that cannot be traded
    does not free the capital it ties up: the engines still add new targets on top of it."""
    T, N = target.shape
    use_lots = capital is not None
    mtv = np.asarray(min_trade_value, dtype=float)                       # a number, or one per security
    mtv_any = bool(np.any(mtv > 0))
    out = np.zeros_like(target, dtype=float)
    prev = np.zeros(N)
    asked = blocked_turn = skipped_turn = gap = stuck = 0.0
    n_blocked = n_skipped = 0
    streak = np.zeros(N, dtype=int)
    longest = 0
    for t in range(T):
        tgt = target[t]
        if use_lots:
            lt = lot[t] if isinstance(lot, np.ndarray) and lot.ndim == 2 else lot
            tgt = round_to_lots(tgt, capital, price[t], lt)
            # A flat target needs no price to carry out (a delisted name is simply closed); a non-flat target with no price cannot be sized, so the
            # position stays.
            tgt = np.where(target[t] == 0, 0.0, np.where(np.isfinite(tgt), tgt, prev))
            gap += float(np.abs(tgt - target[t]).sum())
        d = tgt - prev
        stay = np.zeros(N, dtype=bool)
        if use_lots and mtv_any:
            small = (np.abs(d) * capital < mtv) & (d != 0)
            stay |= small
        blk = np.zeros(N, dtype=bool)
        if can_buy is not None:
            blk |= (d > 0) & ~can_buy[t]
        if can_sell is not None:
            blk |= (d < 0) & ~can_sell[t]
        cur = np.where(stay | blk, prev, tgt)
        held = blk & ~stay
        stuck += float(np.abs(cur - tgt)[held].sum())                         # weight sitting where it should not be, today
        streak = np.where(held, streak + 1, 0)
        longest = max(longest, int(streak.max())) if N else longest
        if t > 0:
            asked += float(np.abs(d).sum())
            blocked_turn += float(np.abs(d[blk & ~stay]).sum())
            skipped_turn += float(np.abs(d[stay]).sum())
            n_blocked += int((blk & ~stay).sum())
            n_skipped += int(stay.sum())
        out[t] = cur
        prev = cur
    stats = {"mean_stuck_weight": stuck / T if T else 0.0, "longest_freeze_days": longest, "asked_turnover": asked, "blocked_turnover": blocked_turn, "blocked_trades": n_blocked, "blocked_turnover_share": blocked_turn / asked if asked > 0 else 0.0,
             "min_trade_skipped": n_skipped, "min_trade_skipped_share": skipped_turn / asked if asked > 0 else 0.0,
             "mean_abs_rounding_gap": gap / (T * N) if use_lots else 0.0}
    return out, stats
