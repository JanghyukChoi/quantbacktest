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


def halt_runs(panel: Panel) -> np.ndarray:
    """(date x ticker) int: how many consecutive days, counting the day itself, the security has had a price but no (or unknown) volume; 0 on a day it trades or has no price.
    A panel without `volume` has none."""
    c = panel.close
    if panel.volume is None:
        return np.zeros(c.shape, dtype=int)
    halted = (c.notna() & ~(panel.volume.fillna(0.0) > 0)).to_numpy()
    run = np.zeros(c.shape, dtype=int)
    for t in range(c.shape[0]):
        run[t] = np.where(halted[t], (run[t - 1] if t else 0) + 1, 0)
    return run


def freeze_hits(panel: Panel, freeze_days: int | None) -> np.ndarray | None:
    """(date x ticker) bool indexed by the **signal date**: True where the execution day (signal date + `entry_lag`) is exactly the `freeze_days`-th day of a suspension.
    None when `freeze_days` is None. A suspension marks a position down once and not again however long it lasts. The markdown is booked on the signal-date row, which is
    the `freeze_days`-th suspended day minus `entry_lag`, and it enters the return of that row (close of the execution day to the close of the next), so it is the
    `freeze_days`-th suspended day's close that takes the loss."""
    if freeze_days is None:
        return None
    if isinstance(freeze_days, bool) or not isinstance(freeze_days, (int, np.integer)) or freeze_days < 1:
        raise ValueError(f"freeze_days must be a whole number of days, at least 1, or None, got {freeze_days!r}")
    hit = halt_runs(panel) == freeze_days
    lag = panel.entry_lag
    return np.vstack([hit[lag:], np.zeros((lag, hit.shape[1]), bool)]) if lag else hit


def check_freeze_return(freeze_return: float) -> float:
    """`freeze_return` as a float, or ValueError unless it is between -1 and 0 (a markdown is never a gain)."""
    if isinstance(freeze_return, (bool, np.bool_)) or not isinstance(freeze_return, (int, float, np.integer, np.floating)):
        raise ValueError(f"freeze_return must be a number between -1 and 0, got {freeze_return!r}")
    if not (np.isfinite(freeze_return) and -1.0 <= freeze_return <= 0.0):
        raise ValueError(f"freeze_return must be between -1 and 0 (a markdown is never a gain), got {freeze_return!r}")
    return float(freeze_return)


def freeze_episodes(panel: Panel, *, min_days: int = 20, relist_days: int = 20) -> pd.DataFrame:
    """The stretches of at least `min_days` consecutive days on which a security had a price but no volume (a trading suspension), and how each ended.

    One row per episode: ticker, start, end (the last suspended day), days, outcome and `ret`, the return from the last price before the suspension
    to the price at the end of the story.
    A day without a price (NaN) ends a suspension, so a halt with a gap of missing prices inside it is cut into two episodes. A suspension that is already running
    on the panel's first day has no price before it: such an episode is listed, but its `ret` is NaN, and its length is truncated.
    outcome  "resumed"               the first priced day after the suspension, and the data goes on (or the security is not flagged in `delist_after`):
                                      ret is that day's price over the price before the suspension. That day may itself have no volume if missing prices sit between.
             "resumed_then_delisted"  the security is flagged in `delist_after` and its last bar falls within `relist_days` of that day (a liquidation window): ret is its
                                      last price over the price before. Without the flag the same story is labeled "resumed".
             "ended_in_halt"          the security's last bar is a suspended day and it is flagged in `delist_after`: the price never moved, ret is 0 (NaN if there is
                                      no price before), and the loss, if any, was never in the data
             "ongoing"                no priced day follows: either still suspended on the last day of the panel, or the data ends in the suspension without a delisting
                                      flag; ret is NaN
    The point is to measure how often a position that cannot be sold is lost, instead of assuming it. Counts are small and depend on the market and
    the period; read the distribution of `ret`, not a mean."""
    c = panel.close
    halted = (c.notna() & ~(panel.volume.fillna(0.0) > 0)).to_numpy() if panel.volume is not None else np.zeros(c.shape, bool)
    px = c.to_numpy(float)
    da = panel.delist_after.reindex(index=c.index, columns=c.columns).fillna(False).to_numpy(bool) if panel.delist_after is not None else np.zeros(c.shape, bool)
    T, N = c.shape
    rows = []
    for j in range(N):
        h = halted[:, j]
        if not h.any():
            continue
        t = 0
        valid = np.where(~np.isnan(px[:, j]))[0]
        last_bar = int(valid[-1]) if len(valid) else -1
        flagged = bool(da[:, j].any())
        while t < T:
            if not h[t]:
                t += 1
                continue
            e = t
            while e + 1 < T and h[e + 1]:
                e += 1
            if e - t + 1 >= min_days:
                before = next((px[k, j] for k in range(t - 1, -1, -1) if not np.isnan(px[k, j]) and not h[k]), np.nan)
                if e == T - 1 and not (flagged and last_bar == e):
                    outcome, ret = "ongoing", np.nan
                elif flagged and last_bar == e:
                    outcome, ret = "ended_in_halt", 0.0 if np.isfinite(before) else np.nan
                else:
                    nxt = e + 1
                    while nxt < T and np.isnan(px[nxt, j]):
                        nxt += 1
                    if nxt >= T:
                        outcome, ret = "ongoing", np.nan
                    elif flagged and last_bar - nxt <= relist_days:
                        outcome, ret = "resumed_then_delisted", px[last_bar, j] / before - 1.0 if np.isfinite(before) else np.nan
                    else:
                        outcome, ret = "resumed", px[nxt, j] / before - 1.0 if np.isfinite(before) else np.nan
                rows.append({"ticker": c.columns[j], "start": c.index[t], "end": c.index[e], "days": e - t + 1, "outcome": outcome, "ret": ret})
            t = e + 1
    return pd.DataFrame(rows, columns=["ticker", "start", "end", "days", "outcome", "ret"])


def at_prices(panel: Panel, price="open") -> Panel:
    """A copy of `panel` whose prices are another series, so that a position is entered and marked at it. `price` is the name of a panel
    field (`"open"`, `"high"`, `"low"`) or a (date x ticker) frame. With the open, a signal from the close of day d is entered at the open
    of d+lag and held to the open of d+lag+1. A price that is zero or negative is treated as missing. Only `close` changes: eligibility, volume and the rest were built from the original
    closes, and `screen` and `backtest_event` still compute their controls from whatever `close` now holds."""
    px = getattr(panel, price) if isinstance(price, str) else price
    if px is None:
        raise ValueError(f"the panel has no {price!r}")
    if not isinstance(px, pd.DataFrame):
        raise ValueError("price must be a field name or a (date x ticker) frame")
    px = px.reindex(index=panel.dates, columns=panel.tickers)
    return replace(panel, close=px.where(px > 0))                # a zero or negative price is no price: the engines treat a missing price as a return of 0, a zero would give infinity


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
            capital: float | None = None, price: np.ndarray | None = None, lot=1.0, min_trade_value: float = 0.0, cap_gross: bool = False):
    """The positions that result from asking for `target` row by row (row 0 is a build from cash).

    Each day, per name: round the target to whole lots (when `capital` is given; `price` is read on the **same row** as the target, so pass the price of the day the
    trade is done: `backtest_weights` shifts it by `entry_lag`); skip the trade if its value is under `min_trade_value` (a number, or one per security);
    skip it if it increases the position where `can_buy` is False, or decreases it where `can_sell` is False. A skipped trade leaves the
    position as it was the day before. A name with no price on a day (NaN) keeps its position.
    The weight is what the engines keep constant, so a day without a trade keeps the previous **weight**, not the previous share count: after the price moves, the
    shares it implies need not be a multiple of the lot (a halted name does not move, so there it is).

    `cap_gross`: a position that cannot be traded ties up capital. Without this, new targets are added on top of it and the gross exposure can exceed the
    target's. With it, the day's whole target is scaled by a k in [0, 1], the largest one found by bisection for which the resulting gross exposure does not exceed
    the gross of the day's target (after lot rounding, so rounding itself is never cut). A position that cannot be sold stays where it is; one that cannot be
    bought stays too unless the scaled target falls below it, in which case it is sold down, so it can shrink. A scaled target can turn a buy into a sell, which
    changes what is blocked, so k is searched for rather than solved for, and the feasible set being an interval [0, k] is not proven (no counter-example was found
    in 653 random small cases). On a day with nothing blocked or skipped k is exactly 1 and the result is the same as without it, with or without `capital`. If the
    stuck positions alone exceed the target gross, nothing can meet the cap: k is 0 and the day is counted in `infeasible_days`.

    Returns (positions, stats). `mean_stuck_weight` is the average over days of the weight held where the target said otherwise because a trade was
    blocked; `longest_freeze_days` the longest run of consecutive days a wanted trade in one name was blocked; `mean_free_scale` the average k (1 without `cap_gross`), `infeasible_days` the days on which the stuck positions alone exceeded the target gross."""
    T, N = target.shape
    use_lots = capital is not None
    mtv = np.asarray(min_trade_value, dtype=float)                       # a number, or one per security
    mtv_any = bool(np.any(mtv > 0))
    out = np.zeros_like(target, dtype=float)
    prev = np.zeros(N)
    asked = blocked_turn = skipped_turn = gap = stuck = k_sum = 0.0
    n_blocked = n_skipped = infeasible = 0
    streak = np.zeros(N, dtype=int)
    longest = 0

    def decide(raw: np.ndarray, t: int):
        tgt = raw
        if use_lots:
            lt = lot[t] if isinstance(lot, np.ndarray) and lot.ndim == 2 else lot
            tgt = round_to_lots(raw, capital, price[t], lt)
            # A flat target needs no price to carry out (a delisted name is simply closed); a non-flat target with no price cannot be sized, so the
            # position stays.
            tgt = np.where(raw == 0, 0.0, np.where(np.isfinite(tgt), tgt, prev))
        d = tgt - prev
        stay = np.zeros(N, dtype=bool)
        if use_lots and mtv_any:
            stay |= (np.abs(d) * capital < mtv) & (d != 0)
        blk = np.zeros(N, dtype=bool)
        if can_buy is not None:
            blk |= (d > 0) & ~can_buy[t]
        if can_sell is not None:
            blk |= (d < 0) & ~can_sell[t]
        return tgt, d, stay, blk

    for t in range(T):
        raw0 = target[t]
        k = 1.0
        tgt, d, stay, blk = decide(raw0, t)
        if cap_gross:
            budget = float(np.abs(tgt).sum()) + 1e-12                       # what the day would hold with nothing in the way (after lot rounding)

            def gross_at(kk: float) -> float:
                g_tgt, _, g_stay, g_blk = decide(raw0 * kk, t)
                return float(np.abs(np.where(g_stay | g_blk, prev, g_tgt)).sum())

            if float(np.abs(np.where(stay | blk, prev, tgt)).sum()) > budget:
                if gross_at(0.0) > budget:
                    k = 0.0
                    infeasible += 1
                else:
                    lo, hi = 0.0, 1.0                                       # lo is always feasible, hi is not
                    while hi - lo > 1e-14:
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if gross_at(mid) <= budget else (lo, mid)
                    k = lo
                raw = raw0 * k
                tgt, d, stay, blk = decide(raw, t)
            else:
                raw = raw0
        else:
            raw = raw0
        k_sum += k
        if use_lots:
            gap += float(np.abs(tgt - raw).sum())
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
             "mean_abs_rounding_gap": gap / (T * N) if use_lots else 0.0, "mean_free_scale": k_sum / T if T else 1.0, "infeasible_days": infeasible}
    return out, stats
