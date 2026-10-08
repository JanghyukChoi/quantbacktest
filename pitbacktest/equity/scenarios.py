"""Survivorship scenarios: put delistings back into a survivors-only panel and see how the result moves.

This is a **what-if**, not a correction. The actual delisted companies are unknown here, so we draw them at random
(optionally tilted toward small, illiquid or volatile names, which is where delistings concentrate in the literature) and
give them an explicit delisting return. Run it over a grid of annual rates and delisting returns and report the range, not a
single "corrected" number.

Order-of-magnitude guide from the delisting literature (check it for your period and market): a few percent of listed
names disappear each year, roughly half of them for performance reasons; the average return on a performance-related
delisting is about -30% on NYSE/AMEX (Shumway 1997) and about -55% on Nasdaq (Shumway and Warther 1999).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from ..core.panel import Panel


def _weights(panel: Panel, year_start: pd.Timestamp, names: pd.Index, hazard: str) -> np.ndarray:
    if hazard == "uniform":
        return np.ones(len(names))
    before = panel.dates < year_start
    if not before.any():
        return np.ones(len(names))                                 # first year: nothing earlier to tilt on, so uniform
    if hazard == "volatile":
        v = panel.close.pct_change(fill_method=None).loc[before].tail(60).std().reindex(names)
    elif hazard == "illiquid":
        adv = panel.adv(30)
        v = -(adv.loc[before].tail(1).iloc[0].reindex(names)) if adv is not None else pd.Series(0.0, index=names)
    else:
        raise ValueError(hazard)
    r = v.rank(pct=True).fillna(0.5).to_numpy()
    return 0.2 + r ** 2                                            # tilted, but every name keeps a chance


def inject_delistings(panel: Panel, *, annual_rate: float = 0.03, hazard: str = "uniform", seed: int = 0,
                      min_age_days: int = 60) -> Panel:
    """Return a copy of `panel` in which a fraction `annual_rate` of the names alive at the start of each calendar year stop
    trading at a random date later that year. The last real bar is flagged in `delist_after` and prices, volume and
    eligibility are removed afterwards."""
    if hazard not in ("uniform", "volatile", "illiquid"):
        raise ValueError(f"hazard must be 'uniform', 'volatile' or 'illiquid', not {hazard!r}")
    if hazard == "illiquid" and panel.volume is None:
        raise ValueError("hazard='illiquid' needs panel.volume")
    if not (0 <= annual_rate <= 1):
        raise ValueError(f"annual_rate must be between 0 and 1, got {annual_rate!r}")
    rng = np.random.default_rng(seed)
    close = panel.close.copy()
    elig = panel.eligible.copy()
    vol = None if panel.volume is None else panel.volume.copy()
    da = (pd.DataFrame(False, index=panel.dates, columns=panel.tickers) if panel.delist_after is None
          else panel.delist_after.copy())
    for y in sorted(set(panel.dates.year)):
        idx = panel.dates[panel.dates.year == y]
        if len(idx) < 100:
            continue                                               # skip partial years
        alive = close.loc[idx[0]].dropna().index
        k = int(round(annual_rate * len(alive)))
        if k == 0:
            continue
        w = _weights(panel, idx[0], alive, hazard)
        pick = rng.choice(len(alive), size=min(k, len(alive)), replace=False, p=w / w.sum())
        for j in pick:
            name = alive[j]
            when = idx[int(rng.integers(min_age_days // 3, len(idx) - 1))]
            close.loc[when + pd.Timedelta(days=1):, name] = np.nan
            elig.loc[when + pd.Timedelta(days=1):, name] = False
            if vol is not None:
                vol.loc[when + pd.Timedelta(days=1):, name] = np.nan
            da.loc[when, name] = True
    return replace(panel, close=close, eligible=elig, volume=vol, delist_after=da)


def survivorship_scenarios(panel: Panel, evaluate: Callable[[Panel, float], float], *,
                           rates=(0.01, 0.03, 0.05), delist_returns=(-0.30, -0.55), hazards=("uniform", "volatile"),
                           n: int = 10, seed: int = 0) -> pd.DataFrame:
    """Run `evaluate(panel, delist_return)` (it should return one number, for example the net Sharpe) on the original panel
    and on `n` randomly delisting copies for every combination. Returns the baseline and the spread of outcomes."""
    base = float(evaluate(panel, 0.0))
    rows = []
    for hz in hazards:
        for r in rates:
            for d in delist_returns:
                vals = [float(evaluate(inject_delistings(panel, annual_rate=r, hazard=hz, seed=seed + i), d)) for i in range(n)]
                rows.append({"hazard": hz, "annual_rate": r, "delist_return": d, "baseline": base,
                             "mean": float(np.mean(vals)), "p5": float(np.percentile(vals, 5)),
                             "p95": float(np.percentile(vals, 95)), "mean_change": float(np.mean(vals) - base)})
    return pd.DataFrame(rows)


def survivors_only(panel: Panel, end_gap_days: int = 5) -> Panel:
    """The usual shortcut, made explicit: keep only the securities that still have a price on the last day.
    Compare a result on `panel` with the same result on `survivors_only(panel)` to measure survivorship bias directly."""
    last = panel.close.apply(lambda c: c.last_valid_index())
    keep = [c for c in panel.close.columns if last[c] is not None and last[c] >= panel.dates[-1] - pd.Timedelta(days=end_gap_days)]
    sub = lambda d: None if d is None else d[keep]
    return replace(panel, close=panel.close[keep], eligible=panel.eligible[keep], open=sub(panel.open), high=sub(panel.high),
                   low=sub(panel.low), volume=sub(panel.volume), mkt_cap=sub(panel.mkt_cap),
                   chars={k: v[keep] for k, v in panel.chars.items()}, funding=sub(panel.funding),
                   delist_after=sub(panel.delist_after))
