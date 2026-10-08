"""Panel convention: the input contract of every test.

Design principles
    - The core is **plain pandas and numpy** and does not depend on any external data source (adapters do that).
    - Every matrix is (date x ticker) and the index and columns must be identical.
    - `eligible` is the **point-in-time universe**: True only for securities that could actually be traded that day.
        It is the only device against survivorship bias, so it is a required input.

Timing convention (the only one used in the whole project)
    A signal is **fixed** at close(t). Entry is at close(t+entry_lag) and exit at close(t+entry_lag+h).
    entry_lag defaults to 1: "see the signal, buy the next day". 0 assumes a fill at the same close (aggressive).
    Breaking this convention is caught by Panel.assert_no_lookahead().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED = ("close", "eligible")
OPTIONAL = ("open", "high", "low", "volume", "mkt_cap", "funding", "delist_after")


@dataclass
class Panel:
    """A bundle of (date x ticker) matrices.

    close    adjusted closing price: the only source of returns
    eligible point-in-time universe bool: listed, tradable and passing the liquidity condition that day
    The rest is optional. With volume and mkt_cap the liquidity and size controls are switched on.
    """

    close: pd.DataFrame
    eligible: pd.DataFrame
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    mkt_cap: pd.DataFrame | None = None
    chars: dict[str, pd.DataFrame] = field(default_factory=dict)
    market: str = "KR"
    entry_lag: int = 1
    # Trading days per year used for annualising: 252 for stocks, 365 for 24/7 markets (crypto). A wrong value makes CAGR, Sharpe and volatility wrong.
    periods_per_year: int = 252
    # Optional: futures funding (date x ticker, daily sum, positive means longs pay). Delisting flag (True on the last real bar). Notes for adapters.
    funding: pd.DataFrame | None = None
    delist_after: pd.DataFrame | None = None
    # Optional: (date x ticker) bool, True where a security can be sold short that day (borrowable, and no short-selling ban). None means every
    # security can. A date or security missing from the frame counts as **not** shortable: an unknown is not permission.
    shortable: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- construction and validation
    def __post_init__(self) -> None:
        self.close = self.close.sort_index()
        if self.close.index.has_duplicates:                    # before any reindex, which would fail with a less helpful pandas message
            raise ValueError("close.index has duplicate dates")
        self.eligible = self.eligible.reindex(
            index=self.close.index, columns=self.close.columns
        ).fillna(False).astype(bool)
        for k in OPTIONAL:
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.reindex(index=self.close.index, columns=self.close.columns))
        self.chars = {
            k: v.reindex(index=self.close.index, columns=self.close.columns)
            for k, v in self.chars.items()
        }
        if self.delist_after is not None:
            self.delist_after = self.delist_after.fillna(False).astype(bool)
        if self.shortable is not None:
            self.shortable = self.shortable.reindex(index=self.close.index, columns=self.close.columns).fillna(False).astype(bool)
        self.validate()

    def validate(self) -> None:
        """Raise ValueError if the panel breaks its invariants: the date index ascending and without duplicates, at least one eligible
        security, `entry_lag` not negative, and at least 10 eligible securities on half of the dates or more. Runs on construction."""
        if not self.close.index.is_monotonic_increasing:
            raise ValueError("close.index is not ascending")
        if self.close.index.has_duplicates:
            raise ValueError("close.index has duplicate dates")
        if self.eligible.values.sum() == 0:
            raise ValueError("eligible is all False")
        if self.entry_lag < 0:
            raise ValueError("entry_lag must be 0 or more")
        n = self.eligible.sum(axis=1)
        if (n[n > 0] < 10).mean() > 0.5:
            raise ValueError(
                "eligible has fewer than 10 securities on more than half of the dates: cross-sectional tests are impossible"
            )

    # ---------------------------------------------------------------- derived
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    def ret1(self) -> pd.DataFrame:
        """Daily return close(t)/close(t-1) - 1."""
        return self.close.pct_change(fill_method=None)

    def forward(self, h: int, delist_return: float | None = None) -> pd.DataFrame:
        """Return from close(t + lag) to close(t + lag + h) for a signal at t: a plain price ratio, NaN when either price is missing.

        A security flagged in `delist_after` is carried at its last price after its last bar (as cash), or at `last price x (1 + delist_return)`
        when `delist_return` is given, so a holding period that runs into a delisting keeps its result instead of dropping out. Without that
        the statistics of `screen` and `backtest_event` would leave out exactly the trades that ended in a delisting. A security that is only
        missing for some days (a halt) still gives NaN when an end price is missing. Compare `analytics.forward_returns`, which compounds daily
        returns and treats a missing price as a return of 0."""
        lag = self.entry_lag
        px = self.close
        if self.delist_after is not None and bool(np.asarray(self.delist_after.values).any()):
            da = self.delist_after.reindex(index=px.index, columns=px.columns).fillna(False).astype(bool).astype(int)
            gone = (da.cumsum() - da) > 0                                   # strictly after the security's last bar
            carried = px.ffill() * (1.0 + (0.0 if delist_return is None else float(delist_return)))
            px = px.where(~gone, carried)
        entry = px.shift(-lag)
        exit_ = px.shift(-(lag + h))
        return (exit_ / entry - 1.0).astype(np.float32)

    def fingerprint(self) -> str:
        """Short hash of the numbers this panel holds (every matrix that is present, including volume and market cap, which
        the impact model and the size controls read), used by the trial ledger to tell whether two runs saw the same data.
        The date index and the ticker names are not hashed: the same values in the same shape give the same fingerprint."""
        import hashlib
        h = hashlib.sha256()
        mats = [(n, getattr(self, n)) for n in ("close", "eligible", "open", "high", "low", "volume", "mkt_cap", "funding", "delist_after", "shortable")]
        mats += [(f"chars.{k}", self.chars[k]) for k in sorted(self.chars)]
        for name, v in mats:
            if v is not None:
                a = np.ascontiguousarray(v.to_numpy(dtype=np.float32))
                h.update(name.encode() + repr(a.shape).encode() + a.tobytes())
        h.update(f"{self.entry_lag}|{self.periods_per_year}|{self.market}".encode())
        return h.hexdigest()[:16]

    def adv(self, window: int = 20) -> pd.DataFrame:
        """Average traded value. None if there is no volume."""
        if self.volume is None:
            return None
        return (self.close * self.volume).rolling(window, min_periods=window).mean()

    # ---------------------------------------------------------------- safeguards
    def assert_no_lookahead(self, signal: pd.DataFrame, h: int = 5,
                            n_null: int = 8, seed: int = 0) -> dict:
        """Look-ahead detection: if the result improves when the signal is pushed one day **later**, it is looking at the future.

        A normal signal gets worse when delayed (the information decays).
        If it improves after the delay, the signal already contains future information.

        Warning: judging by a bare `lagged > base` makes a **random factor wrong 50% of the time**:
              both are near zero so the sign flips by chance. So the noise scale (sd) of the spread is measured with random shuffles
              and twice that sd is used as the tolerance.
        """
        fwd = self.forward(h)
        base = _spread(signal, fwd, self.eligible)
        lagged = _spread(signal.shift(1), fwd, self.eligible)
        rng = np.random.default_rng(seed)
        nulls = []
        for _ in range(n_null):
            sh = signal.values.copy()
            ev = self.eligible.values
            for i in range(sh.shape[0]):
                j = np.where(ev[i])[0]
                if len(j) > 1:
                    sh[i, j] = sh[i, rng.permutation(j)]
            v = _spread(pd.DataFrame(sh, index=signal.index, columns=signal.columns),
                        fwd, self.eligible)
            if np.isfinite(v):
                nulls.append(v)
        tol = 2 * float(np.std(nulls)) if len(nulls) >= 3 else 0.0
        ok = not (np.isfinite(base) and np.isfinite(lagged) and lagged > base + tol)
        return {"base_bp": base * 1e4, "lagged_bp": lagged * 1e4,
                "tol_bp": tol * 1e4, "pass": bool(ok)}

    def audit(self) -> dict:
        """Data integrity audit: run it once before any test.

        It catches common data defects:
            - silent truncation   (data vanishes wholesale after some date)
            - adjusted-price mismatch (more extreme gaps than daily moves: the adj_open defect type)
            - universe jumps      (the population triples over time, for example)
        """
        out = {}
        n = self.eligible.sum(axis=1)
        out["dates"] = len(self.dates)
        out["tickers"] = int(self.eligible.any(axis=0).sum())
        out["eligible_mean"] = float(n.mean())
        yr = n.groupby(self.dates.year).mean()
        out["eligible_by_year"] = {int(k): float(v) for k, v in yr.items()}
        out["universe_growth"] = float(yr.iloc[-1] / yr.iloc[0]) if len(yr) > 1 and yr.iloc[0] else np.nan

        r = self.ret1().where(self.eligible)
        out["ret_extreme_pct"] = float((r.abs() > 0.30).mean().mean() * 100)
        gap = None
        if self.open is not None:
            gap = (self.open / self.close.shift(1) - 1.0).where(self.eligible)
            out["gap_extreme_pct"] = float((gap.abs() > 0.30).mean().mean() * 100)
            ratio = out["gap_extreme_pct"] / max(out["ret_extreme_pct"], 1e-9)
            out["gap_vs_daily_ratio"] = float(ratio)
            out["adj_price_consistent"] = bool(ratio < 3.0)

        # silent truncation: security coverage drops sharply after some date
        cov = self.close.notna().sum(axis=1)
        if len(cov) > 60:
            tail = cov.iloc[-20:].mean(); body = cov.iloc[:-20].median()
            out["tail_coverage_ratio"] = float(tail / body) if body else np.nan
            out["truncation_suspected"] = bool(tail < body * 0.5)

        # suspected survivorship bias: in a real market a steady share of securities disappears every year. If almost no security's price stops before the panel's end,
        # the panel holds only 'securities alive today'. Measured: of 41 well-known delisted or acquired stocks looked up in yfinance,
        # none came back with a correct history (docs/survivorship.md).
        last = self.close.apply(lambda c: c.last_valid_index())
        ended = last.dropna() < (self.dates[-1] - pd.Timedelta(days=30))
        span_years = (self.dates[-1] - self.dates[0]).days / 365.25
        out["ended_before_end_share"] = float(ended.mean()) if len(ended) else np.nan
        out["survivorship_suspected"] = bool(len(ended) >= 30 and span_years >= 3 and ended.mean() < 0.01)
        return out


def _spread(signal: pd.DataFrame, fwd: pd.DataFrame, eligible: pd.DataFrame,
            q: float = 0.10) -> float:
    """Mean top-q minus bottom-q spread (internal helper for the look-ahead check)."""
    rk = signal.where(eligible).rank(axis=1, pct=True, na_option="keep")
    hi, lo = ((rk >= 1 - q) & eligible).values, ((rk <= q) & eligible).values
    c = fwd.values.astype(np.float64)
    out = []
    for i in range(c.shape[0]):
        a, b = hi[i] & np.isfinite(c[i]), lo[i] & np.isfinite(c[i])
        if a.sum() >= 3 and b.sum() >= 3:
            out.append(c[i][a].mean() - c[i][b].mean())
    return float(np.mean(out)) if out else np.nan


def build_pit_eligible(close: pd.DataFrame, *, listed: pd.DataFrame | None = None,
                       min_adv: float = 0.0, volume: pd.DataFrame | None = None,
                       exclude: pd.DataFrame | None = None) -> pd.DataFrame:
    """Helper that builds a point-in-time universe mask.

    listed   listing flag bool (falls back to whether close exists)
    min_adv  minimum average traded value (needs volume)
    exclude  mask of securities to exclude, such as administrative issues and trading halts (True = exclude)
    """
    ok = close.notna() & (close > 0)
    if listed is not None:
        ok &= listed.reindex_like(close).fillna(False).astype(bool)
    if min_adv > 0 and volume is not None:
        adv = (close * volume).rolling(20, min_periods=20).mean()
        ok &= adv >= min_adv
    if exclude is not None:
        ok &= ~exclude.reindex_like(close).fillna(False).astype(bool)
    return ok.fillna(False)
