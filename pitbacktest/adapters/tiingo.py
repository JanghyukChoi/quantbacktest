"""Tiingo end-of-day prices (free account) as a point-in-time panel that keeps delisted securities.

What it is for
  Tiingo's end-of-day API returns the full history of many US stocks that were later acquired or delisted, which yfinance
  does not. The free tier is small, so this adapter is built for a **random sample** of tickers, not the whole market:
  `draw_order` gives a seeded random order and `fetch_symbols` downloads in that order, resumably, and stops cleanly when
  the account runs out of requests. Any prefix of the order is a random sample.

What it does not fix
  A probe (`docs/tiingo_probe.py`) found the history of 23 of 29 takeover or rename cases but of **none** of 12 bankruptcies
  and rescue sales. A panel from this adapter is less survivor-biased than one from yfinance, not unbiased. Treat any
  survivors-versus-this comparison as a lower bound.

Choices worth knowing
  - Returns use `adjClose` (split and dividend adjusted, so total return). Eligibility uses the raw `close` and `volume`.
  - A ticker with several listing windows in the ticker list (reuse) becomes several securities, each cut to its own window.
  - A security whose last bar is more than `end_gap_days` before the last calendar date is flagged in `delist_after`.
  - A ticker for which the API has nothing is recorded (`.none`) and counted, never silently replaced.
  - Only the `TIINGO_API_KEY` variable is read.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.panel import Panel

BASE = "https://api.tiingo.com/tiingo/daily"
KEEP = ("date", "close", "volume", "adjClose")
_NOT_PLAIN = re.compile(r"[-.]")


class QuotaExceeded(RuntimeError):
    """The API refused further requests (rate or monthly limit). Run again later; the cache keeps what was fetched."""


def load_key(env_path: str | Path | None = None) -> str:
    """TIINGO_API_KEY from the environment, else from a .env file. Only that one variable is read."""
    if os.environ.get("TIINGO_API_KEY"):
        return os.environ["TIINGO_API_KEY"]
    if env_path:
        for line in Path(env_path).expanduser().read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "TIINGO_API_KEY":
                return v.strip().strip('"').strip("'")
    raise KeyError("TIINGO_API_KEY is not set")


# ---------------------------------------------------------------------------------------------------- the sample
def study_frame(master: pd.DataFrame, since: str = "2013-01-01") -> pd.DataFrame:
    """Rows of the ticker list that can enter a study starting at `since`: plain tickers (no '-' or '.', and no
    five-letter ticker ending in W, U, R or P: warrants, units, rights, preferreds) whose window ends on or after `since`."""
    t = master["ticker"].astype(str)
    plain = ~t.str.contains(_NOT_PLAIN) & ~((t.str.len() == 5) & t.str[-1].isin(list("WURP")))
    return master[plain & (master["end"] >= pd.Timestamp(since))].reset_index(drop=True)


def draw_order(tickers, seed: int = 0) -> list[str]:
    """A seeded random permutation of the unique tickers (sorted first, so it does not depend on the input order).
    Download in this order and every prefix is a simple random sample."""
    u = sorted(set(map(str, tickers)))
    rng = np.random.default_rng(seed)
    return [u[i] for i in rng.permutation(len(u))]


# ---------------------------------------------------------------------------------------------------- downloading
def _call(ticker: str, start: str, key: str, retries: int = 3):
    """Bars for one ticker, or None if the API does not know it. Raises QuotaExceeded on HTTP 429."""
    url = f"{BASE}/{ticker.lower()}/prices?startDate={start}&format=json"
    req = urllib.request.Request(url, headers={"Authorization": f"Token {key}", "Content-Type": "application/json",
                                               "User-Agent": "pitbacktest-research/0.2"})
    delay = 1.0
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            # The monthly limit on distinct symbols (500 on a free account) comes back as HTTP 200 with a message, not as a 429.
            if isinstance(data, dict) and re.search(r"run over|allocation|look ?up|upgrade", str(data.get("detail", "")), re.I):
                raise QuotaExceeded(f"HTTP 200: {data.get('detail')}")
            return data
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf-8", "replace")
            if e.code == 429:
                raise QuotaExceeded(f"HTTP 429: {body}") from None
            if e.code == 404:
                return None
            if e.code in (401, 403):
                raise PermissionError(f"HTTP {e.code}: {body}") from None
            if e.code >= 500 and i < retries:
                time.sleep(delay); delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
            if i < retries:
                time.sleep(delay); delay *= 2
                continue
            raise
    return None


def _check(ticker: str, rows) -> list[dict]:
    if not isinstance(rows, list):
        raise RuntimeError(f"Tiingo {ticker}: unexpected response type {type(rows).__name__}")
    out = []
    for r in rows:
        if not all(k in r for k in KEEP):
            raise RuntimeError(f"Tiingo {ticker}: bar without {[k for k in KEEP if k not in r]}; the format changed")
        out.append({"date": str(r["date"])[:10], "close": r["close"], "volume": r["volume"], "adjClose": r["adjClose"]})
    return out


def _safe(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", ticker.upper())


def fetch_symbols(tickers, store_dir, *, start: str = "2012-01-01", key: str | None = None, sleep: float = 0.3,
                  max_new: int | None = None, caller=None, progress: bool = True) -> dict:
    """Download one file per ticker into `store_dir` (`T.json` with the needed fields, or `T.none` if the API has nothing).
    Resumable: tickers already stored are skipped. `max_new` limits requests in this run. Stops cleanly on a quota error.
    `caller(ticker, start)` can be injected for tests (return None for an unknown ticker)."""
    store = Path(store_dir).expanduser()
    store.mkdir(parents=True, exist_ok=True)
    key = key or (None if caller else load_key())
    call = caller or (lambda t, s: _call(t, s, key))
    done = {"fetched": 0, "none": 0, "skipped": 0, "calls": 0, "stopped": None}
    for t in tickers:
        f = store / f"{_safe(t)}.json"
        n = store / f"{_safe(t)}.none"
        if f.exists() or n.exists():
            done["skipped"] += 1
            continue
        if max_new is not None and done["calls"] >= max_new:
            done["stopped"] = "max_new"
            break
        try:
            rows = call(t, start)
        except QuotaExceeded as e:
            done["stopped"] = f"quota: {e}"
            break
        done["calls"] += 1
        if rows is None or (isinstance(rows, list) and len(rows) == 0):
            n.write_text("")
            done["none"] += 1
        else:
            f.write_text(json.dumps(_check(t, rows)))        # validated first: a malformed answer is never saved
            done["fetched"] += 1
        if progress and done["calls"] % 25 == 0:
            print(f"  {done['calls']} requests, {done['fetched']} with data, {done['none']} none", flush=True)
        time.sleep(sleep)
    return done


# ---------------------------------------------------------------------------------------------------- the panel
def build_tiingo_panel(store_dir, master: pd.DataFrame, tickers, *, start: str = "2012-01-01", end: str | None = None,
                       min_age_bars: int = 60, adv_window: int = 30, min_dollar_volume: float = 2e6, min_price: float = 1.0,
                       min_names_per_date: int = 100, end_gap_days: int = 5, entry_lag: int = 1) -> Panel:
    """Point-in-time panel from the stored files. Eligibility on day t uses only data up to t.

    Each listing window of a ticker in `master` (columns ticker, start, end) is one security, cut to that window.
    The trading calendar is the set of dates on which at least `min_names_per_date` securities have a bar.
    `start` is the first date kept: pass the start of the warm-up year and cut the study period afterwards. The age of a
    security counts bars from the start of the stored data (`fetch_symbols(start=...)`), so a warm-up year of 60+ bars
    makes that irrelevant for the study period."""
    store = Path(store_dir).expanduser()
    adj, raw, vol, meta = {}, {}, {}, {"tickers_with_data": 0, "tickers_none": 0, "tickers_missing": 0, "windows_cut": 0}
    for t in map(str, tickers):
        f, n = store / f"{_safe(t)}.json", store / f"{_safe(t)}.none"
        if n.exists():
            meta["tickers_none"] += 1
            continue
        if not f.exists():
            meta["tickers_missing"] += 1
            continue
        bars = pd.DataFrame(json.loads(f.read_text()))
        if bars.empty:
            meta["tickers_none"] += 1
            continue
        meta["tickers_with_data"] += 1
        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.drop_duplicates("date", keep="last").set_index("date").sort_index()
        wins = master[master["ticker"].astype(str) == t].sort_values("start")
        if wins.empty:
            wins = pd.DataFrame({"start": [bars.index[0]], "end": [bars.index[-1]]})
        multi = len(wins) > 1
        meta["windows_cut"] += int(multi)
        for _, w in wins.iterrows():
            b = bars.loc[(bars.index >= w["start"]) & (bars.index <= w["end"])]
            b = b[(b["adjClose"] > 0) & (b["close"] > 0)]
            if b.empty:
                continue
            sid = f"{t}@{w['start']:%Y%m%d}" if multi else t
            adj[sid], raw[sid], vol[sid] = b["adjClose"], b["close"], b["volume"]
    if not adj:
        raise ValueError("no usable securities in the store")
    A, R, V = pd.DataFrame(adj), pd.DataFrame(raw), pd.DataFrame(vol)
    cal = A.index[(A.notna().sum(axis=1) >= min_names_per_date) & (A.index >= pd.Timestamp(start))]
    if end:
        cal = cal[cal <= pd.Timestamp(end)]
    A, R, V = A.reindex(cal), R.reindex(cal), V.reindex(cal)
    dvol = R * V
    # eligibility, all trailing
    age = A.notna().cumsum()
    med = dvol.rolling(adv_window, min_periods=adv_window).median()
    elig = (age >= min_age_bars) & (med >= min_dollar_volume) & (R >= min_price) & (V > 0) & A.notna()
    last = A.apply(lambda c: c.last_valid_index())
    last_date = cal[-1]
    da = pd.DataFrame(False, index=cal, columns=A.columns)
    for c in A.columns:
        if last[c] is not None and last[c] < last_date - pd.Timedelta(days=end_gap_days):
            da.loc[last[c], c] = True
    r = A.pct_change(fill_method=None)
    meta["suspect_returns_gt_10x"] = int((r.abs() > 10).sum().sum())
    meta["securities"] = int(A.shape[1])
    meta["raw_close"] = R.astype("float32")                      # the real price level (dollars), for whole-share sizes; `close` is adjusted
    return Panel(close=A, eligible=elig, volume=dvol / A, market="US", entry_lag=entry_lag, periods_per_year=252,
                 delist_after=da, meta=meta)
