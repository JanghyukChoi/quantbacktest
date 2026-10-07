"""Korean equities from the official KRX OpenAPI, point in time and survivorship-free.

How it removes survivorship bias: the API returns **every stock that was listed on a given date**, including the ones
delisted or merged later (on 2015-01-02, 125 of the 899 KOSPI names are gone by 2024). Fetching day by day therefore
builds the universe the way it really was.

How it handles corporate actions without a price-adjustment table: each row carries the change versus the *reference
price* (`CMPPREVDD_PRC`), and the reference price already reflects splits and rights issues. So
`close / (close - change) - 1` is the adjusted return. It matched the published return to within rounding for all 953
KOSPI stocks on 2024-01-02, and gives -2.08% (not -98%) for Samsung Electronics on its 50:1 split day (2018-05-04).
The adjusted close used by the panel is the cumulative product of those returns. Dividends are not included (price
return only).

You need a KRX OpenAPI key (free, the services have to be approved on openapi.krx.co.kr). Put it in the environment as
KRX_OPENAPI_KEY or pass `key=`; `load_key(env_path)` reads only that one variable from a .env file.

    fetch_days("2013-01-01", "2026-10-06", "~/.cache/quantbt/krx")      # resumable, about 2 calls per trading day
    panel = build_krx_panel("~/.cache/quantbt/krx", start="2014-01-01")
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

BASE = "https://data-dbg.krx.co.kr/svc/apis"
ENDPOINTS = {"KOSPI": "sto/stk_bydd_trd", "KOSDAQ": "sto/ksq_bydd_trd"}
RENAME = {"ISU_CD": "code", "ISU_NM": "name", "MKT_NM": "market", "TDD_CLSPRC": "close", "CMPPREVDD_PRC": "change",
          "TDD_OPNPRC": "open", "TDD_HGPRC": "high", "TDD_LWPRC": "low", "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value",
          "MKTCAP": "mktcap", "LIST_SHRS": "shares"}
# Not tradable common stock for a cross-sectional study: SPACs, REITs and funds listed as stocks.
DEFAULT_EXCLUDE = re.compile(r"스팩|리츠|\d+호|인프라|맥쿼리")


class QuotaExceeded(RuntimeError):
    """The API refused further calls (daily limit). Run again later; the cache keeps what was fetched."""


def load_key(env_path: str | Path | None = None) -> str:
    """KRX_OPENAPI_KEY from the environment, else from a .env file. Only that one variable is read."""
    if os.environ.get("KRX_OPENAPI_KEY"):
        return os.environ["KRX_OPENAPI_KEY"]
    if env_path:
        for line in Path(env_path).expanduser().read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "KRX_OPENAPI_KEY":
                return v.strip().strip('"').strip("'")
    raise KeyError("KRX_OPENAPI_KEY is not set")


def _call(path: str, date: str, key: str, retries: int = 4) -> list[dict]:
    req = urllib.request.Request(f"{BASE}/{path}?basDd={date}", headers={"AUTH_KEY": key, "User-Agent": "pitbacktest-research/0.2"})
    delay = 1.0
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read()).get("OutBlock_1", [])
        except urllib.error.HTTPError as e:
            body = e.read()[:200].decode("utf-8", "replace")
            if e.code in (429, 403) or "limit" in body.lower() or "초과" in body:
                raise QuotaExceeded(f"HTTP {e.code}: {body}") from None
            if e.code >= 500 and i < retries:
                time.sleep(delay); delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):   # ValueError: an empty or non-JSON body
            if i < retries:
                time.sleep(delay); delay *= 2
                continue
            raise
    return []


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).rename(columns=RENAME)[list(RENAME.values())]
    for c in ("close", "change", "open", "high", "low", "volume", "value", "mktcap", "shares"):
        df[c] = pd.to_numeric(df[c].replace({"-": np.nan, "": np.nan}), errors="coerce")
    return df


def fetch_days(start, end, store_dir, *, key: str | None = None, markets=("KOSPI", "KOSDAQ"), sleep: float = 0.1,
               max_calls: int | None = None, caller=None, progress: bool = True) -> dict:
    """Download one file per weekday into `store_dir`. Resumable: finished days (and holidays) are skipped.
    `caller(path, date)` can be injected for tests. Stops cleanly if the API reports a quota limit."""
    store = Path(store_dir).expanduser()
    store.mkdir(parents=True, exist_ok=True)
    key = key or (None if caller else load_key())
    call = caller or (lambda p, d: _call(p, d, key))
    done = {"days": 0, "holidays": 0, "skipped": 0, "calls": 0, "stopped": None}
    for ts in pd.bdate_range(start, end):
        d = ts.strftime("%Y%m%d")
        if (store / f"{d}.pkl").exists() or (store / f"{d}.hol").exists():
            done["skipped"] += 1
            continue
        if max_calls is not None and done["calls"] + len(markets) > max_calls:
            done["stopped"] = "max_calls"
            break
        frames = []
        try:
            for m in markets:
                rows = call(ENDPOINTS[m], d)
                done["calls"] += 1
                if rows:
                    frames.append(_frame(rows))
                time.sleep(sleep)
        except QuotaExceeded as e:
            done["stopped"] = f"quota: {e}"
            break
        if frames and len(frames) < len(markets) and caller is None:
            done.setdefault("partial", []).append(d)                   # one market answered empty: do not save, retry next run
            continue
        if frames:
            pd.concat(frames).to_pickle(store / f"{d}.pkl")
            done["days"] += 1
        else:
            (store / f"{d}.hol").write_text("")                      # holiday: both markets empty
            done["holidays"] += 1
        if progress and (done["days"] + done["holidays"]) % 100 == 0:
            print(f"  {d}  {done}", flush=True)
    return done


def build_krx_panel(store_dir, *, start=None, end=None, min_age_days: int = 60, adv_window: int = 30,
                    min_value_krw: float = 1e9, common_only: bool = True, exclude=DEFAULT_EXCLUDE,
                    reuse_gap_days: int = 120, entry_lag: int = 1) -> Panel:
    """Point-in-time Panel of Korean stocks from the cached daily files.

    eligible on day t uses data up to t only: at least `min_age_days` bars of history, trailing median traded value of
    at least `min_value_krw`, positive volume that day, common shares only (code ends in 0) and no SPAC or REIT names.
    A code that disappears for more than `reuse_gap_days` and comes back is treated as a different security (suffix #2)."""
    store = Path(store_dir).expanduser()
    files = sorted(store.glob("*.pkl"))
    if start:
        files = [f for f in files if f.stem >= pd.Timestamp(start).strftime("%Y%m%d")]
    if end:
        files = [f for f in files if f.stem <= pd.Timestamp(end).strftime("%Y%m%d")]
    if not files:
        raise ValueError("no cached days: run fetch_days() first")
    parts = []
    for f in files:
        d = pd.read_pickle(f)
        d["date"] = pd.Timestamp(f.stem)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True).sort_values(["code", "date"])
    gap = df.groupby("code")["date"].diff().dt.days
    life = (gap > reuse_gap_days).groupby(df["code"]).cumsum()
    df["sid"] = np.where(life == 0, df["code"], df["code"] + "#" + (life + 1).astype(str))
    names = df.drop_duplicates("sid", keep="last").set_index("sid")["name"].to_dict()

    def mat(c: str) -> pd.DataFrame:
        return df.pivot(index="date", columns="sid", values=c).sort_index()

    close, chg = mat("close"), mat("change")
    ref = close - chg
    ret = (close / ref - 1.0).where(ref > 0)
    first = close.notna() & ~close.notna().cumsum().shift(1, fill_value=0).astype(bool)   # a security's first bar
    ret = ret.mask(first)                                                    # the listing-day move is not tradable
    adj = 100.0 * np.exp(np.log1p(ret.fillna(0.0)).cumsum().where(close.notna()))
    k = adj / close                                                          # adjust open/high/low to the same level
    value, vol_raw = mat("value"), mat("volume")
    age = close.notna().cumsum()
    adv = value.rolling(adv_window, min_periods=adv_window).median()
    ok = close.notna() & (vol_raw > 0) & (age >= min_age_days) & (adv >= min_value_krw)
    if common_only:
        ok = ok & pd.Series({c: c.split("#")[0].endswith("0") for c in ok.columns}).reindex(ok.columns).values
    if exclude is not None:
        bad = [s for s in ok.columns if exclude.search(names.get(s, ""))]
        ok[bad] = False
    last = close.apply(lambda c: c.last_valid_index())
    da = pd.DataFrame(False, index=close.index, columns=close.columns)
    for s in close.columns:
        if last[s] is not None and last[s] < close.index[-1] - pd.Timedelta(days=5):
            da.loc[last[s], s] = True
    p = Panel(close=adj, eligible=ok, open=mat("open") * k, high=mat("high") * k, low=mat("low") * k,
              volume=value / adj, mkt_cap=mat("mktcap"), market="KR", periods_per_year=245, entry_lag=entry_lag,
              delist_after=da)
    n = ok.sum(axis=1)
    p.meta = {"securities": int(close.shape[1]), "delisted_in_panel": int(da.values.any(axis=0).sum()),
              "eligible_median": float(n[n > 0].median()), "first_day": str(close.index[0].date()),
              "last_day": str(close.index[-1].date()), "names": names}
    return p
