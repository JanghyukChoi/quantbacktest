"""Binance public data archive (data.binance.vision) for USDT-margined perpetuals.

Why the archive and not the live API: the archive keeps **delisted** contracts (LUNA, FTT and hundreds of
others), so a universe built from it is not limited to today's survivors. The live API is used only to fill the
last partial month for contracts that still trade.

Daily bars are UTC days. Funding is settled every 8 hours (older contracts) or per `funding_interval_hours`;
`daily_funding` sums every settlement that falls inside the UTC day.

Everything is cached under `cache_dir` (default ~/.cache/quantbt/binance_um) so a download happens once.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL_URL = "https://data.binance.vision"
API = "https://fapi.binance.com"
KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
           "taker_buy_base", "taker_buy_quote", "ignore"]
UA = {"User-Agent": "pitbacktest-research/0.2 (public data only)"}


def _get(url: str, retries: int = 4, timeout: float = 30) -> bytes | None:
    """GET with backoff. Returns None on 404 (a missing archive file is normal)."""
    url = urllib.parse.quote(url, safe=":/?&=%")                  # non-ASCII symbols (e.g. Chinese tickers) need encoding
    delay = 1.0
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (418, 429, 500, 502, 503, 504) and i < retries:
                time.sleep(delay * (3 if e.code in (418, 429) else 1))
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return None


def _list(prefix: str) -> list[str]:
    """All keys under a prefix (S3 listing, paginated)."""
    keys, marker = [], ""
    while True:
        q = {"prefix": prefix, "max-keys": "1000"}
        if marker:
            q["marker"] = marker
        x = (_get(f"{LIST_URL}?{urllib.parse.urlencode(q)}") or b"").decode("utf-8")
        got = re.findall(r"<Key>([^<]+)</Key>", x)
        keys += got
        if "<IsTruncated>true</IsTruncated>" in x and got:
            marker = got[-1]
        else:
            return keys


def _list_dirs(prefix: str) -> list[str]:
    out, marker = [], ""
    while True:
        q = {"prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        if marker:
            q["marker"] = marker
        x = (_get(f"{LIST_URL}?{urllib.parse.urlencode(q)}") or b"").decode("utf-8")
        got = re.findall(r"<Prefix>([^<]+)</Prefix>", x)
        got = [g for g in got if g != prefix]
        out += got
        nm = re.search(r"<NextMarker>([^<]+)</NextMarker>", x)
        if "<IsTruncated>true</IsTruncated>" in x and nm:
            marker = nm.group(1)
        else:
            return out


def _read_klines_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None, names=KL_COLS, dtype=str)
    t = pd.to_numeric(raw["open_time"], errors="coerce")        # some files carry a header row
    raw = raw[t.notna()].copy()
    raw["open_time"] = t[t.notna()].astype("int64")
    return raw


def _frame_from_klines(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    ts = raw["open_time"].astype("int64")
    unit = "us" if ts.iloc[0] > 10**14 else "ms"                # newer archives use microseconds
    idx = pd.to_datetime(ts, unit=unit, utc=True).dt.tz_localize(None).dt.normalize()
    df = raw[["open", "high", "low", "close", "volume", "quote_volume", "trades"]].apply(pd.to_numeric, errors="coerce")
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


class ArchiveStore:
    """Local cache plus downloader for one market (USDT-M perpetuals)."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.dir = Path(cache_dir or Path.home() / ".cache" / "quantbt" / "binance_um").expanduser()
        (self.dir / "daily").mkdir(parents=True, exist_ok=True)
        (self.dir / "funding").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ symbols
    def symbols(self, refresh: bool = False, quote: str = "USDT") -> list[str]:
        """Every USDT perpetual that ever existed in the archive (delisted ones included).
        Delivery contracts such as BTCUSDT_250627 are excluded."""
        f = self.dir / "symbols.json"
        if f.exists() and not refresh:
            return json.loads(f.read_text())
        pre = "data/futures/um/monthly/klines/"
        names = [d[len(pre):].strip("/") for d in _list_dirs(pre)]
        syms = sorted(n for n in names if n.endswith(quote) and "_" not in n)
        f.write_text(json.dumps(syms))
        return syms

    def trading_now(self) -> set[str]:
        """Symbols the live exchange reports as TRADING (used to know which ones need the live top-up)."""
        raw = _get(f"{API}/fapi/v1/exchangeInfo")
        info = json.loads(raw)
        return {s["symbol"] for s in info["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"}

    # ------------------------------------------------------------ one symbol
    def _months(self, sym: str, kind: str) -> list[str]:
        keys = _list(f"data/futures/um/monthly/{kind}/{sym}/" + ("1d/" if kind == "klines" else ""))
        return sorted(k for k in keys if k.endswith(".zip"))

    def fetch_daily(self, sym: str, live: bool) -> pd.DataFrame | None:
        f = self.dir / "daily" / f"{sym}.pkl"
        if f.exists():
            return pd.read_pickle(f)
        parts = []
        for k in self._months(sym, "klines"):
            blob = _get(f"{DL_URL}/{k}")
            if blob:
                parts.append(_read_klines_zip(blob))
        raw = pd.concat(parts) if parts else pd.DataFrame(columns=KL_COLS)
        df = _frame_from_klines(raw)
        if live:                                                   # top-up from the live API
            start = (df.index[-1] + pd.Timedelta(days=1)) if len(df) else pd.Timestamp("2019-09-01")
            ms = int(start.tz_localize("UTC").timestamp() * 1000)
            blob = _get(f"{API}/fapi/v1/klines?symbol={sym}&interval=1d&startTime={ms}&limit=1500")
            if blob:
                rows = json.loads(blob)
                if rows:
                    live_df = _frame_from_klines(pd.DataFrame(rows, columns=KL_COLS).assign(
                        open_time=lambda d: d["open_time"].astype("int64")))
                    df = pd.concat([df, live_df]).pipe(lambda d: d[~d.index.duplicated(keep="last")]).sort_index()
        if len(df):
            today = pd.Timestamp.utcnow().tz_localize(None).normalize()
            df = df[df.index < today]                              # never keep an unfinished bar
        df.to_pickle(f)
        return df

    def fetch_funding(self, sym: str, live: bool) -> pd.Series | None:
        f = self.dir / "funding" / f"{sym}.pkl"
        if f.exists():
            return pd.read_pickle(f)
        frames = []
        for k in self._months(sym, "fundingRate"):
            blob = _get(f"{DL_URL}/{k}")
            if not blob:
                continue
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                d = pd.read_csv(z.open(z.namelist()[0]), dtype=str)
            if d.shape[1] >= 3:
                d.columns = ["calc_time", "interval", "rate"][: d.shape[1]] + list(d.columns[3:])
                frames.append(d[["calc_time", "rate"]])
        out = pd.concat(frames) if frames else pd.DataFrame(columns=["calc_time", "rate"])
        out["calc_time"] = pd.to_numeric(out["calc_time"], errors="coerce")
        out["rate"] = pd.to_numeric(out["rate"], errors="coerce")
        out = out.dropna()
        if live:
            last = out["calc_time"].max() if len(out) else 1567296000000
            blob = _get(f"{API}/fapi/v1/fundingRate?symbol={sym}&startTime={int(last) + 1}&limit=1000")
            if blob:
                rows = json.loads(blob)
                if rows:
                    add = pd.DataFrame({"calc_time": [r["fundingTime"] for r in rows], "rate": [float(r["fundingRate"]) for r in rows]})
                    out = pd.concat([out, add])
        if out.empty:
            s = pd.Series(dtype=float)
        else:
            ts = out["calc_time"].astype("int64")
            unit = "us" if ts.iloc[0] > 10**14 else "ms"
            day = pd.to_datetime(ts, unit=unit, utc=True).dt.tz_localize(None).dt.normalize()
            s = out.set_index(day)["rate"].groupby(level=0).sum().sort_index()   # sum of settlements in the UTC day
        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        s = s[s.index < today]
        s.to_pickle(f)
        return s


def fetch_all(store: ArchiveStore | None = None, symbols: list[str] | None = None, workers: int = 12,
              progress: bool = True) -> dict:
    """Download (or load from cache) daily bars and funding for every symbol. Returns counts."""
    store = store or ArchiveStore()
    syms = symbols or store.symbols()
    live = store.trading_now()
    done = {"ok": 0, "empty": 0, "error": 0}
    errs: list[str] = []

    def one(s: str):
        d = store.fetch_daily(s, s in live)
        f = store.fetch_funding(s, s in live)
        return s, (0 if d is None else len(d)), (0 if f is None else len(f))

    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, s): s for s in syms}
        for n, fu in enumerate(cf.as_completed(futs), 1):
            try:
                _, nd, nf = fu.result()
                done["ok" if nd else "empty"] += 1
            except Exception as e:                                  # one bad symbol must not stop the run
                done["error"] += 1
                errs.append(f"{futs[fu]}: {type(e).__name__}: {e}")
            if progress and n % 25 == 0:
                print(f"  {n}/{len(syms)}  {done}", flush=True)
    done["errors"] = errs[:20]
    return done
