"""A free, keyless list of US stock tickers with their listing windows, delisted ones included.

Source: Tiingo's published `supported_tickers.zip` (no API key needed). It is a **security master, not a price
source**: it tells you which tickers existed in which period. It is incomplete before about 2013 and misses some recent
failures (for example SIVB and FRC were absent when this was written), so coverage numbers built on it are lower
bounds on how many names a survivors-only panel is missing.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
US_EXCHANGES = {"NYSE", "NASDAQ", "NYSE MKT", "NYSE ARCA", "AMEX", "NYSE American"}


def load_us_master(cache_dir: str | Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """DataFrame [ticker, exchange, start, end, alive_today] for US-listed common stocks."""
    d = Path(cache_dir or Path.home() / ".cache" / "quantbt").expanduser()
    d.mkdir(parents=True, exist_ok=True)
    f = d / "tiingo_supported_tickers.zip"
    if refresh or not f.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "pitbacktest-research/0.2 (public data only)"})
        f.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    with zipfile.ZipFile(f) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["assetType"] == "Stock") & df["exchange"].isin(US_EXCHANGES)].copy()
    df["start"] = pd.to_datetime(df["startDate"], errors="coerce")
    df["end"] = pd.to_datetime(df["endDate"], errors="coerce")
    df = df.dropna(subset=["start", "end"])
    df["alive_today"] = df["end"] >= df["end"].max() - pd.Timedelta(days=7)
    return df[["ticker", "exchange", "start", "end", "alive_today"]].reset_index(drop=True)


def universe_coverage(panel_tickers, master: pd.DataFrame, years=range(2010, 2025)) -> pd.DataFrame:
    """For each year: how many US stocks were listed, how many of them are in your panel, and (the telling number)
    how many of the names that stopped trading that year are in your panel.

    A ticker only counts as present if its listing window in the master overlaps the year, so a reused ticker is not
    credited to the company that used to have it (the master has one row per ticker and window)."""
    have = set(panel_tickers)
    rows = []
    for y in years:
        a, b = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        alive = master[(master["start"] <= b) & (master["end"] >= a)]
        died = alive[(alive["end"] <= b) & ~alive["alive_today"]]
        rows.append({"year": y, "listed": len(alive), "in_panel": int(alive["ticker"].isin(have).sum()),
                     "stopped_trading": len(died), "stopped_in_panel": int(died["ticker"].isin(have).sum())})
    out = pd.DataFrame(rows).set_index("year")
    out["share_listed_in_panel"] = out["in_panel"] / out["listed"].replace(0, float("nan"))
    out["share_stopped_in_panel"] = out["stopped_in_panel"] / out["stopped_trading"].replace(0, float("nan"))
    return out
