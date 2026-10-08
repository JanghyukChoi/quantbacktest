"""Measure the quoted spread of Binance USDT-M perpetuals from the public daily top-of-book files and compare it with what the library assumes.

    python docs/crypto_spread_probe.py SYMBOL YYYY-MM-DD [...]      # e.g. DOGEUSDT 2023-12-06

Downloads `bookTicker/SYMBOL-bookTicker-DAY.zip` (public, no key; 30 to 500 MB), streams it in chunks, and prints the time-weighted
distribution of the half spread (ask - bid) / 2 / mid in basis points. Daily files exist from 2023-05-16 to 2024-03-30. The file is deleted
after it has been read. See docs/crypto_spread_check.md for the study that used it."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily/bookTicker"
UA = {"User-Agent": "pitbacktest-research/0.2 (public data only)"}


def fetch(symbol: str, day: str, dest: str) -> int:
    url = f"{BASE}/{symbol}/{symbol}-bookTicker-{day}.zip"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(dest, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    return os.path.getsize(dest)


def measure(path: str, cap_s: float = 60.0, nbins: int = 20000, hi_bp: float = 200.0) -> dict:
    """Time-weighted half spread over one day. Each quote counts for the time until the next update, at most `cap_s` seconds (a feed gap is not
    a quote). The median and the 90th percentile come from a fine histogram, so they are exact to hi_bp / nbins = 0.01 bp."""
    edges = np.linspace(0, hi_bp, nbins + 1)
    hist = np.zeros(nbins)
    tot = wsum = 0.0
    n = crossed = 0
    last = None
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        for ch in pd.read_csv(f, chunksize=2_000_000, usecols=["best_bid_price", "best_ask_price", "event_time"]):
            t = ch["event_time"].to_numpy(np.int64) / 1000.0
            bid, ask = ch["best_bid_price"].to_numpy(float), ch["best_ask_price"].to_numpy(float)
            if last is not None:                                     # carry the last quote of the previous chunk so that no interval is lost
                t, bid, ask = np.concatenate([[last[0]], t]), np.concatenate([[last[1]], bid]), np.concatenate([[last[2]], ask])
            last = (t[-1], bid[-1], ask[-1])
            dt = np.minimum(np.diff(t), cap_s)
            half = (0.5 * (ask - bid) / ((ask + bid) / 2) * 1e4)[:-1]
            ok = (dt > 0) & np.isfinite(half) & (half >= 0)
            crossed += int(((ask - bid) < 0).sum())
            hist += np.histogram(np.clip(half[ok], 0, hi_bp - 1e-9), bins=edges, weights=dt[ok])[0]
            tot += float((half[ok] * dt[ok]).sum())
            wsum += float(dt[ok].sum())
            n += int(ok.sum())
    cum = np.cumsum(hist) / hist.sum()
    return {"updates": n, "seconds": round(wsum), "half_mean_bp": tot / wsum, "half_median_bp": float(edges[np.searchsorted(cum, 0.5)]),
            "half_p90_bp": float(edges[np.searchsorted(cum, 0.9)]), "crossed": crossed}


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) % 2 == 0:
        sys.exit("usage: crypto_spread_probe.py SYMBOL DAY [SYMBOL DAY ...]")
    for sym, day in zip(sys.argv[1::2], sys.argv[2::2]):
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "x.zip")
            t0 = time.time()
            mb = fetch(sym, day, dest) / 1e6
            r = measure(dest)
        print(sym, day, f"{mb:.0f} MB, {time.time() - t0:.0f} s", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
