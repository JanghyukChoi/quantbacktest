"""What the order book implies for the square-root impact coefficient Y of `ImpactModel`, for Binance USDT-M perpetuals.

    python docs/crypto_impact_study.py            # needs the cached daily bars (ArchiveStore) and network for the public bookDepth files

`bookDepth` (public, daily, 2023-01-01 to 2024-05-17) gives, every 30 seconds, the cumulative resting notional within 1 to 5 percent of the mid
on each side. A market order of notional Q walks that book; its average price against the mid is the slippage. Setting that equal to the model's
unit cost `Y * sigma * sqrt(Q / ADV)` gives the Y the book implies at that size. What this is NOT: a measurement of impact, because resting depth
refills, orders are split, and the information in a trade is not in the book; it is the cost of taking the visible book at once. See
docs/crypto_impact_check.md."""
from __future__ import annotations

import io
import os
import sys
import urllib.error
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
UA = {"User-Agent": "pitbacktest-research/0.2 (public data only)"}
FRACTIONS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)          # order size as a share of the day's turnover


def depth_profile(symbol: str, day: str) -> pd.DataFrame | None:
    """Median over the day's snapshots of the cumulative notional within p percent: a frame indexed by p = -5..-1, 1..5. None if there is no file."""
    url = f"{BASE}/{symbol}/{symbol}-bookDepth-{day}.zip"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            blob = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    return df.groupby("percentage")["notional"].median().to_frame("notional")


def average_slippage(cum_notional: np.ndarray, q: float) -> float:
    """Average price paid above the mid, as a fraction, by a market order of notional q that walks a book whose cumulative notional within 1, 2, ..., 5
    percent is `cum_notional` (linear inside each band). NaN if q is more than the book shown (5 percent)."""
    n = np.concatenate([[0.0], np.asarray(cum_notional, float)])
    p = np.arange(0, len(n)) * 0.01
    if not np.all(np.diff(n) > 0) or q > n[-1] or q <= 0:
        return float("nan")
    qs = np.linspace(0, q, 2001)
    price = np.interp(qs, n, p)                       # price level reached after consuming qs
    return float(np.trapezoid(price, qs) / q)


def implied_y(profile: pd.DataFrame, adv: float, sigma_daily: float, fractions=FRACTIONS) -> dict:
    """{fraction: Y} where Y = slippage / (sigma * sqrt(fraction)), averaging the buy side (+1..+5) and the sell side (-1..-5)."""
    ask = profile.loc[[1, 2, 3, 4, 5], "notional"].to_numpy()
    bid = profile.loc[[-1, -2, -3, -4, -5], "notional"].to_numpy()
    out = {}
    for f in fractions:
        q = f * adv
        s = np.nanmean([average_slippage(ask, q), average_slippage(bid, q)])
        out[f] = s / (sigma_daily * np.sqrt(f)) if np.isfinite(s) else float("nan")
    return out


def main(n_symbols: int = 30, root: str | None = None) -> pd.DataFrame:
    from pitbacktest.crypto import ArchiveStore, panel as cp
    root = root or os.path.expanduser("~/.cache/quantbt/binance_um")
    names = sorted(f[:-4] for f in os.listdir(root + "/daily") if f.endswith(".pkl"))
    warnings.filterwarnings("ignore")
    P = cp.build_panel(ArchiveStore(root), symbols=names, start="2022-06-01", end="2024-05-17", min_adv_usd=5e6, min_age_days=60, live=set(names))
    adv30 = P.adv(30)
    sig20 = P.ret1().rolling(20, min_periods=15).std()
    days = [str(d.date()) for d in pd.to_datetime(["2023-02-15", "2023-04-12", "2023-06-14", "2023-08-16", "2023-10-11", "2023-12-13",
                                                     "2024-01-17", "2024-02-14", "2024-03-13", "2024-04-17"])]
    a = adv30.loc[days].mean()[P.eligible.loc[days].all()].dropna().sort_values(ascending=False)
    pick = [a.index[int(i)] for i in np.unique(np.linspace(0, len(a) - 1, n_symbols).round().astype(int))]
    rows = []
    for s in pick:
        for d in days:
            prof = depth_profile(s, d)
            if prof is None:
                continue
            y = implied_y(prof, float(adv30.loc[d, s]), float(sig20.loc[d, s]))
            rows.append({"symbol": s, "day": d, "adv_usd": float(adv30.loc[d, s]), "sigma": float(sig20.loc[d, s]),
                         "depth1pct_usd": float(prof.loc[[1, -1], "notional"].mean()), **{f"y@{f:g}": v for f, v in y.items()}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    df = main()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_impact_study_results.csv")
    df.to_csv(out, index=False)
    print(len(df), "contract-days ->", out)
