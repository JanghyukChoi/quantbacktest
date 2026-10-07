"""Probe: what does yfinance return for well-known US stocks that were delisted, acquired or renamed?

    python docs/survivorship_probe.py

A convenience sample chosen from memory, NOT a statistical sample, so do not read a rate off it. The delisting years
are approximate. The point is to see which kinds of failure happen. Results depend on Yahoo's data on the day you run it.
"""
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
L = [("LEH", 2008), ("WB", 2008), ("BSC", 2008), ("CFC", 2008), ("MER", 2008), ("WM", 2008), ("GM", 2009), ("CIT", 2009),
     ("SIVB", 2023), ("FRC", 2023), ("SBNY", 2023), ("TWX", 2018), ("MON", 2018), ("CELG", 2019), ("ESRX", 2018),
     ("AGN", 2020), ("RTN", 2020), ("TWC", 2016), ("SHLD", 2018), ("JCP", 2020), ("GGP", 2018), ("DPS", 2018),
     ("AET", 2018), ("CA", 2018), ("XLNX", 2022), ("ATVI", 2023), ("TWTR", 2022), ("CERN", 2022), ("VMW", 2023),
     ("CTXS", 2022), ("NLSN", 2022), ("ANTM", 2022), ("PBCT", 2022), ("FB", 2022), ("SYMC", 2019), ("LLTC", 2017),
     ("YHOO", 2017), ("HOT", 2016), ("TYC", 2016), ("BRCM", 2016), ("KRFT", 2015)]
d = yf.download([t for t, _ in L], start="2000-01-01", end="2025-06-01", group_by="ticker", progress=False,
                threads=True, auto_adjust=False)
out = {"no data": 0, "another company (ticker reused)": 0, "history present": 0, "cut short": 0}
for t, y in L:
    try:
        c = d[t]["Close"].dropna()
    except Exception:
        c = pd.Series(dtype=float)
    if len(c) == 0:
        k = "no data"
    elif c.index[-1].year > y + 1:
        k = "another company (ticker reused)"
    elif c.index[-1].year >= y - 1:
        k = "history present"
    else:
        k = "cut short"
    out[k] += 1
print(f"n = {len(L)}", out)
