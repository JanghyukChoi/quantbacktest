"""Probe: what does Tiingo's end-of-day API return for the same 41 delisted, acquired or renamed US stocks as
`survivorship_probe.py` (which measured yfinance)?

    TIINGO_API_KEY=... python docs/tiingo_probe.py

A convenience sample chosen from memory, NOT a statistical sample, so do not read a rate off it. Delisting years are
approximate. It makes one request per ticker; a free account has hourly limits, so the run stops at the first HTTP 429.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

L = [("LEH", 2008), ("WB", 2008), ("BSC", 2008), ("CFC", 2008), ("MER", 2008), ("WM", 2008), ("GM", 2009), ("CIT", 2009),
     ("SIVB", 2023), ("FRC", 2023), ("SBNY", 2023), ("TWX", 2018), ("MON", 2018), ("CELG", 2019), ("ESRX", 2018),
     ("AGN", 2020), ("RTN", 2020), ("TWC", 2016), ("SHLD", 2018), ("JCP", 2020), ("GGP", 2018), ("DPS", 2018),
     ("AET", 2018), ("CA", 2018), ("XLNX", 2022), ("ATVI", 2023), ("TWTR", 2022), ("CERN", 2022), ("VMW", 2023),
     ("CTXS", 2022), ("NLSN", 2022), ("ANTM", 2022), ("PBCT", 2022), ("FB", 2022), ("SYMC", 2019), ("LLTC", 2017),
     ("YHOO", 2017), ("HOT", 2016), ("TYC", 2016), ("BRCM", 2016), ("KRFT", 2015)]
KEY = os.environ.get("TIINGO_API_KEY")
if not KEY:
    sys.exit("set TIINGO_API_KEY")


def fetch(t):
    req = urllib.request.Request(f"https://api.tiingo.com/tiingo/daily/{t.lower()}/prices?startDate=2000-01-01&format=json",
                                 headers={"Authorization": f"Token {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


out, detail = {"no data": 0, "another company (ticker reused)": 0, "history present": 0, "cut short": 0}, []
for t, y in L:
    st, rows = fetch(t)
    if st == 429:
        print("rate limited at", t, "- stopped"); break
    if st != 200 or not rows:
        k, last = "no data", "-"
    else:
        last = rows[-1]["date"][:10]
        ly = int(last[:4])
        k = ("another company (ticker reused)" if ly > y + 1 else "history present" if ly >= y - 1 else "cut short")
    out[k] += 1
    detail.append((t, y, k, last))
    time.sleep(0.3)
for d in detail:
    print(f"{d[0]:6s} delisted ~{d[1]}  {d[2]:32s} last bar {d[3]}")
print(f"n = {len(detail)}", out)
