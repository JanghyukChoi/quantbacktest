"""Check the intraday parser against real Binance files, with an independent reference: Binance publishes 1-day klines
separately, so 1-minute bars summed to days must equal them. Needs network, no key.

    python docs/intraday_realdata_check.py [SYMBOL] [YYYY-MM]
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitbacktest.crypto import binance_archive as ba, intraday as ib

sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
month = sys.argv[2] if len(sys.argv) > 2 else "2024-01"
base = f"{ba.DL_URL}/data/futures/um/monthly"
m1 = ba._get(f"{base}/klines/{sym}/1m/{sym}-1m-{month}.zip")
d1 = ba._get(f"{base}/klines/{sym}/1d/{sym}-1d-{month}.zip")
fr = ba._get(f"{base}/fundingRate/{sym}/{sym}-fundingRate-{month}.zip")
assert m1 and d1, "file not in the archive"

one = ib.parse_minute_zip(m1)
days = pd.Period(month).days_in_month
print(f"{sym} {month}: {len(one):,} minutes parsed, expected {days * 1440:,}; first label {one.index[0]}, last {one.index[-1]}")
ref = ba._frame_from_klines(ba._read_klines_zip(d1))
agg = ib.aggregate_bars(one, "1440min")
agg.index = (agg.index - pd.Timedelta(days=1)).normalize()
j = agg.join(ref, rsuffix="_d", how="inner")
print(f"days compared: {len(j)}")
for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades"):
    rel = ((j[c] - j[c + "_d"]).abs() / j[c + "_d"].abs().clip(lower=1e-12)).max()
    print(f"  {c:13s} max relative difference {rel:.2e}")
if fr:
    s = ib.funding_series([ib.parse_funding_zip(fr)])
    print(f"funding settlements: {len(s)} (3 a day expected: {days * 3}); spacings {set(pd.Series(s.index).diff().dropna())}")
