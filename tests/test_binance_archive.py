"""The Binance archive downloader on a fake network (no network).

B1 transport     404 is None, a 503 is retried, a 403 is not, a Chinese symbol is percent-encoded
B2 listing       paginated S3 listings are followed to the end; the prefix itself is not a directory
B3 parsing       a header row, string numbers and millisecond or microsecond stamps all give the same daily frame
B4 daily bars    months are joined, duplicates dropped, the live top-up is added and the unfinished day is dropped; the result is cached
B5 funding       settlements are summed per UTC day (header and microsecond files included), the live top-up is added
B6 symbols       delivery contracts and other quotes are excluded; the list is cached
B7 fetch_all     one bad symbol is counted and does not stop the run
"""
from __future__ import annotations
import io, json, sys, tempfile, time, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zipfile
import numpy as np
import pandas as pd
from pitbacktest.crypto import binance_archive as ba
from pitbacktest.crypto.binance_archive import ArchiveStore


class Resp:
    def __init__(self, b): self.b = b
    def read(self): return self.b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _zip(csv: str) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("x.csv", csv)
    return out.getvalue()


def _daily_csv(days, unit="ms", header=False, start_px=100.0, qv=1e6):
    rows = []
    for i, d in enumerate(days):
        t = int(pd.Timestamp(d).timestamp() * (1000 if unit == "ms" else 1_000_000))
        px = start_px + i
        rows.append(f"{t},{px},{px + 1},{px - 1},{px + 0.5},{qv / px},{t + 86_399_999},{qv},100,0,0,0")
    head = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n" if header else ""
    return head + "\n".join(rows) + "\n"


def test_transport():
    seen = []
    calls = {"n": 0}
    real_urlopen, real_sleep = ba.urllib.request.urlopen, ba.time.sleep
    ba.time.sleep = lambda s: None
    try:
        def fake(req, timeout=0):
            seen.append(req.full_url)
            calls["n"] += 1
            u = req.full_url
            if "missing" in u:
                raise urllib.error.HTTPError(u, 404, "nf", {}, None)
            if "flaky" in u and calls["n"] % 3 != 0:
                raise urllib.error.HTTPError(u, 503, "busy", {}, None)
            if "forbidden" in u:
                raise urllib.error.HTTPError(u, 403, "no", {}, None)
            if "down" in u:
                raise urllib.error.URLError("unreachable")
            return Resp(b"ok")
        ba.urllib.request.urlopen = fake
        assert ba._get("https://x/missing") is None
        calls["n"] = 0
        assert ba._get("https://x/flaky") == b"ok" and calls["n"] == 3                              # two 503s, then success
        n = calls["n"]
        try:
            ba._get("https://x/forbidden")
        except urllib.error.HTTPError as e:
            assert e.code == 403 and calls["n"] == n + 1                                            # not retried
        else:
            raise AssertionError("403 must raise")
        calls["n"] = 0
        try:
            ba._get("https://x/down", retries=2)
        except urllib.error.URLError:
            assert calls["n"] == 3                                                                  # the first try and two retries
        else:
            raise AssertionError("a dead network must raise after the retries")
        ba._get("https://data.binance.vision/data/futures/um/monthly/klines/币安人生USDT/1d/a.zip")
        assert "%E5%B8%81" in seen[-1] and "币" not in seen[-1], seen[-1]
    finally:
        ba.urllib.request.urlopen, ba.time.sleep = real_urlopen, real_sleep
    print("B1 404 -> None; a 503 is retried to success; a 403 is not; a dead network raises after the retries; a Chinese symbol is percent-encoded  PASS")


def test_listing():
    pages = {
        "": '<R><Key>a/1.zip</Key><Key>a/2.zip</Key><IsTruncated>true</IsTruncated></R>',
        "a/2.zip": '<R><Key>a/3.zip</Key><IsTruncated>false</IsTruncated></R>',
    }
    dirs = {
        "": '<R><Prefix>p/</Prefix><Prefix>p/AAA/</Prefix><IsTruncated>true</IsTruncated><NextMarker>p/AAA/</NextMarker></R>',
        "p/AAA/": '<R><Prefix>p/BBB/</Prefix><IsTruncated>false</IsTruncated></R>',
    }
    real = ba._get
    def fake(url, retries=4, timeout=30):
        q = dict(x.split("=", 1) for x in url.split("?", 1)[1].split("&"))
        mk = q.get("marker", "").replace("%2F", "/")
        return (dirs if "delimiter" in q else pages)[mk].encode()
    ba._get = fake
    try:
        assert ba._list("a/") == ["a/1.zip", "a/2.zip", "a/3.zip"]
        assert ba._list_dirs("p/") == ["p/AAA/", "p/BBB/"]                                         # "p/" itself is not a directory of itself
    finally:
        ba._get = real
    print("B2 paginated listings are followed to the end and the prefix is not returned as its own directory  PASS")


def test_parsing():
    days = pd.date_range("2024-03-01", periods=5)
    plain = ba._frame_from_klines(ba._read_klines_zip(_zip(_daily_csv(days))))
    hdr = ba._frame_from_klines(ba._read_klines_zip(_zip(_daily_csv(days, header=True))))
    us = ba._frame_from_klines(ba._read_klines_zip(_zip(_daily_csv(days, unit="us"))))
    dup = ba._frame_from_klines(ba._read_klines_zip(_zip(_daily_csv(days) + _daily_csv(days[:2], start_px=500.0))))
    for x in (hdr, us):
        assert x.index.equals(plain.index) and np.allclose(x["close"], plain["close"]), x
    assert plain.index[0] == pd.Timestamp("2024-03-01") and len(plain) == 5 and plain["close"].iloc[0] == 100.5
    assert len(dup) == 5 and dup["close"].iloc[0] == 500.5                                          # a later duplicate wins
    assert ba._frame_from_klines(pd.DataFrame()).empty
    print("B3 header rows, microsecond stamps and duplicate days give the same clean daily frame  PASS")


def test_fetch_daily_cache_live_and_unfinished_day():
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    keys = ["data/futures/um/monthly/klines/AAAUSDT/1d/AAAUSDT-1d-2024-01.zip", "data/futures/um/monthly/klines/AAAUSDT/1d/AAAUSDT-1d-2024-02.zip"]
    blobs = {keys[0]: _zip(_daily_csv(pd.date_range("2024-01-01", "2024-01-31"))),
             keys[1]: _zip(_daily_csv(pd.date_range("2024-02-01", "2024-02-29"), start_px=200.0))}
    live_rows = []
    for d in (pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-02"), today):                         # the last one is today: unfinished
        t = int(d.timestamp() * 1000)
        live_rows.append([t, "300", "301", "299", "300.5", "10", t + 86_399_999, "3000", 5, "0", "0", "0"])
    real = (ba._list, ba._get)
    n = {"get": 0}
    def fake_get(url, retries=4, timeout=30):
        n["get"] += 1
        if "fapi/v1/klines" in url:
            return json.dumps(live_rows).encode()
        return blobs[url.split("data.binance.vision/")[1]]
    ba._list, ba._get = (lambda prefix: keys), fake_get
    try:
        with tempfile.TemporaryDirectory(prefix="batest-") as d:
            store = ArchiveStore(Path(d))
            df = store.fetch_daily("AAAUSDT", live=True)
            assert len(df) == 31 + 29 + 2 and df.index.max() == pd.Timestamp("2024-03-02") and today not in df.index
            assert df.index.is_monotonic_increasing and not df.index.has_duplicates
            assert df["close"].loc["2024-03-01"] == 300.5 and df["close"].loc["2024-02-01"] == 200.5
            calls = n["get"]
            again = store.fetch_daily("AAAUSDT", live=True)                                          # cached: no request, same frame
            assert n["get"] == calls and again.equals(df)
            off = ArchiveStore(Path(d) / "x").fetch_daily("AAAUSDT", live=False)
            assert off.index.max() == pd.Timestamp("2024-02-29") and len(off) == 60                  # no top-up when the contract no longer trades
    finally:
        ba._list, ba._get = real
    print("B4 months joined, live top-up added, today's unfinished bar dropped, second call served from the cache  PASS")


def test_fetch_funding_daily_sums():
    key = "data/futures/um/monthly/fundingRate/AAAUSDT/AAAUSDT-fundingRate-2024-01.zip"
    def ev(day, hours, rate, unit):
        k = 1000 if unit == "ms" else 1_000_000
        return [f"{int((pd.Timestamp(day) + pd.Timedelta(hours=h)).timestamp() * k)},8,{rate}" for h in hours]
    rows = ev("2024-01-01", (0, 8, 16), 0.0001, "ms") + ev("2024-01-02", (0, 8), 0.0002, "ms")
    csv = "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(rows) + "\n"
    live = [{"fundingTime": int(pd.Timestamp("2024-01-03").timestamp() * 1000), "fundingRate": "0.0005"}]
    real = (ba._list, ba._get)
    ba._list = lambda prefix: [key]
    ba._get = lambda url, retries=4, timeout=30: json.dumps(live).encode() if "fundingRate?symbol" in url else _zip(csv)
    try:
        with tempfile.TemporaryDirectory(prefix="batest-") as d:
            s = ArchiveStore(Path(d)).fetch_funding("AAAUSDT", live=True)
            assert abs(s.loc["2024-01-01"] - 0.0003) < 1e-15 and abs(s.loc["2024-01-02"] - 0.0004) < 1e-15 and abs(s.loc["2024-01-03"] - 0.0005) < 1e-15, s
            assert s.index.is_monotonic_increasing
        usv = "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(ev("2024-01-01", (0, 8, 16), 0.0001, "us")) + "\n"
        ba._get = lambda url, retries=4, timeout=30: _zip(usv)
        with tempfile.TemporaryDirectory(prefix="batest-") as d:
            s = ArchiveStore(Path(d)).fetch_funding("AAAUSDT", live=False)
            assert abs(s.loc["2024-01-01"] - 0.0003) < 1e-15 and len(s) == 1                          # microsecond stamps land on the right day
    finally:
        ba._list, ba._get = real
    print("B5 three 8-hour settlements sum to one UTC day (milliseconds or microseconds), the live top-up is added  PASS")


def test_symbols_and_fetch_all():
    real = (ba._list_dirs, ba._get)
    ba._list_dirs = lambda prefix: [prefix + n + "/" for n in ("BTCUSDT", "ETHUSDT", "BTCUSDT_250627", "ETHBUSD", "LUNAUSDT", "FOO_USDT")]
    try:
        with tempfile.TemporaryDirectory(prefix="batest-") as d:
            store = ArchiveStore(Path(d))
            syms = store.symbols()
            assert syms == ["BTCUSDT", "ETHUSDT", "LUNAUSDT"], syms                                  # delivery and BUSD contracts are out
            ba._list_dirs = lambda prefix: (_ for _ in ()).throw(AssertionError("must come from the cache"))
            assert store.symbols() == syms and (Path(d) / "symbols.json").exists()
            # fetch_all: one symbol blows up, one is empty, one is fine; the run completes and counts them
            store.trading_now = lambda: {"BTCUSDT"}
            def fd(sym, live):
                if sym == "ETHUSDT":
                    raise RuntimeError("boom")
                return pd.DataFrame({"close": [1.0]}) if sym == "BTCUSDT" else pd.DataFrame()
            store.fetch_daily, store.fetch_funding = fd, (lambda sym, live: pd.Series([0.0]))
            res = ba.fetch_all(store, workers=2, progress=False)
            assert res["ok"] == 1 and res["empty"] == 1 and res["error"] == 1, res
    finally:
        ba._list_dirs, ba._get = real
    print("B6/B7 delivery and BUSD contracts are excluded and the list is cached; one failing symbol is counted and the run finishes  PASS")


if __name__ == "__main__":
    test_transport()
    test_listing()
    test_parsing()
    test_fetch_daily_cache_live_and_unfinished_day()
    test_fetch_funding_daily_sums()
    test_symbols_and_fetch_all()
    print("binance archive tests: all passed")
