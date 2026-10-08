"""Tiingo adapter, tested offline with a fake API whose right answers are known.

U1 delisting   a security whose bars stop is in the panel, flagged, and removed by survivors_only
U2 prices      a split leaves the adjusted return unchanged; a sub-dollar stock is never eligible
U3 reuse       a ticker with two listing windows becomes two securities; bars outside the windows are dropped
U4 resumable   a rerun makes zero requests; a quota error stops cleanly; unknown tickers are recorded, not retried
U5 point in time   changing the future does not change eligibility in the past
U6 sample      the draw is seeded, independent of input order, and the frame excludes warrants, units and preferreds
U7 loud        a malformed answer raises and nothing is saved
U8 calendar    a date with too few bars is not a trading day
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pitbacktest.adapters import tiingo
from pitbacktest.equity import survivors_only

DAYS = pd.bdate_range("2021-01-04", periods=300)


def _bars(dates, level, seed, split_at=None, vol=100_000, noise=0.01):
    rng = np.random.default_rng(seed)
    adj = level * np.exp(np.cumsum(rng.normal(0, noise, len(dates))))
    raw = adj.copy()
    if split_at is not None:                                   # a 2:1 split: raw price halves, adjusted does not move
        raw[split_at:] = raw[split_at:] / 2
    return [{"date": f"{d:%Y-%m-%d}T00:00:00.000Z", "close": float(r), "volume": int(vol), "adjClose": float(a),
             "high": 1.0, "low": 1.0} for d, r, a in zip(dates, raw, adj)]


def _world():
    api, rows = {}, []
    for i in range(14):
        api[f"S{i:02d}"] = _bars(DAYS, 50.0, i)
        rows.append((f"S{i:02d}", DAYS[0], DAYS[-1]))
    api["DEAD"] = _bars(DAYS[:201], 40.0, 90); rows.append(("DEAD", DAYS[0], DAYS[200]))
    api["SPLIT"] = _bars(DAYS, 60.0, 91, split_at=150); rows.append(("SPLIT", DAYS[0], DAYS[-1]))
    api["PENNY"] = _bars(DAYS, 0.5, 92, vol=10_000_000); rows.append(("PENNY", DAYS[0], DAYS[-1]))
    reuse = _bars(DAYS[:41], 30.0, 93) + _bars(DAYS[41:240], 5.0, 94) + _bars(DAYS[240:], 80.0, 95)
    api["REUSE"] = reuse                                        # the middle is another company's data, outside both windows
    rows += [("REUSE", DAYS[0], DAYS[40]), ("REUSE", DAYS[240], DAYS[-1])]
    api["GHOST"] = None; rows.append(("GHOST", DAYS[0], DAYS[-1]))
    master = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    master["alive_today"] = master["end"] >= DAYS[-1]
    master["exchange"] = "NYSE"
    return api, master


class Fake:
    def __init__(self, api, quota_after=None):
        self.api, self.calls, self.quota_after = api, [], quota_after

    def __call__(self, t, start):
        if self.quota_after is not None and len(self.calls) >= self.quota_after:
            raise tiingo.QuotaExceeded("fake limit")
        self.calls.append(t)
        return self.api[t]


def _panel(store, api, master, **kw):
    tiingo.fetch_symbols(list(api), store, caller=Fake(api), sleep=0, progress=False)
    return tiingo.build_tiingo_panel(store, master, list(api), start="2021-01-04", min_names_per_date=10, **kw)


def test_delisting_split_and_penny():
    api, master = _world()
    with tempfile.TemporaryDirectory(prefix="tiingotest-") as d:
        p = _panel(d, api, master)
        assert "DEAD" in p.close.columns
        assert p.delist_after["DEAD"].sum() == 1 and p.delist_after["DEAD"].idxmax() == DAYS[200]
        assert p.close["DEAD"].dropna().index[-1] == DAYS[200]
        assert not p.eligible["DEAD"].iloc[201:].any() and p.eligible["DEAD"].iloc[100]
        b = survivors_only(p)
        assert "DEAD" not in b.close.columns and "S00" in b.close.columns
        r = p.ret1()["SPLIT"]
        assert abs(r.iloc[150]) < 0.06, r.iloc[150]            # the raw price halved; the return did not
        assert not p.eligible["PENNY"].any()                   # raw close under 1 dollar
        assert p.eligible["S00"].iloc[100:].all() and not p.eligible["S00"].iloc[:30].any()   # age and ADV warm-up
        print("U1/U2 delisting kept and flagged, survivors_only drops it; split and penny handled  PASS")


def test_reuse_windows():
    api, master = _world()
    with tempfile.TemporaryDirectory(prefix="tiingotest-") as d:
        p = _panel(d, api, master)
        cols = [c for c in p.close.columns if c.startswith("REUSE")]
        assert len(cols) == 2, cols
        a, b = sorted(cols)
        assert p.close[a].dropna().index.max() <= DAYS[40] and p.close[b].dropna().index.min() >= DAYS[240]
        assert p.close[a].notna().sum() == 41 and p.close[b].notna().sum() == 60
        assert p.delist_after[a].sum() == 1 and p.delist_after[b].sum() == 0     # the first window ended; the second runs on
        print("U3 two listing windows become two securities, the other company's bars are dropped  PASS")


def test_resumable_quota_and_none():
    api, master = _world()
    with tempfile.TemporaryDirectory(prefix="tiingotest-") as d:
        f1 = Fake(api, quota_after=6)
        r1 = tiingo.fetch_symbols(list(api), d, caller=f1, sleep=0, progress=False)
        assert r1["calls"] == 6 and r1["stopped"].startswith("quota"), r1
        f2 = Fake(api)
        r2 = tiingo.fetch_symbols(list(api), d, caller=f2, sleep=0, progress=False)
        assert r2["skipped"] == 6 and r2["stopped"] is None and r2["none"] == 1       # GHOST recorded as none
        f3 = Fake(api)
        r3 = tiingo.fetch_symbols(list(api), d, caller=f3, sleep=0, progress=False)
        assert r3["calls"] == 0 and f3.calls == [] and r3["skipped"] == len(api)
        f4 = Fake(api)
        tiingo.fetch_symbols(["S00"], Path(d) / "other", caller=f4, sleep=0, progress=False, max_new=0)
        assert f4.calls == []
        print("U4 quota stops cleanly, resumes where it stopped, a rerun makes zero requests, unknown ticker recorded once  PASS")


def test_point_in_time():
    api, master = _world()
    with tempfile.TemporaryDirectory(prefix="tiingotest-") as d1, tempfile.TemporaryDirectory(prefix="tiingotest-") as d2:
        p1 = _panel(d1, api, master)
        api2 = dict(api)
        api2["S03"] = api["S03"][:151] + _bars(DAYS[151:], 50.0, 77, vol=1)         # the future changes: volume collapses after day 150
        p2 = _panel(d2, api2, master)
        assert (p1.eligible.iloc[:151] == p2.eligible.iloc[:151]).all().all()
        assert p1.eligible["S03"].iloc[100] and not p2.eligible["S03"].iloc[-1]
        print("U5 eligibility in the past is unchanged when the future changes  PASS")


def test_sample_and_frame():
    t = [f"A{i:03d}" for i in range(200)]
    o1 = tiingo.draw_order(t, seed=0)
    assert o1 == tiingo.draw_order(list(reversed(t)), seed=0) and o1 != tiingo.draw_order(t, seed=1)
    assert sorted(o1) == sorted(t) and o1[:50] != sorted(t)[:50]
    m = pd.DataFrame({"ticker": ["AAPL", "BRK-B", "BF.B", "ABCDW", "ABCDU", "ABCDR", "ABCDP", "ABCDQ", "WWW", "OLD"],
                      "start": pd.Timestamp("2005-01-01"),
                      "end": [pd.Timestamp("2026-10-06")] * 9 + [pd.Timestamp("2010-01-01")]})
    f = tiingo.study_frame(m, since="2013-01-01")
    assert list(f["ticker"]) == ["AAPL", "ABCDQ", "WWW"], list(f["ticker"])      # Q (bankruptcy) is kept, warrants and old ones are not
    print("U6 the draw is seeded and order independent; the frame drops warrants, units, preferreds, share classes, old windows  PASS")


def test_malformed_is_loud():
    with tempfile.TemporaryDirectory(prefix="tiingotest-") as d:
        bad = lambda t, s: [{"date": "2021-01-04", "close": 1.0, "volume": 5}]            # no adjClose
        try:
            tiingo.fetch_symbols(["X"], d, caller=bad, sleep=0, progress=False)
        except RuntimeError as e:
            assert "adjClose" in str(e)
        else:
            raise AssertionError("a malformed answer must raise")
        assert not list(Path(d).glob("*")), "nothing may be saved for a malformed answer"
        try:
            tiingo.fetch_symbols(["X"], d, caller=lambda t, s: {"detail": "x"}, sleep=0, progress=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a non-list answer must raise")
        print("U7 malformed answers raise and leave no file  PASS")


def test_calendar():
    api, master = _world()
    extra = DAYS[-1] + pd.offsets.BDay(1)                                  # a date only one name has a bar on
    api["S00"] = api["S00"] + [{"date": f"{extra:%Y-%m-%d}T00:00:00.000Z", "close": 50.0, "volume": 1000, "adjClose": 50.0}]
    master.loc[master.ticker == "S00", "end"] = extra
    with tempfile.TemporaryDirectory(prefix="tiingotest-") as d:
        p = _panel(d, api, master)
        assert extra not in p.close.index and DAYS[-1] in p.close.index
        # without the rule the date would be there: the test would be vacuous if it were not
        tiingo.fetch_symbols(list(api), d, caller=Fake(api), sleep=0, progress=False)
        loose = tiingo.build_tiingo_panel(d, master, list(api), start="2021-01-04", min_names_per_date=1)
        assert extra in loose.close.index
        print("U8 a date with too few bars is not a trading day  PASS")


def test_quota_message_in_a_200_response_stops_cleanly():
    """The monthly symbol limit arrives as HTTP 200 with a `detail` message (seen on a free account); it must stop the run like a 429,
    not be mistaken for a format change, and a dict that is not a quota message must still raise."""
    class R:
        def __init__(self, b): self.b = b
        def read(self): return self.b
        def __enter__(self): return self
        def __exit__(self, *a): return False
    import json as _json
    real = tiingo.urllib.request.urlopen
    msg = {"detail": "You have run over your 500 symbol look up for this month. Please upgrade at https://api.tiingo.com/pricing"}
    try:
        tiingo.urllib.request.urlopen = lambda req, timeout=0: R(_json.dumps(msg).encode())
        with tempfile.TemporaryDirectory(prefix="tiingotest-") as d:
            r = tiingo.fetch_symbols(["AAA", "BBB"], d, key="k", sleep=0, progress=False)
            assert r["stopped"].startswith("quota") and "500 symbol" in r["stopped"] and r["calls"] == 0, r
            assert not list(Path(d).glob("*")), "a quota message must not be stored as data"
        tiingo.urllib.request.urlopen = lambda req, timeout=0: R(_json.dumps({"detail": "Something else is wrong"}).encode())
        with tempfile.TemporaryDirectory(prefix="tiingotest-") as d:
            try:
                tiingo.fetch_symbols(["AAA"], d, key="k", sleep=0, progress=False)
            except RuntimeError as e:
                assert "unexpected response" in str(e)
            else:
                raise AssertionError("an unknown dict must still be loud")
    finally:
        tiingo.urllib.request.urlopen = real
    print("U9 the monthly 500-symbol message (HTTP 200) stops the run cleanly and is never stored; an unknown dict still raises  PASS")


def _full(api):
    """The same world with the extra fields of a full download: the open is 1% under the close, the dividend is paid on one day."""
    out = {}
    for t, rows in api.items():
        if rows is None:
            out[t] = None
            continue
        out[t] = [{**r, "open": r["close"] * 0.99, "adjOpen": r["adjClose"] * 0.99, "adjHigh": r["adjClose"] * 1.01, "adjLow": r["adjClose"] * 0.98,
                   "divCash": 0.5 if i == 30 else 0.0, "splitFactor": 1.0} for i, r in enumerate(rows)]
    return out


def test_full_fields():
    api, master = _world()
    fapi = _full(api)
    with tempfile.TemporaryDirectory(prefix="tiingotest-") as d, tempfile.TemporaryDirectory(prefix="tiingotest-") as d2:
        tiingo.fetch_symbols(list(fapi), d, caller=Fake(fapi), sleep=0, progress=False, full=True)
        tiingo.fetch_symbols(list(api), d2, caller=Fake(api), sleep=0, progress=False)
        pf = tiingo.build_tiingo_panel(d, master, list(fapi), start="2021-01-04", min_names_per_date=10)
        pp = tiingo.build_tiingo_panel(d2, master, list(api), start="2021-01-04", min_names_per_date=10)
        assert pp.open is None and pp.meta.get("div_cash") is None                       # a store made without full=True is read as before
        assert pf.open is not None and pf.high is not None and pf.low is not None
        assert np.allclose((pf.open / pf.close).stack().dropna(), 0.99)                  # adjOpen is 0.99 x adjClose in this world
        assert np.array_equal(pf.close.fillna(-1).to_numpy(), pp.close.fillna(-1).to_numpy())     # prices and eligibility are the same either way
        assert pf.eligible.equals(pp.eligible)
        dc = pf.meta["div_cash"]
        assert dc.shape == pf.close.shape and (dc.fillna(0).sum(axis=0) > 0).sum() > 10
        # a full download that lacks a field is loud and saves nothing
        bad = {"X": [{k: v for k, v in r.items() if k != "adjOpen"} for r in fapi["S00"]]}
        with tempfile.TemporaryDirectory(prefix="tiingotest-") as d3:
            try:
                tiingo.fetch_symbols(["X"], d3, caller=lambda t, s: bad[t], sleep=0, progress=False, full=True)
            except RuntimeError as e:
                assert "adjOpen" in str(e)
            else:
                raise AssertionError("a bar without adjOpen must be loud in a full download")
            assert not list(Path(d3).glob("*"))
    print("U10 a full download keeps open, high, low and dividends; the panel gets them; a store without them reads as before; a missing field is loud  PASS")


if __name__ == "__main__":
    test_full_fields()
    test_delisting_split_and_penny()
    test_reuse_windows()
    test_resumable_quota_and_none()
    test_point_in_time()
    test_sample_and_frame()
    test_malformed_is_loud()
    test_calendar()
    test_quota_message_in_a_200_response_stops_cleanly()
    print("tiingo tests: all passed")
