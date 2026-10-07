"""Intraday layer on a fake archive (no network). The design is in docs/intraday_design.md; each test follows it.

I1 parsing       ms and us timestamps agree, rows are labelled by bar end, duplicates and a header row are tolerated
I2 aggregation   1m -> 5m equals an independent loop, with gaps and a ragged end
I3 point in time eligibility for day D uses days up to D-1 only; changing the future never changes the past
I4 latency       a signal that knows exactly one bar ahead earns at lag 1 and nothing at lag 2
I5 funding       a position held through a settlement pays it once; a flat one pays nothing; the 3 ms stamp lands on the right bar
I6 annualise     periods per year follows the bar length; a bar that does not divide a day is refused
I7 break-even    the cost found from two runs makes the net mean zero in a third
I8 memory        the guard names a bar length that fits
I9 download      resumable, a 404 is recorded and not retried, complete months and partial month use different files
I10 halts        a terminal zero-volume tail is cut and flagged; interior zero-volume bars keep their price
I11 guards       entry_lag 0 is refused; a short sample warns instead of going NaN silently
"""
from __future__ import annotations
import io, sys, tempfile, warnings, zipfile
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.crypto import intraday as ib
from pitbacktest.crypto.binance_archive import ArchiveStore
from pitbacktest.weights import backtest_weights

KL = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "tbb", "tbq", "ignore"]


def _zip_bytes(df_end: pd.DataFrame, unit="ms", header=False, dup=False) -> bytes:
    """An archive-style file from a frame indexed by bar END time."""
    op = (df_end.index - pd.Timedelta(minutes=1)).tz_localize("UTC")
    t = op.astype("datetime64[ns, UTC]").asi8 // (10**6 if unit == "ms" else 10**3)
    f = pd.DataFrame({"open_time": t, "open": df_end["open"].to_numpy(), "high": df_end["high"].to_numpy(), "low": df_end["low"].to_numpy(),
                      "close": df_end["close"].to_numpy(), "volume": df_end["volume"].to_numpy(), "close_time": t + 59999,
                      "quote_volume": df_end["quote_volume"].to_numpy(), "count": df_end["trades"].to_numpy(), "tbb": 0, "tbq": 0, "ignore": 0})
    if dup:
        f = pd.concat([f, f.iloc[:3]])
    buf = io.StringIO()
    f.to_csv(buf, index=False, header=header)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("x.csv", ("open_time,open,high,low,close,volume,close_time,quote_volume,count,tbb,tbq,ignore\n" if header else "") + buf.getvalue()
                   if False else buf.getvalue())
    return out.getvalue()


def _minutes(start, n, level=100.0, seed=0, sd=0.0008, vol=2e6):
    rng = np.random.default_rng(seed)
    end = pd.date_range(pd.Timestamp(start) + pd.Timedelta(minutes=1), periods=n, freq="min")
    c = level * np.exp(np.cumsum(rng.normal(0, sd, n)))
    o = np.r_[level, c[:-1]]
    qv = rng.lognormal(np.log(vol), 0.3, n)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0004, "low": np.minimum(o, c) * 0.9996, "close": c,
                         "volume": qv / c, "quote_volume": qv, "trades": 50}, index=end)


# ---------------------------------------------------------------------------------------------------- I1
def test_parse_units_labels_duplicates():
    m = _minutes("2024-01-01", 600, seed=1)
    a = ib.parse_minute_zip(_zip_bytes(m, "ms"))
    b = ib.parse_minute_zip(_zip_bytes(m, "us"))
    c = ib.parse_minute_zip(_zip_bytes(m, "ms", dup=True))
    assert str(a.index.dtype) == str(b.index.dtype) == "datetime64[ns]", (a.index.dtype, b.index.dtype)             # the same unit in every pandas
    for x in (a, b, c):
        assert x.index.equals(m.index.astype("datetime64[ns]")) and np.allclose(x["close"], m["close"]) and np.allclose(x["quote_volume"], m["quote_volume"])
    assert a.index[0] == pd.Timestamp("2024-01-01 00:01")                                      # open 00:00 -> labelled 00:01
    assert ib.parse_minute_zip(_zip_bytes(m.iloc[0:0], "ms")).empty
    print("I1 ms and us timestamps agree; a bar opening at 00:00 is labelled 00:01; duplicates dropped  PASS")


# ---------------------------------------------------------------------------------------------------- I2
def _loop_bars(m: pd.DataFrame, k: int) -> pd.DataFrame:
    """Independent aggregation: group minutes by floor(open / k minutes), plain Python."""
    op = m.index - pd.Timedelta(minutes=1)
    key = (op.astype("datetime64[ns]").asi8 // (k * 60 * 10**9)).astype("int64")
    rows = []
    for kk in sorted(set(key)):
        sub = m[key == kk]
        end = pd.Timestamp(int(kk) * k * 60 * 10**9) + pd.Timedelta(minutes=k)
        rows.append((end, sub["open"].iloc[0], sub["high"].max(), sub["low"].min(), sub["close"].iloc[-1], sub["volume"].sum(),
                     sub["quote_volume"].sum(), len(sub)))
    out = pd.DataFrame(rows, columns=["end", "open", "high", "low", "close", "volume", "quote_volume", "n_min"]).set_index("end")
    return out[out.index <= m.index.max()]


def test_aggregation_matches_loops():
    rng = np.random.default_rng(2)
    m = _minutes("2024-01-01", 1000, seed=3)
    m = m.drop(m.index[rng.choice(len(m), 60, replace=False)])                                  # gaps
    m = m.iloc[:-3]                                                                             # a ragged end
    for k in (5, 15, 60):
        got = ib.aggregate_bars(m, f"{k}min")
        ref = _loop_bars(m, k)
        assert got.index.astype('datetime64[ns]').equals(ref.index.astype('datetime64[ns]')), (k, len(got), len(ref))
        for c in ("open", "high", "low", "close", "volume", "quote_volume"):
            assert np.allclose(got[c], ref[c], rtol=1e-12), (k, c)
        assert (got["n_min"].to_numpy() == ref["n_min"].to_numpy()).all()
        assert got.index.max() <= m.index.max()
    assert (ib.aggregate_bars(m, "5min")["n_min"] < 5).sum() > 0                                # partial bars exist and are counted
    print("I2 1m -> 5m, 15m, 1h equals an independent loop with gaps and a ragged end (incomplete last bar dropped)  PASS")


# ---------------------------------------------------------------------------------------------------- store helpers
def _write_months(store, sym, m: pd.DataFrame, span=("2023-09", "2024-04")):
    d = ib._paths(store, sym)
    for per, g in m.groupby(m.index.to_period("M")):
        g.to_pickle(d / f"{per}.pkl")
    for per in pd.period_range(*span, freq="M"):                  # what fetch_minutes records for a month the archive does not have
        if not (d / f"{per}.pkl").exists():
            (d / f"{per}.none").write_text("")


def _world(tmp, n_days=150, n_sym=12, start="2023-10-01", seed=5, shock_from=None, listing_day=None, halt_after=None):
    rng = np.random.default_rng(seed)
    store = ArchiveStore(Path(tmp))
    names = [f"C{i:02d}USDT" for i in range(n_sym)]
    for i, s in enumerate(names):
        m = _minutes(start, n_days * 1440, seed=seed + i, vol=3e6)
        if s == "C00USDT" and listing_day is not None:
            m = m[m.index > pd.Timestamp(start) + pd.Timedelta(days=listing_day)]
        if shock_from is not None:
            m = m.copy()
            m.loc[m.index > pd.Timestamp(start) + pd.Timedelta(days=shock_from), ["volume", "quote_volume"]] *= 1e-4
        if s == "C01USDT" and halt_after is not None:
            cut = pd.Timestamp(start) + pd.Timedelta(days=halt_after)
            tail = m[m.index > cut].copy()
            tail[["open", "high", "low"]] = tail["close"].iloc[0]
            tail["close"] = tail["close"].iloc[0]
            tail[["volume", "quote_volume"]] = 0.0
            m = pd.concat([m[m.index <= cut], tail])
        _write_months(store, s, m)
    return store, names


def _pin_no_funding():
    return {}


# ---------------------------------------------------------------------------------------------------- I3
def test_point_in_time_eligibility():
    with tempfile.TemporaryDirectory(prefix="intratest-") as d:
        store, names = _world(d, n_days=120, listing_day=95)                          # C00 lists at 2024-01-04 00:00 (day 95 after 2023-10-01)
        kw = dict(bar="1h", adv_window=3, min_age_days=5, min_adv_usd=1e6, funding=False)
        p = ib.build_intraday_panel(store, names, "2024-01-03", "2024-01-25", **kw)
        e0 = p.eligible["C00USDT"]
        first = e0[e0].index[0]
        # whole days of history up to D-1 must be >= 5: days 01-04..01-08 are five, so the first eligible day is 01-09. A rule that let
        # day D itself count (as the daily panel may) would start on 01-08.
        assert first == pd.Timestamp("2024-01-09 01:00"), first
        assert not e0.loc[:"2024-01-09 00:00"].any()
        # everything from 2024-01-15 on is changed: the past must not move, and day D itself must not leak into its own flag
        store2, _ = _world(d + "/b", n_days=120, listing_day=95, shock_from=106)                  # day 106 = 2024-01-15
        p2 = ib.build_intraday_panel(store2, names, "2024-01-03", "2024-01-25", **kw)
        cutoff = pd.Timestamp("2024-01-16 00:00")                                                  # day 01-15 is judged by days <= 01-14
        assert (p.eligible.loc[:cutoff].to_numpy() == p2.eligible.loc[:cutoff].to_numpy()).all()
        later = slice(cutoff + pd.Timedelta(days=4), None)
        assert (p.eligible.loc[later].to_numpy() != p2.eligible.loc[later].to_numpy()).any()      # and the future does change once the collapse is visible
        print(f"I3 first eligible bar {first} = listing day + 5 whole days; altering 01-15 onward leaves everything up to 01-16 unchanged  PASS")


# ---------------------------------------------------------------------------------------------------- I4
def _direct_panel(n=63_000, k=12, seed=7, bar_min=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 00:05", periods=n, freq=f"{bar_min}min")
    cols = [f"S{i:02d}" for i in range(k)]
    r = pd.DataFrame(rng.normal(0, 0.002, (n, k)), index=idx, columns=cols)
    close = 100 * np.exp(r.cumsum())
    p = q.Panel(close=close, eligible=pd.DataFrame(True, index=idx, columns=cols), volume=pd.DataFrame(1e6, index=idx, columns=cols) / close,
                market="TEST", entry_lag=1, periods_per_year=int(365 * 1440 / bar_min))
    p.meta = {"bar_minutes": bar_min}
    return p, r


def test_latency_known_answer():
    p, r = _direct_panel()
    oracle = r.shift(-2)                                   # the return of bar t+2, the one a lag-1 entry earns
    sw = ib.latency_sweep(p, oracle, lags=(1, 2, 3, 6), hold=1)
    s = sw.set_index("lag_bars")["sharpe"]
    assert s[1] > 50 and abs(s[2]) < 5 and abs(s[3]) < 5 and abs(s[6]) < 5, s.to_dict()
    assert list(sw["delay_minutes"]) == [5, 10, 15, 30]
    blend = r.shift(-2) + 0.7 * r.shift(-3)               # knows bar t+2 fully and bar t+3 partly: the edge decays, it does not vanish at once
    sw2 = ib.latency_sweep(p, blend, lags=(1, 2, 3), hold=1).set_index("lag_bars")["sharpe"]
    assert sw2[1] > sw2[2] > 5 and abs(sw2[3]) < 5, sw2.to_dict()
    both = r.shift(-2) + r.shift(-3)                       # knows both bars equally: lag 1 and lag 2 are each perfectly informed
    sw3 = ib.latency_sweep(p, both, lags=(1, 2, 3), hold=1).set_index("lag_bars")["sharpe"]
    assert sw3[1] > 100 and sw3[2] > 100 and abs(sw3[3]) < 5, sw3.to_dict()
    print(f"I4 knows-one-bar-ahead signal: Sharpe {s[1]:.0f} at lag 1, {s[2]:.1f} at lag 2; a signal that knows bar t+2 fully and t+3 partly decays {sw2[1]:.0f} -> {sw2[2]:.0f} -> {sw2[3]:.1f}  PASS")


# ---------------------------------------------------------------------------------------------------- I5
def test_funding_alignment_and_payment():
    with tempfile.TemporaryDirectory(prefix="intratest-") as d:
        store, names = _world(d, n_days=100, start="2023-11-01")
        ev_times = pd.date_range("2024-01-06 00:00", "2024-01-14 16:00", freq="8h")                      # settlements at 00, 08, 16
        jitter = (ev_times + pd.Timedelta(milliseconds=3))
        ms = (jitter.astype("datetime64[ns]").tz_localize("UTC").asi8 // 10**6)
        frame = pd.DataFrame({"calc_time": ms, "rate": 0.0005})
        events = {s: ib.funding_series([frame]) for s in names}                                           # the 3 ms stamp is rounded away
        assert events[names[0]].index.equals(pd.DatetimeIndex(ev_times).astype('datetime64[ns]'))
        for bar, expect_shift in (("1h", 0), ("15min", 0), ("3h", 1)):                                    # 08:00 is a bar end for 1h/15min, not for 3h (-> 09:00)
            p = ib.build_intraday_panel(store, names, "2024-01-05", "2024-01-15", bar=bar, adv_window=3, min_age_days=5, min_adv_usd=1e6,
                                        funding_events=events)
            row = p.funding[names[0]]
            m = p.meta["bar_minutes"]
            for t in ev_times:
                end = t if (t.hour * 60 + t.minute) % m == 0 else t.ceil(f"{m}min")
                assert abs(row.loc[end] - 0.0005) < 1e-15, (bar, t, end)
            assert abs(row.sum() - 0.0005 * len(ev_times)) < 1e-12, row.sum()
        # payment: flat prices, so any net return is funding
        p = ib.build_intraday_panel(store, names, "2024-01-05", "2024-01-15", bar="1h", adv_window=3, min_age_days=5, min_adv_usd=1e6, funding_events=events)
        flat = replace(p, close=p.close.ffill().iloc[[0]].reindex(p.close.index).ffill().bfill() * 0 + 100.0)
        W = pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
        for sgn, name in ((1.0, "long"), (-1.0, "short"), (0.0, "flat")):
            W[names[0]] = sgn
            r = backtest_weights(flat, W, benchmark=None, check_universe=False)
            f = flat.funding[names[0]].to_numpy()
            lag = flat.entry_lag
            expect = -sgn * sum(f[rr] for rr in range(len(f)) if 0 <= rr - lag - 1 < len(r.net_returns))
            assert abs(r.net_returns.sum() - expect) < 1e-12, (name, r.net_returns.sum(), expect)
            assert sgn != 0.0 or r.net_returns.abs().sum() == 0.0
        print("I5 3 ms stamps land on the bar ending at the settlement (3h bars: the next end); long pays, short receives, flat pays nothing  PASS")


# ---------------------------------------------------------------------------------------------------- I6, I11
def test_periods_per_year_and_guards():
    with tempfile.TemporaryDirectory(prefix="intratest-") as d:
        store, names = _world(d, n_days=100, start="2023-11-01")
        for bar, ppy in (("1min", 525600), ("5min", 105120), ("15min", 35040), ("1h", 8760)):
            if bar == "1min":
                p = ib.build_intraday_panel(store, names[:12], "2024-01-05", "2024-01-06", bar=bar, adv_window=3, min_age_days=5, min_adv_usd=1e6, funding=False)
            else:
                p = ib.build_intraday_panel(store, names, "2024-01-05", "2024-01-08", bar=bar, adv_window=3, min_age_days=5, min_adv_usd=1e6, funding=False)
            assert p.periods_per_year == ppy, (bar, p.periods_per_year)
        try:
            ib.build_intraday_panel(store, names, "2024-01-05", "2024-01-08", bar="7min", funding=False)
        except ValueError as e:
            assert "divides a day" in str(e)
        else:
            raise AssertionError("7min must be refused")
        try:
            ib.build_intraday_panel(store, names, "2024-01-05", "2024-01-08", bar="1h", entry_lag=0, funding=False)
        except ValueError as e:
            assert "at least 1 bar" in str(e)
        else:
            raise AssertionError("entry_lag 0 must be refused")
    p, r = _direct_panel(n=3000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = ib.backtest_intraday(p, r.shift(-2))
    assert np.isnan(res.metrics["Sharpe"]) and any("observations" in str(x.message) and "years" in str(x.message) for x in w)
    print("I6/I11 periods per year 525600, 105120, 35040, 8760; a 7-minute bar and entry_lag 0 are refused; a short sample warns  PASS")


# ---------------------------------------------------------------------------------------------------- I7
def test_breakeven_cost():
    p, r = _direct_panel(n=63_000, seed=9)
    sig = r.shift(-2) * 0.02 + pd.DataFrame(np.random.default_rng(3).normal(0, 0.002, r.shape), index=r.index, columns=r.columns)  # a weak edge
    be = ib.breakeven_cost(p, sig, hold=1)
    c = be["breakeven_one_way_bp"]
    assert c > 0, be
    at = ib.backtest_intraday(p, sig, one_way_bp=c, hold=1).net_returns.mean()
    assert abs(at) < 1e-12 * 1 + abs(be["net_mean_at_0bp"]) * 1e-6, (at, be)
    assert ib.backtest_intraday(p, sig, one_way_bp=0.5 * c, hold=1).net_returns.mean() > 0 > ib.backtest_intraday(p, sig, one_way_bp=2 * c, hold=1).net_returns.mean()
    print(f"I7 break-even one-way cost {c:.3f} bp: net mean at that cost is {at:.1e}; positive at half, negative at double  PASS")


# ---------------------------------------------------------------------------------------------------- I8
def test_memory_guard():
    with tempfile.TemporaryDirectory(prefix="intratest-") as d:
        store, names = _world(d, n_days=100, start="2023-11-01")
        try:
            ib.build_intraday_panel(store, names, "2024-01-05", "2024-01-31", bar="1min", max_gb=0.001, funding=False)
        except ValueError as e:
            msg = str(e)
            assert "GB" in msg and "longer bar" in msg, msg
        else:
            raise AssertionError("the memory guard did not fire")
    assert 3.0 < ib.estimate_memory(525_600, 40) < 4.0 and ib.estimate_memory(105_120, 40) < 1.0     # a year of 1-minute bars for 40 contracts is over the 3 GB default
    print("I8 a tiny limit refuses with a bar length that fits; a year of 1-minute bars for 40 contracts (about 3.4 GB) is over the default limit, 5-minute bars (0.7 GB) are not  PASS")


# ---------------------------------------------------------------------------------------------------- I9
def test_download_resumable_and_missing():
    calls = []
    month_df = {f"2024-0{k}": _minutes(f"2024-0{k}-01", 1440 * (29 if k == 2 else 31 if k in (1, 3) else 30), seed=k) for k in (1, 2, 3)}

    def getter(key):
        calls.append(key)
        for per, df in month_df.items():
            if f"monthly/klines/AAAUSDT/1m/AAAUSDT-1m-{per}.zip" in key:
                return _zip_bytes(df)
        if "daily/klines/AAAUSDT/1m/AAAUSDT-1m-2024-04-" in key:
            day = key[-14:-4]
            return _zip_bytes(_minutes(day, 1440, seed=int(day[-2:])))
        return None
    with tempfile.TemporaryDirectory(prefix="intratest-") as d:
        store = ArchiveStore(Path(d))
        r1 = ib.fetch_minutes(store, ["AAAUSDT", "ZZZUSDT"], "2024-02-10", "2024-04-03", warmup_days=20, getter=getter, today="2024-04-04")
        monthly = [k for k in calls if "/monthly/" in k]
        daily = [k for k in calls if "/daily/" in k]
        assert len([k for k in monthly if "AAAUSDT" in k]) == 3 and len([k for k in daily if "AAAUSDT" in k]) == 3, (monthly, daily)   # Jan, Feb, Mar; Apr 1 to 3
        assert not any("2024-04.zip" in k for k in monthly), "the partial month must come from daily files"
        assert r1["missing"] >= 3                                         # ZZZ: 3 months and 3 days absent
        n = len(calls)
        r2 = ib.fetch_minutes(store, ["AAAUSDT", "ZZZUSDT"], "2024-02-10", "2024-04-03", warmup_days=20, getter=getter, today="2024-04-04")
        assert len(calls) == n and r2["months"] == 0 and r2["days"] == 0 and r2["cached"] > 0
        df = ib.load_minutes(store, "AAAUSDT", "2024-02-10", "2024-04-03")
        assert df.index.is_monotonic_increasing and not df.index.has_duplicates
        bad = lambda key: b"not a zip"
        try:
            ib.fetch_minutes(ArchiveStore(Path(d) / "x"), ["AAAUSDT"], "2024-02-10", "2024-02-20", warmup_days=0, getter=bad, today="2024-04-04")
        except Exception:
            pass
        else:
            raise AssertionError("a file that does not parse must raise")
        assert not list((Path(d) / "x" / "minutes" / "AAAUSDT").glob("*.pkl")), "nothing may be cached for a bad file"
    print("I9 monthly zips for complete months, daily zips for the partial one; a rerun makes no request; 404s recorded; a bad file raises and caches nothing  PASS")


# ---------------------------------------------------------------------------------------------------- I10
def test_halts_and_zero_volume():
    with tempfile.TemporaryDirectory(prefix="intratest-") as d:
        store, names = _world(d, n_days=130, halt_after=110)                      # C01 stops after day 110 (2024-01-19) and prints flat bars
        # an interior stretch in C02 with no trades for 30 minutes (2024-01-25 12:00): the price must stay
        m = ib.load_minutes(store, "C02USDT", "2023-10-01", "2024-02-07")
        i0 = m.index[116 * 1440 + 720]
        z = (m.index >= i0) & (m.index < i0 + pd.Timedelta(minutes=30))
        m.loc[z, ["open", "high", "low", "close"]] = m["close"].shift(1).loc[z].iloc[0]
        m.loc[z, ["volume", "quote_volume"]] = 0.0
        _write_months(store, "C02USDT", m)
        p = ib.build_intraday_panel(store, names, "2024-01-05", "2024-02-05", bar="15min", adv_window=3, min_age_days=5, min_adv_usd=1e6, funding=False)
        cut = pd.Timestamp("2023-10-01") + pd.Timedelta(days=110)
        c1 = p.close["C01USDT"]
        assert c1.dropna().index.max() <= cut + pd.Timedelta(minutes=15), c1.dropna().index.max()
        assert p.delist_after["C01USDT"].sum() == 1 and p.delist_after["C01USDT"].idxmax() == c1.dropna().index.max()
        assert not p.eligible["C01USDT"].iloc[-100:].any()
        c2 = p.close["C02USDT"]
        assert c2.loc[i0:i0 + pd.Timedelta(minutes=45)].notna().all(), "interior zero-volume bars must keep their price"
        assert p.meta["zero_volume_bars_kept"] > 0 and p.meta["delisted_in_panel"] == 1, p.meta
        print("I10 the terminal zero-volume tail is cut and the last real bar flagged; an interior no-trade stretch keeps its price (counted)  PASS")


# ---------------------------------------------------------------------------------------------------- I12
def test_one_way_cost_unit_is_pinned():
    """Flat prices, so any return is cost. The two legs are fully replaced every bar: each leg trades out 1 and in 1 (|change| 2), so
    the bar pays 4 x the one-way cost. Break-even alone would not catch a wrong conversion, because it uses the same one on both sides."""
    n, k = 63_000, 10
    idx = pd.date_range("2024-01-01 00:05", periods=n, freq="5min")
    cols = [f"S{i}" for i in range(k)]
    p = q.Panel(close=pd.DataFrame(100.0, index=idx, columns=cols), eligible=pd.DataFrame(True, index=idx, columns=cols),
                volume=pd.DataFrame(1e4, index=idx, columns=cols), market="TEST", entry_lag=1, periods_per_year=105120)
    p.meta = {"bar_minutes": 5}
    even = np.arange(k, dtype=float)
    F = pd.DataFrame(np.where((np.arange(n) % 2 == 0)[:, None], even[None, :], even[::-1][None, :]), index=idx, columns=cols)
    for c in (1.0, 3.0, 7.5):
        r = ib.backtest_intraday(p, F, one_way_bp=c, hold=1)
        net = r.net_returns.to_numpy()
        steady = net[2:]
        assert np.allclose(steady, -4.0 * c / 1e4, atol=1e-15), (c, steady[:3])
        assert abs(r.metrics["turnover_daily"] - 2.0) < 1e-3                  # one-way turnover per bar: half of 4
    be = ib.breakeven_cost(p, F)
    assert not np.isfinite(be["breakeven_one_way_bp"]) or be["breakeven_one_way_bp"] <= 0                                  # no edge: never pays
    print("I12 a full rotation of both legs costs exactly 4 x the one-way cost per bar (1, 3 and 7.5 bp)  PASS")


if __name__ == "__main__":
    test_parse_units_labels_duplicates()
    test_aggregation_matches_loops()
    test_point_in_time_eligibility()
    test_latency_known_answer()
    test_funding_alignment_and_payment()
    test_periods_per_year_and_guards()
    test_breakeven_cost()
    test_memory_guard()
    test_download_resumable_and_missing()
    test_halts_and_zero_volume()
    test_one_way_cost_unit_is_pinned()
    print("intraday tests: all passed")
