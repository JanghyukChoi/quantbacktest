"""Intraday bars for USDT perpetuals from the Binance public archive, as a point-in-time Panel.

Read `docs/intraday_design.md` first. What this layer can say: how fast an edge decays with latency, and at what cost it stops
paying. What it cannot say: whether the signal can be traded. The archive has bars, not a book, so there is no queue position, no
partial fill and no impact below the bar.

Time convention. Rows are indexed by bar **end** time (UTC, naive). A row holds only what was known at that instant, so a signal
from row t is entered at row t + lag with lag >= 1.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.panel import Panel
from ..portfolio import backtest_portfolio
from . import binance_archive as ba

COLS = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]
MIN_PER_DAY = 1440


# ------------------------------------------------------------------------------------------------------ parsing
def parse_minute_zip(blob: bytes) -> pd.DataFrame:
    """One archive file of 1-minute klines -> frame indexed by the bar **end** time (open time + 1 minute).
    Millisecond and microsecond timestamps are told apart per row; header rows and duplicates are dropped."""
    raw = ba._read_klines_zip(blob)
    if raw.empty:
        return pd.DataFrame(columns=COLS)
    ts = raw["open_time"].astype("int64").to_numpy()
    ms = np.where(ts > 10**14, ts // 1000, ts)
    idx = (pd.to_datetime(ms, unit="ms", utc=True).tz_localize(None).astype("datetime64[ns]") + pd.Timedelta(minutes=1))   # ns in every pandas
    df = raw[COLS].apply(pd.to_numeric, errors="coerce")
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["close"])


def _bar_minutes(bar: str) -> int:
    m = int(pd.Timedelta(bar) / pd.Timedelta(minutes=1))
    if m < 1 or MIN_PER_DAY % m:
        raise ValueError(f"bar {bar!r} must be a whole number of minutes that divides a day (1min, 5min, 15min, 1h, ...)")
    return m


def aggregate_bars(df1m: pd.DataFrame, bar: str) -> pd.DataFrame:
    """1-minute frame (end-time index) -> bars labelled by their end. Open is the first minute's, close the last, high and low the
    extremes, volumes summed; `n_min` is how many minutes the bar holds. A last bar that would end after the data ends is dropped."""
    m = _bar_minutes(bar)
    if df1m.empty:
        return pd.DataFrame(columns=COLS + ["n_min"])
    if m == 1:
        out = df1m.copy()
        out["n_min"] = 1
        return out
    opn = df1m.copy()
    opn.index = opn.index - pd.Timedelta(minutes=1)                      # back to open time so bars align to the clock
    g = opn.resample(f"{m}min", label="left", closed="left", origin="epoch")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum(min_count=1), "quote_volume": g["quote_volume"].sum(min_count=1),
                        "trades": g["trades"].sum(min_count=1), "n_min": g["close"].count()})
    out = out[out["n_min"] > 0]
    out.index = out.index + pd.Timedelta(minutes=m)                      # label by end
    return out[out.index <= df1m.index.max()]


# ------------------------------------------------------------------------------------------------------ download
def _month_keys(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Period]:
    return list(pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"))


def _paths(store, sym: str) -> Path:
    d = store.dir / "minutes" / sym
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_minutes(store, symbols, start, end, *, workers: int = 6, warmup_days: int = 91, getter=None, today=None) -> dict:
    """Download 1-minute klines for `symbols` from `start - warmup_days` to `end`. Complete months come from the monthly archive,
    the current partial month from the daily archive. Resumable: a month or day already cached, or recorded as missing (404), is
    not requested again. `getter(key) -> bytes | None` can be injected for tests; `today` is the current UTC date."""
    get = getter or (lambda key: ba._get(f"{ba.DL_URL}/{key}"))
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.utcnow().tz_localize(None).normalize()
    lo = pd.Timestamp(start).normalize() - pd.Timedelta(days=warmup_days)
    hi = min(pd.Timestamp(end).normalize(), today - pd.Timedelta(days=1))
    cur = today.to_period("M")
    stats = {"months": 0, "days": 0, "missing": 0, "cached": 0}

    def one(sym: str) -> dict:
        s = {"months": 0, "days": 0, "missing": 0, "cached": 0}
        d = _paths(store, sym)
        for per in _month_keys(lo, hi):
            if per < cur:                                                  # a complete month
                f, none = d / f"{per}.pkl", d / f"{per}.none"
                if f.exists() or none.exists():
                    s["cached"] += 1
                    continue
                blob = get(f"data/futures/um/monthly/klines/{sym}/1m/{sym}-1m-{per}.zip")
                if blob is None:
                    none.write_text("")
                    s["missing"] += 1
                    continue
                parse_minute_zip(blob).to_pickle(f)                       # parsed fully first: a bad file raises and caches nothing
                s["months"] += 1
            else:                                                          # the partial month, one file per finished day
                for day in pd.date_range(max(lo, per.start_time), min(hi, per.end_time.normalize())):
                    f, none = d / f"d{day:%Y-%m-%d}.pkl", d / f"d{day:%Y-%m-%d}.none"
                    if f.exists() or none.exists():
                        s["cached"] += 1
                        continue
                    blob = get(f"data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{day:%Y-%m-%d}.zip")
                    if blob is None:
                        none.write_text("")
                        s["missing"] += 1
                        continue
                    parse_minute_zip(blob).to_pickle(f)
                    s["days"] += 1
        return s

    with cf.ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, list(symbols)):
            for k in stats:
                stats[k] += r[k]
    return stats


def load_minutes(store, sym: str, start, end) -> pd.DataFrame:
    """Cached 1-minute frame for `sym` between `start` (inclusive) and `end` (inclusive day), end-time index."""
    d = _paths(store, sym)
    parts = [pd.read_pickle(f) for f in sorted(d.glob("*.pkl"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df.index > pd.Timestamp(start)) & (df.index <= pd.Timestamp(end).normalize() + pd.Timedelta(days=1))]


def parse_funding_zip(blob: bytes) -> pd.DataFrame:
    """One monthly fundingRate archive file -> DataFrame[calc_time (ms or us), rate]."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        dd = pd.read_csv(z.open(z.namelist()[0]), dtype=str)
    if dd.shape[1] < 3:
        return pd.DataFrame(columns=["calc_time", "rate"])
    dd.columns = ["calc_time", "interval", "rate"][: dd.shape[1]] + list(dd.columns[3:])
    return dd[["calc_time", "rate"]]


def funding_series(frames: list[pd.DataFrame]) -> pd.Series:
    """Settlements with timestamps rounded to the nearest minute (the archive stamps them a few milliseconds late)."""
    out = pd.concat(frames) if frames else pd.DataFrame(columns=["calc_time", "rate"])
    t = pd.to_numeric(out["calc_time"], errors="coerce")
    r = pd.to_numeric(out["rate"], errors="coerce")
    ok = t.notna() & r.notna()
    t, r = t[ok].astype("int64").to_numpy(), r[ok].to_numpy(float)
    ms = np.where(t > 10**14, t // 1000, t)
    idx = pd.to_datetime(ms, unit="ms", utc=True).tz_localize(None).astype("datetime64[ns]").round("min")
    return pd.Series(r, index=idx, dtype=float).groupby(level=0).sum().sort_index()


def fetch_funding_events(store, sym: str, *, lister=None, getter=None) -> pd.Series:
    """Funding settlements with their instants, cached. The daily store sums them per day and loses the instant, which an
    intraday simulation needs. `lister(sym) -> keys` and `getter(key) -> bytes | None` can be injected for tests."""
    f = store.dir / "funding_events" / f"{sym}.pkl"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return pd.read_pickle(f)
    keys = lister(sym) if lister else store._months(sym, "fundingRate")
    get = getter or (lambda key: ba._get(f"{ba.DL_URL}/{key}"))
    frames = [parse_funding_zip(blob) for k in keys if (blob := get(k))]
    s = funding_series(frames)
    s.to_pickle(f)
    return s


# ------------------------------------------------------------------------------------------------------ the panel
def estimate_memory(n_bars: int, n_symbols: int, n_matrices: int = 20) -> float:
    """Rough peak gigabytes: `backtest_portfolio` peaked at about 14 (bar x contract) float64 matrices when measured on a panel
    with close, eligible and volume (tests/memprobe in the design notes); a panel from `build_intraday_panel` also carries open,
    high, low, funding and delisting flags, so 20 is used. Measure again if you change the engine."""
    return n_bars * n_symbols * 8 * n_matrices / 1e9


def build_intraday_panel(store, symbols, start, end, *, bar: str = "5min", adv_window: int = 30, min_age_days: int = 60,
                         min_adv_usd: float = 5e6, top_n: int | None = None, entry_lag: int = 1, max_gb: float = 3.0,
                         funding: bool = True, funding_events: dict | None = None) -> Panel:
    """Point-in-time intraday panel. Eligibility for every bar of day D uses only days up to D - 1. Download first with
    `fetch_minutes(store, symbols, start, end)`, which also loads the warm-up days."""
    symbols = list(symbols)
    if entry_lag < 1:
        raise ValueError("entry_lag must be at least 1 bar: a bar's close is only known at its end, so it cannot be traded on")
    m = _bar_minutes(bar)
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    warm = max(adv_window, min_age_days) + 1
    lo = start - pd.Timedelta(days=warm)
    grid_end = end + pd.Timedelta(days=1)
    grid = pd.date_range(start + pd.Timedelta(minutes=m), grid_end, freq=f"{m}min").astype("datetime64[ns]")   # pandas 3 defaults to us
    mem = estimate_memory(len(grid), len(symbols))
    if mem > max_gb:
        fit = [b for b in ("1min", "5min", "15min", "30min", "1h", "2h", "4h") if _bar_minutes(b) >= m and
               estimate_memory(len(grid) * m // _bar_minutes(b), len(symbols)) <= max_gb]
        raise ValueError(f"about {mem:.1f} GB needed for {len(grid):,} bars x {len(symbols)} contracts (limit {max_gb}); "
                         f"use a longer bar ({fit[0] if fit else 'fewer contracts or a shorter period'}), fewer contracts or a shorter period")
    bars: dict[str, pd.DataFrame] = {}
    for s in symbols:
        d = _paths(store, s)
        missing = [p for p in _month_keys(lo, min(end, pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)))
                   if p < pd.Timestamp.utcnow().tz_localize(None).to_period("M")
                   and not (d / f"{p}.pkl").exists() and not (d / f"{p}.none").exists()]
        if missing:
            raise ValueError(f"{s}: months not downloaded {[str(p) for p in missing[:4]]}; run fetch_minutes(store, symbols, start, end) "
                             f"first (it loads {warm} warm-up days before the start)")
        raw = load_minutes(store, s, lo, end)
        if raw.empty:
            continue
        b = aggregate_bars(raw, bar)
        bars[s] = b
    if not bars:
        raise ValueError("no data for any symbol")
    cols = sorted(bars)

    # ---- terminal zero-volume tail (a halted or delisted contract keeps printing flat bars): drop only that tail ------------
    zero_interior = 0
    for s in cols:
        b = bars[s]
        live = np.flatnonzero((b["quote_volume"] > 0).to_numpy())
        if len(live) == 0:
            bars[s] = b.iloc[0:0]
            continue
        zero_interior += int((b["quote_volume"].iloc[: live[-1] + 1] <= 0).sum())
        bars[s] = b.iloc[: live[-1] + 1]
    cols = [s for s in cols if len(bars[s])]

    # ---- daily liquidity, then eligibility from the PREVIOUS day -----------------------------------------------------
    day_of = lambda ix: (ix - pd.Timedelta(minutes=1)).floor("D")
    dq = pd.DataFrame({s: bars[s]["quote_volume"].groupby(day_of(bars[s].index)).sum() for s in cols})
    dq = dq.reindex(pd.date_range(dq.index.min(), max(dq.index.max(), end), freq="D"))
    has = dq.fillna(0) > 0
    age = has.cumsum()
    adv = dq.where(has).rolling(adv_window, min_periods=adv_window).median()
    ok_day = (age >= min_age_days) & (adv >= min_adv_usd)
    if top_n:
        rank = adv.where(ok_day).rank(axis=1, ascending=False, method="first")
        ok_day = ok_day & (rank <= top_n)
    ok_for_day = ok_day.shift(1, fill_value=False)                        # day D is judged by days <= D - 1

    def mat(col: str) -> pd.DataFrame:
        return pd.DataFrame({s: bars[s][col] for s in cols}).reindex(grid)

    close, high, low, opn = mat("close"), mat("high"), mat("low"), mat("open")
    qv = mat("quote_volume")
    elig = ok_for_day.reindex(day_of(grid)).fillna(False).astype(bool).to_numpy()
    eligible = pd.DataFrame(elig, index=grid, columns=cols) & close.notna()

    # ---- delisting: last real bar of a contract that ended well before the panel end ---------------------------------------
    last = close.apply(lambda c: c.last_valid_index())
    da = pd.DataFrame(False, index=grid, columns=cols)
    for s in cols:
        if last[s] is not None and last[s] < grid[-1] - pd.Timedelta(days=3):
            da.loc[last[s], s] = True

    # ---- funding at the settlement instant ----------------------------------------------------------------------------
    fund, n_events = None, 0
    if funding:
        F = np.zeros((len(grid), len(cols)))
        for j, s in enumerate(cols):
            ev = funding_events[s] if funding_events is not None and s in funding_events else fetch_funding_events(store, s)
            ev = ev[(ev.index > grid[0] - pd.Timedelta(minutes=m)) & (ev.index <= grid[-1])]
            if ev.empty:
                continue
            pos = grid.searchsorted(ev.index.to_numpy(), side="left")       # first bar whose end is at or after the settlement
            ok = pos < len(grid)
            np.add.at(F, (pos[ok], np.full(int(ok.sum()), j)), ev.to_numpy(float)[ok])
            n_events += int(ok.sum())
        fund = pd.DataFrame(F, index=grid, columns=cols)
    panel = Panel(close=close, eligible=eligible, open=opn, high=high, low=low, volume=qv / close, market="CRYPTO-INTRADAY",
                  entry_lag=entry_lag, periods_per_year=int(365 * MIN_PER_DAY / m), funding=fund, delist_after=da)
    n = panel.eligible.sum(axis=1)
    panel.meta = {"bar": bar, "bar_minutes": m, "symbols_used": len(cols), "bars": len(grid), "partial_bars": int(sum((bars[s]["n_min"] < m).sum() for s in cols)),
                  "zero_volume_bars_kept": zero_interior, "delisted_in_panel": int(da.values.any(axis=0).sum()),
                  "eligible_median": float(n[n > 0].median()) if (n > 0).any() else 0.0, "estimated_gb": round(mem, 2),
                  "funding_events": n_events}
    return panel


# ------------------------------------------------------------------------------------------------------ analysis
def backtest_intraday(panel: Panel, factor: pd.DataFrame, *, one_way_bp: float = 0.0, long_q: float = 0.2, short_q: float | None = 0.2,
                      hold: int = 1, **kw):
    """`backtest_portfolio` with the cost given as **one-way** basis points (the engine's scalar is a round trip), no grid and no
    benchmark. `hold` is in bars."""
    if panel.entry_lag < 1:
        raise ValueError("an intraday panel needs entry_lag >= 1: a bar's close is only known when the bar ends, so it cannot be traded on")
    kw.setdefault("benchmark", None)
    kw.setdefault("grid", False)
    return backtest_portfolio(panel, factor, long_q=long_q, short_q=short_q, hold=hold, spread_bp=2.0 * one_way_bp, **kw)


def latency_sweep(panel: Panel, factor: pd.DataFrame, lags=(1, 2, 5, 15), *, one_way_bp: float = 0.0, **kw) -> pd.DataFrame:
    """The same portfolio entered `lag` bars after the signal. A real edge decays smoothly with the delay."""
    m = panel.meta.get("bar_minutes", 1)
    rows = []
    for k in lags:
        r = backtest_intraday(replace(panel, entry_lag=int(k)), factor, one_way_bp=one_way_bp, **kw)
        met = r.metrics
        rows.append({"lag_bars": int(k), "delay_minutes": int(k) * m, "sharpe": met["Sharpe"], "cagr": met["CAGR"],
                     "gross_cagr": met["gross_CAGR"], "turnover_per_bar": met["turnover_daily"]})
    return pd.DataFrame(rows)


def breakeven_cost(panel: Panel, factor: pd.DataFrame, **kw) -> dict:
    """One-way cost in bp at which the net mean return is zero. Net return is linear in the cost per unit traded, so two runs
    give it exactly: c* = mean(net at 0 bp) / mean(net at 0 bp - net at 1 bp)."""
    kw.pop("one_way_bp", None)
    r0 = backtest_intraday(panel, factor, one_way_bp=0.0, **kw)
    r1 = backtest_intraday(panel, factor, one_way_bp=1.0, **kw)
    n0, n1 = r0.net_returns, r1.net_returns
    per_bp = float((n0 - n1).mean())
    mean0 = float(n0.mean())
    c = mean0 / per_bp if per_bp > 0 else float("nan")
    return {"breakeven_one_way_bp": c, "net_mean_at_0bp": mean0, "cost_per_bp": per_bp, "gross_sharpe": r0.metrics["Sharpe"]}
