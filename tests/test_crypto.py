"""Crypto bias controls, tested offline on a fake archive where the right answer is known.

C1 survivorship   delisted contracts are in the universe, and `survivors_only` removes exactly them
C2 PIT eligibility  changing data after day t never changes eligibility on or before day t (no look-ahead)
C3 stale tail     zero-volume frozen bars are dropped and the last real bar is marked as the delisting
C4 min age        a new listing is not eligible until it has `min_age_days` bars
C5 funding        constant funding: a long pays it, a short receives it, a balanced long-short roughly nets out
C6 delisting      a held contract that delists takes `delist_return` (default: silently flat, flagged as an event)
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import quantbt as q
from quantbt.crypto.binance_archive import ArchiveStore
from quantbt.crypto.panel import build_panel

DAYS = pd.date_range("2021-01-01", periods=420, freq="D")
DEAD = {"DEAD1USDT": 250, "DEAD2USDT": 300, "DEAD3USDT": 330}      # symbol -> last real bar index


def _fake_store(tmp: Path, *, mutate_after: int | None = None) -> tuple[ArchiveStore, set]:
    rng = np.random.default_rng(11)
    store = ArchiveStore(tmp)
    names = [f"C{i:02d}USDT" for i in range(24)] + ["BTCUSDT"] + list(DEAD) + ["NEWUSDT"]
    (tmp / "symbols.json").write_text(json.dumps(sorted(names)))
    live = set()
    for s in names:
        start = 300 if s == "NEWUSDT" else 0
        end = DEAD.get(s, len(DAYS) - 1)
        idx = DAYS[start:end + 1]
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx))))
        qv = rng.lognormal(18, 0.3, len(idx))                       # about 6.6e7 USDT a day, above the 5e6 floor
        if mutate_after is not None:
            qv = qv.copy(); qv[mutate_after - start:] *= 1e-3        # future volume collapses (must not touch the past)
        df = pd.DataFrame({"open": close, "high": close * 1.02, "low": close * 0.98, "close": close,
                           "volume": qv / close, "quote_volume": qv, "trades": 1000}, index=idx)
        if s in DEAD:                                                # frozen, zero-volume tail like a halted contract
            tail = pd.DataFrame({"open": close[-1], "high": close[-1], "low": close[-1], "close": close[-1], "volume": 0.0,
                                 "quote_volume": 0.0, "trades": 0}, index=pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=5))
            df = pd.concat([df, tail])
        df.to_pickle(tmp / "daily" / f"{s}.pkl")
        pd.Series(0.0003, index=df.index).to_pickle(tmp / "funding" / f"{s}.pkl")
        if s not in DEAD:
            live.add(s)
    return store, live


def test_survivorship_and_stale_tail_and_min_age():
    with tempfile.TemporaryDirectory() as d:
        store, live = _fake_store(Path(d))
        full = build_panel(store, start="2021-03-01", min_adv_usd=5e6, min_age_days=60, live=live)
        surv = build_panel(store, start="2021-03-01", min_adv_usd=5e6, min_age_days=60, survivors_only=True, live=live)
        assert set(DEAD) <= set(full.tickers) and not (set(DEAD) & set(surv.tickers))            # C1
        assert full.meta["delisted_in_panel"] == 3 and surv.meta["delisted_in_panel"] == 0
        for s, last in DEAD.items():                                                             # C3
            lv = full.close[s].last_valid_index()
            assert lv == DAYS[last], (s, lv)                                                     # frozen tail removed
            assert bool(full.delist_after.loc[lv, s]) and full.delist_after[s].sum() == 1
        e = full.eligible["NEWUSDT"]                                                             # C4
        first = e[e].index[0]
        assert (first - DAYS[300]).days == 59, first                                             # 60th bar is the first eligible
        print("C1 survivorship, C3 stale tail, C4 min age  PASS")


def test_pit_eligibility_has_no_lookahead():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a, live = _fake_store(Path(d1))
        b, _ = _fake_store(Path(d2), mutate_after=300)
        pa = build_panel(a, start="2021-02-01", live=live)
        pb = build_panel(b, start="2021-02-01", live=live)
        cut = DAYS[299]
        ea, eb = pa.eligible.loc[:cut], pb.eligible.loc[:cut]
        assert (ea.reindex(columns=eb.columns, fill_value=False).values == eb.values).all()      # C2
        assert not (pa.eligible.loc[DAYS[300:]].values == pb.eligible.loc[DAYS[300:]].values).all()   # sanity: the future did change
        print("C2 PIT eligibility (future changes leave the past untouched)  PASS")


def _panel(fund: float, n=30, T=300, delist_at: int | None = None) -> q.Panel:
    rng = np.random.default_rng(2)
    dates = pd.date_range("2022-01-01", periods=T, freq="D")
    tick = [f"T{i:02d}" for i in range(n)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, n)), axis=0)), index=dates, columns=tick)
    vol = pd.DataFrame(rng.lognormal(12, 0.3, (T, n)), index=dates, columns=tick)
    el = pd.DataFrame(True, index=dates, columns=tick)
    funding = pd.DataFrame(fund, index=dates, columns=tick)
    da = pd.DataFrame(False, index=dates, columns=tick)
    if delist_at is not None:
        close.iloc[delist_at + 1:, 0] = np.nan                       # contract ends after bar `delist_at`
        da.iloc[delist_at, 0] = True
    return q.Panel(close=close, eligible=el, volume=vol, market="CRYPTO", periods_per_year=365, funding=funding, delist_after=da)


def test_funding_sign_and_size():
    f = 0.0004
    p = _panel(f)
    fac = pd.DataFrame(np.random.default_rng(4).normal(size=p.close.shape), index=p.dates, columns=p.tickers)
    kw = dict(long_q=0.3, hold=5, spread_bp=0.0, benchmark=None, grid=False)
    lo = q.backtest_portfolio(p, fac, short_q=None, **kw).metrics
    assert abs(lo["funding_annual_bp"] - f * 365 * 1e4) < 1.0, lo["funding_annual_bp"]            # long pays 0.04%/day
    # Short side only: fix the factor (so the legs never change) and charge funding only on the short names.
    n = len(p.tickers)
    static = pd.DataFrame(np.tile(np.arange(n, dtype=float), (len(p.dates), 1)), index=p.dates, columns=p.tickers)
    fshort = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); fshort.iloc[:, : int(0.3 * n)] = f   # bottom 30% = the shorts
    ps = q.Panel(close=p.close, eligible=p.eligible, volume=p.volume, market="CRYPTO", periods_per_year=365, funding=fshort)
    sh = q.backtest_portfolio(ps, static, **kw).metrics
    assert abs(sh["funding_annual_bp"] + f * 365 * 1e4) < 1.0, sh["funding_annual_bp"]            # the short leg RECEIVES f
    ls = q.backtest_portfolio(p, fac, short_q=0.3, **kw).metrics
    assert abs(ls["funding_annual_bp"]) < 5.0, ls["funding_annual_bp"]                            # symmetric legs net out
    off = q.backtest_portfolio(p, fac, short_q=None, funding=False, **kw).metrics
    assert off["funding_annual_bp"] == 0.0
    print(f"C5 funding  long pays {lo['funding_annual_bp']:.0f}bp/yr, short receives {-sh['funding_annual_bp']:.0f}bp/yr, "
          f"long-short {ls['funding_annual_bp']:.1f}bp/yr  PASS")


def test_delisting_event_is_explicit():
    p = _panel(0.0, delist_at=150)
    fac = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    fac["T00"] = 10.0                                                # always the top pick, so it is held into the delisting
    kw = dict(long_q=0.04, short_q=None, hold=1, spread_bp=0.0, benchmark=None, grid=False)
    base = q.backtest_portfolio(p, fac, **kw)
    hit = q.backtest_portfolio(p, fac, delist_return=-0.9, **kw)
    assert base.metrics["delist_events_held"] >= 1                                               # C6: flagged either way
    assert hit.metrics["CAGR"] < base.metrics["CAGR"] - 0.5, (base.metrics["CAGR"], hit.metrics["CAGR"])
    print(f"C6 delisting  silent {base.metrics['CAGR']:+.2f} vs delist_return=-90% {hit.metrics['CAGR']:+.2f}, "
          f"events held {base.metrics['delist_events_held']}  PASS")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    test_survivorship_and_stale_tail_and_min_age()
    test_pit_eligibility_has_no_lookahead()
    test_funding_sign_and_size()
    test_delisting_event_is_explicit()
    print("crypto tests: all passed")
