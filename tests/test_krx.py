"""KRX adapter, tested offline with a fake API whose right answers are known.

K1 survivorship  a stock that was listed on day t but is gone later is in the panel and is flagged as delisted
K2 split         a 50:1 split (raw price falls 98%) leaves the adjusted return unchanged
K3 listing day   the first bar of a new listing has no return, and the stock waits `min_age_days` to be eligible
K4 code reuse    a code that comes back after a long gap becomes a different security
K7 no trades     a day without trades has no open, high or low (NaN, not 0)
K6 suspension   a consolidation during a suspension makes a huge false return: it is listed, kept by default, cut on request
K5 resumable     a second fetch calls the API zero times; a quota error stops cleanly and keeps what was saved
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pitbacktest.adapters import krx

DAYS = pd.bdate_range("2021-01-04", periods=300)


def _row(code, name, close, ref, vol=1_000_000, value=5e9):
    return {"ISU_CD": code, "ISU_NM": name, "MKT_NM": "KOSPI", "TDD_CLSPRC": str(int(close)), "CMPPREVDD_PRC": str(int(close - ref)),
            "TDD_OPNPRC": str(int(close)), "TDD_HGPRC": str(int(close * 1.01)), "TDD_LWPRC": str(int(close * 0.99)),
            "ACC_TRDVOL": str(vol), "ACC_TRDVAL": str(int(value)), "MKTCAP": str(int(close * 1e7)), "LIST_SHRS": "10000000",
            "FLUC_RT": "0", "SECT_TP_NM": "", "BAS_DD": ""}


def _world():
    """date -> rows. Stocks: 15 steady ones, SPLIT (50:1 on day 150), DEAD (last bar day 200), NEW (lists day 100),
    REUSE (alive days 0..40, gone, back from day 240 with another price level)."""
    rng = np.random.default_rng(5)
    px = {f"{(i + 1) * 10:06d}": 10000.0 for i in range(15)}        # 000010, 000020, ... all end in 0 (common shares)
    world = {}
    p_split, p_dead, p_new, p_reuse = 100000.0, 20000.0, 8000.0, 3000.0
    for t, d in enumerate(DAYS):
        rows = []
        for c in list(px):
            new = px[c] * (1 + rng.normal(0, 0.01)); rows.append(_row(c, f"STEADY{c}", new, px[c])); px[c] = new
        r = rng.normal(0, 0.01)
        if t == 150:
            rows.append(_row("111110", "SPLITCO", p_split * (1 + r) / 50, p_split / 50)); p_split = p_split * (1 + r) / 50
        else:
            rows.append(_row("111110", "SPLITCO", p_split * (1 + r), p_split)); p_split *= 1 + r
        if t <= 200:
            nd = p_dead * (1 + rng.normal(0, 0.02)); rows.append(_row("222220", "DEADCO", nd, p_dead)); p_dead = nd
        if t >= 100:
            ref = p_new if t > 100 else p_new * 0.6                        # listing day: reference is the offering price
            nn = p_new * (1 + rng.normal(0, 0.02)); rows.append(_row("333330", "NEWCO", nn, ref)); p_new = nn
        if t <= 40 or t >= 240:
            nr = p_reuse * (1 + rng.normal(0, 0.02)); rows.append(_row("444440", "REUSECO", nr, p_reuse)); p_reuse = nr
        world[d.strftime("%Y%m%d")] = rows
    return world


def test_adapter():
    world = _world()
    calls = []

    def caller(path, date):
        calls.append(date)
        return world.get(date, []) if path.endswith("stk_bydd_trd") else []      # KOSDAQ: empty

    with tempfile.TemporaryDirectory() as d:
        out = krx.fetch_days(DAYS[0], DAYS[-1], d, caller=caller, sleep=0, progress=False)
        n1 = len(calls)
        again = krx.fetch_days(DAYS[0], DAYS[-1], d, caller=caller, sleep=0, progress=False)
        assert out["days"] == 300 and again["skipped"] == 300 and len(calls) == n1                 # K5 resumable
        p = krx.build_krx_panel(d, min_age_days=30, min_value_krw=1e9)
    c, r = p.close, p.ret1()
    assert "222220" in c.columns and bool(p.delist_after["222220"].sum() == 1)                      # K1
    assert c["222220"].last_valid_index() == DAYS[200] and p.meta["delisted_in_panel"] >= 1
    assert p.meta["return_basis"].startswith("price (") and "dividends are not in the series" in p.meta["return_basis"]                  # the page must be able to say so
    assert abs(float(r.loc[DAYS[150], "111110"])) < 0.05                                           # K2: no -98% on the split day
    assert float(c["111110"].iloc[0]) == 100.0 or float(c["111110"].iloc[1]) > 50
    assert np.isnan(r.loc[DAYS[100], "333330"])                                                    # K3: listing-day return dropped
    el = p.eligible["333330"]
    assert el[el].index[0] == DAYS[100 + 29], el[el].index[0]
    assert "444440#2" in c.columns and c["444440"].last_valid_index() == DAYS[40]                   # K4
    print("K1 survivorship, K2 split, K3 listing day, K4 code reuse, K5 resumable  PASS")


def test_quota_stops_cleanly():
    world = _world(); state = {"n": 0}

    def caller(path, date):
        state["n"] += 1
        if state["n"] > 40:
            raise krx.QuotaExceeded("HTTP 429: limit")
        return world.get(date, []) if path.endswith("stk_bydd_trd") else []

    with tempfile.TemporaryDirectory() as d:
        out = krx.fetch_days(DAYS[0], DAYS[-1], d, caller=caller, sleep=0, progress=False)
        saved = len(list(Path(d).glob("*.pkl")))
    assert out["stopped"].startswith("quota") and saved == 20, (out, saved)
    print(f"K5 quota error stops cleanly, {saved} days kept  PASS")


def test_suspect_returns_after_a_suspension():
    """K6 a security suspended for 30 days, during which it was consolidated 300 to 1: the first day of trading again carries a change against the old close, and
    the adjusted return becomes +29,948%. It is listed, it can be cut, and nothing else moves."""
    import tempfile
    days = pd.bdate_range("2021-01-04", periods=120)
    rng = np.random.default_rng(3)
    codes = [f"{(i + 1) * 10:06d}" for i in range(14)]
    px = {c: 10000.0 for c in codes}
    sus = "099990"; px[sus] = 2080.0
    prev_close = 0                                                                      # the last close after the consolidation, set below
    with tempfile.TemporaryDirectory(prefix="krxtest-") as d:
        for t, day in enumerate(days):
            rows = []
            for c in codes:
                new = px[c] * (1 + rng.normal(0, 0.01)) * (2.5 if (c == codes[1] and t == 60) else 1.0)     # codes[1]: one ordinary +150% day, no suspension
                rows.append(dict(code=c, name=f"N{c}", market="KOSPI", close=int(new), change=int(new - px[c]), open=int(new), high=int(new), low=int(new),
                                 volume=1_000_000, value=int(5e9), mktcap=int(new * 1e7), shares=10_000_000)); px[c] = new
            if t < 40:                                                                       # trades normally until day 39
                nw = px[sus] * (1 + rng.normal(0, 0.01)); rows.append(dict(code=sus, name="SUSCO", market="KOSPI", close=int(nw), change=int(nw - px[sus]), open=int(nw), high=int(nw), low=int(nw),
                                                                           volume=500_000, value=int(5e9), mktcap=int(nw * 1e7), shares=10_000_000)); px[sus] = nw
            elif t < 70:                                                                     # suspended: same close, no volume
                rows.append(dict(code=sus, name="SUSCO", market="KOSPI", close=int(px[sus]), change=0, open=int(px[sus]), high=int(px[sus]), low=int(px[sus]), volume=0, value=0, mktcap=int(px[sus] * 1e7), shares=10_000_000))
            else:                                                                            # trading again after a 300:1 consolidation; the change is against the OLD close
                new_close = int(round(px[sus] * 300 * (1 + (0.02 if t == 70 else rng.normal(0, 0.01)))))
                chg = new_close - int(px[sus]) if t == 70 else new_close - int(prev_close)
                rows.append(dict(code=sus, name="SUSCO", market="KOSPI", close=new_close, change=chg, open=new_close, high=new_close, low=new_close, volume=400_000, value=int(5e9), mktcap=new_close * 10_000_000, shares=10_000_000))
                prev_close = new_close
            pd.DataFrame(rows).to_pickle(f"{d}/{day:%Y%m%d}.pkl")
        old = krx.build_krx_panel(d, min_age_days=5, adv_window=5, min_value_krw=1e6)
        cut = krx.build_krx_panel(d, min_age_days=5, adv_window=5, min_value_krw=1e6, drop_suspect_above=1.0)
        sr = old.meta["suspect_returns"]
        assert len(sr) == 2, sr
        a = sr[sr.ticker == sus].iloc[0]; b = sr[sr.ticker == codes[1]].iloc[0]
        assert bool(a["after_suspension"]) and a["ret"] > 100 and not bool(b["after_suspension"]) and abs(b["ret"] - 1.5) < 0.02, sr
        mid = krx.build_krx_panel(d, min_age_days=5, adv_window=5, min_value_krw=1e6, drop_suspect_above=2.0)       # between the two: only the false one is cut
        assert mid.close[codes[1]].pct_change().max() > 1.4 and mid.close[sus].pct_change().abs().max() < 0.2
        assert old.close[sus].pct_change().max() > 100                                       # the default keeps the series as it was
        r2 = cut.close[sus].pct_change()
        assert abs(r2.iloc[70]) < 1e-12 and r2.abs().max() < 0.2, r2.abs().max()           # cut: that day is flat, every other day is a normal return
        assert cut.close[codes[1]].pct_change().abs().max() < 0.2                             # at 1.0 the ordinary +150% day is cut as well: the threshold is on size, not on cause
        others = [c for c in codes if c != codes[1]]
        assert np.allclose(old.close[others].to_numpy(), cut.close[others].to_numpy(), equal_nan=True)   # nothing else changed (codes[1]'s +150% day is cut at 1.0 too)
        r_old = old.close[sus].pct_change().drop(old.dates[70]); r_cut = cut.close[sus].pct_change().drop(cut.dates[70])
        assert np.allclose(r_old.fillna(0).to_numpy(), r_cut.fillna(0).to_numpy(), atol=1e-12)          # every other return of the suspended name is the same
        try:
            krx.build_krx_panel(d, min_age_days=5, adv_window=5, min_value_krw=1e6, drop_suspect_above=0.0)
        except ValueError as e:
            assert "positive" in str(e)
        else:
            raise AssertionError("a zero threshold must raise")
    print("K6 a consolidation during a suspension gives a +29,948% return: listed, kept by default, cut on request, and nothing else changes  PASS")


def test_no_trade_days_have_no_intraday_price():
    """K7 a day without trades is written with open, high and low 0: that is no price. The panel makes it NaN, the close stays, and trading at the open stays finite."""
    import tempfile
    from pitbacktest import execution as ex
    days = pd.bdate_range("2021-01-04", periods=80)
    rng = np.random.default_rng(8)
    codes = [f"{(i + 1) * 10:06d}" for i in range(14)]
    px = {c: 10000.0 for c in codes}
    with tempfile.TemporaryDirectory(prefix="krxtest-") as d:
        for t, day in enumerate(days):
            rows = []
            for c in codes:
                idle = (c == codes[0] and 30 <= t < 36)                                    # codes[0] does not trade for six days
                nw = px[c] * (1 + (0 if idle else rng.normal(0, 0.01)))
                rows.append(dict(code=c, name=f"N{c}", market="KOSPI", close=int(nw), change=int(nw - px[c]), open=0 if idle else int(nw), high=0 if idle else int(nw * 1.01),
                                 low=0 if idle else int(nw * 0.99), volume=0 if idle else 1_000_000, value=0 if idle else int(5e9), mktcap=int(nw * 1e7), shares=10_000_000))
                px[c] = nw
            pd.DataFrame(rows).to_pickle(f"{d}/{day:%Y%m%d}.pkl")
        P = krx.build_krx_panel(d, min_age_days=5, adv_window=5, min_value_krw=1e6)
        idle = P.open.index[30:36]
        assert P.open.loc[idle, codes[0]].isna().all() and P.high.loc[idle, codes[0]].isna().all() and P.low.loc[idle, codes[0]].isna().all()
        assert P.close.loc[idle, codes[0]].notna().all()                                   # the close (carried) is still there
        assert (P.open.stack().dropna() > 0).all() and (P.high.stack().dropna() > 0).all() and (P.low.stack().dropna() > 0).all()
        assert P.open.loc[P.dates[10], codes[1]] > 0 and abs(P.open.loc[P.dates[10], codes[1]] / P.close.loc[P.dates[10], codes[1]] - 1) < 0.05   # a trading day keeps its open
        r = ex.at_prices(P, "open").close.pct_change(fill_method=None)
        assert np.isfinite(r.to_numpy()[~np.isnan(r.to_numpy())]).all()
    print("K7 a day without trades has no open, high or low (NaN, not 0); the close stays; trading at the open has no infinite return  PASS")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    test_adapter()
    test_quota_stops_cleanly()
    test_suspect_returns_after_a_suspension()
    test_no_trade_days_have_no_intraday_price()
    print("krx tests: all passed")
