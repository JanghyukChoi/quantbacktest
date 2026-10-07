"""KRX adapter, tested offline with a fake API whose right answers are known.

K1 survivorship  a stock that was listed on day t but is gone later is in the panel and is flagged as delisted
K2 split         a 50:1 split (raw price falls 98%) leaves the adjusted return unchanged
K3 listing day   the first bar of a new listing has no return, and the stock waits `min_age_days` to be eligible
K4 code reuse    a code that comes back after a long gap becomes a different security
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


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    test_adapter()
    test_quota_stops_cleanly()
    print("krx tests: all passed")
