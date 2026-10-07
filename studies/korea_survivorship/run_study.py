"""Run the preregistered Korean survivorship measurement in PREREGISTRATION.md.

    python run_study.py          # needs the KRX cache: see quantbt/adapters/krx.py (fetch_days)

Writes results.json next to this file; make_report.py turns it into REPORT.md.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import quantbt as q                                                       # noqa: E402
from quantbt.adapters.krx import build_krx_panel                           # noqa: E402
from quantbt.equity import survivors_only                                  # noqa: E402

warnings.filterwarnings("ignore")
STORE = Path.home() / ".cache" / "quantbt" / "krx"
START = "2013-01-01"
HOLDS = (5, 20)
COSTS = {"15bp": 15.0, "0bp": 0.0}


def factors(p) -> dict:
    c = p.close
    r = c.pct_change(fill_method=None)
    return {
        "K1_reversal_5d": -c.pct_change(5, fill_method=None),
        "K2_momentum_240_20": c.shift(20) / c.shift(240) - 1.0,
        "K3_low_vol_60": -r.rolling(60, min_periods=50).std(),
        "K4_small_size": -np.log(p.mkt_cap.clip(lower=1)),
    }


def run(p, f, h, one_way_bp):
    # scalar spread_bp is a round trip: pass twice the one-way cost
    return q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=h, spread_bp=2.0 * one_way_bp, benchmark=None, grid=False)


def main() -> None:
    t0 = time.time()
    A = build_krx_panel(STORE, start="2012-01-01")                          # one year of warm-up before the first signal
    fa = {k: v.loc[START:] for k, v in factors(A).items()}
    k = A.close.index >= pd.Timestamp(START)
    from dataclasses import replace
    sl = lambda d: None if d is None else d.loc[k]
    A2 = replace(A, close=sl(A.close), eligible=sl(A.eligible), open=sl(A.open), high=sl(A.high), low=sl(A.low),
                 volume=sl(A.volume), mkt_cap=sl(A.mkt_cap), delist_after=sl(A.delist_after))
    B2 = survivors_only(A2)
    fb = {n: v[B2.close.columns] for n, v in fa.items()}
    print("A:", A2.close.shape, "B:", B2.close.shape, flush=True)
    rows = {}
    for fname in fa:
        for h in HOLDS:
            tid = f"{fname}_h{h}"
            rows[tid] = {}
            for cname, c in COSTS.items():
                ra, rb = run(A2, fa[fname], h, c), run(B2, fb[fname], h, c)
                rows[tid][cname] = {"A_sharpe": float(ra.metrics["Sharpe"]), "B_sharpe": float(rb.metrics["Sharpe"]),
                                    "A_cagr": float(ra.metrics["CAGR"]), "B_cagr": float(rb.metrics["CAGR"]),
                                    "A_turnover": float(ra.metrics["turnover_daily"])}
            print(tid, {c: (round(v["A_sharpe"], 2), round(v["B_sharpe"], 2)) for c, v in rows[tid].items()}, flush=True)
    # share of eligible security-days that belong to securities missing from B, by year
    missing = [c for c in A2.close.columns if c not in B2.close.columns]
    e = A2.eligible
    share = (e[missing].sum(axis=1) / e.sum(axis=1)).groupby(e.index.year).mean()
    out = {"first_day": str(A2.dates[0].date()), "last_day": str(A2.dates[-1].date()), "days": int(len(A2.dates)),
           "securities_A": int(A2.close.shape[1]), "securities_B": int(B2.close.shape[1]),
           "eligible_median_A": float(e.sum(axis=1)[e.sum(axis=1) > 0].median()),
           "eligible_median_B": float(B2.eligible.sum(axis=1)[B2.eligible.sum(axis=1) > 0].median()),
           "missing_share_by_year": {int(y): float(v) for y, v in share.items()}, "trials": rows,
           "seconds": round(time.time() - t0)}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    print("done", out["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
