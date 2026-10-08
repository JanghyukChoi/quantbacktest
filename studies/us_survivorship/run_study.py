"""Run the preregistered US survivorship measurement: PREREGISTRATION.md as amended by AMENDMENT_1.md.

    python run_study.py            # the full study (needs the Tiingo cache from fetch_sample.py); takes a few minutes with 8+ cores
    SMOKE=1 python run_study.py    # a tiny configuration, only to check that the script runs; writes nothing to results.json

Writes results.json next to this file; make_report.py turns it into REPORT.md.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import pitbacktest as q                                                   # noqa: E402
from pitbacktest.adapters import tiingo                                    # noqa: E402
from pitbacktest.analytics import sharpe_diff_ci                           # noqa: E402
from pitbacktest.core.controls import neutralize, xs_norm                  # noqa: E402
from pitbacktest.equity import inject_delistings, load_us_master, survivors_only   # noqa: E402

warnings.filterwarnings("ignore")
SMOKE = bool(os.environ.get("SMOKE"))
STORE = Path.home() / ".cache" / "quantbt" / "tiingo"
WARM, START, SINCE = "2012-01-01", "2013-01-01", "2013-01-01"
HOLDS = (5, 20)
COSTS = {"10bp": 10.0, "0bp": 0.0}
N_BOOT, SEED = (100 if SMOKE else 2000), 0
RATES = (0.02,) if SMOKE else (0.01, 0.02, 0.03)
DRETS = (-0.30, -0.55)
HAZARDS = ("uniform",) if SMOKE else ("uniform", "volatile")
N_DRAWS = 1 if SMOKE else 10
SCEN_COST = "10bp"
FACTORS = ("U1_reversal_5d", "U2_momentum_240_20", "U3_low_vol_60", "U4_low_dollar_volume")

A_FULL = None                                                              # set before the worker pool is created (fork shares it)


# ------------------------------------------------------------------------------------------------ definitions
def styles(p) -> dict:
    """The four style exposures from data up to day t: the factors without the sign (U3 and U4) and the raw measures."""
    c = p.close
    r = c.pct_change(fill_method=None)
    dv = (c * p.volume).rolling(60, min_periods=50).median().clip(lower=1)
    return {FACTORS[0]: c.pct_change(5, fill_method=None), FACTORS[1]: c.shift(20) / c.shift(240) - 1.0,
            FACTORS[2]: r.rolling(60, min_periods=50).std(), FACTORS[3]: np.log(dv)}


def factors(p) -> dict:
    """Higher = long. U1 minus the 5-day return, U2 return t-240 to t-20, U3 minus the 60-day volatility, U4 minus log dollar volume."""
    s = styles(p)
    return {FACTORS[0]: -s[FACTORS[0]], FACTORS[1]: s[FACTORS[1]], FACTORS[2]: -s[FACTORS[2]], FACTORS[3]: -s[FACTORS[3]]}


def cut(p, start=START):
    k = p.close.index >= pd.Timestamp(start)
    sl = lambda d: None if d is None else d.loc[k]
    return replace(p, close=sl(p.close), eligible=sl(p.eligible), volume=sl(p.volume), delist_after=sl(p.delist_after))


def run(p, f, h, one_way_bp, delist_return=None):
    # scalar spread_bp is a round trip: pass twice the one-way cost
    return q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=h, spread_bp=2.0 * one_way_bp, benchmark="equal", grid=False,
                                delist_return=delist_return)


def neutral_factors(p, raw: dict, st: dict) -> dict:
    """Each factor residualised against the other three styles within this universe's own cross-section."""
    out = {}
    for name, f in raw.items():
        ctrl = {k: xs_norm(v.reindex(columns=f.columns), p.eligible) for k, v in st.items() if k != name}
        out[name] = neutralize(xs_norm(f, p.eligible), ctrl, p.eligible)
    return out


def compare(ra, rb, bench_a, ppy) -> dict:
    ci = sharpe_diff_ci(ra.net_returns, rb.net_returns, periods_per_year=ppy, n=N_BOOT, seed=SEED)
    aa, ab = ra.alpha_beta(bench_a), rb.alpha_beta(bench_a)
    k = "benchmark"
    return {"A_sharpe": float(ra.metrics["Sharpe"]), "B_sharpe": float(rb.metrics["Sharpe"]),
            "A_cagr": float(ra.metrics["CAGR"]), "B_cagr": float(rb.metrics["CAGR"]),
            "diff": ci["diff"], "diff_lo": ci["lo"], "diff_hi": ci["hi"], "diff_se": ci["se"], "p_boot": ci["p_boot"],
            "A_beta": aa["betas"][k]["beta"], "B_beta": ab["betas"][k]["beta"],
            "A_alpha_annual": aa["alpha_annual"], "B_alpha_annual": ab["alpha_annual"],
            "A_alpha_t": aa["alpha_t"], "B_alpha_t": ab["alpha_t"]}


# ------------------------------------------------------------------------------------------------ sample and panel
def load_sample():
    master = load_us_master()
    frame = tiingo.study_frame(master, SINCE)
    order = tiingo.draw_order(frame["ticker"], seed=SEED)
    prefix = 0
    for t in order:
        if (STORE / f"{tiingo._safe(t)}.json").exists() or (STORE / f"{tiingo._safe(t)}.none").exists():
            prefix += 1
        else:
            break
    drawn = order[:prefix]
    got = [t for t in drawn if (STORE / f"{tiingo._safe(t)}.json").exists()]
    none = [t for t in drawn if (STORE / f"{tiingo._safe(t)}.none").exists()]
    return master, frame, order, drawn, got, none


def audit(master, frame, drawn, got, none, A_full, A2, B2) -> dict:
    last_date = master["end"].max()
    alive_of = lambda t: bool((master.loc[master.ticker == t, "end"].max() >= last_date - pd.Timedelta(days=7)))
    frame_alive = frame.groupby("ticker")["end"].max() >= last_date - pd.Timedelta(days=7)
    frame_delisted_share = float((~frame_alive).mean())
    drawn_alive = [alive_of(t) for t in drawn]
    got_alive = [alive_of(t) for t in got]
    # distress-type delistings actually present: delisted securities whose price fell more than 50% over their last 252 bars
    c = A_full.close
    flagged = [col for col in c.columns if A_full.delist_after[col].any()]
    big_fall = 0
    for col in flagged:
        s = c[col].dropna()
        if len(s) > 252 and s.iloc[-1] / s.iloc[-253] - 1 < -0.5:
            big_fall += 1
    el = A2.eligible
    n = el.sum(axis=1)
    yr = c.loc[START:].notna().groupby(c.loc[START:].index.year).any().sum(axis=1)
    big = (A2.close.pct_change(fill_method=None).where(A2.eligible)).abs().stack().sort_values(ascending=False).head(5)
    return {"frame_tickers": int(len(frame["ticker"].unique())), "frame_delisted_share": frame_delisted_share,
            "drawn": len(drawn), "with_prices": len(got), "none": len(none),
            "none_tickers": [{"ticker": t, "alive": alive_of(t)} for t in none],
            "drawn_delisted_share": float(1 - np.mean(drawn_alive)), "obtained_delisted_share": float(1 - np.mean(got_alive)),
            "obtained_delisted": int(sum(not a for a in got_alive)),
            "delisted_flagged_in_panel": len(flagged), "delisted_last_year_fell_over_50pct": int(big_fall),
            "securities_A": int(A2.close.shape[1]), "securities_B": int(B2.close.shape[1]),
            "eligible_per_day_min": int(n[n > 0].min()), "eligible_per_day_median": float(n[n > 0].median()), "eligible_per_day_max": int(n.max()),
            "ever_eligible": int((el.sum() > 0).sum()), "securities_with_a_bar_by_year": {int(k): int(v) for k, v in yr.items()},
            "windows_cut_for_ticker_reuse": int(A_full.meta.get("windows_cut", 0)),
            "suspect_returns_gt_10x": int(A_full.meta.get("suspect_returns_gt_10x", 0)),
            "largest_abs_daily_returns_among_eligible": [{"security": a, "date": str(b.date()), "ret": float(v)} for (b, a), v in big.items()],
            "inconclusive": bool(len(got) < 300), "delisted_share_flag": bool(float(1 - np.mean(got_alive)) < 0.7 * frame_delisted_share)}


# ------------------------------------------------------------------------------------------------ scenario worker
def _scenario_task(args):
    hz, rate, i = args
    P = inject_delistings(A_FULL, annual_rate=rate, hazard=hz, seed=SEED + i)
    Pc = cut(P)
    fx = {k: v.loc[START:] for k, v in factors(P).items()}
    out = []
    for fname in FACTORS:
        for h in HOLDS:
            for dr in DRETS:
                r = run(Pc, fx[fname], h, COSTS[SCEN_COST], delist_return=dr)
                out.append((f"{fname}_h{h}", hz, rate, dr, i, float(r.metrics["Sharpe"])))
    return out


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    global A_FULL
    t0 = time.time()
    master, frame, order, drawn, got, none = load_sample()
    print(f"frame {len(order)} | drawn prefix {len(drawn)} | with prices {len(got)} | none {len(none)}", flush=True)
    A_FULL = tiingo.build_tiingo_panel(STORE, master, got, start=WARM, min_names_per_date=100)
    A2 = cut(A_FULL)
    B2 = survivors_only(A2)
    ppy = A2.periods_per_year
    print(f"A_T {A2.close.shape} B {B2.close.shape} | eligible per day median {int(A2.eligible.sum(axis=1).median())} ({time.time()-t0:.0f}s)", flush=True)

    aud = audit(master, frame, drawn, got, none, A_FULL, A2, B2)
    rawA = {k: v.loc[START:] for k, v in factors(A_FULL).items()}
    rawB = {k: v[B2.close.columns] for k, v in rawA.items()}
    stA = {k: v.loc[START:] for k, v in styles(A_FULL).items()}
    stB = {k: v[B2.close.columns] for k, v in stA.items()}
    neuA, neuB = neutral_factors(A2, rawA, stA), neutral_factors(B2, rawB, stB)

    trials = {}
    for fname in FACTORS:
        for h in HOLDS:
            tid = f"{fname}_h{h}"
            trials[tid] = {}
            for cname, c in COSTS.items():
                ra, rb = run(A2, rawA[fname], h, c), run(B2, rawB[fname], h, c)
                bench = ra.benchmark_returns
                na, nb = run(A2, neuA[fname], h, c), run(B2, neuB[fname], h, c)
                trials[tid][cname] = {"raw": compare(ra, rb, bench, ppy), "neutral": compare(na, nb, bench, ppy)}
            r = trials[tid]["10bp"]
            print(tid, f"raw diff {r['raw']['diff']:+.2f} [{r['raw']['diff_lo']:+.2f},{r['raw']['diff_hi']:+.2f}]  neutral diff {r['neutral']['diff']:+.2f}  ({time.time()-t0:.0f}s)", flush=True)

    # Part 3: the scenario (assumed, not measured). Same grid for every trial, 10 draws per cell, at 10 bp.
    tasks = [(hz, rate, i) for hz in HAZARDS for rate in RATES for i in range(N_DRAWS)]
    print(f"scenario: {len(tasks)} injected panels x {len(FACTORS) * len(HOLDS) * len(DRETS)} backtests", flush=True)
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4), mp_context=mp.get_context("fork")) as ex:
        rows = [r for part in ex.map(_scenario_task, tasks) for r in part]
    df = pd.DataFrame(rows, columns=["tid", "hazard", "rate", "delist_return", "draw", "sharpe"])
    scen = {}
    for tid, g in df.groupby("tid"):
        cells = []
        for (hz, rate, dr), gg in g.groupby(["hazard", "rate", "delist_return"]):
            cells.append({"hazard": hz, "annual_rate": float(rate), "delist_return": float(dr), "mean_sharpe_A_scenario": float(gg["sharpe"].mean()),
                          "p5": float(np.percentile(gg["sharpe"], 5)), "p95": float(np.percentile(gg["sharpe"], 95)), "n_draws": int(len(gg))})
        scen[tid] = {"A_T_sharpe": trials[tid][SCEN_COST]["raw"]["A_sharpe"], "B_sharpe": trials[tid][SCEN_COST]["raw"]["B_sharpe"], "cells": cells}
    out = {"smoke": SMOKE, "config": {"holds": HOLDS, "costs": COSTS, "n_boot": N_BOOT, "seed": SEED, "rates": RATES, "delist_returns": DRETS,
                                      "hazards": HAZARDS, "n_draws": N_DRAWS, "scenario_cost": SCEN_COST, "start": START},
           "audit": aud, "trials": trials, "scenario": scen, "seconds": round(time.time() - t0)}
    target = HERE / ("results_smoke.json" if SMOKE else "results.json")
    target.write_text(json.dumps(out, indent=1))
    print("done", out["seconds"], "s ->", target.name, flush=True)


if __name__ == "__main__":
    main()
