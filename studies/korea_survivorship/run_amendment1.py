"""Amendment 1 (see AMENDMENT_1.md): precision of the survivorship difference and an exposure check.

    python run_amendment1.py     # needs the KRX cache; about 10 minutes

Writes results_amendment1.json. Run only after AMENDMENT_1.md was committed.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import pitbacktest as q                                                       # noqa: E402
from pitbacktest.adapters.krx import build_krx_panel                           # noqa: E402
from pitbacktest.analytics import sharpe_diff_ci                               # noqa: E402
from pitbacktest.core.controls import neutralize, xs_norm                      # noqa: E402
from pitbacktest.equity import survivors_only                                  # noqa: E402
from run_study import COSTS, HOLDS, START, STORE, factors                  # noqa: E402

warnings.filterwarnings("ignore")
N_BOOT, SEED = 2000, 0


def styles(p) -> dict:
    """The four style exposures, computed from data up to day t only."""
    c = p.close
    r = c.pct_change(fill_method=None)
    return {"K1_reversal_5d": c.pct_change(5, fill_method=None),
            "K2_momentum_240_20": c.shift(20) / c.shift(240) - 1.0,
            "K3_low_vol_60": r.rolling(60, min_periods=50).std(),
            "K4_small_size": np.log(p.mkt_cap.clip(lower=1))}


def neutral_factors(p, raw: dict, st: dict) -> dict:
    """Each factor residualised against the other three styles `st`, within this universe's own cross-section."""
    el = p.eligible
    out = {}
    for name, f in raw.items():
        ctrl = {k: xs_norm(v.reindex(columns=f.columns), el) for k, v in st.items() if k != name}
        out[name] = neutralize(xs_norm(f, el), ctrl, el)
    return out


def run(p, f, h, one_way_bp):
    # scalar spread_bp is a round trip: pass twice the one-way cost
    return q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=h, spread_bp=2.0 * one_way_bp, benchmark="equal", grid=False)


def compare(ra, rb, bench_a, ppy) -> dict:
    ci = sharpe_diff_ci(ra.net_returns, rb.net_returns, periods_per_year=ppy, n=N_BOOT, seed=SEED)
    aa, ab = ra.alpha_beta(bench_a), rb.alpha_beta(bench_a)
    k = "benchmark"
    return {"A_sharpe": float(ra.metrics["Sharpe"]), "B_sharpe": float(rb.metrics["Sharpe"]),
            "A_cagr": float(ra.metrics["CAGR"]), "B_cagr": float(rb.metrics["CAGR"]),
            "diff": ci["diff"], "diff_lo": ci["lo"], "diff_hi": ci["hi"], "diff_se": ci["se"], "p_boot": ci["p_boot"],
            "A_beta": aa["betas"][k]["beta"], "B_beta": ab["betas"][k]["beta"],
            "A_alpha_annual": aa["alpha_annual"], "B_alpha_annual": ab["alpha_annual"],
            "A_alpha_t": aa["alpha_t"], "B_alpha_t": ab["alpha_t"], "A_r2": aa["r2"], "B_r2": ab["r2"]}


def main() -> None:
    t0 = time.time()
    A = build_krx_panel(STORE, start="2012-01-01")
    k = A.close.index >= pd.Timestamp(START)
    sl = lambda d: None if d is None else d.loc[k]
    A2 = replace(A, close=sl(A.close), eligible=sl(A.eligible), open=sl(A.open), high=sl(A.high), low=sl(A.low),
                 volume=sl(A.volume), mkt_cap=sl(A.mkt_cap), delist_after=sl(A.delist_after))
    # factors and styles need the warm-up year, so compute on the full panel and cut afterwards
    rawA = {n: v.loc[START:] for n, v in factors(A).items()}
    B2 = survivors_only(A2)
    rawB = {n: v[B2.close.columns] for n, v in rawA.items()}
    stA = {n: v.loc[START:] for n, v in styles(A).items()}          # styles need the warm-up year too, then cut
    stB = {n: v[B2.close.columns] for n, v in stA.items()}
    neuA, neuB = neutral_factors(A2, rawA, stA), neutral_factors(B2, rawB, stB)
    print("A:", A2.close.shape, "B:", B2.close.shape, f"({time.time()-t0:.0f}s)", flush=True)

    ref = json.loads((HERE / "results.json").read_text())["trials"]
    ppy = A2.periods_per_year
    rows, worst = {}, 0.0
    for fname in rawA:
        for h in HOLDS:
            tid = f"{fname}_h{h}"
            rows[tid] = {}
            for cname, c in COSTS.items():
                ra, rb = run(A2, rawA[fname], h, c), run(B2, rawB[fname], h, c)
                bench = ra.benchmark_returns
                raw = compare(ra, rb, bench, ppy)
                for key in ("A_sharpe", "B_sharpe", "A_cagr", "B_cagr"):                  # reproduction check
                    worst = max(worst, abs(raw[key] - ref[tid][cname][key]))
                na, nb = run(A2, neuA[fname], h, c), run(B2, neuB[fname], h, c)
                neu = compare(na, nb, bench, ppy)
                rows[tid][cname] = {"raw": raw, "neutral": neu}
            print(tid, f"raw diff {rows[tid]['15bp']['raw']['diff']:+.2f}  neutral diff {rows[tid]['15bp']['neutral']['diff']:+.2f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    out = {"n_boot": N_BOOT, "seed": SEED, "reproduction_max_abs_diff": worst, "trials": rows, "seconds": round(time.time() - t0)}
    (HERE / "results_amendment1.json").write_text(json.dumps(out, indent=1))
    print("reproduction max |diff| vs results.json:", worst, flush=True)
    if worst > 1e-9:
        print("WARNING: the original trials were not reproduced; the new numbers are not comparable", flush=True)


if __name__ == "__main__":
    main()
