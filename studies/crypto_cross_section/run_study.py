"""Run the preregistered study in PREREGISTRATION.md. Nothing here is tuned after seeing a result.

    python run_study.py            # needs the archive cache: python -c "import quantbt.crypto as c; c.fetch_all()"

Writes results.json and REPORT.md next to this file.
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
import quantbt as q                                                     # noqa: E402
from quantbt import validation as val                                    # noqa: E402
from quantbt.core.panel import Panel                                     # noqa: E402
from quantbt.crypto import build_panel, liquidity_cost_bp, participation_report   # noqa: E402

warnings.filterwarnings("ignore")
START, WARM = "2021-01-01", "2020-08-01"
MIN_ADV, MIN_AGE = 2e7, 90
HOLDS = (1, 5, 10)
PPY = 365


def make_factors(p: Panel) -> dict[str, pd.DataFrame]:
    c = p.close
    r = c.pct_change(fill_method=None)
    turnover = c * p.volume                                              # USDT turnover
    return {
        "F1_reversal_5d": -c.pct_change(5, fill_method=None),
        "F2_momentum_30_5": c.shift(5) / c.shift(30) - 1.0,
        "F3_low_vol_30": -r.rolling(30, min_periods=25).std(),
        "F4_illiquidity": -np.log(turnover.rolling(30, min_periods=30).mean().clip(lower=1)),
    }


def cut(p: Panel, df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[START:]


def sliced(p: Panel) -> Panel:
    k = p.close.index >= pd.Timestamp(START)
    sl = lambda d: None if d is None else d.loc[k]
    return Panel(close=sl(p.close), eligible=sl(p.eligible), open=sl(p.open), high=sl(p.high), low=sl(p.low),
                 volume=sl(p.volume), chars={n: sl(v) for n, v in p.chars.items()}, market=p.market,
                 entry_lag=p.entry_lag, periods_per_year=p.periods_per_year, funding=sl(p.funding),
                 delist_after=sl(p.delist_after), meta=p.meta)


def run(panel: Panel, factor: pd.DataFrame, h: int, cost, *, funding=True, delist_return=0.0, entry_lag=1):
    panel.entry_lag = entry_lag
    return q.backtest_portfolio(panel, factor, long_q=0.2, short_q=0.2, hold=h, spread_bp=cost, benchmark=None,
                                grid=False, funding=funding, delist_return=delist_return)


def sharpe(res) -> float:
    return float(res.metrics["Sharpe"])


def year_stats(s: pd.Series) -> dict:
    out = {}
    for y, g in s.groupby(s.index.year):
        if len(g) >= 180 or len(g) >= 360:
            out[str(y)] = {"days": int(len(g)), "ret": float((1 + g).prod() - 1)}
    return out


def main() -> None:
    t0 = time.time()
    full = build_panel(start=WARM, min_adv_usd=MIN_ADV, min_age_days=MIN_AGE)
    surv = build_panel(start=WARM, min_adv_usd=MIN_ADV, min_age_days=MIN_AGE, survivors_only=True)
    arms = {}
    for name, pf in (("A", full), ("B", surv)):
        fac = {k: cut(pf, v) for k, v in make_factors(pf).items()}
        cost = cut(pf, liquidity_cost_bp(pf, taker_fee_bp=5.0, thin_usd=1e8))
        arms[name] = (sliced(pf), fac, cost, pf)
    pA, facA, costA, pfA = arms["A"]
    pB, facB, costB, _ = arms["B"]
    print("panels built", round(time.time() - t0), "s | A:", pA.meta, "| B:", pB.meta, flush=True)

    trials, rows = [], {}
    for fname in facA:
        for h in HOLDS:
            tid = f"{fname}_h{h}"
            rA = run(pA, facA[fname], h, costA)
            rB = run(pB, facB[fname], h, costB)
            rC = run(pA, facA[fname], h, costA, funding=False)
            rD = run(pA, facA[fname], h, 4.0, funding=False)           # scalar spread_bp is a round trip: 4 bp = 2 bp one way
            rows[tid] = {"A": sharpe(rA), "B": sharpe(rB), "C": sharpe(rC), "D": sharpe(rD),
                         "A_CAGR": float(rA.metrics["CAGR"]), "A_MDD": float(rA.metrics["MDD"]),
                         "A_turnover_daily": float(rA.metrics["turnover_daily"]),
                         "A_funding_annual_bp": float(rA.metrics["funding_annual_bp"]),
                         "A_cost_annual_bp": float(rA.metrics["cost_annual_bp"]),
                         "A_delist_events_held": int(rA.metrics["delist_events_held"])}
            trials.append((tid, fname, h, rA))
            print(tid, {k: round(v, 2) for k, v in rows[tid].items() if k in "ABCD"}, flush=True)

    # --- verdict on the best trial in specification A ---
    best_i = int(np.nanargmax([rows[t[0]]["A"] for t in trials]))
    tid, fname, h, rbest = trials[best_i]
    R = pd.concat({t[0]: t[3].net_returns for t in trials}, axis=1).dropna()
    dsr = val.deflated_sharpe(R.values, trials=len(trials), periods_per_year=PPY)
    pbo = val.pbo_cscv(R.values, blocks=16, periods_per_year=PPY)
    ys = year_stats(rbest.net_returns)
    pos_share = float(np.mean([v["ret"] > 0 for v in ys.values()])) if ys else float("nan")
    stress = {
        "delist_return_-0.30": sharpe(run(pA, facA[fname], h, costA, delist_return=-0.30)),
        "taker_fee_8bp": sharpe(run(pA, facA[fname], h, cut(pfA, liquidity_cost_bp(pfA, taker_fee_bp=8.0, thin_usd=1e8)))),
        "entry_lag_2": sharpe(run(pA, facA[fname], h, costA, entry_lag=2)),
    }
    pA.entry_lag = 1
    gates = {
        "G1": {"label": "net Sharpe >= 0.5", "value": sharpe(rbest), "pass": bool(sharpe(rbest) >= 0.5)},
        "G2": {"label": "deflated Sharpe >= 0.95", "value": dsr["dsr"], "pass": bool(dsr["dsr"] >= 0.95)},
        "G3": {"label": "PBO < 0.20", "value": pbo["pbo"], "pass": bool(pbo["pbo"] < 0.20)},
        "G4": {"label": "positive in >= 60% of calendar years", "value": pos_share, "pass": bool(pos_share >= 0.6)},
        "G5": {"label": "net Sharpe > 0 under all stresses", "value": min(stress.values()), "pass": bool(min(stress.values()) > 0)},
    }
    verdict = all(g["pass"] for g in gates.values())
    part = {f"{int(a/1e6)}M": participation_report(pA, rbest.holdings, a) for a in (1e6, 1e7, 1e8)}
    bias = {k: float(np.nanmean([rows[t][k] - rows[t]["A"] for t in rows])) for k in "BCD"}
    out = {"run_at_utc": pd.Timestamp.utcnow().isoformat(), "last_signal_date": str(pA.dates[-1].date()),
           "first_signal_date": str(pA.dates[0].date()), "days": int(len(pA.dates)),
           "universe_A": pA.meta, "universe_B": pB.meta, "trials": rows, "bias_mean_sharpe_diff_vs_A": bias,
           "best_trial": tid, "dsr": dsr, "pbo": pbo, "year_stats": ys, "stress": stress, "gates": gates,
           "verdict": verdict, "participation": part, "seconds": round(time.time() - t0)}
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nBEST", tid, "| gates", {k: g["pass"] for k, g in gates.items()}, "| verdict", verdict, flush=True)


if __name__ == "__main__":
    main()
