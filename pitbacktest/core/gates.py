"""Gate runner: verdict gates fixed before the result is seen.

The order of the gates matters: statistics -> cost -> stability -> structure.
**Each later gate is harder to pass.** With the statistical gate alone many candidates survive, but
after the cost, stability and structure gates far fewer remain, and that is the normal outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .estimators import newey_west_t


@dataclass
class GateConfig:
    """Gate thresholds. Fix them before testing starts and do not change them afterwards."""
    null_threshold: float | None = None   # None means measure it with a shuffled null
    fallback_t: float = 3.0               # only when the null cannot be measured
    require_net_positive: bool = True     # net profit after costs > 0
    require_subperiod_sign: bool = True   # the sign agrees in the first and second half
    min_monotonicity: float = 0.5         # |rho|
    min_yearly_positive: float = 0.6      # share of years that are positive
    max_single_name_share: float = 0.5    # firing share of the most frequent name (static basket)
    min_turnover: float = 0.05            # lower bound of daily turnover


def fire_structure(fire: np.ndarray, dates: pd.DatetimeIndex,
                   tickers: pd.Index) -> dict:
    """Firing structure: a signal or a static basket?

    A factor that keeps picking the same large stocks for years is a size factor (a static basket), not a signal.
    The top-name share and the longest run of consecutive firing days catch it.
    """
    n = fire.sum(axis=1)
    act = n[n > 0]
    cnt = pd.Series(fire.sum(axis=0), index=tickers)
    cnt = cnt[cnt > 0].sort_values(ascending=False)
    tot = int(fire.sum())
    if tot == 0 or len(cnt) == 0:
        return {"fires_per_day": 0.0, "active_days_pct": 0.0, "n_names": 0,
                "top10_share": np.nan, "max_name_share": np.nan,
                "max_consecutive": 0, "daily_turnover": np.nan, "top_names": {},
                "note": "no firings"}
    turn = []
    for i in range(1, fire.shape[0]):
        a, b = fire[i - 1], fire[i]
        u = (a | b).sum()
        if u and a.sum() and b.sum():
            turn.append(1 - (a & b).sum() / u)
    runs = []
    for c in cnt.head(30).index:
        s = fire[:, tickers.get_loc(c)].astype(int)
        best = cur = 0
        for x in s:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        runs.append(best)
    return {
        "fires_per_day": float(act.mean()) if len(act) else 0.0,
        "active_days_pct": float(len(act) / max(len(dates), 1) * 100),
        "n_names": int(len(cnt)),
        "top10_share": float(cnt.head(10).sum() / tot) if tot else np.nan,
        "max_name_share": float(cnt.iloc[0] / len(dates)) if len(cnt) else np.nan,
        "max_consecutive": int(max(runs)) if runs else 0,
        "daily_turnover": float(np.mean(turn)) if turn else np.nan,
        "top_names": cnt.head(8).to_dict(),
    }


def subperiod(values: np.ndarray, dates: pd.DatetimeIndex, lag: int = 21) -> dict:
    """First half versus second half. If the signs differ the result depends on the period."""
    values = np.asarray(values)
    if len(values) == 0:
        return {"front": {"mean_bp": np.nan, "t": np.nan, "n": 0},
                "back": {"mean_bp": np.nan, "t": np.nan, "n": 0}, "sign_match": False}
    mid = len(values) // 2
    out = {}
    for lab, sl in (("front", slice(0, mid)), ("back", slice(mid, len(values)))):
        mu, _, t, n = newey_west_t(values[sl], lag=lag)
        out[lab] = {"mean_bp": mu * 1e4 if np.isfinite(mu) else np.nan, "t": t, "n": n}
    f, b = out["front"]["mean_bp"], out["back"]["mean_bp"]
    out["sign_match"] = bool(np.isfinite(f) and np.isfinite(b) and (f > 0) == (b > 0))
    return out


def yearly(values: np.ndarray, dates: pd.DatetimeIndex, lag: int = 21) -> dict:
    """Consistency across years."""
    values = np.asarray(values)
    if len(values) == 0:
        return {"by_year": {}, "positive": 0, "total": 0, "positive_ratio": np.nan}
    s = pd.Series(values, index=dates[: len(values)])
    out, pos, tot = {}, 0, 0
    for y, g in s.groupby(s.index.year):
        if len(g) < 40:
            continue
        mu, _, t, n = newey_west_t(g.values, lag=lag)
        out[int(y)] = {"mean_bp": mu * 1e4 if np.isfinite(mu) else np.nan, "t": t}
        tot += 1
        pos += int(np.isfinite(mu) and mu > 0)
    return {"by_year": out, "positive": pos, "total": tot,
            "positive_ratio": pos / tot if tot else np.nan}


def oos_holdout(values: np.ndarray, dates: pd.DatetimeIndex,
                months: int = 12, lag: int = 21) -> dict:
    """OOS holdout: set the last N months aside and test on them.

    With a small sample a low t is normal. To tell 'dead' from 'cannot be measured'
    look at n_days as well.
    """
    values = np.asarray(values)
    if len(values) == 0:
        return {"IS": {"mean_bp": np.nan, "t": np.nan, "n": 0},
                "OOS": {"mean_bp": np.nan, "t": np.nan, "n": 0},
                "cutoff": None, "note": "no sample: firings are too rare or the universe is too small"}
    d = dates[: len(values)]
    cut = d[-1] - pd.DateOffset(months=months)
    out = {}
    for lab, m in (("IS", d <= cut), ("OOS", d > cut)):
        v = values[m]
        if len(v) < 30:
            out[lab] = {"mean_bp": np.nan, "t": np.nan, "n": len(v)}
            continue
        mu, _, t, n = newey_west_t(v, lag=lag)
        out[lab] = {"mean_bp": mu * 1e4 if np.isfinite(mu) else np.nan, "t": t, "n": n}
    out["cutoff"] = str(cut.date())
    return out


def run_gates(daily_excess: np.ndarray, dates: pd.DatetimeIndex, *,
              t_stat: float, net_bp: float, rho: float,
              fire: np.ndarray | None = None, tickers: pd.Index | None = None,
              cfg: GateConfig | None = None, lag: int = 21) -> dict:
    """Pass every gate in order and record where the candidate dropped out."""
    cfg = cfg or GateConfig()
    thr = cfg.null_threshold if cfg.null_threshold is not None else cfg.fallback_t
    res = {"config": asdict(cfg), "threshold": thr}
    steps: list[tuple[str, bool, str]] = []

    ok1 = abs(t_stat) > thr
    steps.append(("G1 statistical threshold", ok1, f"|t|={abs(t_stat):.2f} vs {thr:.2f}"))

    ok2 = (not cfg.require_net_positive) or (np.isfinite(net_bp) and net_bp > 0)
    steps.append(("G2 net profit after costs", ok2, f"{net_bp:+.1f}bp"))

    sp = subperiod(daily_excess, dates, lag=lag)
    ok3 = (not cfg.require_subperiod_sign) or sp["sign_match"]
    steps.append(("G3 sub-period sign", ok3,
                  f"first {sp['front']['mean_bp']:+.1f} / second {sp['back']['mean_bp']:+.1f}"))

    ok4 = np.isfinite(rho) and abs(rho) >= cfg.min_monotonicity
    steps.append(("G4 decile monotonicity", ok4, f"rho={rho:+.2f}"))

    yr = yearly(daily_excess, dates, lag=lag)
    ok5 = np.isfinite(yr["positive_ratio"]) and yr["positive_ratio"] >= cfg.min_yearly_positive
    steps.append(("G5 yearly consistency", ok5, f"{yr['positive']}/{yr['total']} years"))

    st = None
    if fire is not None and tickers is not None:
        st = fire_structure(fire, dates, tickers)
        ok6 = (st["max_name_share"] <= cfg.max_single_name_share
               and st["daily_turnover"] >= cfg.min_turnover)
        steps.append(("G6 firing structure", ok6,
                      f"top name {st['max_name_share']*100:.1f}% | turnover {st['daily_turnover']*100:.1f}%"))

    res["steps"] = [{"gate": g, "pass": bool(p), "detail": d} for g, p, d in steps]
    res["passed"] = all(p for _, p, _ in steps)
    res["failed_at"] = next((g for g, p, _ in steps if not p), None)
    res["subperiod"], res["yearly"], res["structure"] = sp, yr, st
    res["oos"] = oos_holdout(daily_excess, dates, lag=lag)
    return res
