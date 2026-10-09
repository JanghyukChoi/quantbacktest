"""A. Factor screening with neutralisation: the first gate of the research stage.

This stage decides the conclusion. A candidate that clears the statistical threshold usually disappears after the cost, timing and
    monotonicity gates and the neutralisation against firm characteristics; that is the normal outcome.

Principles
    1. The statistic that decides the first gate (`t`) is a Fama-MacBeth t **with controls**, and the factor is re-tested after neutralisation
       (`neu_t`, `neu_survival_%`). Uncontrolled figures (`excess_bp`, `net_bp`, `rho`, and `coef_bp_raw`) are returned next to them, labelled as such.
    2. Missing firm characteristics produce a warning (a missing ROA or book-to-market is the main cause of false findings).
    3. The multiple-testing threshold is **measured with a shuffled null**.
    4. The whole **distribution** of a parameter grid is returned, not its best cell.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .core.controls import build_controls, neutralize, xs_norm
from .core.estimators import (decile_profile, fama_macbeth, newey_west_t,
                              shuffle_columns, shuffle_null)
from .core.gates import GateConfig, fire_structure, run_gates
from .core.panel import Panel, check_alignment


@dataclass
class ScreenResult:
    null: dict
    factors: dict
    survivors: list[str]
    funnel: dict
    summary: pd.DataFrame


def _deploy(fire: np.ndarray, cum: np.ndarray, ev: np.ndarray,
            cost_bp: float) -> tuple[np.ndarray, dict]:
    # adapt to the universe size: a fixed threshold throws away every date in a small universe
    med = float(np.median(ev.sum(axis=1)))
    min_pool = int(max(5, min(30, med * 0.5)))
    exc = []
    for i in range(cum.shape[0]):
        s = fire[i] & np.isfinite(cum[i])
        pool = ev[i] & np.isfinite(cum[i])
        if s.sum() >= 1 and pool.sum() >= min_pool:
            exc.append(cum[i][s].mean() - cum[i][pool].mean())
    a = np.array(exc)
    if len(a) < 100:
        return a, {"excess_bp": np.nan, "net_bp": np.nan, "t": np.nan}
    mu, _, t, _ = newey_west_t(a, lag=21)
    return a, {"excess_bp": float(mu * 1e4), "net_bp": float(mu * 1e4 - cost_bp), "t": float(t)}


def screen(panel: Panel, factors: dict[str, pd.DataFrame], *,
           horizons: tuple[int, ...] = (1, 5, 20, 60),
           primary_h: int = 20, cost_bp: float = 20.0,
           fire_q: float = 0.10, n_null: int = 20,
           gate_config: GateConfig | None = None,
           neutralize_all: bool = True, delist_return: float | None = None) -> ScreenResult:
    """Screen many factors at once.

    factors  {name: (date x ticker) continuous factor}
    delist_return  return assumed on the day after a security's last bar when it is flagged in panel.delist_after (None: carried at its last price).

    In the summary, `t` is the Fama-MacBeth t **with controls**, and `neu_t` and `neu_survival_%` re-test the factor after residualising it
    against the controls. `excess_bp`, `net_bp`, `rho` and the gates G2 to G6 come from the **uncontrolled** top `fire_q` firing. The uncontrolled
    coefficient and t of a factor are in result.factors[name]['fm'][h]['coef_bp_raw'] and ['t_raw'].
    """
    el = panel.eligible
    ev = el.values
    ctrl = build_controls(panel)
    if not horizons or any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1 for h in horizons):
        raise ValueError(f"horizons must be whole numbers of periods, at least 1, got {horizons!r}")
    if n_null < 1:
        raise ValueError(f"n_null must be at least 1, got {n_null!r}")
    if n_null < 10:
        warnings.warn(f"n_null={n_null}: the G1 threshold is the 95th percentile of only {n_null} shuffled |t| values, which is biased low and moves between "
                      f"seeds (on a test panel it averaged 0.9 with n_null=2 and 1.8 with n_null=20; the 95th percentile of pure noise is about 1.96). "
                      f"Use at least 10.", stacklevel=2)
    if primary_h not in horizons:
        raise ValueError(f"primary_h={primary_h} must be one of horizons {tuple(horizons)}")
    if not (cost_bp >= 0 and np.isfinite(cost_bp)):
        raise ValueError(f"cost_bp must be finite and not negative, got {cost_bp!r}")
    if not (0 < fire_q <= 1):
        raise ValueError(f"fire_q must be in (0, 1], got {fire_q!r}")
    for fname, fv in factors.items():
        if isinstance(fv, pd.DataFrame) and len(fv.columns) and (fv.dtypes == bool).all():
            raise ValueError(f"factor {fname!r} is boolean: screen ranks a numeric score, use backtest_event for a yes/no signal")
    fwd = {h: panel.forward(h, delist_return) for h in horizons}
    cums = {h: fwd[h].values.astype(np.float64) for h in horizons}

    # 1) measure the threshold with a shuffled null: running it for every factor is expensive, so a representative factor stands in
    rep = next(iter(factors.values()))
    null = shuffle_null(lambda rng: xs_norm(shuffle_columns(rep, el, rng), el),
                        el, {primary_h: fwd[primary_h]}, ctrl, n_rep=n_null)
    thr = null.get("p95", np.nan)
    if not np.isfinite(thr):
        thr = 3.0
    cfg = GateConfig(null_threshold=thr) if gate_config is None else (
        gate_config if gate_config.null_threshold is not None else replace(gate_config, null_threshold=thr))

    rows, detail = [], {}
    for name, raw in factors.items():
        check_alignment(panel, raw, f"factor {name!r}")
        f = xs_norm(raw.reindex(index=panel.dates, columns=panel.tickers), el)
        fm = fama_macbeth(f, fwd, ctrl, el)
        rk = f.rank(axis=1, pct=True, na_option="keep")
        fire = ((rk >= 1 - fire_q) & el).values
        de, dep = _deploy(fire, cums[primary_h], ev, cost_bp)
        dec = decile_profile(f, fwd[primary_h], el, lag=max(primary_h, 21))
        g = run_gates(de, panel.dates, t_stat=fm[primary_h]["t"], net_bp=dep["net_bp"],
                      rho=dec["monotonicity_rho"], fire=fire, tickers=panel.tickers,
                      cfg=cfg, lag=max(primary_h, 21))

        neu = None
        if neutralize_all:
            fn = xs_norm(neutralize(f, ctrl, el), el)
            fmn = fama_macbeth(fn, fwd, ctrl, el)
            rkn = fn.rank(axis=1, pct=True, na_option="keep")
            firen = ((rkn >= 1 - fire_q) & el).values
            _, depn = _deploy(firen, cums[primary_h], ev, cost_bp)
            keep = (depn["excess_bp"] / dep["excess_bp"] * 100
                    if np.isfinite(dep["excess_bp"]) and abs(dep["excess_bp"]) > 1e-9 else np.nan)
            neu = {"t": fmn[primary_h]["t"], "excess_bp": depn["excess_bp"],
                   "net_bp": depn["net_bp"], "survival_pct": float(keep) if np.isfinite(keep) else np.nan}

        detail[name] = {"fm": {h: fm[h] for h in horizons}, "deploy": dep,
                        "decile": dec, "gates": g, "neutralized": neu}
        rows.append({
            "factor": name,
            "t": fm[primary_h]["t"],                       # after controls
            "excess_bp": dep["excess_bp"], "net_bp": dep["net_bp"],
            "rho": dec["monotonicity_rho"],
            "neu_t": neu["t"] if neu else np.nan,
            "neu_survival_%": neu["survival_pct"] if neu else np.nan,
            "passed": g["passed"], "failed_at": g["failed_at"],
        })

    df = pd.DataFrame(rows).sort_values("net_bp", ascending=False)
    surv = df.loc[df.passed, "factor"].tolist()
    n = len(df)
    funnel = {
        "all": n,
        f"G1 |t|>{thr:.2f}": int((df.t.abs() > thr).sum()),
        "G2 net of costs > 0": int(((df.t.abs() > thr) & (df.net_bp > 0)).sum()),
        "passed all gates": len(surv),
        "survival after neutralisation > 50%": int((df["neu_survival_%"] > 50).sum()) if neutralize_all else None,
    }
    return ScreenResult(null=null, factors=detail, survivors=surv, funnel=funnel, summary=df)
