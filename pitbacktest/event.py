"""C. Event-signal test: does it hold up as individual alerts?

Portfolio metrics (mean excess return) assume the mean is realised. A user receives alerts one by one, though, so the **win rate,
median and payoff ratio** decide what it feels like.

What is always decomposed
    win rate = base rate (how the sample was picked) + lift (what the signal adds)
        - A longer holding period raises the win rate and the base rate together. Judge by the lift only.
        - A win rate below the base rate (lift < 0) means the signal does harm.
    mean vs median
        - If the signs differ, a few big winners cover many losses: it works as a portfolio and not as alerts.
        - The typical case is a positive mean with a negative median.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core.controls import build_controls, xs_norm
from .core.estimators import fama_macbeth, newey_west_t, paired_diff
from .core.gates import GateConfig, fire_structure, run_gates
from .core.notes import keep_warnings
from .core.panel import Panel, check_alignment


@dataclass
class EventResult:
    spec: dict
    per_horizon: dict
    gates: dict
    structure: dict
    neutralized: dict | None
    lookahead: dict

    notes: list = field(default_factory=list)   # the warnings the run raised (text), kept for programs that read the result

    def to_dict(self) -> dict:
        """The result as a JSON-ready dictionary (see `pitbacktest.export`)."""
        from .export import event_to_dict
        return event_to_dict(self)

    def to_json(self, path=None, indent: int | None = 2) -> str:
        """`to_dict()` as JSON text, or written to `path`."""
        from .export import dump_json
        return dump_json(self.to_dict(), path, indent)


def _trade_stats(fire: np.ndarray, cum: np.ndarray, eligible: np.ndarray,
                 cost: float) -> dict | None:
    """Per-trade profit distribution plus the base win rate."""
    tr, bh, bn = [], 0.0, 0
    for i in range(cum.shape[0]):
        s = fire[i] & np.isfinite(cum[i])
        k = int(s.sum())
        if k < 1:
            continue
        tr.append(cum[i][s] - cost)
        pool = eligible[i] & np.isfinite(cum[i])
        if pool.sum() >= 20:
            bh += float((cum[i][pool] - cost > 0).mean()) * k
            bn += k
    if not tr:
        return None
    t = np.concatenate(tr)
    if len(t) < 100:
        return None
    win = float((t > 0).mean())
    base = bh / bn if bn else np.nan
    w, l = t[t > 0], t[t < 0]
    return {
        "n_trades": int(len(t)),
        "win_rate": win * 100,
        "base_rate": base * 100 if np.isfinite(base) else np.nan,
        "lift_pp": (win - base) * 100 if np.isfinite(base) else np.nan,
        "mean_bp": float(t.mean() * 1e4),
        "median_bp": float(np.median(t) * 1e4),
        "avg_win_bp": float(w.mean() * 1e4) if len(w) else np.nan,
        "avg_loss_bp": float(l.mean() * 1e4) if len(l) else np.nan,
        "payoff": float(w.mean() / abs(l.mean())) if len(l) else np.nan,
        "p25_bp": float(np.percentile(t, 25) * 1e4),
        "p75_bp": float(np.percentile(t, 75) * 1e4),
        "skew_warning": bool(t.mean() > 0 > np.median(t)),
    }


def _daily_excess(fire: np.ndarray, cum: np.ndarray, eligible: np.ndarray,
                  *, min_fire: int = 1, min_pool: int | None = None) -> np.ndarray:
    """Daily excess return of the firing group (gate input).

    A fixed min_pool throws away every date in a small universe (for example a 20-name test).
    The default **adapts** to half the median universe size (at most 30).
    """
    if min_pool is None:
        med = float(np.median(eligible.sum(axis=1)))
        min_pool = int(max(5, min(30, med * 0.5)))
    out = []
    for i in range(cum.shape[0]):
        s = fire[i] & np.isfinite(cum[i])
        pool = eligible[i] & np.isfinite(cum[i])
        if s.sum() >= min_fire and pool.sum() >= min_pool:
            out.append(cum[i][s].mean() - cum[i][pool].mean())
    return np.array(out)


@keep_warnings
def backtest_event(panel: Panel, signal: pd.DataFrame, *,
                   horizons: tuple[int, ...] = (1, 2, 5, 10, 21),
                   cost_bp: float = 20.0,
                   neutralize_check: bool = True,
                   gate_config: GateConfig | None = None,
                   null_threshold: float | None = None,
                   delist_return: float | None = None) -> EventResult:
    """Event-signal test.

    signal   bool or 0/1 matrix (date x ticker). True = fires that day.
    cost_bp  round-trip trading cost. If per-security measured costs exist, use the spread panel on the portfolio side.
    delist_return  return assumed on the day after a security's last bar when it is flagged in panel.delist_after (None: carried at its last price).

    With the neutralisation test on (the default) the contribution after controls is re-measured with a **dummy regression**.
    An event signal is not a continuous factor, so it is read through a dummy coefficient and not through residuals.
    """
    if not horizons or any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1 for h in horizons):
        raise ValueError(f"horizons must be whole numbers of periods, at least 1, got {horizons!r}")
    if not (cost_bp >= 0 and np.isfinite(cost_bp)):
        raise ValueError(f"cost_bp must be finite and not negative, got {cost_bp!r}")
    check_alignment(panel, signal, "signal")
    sig = signal.reindex(index=panel.dates, columns=panel.tickers).fillna(False)
    fire = (sig.astype(bool) & panel.eligible).values
    ev = panel.eligible.values
    cost = cost_bp / 1e4

    lookahead = panel.assert_no_lookahead(sig.astype(float), h=max(horizons))
    cums = {h: panel.forward(h, delist_return).values.astype(np.float64) for h in horizons}

    per_h = {}
    for h in horizons:
        st = _trade_stats(fire, cums[h], ev, cost)
        if st is None:
            continue
        de = _daily_excess(fire, cums[h], ev)
        mu, _, t, n = newey_west_t(de, lag=max(h, 21))
        per_h[h] = {**st, "excess_bp": float(mu * 1e4), "t": float(t), "n_days": int(n),
                    "net_bp": float(mu * 1e4 - cost_bp)}

    if not per_h:
        raise ValueError(
            "No horizon can be tested. Either the signal fires too rarely (fewer than 100 events) or "
            "too few securities are eligible. Check signal, eligible and horizons.")
    # representative horizon = where the lift is largest
    best_h = max(per_h, key=lambda h: per_h[h].get("lift_pp", -99))
    de = _daily_excess(fire, cums[best_h], ev)
    if len(de) < 100:
        warnings.warn(
            f"Only {len(de)} days of daily excess return, so the gates cannot be trusted. "
            f"_daily_excess counts only days with at least one firing name and a pool of at least max(5, min(30, half the "
            f"median universe)) names, so a short sample or a rarely firing signal leaves few days.", stacklevel=2)

    from .core.estimators import decile_profile
    dec = decile_profile(sig.astype(float), panel.forward(best_h, delist_return), panel.eligible,
                         lag=max(best_h, 21))
    struct = fire_structure(fire, panel.dates, panel.tickers)
    cfg = gate_config or GateConfig(null_threshold=null_threshold)
    gates = run_gates(de, panel.dates, t_stat=per_h[best_h]["t"],
                      net_bp=per_h[best_h]["net_bp"], rho=dec["monotonicity_rho"],
                      fire=fire, tickers=panel.tickers, cfg=cfg, lag=max(best_h, 21))

    neu = None
    if neutralize_check:
        neu = _dummy_neutralized(panel, fire, cums, horizons)

    return EventResult(
        spec={"horizons": list(horizons), "cost_bp": cost_bp, "entry_lag": panel.entry_lag,
              "market": panel.market, "best_horizon": best_h,
              "decile": dec},
        per_horizon=per_h, gates=gates, structure=struct,
        neutralized=neu, lookahead=lookahead)


def _dummy_neutralized(panel: Panel, fire: np.ndarray, cums: dict,
                       horizons: tuple[int, ...], min_obs: int = 30) -> dict:
    """Event dummy regression: does firing still predict returns after controlling for characteristics?

    Sample: every eligible name that day. Dependent: the future return. Independent: [firing dummy] + controls.
    The dummy coefficient is 'what firing adds among names with the same characteristics'.
    """
    ctrl = build_controls(panel)
    cvs = [c.values.astype(np.float64) for c in ctrl.values()]
    ev = panel.eligible.values
    out = {}
    for h in horizons:
        cum = cums[h]
        b_raw = np.full(cum.shape[0], np.nan)
        b_neu = np.full(cum.shape[0], np.nan)
        for i in range(cum.shape[0]):
            pool = ev[i] & np.isfinite(cum[i])
            for c in cvs:
                pool = pool & np.isfinite(c[i])
            n = int(pool.sum())
            if n < min_obs:
                continue
            d = fire[i][pool].astype(np.float64)
            if d.sum() < 3 or d.sum() > n - 3:
                continue
            y = cum[i][pool]
            try:
                b_raw[i] = np.linalg.lstsq(np.column_stack([np.ones(n), d]), y, rcond=None)[0][1]
                X = np.column_stack([np.ones(n), d] + [c[i][pool] for c in cvs])
                b_neu[i] = np.linalg.lstsq(X, y, rcond=None)[0][1]
            except np.linalg.LinAlgError:
                continue
        m0, _, t0, _ = newey_west_t(b_raw, lag=max(h, 21))
        m1, _, t1, _ = newey_west_t(b_neu, lag=max(h, 21))
        keep = (m1 / m0 * 100) if (np.isfinite(m0) and abs(m0) > 1e-9) else np.nan
        out[h] = {"raw_bp": m0 * 1e4 if np.isfinite(m0) else np.nan, "raw_t": t0,
                  "neutral_bp": m1 * 1e4 if np.isfinite(m1) else np.nan, "neutral_t": t1,
                  "survival_pct": float(keep) if np.isfinite(keep) else np.nan}
    return out
