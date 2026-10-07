"""C · 이벤트형 시그널 검정 — 개별 알림으로 성립하는가.

포트폴리오 지표(평균 초과수익)는 평균이 실현된다고 가정한다. 그러나 알림은 사용자가
건별로 받으므로 **승률·중앙값·손익비**가 실제 체감을 결정한다.

반드시 분해하는 것
  승률 = 기저승률(표본을 어떻게 골랐나) + lift(신호가 더한 것)
    · 보유가 길수록 승률과 기저가 함께 오른다. lift 로만 판단해야 한다.
    · 승률이 기저승률보다 낮으면(lift < 0) 신호가 오히려 해로운 것이다.
  평균 vs 중앙값
    · 부호가 갈리면 소수 대박이 다수 손실을 덮는 구조다. 포트폴리오는 되지만 알림은 안 된다.
    · 평균은 양수인데 중앙값이 음수인 경우가 대표적이다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core.controls import build_controls, xs_norm
from .core.estimators import fama_macbeth, newey_west_t, paired_diff
from .core.gates import GateConfig, fire_structure, run_gates
from .core.panel import Panel


@dataclass
class EventResult:
    spec: dict
    per_horizon: dict
    gates: dict
    structure: dict
    neutralized: dict | None
    lookahead: dict


def _trade_stats(fire: np.ndarray, cum: np.ndarray, eligible: np.ndarray,
                 cost: float) -> dict | None:
    """건별 손익 분포 + 기저승률."""
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
    """날짜별 발화군 초과수익 (게이트 입력).

    min_pool 을 고정하면 소형 유니버스(예: 20종목 테스트)에서 전 날짜가 버려진다.
    기본값은 유니버스 중앙 크기의 절반(최대 30)으로 **적응**시킨다.
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


def backtest_event(panel: Panel, signal: pd.DataFrame, *,
                   horizons: tuple[int, ...] = (1, 2, 5, 10, 21),
                   cost_bp: float = 20.0,
                   neutralize_check: bool = True,
                   gate_config: GateConfig | None = None,
                   null_threshold: float | None = None) -> EventResult:
    """이벤트형 시그널 검정.

    signal   bool 또는 0/1 매트릭스 (date × ticker). True = 그날 발화.
    cost_bp  왕복 거래비용. 종목별 실측이 있으면 portfolio 쪽 spread_panel 을 쓸 것.

    중립화 검정이 켜져 있으면(기본) **더미 회귀**로 특성 통제 후 기여를 재측정한다.
    이벤트 신호는 연속 팩터가 아니므로 잔차화가 아니라 더미 계수로 본다.
    """
    sig = signal.reindex(index=panel.dates, columns=panel.tickers).fillna(False)
    fire = (sig.astype(bool) & panel.eligible).values
    ev = panel.eligible.values
    cost = cost_bp / 1e4

    lookahead = panel.assert_no_lookahead(sig.astype(float), h=max(horizons))
    cums = {h: panel.forward(h).values.astype(np.float64) for h in horizons}

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
            "검정 가능한 호라이즌이 없습니다. 발화가 너무 드물거나(건수<100) "
            "eligible 종목이 부족합니다. signal·eligible·horizons 를 확인하십시오.")
    # 대표 호라이즌 = lift 가 최대인 곳
    best_h = max(per_h, key=lambda h: per_h[h].get("lift_pp", -99))
    de = _daily_excess(fire, cums[best_h], ev)
    if len(de) < 100:
        warnings.warn(
            f"날짜별 초과수익 표본이 {len(de)}일뿐이라 게이트를 신뢰할 수 없습니다. "
            f"_daily_excess 는 발화 3종목 이상 & 유니버스 30종목 이상인 날만 셉니다 — "
            f"유니버스가 작으면(예: 20종목) 대부분의 날이 제외됩니다.", stacklevel=2)

    from .core.estimators import decile_profile
    dec = decile_profile(sig.astype(float), panel.forward(best_h), panel.eligible,
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
    """이벤트 더미 회귀 — 특성 통제 후에도 발화가 수익률을 예측하는가.

    표본: 그날 eligible 전체. 종속: 미래수익. 독립: [발화더미] + 통제변수.
    더미 계수가 곧 '같은 특성에서 발화가 더하는 것'이다.
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
