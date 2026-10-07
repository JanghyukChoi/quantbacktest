"""A · 팩터 스크리닝 + 중립화 엔진 — 리서치 단계의 첫 관문.

이 단계가 결론을 가른다. 통계 문턱을 넘은 후보도 비용·시기·단조성 게이트와
  재무 특성 중립화를 거치면 대부분 사라지는 것이 정상이다.

원칙
  1. 무통제 결과는 리턴하되 **기본 요약에서 제외**한다. 보고는 통제 후만.
  2. 재무 특성이 없으면 경고한다 (ROA·BM 누락이 오판의 주원인).
  3. 다중검정 문턱은 **셔플 귀무로 실측**한다.
  4. 격자 최고값이 아니라 **전 분포**를 리턴한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core.controls import build_controls, neutralize, xs_norm
from .core.estimators import (decile_profile, fama_macbeth, newey_west_t,
                              shuffle_columns, shuffle_null)
from .core.gates import GateConfig, fire_structure, run_gates
from .core.panel import Panel


@dataclass
class ScreenResult:
    null: dict
    factors: dict
    survivors: list[str]
    funnel: dict
    summary: pd.DataFrame


def _deploy(fire: np.ndarray, cum: np.ndarray, ev: np.ndarray,
            cost_bp: float) -> tuple[np.ndarray, dict]:
    # 유니버스 크기에 적응 — 고정 문턱은 소형 유니버스에서 전 날짜를 버린다
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
           fire_q: float = 0.10, n_null: int = 2,
           gate_config: GateConfig | None = None,
           neutralize_all: bool = True) -> ScreenResult:
    """팩터 전수 스크리닝.

    factors  {이름: (date × ticker) 연속 팩터}
    반환 summary 는 **통제 후 지표만** 담는다 (무통제는 factors[name]['raw'] 에).
    """
    el = panel.eligible
    ev = el.values
    ctrl = build_controls(panel)
    fwd = {h: panel.forward(h) for h in horizons}
    cums = {h: fwd[h].values.astype(np.float64) for h in horizons}

    # 1) 셔플 귀무로 문턱 실측 — 팩터 개수만큼 돌리면 비싸므로 대표 팩터로 추정
    rep = next(iter(factors.values()))
    null = shuffle_null(lambda rng: xs_norm(shuffle_columns(rep, el, rng), el),
                        el, {primary_h: fwd[primary_h]}, ctrl, n_rep=n_null)
    thr = null.get("p95", np.nan)
    if not np.isfinite(thr):
        thr = 3.0
    cfg = gate_config or GateConfig(null_threshold=thr)

    rows, detail = [], {}
    for name, raw in factors.items():
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
            "t": fm[primary_h]["t"],                       # 통제 후
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
        "전체": n,
        f"G1 |t|>{thr:.2f}": int((df.t.abs() > thr).sum()),
        "G2 비용후 순수익>0": int(((df.t.abs() > thr) & (df.net_bp > 0)).sum()),
        "최종 통과": len(surv),
        "중립화 후 잔존>50%": int((df["neu_survival_%"] > 50).sum()) if neutralize_all else None,
    }
    return ScreenResult(null=null, factors=detail, survivors=surv, funnel=funnel, summary=df)
