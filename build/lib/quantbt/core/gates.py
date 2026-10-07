"""게이트 러너 — 결과를 보기 전에 고정하는 판정 관문.

게이트 순서에 의미가 있다. 통계 → 비용 → 안정성 → 구조 순으로,
**뒤로 갈수록 통과가 어렵다**. 통계 게이트만 두면 후보가 많이 살아남지만
비용·안정성·구조 게이트를 거치면 크게 줄어드는 것이 정상이다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .estimators import newey_west_t


@dataclass
class GateConfig:
    """게이트 문턱. 검정 시작 전에 고정하고 이후 바꾸지 않는다."""
    null_threshold: float | None = None   # None 이면 셔플 귀무로 실측
    fallback_t: float = 3.0               # 귀무 실측이 불가할 때만
    require_net_positive: bool = True     # 비용 후 순수익 > 0
    require_subperiod_sign: bool = True   # 전·후반 부호 일치
    min_monotonicity: float = 0.5         # |rho|
    min_yearly_positive: float = 0.6      # 양수 연도 비율
    max_single_name_share: float = 0.5    # 최다 종목 발화 비율 (정적 바스켓)
    min_turnover: float = 0.05            # 일간 회전율 하한


def fire_structure(fire: np.ndarray, dates: pd.DatetimeIndex,
                   tickers: pd.Index) -> dict:
    """발화 구조 — 시그널인가 정적 바스켓인가.

    여러 해 동안 같은 대형주만 계속 뽑는 팩터는 시그널이 아니라 사이즈 팩터(정적 바스켓)다.
    최다 종목 비율과 최대 연속 발화일로 이를 잡는다.
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
                "note": "발화 0건"}
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
    """전·후반 분할. 부호가 갈리면 시기 의존이다."""
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
    """연도별 일관성."""
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
    """OOS 홀드아웃 — 최근 N개월을 떼어 검정.

    표본이 작으면 t 가 낮게 나오는 게 정상이다. '죽었다'와 '못 잰다'를 구분하려면
    n_days 를 함께 봐야 한다.
    """
    values = np.asarray(values)
    if len(values) == 0:
        return {"IS": {"mean_bp": np.nan, "t": np.nan, "n": 0},
                "OOS": {"mean_bp": np.nan, "t": np.nan, "n": 0},
                "cutoff": None, "note": "표본 없음 — 발화가 너무 드물거나 유니버스가 작습니다"}
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
    """전 게이트를 순서대로 통과시키고 어디서 탈락했는지 기록한다."""
    cfg = cfg or GateConfig()
    thr = cfg.null_threshold if cfg.null_threshold is not None else cfg.fallback_t
    res = {"config": asdict(cfg), "threshold": thr}
    steps: list[tuple[str, bool, str]] = []

    ok1 = abs(t_stat) > thr
    steps.append(("G1 통계 문턱", ok1, f"|t|={abs(t_stat):.2f} vs {thr:.2f}"))

    ok2 = (not cfg.require_net_positive) or (np.isfinite(net_bp) and net_bp > 0)
    steps.append(("G2 비용 후 순수익", ok2, f"{net_bp:+.1f}bp"))

    sp = subperiod(daily_excess, dates, lag=lag)
    ok3 = (not cfg.require_subperiod_sign) or sp["sign_match"]
    steps.append(("G3 시기분할 부호", ok3,
                  f"전 {sp['front']['mean_bp']:+.1f} / 후 {sp['back']['mean_bp']:+.1f}"))

    ok4 = np.isfinite(rho) and abs(rho) >= cfg.min_monotonicity
    steps.append(("G4 십분위 단조성", ok4, f"rho={rho:+.2f}"))

    yr = yearly(daily_excess, dates, lag=lag)
    ok5 = np.isfinite(yr["positive_ratio"]) and yr["positive_ratio"] >= cfg.min_yearly_positive
    steps.append(("G5 연도 일관성", ok5, f"{yr['positive']}/{yr['total']}년"))

    st = None
    if fire is not None and tickers is not None:
        st = fire_structure(fire, dates, tickers)
        ok6 = (st["max_name_share"] <= cfg.max_single_name_share
               and st["daily_turnover"] >= cfg.min_turnover)
        steps.append(("G6 발화 구조", ok6,
                      f"최다종목 {st['max_name_share']*100:.1f}% · 회전율 {st['daily_turnover']*100:.1f}%"))

    res["steps"] = [{"gate": g, "pass": bool(p), "detail": d} for g, p, d in steps]
    res["passed"] = all(p for _, p, _ in steps)
    res["failed_at"] = next((g for g, p, _ in steps if not p), None)
    res["subperiod"], res["yearly"], res["structure"] = sp, yr, st
    res["oos"] = oos_holdout(daily_excess, dates, lag=lag)
    return res
