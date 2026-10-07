"""B · 포트폴리오 알파 백테스트 — 신호를 운용 결과로.

시점 규약 (버그가 난 지점이므로 명시)
  holdings[d] = t = d-hold+1 … d 신호의 트랜치 평균. 신호는 close(d) 확정,
  진입 close(d+entry_lag), 수익은 close(d+entry_lag) → close(d+entry_lag+1).
  진입이 하루 늦어지는 off-by-one 버그는 1일 평균회귀 전략에서는 수익 대부분을 지운다.
  assert_timing() 이 이를 잡는다.

비용
  회전율 비례. **종목별 실측 스프레드 패널 사용을 권장**한다.
  일괄 가정은 회전율이 높은 전략의 성과를 크게 부풀릴 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core.costs import apply_turnover_cost, turnover
from .core.panel import Panel

ANN = 252


@dataclass
class PortfolioResult:
    spec: dict
    metrics: dict
    benchmark: dict | None
    excess: dict | None
    yearly: dict
    grid: dict | None


def _tranche(weights: np.ndarray, hold: int) -> np.ndarray:
    """중첩 트랜치: d일 보유 = t = d-hold+1 … d 신호의 평균 (매일 자본 1/hold 씩 진입)."""
    T, N = weights.shape
    out = np.zeros((T, N))
    run = np.zeros(N)
    cnt = 0
    for d in range(T):
        run += weights[d]
        cnt += 1
        if d - hold >= 0:
            run -= weights[d - hold]
            cnt -= 1
        if cnt > 0:
            out[d] = run / cnt
    return out


def _normalize(mask_or_w: np.ndarray) -> np.ndarray:
    w = np.asarray(mask_or_w, dtype=np.float64)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = np.nan
    return np.where(np.isfinite(s), w / s, 0.0)


def metrics(net: np.ndarray, dates: pd.DatetimeIndex, ann: int = ANN) -> dict:
    s = pd.Series(net, index=dates[: len(net)]).dropna()
    if len(s) < ann // 2:
        return {"CAGR": np.nan, "MDD": np.nan, "Sharpe": np.nan}
    eq = (1 + s).cumprod()
    yrs = len(s) / ann
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:]
    recov = rec[rec >= eq.loc[peak]].index
    return {"CAGR": float(cagr), "MDD": float(mdd),
            "Sharpe": float(s.mean() / (s.std() + 1e-12) * np.sqrt(ann)),
            "Sortino": float(s.mean() * ann / (s[s < 0].std() * np.sqrt(ann) + 1e-12)),
            "Calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
            "vol": float(s.std() * np.sqrt(ann)),
            "years": float(yrs), "pos_days": float((s > 0).mean()),
            "mdd_peak": str(peak.date()), "mdd_trough": str(trough.date()),
            "mdd_recovered": str(recov[0].date()) if len(recov) else "미회복"}


def backtest_portfolio(panel: Panel, factor: pd.DataFrame, *,
                       long_q: float = 0.10, short_q: float | None = 0.10,
                       hold: int = 5, weighting: str = "equal",
                       spread_bp=20.0, benchmark: str | None = "cap",
                       grid: bool = True) -> PortfolioResult:
    """팩터 → 포트폴리오 성과.

    factor      연속 팩터 (높을수록 롱). bool 이면 롱온리 발화로 해석.
    long_q      롱 분위 (상위 q). short_q=None 이면 롱온리.
    weighting   equal | signal (신호강도 비례) | rank
    spread_bp   스칼라 또는 (date × ticker) 편도 스프레드 패널
    benchmark   cap(시총가중) | equal(동일가중) | None
    """
    f = factor.reindex(index=panel.dates, columns=panel.tickers)
    el = panel.eligible
    rk = f.where(el).rank(axis=1, pct=True, na_option="keep")
    ret = np.nan_to_num(panel.ret1().values.astype(np.float64), nan=0.0)
    # 신호 d → 진입 close(d+lag) → 수익 close(d+lag)→close(d+lag+1)
    fwd = np.roll(ret, -(panel.entry_lag + 1), axis=0)
    fwd[-(panel.entry_lag + 1):] = 0.0
    ev = el.values

    def leg(q: float, top: bool) -> np.ndarray:
        m = ((rk >= 1 - q) if top else (rk <= q)) & el
        mv = m.values
        if weighting == "equal":
            w = mv.astype(float)
        elif weighting == "signal":
            w = np.where(mv, np.abs(np.nan_to_num(f.values, nan=0.0)), 0.0)
        else:  # rank
            r = np.nan_to_num(rk.values, nan=0.0)
            w = np.where(mv, (r - (1 - q)) if top else (q - r), 0.0)
        return _normalize(w)

    hl = _tranche(leg(long_q, True), hold)
    gross = (hl * fwd).sum(axis=1)
    cost = apply_turnover_cost(hl, spread_bp)
    turn = turnover(hl)
    if short_q:
        hs = _tranche(leg(short_q, False), hold)
        gross = gross - (hs * fwd).sum(axis=1)
        cost = cost + apply_turnover_cost(hs, spread_bp)
        turn = turn + turnover(hs)
    net = gross - cost

    cut = len(panel.dates) - (panel.entry_lag + 1)
    m = metrics(net[:cut], panel.dates, panel.periods_per_year)
    m["turnover_daily"] = float(turn[:cut].mean())
    m["cost_annual_bp"] = float(cost[:cut].mean() * panel.periods_per_year * 1e4)
    m["gross_CAGR"] = metrics(gross[:cut], panel.dates, panel.periods_per_year)["CAGR"]
    m["avg_positions"] = float((hl > 0).sum(axis=1)[(hl > 0).sum(axis=1) > 0].mean())

    bench = bexc = None
    if benchmark:
        if benchmark == "cap" and panel.mkt_cap is not None:
            w = np.where(ev, np.nan_to_num(panel.mkt_cap.values, nan=0.0), 0.0)
        else:
            w = ev.astype(float)
        bw = _normalize(w)
        br = (bw * fwd).sum(axis=1)
        bench = metrics(br[:cut], panel.dates, panel.periods_per_year)
        bexc = metrics((net - br)[:cut], panel.dates, panel.periods_per_year)

    s = pd.Series(net[:cut], index=panel.dates[:cut])
    yr = s.groupby(s.index.year).apply(lambda g: float((1 + g).prod() - 1))

    g = None
    if grid:
        g = {}
        for h in (1, 2, 5, 10, 21):
            for c in (0, 2, 5, 10, 20):
                hl2 = _tranche(leg(long_q, True), h)
                gr = (hl2 * fwd).sum(axis=1)
                co = apply_turnover_cost(hl2, c)
                if short_q:
                    hs2 = _tranche(leg(short_q, False), h)
                    gr = gr - (hs2 * fwd).sum(axis=1)
                    co = co + apply_turnover_cost(hs2, c)
                mm = metrics((gr - co)[:cut], panel.dates, panel.periods_per_year)
                g[f"h{h}_c{c}"] = {"CAGR": mm["CAGR"], "Sharpe": mm["Sharpe"], "MDD": mm["MDD"]}

    return PortfolioResult(
        spec={"long_q": long_q, "short_q": short_q, "hold": hold, "weighting": weighting,
              "entry_lag": panel.entry_lag, "market": panel.market,
              "spread": "panel" if not np.isscalar(spread_bp) else f"{spread_bp}bp 일괄"},
        metrics=m, benchmark=bench, excess=bexc,
        yearly={str(k): v for k, v in yr.items()}, grid=g)


def assert_timing(panel: Panel) -> dict:
    """시점 정렬 자기검증 — 완전예지 팩터를 넣어 수익이 나오는지 본다.

    미래 수익률을 그대로 팩터로 쓰면 CAGR 이 크게 양수여야 한다. 아니면 시점이 어긋난 것이다.
    (진입 지연 버그를 잡는 회귀 테스트)
    """
    lag = panel.entry_lag
    oracle = panel.close.shift(-(lag + 1)) / panel.close.shift(-lag) - 1.0
    r = backtest_portfolio(panel, oracle, long_q=0.10, short_q=0.10, hold=1,
                           spread_bp=0.0, benchmark=None, grid=False)
    ok = np.isfinite(r.metrics["CAGR"]) and r.metrics["CAGR"] > 0.5
    return {"oracle_CAGR": r.metrics["CAGR"], "oracle_Sharpe": r.metrics["Sharpe"],
            "pass": bool(ok),
            "note": "완전예지 팩터가 큰 양수를 내야 정상. 실패면 entry_lag 또는 fwd 정렬 오류."}
