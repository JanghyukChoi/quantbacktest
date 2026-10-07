"""거래비용 — 일괄 가정 금지, 종목별 실측을 원칙으로.

왜 일괄 가정이 위험한가
  종목마다 스프레드가 다른데 왕복 몇 bp 를 일괄로 가정하면, 회전율이 높은 전략일수록
  비용을 과소평가해 수익률의 부호까지 뒤집힐 수 있다. 일간 회전율이 크면 편도 비용의
  작은 차이가 연 단위로 크게 누적된다.

추정량
  Roll(1984)        체결가 자기공분산. 분봉이 있으면 가장 신뢰할 만하다.
  Corwin-Schultz    고저가 기반. **ETF·저변동 종목에서 0 을 반환할 수 있으니 주의**.
  거래대금 회귀      실측이 없는 종목을 log(bp) = a + b·log(거래대금) 로 보간.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def roll_spread(prices: np.ndarray) -> float:
    """Roll(1984) 실효 스프레드 (가격 단위). 자기공분산이 양수면 NaN."""
    p = np.asarray(prices, dtype=np.float64)
    p = p[np.isfinite(p)]
    if len(p) < 30:
        return np.nan
    d = np.diff(p)
    c = np.cov(d[1:], d[:-1])[0, 1]
    return 2 * np.sqrt(-c) if c < 0 else np.nan


def corwin_schultz(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Corwin-Schultz(2012) 고저가 스프레드 추정 (비율).

    ⚠️ 고저 폭이 좁은 자산(ETF·대형 저변동주)에서는 추정량이 음수가 되어 0 으로 클립된다.
       결과가 전부 0 이면 이 추정량은 쓰지 말고 Roll 이나 거래대금 회귀로 넘어갈 것.
    """
    h, l = np.log(high), np.log(low)
    beta = (h - l) ** 2 + (h.shift(1) - l.shift(1)) ** 2
    h2 = np.maximum(high, high.shift(1))          # 원소별 — pd.concat(axis=1) 은 컬럼을 두 배로 늘린다
    l2 = np.minimum(low, low.shift(1))
    gamma = (np.log(h2) - np.log(l2)) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    return (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0, upper=0.2)


def fit_spread_model(dvol_m: np.ndarray, spread_bp: np.ndarray) -> tuple[float, float]:
    """log(편도 bp) = a + b·log(거래대금 M) 회귀. 실측 없는 종목 보간용."""
    m = np.isfinite(dvol_m) & np.isfinite(spread_bp) & (dvol_m > 0) & (spread_bp > 0)
    if m.sum() < 20:
        return np.nan, np.nan
    b, a = np.polyfit(np.log(dvol_m[m]), np.log(spread_bp[m]), 1)
    return float(a), float(b)


def spread_panel(adv: pd.DataFrame, *, a: float, b: float,
                 measured: dict[str, float] | None = None,
                 lo: float = 0.05, hi: float = 50.0) -> pd.DataFrame:
    """(date × ticker) 편도 스프레드 패널 (bp). 실측이 있으면 덮어쓴다."""
    dv_m = (adv / 1e6).clip(lower=0.1)
    est = np.exp(a + b * np.log(dv_m))
    if measured:
        for k, v in measured.items():
            if k in est.columns and np.isfinite(v):
                est[k] = v
    return est.clip(lower=lo, upper=hi)


def apply_turnover_cost(holdings: np.ndarray, spread_bp: np.ndarray | float) -> np.ndarray:
    """회전율 비례 비용. 반환은 일별 비용(수익률 단위).

    holdings   (date × ticker) 비중. 합=1 (롱) 또는 롱숏 각각.
    spread_bp  스칼라(일괄) 또는 (date × ticker) 실측 패널. **패널 사용을 권장**.
    """
    d = np.abs(np.diff(holdings, axis=0))
    cost = np.zeros(holdings.shape[0])
    if np.isscalar(spread_bp):
        cost[1:] = d.sum(axis=1) * (spread_bp / 1e4) * 0.5
    else:
        s = np.nan_to_num(np.asarray(spread_bp), nan=float(np.nanmedian(spread_bp)))
        cost[1:] = (d * s[1:] / 1e4).sum(axis=1)
    return cost


def turnover(holdings: np.ndarray) -> np.ndarray:
    """일별 편도 회전율."""
    t = np.zeros(holdings.shape[0])
    t[1:] = 0.5 * np.abs(np.diff(holdings, axis=0)).sum(axis=1)
    return t
