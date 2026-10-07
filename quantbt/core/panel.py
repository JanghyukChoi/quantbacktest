"""패널 규약 — 모든 검정의 입력 계약.

설계 원칙
  · 코어는 **순수 pandas/numpy**. 외부 데이터 소스에 의존하지 않는다 (어댑터가 담당).
  · 모든 매트릭스는 (date × ticker) 이고 index/columns 가 동일해야 한다.
  · `eligible` 은 **PIT 유니버스**다. 그날 실제 거래 가능했던 종목만 True.
    생존편향을 막는 유일한 장치이므로 필수 입력이다.

시점 규약 (이 프로젝트 전체에서 단 하나만 쓴다)
  신호는 close(t) 에 **확정**된다. 진입은 close(t+entry_lag), 청산은 close(t+entry_lag+h).
  entry_lag 기본 1 — "신호를 보고 다음 날 산다". 0 은 같은 종가 체결 가정(공격적).
  이 규약을 어기면 Panel.assert_no_lookahead() 가 잡는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED = ("close", "eligible")
OPTIONAL = ("open", "high", "low", "volume", "mkt_cap", "funding", "delist_after")


@dataclass
class Panel:
    """(date × ticker) 매트릭스 묶음.

    close    수정주가 종가 — 수익률의 유일한 원천
    eligible PIT 유니버스 bool. 그날 상장·거래가능·유동성조건 충족
    나머지는 선택. volume·mkt_cap 이 있으면 유동성/규모 통제가 켜진다.
    """

    close: pd.DataFrame
    eligible: pd.DataFrame
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    mkt_cap: pd.DataFrame | None = None
    chars: dict[str, pd.DataFrame] = field(default_factory=dict)
    market: str = "KR"
    entry_lag: int = 1
    # 연율화에 쓰는 연간 거래일. 주식 252, 24/7 시장(크립토)은 365 로 둔다. 잘못 두면 CAGR, 샤프, 변동성이 틀린다.
    periods_per_year: int = 252
    # 선택: 선물 펀딩비(날짜 x 종목, 일합계, 양수면 롱이 지불). 상장폐지 표시(마지막 실제 봉이면 True). 어댑터 메모.
    funding: pd.DataFrame | None = None
    delist_after: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- 생성·검증
    def __post_init__(self) -> None:
        self.close = self.close.sort_index()
        self.eligible = self.eligible.reindex(
            index=self.close.index, columns=self.close.columns
        ).fillna(False).astype(bool)
        for k in OPTIONAL:
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.reindex(index=self.close.index, columns=self.close.columns))
        self.chars = {
            k: v.reindex(index=self.close.index, columns=self.close.columns)
            for k, v in self.chars.items()
        }
        if self.delist_after is not None:
            self.delist_after = self.delist_after.fillna(False).astype(bool)
        self.validate()

    def validate(self) -> None:
        if not self.close.index.is_monotonic_increasing:
            raise ValueError("close.index 가 오름차순이 아닙니다")
        if self.close.index.has_duplicates:
            raise ValueError("close.index 에 중복 날짜가 있습니다")
        if self.eligible.values.sum() == 0:
            raise ValueError("eligible 이 전부 False 입니다")
        if self.entry_lag < 0:
            raise ValueError("entry_lag 는 0 이상이어야 합니다")
        n = self.eligible.sum(axis=1)
        if (n[n > 0] < 10).mean() > 0.5:
            raise ValueError(
                "eligible 종목이 절반 이상의 날짜에서 10개 미만입니다 — 횡단면 검정이 불가능합니다"
            )

    # ---------------------------------------------------------------- 파생
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    def ret1(self) -> pd.DataFrame:
        """일간 수익률 close(t)/close(t-1) − 1."""
        return self.close.pct_change(fill_method=None)

    def forward(self, h: int) -> pd.DataFrame:
        """신호 t 기준 미래 h일 누적 수익률.

        entry_lag=1 이면 close(t+1) 진입 → close(t+1+h) 청산.
        로그 누적 후 되돌려 중간 결측이 있어도 안전하게 합산한다.
        """
        lag = self.entry_lag
        entry = self.close.shift(-lag)
        exit_ = self.close.shift(-(lag + h))
        return (exit_ / entry - 1.0).astype(np.float32)

    def adv(self, window: int = 20) -> pd.DataFrame:
        """평균 거래대금. volume 이 없으면 None."""
        if self.volume is None:
            return None
        return (self.close * self.volume).rolling(window, min_periods=window).mean()

    # ---------------------------------------------------------------- 안전장치
    def assert_no_lookahead(self, signal: pd.DataFrame, h: int = 5,
                            n_null: int = 8, seed: int = 0) -> dict:
        """룩어헤드 탐지 — 신호를 하루 **뒤로** 밀었을 때 성과가 오르면 미래를 보고 있다.

        정상 신호는 시점을 늦추면 성과가 떨어진다(정보가 소멸하므로).
        늦췄는데 오르면 그 신호는 이미 미래 정보를 담고 있다는 뜻이다.

        ⚠️ 단순히 `lagged > base` 로 판정하면 **무작위 팩터가 50% 확률로 오판**된다.
           둘 다 0 근처라 부호가 우연히 갈리기 때문. 그래서 무작위 셔플로 스프레드의
           노이즈 규모(sd)를 실측하고, 그 2σ 를 허용오차로 쓴다.
        """
        fwd = self.forward(h)
        base = _spread(signal, fwd, self.eligible)
        lagged = _spread(signal.shift(1), fwd, self.eligible)
        rng = np.random.default_rng(seed)
        nulls = []
        for _ in range(n_null):
            sh = signal.values.copy()
            ev = self.eligible.values
            for i in range(sh.shape[0]):
                j = np.where(ev[i])[0]
                if len(j) > 1:
                    sh[i, j] = sh[i, rng.permutation(j)]
            v = _spread(pd.DataFrame(sh, index=signal.index, columns=signal.columns),
                        fwd, self.eligible)
            if np.isfinite(v):
                nulls.append(v)
        tol = 2 * float(np.std(nulls)) if len(nulls) >= 3 else 0.0
        ok = not (np.isfinite(base) and np.isfinite(lagged) and lagged > base + tol)
        return {"base_bp": base * 1e4, "lagged_bp": lagged * 1e4,
                "tol_bp": tol * 1e4, "pass": bool(ok)}

    def audit(self) -> dict:
        """데이터 무결성 감사 — 검정 전에 반드시 한 번 돌린다.

        흔한 데이터 결함을 잡는다:
          · 조용한 절단   (특정 날짜 이후 데이터가 통째로 사라짐)
          · 수정주가 불일치 (갭 극단값이 일간 변동보다 많음 — adj_open 결함 유형)
          · 유니버스 급변  (모집단이 시간에 따라 3배 늘어나는 등)
        """
        out = {}
        n = self.eligible.sum(axis=1)
        out["dates"] = len(self.dates)
        out["tickers"] = int(self.eligible.any(axis=0).sum())
        out["eligible_mean"] = float(n.mean())
        yr = n.groupby(self.dates.year).mean()
        out["eligible_by_year"] = {int(k): float(v) for k, v in yr.items()}
        out["universe_growth"] = float(yr.iloc[-1] / yr.iloc[0]) if len(yr) > 1 and yr.iloc[0] else np.nan

        r = self.ret1().where(self.eligible)
        out["ret_extreme_pct"] = float((r.abs() > 0.30).mean().mean() * 100)
        gap = None
        if self.open is not None:
            gap = (self.open / self.close.shift(1) - 1.0).where(self.eligible)
            out["gap_extreme_pct"] = float((gap.abs() > 0.30).mean().mean() * 100)
            ratio = out["gap_extreme_pct"] / max(out["ret_extreme_pct"], 1e-9)
            out["gap_vs_daily_ratio"] = float(ratio)
            out["adj_price_consistent"] = bool(ratio < 3.0)

        # 조용한 절단: 종목 커버리지가 특정 시점 이후 급감
        cov = self.close.notna().sum(axis=1)
        if len(cov) > 60:
            tail = cov.iloc[-20:].mean(); body = cov.iloc[:-20].median()
            out["tail_coverage_ratio"] = float(tail / body) if body else np.nan
            out["truncation_suspected"] = bool(tail < body * 0.5)

        # 생존편향 의심: 실제 시장에서는 해마다 일정 비율의 종목이 사라진다. 패널 끝 이전에 가격이 끊긴 종목이
        # 거의 없다면 그 패널은 '오늘 살아 있는 종목'만 모은 것이다. 측정: yfinance 로 상장폐지·인수된 유명 종목
        # 42개를 조회했더니 정확한 이력을 돌려준 것이 0개였다(docs/survivorship.md).
        last = self.close.apply(lambda c: c.last_valid_index())
        ended = last.dropna() < (self.dates[-1] - pd.Timedelta(days=30))
        span_years = (self.dates[-1] - self.dates[0]).days / 365.25
        out["ended_before_end_share"] = float(ended.mean()) if len(ended) else np.nan
        out["survivorship_suspected"] = bool(len(ended) >= 30 and span_years >= 3 and ended.mean() < 0.01)
        return out


def _spread(signal: pd.DataFrame, fwd: pd.DataFrame, eligible: pd.DataFrame,
            q: float = 0.10) -> float:
    """상위 q − 하위 q 스프레드 평균 (룩어헤드 검사용 내부 헬퍼)."""
    rk = signal.where(eligible).rank(axis=1, pct=True, na_option="keep")
    hi, lo = ((rk >= 1 - q) & eligible).values, ((rk <= q) & eligible).values
    c = fwd.values.astype(np.float64)
    out = []
    for i in range(c.shape[0]):
        a, b = hi[i] & np.isfinite(c[i]), lo[i] & np.isfinite(c[i])
        if a.sum() >= 3 and b.sum() >= 3:
            out.append(c[i][a].mean() - c[i][b].mean())
    return float(np.mean(out)) if out else np.nan


def build_pit_eligible(close: pd.DataFrame, *, listed: pd.DataFrame | None = None,
                       min_adv: float = 0.0, volume: pd.DataFrame | None = None,
                       exclude: pd.DataFrame | None = None) -> pd.DataFrame:
    """PIT 유니버스 마스크 생성 헬퍼.

    listed   상장 여부 bool (없으면 close 존재 여부로 대체)
    min_adv  최소 평균거래대금 (volume 필요)
    exclude  관리종목·거래정지 등 제외 마스크 (True = 제외)
    """
    ok = close.notna() & (close > 0)
    if listed is not None:
        ok &= listed.reindex_like(close).fillna(False).astype(bool)
    if min_adv > 0 and volume is not None:
        adv = (close * volume).rolling(20, min_periods=20).mean()
        ok &= adv >= min_adv
    if exclude is not None:
        ok &= ~exclude.reindex_like(close).fillna(False).astype(bool)
    return ok.fillna(False)
