"""통제변수 — 가격 7종 + 재무 특성 5종.

왜 재무 특성이 기본값인가
  가격 통제만 넣고 검정하면 **남의 알파를 자기 것으로 착각**한다. 수익성·밸류 같은 재무 특성을
  통제하지 않으면 겉보기 알파가 크게 줄거나 부호가 바뀌는 일이 흔하다
  (tests/test_synthetic.py 의 T6 이 통제변수 자체를 팩터로 넣어 이 소멸을 재현한다).
  가격 통제(mom20/mom120)만으로는 표준 모멘텀(12-1개월)·수익성·밸류를 못 잡는다.
  그래서 chars 가 있으면 **자동으로 포함**하고, 없으면 경고한다.

정규화
  전부 날짜별 백분위 랭크 → 평균0·분산1. 이상치에 강건하고 계수를 '1SD당 bp' 로 읽게 한다.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

EPS = 1e-12
PRICE_CONTROLS = ("rev1", "rev5", "mom21", "mom63", "mom252_21", "logsize", "turnover", "vol21")
CHAR_NAMES = ("log_size", "log_bm", "momentum", "roa", "asset_growth")


def xs_rank(v: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """날짜별 eligible 내 백분위 랭크 [0,1]. 횡단면 연산이라 시장공통 성분이 자동 제거된다."""
    return v.where(eligible).rank(axis=1, pct=True, na_option="keep")


def xs_norm(v: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """랭크 → 평균0·분산1 스케일."""
    return ((xs_rank(v, eligible) - 0.5) * np.sqrt(12.0)).astype(np.float32)


def build_controls(panel, *, include_chars: bool = True) -> dict[str, pd.DataFrame]:
    """통제변수 묶음. panel.chars 가 있으면 재무 특성도 포함한다."""
    close, el = panel.close, panel.eligible
    ret = close.pct_change()
    out: dict[str, pd.DataFrame] = {
        # rev1 필수 — 역추세 매매자의 수급은 곧 '당일 하락'을 대리한다.
        # 1일 반전은 단기 최강 효과이므로 빼면 신호가 아니라 반전을 재발견하게 된다.
        "rev1": ret,
        "rev5": close.pct_change(5),
        "mom21": close.pct_change(21),
        "mom63": close.pct_change(63),
        # 표준 모멘텀 12-1 — mom21/63 만으로는 이걸 못 잡는다
        "mom252_21": close.pct_change(getattr(panel, "periods_per_year", 252)) - close.pct_change(21),
        "vol21": ret.rolling(21, min_periods=21).std(),
    }
    if panel.mkt_cap is not None:
        out["logsize"] = np.log(panel.mkt_cap.replace(0, np.nan))
        adv = panel.adv(20)
        if adv is not None:
            out["turnover"] = adv / (panel.mkt_cap + EPS)

    if include_chars:
        if not panel.chars:
            warnings.warn(
                "재무 특성(chars)이 없습니다. 가격 통제만으로는 ROA·BM 노출을 못 잡아 "
                "'남의 알파'를 고유 알파로 오판할 수 있습니다. "
                "adapters 로 chars 를 채우거나 include_chars=False 로 명시하십시오.",
                stacklevel=2,
            )
        for k, v in panel.chars.items():
            out[f"char_{k}"] = v

    return {k: xs_norm(v, el).astype(np.float32) for k, v in out.items() if v is not None}


def neutralize(factor: pd.DataFrame, controls: dict[str, pd.DataFrame],
               eligible: pd.DataFrame, *, min_obs: int = 50) -> pd.DataFrame:
    """날짜별 횡단면 회귀 잔차 — 통제변수와 직교화.

    연속 팩터용. 이벤트(0/1) 신호는 event.py 의 더미 회귀를 쓴다.
    잔차가 팩터 크기의 1e-5 미만이면(통제변수가 팩터를 거의 완전히 설명) 그날은 NaN 이다.
    """
    fv = factor.values.astype(np.float64)
    ev = eligible.values
    cvs = [c.values.astype(np.float64) for c in controls.values()]
    out = np.full(fv.shape, np.nan)
    for i in range(fv.shape[0]):
        ok = ev[i] & np.isfinite(fv[i])
        for c in cvs:
            ok &= np.isfinite(c[i])
        if ok.sum() < min_obs:
            continue
        y = fv[i][ok]
        X = np.column_stack([np.ones(ok.sum())] + [c[i][ok] for c in cvs])
        try:
            Q, _ = np.linalg.qr(X)
        except np.linalg.LinAlgError:
            continue
        res = y - Q @ (Q.T @ y)
        # A factor inside the span of the controls leaves only rounding noise (about 1e-8 of its size in float32).
        # Left in, a later rank-normalisation would blow that noise up to unit scale and the result would depend on the
        # BLAS build. No information is left, so the day stays NaN.
        if not (res.std() > 1e-5 * y.std()):
            continue
        out[i][ok] = res
    return pd.DataFrame(out, index=factor.index, columns=factor.columns)


def build_chars_from_financials(close: pd.DataFrame, mkt_cap: pd.DataFrame,
                                book_equity: pd.DataFrame | None = None,
                                net_income: pd.DataFrame | None = None,
                                total_assets: pd.DataFrame | None = None) -> dict:
    """재무 특성 5종 — Fama-French 계열 표준 정의.

    분기 재무는 일별로 forward-fill 된 상태로 넘긴다 (어댑터 책임).
    """
    chars = {"log_size": np.log(mkt_cap.clip(lower=1)),
             "momentum": close.pct_change(252) - close.pct_change(21)}
    if book_equity is not None:
        chars["log_bm"] = np.log((book_equity / mkt_cap.replace(0, np.nan)).clip(lower=1e-6))
    if net_income is not None and total_assets is not None:
        chars["roa"] = net_income / total_assets.replace(0, np.nan)
    if total_assets is not None:
        chars["asset_growth"] = total_assets.pct_change(252)
    return chars
