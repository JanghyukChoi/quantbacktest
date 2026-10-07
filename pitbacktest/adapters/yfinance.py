"""공개 데이터 어댑터 — yfinance 로 Panel 을 만든다.

의존성: `pip install yfinance`  (코어에는 불필요, 이 어댑터에서만 쓴다)

사용
    from pitbacktest.adapters.yfinance import load_panel
    panel = load_panel(["AAPL", "MSFT", "NVDA"], "2018-01-01", "2024-12-31")

주의 — 무료 데이터의 한계
  · **생존편향**: 티커 목록을 오늘 기준으로 주면 그동안 상장폐지된 종목이 빠진다.
    PIT 유니버스가 아니므로 성과가 낙관 편향된다. 연구용으로만 쓸 것.
  · **수정주가**: auto_adjust=True 로 배당·분할을 반영한다. 이를 끄면 분할일에
    가짜 수익률이 생긴다.
  · 재무 특성(ROA·BM 등)은 제공하지 않는다. chars 없이 돌리면 경고가 뜨는데,
    그건 "가격 통제만으로는 남의 알파를 걸러내지 못한다"는 정당한 경고다.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..core.panel import Panel, build_pit_eligible


def load_panel(tickers: list[str], start: str, end: str, *,
               min_dollar_volume: float = 1e6,
               market: str = "US", entry_lag: int = 1,
               with_market_cap: bool = True,
               with_chars: bool = False,
               shares_outstanding: dict[str, float] | None = None) -> Panel:
    """yfinance 로 (date × ticker) 패널 구성.

    tickers            조회할 종목. **오늘 상장된 것만 주면 생존편향이 생긴다.**
    min_dollar_volume  eligible 최소 평균거래대금 (20일)
    with_market_cap    get_shares_full(시계열 주식수) × 종가로 시가총액 산출.
                       규모·회전율 통제가 켜진다. 종목당 API 1회라 다소 느리다.
    with_chars         분기 재무로 재무 특성 5종 구성 (log_size·log_bm·momentum·roa·asset_growth).
                       **중립화 검정의 핵심 입력**이다. 종목당 API 2회 추가.
    shares_outstanding 수동 지정 {티커: 주식수}. with_market_cap 실패 시 폴백.
    """
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "yfinance 가 필요합니다: pip install yfinance"
        ) from e

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column")
    if raw is None or raw.empty:
        raise ValueError("yfinance 가 데이터를 반환하지 않았습니다 (티커·기간 확인)")

    def pick(field: str) -> pd.DataFrame | None:
        if isinstance(raw.columns, pd.MultiIndex):
            if field not in raw.columns.get_level_values(0):
                return None
            d = raw[field]
        else:
            if field not in raw.columns:
                return None
            d = raw[[field]].rename(columns={field: tickers[0]})
        return d.sort_index()

    close = pick("Close")
    if close is None:
        raise ValueError("Close 컬럼이 없습니다")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume = pick("Volume")
    high, low, open_ = pick("High"), pick("Low"), pick("Open")
    for d in (volume, high, low, open_):
        if d is not None:
            d.index = close.index

    mkt_cap = None
    if with_market_cap:
        mkt_cap = _market_cap(yf, list(close.columns), close)
    if mkt_cap is None and shares_outstanding:
        so = pd.Series(shares_outstanding).reindex(close.columns)
        if so.notna().any():
            mkt_cap = close.mul(so, axis=1)
    if mkt_cap is None:
        warnings.warn("시가총액을 만들지 못했습니다 — 규모·회전율 통제가 꺼집니다", stacklevel=2)

    chars = {}
    if with_chars:
        if mkt_cap is None:
            warnings.warn("시가총액이 없어 재무 특성을 만들 수 없습니다", stacklevel=2)
        else:
            chars = _characteristics(yf, close, mkt_cap)

    eligible = build_pit_eligible(close, min_adv=min_dollar_volume, volume=volume)
    n_drop = int((~eligible).sum().sum())
    if eligible.values.sum() == 0:
        raise ValueError("eligible 이 전부 False 입니다 — min_dollar_volume 을 낮추십시오")

    warnings.warn(
        f"yfinance 패널은 **PIT 유니버스가 아닙니다**. 오늘 상장된 티커만 조회되므로 "
        f"상장폐지 종목이 빠져 성과가 낙관 편향됩니다. (제외 셀 {n_drop:,}개)",
        stacklevel=2)

    ppy = 365 if market.upper() == "CRYPTO" else 252   # 24/7 시장은 연 365 거래일
    n_ended = sum(1 for c in close.columns if close[c].last_valid_index() is not None
                  and close[c].last_valid_index() < close.index[-1] - pd.Timedelta(days=30))
    if len(close.columns) >= 30 and n_ended == 0:
        warnings.warn("이 패널에는 기간 안에 가격이 끊긴 종목이 하나도 없습니다 — 오늘 살아 있는 종목만 모은 것이라 "
                      "생존편향이 있습니다(yfinance 는 상장폐지, 인수 종목의 과거 가격을 거의 주지 않는다). "
                      "docs/survivorship.md 참고.", stacklevel=2)
    return Panel(close=close, eligible=eligible, open=open_, high=high, low=low,
                 volume=volume, mkt_cap=mkt_cap, chars=chars,
                 market=market, entry_lag=entry_lag, periods_per_year=ppy)


def _market_cap(yf, tickers: list[str], close: pd.DataFrame) -> pd.DataFrame | None:
    """시계열 주식수 × 종가. get_shares_full 은 발행주식수 변화(자사주·증자)를 반영한다."""
    cols = {}
    for t in tickers:
        try:
            sh = yf.Ticker(t).get_shares_full(start=str(close.index[0].date()))
        except Exception:  # noqa: BLE001, PERF203
            continue
        if sh is None or not len(sh):
            continue
        s = pd.Series(sh)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        cols[t] = s.reindex(close.index, method="ffill")
    if not cols:
        return None
    shares = pd.DataFrame(cols).reindex(columns=close.columns)
    got = int(shares.notna().any().sum())
    if got < len(tickers):
        warnings.warn(f"주식수를 {got}/{len(tickers)} 종목만 받았습니다 — "
                      f"나머지는 시가총액이 NaN 이라 통제에서 제외됩니다", stacklevel=3)
    return close * shares


def _characteristics(yf, close: pd.DataFrame, mkt_cap: pd.DataFrame) -> dict:
    """재무 특성 5종 — Fama-French 계열 표준 정의.

    ⚠️ yfinance 의 재무 커버리지는 짧다 — 분기 5~7개, 연간 5개뿐이다.
       2018~2024 같은 장기 백테스트에서는 대부분 구간이 비므로, 분기+연간을 합쳐 쓰고
       **커버리지가 부족하면 해당 특성을 아예 뺀다**(빈 컬럼이 회귀를 망치기 때문).
       장기 연구에는 유료 재무 소스(Compustat·Sharadar 등)를 붙이는 것이 맞다.

    공시 지연: yfinance 는 공시일을 주지 않아 **기말 + 45영업일**로 근사한다.
    """
    LAG = 45
    MIN_COVER = 0.30      # 유효셀 비율이 이 밑이면 특성을 버린다
    ta, be, ni = {}, {}, {}
    for t in close.columns:
        try:
            tk = yf.Ticker(t)
            bs, inc = tk.quarterly_balance_sheet, tk.quarterly_income_stmt
        except Exception:  # noqa: BLE001, PERF203
            continue
        if bs is None or bs.empty:
            continue

        def grab(df, keys):
            if df is None or df.empty:
                return None
            for k in keys:
                hit = [i for i in df.index if k in str(i)]
                if hit:
                    s = df.loc[hit[0]].dropna()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    return s.sort_index()
            return None

        # 분기(최근 5~7개) + 연간(최근 5년)을 합쳐 커버리지를 넓힌다
        try:
            abs_, ainc = tk.balance_sheet, tk.income_stmt
        except Exception:  # noqa: BLE001
            abs_ = ainc = None

        def merge(q_, a_):
            parts = [x for x in (q_, a_) if x is not None and len(x)]
            if not parts:
                return None
            m = pd.concat(parts)
            return m[~m.index.duplicated(keep="first")].sort_index()

        a = merge(grab(bs, ["Total Assets"]), grab(abs_, ["Total Assets"]))
        e = merge(grab(bs, ["Stockholders Equity", "Common Stock Equity"]),
                  grab(abs_, ["Stockholders Equity", "Common Stock Equity"]))
        n = merge(grab(inc, ["Net Income From Continuing Operation Net Minority Interest",
                             "Net Income"]),
                  grab(ainc, ["Net Income From Continuing Operation Net Minority Interest",
                              "Net Income"]))
        for src, dst in ((a, ta), (e, be), (n, ni)):
            if src is not None and len(src):
                dst[t] = src

    def to_daily(d: dict) -> pd.DataFrame | None:
        if not d:
            return None
        df = pd.DataFrame(d)
        # 분기말 + LAG영업일 이후부터 사용 (공시 지연 근사)
        df.index = df.index + pd.tseries.offsets.BDay(LAG)
        return df.reindex(close.index.union(df.index)).ffill().reindex(close.index)\
                 .reindex(columns=close.columns)

    TA, BE, NI = to_daily(ta), to_daily(be), to_daily(ni)
    if TA is None and BE is None:
        warnings.warn("분기 재무를 받지 못해 재무 특성이 비었습니다", stacklevel=3)
        return {}
    out = {"log_size": np.log(mkt_cap.clip(lower=1)),
           "momentum": close.pct_change(252) - close.pct_change(21)}
    if BE is not None:
        out["log_bm"] = np.log((BE / mkt_cap.replace(0, np.nan)).clip(lower=1e-6))
    if NI is not None and TA is not None:
        out["roa"] = NI / TA.replace(0, np.nan)
    if TA is not None:
        out["asset_growth"] = TA.pct_change(252)
    # 커버리지가 낮은 특성은 버린다 — 빈 컬럼이 회귀 표본을 통째로 날린다
    kept, dropped = {}, {}
    for k, v in out.items():
        cover = float(v.notna().values.mean())
        (kept if cover >= MIN_COVER else dropped)[k] = cover
        if cover >= MIN_COVER:
            kept[k] = v
    dropped_names = {k: f"{c*100:.0f}%" for k, c in dropped.items()}
    kept_out = {k: out[k] for k in out if k not in dropped}
    if dropped_names:
        warnings.warn(
            f"재무 특성 {list(dropped_names)} 은 커버리지가 {dropped_names} 로 낮아 제외했습니다 "
            f"(yfinance 는 분기 5~7개·연간 5년만 제공). "
            f"장기 백테스트에는 유료 재무 소스를 붙이십시오.", stacklevel=3)
    got = {k: f"{float(v.notna().values.mean())*100:.0f}%" for k, v in kept_out.items()}
    warnings.warn(f"재무 특성 유효셀: {got} · 기말+{LAG}영업일 지연 적용", stacklevel=3)
    return kept_out


def sp500_tickers() -> list[str]:
    """위키백과 S&P 500 현재 구성종목. **현재 시점 목록이라 생존편향이 있다.**"""
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return sorted(t["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())
    except Exception as e:  # pragma: no cover  # noqa: BLE001
        raise RuntimeError(f"S&P 500 목록을 가져오지 못했습니다: {e}") from e
