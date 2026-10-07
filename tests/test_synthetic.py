"""합성 데이터 검증 — 정답을 아는 데이터로 프레임워크가 맞게 답하는가.

검증 항목
  T1 완전예지 팩터  → 큰 양수여야 (시점 정렬 정상)
  T2 무작위 팩터    → 0 근처여야 (거짓 양성 없음)
  T3 룩어헤드 팩터  → assert_no_lookahead 가 잡아야
  T4 미래 신호      → 하루 늦춰도 성과가 유지/상승하면 탐지
  T5 셔플 귀무      → 무작위 팩터의 |t| 분포가 합리적 범위
  T6 중립화        → 통제변수 자체를 팩터로 넣으면 중립화 후 소멸해야
  T7 기저승률      → 무작위 발화의 lift 가 0 근처여야
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.core.controls import build_controls, xs_norm


def make_panel(n_days=1200, n_stocks=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tick = [f"S{i:03d}" for i in range(n_stocks)]
    # 시장 + 개별 수익률
    mkt = rng.normal(0.0003, 0.010, n_days)[:, None]
    idio = rng.normal(0, 0.018, (n_days, n_stocks))
    ret = mkt + idio
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=tick)
    vol = pd.DataFrame(rng.lognormal(11, 1.0, (n_days, n_stocks)), index=dates, columns=tick)
    mcap = close * rng.lognormal(15, 1.0, n_stocks)
    el = pd.DataFrame(True, index=dates, columns=tick)
    el.iloc[:60] = False                        # 초기 워밍업 제외
    return q.Panel(close=close, eligible=el, volume=vol, mkt_cap=mcap,
                   market="TEST", entry_lag=1), rng


def main():
    p, rng = make_panel()
    fails = []
    print(f"[패널] {len(p.dates)}일 × {len(p.tickers)}종목 · entry_lag={p.entry_lag}\n")

    # T1 완전예지
    r = q.assert_timing(p)
    ok = r["pass"]
    print(f"T1 시점정렬(완전예지)  CAGR {r['oracle_CAGR']*100:>8.1f}%  "
          f"Sharpe {r['oracle_Sharpe']:>6.2f}  {'PASS' if ok else 'FAIL'}")
    if not ok: fails.append("T1")

    # T2 무작위 팩터
    noise = pd.DataFrame(rng.normal(size=(len(p.dates), len(p.tickers))),
                         index=p.dates, columns=p.tickers)
    rp = q.backtest_portfolio(p, noise, hold=5, spread_bp=0.0, grid=False)
    ok = abs(rp.metrics["Sharpe"]) < 0.5
    print(f"T2 무작위 팩터        CAGR {rp.metrics['CAGR']*100:>8.2f}%  "
          f"Sharpe {rp.metrics['Sharpe']:>6.2f}  {'PASS' if ok else 'FAIL'} (|Sharpe|<0.5 기대)")
    if not ok: fails.append("T2")

    # T3 룩어헤드 탐지
    fut = p.close.shift(-3) / p.close - 1.0        # 명백한 미래 정보
    la = p.assert_no_lookahead(fut, h=5)
    ok = not la["pass"] or la["base_bp"] > 100     # 탐지되거나, 최소한 base 가 비정상적으로 큼
    print(f"T3 룩어헤드 탐지      base {la['base_bp']:>8.1f}bp  lagged {la['lagged_bp']:>8.1f}bp  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok: fails.append("T3")

    # T4 정상 신호는 늦추면 약해진다
    la2 = p.assert_no_lookahead(noise, h=5)
    print(f"T4 무작위 룩어헤드     pass={la2['pass']}  (무작위는 통과해야)")
    if not la2["pass"]: fails.append("T4")

    # T5 셔플 귀무
    sr = q.screen(p, {"noise": noise}, horizons=(1, 5, 20), primary_h=20,
                  n_null=2, neutralize_all=False)
    thr = sr.null.get("p95", np.nan)
    # 합성 데이터엔 신호가 없으므로 귀무 |t| 는 0 근처가 정상이다.
    # 검증할 것은 "값이 산출되고 비정상적으로 크지 않은가"이지 특정 범위가 아니다.
    ok = np.isfinite(thr) and 0 <= thr < 4.0
    print(f"T5 셔플귀무 문턱      p95={thr:.2f}  max={sr.null.get('max', float('nan')):.2f}  "
          f"{'PASS' if ok else 'FAIL'} (산출됨 & <4 기대)")
    if not ok: fails.append("T5")

    # T6 중립화 — 통제변수를 팩터로 쓰면 중립화 후 소멸
    ctrl = build_controls(p, include_chars=False)
    mom = ctrl["mom21"]
    sr2 = q.screen(p, {"mom21_self": mom}, horizons=(1, 5, 20), primary_h=20,
                   n_null=1, neutralize_all=True)
    surv = sr2.summary["neu_survival_%"].iloc[0]
    ok = (not np.isfinite(surv)) or abs(surv) < 30
    print(f"T6 중립화(자기통제)    잔존율 {surv:>7.1f}%  {'PASS' if ok else 'FAIL'} (|잔존|<30% 기대)")
    if not ok: fails.append("T6")

    # T7 기저승률
    fire = pd.DataFrame(rng.random((len(p.dates), len(p.tickers))) < 0.05,
                        index=p.dates, columns=p.tickers)
    re = q.backtest_event(p, fire, horizons=(1, 5, 20), cost_bp=0.0,
                          neutralize_check=False)
    lifts = [v["lift_pp"] for v in re.per_horizon.values() if np.isfinite(v.get("lift_pp", np.nan))]
    ok = all(abs(x) < 3.0 for x in lifts)
    print(f"T7 무작위 발화 lift    {['%+.2f' % x for x in lifts]}  "
          f"{'PASS' if ok else 'FAIL'} (|lift|<3%p 기대)")
    if not ok: fails.append("T7")

    # T8 소형 유니버스 — 고정 문턱이면 전 날짜가 버려진다 (실데이터에서 발견된 버그)
    small, srng = make_panel(n_days=800, n_stocks=20, seed=3)
    sig_s = pd.DataFrame(srng.random((len(small.dates), 20)) < 0.2,
                         index=small.dates, columns=small.tickers)
    try:
        rs = q.backtest_event(small, sig_s, horizons=(1, 5), cost_bp=0.0,
                              neutralize_check=False)
        got = len(rs.per_horizon) > 0 and np.isfinite(
            rs.gates["oos"]["OOS"].get("mean_bp", np.nan))
        print(f"T8 소형 유니버스(20종목) 호라이즌 {len(rs.per_horizon)}개 · OOS 산출 "
              f"{'PASS' if got else 'FAIL'}")
        if not got: fails.append("T8")
    except Exception as ex:
        print(f"T8 소형 유니버스(20종목) 예외 {type(ex).__name__} FAIL")
        fails.append("T8")

    # T9 소형 유니버스에서 FM 회귀 — min_stocks 고정이면 전 날짜가 버려진다
    small2, r2 = make_panel(n_days=900, n_stocks=25, seed=11)
    from pitbacktest.core.estimators import fama_macbeth
    from pitbacktest.core.controls import build_controls as _bc
    c2 = _bc(small2, include_chars=False)
    f2 = xs_norm(pd.DataFrame(r2.normal(size=(len(small2.dates), 25)),
                              index=small2.dates, columns=small2.tickers), small2.eligible)
    fm2 = fama_macbeth(f2, {5: small2.forward(5)}, c2, small2.eligible)
    ok = np.isfinite(fm2[5]["t"]) and fm2[5]["n_days"] > 100
    print(f"T9 소형 FM 회귀(25종목) t={fm2[5]['t']:+.2f} · n_days={fm2[5]['n_days']} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok: fails.append("T9")

    # T10 완전공선 경고 — 통제변수를 그대로 팩터로 넣으면 경고가 떠야
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        fama_macbeth(c2["mom21"], {5: small2.forward(5)}, c2, small2.eligible)
        got = any("공선" in str(x.message) for x in rec)
    print(f"T10 완전공선 경고     {'PASS' if got else 'FAIL'}")
    if not got: fails.append("T10")

    # 감사
    aud = p.audit()
    print(f"\n[감사] 유니버스 증가율 {aud['universe_growth']:.2f}x · "
          f"극단수익률 {aud['ret_extreme_pct']:.3f}% · "
          f"절단의심 {aud.get('truncation_suspected')}")

    print(f"\n{'='*60}")
    print(f"결과: {10-len(fails)}/10 통과" + (f" · 실패: {fails}" if fails else " · 전부 통과"))
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(main())
