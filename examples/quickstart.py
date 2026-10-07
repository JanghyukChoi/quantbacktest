"""pitbacktest 사용 예 — 세 진입점.

여기서는 합성 데이터로 흐름만 보인다. 실데이터는 adapters.yfinance 를 쓴다.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q

# ── 1. 패널 준비 ───────────────────────────────────────────────
rng = np.random.default_rng(7)
dates = pd.bdate_range("2019-01-01", periods=1000)
tick = [f"S{i:03d}" for i in range(150)]
ret = rng.normal(0.0003, 0.015, (1000, 150))
close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=tick)
vol = pd.DataFrame(rng.lognormal(11, 1, (1000, 150)), index=dates, columns=tick)
mcap = close * rng.lognormal(15, 1, 150)
el = pd.DataFrame(True, index=dates, columns=tick); el.iloc[:60] = False

panel = q.Panel(close=close, eligible=el, volume=vol, mkt_cap=mcap, market="TEST")
print("[감사]", {k: v for k, v in panel.audit().items() if not isinstance(v, dict)})
print("[시점 검증]", q.assert_timing(panel)["pass"])

# ── 2. A · 팩터 스크리닝 + 중립화 ──────────────────────────────
# ⚠️ 통제변수(rev1·rev5·mom21·mom63·mom252_21·vol21·logsize·turnover)와
#    같은 팩터를 넣으면 잔차가 0 이 되어 t 가 NaN 이 된다. 겹치지 않는 축을 쓴다.
r = close.pct_change()
factors = {
    "거래량충격":   (vol / vol.rolling(60, min_periods=60).mean()),
    "가격_거래량_괴리": (close.pct_change(10).rank(axis=1, pct=True)
                    - (vol.rolling(10).mean() / vol.rolling(60).mean()).rank(axis=1, pct=True)),
    "고저변동_대비수익": close.pct_change(10) / (r.rolling(10).std() + 1e-9),
}
sr = q.screen(panel, factors, horizons=(1, 5, 20), primary_h=20, n_null=2)
print("\n[스크리닝 퍼널]", sr.funnel)
print(sr.summary[["factor", "t", "net_bp", "rho", "neu_survival_%", "passed"]].round(2).to_string(index=False))

# ── 3. B · 포트폴리오 ──────────────────────────────────────────
rp = q.backtest_portfolio(panel, factors["거래량충격"], long_q=0.1, short_q=0.1,
                          hold=5, weighting="signal", spread_bp=5.0)
print(f"\n[포트폴리오] CAGR {rp.metrics['CAGR']*100:+.2f}% · Sharpe {rp.metrics['Sharpe']:.2f} "
      f"· MDD {rp.metrics['MDD']*100:.1f}% · 회전율 {rp.metrics['turnover_daily']*100:.1f}%/일")

# ── 4. C · 이벤트형 ────────────────────────────────────────────
sig = factors["거래량충격"].rank(axis=1, pct=True) >= 0.95
re = q.backtest_event(panel, sig, horizons=(1, 5, 20), cost_bp=20.0)
for h, v in re.per_horizon.items():
    print(f"[이벤트 h={h:>2}] 승률 {v['win_rate']:.1f}% · 기저 {v['base_rate']:.1f}% "
          f"· lift {v['lift_pp']:+.2f}%p · 중앙값 {v['median_bp']:+.1f}bp")
print("[게이트]", re.gates["passed"], "· 탈락지점:", re.gates["failed_at"])
