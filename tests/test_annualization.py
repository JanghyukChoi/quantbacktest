"""연율화 시험 — 같은 일별 수익률이면 연간 거래일 설정만큼만 연율화 수치가 달라져야 한다.

주식은 연 252일, 24/7 시장(크립토)은 365일이다. 252 가 하드코딩돼 있으면 크립토의 CAGR, 샤프, 변동성이 틀린다.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import quantbt as q


def _panel(ppy: int) -> q.Panel:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=900, freq="D")      # 달력 일수 그대로(주말 포함)
    tick = [f"C{i:02d}" for i in range(30)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.03, (900, 30)), axis=0)), index=dates, columns=tick)
    vol = pd.DataFrame(rng.lognormal(11, 1, (900, 30)), index=dates, columns=tick)
    el = pd.DataFrame(True, index=dates, columns=tick)
    return q.Panel(close=close, eligible=el, volume=vol, mkt_cap=close * 1e6, market="CRYPTO", periods_per_year=ppy)


def test_annualization_scales_with_periods_per_year():
    factor = lambda p: -p.close.pct_change(5)
    a, b = _panel(252), _panel(365)
    ma = q.backtest_portfolio(a, factor(a), long_q=0.2, short_q=0.2, hold=5, spread_bp=5).metrics
    mb = q.backtest_portfolio(b, factor(b), long_q=0.2, short_q=0.2, hold=5, spread_bp=5).metrics
    assert abs(mb["years"] / ma["years"] - 252 / 365) < 1e-9, (ma["years"], mb["years"])
    assert abs(mb["vol"] / ma["vol"] - np.sqrt(365 / 252)) < 1e-6, (ma["vol"], mb["vol"])
    assert abs(mb["Sharpe"] / ma["Sharpe"] - np.sqrt(365 / 252)) < 1e-6, (ma["Sharpe"], mb["Sharpe"])
    print("T11 연율화 설정 252 vs 365  비율 정확  PASS")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    test_annualization_scales_with_periods_per_year()
