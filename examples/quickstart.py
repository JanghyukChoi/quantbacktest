"""pitbacktest usage: the three entry points.

This shows the flow on synthetic data. For real data use adapters.yfinance.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q

# ── 1. Prepare the panel ───────────────────────────────────────────────
rng = np.random.default_rng(7)
dates = pd.bdate_range("2019-01-01", periods=1000)
tick = [f"S{i:03d}" for i in range(150)]
ret = rng.normal(0.0003, 0.015, (1000, 150))
close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=tick)
vol = pd.DataFrame(rng.lognormal(11, 1, (1000, 150)), index=dates, columns=tick)
mcap = close * rng.lognormal(15, 1, 150)
el = pd.DataFrame(True, index=dates, columns=tick); el.iloc[:60] = False

panel = q.Panel(close=close, eligible=el, volume=vol, mkt_cap=mcap, market="TEST")
print("[audit]", {k: v for k, v in panel.audit().items() if not isinstance(v, dict)})
print("[timing check]", q.assert_timing(panel)["pass"])

# ── 2. A. Factor screening + neutralisation ──────────────────────────────
# Warning: a factor equal to a control (rev1, rev5, mom21, mom63, mom252_21, vol21, logsize, turnover)
#    leaves a residual of 0 and t becomes NaN. Use axes that do not overlap them.
r = close.pct_change()
factors = {
    "volume_shock":   (vol / vol.rolling(60, min_periods=60).mean()),
    "price_volume_gap": (close.pct_change(10).rank(axis=1, pct=True)
                    - (vol.rolling(10).mean() / vol.rolling(60).mean()).rank(axis=1, pct=True)),
    "return_per_vol": close.pct_change(10) / (r.rolling(10).std() + 1e-9),
}
sr = q.screen(panel, factors, horizons=(1, 5, 20), primary_h=20)
print("\n[screening funnel]", sr.funnel)
print(sr.summary[["factor", "t", "net_bp", "rho", "neu_survival_%", "passed"]].round(2).to_string(index=False))

# ── 3. B. Portfolio ──────────────────────────────────────────
rp = q.backtest_portfolio(panel, factors["volume_shock"], long_q=0.1, short_q=0.1,
                          hold=5, weighting="signal", spread_bp=5.0)
print(f"\n[portfolio] CAGR {rp.metrics['CAGR']*100:+.2f}% · Sharpe {rp.metrics['Sharpe']:.2f} "
      f"· MDD {rp.metrics['MDD']*100:.1f}% · turnover {rp.metrics['turnover_daily']*100:.1f}%/day")

# ── 4. C. Events ────────────────────────────────────────────
sig = factors["volume_shock"].rank(axis=1, pct=True) >= 0.95
re = q.backtest_event(panel, sig, horizons=(1, 5, 20), cost_bp=20.0)
for h, v in re.per_horizon.items():
    print(f"[event h={h:>2}] win rate {v['win_rate']:.1f}% · base {v['base_rate']:.1f}% "
          f"· lift {v['lift_pp']:+.2f}%p · median {v['median_bp']:+.1f}bp")
print("[gates]", re.gates["passed"], "· failed at:", re.gates["failed_at"])
