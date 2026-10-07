"""Run against an **installed** package, from a directory that is not the repository: imports work, the version matches the
metadata, and a small backtest, the ledger and the cost simulator run end to end. Nothing here may import from the repo."""
import importlib.metadata as md
import tempfile

import numpy as np
import pandas as pd

import quantbt as q

assert "site-packages" in q.__file__.replace("\\", "/") or "dist-packages" in q.__file__, q.__file__
dist = next(d for d in md.distributions() if (d.read_text("top_level.txt") or "").split() and "quantbt" in (d.read_text("top_level.txt") or "").split())
assert dist.version == q.__version__, (dist.version, q.__version__)

rng = np.random.default_rng(0)
dates = pd.bdate_range("2020-01-01", periods=400)
cols = [f"S{i:02d}" for i in range(40)]
close = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0, 0.015, (400, 40)), axis=0)), index=dates, columns=cols)
vol = pd.DataFrame(rng.lognormal(10, 0.3, (400, 40)), index=dates, columns=cols)
el = pd.DataFrame(True, index=dates, columns=cols); el.iloc[:60] = False
panel = q.Panel(close=close, eligible=el, volume=vol, market="TEST")
factor = -close.pct_change(5)
with tempfile.TemporaryDirectory() as d:
    led = q.Ledger(d)
    r = q.backtest_portfolio(panel, factor, long_q=0.2, short_q=0.2, hold=5, spread_bp=10, ledger=led, family="smoke", name="rev5")
    assert led.n_trials("smoke") == 1 and led.verify()["ok"]
ab = r.alpha_beta()
ci = r.sharpe_ci(n=200)
w = q.backtest_weights(panel, pd.DataFrame(r.holdings, index=panel.dates, columns=panel.tickers), spread_bp=10,
                       impact=q.ImpactModel(aum=1e6), check_universe=False)
cap = q.capacity_curve(panel, pd.DataFrame(r.holdings, index=panel.dates, columns=panel.tickers), aums=[1e5, 1e6], spread_bp=10,
                       check_universe=False)
assert np.isfinite(ab["alpha_annual"]) and ci["lo"] < ci["hi"] and len(cap) == 2 and np.isfinite(w.metrics["Sharpe"])
print("installed package", q.__version__, "ok:", q.__file__)
