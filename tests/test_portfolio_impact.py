"""Market impact in `backtest_portfolio`.

PI1 same as weights   the same holdings through `backtest_weights` give the same net returns and impact (1e-12); `ImpactModel` is one class under both import paths
PI2 square root       four times the money costs twice the impact (the cap out of the way); zero money costs nothing and changes nothing
PI3 inputs            no volume raises; the 25-cell grid is left out with an impact model; no impact keeps the old metrics and grid
PI4 ledger            two sizes of money are two trials, the same size again is one
PI5 report            the report names the impact cost
"""
from __future__ import annotations
import sys, tempfile, warnings
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.core.impact import ImpactModel as CoreImpact
from pitbacktest.weights import ImpactModel as WeightsImpact
from test_argument_checks import _raises
from test_synthetic import make_panel


def _setup(n_days=700, n_stocks=60):
    p, rng = make_panel(n_days=n_days, n_stocks=n_stocks)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    return p, f


def _bp(p, f, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return q.backtest_portfolio(p, f, hold=5, spread_bp=0, benchmark=None, **kw)


def test_same_as_weights():
    p, f = _setup()
    im = q.ImpactModel(aum=5e7, y=1.3)
    r = _bp(p, f, impact=im)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = q.backtest_weights(p, pd.DataFrame(r.holdings, index=p.dates, columns=p.tickers), spread_bp=0, impact=im, benchmark=None, check_universe=False)
    assert float(np.abs(r.net_returns.to_numpy() - w.net_returns.to_numpy()).max()) < 1e-12
    for k in ("impact_annual_bp", "participation_p99", "participation_max", "trades_over_10pct_adv"):
        assert abs(r.metrics[k] - w.metrics[k]) < 1e-9 * max(1.0, abs(w.metrics[k])), (k, r.metrics[k], w.metrics[k])
    assert r.metrics["impact_annual_bp"] > 100.0                                          # the money is large enough to matter
    assert WeightsImpact is CoreImpact is q.ImpactModel
    print(f"PI1 the same holdings through backtest_weights give the same net returns (1e-12) and {r.metrics['impact_annual_bp']:,.0f} bp a year of impact; one ImpactModel class  PASS")


def test_square_root_and_zero():
    p, f = _setup()
    big = 1e12                                                                            # the cap is out of the way: the law itself is tested
    a = _bp(p, f, impact=q.ImpactModel(aum=2e6, y=1.0, max_cost_bp=big))
    b = _bp(p, f, impact=q.ImpactModel(aum=8e6, y=1.0, max_cost_bp=big))
    ratio = b.metrics["impact_annual_bp"] / a.metrics["impact_annual_bp"]
    assert abs(ratio - 2.0) < 1e-9, ratio
    y2 = _bp(p, f, impact=q.ImpactModel(aum=2e6, y=3.0, max_cost_bp=big))
    assert abs(y2.metrics["impact_annual_bp"] / a.metrics["impact_annual_bp"] - 3.0) < 1e-9
    none = _bp(p, f)
    zero = _bp(p, f, impact=q.ImpactModel(aum=0.0, max_cost_bp=big))
    assert float(np.abs(zero.net_returns.to_numpy() - none.net_returns.to_numpy()).max()) < 1e-15 and zero.metrics["impact_annual_bp"] == 0.0
    assert (a.net_returns < none.net_returns).sum() > 0 and (a.net_returns <= none.net_returns + 1e-15).all()          # impact only ever costs
    print(f"PI2 four times the money costs {ratio:.6f}x the impact, three times Y costs 3x, zero money changes nothing, impact never pays  PASS")


def test_inputs():
    p, f = _setup()
    _raises(lambda: _bp(replace(p, volume=None), f, impact=q.ImpactModel(aum=1e6)), "volume")
    with_impact = _bp(p, f, impact=q.ImpactModel(aum=1e6), grid=True)
    assert with_impact.grid is None and with_impact.spec["impact"]["aum"] == 1e6
    plain = _bp(p, f, grid=True)
    assert plain.grid is not None and "impact_annual_bp" not in plain.metrics and plain.spec["impact"] is None
    print("PI3 no volume raises, the grid is left out with an impact model, and a run without one keeps its grid and metrics  PASS")


def test_ledger_and_report():
    p, f = _setup()
    led = q.Ledger(tempfile.mkdtemp(prefix="pi-ledger-"))
    for aum in (1e6, 1e8, 1e6):
        _bp(p, f, impact=q.ImpactModel(aum=aum), ledger=led, family="size", name="s")
    assert led.n_trials("size") == 2, led.n_trials("size")
    _bp(p, f, ledger=led, family="size", name="s")
    assert led.n_trials("size") == 3                                                      # no impact at all is a third configuration
    page = _bp(p, f, impact=q.ImpactModel(aum=5e7)).report()
    assert "Market impact" in page and "bp a year" in page
    assert "Market impact" not in _bp(p, f).report()
    print("PI4/PI5 two sizes of money are two trials and a repeat is one; the report names the impact cost only when there is one  PASS")


if __name__ == "__main__":
    test_same_as_weights()
    test_square_root_and_zero()
    test_inputs()
    test_ledger_and_report()
    print("portfolio impact tests: all passed")
