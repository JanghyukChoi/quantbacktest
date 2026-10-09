"""Bad arguments must stop with a clear message instead of quietly running something else.

Each case below used to run without a word (found while writing the library's specification): a mistyped `weighting` or `benchmark` was
read as a different option, `long_q=1.5` or `0` ran, `short_q=0` silently meant long-only, `hold=0` gave a Sharpe of 0, a negative
cost paid the strategy for trading (Sharpe -1.01 became +0.96), and a boolean factor was documented as supported but was not.

AC1 portfolio  every bad argument raises ValueError naming the argument; the boundary values still run
AC2 weights    the same for cost, benchmark and the impact model
AC3 screen and event   horizons, primary_h and costs
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.weights import ImpactModel
from test_synthetic import make_panel


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return str(e)
    raise AssertionError(f"no ValueError for {needles}")


def test_portfolio_arguments():
    p, _ = make_panel(n_days=500, n_stocks=60)
    f = -p.close.pct_change(5)
    run = lambda **kw: q.backtest_portfolio(p, kw.pop("factor", f), **{"long_q": 0.2, "short_q": 0.2, "hold": 5, "spread_bp": 10,
                                                                        "grid": False, "benchmark": None, **kw})
    _raises(lambda: run(weighting="rnak"), "weighting", "'rnak'")
    _raises(lambda: run(benchmark="equl"), "benchmark", "'equl'")
    _raises(lambda: run(long_q=1.5), "long_q")
    _raises(lambda: run(long_q=0.0), "long_q")
    _raises(lambda: run(long_q=float("nan")), "long_q")
    _raises(lambda: run(short_q=0.0), "short_q", "None for a long-only")
    _raises(lambda: run(short_q=-0.1), "short_q")
    _raises(lambda: run(long_q=0.7, short_q=0.7), "long_q + short_q", "both legs")       # the same security long and short at once
    _raises(lambda: run(hold=0), "hold")
    _raises(lambda: run(hold=2.5), "hold")
    _raises(lambda: run(hold=True), "hold")
    _raises(lambda: run(spread_bp=-10), "spread_bp", "not negative")
    _raises(lambda: run(spread_bp=float("inf")), "spread_bp")
    _raises(lambda: run(spread_bp=pd.DataFrame(-1.0, index=p.dates, columns=p.tickers).to_numpy()), "spread_bp")
    _raises(lambda: run(factor=(f.rank(axis=1, pct=True) >= 0.9)), "boolean factor", "backtest_event")
    # the boundary values and every documented option still run
    for kw in (dict(long_q=1.0, short_q=None), dict(long_q=0.5, short_q=0.5), dict(hold=1), dict(spread_bp=0.0), dict(spread_bp=0),
               dict(weighting="equal"), dict(weighting="signal"), dict(weighting="rank"),
               dict(benchmark="cap"), dict(benchmark="equal"), dict(benchmark=None), dict(hold=np.int64(3))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert np.isfinite(run(**kw).metrics["turnover_daily"]), kw
    print("AC1 portfolio: typos in weighting and benchmark, quantiles outside (0, 1], short_q=0, hold below 1, a negative or infinite cost and a boolean factor all raise; the boundary values run  PASS")


def test_weights_arguments():
    p, _ = make_panel(n_days=400, n_stocks=60)
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    W.iloc[100:, :5] = 0.2
    ok = lambda **kw: q.backtest_weights(p, W, **{"benchmark": None, **kw})
    _raises(lambda: ok(benchmark="equl"), "benchmark")
    _raises(lambda: ok(spread_bp=-1.0), "spread_bp")
    _raises(lambda: ok(borrow_bp=-5.0), "borrow_bp")
    _raises(lambda: ImpactModel(aum=-1.0), "aum")
    _raises(lambda: ImpactModel(aum=1e6, y=-0.5), "y must")
    _raises(lambda: ImpactModel(aum=1e6, vol_window=1), "vol_window")
    _raises(lambda: ImpactModel(aum=1e6, max_cost_bp=-1.0), "max_cost_bp")
    _raises(lambda: ImpactModel(aum=float("nan")), "aum")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok(benchmark="cap"); ok(benchmark="equal"); ok(spread_bp=0.0, borrow_bp=0.0); ImpactModel(aum=0.0, y=0.0, max_cost_bp=0.0)
    print("AC2 weights: benchmark typos, negative spread or borrow, and a nonsensical impact model raise; zero costs and zero AUM are allowed  PASS")


def test_screen_and_event_arguments():
    p, rng = make_panel(n_days=500, n_stocks=60)
    f = -p.close.pct_change(5)
    sig = f.rank(axis=1, pct=True) >= 0.95
    _raises(lambda: q.screen(p, {"a": f}, horizons=(1, 5), primary_h=20), "primary_h", "(1, 5)")
    _raises(lambda: q.screen(p, {"a": f}, horizons=(), primary_h=5), "horizons")
    _raises(lambda: q.screen(p, {"a": f}, horizons=(0, 5), primary_h=5), "horizons")
    _raises(lambda: q.screen(p, {"a": f}, horizons=(5,), primary_h=5, cost_bp=-3.0), "cost_bp")
    _raises(lambda: q.backtest_event(p, sig, horizons=()), "horizons")
    _raises(lambda: q.backtest_event(p, sig, horizons=(5, -2)), "horizons")
    _raises(lambda: q.backtest_event(p, sig, horizons=(5.5,)), "horizons")
    _raises(lambda: q.backtest_event(p, sig, horizons=(5,), cost_bp=-1.0), "cost_bp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert q.backtest_event(p, sig, horizons=(5,), cost_bp=0.0, neutralize_check=False).per_horizon
    print("AC3 screen and event: primary_h outside horizons, empty or non-positive horizons and negative costs raise; zero cost runs  PASS")


def test_screen_null_repetitions():
    import inspect
    assert inspect.signature(q.screen).parameters["n_null"].default >= 10                     # the default is a usable threshold
    p, _ = make_panel(n_days=500, n_stocks=60)
    f = -p.close.pct_change(7)
    _raises(lambda: q.screen(p, {"a": f}, horizons=(5,), primary_h=5, n_null=0), "n_null")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        q.screen(p, {"a": f}, horizons=(5,), primary_h=5, n_null=2, neutralize_all=False)
    assert any("n_null=2" in str(x.message) and "biased low" in str(x.message) for x in w), [str(x.message)[:50] for x in w]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        q.screen(p, {"a": f}, horizons=(5,), primary_h=5, n_null=10, neutralize_all=False)
    assert not any("n_null" in str(x.message) for x in w)
    print("AC4 screen: n_null below 10 warns that the threshold is biased low, 0 raises, and the default is 20  PASS")


if __name__ == "__main__":
    test_portfolio_arguments()
    test_weights_arguments()
    test_screen_and_event_arguments()
    test_screen_null_repetitions()
    print("argument check tests: all passed")
