"""Portfolio engine arithmetic that the mutation runs showed no test pinned down.

PV1 metrics      Calmar, the share of positive bars, CAGR and drawdown of a worked return series
PV2 arguments    long_q above 1 is refused alone, 1 is allowed; the defaults are hold 5 and equal weights
PV3 weighting    the rank and the signal weights of a ten-security book, worked by hand
PV4 held count   the delisting count only counts names the book holds; a normal book raises no "no position" note
PV5 cost grid    the 5 x 5 grid has its keys, and each cell equals a run with that holding period and spread (with funding, side costs and a short leg)
PV6 ledger       the ledger's record of the funding switch is the effective one (on only when asked for and the panel has funding)
PV7 freeze       a long held through a suspension is marked down once and counted
"""
from __future__ import annotations
import sys, tempfile, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.portfolio import metrics
from test_argument_checks import _raises
from test_synthetic import make_panel


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def test_metrics_values():
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = np.where(np.arange(252) % 2 == 0, 0.01, -0.004)
    m = metrics(r, idx, 252)
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum(np.maximum.accumulate(eq), 1.0) - 1).min()
    cagr = eq[-1] ** (252 / 252) - 1
    assert abs(m["pos_days"] - 0.5) < 1e-12 and abs(m["MDD"] - dd) < 1e-12 and abs(m["CAGR"] - cagr) < 1e-12
    assert abs(m["Calmar"] - cagr / abs(dd)) < 1e-12 and m["Calmar"] > 0
    up = metrics(np.full(252, 0.001), idx, 252)                                 # never down: no drawdown, so no Calmar
    assert np.isnan(up["Calmar"]) and up["pos_days"] == 1.0
    print("PV1 Calmar, positive-bar share, CAGR and drawdown of a worked series  PASS")


def test_arguments_and_defaults():
    p, _ = make_panel(n_days=200, n_stocks=20, seed=1)
    f = pd.DataFrame(np.random.default_rng(1).standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    _raises(lambda: q.backtest_portfolio(p, f, long_q=1.5, short_q=None), "long_q must be in (0, 1]")
    ok = _quiet(lambda: q.backtest_portfolio(p, f, long_q=1.0, short_q=None, benchmark=None, grid=False))
    assert ok.spec["long_q"] == 1.0
    d = _quiet(lambda: q.backtest_portfolio(p, f, benchmark=None, grid=False))
    assert d.spec["hold"] == 5 and d.spec["weighting"] == "equal"
    print("PV2 long_q above 1 is refused with no short leg, 1 is accepted; the defaults are hold 5 and equal weights  PASS")


def _book(n=10, T=120, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=T)
    cols = [f"S{i}" for i in range(n)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, n)), axis=0)), index=idx, columns=cols)
    el = pd.DataFrame(True, index=idx, columns=cols)
    el.iloc[:5] = False
    p = q.Panel(close=close, eligible=el, volume=close * 1e5, mkt_cap=close * 1e7, market="T", entry_lag=1)
    f = pd.DataFrame(np.tile(np.arange(n, dtype=float) + 1.0, (T, 1)), index=idx, columns=cols)         # S0 lowest ... S9 highest, forever
    return p, f


def test_weightings():
    p, f = _book()
    kw = dict(long_q=0.25, short_q=0.25, hold=1, benchmark=None, grid=False)
    rk = _quiet(lambda: q.backtest_portfolio(p, f, weighting="rank", **kw)).holdings
    row = np.asarray(rk)[60]
    want = np.zeros(10)
    want[[7, 8, 9]] = np.array([0.05, 0.15, 0.25]) / 0.45                               # rank 0.8 0.9 1.0 minus (1 - q) = 0.75
    want[[0, 1]] = -np.array([0.15, 0.05]) / 0.2                                         # rank 0.1 0.2 against q = 0.25
    assert np.allclose(row, want, atol=1e-12), row
    sg = np.asarray(_quiet(lambda: q.backtest_portfolio(p, f, weighting="signal", **kw)).holdings)[60]
    top = np.array([8.0, 9.0, 10.0]); bot = np.array([1.0, 2.0])
    want2 = np.zeros(10); want2[[7, 8, 9]] = top / top.sum(); want2[[0, 1]] = -bot / bot.sum()
    assert np.allclose(sg, want2, atol=1e-12), sg
    eqw = np.asarray(_quiet(lambda: q.backtest_portfolio(p, f, weighting="equal", **kw)).holdings)[60]
    assert np.allclose(eqw[[7, 8, 9]], 1 / 3) and np.allclose(eqw[[0, 1]], -0.5)
    print("PV3 rank, signal and equal weights equal the worked weights of a ten-security book  PASS")


def test_held_count_and_notes():
    p, _ = make_panel(n_days=260, n_stocks=24, seed=5)
    da = pd.DataFrame(False, index=p.dates, columns=p.tickers)
    f = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    f.iloc[:, :6] = 3.0 + np.random.default_rng(0).random((len(f), 6))                    # the first six are always the long leg
    f.iloc[:, 6:] = np.random.default_rng(1).random((len(f), 18))
    mid = p.tickers[10]
    close = p.close.copy()
    close.loc[p.dates[150]:, mid] = np.nan
    da.loc[p.dates[149], mid] = True
    p2 = q.Panel(close=close, eligible=p.eligible, volume=p.volume, mkt_cap=p.mkt_cap, delist_after=da, market="T", entry_lag=1)
    f.loc[:, mid] = 0.5                                                                   # strictly between: in neither leg
    r = _quiet(lambda: q.backtest_portfolio(p2, f, long_q=0.2, short_q=0.1, hold=3, benchmark=None, grid=False))
    assert r.metrics["delist_events_held"] == 0, r.metrics["delist_events_held"]
    assert not any("no position" in n for n in r.notes), r.notes
    eq = _quiet(lambda: q.backtest_portfolio(p2, f, long_q=0.1, short_q=0.1, hold=3, benchmark=None, grid=False))
    assert not any("no position" in n for n in eq.notes), eq.notes                       # equal numbers long and short is a normal book
    big_short = _quiet(lambda: q.backtest_portfolio(p2, f, long_q=0.05, short_q=0.5, hold=1, benchmark=None, grid=False))
    assert not any("no position" in n for n in big_short.notes), big_short.notes          # a small long leg against a big short leg: positions were held
    empty = _quiet(lambda: q.backtest_portfolio(p2, f * np.nan, long_q=0.1, short_q=0.1, hold=3, benchmark=None, grid=False))
    assert any("no position" in n for n in empty.notes), empty.notes
    print("PV4 a delisting of a name the book does not hold is not counted; a normal book raises no 'no position' note  PASS")


def test_cost_grid_matches_runs():
    p, _ = make_panel(n_days=320, n_stocks=40, seed=7)
    rng = np.random.default_rng(7)
    fund = pd.DataFrame(rng.normal(0.0002, 0.0004, p.close.shape), index=p.dates, columns=p.tickers)
    p = q.Panel(close=p.close, eligible=p.eligible, volume=p.volume, mkt_cap=p.mkt_cap, funding=fund, market="T", entry_lag=1)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    kw = dict(long_q=0.2, short_q=0.2, funding=True, buy_bp=3.0, sell_bp=4.0, benchmark=None)
    base = _quiet(lambda: q.backtest_portfolio(p, f, hold=5, spread_bp=10.0, grid=True, **kw))
    assert set(base.grid) == {f"h{h}_c{c}" for h in (1, 2, 5, 10, 21) for c in (0, 2, 5, 10, 20)}, sorted(base.grid)
    for h, c in ((1, 0), (2, 5), (5, 10), (10, 20), (21, 2)):
        one = _quiet(lambda: q.backtest_portfolio(p, f, hold=h, spread_bp=float(c), grid=False, **kw))
        g = base.grid[f"h{h}_c{c}"]
        for k in ("CAGR", "Sharpe", "MDD"):
            assert abs(g[k] - one.metrics[k]) < 1e-9, (h, c, k, g[k], one.metrics[k])
    print("PV5 each cell of the cost grid equals a run with that holding period and spread, with funding, side costs and a short leg  PASS")


def test_ledger_funding_flag():
    p, _ = make_panel(n_days=200, n_stocks=20, seed=2)
    f = pd.DataFrame(np.random.default_rng(2).standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    with tempfile.TemporaryDirectory(prefix="pv6-") as d:
        led = q.Ledger(d)
        ra = _quiet(lambda: q.backtest_portfolio(p, f, ledger=led, family="a", name="x", benchmark=None, grid=False))                       # the default asks for funding, the panel has none
        rb_ = _quiet(lambda: q.backtest_portfolio(p, f, ledger=led, family="b", name="y", funding=True, benchmark=None, grid=False))          # asked for, but the panel has none
        pf = q.Panel(close=p.close, eligible=p.eligible, volume=p.volume, mkt_cap=p.mkt_cap, market="T", entry_lag=1,
                     funding=pd.DataFrame(0.0001, index=p.dates, columns=p.tickers))
        rc = _quiet(lambda: q.backtest_portfolio(pf, f, ledger=led, family="c", name="z", funding=False, benchmark=None, grid=False))                          # the panel has funding, the run switches it off
        rd = _quiet(lambda: q.backtest_portfolio(pf, f, ledger=led, family="d", name="w", funding=True, benchmark=None, grid=False))
        a, b, c, d_ = (led.trials(k)[0]["config"]["funding"] for k in "abcd")
    assert (a, b, c, d_) == (False, False, False, True), (a, b, c, d_)
    assert (ra.spec["funding"], rb_.spec["funding"], rc.spec["funding"], rd.spec["funding"]) == (False, False, False, True)           # and the result says the same
    print("PV6 the ledger records funding as on only when it was asked for and the panel has it  PASS")


def test_freeze_events():
    p, _ = make_panel(n_days=260, n_stocks=24, seed=9)
    v = p.volume.copy(); c = p.close.copy()
    v.iloc[100:140, 0] = 0.0; c.iloc[100:140, 0] = float(c.iloc[99, 0])                  # the first name is suspended on rows 100..139
    from dataclasses import replace
    pp = replace(p, close=c, volume=v)
    f = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    f.iloc[:, 0] = 10.0                                                                  # always the top score: held long throughout
    f.iloc[:, 1:] = np.random.default_rng(0).random((len(f), 23))
    kw = dict(long_q=0.1, short_q=None, hold=1, benchmark=None, grid=False)
    base = _quiet(lambda: q.backtest_portfolio(pp, f, **kw))
    fr = _quiet(lambda: q.backtest_portfolio(pp, f, freeze_days=10, freeze_return=-0.4, **kw))
    assert "freeze_markdown_events" not in base.metrics
    assert fr.metrics["freeze_markdown_events"] >= 1 and fr.metrics["freeze_markdown_annual_bp"] < 0, fr.metrics
    print("PV7 a long held through a suspension is marked down once and counted  PASS")


if __name__ == "__main__":
    for fn in (test_metrics_values, test_arguments_and_defaults, test_weightings, test_held_count_and_notes, test_cost_grid_matches_runs, test_ledger_funding_flag, test_freeze_events):
        fn()
    print("portfolio value tests: all passed")
