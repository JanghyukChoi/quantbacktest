"""Factor recipes: each one equals its definition rebuilt by hand, is causal, is invariant to the price level where it should be, and refuses bad input.

RC1 definitions   every recipe against an explicit computation, cell by cell on a small panel
RC2 causality     Panel.assert_causal passes on every recipe; a recipe on a panel cut at a date equals the same row of the full panel
RC3 invariances   multiplying the prices by a constant leaves the price-based recipes' ranks unchanged; the order of securities changes nothing
RC4 input         unknown names, unknown and bad parameters, missing panel fields, short history
RC5 description   describe() lists every recipe with its parameters and defaults, and agrees with the functions
RC6 use           a recipe goes straight into review_portfolio and screen
"""
from __future__ import annotations
import dataclasses, math, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import recipes as rc
from test_argument_checks import _raises
from test_synthetic import make_panel


def _panel(n_days=420, n_stocks=24, seed=3):
    p, _ = make_panel(n_days=n_days, n_stocks=n_stocks, seed=seed)
    return p


def test_definitions():
    p = _panel()
    c, v, cap = p.close.to_numpy(), p.volume.to_numpy(), p.mkt_cap.to_numpy()
    T, N = c.shape
    got = rc.recipe(p, "momentum")
    for t in (260, 300, 419):
        for j in (0, 5, 23):
            assert abs(got.iloc[t, j] - (c[t - 21, j] / c[t - 252, j] - 1.0)) < 1e-12, (t, j)
    assert got.iloc[:252].isna().all().all() and got.iloc[252].notna().all()
    m2 = rc.recipe(p, "momentum", lookback=30, skip=5)
    assert abs(m2.iloc[100, 3] - (c[95, 3] / c[70, 3] - 1.0)) < 1e-12 and m2.iloc[:30].isna().all().all()
    rv = rc.recipe(p, "reversal")
    assert abs(rv.iloc[50, 2] + (c[50, 2] / c[45, 2] - 1.0)) < 1e-12 and rv.iloc[:5].isna().all().all()
    assert abs(rc.recipe(p, "reversal", lookback=1).iloc[50, 2] + (c[50, 2] / c[49, 2] - 1.0)) < 1e-12
    lv = rc.recipe(p, "low_volatility")
    ret = c[1:] / c[:-1] - 1.0
    assert abs(lv.iloc[100, 4] + np.std(ret[100 - 60:100, 4], ddof=1)) < 1e-12
    assert lv.iloc[:30].isna().all().all()                                                      # needs 40 of the 60 returns: the 40th return exists at row 40
    first = int(np.flatnonzero(lv.iloc[:, 0].notna().to_numpy())[0])
    assert first == 40, first
    sz = rc.recipe(p, "size")
    assert np.allclose(sz.to_numpy(), -cap, equal_nan=True)
    cap0 = p.mkt_cap.copy(); cap0.iloc[10, 2] = 0.0
    assert np.isnan(rc.recipe(dataclasses.replace(p, mkt_cap=cap0), "size").iloc[10, 2])
    il = rc.recipe(p, "illiquidity")
    assert abs(il.iloc[100, 6] + np.mean(c[71:101, 6] * v[71:101, 6])) < 1e-6 * abs(il.iloc[100, 6]) and il.iloc[:19].isna().all().all()
    vs = rc.recipe(p, "volume_shock")
    assert abs(vs.iloc[200, 1] - (v[196:201, 1].mean() / v[141:201, 1].mean() - 1.0)) < 1e-12 and vs.iloc[:59].isna().all().all() and vs.iloc[59].notna().all()
    vs2 = rc.recipe(p, "volume_shock", short=2, window=10)
    assert abs(vs2.iloc[50, 0] - (v[49:51, 0].mean() / v[41:51, 0].mean() - 1.0)) < 1e-12
    dh = rc.recipe(p, "distance_to_high")
    assert abs(dh.iloc[300, 8] - (c[300, 8] / c[49:301, 8].max() - 1.0)) < 1e-12 and (dh.dropna(how="all") <= 1e-12).all().all() and dh.iloc[:251].isna().all().all()
    assert abs(rc.recipe(p, "distance_to_high", window=20).iloc[100, 0] - (c[100, 0] / c[81:101, 0].max() - 1.0)) < 1e-12
    print("RC1 every recipe equals its definition cell by cell, with its history rule and its parameters  PASS")


def test_causality():
    p = _panel(n_days=420)
    for name in rc.names():
        out = p.assert_causal(lambda pn, n=name: rc.recipe(pn, n), n_cuts=4)
        assert out["pass"], (name, out)
    full = {n: rc.recipe(p, n) for n in rc.names()}
    cut = p.truncate(p.dates[350])
    for n in rc.names():
        part = rc.recipe(cut, n)
        assert np.allclose(part.iloc[-1].to_numpy(), full[n].loc[p.dates[350]].to_numpy(), atol=1e-12, equal_nan=True), n
    print("RC2 every recipe passes assert_causal and equals the same row of the full panel on a cut panel  PASS")


def test_invariances():
    p = _panel()
    scaled = dataclasses.replace(p, close=p.close * 37.5)
    for name in ("momentum", "reversal", "low_volatility", "distance_to_high"):
        a, b = rc.recipe(p, name), rc.recipe(scaled, name)
        assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-10, equal_nan=True), name
    for name in ("size", "illiquidity"):                                                         # a change of units moves the level, not the ranking
        a = rc.recipe(p, name).rank(axis=1); b = rc.recipe(dataclasses.replace(p, close=p.close * 5, mkt_cap=p.mkt_cap * 5), name).rank(axis=1)
        assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True), name
    cols = list(p.tickers)[::-1]
    flipped = dataclasses.replace(p, close=p.close[cols], eligible=p.eligible[cols], volume=p.volume[cols], mkt_cap=p.mkt_cap[cols])
    for name in rc.names():
        assert np.allclose(rc.recipe(flipped, name)[list(p.tickers)].to_numpy(), rc.recipe(p, name).to_numpy(), atol=1e-12, equal_nan=True), name
    print("RC3 price-based recipes ignore the price level; size and illiquidity keep their ranking under a change of units; the order of securities is irrelevant  PASS")


def test_input():
    p = _panel()
    _raises(lambda: rc.recipe(p, "alpha"), "unknown recipe 'alpha'", "momentum", "distance_to_high")
    _raises(lambda: rc.recipe(p, "momentum", lookbak=5), "no parameter 'lookbak'", "lookback")
    _raises(lambda: rc.recipe(p, "reversal", lookback=0), "at least 1")
    _raises(lambda: rc.recipe(p, "reversal", lookback=2.5), "whole number")
    _raises(lambda: rc.recipe(p, "reversal", lookback=True), "whole number")
    _raises(lambda: rc.recipe(p, "momentum", lookback=10, skip=10), "smaller")
    _raises(lambda: rc.recipe(p, "volume_shock", short=60, window=60), "smaller")
    _raises(lambda: rc.recipe(dataclasses.replace(p, volume=None), "illiquidity"), "needs panel.volume")
    _raises(lambda: rc.recipe(dataclasses.replace(p, volume=None), "volume_shock"), "needs panel.volume")
    _raises(lambda: rc.recipe(dataclasses.replace(p, mkt_cap=None), "size"), "needs panel.mkt_cap")
    short = rc.recipe(_panel(n_days=100), "momentum")                                            # not enough history: empty, not filled
    assert short.isna().all().all() and short.shape == (100, 24)
    out = rc.recipe(p, "reversal")
    assert out.shape == p.close.shape and (out.index == p.dates).all() and list(out.columns) == list(p.tickers) and out.dtypes.eq(float).all()
    print("RC4 unknown names and parameters, bad values, missing fields and short history behave  PASS")


def test_describe():
    d = rc.describe()
    assert list(d) == rc.names() == ["momentum", "reversal", "low_volatility", "size", "illiquidity", "volume_shock", "distance_to_high"]
    for name, e in d.items():
        assert e["higher_is"] == "long" and e["definition"] and e["needs"] and set(e) == {"definition", "parameters", "needs", "higher_is"}, name
    assert d["momentum"]["parameters"] == {"lookback": None, "skip": None} and d["reversal"]["parameters"] == {"lookback": 5}
    assert d["volume_shock"]["parameters"] == {"short": 5, "window": 60} and d["size"]["parameters"] == {} and d["size"]["needs"] == ["mkt_cap"]
    assert rc.describe("reversal") == d["reversal"]
    _raises(lambda: rc.describe("nope"), "unknown recipe")
    assert all("Higher" in e["definition"] or "score" in e["definition"] or "high" in e["definition"] for e in d.values())
    print("RC5 describe() lists every recipe with parameters, defaults and needs, and agrees with the functions  PASS")


def test_use():
    p = _panel(n_days=420)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = q.backtest_portfolio(p, rc.recipe(p, "momentum"), long_q=0.2, short_q=0.2, hold=5, benchmark=None, grid=False)
        rv = q.review_portfolio(p, rc.recipe(p, "low_volatility"), long_q=0.2, short_q=0.2, hold=5, n_permutations=20)
        sr = q.screen(p, {"rev": rc.recipe(p, "reversal"), "vs": rc.recipe(p, "volume_shock")}, horizons=(5,), primary_h=5, n_null=10)
    assert np.isfinite(r.metrics["Sharpe"]) and rv.mean["n"] > 100 and set(sr.summary["factor"]) == {"rev", "vs"}
    print("RC6 a recipe goes straight into backtest_portfolio, review_portfolio and screen  PASS")


def test_edges():
    p = _panel()
    # the default skip is a twelfth of a year of bars, at least one bar
    six = dataclasses.replace(p, periods_per_year=6)
    m = rc.recipe(six, "momentum")
    c = p.close.to_numpy()
    assert abs(m.iloc[50, 0] - (c[49, 0] / c[44, 0] - 1.0)) < 1e-12 and m.iloc[:6].isna().all().all() and m.iloc[6].notna().all()
    # low volatility with the smallest window: two thirds of five returns is three
    lv = rc.recipe(p, "low_volatility", window=5)
    assert int(np.flatnonzero(lv.iloc[:, 0].notna().to_numpy())[0]) == 3
    _raises(lambda: rc.recipe(p, "low_volatility", window=4), "at least 5")
    # a market value under one currency unit is a market value; zero is unknown
    small = p.mkt_cap.copy(); small.iloc[20, 0] = 0.5
    assert rc.recipe(dataclasses.replace(p, mkt_cap=small), "size").iloc[20, 0] == -0.5
    # illiquidity: a window of three is the smallest; a security that never traded is unknown, a thinly traded one is not
    _raises(lambda: rc.recipe(p, "illiquidity", window=2), "at least 3")
    assert rc.recipe(p, "illiquidity", window=3).iloc[10, 0] == -(p.close.to_numpy()[8:11, 0] * p.volume.to_numpy()[8:11, 0]).mean()
    thin = p.volume.copy(); thin.iloc[:, 1] = 0.0; thin.iloc[:, 2] = 0.001; cl = p.close.copy(); cl.iloc[:, 2] = 100.0
    ill = rc.recipe(dataclasses.replace(p, volume=thin, close=cl), "illiquidity")
    assert ill.iloc[100:, 1].isna().all() and abs(ill.iloc[100, 2] + 0.1) < 1e-12
    # volume shock: a security whose volume was zero for the whole window is unknown, not infinite
    quiet = p.volume.copy(); quiet.iloc[:, 3] = 0.0
    vs = rc.recipe(dataclasses.replace(p, volume=quiet), "volume_shock")
    assert vs.iloc[100:, 3].isna().all() and np.isfinite(vs.iloc[100:, 4]).all()
    print("RC7 the history floors, the smallest windows, penny market values and securities that never traded  PASS")


if __name__ == "__main__":
    for fn in (test_definitions, test_causality, test_invariances, test_input, test_describe, test_use, test_edges):
        fn()
    print("recipe tests: all passed")
