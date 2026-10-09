"""Results as JSON, and the warnings kept on the result.

EX1 portfolio   the JSON parses with no NaN token, the metrics equal the result's, every metric the library names has a unit, monthly returns equal a hand computation
EX2 series      'none' leaves the series out, 'full' has every bar, a bad argument raises
EX3 nulls       a figure that could not be computed is null, never 0 and never the text NaN
EX4 notes       a warning is stored in `notes`, still reaches the console once, and still raises under `simplefilter('error')`
EX5 others      event and screen results convert; `jsonable` handles numpy, pandas and odd keys
"""
from __future__ import annotations
import json, sys, warnings
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.export import PORTFOLIO_UNITS, jsonable
from test_argument_checks import _raises
from test_synthetic import make_panel


def _strict_loads(text):
    def bad(c):
        raise ValueError(f"non-JSON constant {c}")
    return json.loads(text, parse_constant=bad)


def _run(**kw):
    p, rng = make_panel(n_days=700, n_stocks=50)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p, f, q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5, spread_bp=10, **kw)


def test_portfolio_json():
    p, f, r = _run()
    text = r.to_json()
    d = _strict_loads(text)
    assert d["schema"] == "pitbacktest/portfolio-result" and d["schema_version"] == 1 and d["library_version"] == q.__version__
    for k, v in r.metrics.items():
        got = d["metrics"][k]
        if isinstance(v, float):
            assert (got is None and not np.isfinite(v)) or abs(got - v) < 1e-12, (k, got, v)
        else:
            assert got == v, (k, got, v)
    known = {"CAGR", "MDD", "Sharpe", "Sortino", "Calmar", "vol", "years", "pos_days", "mdd_peak", "mdd_trough", "mdd_recovered", "ruined", "ruin_date", "gross_CAGR",
             "cost_annual_bp", "side_cost_annual_bp", "funding_annual_bp", "turnover_daily", "avg_positions"}
    assert set(r.metrics) <= known | {"delist_events_held"}, set(r.metrics) - known                     # a metric added later must get a unit here
    assert known - {"ruin_date"} <= set(d["units"]) and all(isinstance(u, str) and u for u in d["units"].values())
    # monthly returns against a hand computation
    s = r.net_returns
    hand = {}
    for ts, v in s.items():
        hand[str(ts.to_period("M"))] = (1 + hand.get(str(ts.to_period("M")), 0.0)) * (1 + v) - 1 if str(ts.to_period("M")) in hand else v
    assert d["series"]["months"] == sorted(hand) and np.allclose(d["series"]["net_return"], [hand[m] for m in sorted(hand)], atol=1e-12)
    eq = (1 + s).cumprod()
    last = {str(ts.to_period("M")): v for ts, v in eq.items()}
    assert np.allclose(d["series"]["equity_at_month_end"], [last[m] for m in sorted(last)], atol=1e-12)
    print(f"EX1 the JSON parses with no NaN, metrics equal the result, every metric has a unit, {len(d['series']['months'])} monthly returns equal a hand computation  PASS")


def test_series_arguments():
    p, f, r = _run()
    assert "series" not in r.to_dict(series="none")
    full = r.to_dict(series="full")["series"]
    assert len(full["dates"]) == len(r.net_returns) == len(full["net_returns"]) and np.allclose(full["net_returns"], r.net_returns.to_numpy(), atol=1e-15)
    assert "benchmark_returns" in full and len(full["benchmark_returns"]) == len(full["dates"])
    _raises(lambda: r.to_dict(series="daily"), "series must be")
    path = Path(__import__("tempfile").mkdtemp(prefix="export-")) / "r.json"
    out = r.to_json(path)
    assert Path(out) == path and _strict_loads(path.read_text(encoding="utf-8"))["metrics"]["CAGR"] == r.metrics["CAGR"]
    print("EX2 series none/monthly/full, a bad argument raises, to_json writes a file  PASS")


def test_nulls():
    p, _ = make_panel(n_days=100, n_stocks=30)
    f = pd.DataFrame(np.random.default_rng(0).standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = q.backtest_portfolio(p, f, hold=5, spread_bp=10)                                            # under half a year: CAGR, MDD and Sharpe are NaN
    assert np.isnan(r.metrics["Sharpe"])
    text = r.to_json()
    d = _strict_loads(text)                                                                              # a bare NaN or Infinity token fails to parse here
    assert d["metrics"]["Sharpe"] is None and d["metrics"]["CAGR"] is None
    print("EX3 a figure that could not be computed is null in the JSON, never 0 and never NaN  PASS")


def test_notes():
    p, rng = make_panel(n_days=300, n_stocks=12)
    c = p.close.copy(); c.iloc[:, 0] = 100.0; c.iloc[150:, 0] = 50.0
    pp = replace(p, close=c)
    w = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); w.iloc[100:, 0] = 30.0
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = q.backtest_weights(pp, w, spread_bp=0, benchmark=None)
    assert any("lost 100%" in n for n in r.notes), r.notes
    assert sum("lost 100%" in str(x.message) for x in ws) == 1, [str(x.message)[:40] for x in ws]    # on the console exactly once
    assert any("lost 100%" in n for n in _strict_loads(r.to_json())["notes"])
    this = Path(__file__).name
    assert [x for x in ws if "lost 100%" in str(x.message)][0].filename.endswith(this)                # attributed to the caller's line, not to the library
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            q.backtest_weights(pp, w, spread_bp=0, benchmark=None)
        raise AssertionError("an error filter must still raise")
    except UserWarning as e:
        assert "lost 100%" in str(e)
    clean = q.backtest_weights(p, w * 0 + (w > 0) * 0.1, spread_bp=0, benchmark=None)
    assert clean.notes == []
    print("EX4 a warning is kept in `notes`, reaches the console once at the caller's line, still raises under an error filter; a quiet run has no notes  PASS")


def test_other_results():
    p, rng = make_panel(n_days=700, n_stocks=60)
    f = -p.close.pct_change(5)
    sig = f.rank(axis=1, pct=True) > 0.97
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = q.backtest_event(p, sig, horizons=(5,), neutralize_check=False)
        s = q.screen(p, {"a": f, "b": -f}, horizons=(5,), primary_h=5, n_null=10, neutralize_all=False)
    de = _strict_loads(e.to_json())
    assert de["schema"] == "pitbacktest/event-result" and de["per_horizon"]["5"]["n_trades"] == e.per_horizon[5]["n_trades"]
    assert set(de["units"]) >= {"win_rate", "base_rate", "lift_pp", "mean_bp"} and isinstance(de["notes"], list)
    ds = _strict_loads(s.to_json())
    assert ds["schema"] == "pitbacktest/screen-result" and len(ds["summary"]["data"]) == 2 and ds["summary"]["columns"][0] == "factor"
    assert ds["survivors"] == s.survivors and set(ds["factors"]) == {"a", "b"}
    j = jsonable({1: np.int64(3), "x": np.float32(1.5), "n": np.nan, "i": np.inf, "t": pd.Timestamp("2020-01-02"), "a": np.array([1.0, np.nan]), "b": np.bool_(True),
                  "df": pd.DataFrame({"u": [1.0, np.nan]}, index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]), "s": pd.Series([1, 2], index=["p", "q"])})
    assert j["1"] == 3 and j["x"] == 1.5 and j["n"] is None and j["i"] is None and j["t"].startswith("2020-01-02") and j["a"] == [1.0, None] and j["b"] is True
    assert j["df"]["columns"] == ["u"] and j["df"]["data"] == [[1.0], [None]] and j["s"] == {"index": ["p", "q"], "data": [1, 2]}
    _strict_loads(json.dumps(j))
    print("EX5 event and screen results convert to JSON; numpy, pandas, NaN, infinity, timestamps and integer keys are handled  PASS")


if __name__ == "__main__":
    test_portfolio_json()
    test_series_arguments()
    test_nulls()
    test_notes()
    test_other_results()
    print("export tests: all passed")
