"""Defects found by independent review of 0.3.0, each pinned.

RV1 grid, excess   after a ruin the 25-cell grid agrees with the headline and the excess figures stay within -100% (no NaN, no negative equity)
RV2 terminal zero  a zero on the last bar of a series is a bankruptcy: it books -100%; a zero in the middle of a series (a halt) is missing
RV3 alignment      the overlap is measured the way the engines align, so a table that aligns is accepted and one that does not is refused; a sparse event signal does not warn
RV4 intraday       session-based intraday bars with their own periods_per_year do not warn; weekly dates with 252 still do; the panel's warnings reach `result.notes`
RV5 deflated       a constant column and too few usable columns no longer give a silent NaN
RV6 error filter   under `simplefilter("error")` the engine stops at the first warning and leaves no ledger entry
RV7 threads        two engine calls that overlap in two threads keep their own notes and leave the global warning filters alone
RV8 drawdown       the starting capital is a peak (a loss on the first bar is a drawdown); a series with no variation has no Sharpe ratio (not 0, not 1e10)
RV9 report         an interval that cannot be computed is said so, a damaged ledger is flagged, a year's drawdown starts from the year before, intraday and time-zone data read right
RV10 export        NaT, NA, 0-d arrays, datetime arrays, colliding keys and sets are written right; a month without data is null
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
from test_argument_checks import _raises
from test_synthetic import make_panel


def _quiet(fn):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in ws]


def test_grid_and_excess_after_ruin():
    rng = np.random.default_rng(5)
    T, N = 400, 30
    idx = pd.bdate_range("2020-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .01, (T, N)), 0)), idx, cols)
    c.iloc[200:, 0] *= 3.5                                                      # the always-shorted name more than triples
    p = q.Panel(close=c, eligible=pd.DataFrame(True, idx, cols))
    f = pd.DataFrame(rng.normal(0, .01, (T, N)), idx, cols)
    f["S0"] = -10.0
    r, _ = _quiet(lambda: q.backtest_portfolio(p, f, long_q=.1, short_q=1 / 30, hold=1, spread_bp=5, benchmark="equal"))
    assert r.metrics["ruined"] and abs(r.metrics["MDD"] + 1.0) < 1e-12
    cell = r.grid["h1_c5"]
    assert abs(cell["MDD"] - r.metrics["MDD"]) < 1e-9 and abs(cell["CAGR"] - r.metrics["CAGR"]) < 1e-9, (cell, r.metrics["CAGR"])
    assert np.isfinite(r.excess["CAGR"]) and r.excess["MDD"] >= -1.0 - 1e-12, r.excess
    w = pd.DataFrame(0.0, idx, cols); w["S0"] = -1.0; w["S1"] = 1.0
    rw, _ = _quiet(lambda: q.backtest_weights(p, w, spread_bp=0))
    assert rw.metrics["ruined"] and np.isfinite(rw.excess["CAGR"]) and rw.excess["MDD"] >= -1.0 - 1e-12, rw.excess
    print("RV1 after a ruin the grid cell equals the headline (MDD -100%) and the excess figures are finite and not below -100%, in both engines  PASS")


def test_terminal_zero():
    p, rng = make_panel(n_days=300, n_stocks=15)
    c = p.close.copy()
    c.iloc[:, 0] = 5.0
    c.iloc[61:, 0] = 0.0                                                         # bankrupt: the last real price 5, then 0.00 for ever
    w = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    w.iloc[40:, 0] = 1.0
    w.iloc[61:, 0] = 0.0
    pp, msgs = _quiet(lambda: replace(p, close=c))
    assert not any("negative or infinite" in m for m in msgs), msgs              # a terminal zero is not a data error
    r, _ = _quiet(lambda: q.backtest_weights(pp, w, spread_bp=0, benchmark=None, check_universe=False))
    assert abs(r.net_returns.iloc[59] + 1.0) < 1e-12, r.net_returns.iloc[57:63].tolist()        # the row whose holding period is close(60) -> close(61)
    assert r.metrics["ruined"] and (r.net_returns.iloc[60:] == 0.0).all()
    # the same zero in the middle of a series, the name trading again afterwards: a halt written as 0, so missing
    c2 = p.close.copy(); c2.iloc[:, 0] = 5.0; c2.iloc[61:64, 0] = 0.0
    w2 = w.copy(); w2.iloc[61:, 0] = 1.0
    pp2, msgs2 = _quiet(lambda: replace(p, close=c2))
    assert any("negative or infinite" in m for m in msgs2) and pp2.close.iloc[61:64, 0].isna().all() and pp2.close.iloc[64, 0] == 5.0
    r2, _ = _quiet(lambda: q.backtest_weights(pp2, w2, spread_bp=0, benchmark=None, check_universe=False))
    assert not r2.metrics["ruined"] and np.isfinite(r2.net_returns.to_numpy()).all()
    assert (pp.close.iloc[61, 0] == 0.0) and pp.close.iloc[62:, 0].isna().all()   # only the first zero of the terminal run stays
    print("RV2 a zero at the end of a series books -100% (the rest of the run is missing); a zero before the name trades again is a missing price  PASS")


def test_alignment_follows_reindex():
    p, rng = make_panel(n_days=400, n_stocks=30)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    for label, x in (("date objects", f.set_axis([d.date() for d in f.index], axis=0)),
                     ("string dates", f.set_axis([str(d.date()) for d in f.index], axis=0)),
                     ("ints as names", f.set_axis(range(f.shape[1]), axis=1))):
        aligns = bool(x.reindex(index=p.dates, columns=p.tickers).notna().any(axis=None))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q.backtest_portfolio(p, x, hold=5, spread_bp=10)
            ran = True
        except ValueError as e:
            ran = False
            assert "in common" in str(e), e
        assert ran == aligns, (label, ran, aligns)                                # what reindex aligns is accepted, and the other way round
    sig = pd.DataFrame(False, index=p.dates, columns=p.tickers)
    sig.iloc[::80, :6] = True                                                     # an event table that is almost empty on most dates
    sparse = sig.loc[sig.any(axis=1)]                                             # only the event dates, 5 of 400
    _, msgs = _quiet(lambda: q.backtest_event(p, sparse, horizons=(5,), neutralize_check=False) if False else q.core.panel.check_alignment(p, sparse, "signal", sparse=True))
    assert not any("covers only" in m for m in msgs), msgs
    _, msgs = _quiet(lambda: q.core.panel.check_alignment(p, f.loc[f.index[::10]], "factor"))
    assert any("covers only 10% of the panel's dates" in m for m in msgs), msgs
    print("RV3 the alignment check accepts exactly what reindex aligns (date objects, string dates, int names), and a sparse event table does not warn about its dates  PASS")


def test_periods_and_panel_notes():
    idx = pd.bdate_range("2021-01-04", periods=40).repeat(78)
    idx = idx + pd.to_timedelta(np.tile(np.arange(78) * 5, 40), unit="min") + pd.Timedelta(hours=9, minutes=30)
    cols = [f"S{i}" for i in range(12)]
    cl = pd.DataFrame(100.0, index=idx, columns=cols)
    _, msgs = _quiet(lambda: q.Panel(close=cl, eligible=pd.DataFrame(True, index=idx, columns=cols), periods_per_year=252 * 78))
    assert not any("periods_per_year" in m for m in msgs), msgs                   # 5-minute session bars: no false alarm
    p, rng = make_panel(n_days=300, n_stocks=15)
    wk = p.close.resample("W-FRI").last()
    _, msgs = _quiet(lambda: q.Panel(close=wk, eligible=pd.DataFrame(True, index=wk.index, columns=wk.columns)))
    assert any("periods_per_year=252" in m for m in msgs)                         # weekly dates with 252 still warn
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    c = p.close.copy(); c.iloc[100, 3] = -5.0
    pp, _ = _quiet(lambda: replace(p, close=c))
    r, _ = _quiet(lambda: q.backtest_portfolio(pp, f, hold=5, spread_bp=10))
    assert any("treated as missing" in n for n in r.notes), r.notes               # the panel's own warning travels with the result
    assert any("treated as missing" in n for n in r.to_dict()["notes"])
    again, _ = _quiet(lambda: replace(pp, volume=p.volume))                       # a copy keeps the notes and does not repeat them
    assert again.meta["construction_notes"] == pp.meta["construction_notes"] and len(pp.meta["construction_notes"]) == 1
    print("RV4 5-minute session bars with 252*78 do not warn, weekly with 252 still does, and the panel's warnings reach result.notes (and survive replace)  PASS")


def test_deflated_sharpe_cases():
    R = np.random.default_rng(1).normal(5e-4, .01, (500, 4))
    R[:, 3] = 0.5                                                                 # a constant column: an infinite Sharpe
    d = q.validation.deflated_sharpe(R)
    assert np.isfinite(d["dsr"]) and np.isfinite(d["sharpe_luck_benchmark"]) and d["best"] != 3, d
    R2 = np.random.default_rng(1).normal(5e-4, .01, (500, 3))
    R2[3, :2] = np.nan
    _raises(lambda: q.validation.deflated_sharpe(R2), "only 1 column(s)", "at least two")
    one = q.validation.deflated_sharpe(R2[:, 2:3])                                # a single strategy and one trial is fine
    assert one["trials"] == 1 and np.isfinite(one["dsr"])
    print("RV5 a constant column is ignored, one usable column among several trials raises, a single strategy still works  PASS")


def test_error_filter_stops_early():
    p, rng = make_panel(n_days=118, n_stocks=30)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    led = q.Ledger(tempfile.mkdtemp(prefix="rv-ledger-"))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            q.backtest_portfolio(p, f, hold=5, spread_bp=10, ledger=led, family="x", name="a")      # under half a year: a warning
        raise AssertionError("it must raise")
    except UserWarning as e:
        assert "observations" in str(e)
    assert led.n_trials("x") == 0, led.n_trials("x")                                                 # nothing was recorded
    _, msgs = _quiet(lambda: q.backtest_portfolio(p, f, hold=5, spread_bp=10, ledger=led, family="x", name="a"))
    assert led.n_trials("x") == 1 and any("observations" in m for m in msgs)
    print("RV6 under an error filter the engine stops at the first warning and leaves no ledger entry; with the default filter it runs and records  PASS")


def test_threads_keep_their_own_notes():
    import threading
    from pitbacktest.core.notes import keep_warnings, warn

    class R:
        notes = None

    a_in, b_in, a_done = threading.Event(), threading.Event(), threading.Event()

    @keep_warnings
    def eng_a():
        a_in.set(); b_in.wait(5)
        warn("A-warning")
        return R()

    @keep_warnings
    def eng_b():
        b_in.set(); a_done.wait(5)                                                 # B leaves after A: the order that corrupted a global filter swap
        warn("B-warning")
        return R()

    out = {}

    def run_a():
        out["a"] = eng_a()
        a_done.set()

    def run_b():
        a_in.wait(5)
        out["b"] = eng_b()

    with warnings.catch_warnings():                                                # the one filter change, made here in the main thread and undone here
        warnings.simplefilter("ignore")
        before = list(warnings.filters)
        ta, tb = threading.Thread(target=run_a), threading.Thread(target=run_b)
        ta.start(); tb.start(); ta.join(10); tb.join(10)
        assert warnings.filters == before                                          # the engines changed nothing global
    assert out["a"].notes == ["A-warning"] and out["b"].notes == ["B-warning"], (out["a"].notes, out["b"].notes)
    _, msgs = _quiet(lambda: warnings.warn("later warning", UserWarning))
    assert msgs == ["later warning"]
    print("RV7 two overlapping engine calls keep separate notes, the global filters are untouched, and later warnings still show  PASS")


def test_drawdown_start_and_no_variation():
    from pitbacktest.portfolio import metrics
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = np.zeros(300); r[0] = -0.30; r[1:] = 0.0002                                    # a loss on the first bar, slow recovery that never gets back to 1
    m = metrics(r, idx)
    eq = (1 + pd.Series(r, index=idx)).cumprod()
    assert abs(m["MDD"] - (eq.min() - 1.0)) < 1e-12 and m["mdd_peak"] == "start" and m["MDD"] < -0.29, m
    flat = metrics(np.full(300, 0.001), idx)
    assert np.isnan(flat["Sharpe"]) and np.isnan(flat["Sortino"]), flat
    zero = metrics(np.zeros(300), idx)
    assert np.isnan(zero["Sharpe"]) and zero["CAGR"] == 0.0
    mixed = metrics(np.tile([0.01, -0.01], 150), idx)
    assert np.isfinite(mixed["Sharpe"]) and mixed["mdd_peak"] != "start"
    print("RV8 a first-bar loss is a drawdown (MDD -30%, peak 'start'); constant and zero returns have no Sharpe ratio; a normal series is unchanged  PASS")


def test_report_cases():
    from test_report import _result, _charts
    p, f, r = _result()
    idx = r.net_returns.index
    # an interval that cannot be computed
    zero = replace(r, net_returns=pd.Series(0.0, index=idx), benchmark_returns=None)
    page = zero.report()
    assert "could not be computed" in page and "it excludes 0" not in page and 'class="ok"><span class="ic">✓</span><span class="lab">Sharpe interval' not in page
    # a damaged ledger
    led = q.Ledger(tempfile.mkdtemp(prefix="rv9-"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in (3, 5, 10, 20):
            q.backtest_portfolio(p, -p.close.pct_change(k), hold=5, spread_bp=10, ledger=led, family="x", name=f"k{k}")
    lines = led.file.read_text().splitlines()
    led.file.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    bad = r.report(ledger=led, family="x")
    assert "Ledger damaged" in bad and 'class="card read bad"' in bad
    # a loss on the first bar of a later year counts in that year's drawdown
    two = pd.bdate_range("2021-01-01", periods=520)
    s = pd.Series(0.001, index=two)
    s.loc[s.index[s.index.year == 2022][0]] = -0.20
    tbl = replace(r, net_returns=s, benchmark_returns=None).report()
    assert "-20.0%" in tbl.split("Worst drawdown")[1], "the year's first-bar loss must be in the year's worst drawdown"
    # intraday
    five = pd.date_range("2024-01-01", periods=600, freq="5min")
    sh = replace(r, net_returns=pd.Series(np.random.default_rng(0).normal(0, 1e-4, 600), index=five), benchmark_returns=None)
    pg = sh.report()
    c = _charts(pg)
    assert c["equity"]["labels"][1] != c["equity"]["labels"][0] and "00:05" in c["equity"]["labels"][1], c["equity"]["labels"][:2]
    assert "Not drawn" in pg or c["rolling"]["series"][0]["y"].count(None) < len(c["rolling"]["series"][0]["y"])
    full = sh.to_dict(series="full")["series"]["dates"]
    assert len(set(full)) == 600 and full[1].endswith("00:05:00"), full[:2]
    # a time zone: the dates read as written, nothing leaks
    kst = pd.bdate_range("2021-01-01", periods=300, tz="Asia/Seoul")
    tz = replace(r, net_returns=pd.Series(np.random.default_rng(1).normal(0, 0.005, 300), index=kst), benchmark_returns=None)
    pgz, msgs = _quiet(lambda: tz.report())
    assert "2021-01-01 to " in pgz and not any("timezone" in m.lower() for m in msgs), msgs
    assert tz.to_dict()["series"]["months"][0] == "2021-01"
    _, msgs = _quiet(lambda: tz.to_dict())
    assert not any("timezone" in m.lower() for m in msgs)
    # negative zero in the heat map, capacity checks
    neg = pd.Series(-1e-7, index=idx)
    assert "-0.0<" not in replace(r, net_returns=neg, benchmark_returns=None).report()
    _raises(lambda: r.report(capacity=pd.DataFrame({"y": [1.0], "aum": [1e6]})), "capacity_curve", "sharpe")
    _raises(lambda: r.report(capacity=pd.DataFrame({"y": [1.0, 1.0], "aum": [1e6, 1e6], "sharpe": [0, 0], "cagr": [0, 0]})), "same y and aum")
    print("RV9 an uncomputable interval is said so, a damaged ledger is flagged, a year's drawdown starts from the year before, intraday and time-zone data read right  PASS")


def test_export_values():
    from pitbacktest.export import jsonable, monthly_returns
    j = jsonable({"nat": pd.NaT, "na": pd.NA, "npnat": np.datetime64("NaT"), "zero_d": np.array(3.5), "dt": np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]"),
                  "days": np.array(["2020-01-01"], dtype="datetime64[D]"), "s": {"b", "a", "c"}, "td": pd.Timedelta("1D"), "per": pd.Period("2020-01", "M")})
    assert j["nat"] is None and j["na"] is None and j["npnat"] is None and j["zero_d"] == 3.5
    assert j["dt"][0].startswith("2020-01-01") and j["dt"][1] is None and j["days"][0].startswith("2020-01-01"), j
    assert j["s"] == ["a", "b", "c"] and j["td"] == "1 days 00:00:00" and j["per"] == "2020-01"
    _raises(lambda: jsonable({1: "a", "1": "b"}), "same text")
    idx = pd.bdate_range("2019-01-01", periods=200)
    s = pd.Series(0.001, index=idx); s.iloc[60:130] = np.nan
    m = monthly_returns(s)
    hole = [p for p in m.index if s[(s.index.year == p.year) & (s.index.month == p.month)].isna().all()]
    assert hole and all(np.isnan(m[p]) for p in hole), "a month with no data is NaN, not 0"
    p, f, r = __import__("test_report")._result()
    d = replace(r, net_returns=s.reindex(r.net_returns.index), benchmark_returns=None).to_dict()
    assert d["series"]["net_return"].count(None) >= 1 and len(d["series"]["months"]) == len(d["series"]["equity_at_month_end"])
    print("RV10 NaT, NA, 0-d arrays, datetime arrays, colliding keys and sets are written right; a month without data is null  PASS")


if __name__ == "__main__":
    test_grid_and_excess_after_ruin()
    test_terminal_zero()
    test_alignment_follows_reindex()
    test_periods_and_panel_notes()
    test_deflated_sharpe_cases()
    test_error_filter_stops_early()
    test_threads_keep_their_own_notes()
    test_drawdown_start_and_no_variation()
    test_report_cases()
    test_export_values()
    print("review-fix tests: all passed")
