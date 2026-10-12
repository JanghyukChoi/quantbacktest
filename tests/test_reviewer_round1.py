"""Changes made after a human read the specification (docs/blind_review/ and the reviewer's fourteen points), each one pinned.

RR1 horizon      `primary_horizon` fixes the horizon before looking; without it the choice is flagged as a search
RR2 ledger       events and screens are trials; unrecorded variants can be declared (`extra_trials`)
RR3 borrow       `backtest_portfolio` takes a borrow fee; an unknown fee on a held short raises; no shortable information is said
RR4 cost date    a rate that varies with the date is the one in force on the day of the trade (signal date + entry_lag)
RR5 missing      a held position with no price to book is counted and warned about; a halt and a delisting are not
RR6 panel        entry_lag=0 is flagged; dates and tickers are in the fingerprint; the date a return was earned is on the result
RR7 delisting    the review shows what a delisting loss would do
RR8 screen       the opt-in gate on the neutralised portfolio, and what `passed` means
RR9 documents    the claim that the portfolio engine has no impact model is gone
"""
from __future__ import annotations
import dataclasses, math, sys, tempfile, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.equity.scenarios import inject_delistings
from test_argument_checks import _raises
from test_synthetic import make_panel


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _notes(fn):
    r = _quiet(fn)
    return r, list(r.notes)


def _event_world(seed=4, n_days=600, n_stocks=40, p_fire=0.05):
    p, _ = make_panel(n_days=n_days, n_stocks=n_stocks, seed=seed)
    sig = pd.DataFrame(np.random.default_rng(seed).random(p.close.shape) < p_fire, index=p.dates, columns=p.tickers) & p.eligible
    return p, sig


def test_horizon():
    p, sig = _event_world()
    free, notes = _notes(lambda: q.backtest_event(p, sig, horizons=(1, 5, 20), neutralize_check=False))
    best = free.spec["best_horizon"]
    assert any(f"best of {len(free.per_horizon)} horizons" in n and "primary_horizon" in n for n in notes) and free.spec["horizon_fixed_in_advance"] is False
    other = next(h for h in free.per_horizon if h != best)
    fixed, notes2 = _notes(lambda: q.backtest_event(p, sig, horizons=(1, 5, 20), neutralize_check=False, primary_horizon=other))
    assert fixed.spec["best_horizon"] == other and fixed.spec["horizon_fixed_in_advance"] is True and not any("best of" in n for n in notes2)
    assert set(fixed.per_horizon) == set(free.per_horizon)                                 # the others are still reported
    assert fixed.per_horizon[other]["t"] == free.per_horizon[other]["t"] and f"{abs(free.per_horizon[other]['t']):.2f}" in fixed.gates["steps"][0]["detail"]            # the gate judges the fixed horizon's t
    _raises(lambda: q.backtest_event(p, sig, horizons=(1, 5), primary_horizon=3), "primary_horizon")
    _raises(lambda: q.backtest_event(p, sig, horizons=(1, 5), primary_horizon=True), "primary_horizon")
    rv = _quiet(lambda: q.review_event(p, sig, horizons=(1, 5, 20), n_permutations=0, as_portfolio=False, primary_horizon=other))
    assert rv.horizon == other and not any("best of" in n for n in rv.notes)
    print("RR1 primary_horizon fixes the horizon in advance; without it the choice is flagged as a search  PASS")


def test_ledger():
    p, sig = _event_world()
    with tempfile.TemporaryDirectory(prefix="rr2-") as d:
        led = q.Ledger(d)
        _quiet(lambda: q.backtest_event(p, sig, horizons=(5,), neutralize_check=False, ledger=led, family="ev", name="a"))
        _quiet(lambda: q.backtest_event(p, sig, horizons=(5,), neutralize_check=False, cost_bp=10.0, ledger=led, family="ev", name="b"))
        _quiet(lambda: q.backtest_event(p, sig, horizons=(5,), neutralize_check=False, ledger=led, family="ev", name="a"))          # the same settings again: one trial
        assert led.n_trials("ev") == 2
        rng = np.random.default_rng(3)
        facs = {f"f{i}": pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers) for i in range(4)}
        _quiet(lambda: q.screen(p, facs, horizons=(5,), primary_h=5, n_null=10, neutralize_all=False, ledger=led, family="scr"))
        assert led.n_trials("scr") == 4 and led.verify()["ok"]
        base = led.deflated_sharpe("scr")
        more = led.deflated_sharpe("scr", extra_trials=96)
        assert base["trials"] == 4 and more["trials"] == 100 and more["extra_trials"] == 96 and more["dsr"] < base["dsr"] and more["sharpe_luck_benchmark"] > base["sharpe_luck_benchmark"]
        _raises(lambda: led.deflated_sharpe("scr", extra_trials=-1), "extra_trials")
        _raises(lambda: led.deflated_sharpe("scr", extra_trials=2.5), "extra_trials")
        f = facs["f0"]
        rv0 = _quiet(lambda: q.review_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5, n_permutations=0, make_factor=lambda panel: f, grid={"hold": [1, 3, 5, 10]}))
        rv1 = _quiet(lambda: q.review_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5, n_permutations=0, make_factor=lambda panel: f, grid={"hold": [1, 3, 5, 10]}, extra_trials=46))
        assert rv0.deflated["trials"] == 4 and rv1.deflated["trials"] == 50 and rv1.deflated["dsr"] <= rv0.deflated["dsr"]
    print("RR2 events and screens are recorded as trials; unrecorded variants can be declared with extra_trials  PASS")


def _book(seed=2, n_days=260, n_stocks=24):
    p, _ = make_panel(n_days=n_days, n_stocks=n_stocks, seed=seed)
    f = pd.DataFrame(np.random.default_rng(seed).standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    return p, f


def test_borrow():
    p, f = _book()
    kw = dict(long_q=0.2, short_q=0.2, hold=5, spread_bp=0.0, benchmark=None, grid=False, funding=False)
    base, nb = _notes(lambda: q.backtest_portfolio(p, f, **kw))
    fee, _ = _notes(lambda: q.backtest_portfolio(p, f, borrow_bp=300.0, **kw))
    assert any("Panel.shortable" in n for n in nb) and base.metrics["borrow_annual_bp"] == 0.0
    cut = len(p.dates) - (p.entry_lag + 1)
    step = 300.0 / 1e4 / p.periods_per_year
    d = (base.net_returns - fee.net_returns).to_numpy()
    assert np.allclose(d[70:], step, atol=1e-13) and abs(fee.metrics["borrow_annual_bp"] - 300.0 * (cut - 62) / cut) < 5.0 and fee.spec["borrow_bp"] == 300.0        # the short leg is one unit of capital, held from the first full day
    assert d[:30].max() < step * 0.5                                                          # nothing is held before the first eligible rows
    shortable = pd.DataFrame(True, index=p.dates, columns=p.tickers)
    withs, nw = _notes(lambda: q.backtest_portfolio(dataclasses.replace(p, shortable=shortable), f, **kw))
    assert not any("Panel.shortable" in n for n in nw)
    # a panel of fees: unknown where a short is held raises, with the name and the day; unknown where nothing is shorted is fine
    rate = pd.DataFrame(200.0, index=p.dates, columns=p.tickers)
    hold_now = np.maximum(-np.asarray(base.holdings), 0.0)
    short_name = p.tickers[int(np.flatnonzero(hold_now[100] > 0)[0])]
    bad = rate.copy(); bad.loc[p.dates[100], short_name] = np.nan
    msg = _raises(lambda: q.backtest_portfolio(p, f, borrow_bp=bad, **kw), "unknown (NaN)", "not zero")
    assert short_name in msg and str(p.dates[100 - p.entry_lag].date()) in msg                 # the fee of day 100 is the one for the trade signalled on day 100 - entry_lag
    never = rate.copy(); never.loc[:, p.tickers[0]] = np.nan
    held_ever = (hold_now[:, 0] > 0).any()
    if not held_ever:
        _quiet(lambda: q.backtest_portfolio(p, f, borrow_bp=never, **kw))
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); W.iloc[50:, 3] = -0.5
    nanw = rate.copy(); nanw.iloc[70, 3] = np.nan
    _raises(lambda: q.backtest_weights(p, W, borrow_bp=nanw, benchmark=None, check_universe=False), "unknown (NaN)")
    ok = _quiet(lambda: q.backtest_weights(p, W, borrow_bp=rate, benchmark=None, check_universe=False))
    assert abs(ok.metrics["borrow_annual_bp"] - 200.0 * 0.5 * (len(p.dates) - 50 - 2) / (len(p.dates) - 2)) < 1e-6
    _raises(lambda: q.backtest_portfolio(p, f, borrow_bp=-1.0, **kw), "borrow_bp")
    series = pd.Series([100.0, 400.0], index=[p.dates[0], p.dates[130]])
    s_run = _quiet(lambda: q.backtest_portfolio(p, f, borrow_bp=series, **kw))
    exp = np.where(np.minimum(np.arange(len(p.dates)) + p.entry_lag, len(p.dates) - 1) >= 130, 400.0, 100.0)[:cut]         # the fee in force on the day of the trade
    assert np.allclose((base.net_returns - s_run.net_returns).to_numpy()[70:], (exp / 1e4 / p.periods_per_year)[70:], atol=1e-13)
    print("RR3 borrow_bp is charged on the short leg, an unknown fee on a held short raises, and a missing shortable frame is said  PASS")


def test_cost_dates():
    p, _ = make_panel(n_days=300, n_stocks=20, seed=6)
    rng = np.random.default_rng(1)
    W = pd.DataFrame(np.where(rng.random(p.close.shape) < 0.3, rng.standard_normal(p.close.shape), 0.0), index=p.dates, columns=p.tickers).where(p.eligible, 0.0)
    W = W / np.maximum(1.0, W.abs().sum(axis=1)).to_numpy()[:, None]
    switch = p.dates[150]
    sched = pd.Series([10.0, 100.0], index=[p.dates[0], switch])
    H = W.to_numpy()
    for lag in (0, 1, 2):
        pl = dataclasses.replace(p, entry_lag=lag)
        a = _quiet(lambda: q.backtest_weights(pl, W, benchmark=None, check_universe=False))
        b = _quiet(lambda: q.backtest_weights(pl, W, buy_bp=sched, sell_bp=sched, benchmark=None, check_universe=False))
        cut = len(H) - (lag + 1)
        ref = np.zeros(len(H))
        for t in range(1, len(H)):
            rate = 100.0 if p.dates[min(t + lag, len(H) - 1)] >= switch else 10.0
            ref[t] = np.abs(H[t] - H[t - 1]).sum() * rate / 1e4
        assert np.allclose((a.net_returns - b.net_returns).to_numpy(), ref[:cut], atol=1e-13), lag
        # the same schedule through the portfolio engine, whose holdings are the weights of the long-short book
        f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
        kw = dict(long_q=0.2, short_q=0.2, hold=1, spread_bp=0.0, benchmark=None, grid=False, funding=False)
        pa = _quiet(lambda: q.backtest_portfolio(pl, f, **kw)); pb = _quiet(lambda: q.backtest_portfolio(pl, f, buy_bp=sched, sell_bp=sched, **kw))
        Hp = np.asarray(pa.holdings); refp = np.zeros(len(Hp))
        for t in range(1, len(Hp)):
            refp[t] = np.abs(Hp[t] - Hp[t - 1]).sum() * (100.0 if p.dates[min(t + lag, len(Hp) - 1)] >= switch else 10.0) / 1e4
        assert np.allclose((pa.net_returns - pb.net_returns).to_numpy(), refp[:cut], atol=1e-13), lag
    print("RR4 a buy or sell rate that varies with the date is read on the execution day, for lags 0, 1 and 2, in both engines  PASS")


def test_missing_price():
    p, f = _book(seed=5, n_days=300, n_stocks=24)
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); W.iloc[60:, 2] = 0.3; W.iloc[60:, 3] = -0.3
    clean = _quiet(lambda: q.backtest_weights(p, W, benchmark=None, check_universe=False))
    assert clean.metrics["held_missing_price_days"] == 0 and not any("no price" in n for n in clean.notes)
    gap = p.close.copy(); gap.iloc[100:110, 2] = np.nan                                      # a data gap in a name held long
    pg = dataclasses.replace(p, close=gap)
    r, notes = _notes(lambda: q.backtest_weights(pg, W, benchmark=None, check_universe=False))
    assert r.metrics["held_missing_price_days"] >= 10 and any("no price" in n and p.tickers[2] in n for n in notes)
    # a halt (a price but no volume) is not a missing price
    vol = p.volume.copy(); vol.iloc[100:110, 2] = 0.0
    halted = _quiet(lambda: q.backtest_weights(dataclasses.replace(p, volume=vol), W, benchmark=None, check_universe=False))
    assert halted.metrics["held_missing_price_days"] == 0
    # a flagged delisting is expected to leave no price, and is not counted
    dd = p.close.copy(); dd.iloc[150:, 2] = np.nan
    da = pd.DataFrame(False, index=p.dates, columns=p.tickers); da.iloc[149, 2] = True
    gone = _quiet(lambda: q.backtest_weights(dataclasses.replace(p, close=dd, delist_after=da), W, benchmark=None, check_universe=False))
    assert gone.metrics["held_missing_price_days"] == 0
    unflagged = _quiet(lambda: q.backtest_weights(dataclasses.replace(p, close=dd), W, benchmark=None, check_universe=False))
    assert unflagged.metrics["held_missing_price_days"] > 100                                  # the same disappearance without a flag is a gap
    pf = _quiet(lambda: q.backtest_portfolio(pg, f, long_q=0.3, short_q=0.3, hold=5, benchmark=None, grid=False))
    assert "held_missing_price_days" in pf.metrics
    print("RR5 a held position with no price is counted and warned about; a halt and a flagged delisting are not  PASS")


def test_panel_and_dates():
    p, f = _book()
    p0 = dataclasses.replace(p, entry_lag=0)
    assert any("entry_lag=0" in n for n in p0.meta.get("construction_notes", [])) and not any("entry_lag=0" in n for n in p.meta.get("construction_notes", []))
    cols = list(p.tickers)[::-1]
    reordered = q.Panel(close=p.close[cols], eligible=p.eligible[cols], volume=p.volume[cols], mkt_cap=p.mkt_cap[cols], market=p.market, entry_lag=p.entry_lag)
    assert reordered.fingerprint() != p.fingerprint()
    renamed = q.Panel(close=p.close.rename(columns=lambda c: "x" + c), eligible=p.eligible.rename(columns=lambda c: "x" + c), volume=p.volume.rename(columns=lambda c: "x" + c), mkt_cap=p.mkt_cap.rename(columns=lambda c: "x" + c), market=p.market, entry_lag=p.entry_lag)
    shifted = q.Panel(close=p.close.set_axis(p.dates + pd.Timedelta(days=1)), eligible=p.eligible.set_axis(p.dates + pd.Timedelta(days=1)), volume=p.volume.set_axis(p.dates + pd.Timedelta(days=1)), mkt_cap=p.mkt_cap.set_axis(p.dates + pd.Timedelta(days=1)), market=p.market, entry_lag=p.entry_lag)
    assert renamed.fingerprint() != p.fingerprint() and shifted.fingerprint() != p.fingerprint() and p.fingerprint() == dataclasses.replace(p).fingerprint()
    for lag in (0, 1, 3):
        pl = dataclasses.replace(p, entry_lag=lag)
        r = _quiet(lambda: q.backtest_portfolio(pl, f, long_q=0.2, short_q=0.2, hold=5, benchmark=None, grid=False))
        assert len(r.pnl_dates) == len(r.net_returns) and (r.pnl_dates == pl.dates[lag + 1:lag + 1 + len(r.net_returns)]).all() and r.pnl_dates[0] > r.net_returns.index[0]
        assert (r.pnl_dates - r.net_returns.index).min() >= pd.Timedelta(days=1)
    w = _quiet(lambda: q.backtest_weights(p, pd.DataFrame(0.0, index=p.dates, columns=p.tickers), benchmark=None, check_universe=False))
    assert (w.pnl_dates == p.dates[2:2 + len(w.net_returns)]).all()
    print("RR6 entry_lag=0 is flagged, the fingerprint covers dates and tickers, and the result carries the date each return was earned  PASS")


def test_delisting_stress():
    p, f = _book(seed=3, n_days=400, n_stocks=40)
    dp = inject_delistings(p, annual_rate=0.2, seed=1)
    kw = dict(long_q=0.2, short_q=0.2, hold=5)
    rv = _quiet(lambda: q.review_portfolio(dp, f, n_permutations=0, **kw))
    t = rv.delist_table
    assert list(t.index) == [0.0, -0.3, -0.5, -1.0] and (t["delist_events_held"] > 0).all()
    for dr in (0.0, -0.5, -1.0):
        r = _quiet(lambda: q.backtest_portfolio(dp, f, delist_return=dr, benchmark=None, grid=False, **kw))
        assert abs(t.loc[dr, "sharpe"] - r.metrics["Sharpe"]) < 1e-12 and abs(t.loc[dr, "cagr"] - r.metrics["CAGR"]) < 1e-12
    assert any("delisting stress" in n for n in rv.notes) and rv.to_dict()["delisting_stress"] is not None and "What if a delisting loses money?" in rv.report()
    plain = _quiet(lambda: q.review_portfolio(p, f, n_permutations=0, **kw))
    assert plain.delist_table is None and "What if a delisting loses money?" not in plain.report() and plain.to_dict()["delisting_stress"] is None
    print("RR7 the review shows Sharpe and CAGR if a delisting loses 0, 30, 50 or 100 percent, equal to direct runs, and only when the panel flags delistings  PASS")


def test_screen_gate():
    p, _ = make_panel(n_days=700, n_stocks=60, seed=9)
    facs = {"mom": p.close.pct_change(20), "rev": -p.close.pct_change(5)}
    plain = _quiet(lambda: q.screen(p, facs, horizons=(20,), n_null=10))
    assert all(len(d["gates"]["steps"]) == 6 for d in plain.factors.values())
    gated = _quiet(lambda: q.screen(p, facs, horizons=(20,), n_null=10, gate_config=q.GateConfig(min_neutral_survival=50.0)))
    for name, d in gated.factors.items():
        steps = d["gates"]["steps"]
        assert len(steps) == 7 and steps[-1]["gate"] == "G7 survives neutralisation"
        assert d["gates"]["passed"] == all(s["pass"] for s in steps) and d["gates"]["failed_at"] == next((s["gate"] for s in steps if not s["pass"]), None)
        nu = d["neutralized"]
        want = bool(np.isfinite(nu["survival_pct"]) and nu["survival_pct"] >= 50.0 and np.isfinite(nu["net_bp"]) and nu["net_bp"] > 0)
        assert steps[-1]["pass"] == want
        assert gated.summary.set_index("factor").loc[name, "passed"] == d["gates"]["passed"]
    assert "does not say that the controlled portfolio is profitable" in q.screen.__doc__
    print("RR8 the opt-in neutralised gate adds a seventh step and decides `passed`; the default stays six gates  PASS")


def test_documents_agree():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "verification_status.md").read_text(encoding="utf-8")
    assert "has no impact model" not in text
    import inspect
    assert "impact" in inspect.signature(q.backtest_portfolio).parameters and "borrow_bp" in inspect.signature(q.backtest_portfolio).parameters
    assert (root / "docs" / "assumptions.md").exists()
    print("RR9 the documents no longer say that the portfolio engine has no impact model  PASS")


if __name__ == "__main__":
    for fn in (test_horizon, test_ledger, test_borrow, test_cost_dates, test_missing_price, test_panel_and_dates, test_delisting_stress, test_screen_gate, test_documents_agree):
        fn()
    print("reviewer round 1 tests: all passed")
