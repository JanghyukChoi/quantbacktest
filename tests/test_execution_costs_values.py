"""Execution, costs, the ledger, survivorship scenarios, export and short-sale bans, on small inputs worked out by hand. A mutation run caught 54 to 90 percent of the changes to these files.

XV1 tradability  what can be bought and sold: delisted names, price limits at the edge, long gaps, the suspension counter and freeze_days
XV2 episodes     every outcome of `freeze_episodes` on a constructed panel
XV3 realize      the blocked-trade statistics of a small hand-worked book
XV4 costs        Roll's spread, the spread regression, rate schedules, side-cost validation
XV5 crypto costs the tick-size spread, the fixed spread and the participation report
XV6 ledger       hash length, the Sharpe of a record, tamper detection, the returns matrix
XV7 scenarios    injected delistings, hazard weights, the scenario table, survivors only
XV8 export       text fallback, intraday labels, the JSON writer
XV9 bans         one-day bans, exemptions
"""
from __future__ import annotations
import dataclasses, json, math, sys, tempfile, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.core.costs import fit_spread_model, rate_schedule, roll_spread, side_cost_input
from pitbacktest.crypto.costs import liquidity_cost_bp, participation_report
from pitbacktest.equity.scenarios import _weights, inject_delistings, survivors_only, survivorship_scenarios
from pitbacktest.execution import freeze_episodes, freeze_hits, halt_runs, realize, tradability
from pitbacktest.export import dump_json, iso_labels, jsonable
from pitbacktest.shorting import shortable_from_bans
from test_argument_checks import _raises
from test_synthetic import make_panel


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _mini(T=120, N=12, seed=1, **kw):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=T); cols = [f"S{i:02d}" for i in range(N)]
    close = pd.DataFrame(100.0 + np.cumsum(rng.normal(0, 0.1, (T, N)), axis=0), index=idx, columns=cols)
    vol = pd.DataFrame(1000.0, index=idx, columns=cols)
    return close, vol, idx, cols


def test_tradability():
    close, vol, idx, cols = _mini()
    close, vol = close.copy(), vol.copy()
    el = pd.DataFrame(True, index=idx, columns=cols)
    # a delisting flagged on row 50 for the first name, which is also halted (no volume) that day: the day itself is halted, the days after allow an exit
    vol.iloc[50, 0] = 0.0
    da = pd.DataFrame(False, index=idx, columns=cols); da.iloc[50, 0] = True
    close.iloc[51:, 0] = np.nan
    p = q.Panel(close=close, eligible=el, volume=vol, delist_after=da, market="T", entry_lag=1)
    cb, cs = tradability(p)
    assert not cb.iloc[50, 0] and not cs.iloc[50, 0] and cb.iloc[51, 0] and cs.iloc[51, 0] and cb.iloc[100, 0]
    assert cb.iloc[49, 0] and cb.iloc[50, 1:].all()
    # price limits: a move of exactly limit - tol is already locked
    c2 = close.copy(); c2.iloc[59, 1], c2.iloc[60, 1] = 100.0, 129.0; c2.iloc[69, 2], c2.iloc[70, 2] = 100.0, 71.0
    p2 = q.Panel(close=c2, eligible=el, volume=vol, market="T", entry_lag=1)
    cb2, cs2 = tradability(p2, limit=0.30, tol=0.01)
    assert not cb2.iloc[60, 1] and cs2.iloc[60, 1] and not cs2.iloc[70, 2] and cb2.iloc[70, 2]
    c3 = close.copy(); c3.iloc[59, 1], c3.iloc[60, 1] = 100.0, 128.9
    cb3, _ = tradability(q.Panel(close=c3, eligible=el, volume=vol, market="T", entry_lag=1), limit=0.30, tol=0.01)
    assert cb3.iloc[60, 1]                                                              # a hair under the locked move can still be bought
    _raises(lambda: tradability(p2, limit=0.30, tol=0.30), "tol")
    _raises(lambda: tradability(p2, limit=1.5), "limit")
    # a gap longer than max_gap_days lets a position be closed either way
    c4 = close.copy(); c4.iloc[30:40, 3] = np.nan
    cb4, _ = tradability(q.Panel(close=c4, eligible=el, volume=vol, market="T", entry_lag=1), max_gap_days=5)
    assert not cb4.iloc[34, 3] and cb4.iloc[35, 3] and cb4.iloc[39, 3] and not cb4.iloc[40 - 1 - 9, 3]
    # halt counter and freeze_hits
    v5 = vol.copy(); v5.iloc[10:40, 4] = 0.0
    p5 = q.Panel(close=close, eligible=el, volume=v5, market="T", entry_lag=2)
    hr = halt_runs(p5)
    assert hr[9, 4] == 0 and hr[10, 4] == 1 and hr[39, 4] == 30 and hr[40, 4] == 0
    fh = freeze_hits(p5, 5)
    assert fh[10 + 4 - 2, 4] and fh[:, 4].sum() == 1                                    # the 5th suspended day minus the entry lag
    assert freeze_hits(p5, 1)[10 - 2, 4] and freeze_hits(p5, None) is None
    _raises(lambda: freeze_hits(p5, 0), "at least 1")
    print("XV1 tradability: delisted exits, limits at the edge, long gaps, the halt counter and freeze_hits  PASS")


def test_episodes():
    close, vol, idx, cols = _mini(T=140, N=12)
    close, vol = close.copy(), vol.copy()
    el = pd.DataFrame(True, index=idx, columns=cols)
    da = pd.DataFrame(False, index=idx, columns=cols)
    # A resumes after a 30-day halt: ret is the first priced day after over the last priced day before
    vol.iloc[10:40, 0] = 0.0
    # B halted 30 days then no price to the end (not flagged): ongoing
    vol.iloc[10:40, 1] = 0.0; close.iloc[40:, 1] = np.nan
    # C halted 30 days, no price for a while, back on day 60: resumed through the gap
    vol.iloc[10:40, 2] = 0.0; close.iloc[40:60, 2] = np.nan
    # D flagged, last bar is a suspended day: ended in halt, ret 0
    vol.iloc[10:40, 3] = 0.0; close.iloc[40:, 3] = np.nan; da.iloc[39, 3] = True
    # E flagged, resumes on day 45, last bar day 50: resumed then delisted
    vol.iloc[10:40, 4] = 0.0; close.iloc[40:45, 4] = np.nan; close.iloc[51:, 4] = np.nan; da.iloc[50, 4] = True
    # F halted from the first day: no price before
    vol.iloc[0:30, 5] = 0.0
    # G halted for 19 days only, still halted on the last day (a different name H for ongoing at the end)
    vol.iloc[10:29, 6] = 0.0
    vol.iloc[100:140, 7] = 0.0
    p = q.Panel(close=close, eligible=el, volume=vol, delist_after=da, market="T", entry_lag=1)
    ep = freeze_episodes(p).set_index("ticker")
    assert set(ep.index) == {"S00", "S01", "S02", "S03", "S04", "S05", "S07"}, list(ep.index)             # G (19 days) is too short
    px = close
    assert ep.loc["S00", "outcome"] == "resumed" and ep.loc["S00", "days"] == 30 and ep.loc["S00", "end"] == idx[39] and ep.loc["S00", "start"] == idx[10]
    assert abs(ep.loc["S00", "ret"] - (px.iloc[40, 0] / px.iloc[9, 0] - 1)) < 1e-12
    assert ep.loc["S01", "outcome"] == "ongoing" and np.isnan(ep.loc["S01", "ret"])
    assert ep.loc["S02", "outcome"] == "resumed" and abs(ep.loc["S02", "ret"] - (px.iloc[60, 2] / px.iloc[9, 2] - 1)) < 1e-12
    assert ep.loc["S03", "outcome"] == "ended_in_halt" and ep.loc["S03", "ret"] == 0.0
    assert ep.loc["S04", "outcome"] == "resumed_then_delisted" and abs(ep.loc["S04", "ret"] - (px.iloc[50, 4] / px.iloc[9, 4] - 1)) < 1e-12
    assert ep.loc["S05", "outcome"] == "resumed" and np.isnan(ep.loc["S05", "ret"]) and ep.loc["S05", "days"] == 30
    assert ep.loc["S07", "outcome"] == "ongoing"
    assert len(freeze_episodes(p, min_days=19)) == len(ep) + 1 and len(freeze_episodes(p, min_days=31)) == 1        # the minimum is inclusive
    print("XV2 every outcome of freeze_episodes on a constructed panel, with its return  PASS")


def test_realize():
    T, N = 6, 3
    target = np.zeros((T, N)); target[0, 1] = 0.3; target[1:, 1] = 0.3; target[1:, 0] = 0.4
    can_buy = np.ones((T, N), bool); can_buy[1:4, 0] = False
    out, st = realize(target, can_buy, None)
    assert np.allclose(out[:, 0], [0, 0, 0, 0, 0.4, 0.4]) and np.allclose(out[:, 1], 0.3)
    # asked turnover starts after the first row; the first row's build from cash is not a request
    assert abs(st["asked_turnover"] - (0.4 * 4)) < 1e-12 and st["asked_turnover"] > 0
    assert abs(st["blocked_turnover"] - 1.2) < 1e-12 and st["blocked_trades"] == 3 and abs(st["blocked_turnover_share"] - 0.75) < 1e-12
    assert abs(st["mean_stuck_weight"] - 1.2 / T) < 1e-12 and st["longest_freeze_days"] == 3 and st["infeasible_days"] == 0 and st["mean_free_scale"] == 1.0
    assert st["min_trade_skipped"] == 0 and st["min_trade_skipped_share"] == 0.0
    print("XV3 the blocked-trade statistics of a hand-worked book  PASS")


def test_costs():
    # Roll: negative first-order autocovariance of price changes
    rng = np.random.default_rng(2)
    base = 100 + np.cumsum(rng.normal(0, 0.05, 400)); half = 0.02
    px = base + half * rng.choice([-1, 1], 400)
    d = np.diff(px); c = np.cov(d[1:], d[:-1])[0, 1]
    assert c < 0 and abs(roll_spread(px) - 2 * math.sqrt(-c)) < 1e-12
    assert np.isnan(roll_spread(np.arange(40.0))) and np.isnan(roll_spread(px[:29])) and np.isfinite(roll_spread(px[:30]))
    assert np.isnan(roll_spread(np.cumsum(rng.normal(0, 1, 400)) * 0 + np.arange(400) ** 1.5))
    # the spread regression: log(bp) = a + b log(M)
    dv = np.exp(rng.uniform(-1, 6, 60)); bp = np.exp(1.0 - 0.3 * np.log(dv))
    a, b = fit_spread_model(dv, bp)
    assert abs(a - 1.0) < 1e-9 and abs(b + 0.3) < 1e-9
    assert np.isfinite(fit_spread_model(dv[:20], bp[:20])[0]) and np.isnan(fit_spread_model(dv[:19], bp[:19])[0])
    bad = dv.copy(); bad[:5] = [0, -1, np.nan, np.inf, 0.0]
    a2, b2 = fit_spread_model(bad[:25], bp[:25])                                          # five unusable rows leave 20: still fitted, on the good ones only
    assert abs(a2 - 1.0) < 1e-9
    zb = bp.copy(); zb[:3] = [0, -1, np.nan]
    assert abs(fit_spread_model(dv[:23], zb[:23])[0] - 1.0) < 1e-9
    # rate schedule: each date takes the latest entry on or before it
    dates = pd.bdate_range("2020-01-01", periods=10)
    rs = rate_schedule(dates, [("2020-01-01", 1.0), ("2020-01-06", 2.0)])
    assert list(rs) == [1.0] * 3 + [2.0] * 7
    _raises(lambda: rate_schedule(dates, [("2020-01-03", 1.0)]), "no rate is known before 2020-01-03")
    _raises(lambda: rate_schedule(dates, []), "empty")
    # side costs: a half basis point is fine, negatives, infinities and unknowns are not
    arr = np.full((10, 3), 0.5)
    assert np.array_equal(side_cost_input(arr, dates, pd.Index(list("abc")), "buy_bp"), arr)
    assert side_cost_input(None, dates, pd.Index(list("abc")), "x") == 0.0 and side_cost_input(0.0, dates, pd.Index(list("abc")), "x") == 0.0
    for badv in (np.where(np.eye(10, 3) > 0, -0.1, 0.5), np.where(np.eye(10, 3) > 0, np.inf, 0.5)):
        _raises(lambda: side_cost_input(badv, dates, pd.Index(list("abc")), "x"), "finite and not negative")
    _raises(lambda: side_cost_input(np.where(np.eye(10, 3) > 0, np.nan, 0.5), dates, pd.Index(list("abc")), "x"), "unknown")
    _raises(lambda: side_cost_input(-1.0, dates, pd.Index(list("abc")), "x"), "not negative")
    _raises(lambda: side_cost_input(np.zeros((9, 3)), dates, pd.Index(list("abc")), "x"), "shape")
    sr = side_cost_input(pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-08"])), dates, pd.Index(list("abc")), "x")
    assert sr.shape == (10, 1) and list(sr[:, 0]) == [1.0] * 5 + [2.0] * 5
    print("XV4 Roll's spread, the spread regression, rate schedules and side-cost validation  PASS")


def test_crypto_costs():
    p, _ = make_panel(n_days=120, n_stocks=12, seed=4)
    fixed = _quiet(lambda: liquidity_cost_bp(p, spread_estimator="fixed", half_spread_bp=2.0, taker_fee_bp=4.0, thin_usd=0.0))
    assert np.allclose(fixed.to_numpy(), 6.0)
    tick = pd.Series(0.01, index=p.tickers)
    tk = _quiet(lambda: liquidity_cost_bp(p, spread_estimator="tick", tick_size=tick, taker_fee_bp=4.0, thin_usd=0.0))
    assert np.allclose(tk.to_numpy(), (4.0 + 0.5 * 0.01 / p.close * 1e4).to_numpy())
    _raises(lambda: liquidity_cost_bp(p, spread_estimator="tick"), "needs tick_size")
    zero = tick.copy(); zero.iloc[2] = 0.0
    msg = _raises(lambda: liquidity_cost_bp(p, spread_estimator="tick", tick_size=zero), "1 tickers")
    assert p.tickers[2] in msg
    nanv = tick.copy(); nanv.iloc[0] = np.nan
    _raises(lambda: liquidity_cost_bp(p, spread_estimator="tick", tick_size=nanv), "not positive")
    _raises(lambda: liquidity_cost_bp(p, spread_estimator="nope"), "nope")
    thin = _quiet(lambda: liquidity_cost_bp(p, spread_estimator="fixed", half_spread_bp=1.0, taker_fee_bp=0.0, thin_usd=1e30, thin_extra_bp=7.0))
    assert np.allclose(thin.iloc[40:].to_numpy(), 8.0)
    # participation: |trade| x AUM / ADV over all real trades
    rng = np.random.default_rng(5)
    W = np.abs(rng.normal(0, 0.1, p.close.shape)); W[10:20] = W[9]                          # ten days without trades
    rep = participation_report(p, W, 1e6, window=30)
    adv = p.adv(30).to_numpy(); dw = np.abs(np.diff(W, axis=0))
    part = (dw * 1e6 / np.where(adv[1:] > 0, adv[1:], np.nan)); part = part[(dw > 1e-9) & np.isfinite(part)]
    assert rep["n_trades"] == part.size and abs(rep["participation_p50"] - np.percentile(part, 50)) < 1e-15
    assert abs(rep["participation_p95"] - np.percentile(part, 95)) < 1e-15 and abs(rep["participation_p99"] - np.percentile(part, 99)) < 1e-15
    assert abs(rep["aum_at_1pct_p95"] - 1e6 * 0.01 / np.percentile(part, 95)) < 1e-6
    still = participation_report(p, np.full(p.close.shape, 0.1), 1e6)
    assert still == {"aum_usd": 1e6, "n_trades": 0}
    print("XV5 crypto spreads (fixed, tick, thin) and the participation report equal their definitions  PASS")


def test_ledger():
    with tempfile.TemporaryDirectory(prefix="xv6-") as d:
        led = q.Ledger(Path(d) / "a" / "b")                                              # parents are created
        q.Ledger(Path(d) / "a" / "b")                                                    # and an existing folder is fine
        idx = pd.bdate_range("2020-01-01", periods=60)
        rng = np.random.default_rng(1)
        for i in range(3):
            s = pd.Series(rng.normal(0.001 * (i + 1), 0.01, 60), index=idx)
            row = led.record("fam", f"trial{i}", s, {"k": i})
        assert len(row["config_hash"]) == 16 and row["family"] == "fam" and row["n_obs"] == 60
        assert abs(row["sharpe_per_period"] - s.mean() / s.std(ddof=1)) < 1e-15
        three = led.record("fam2", "short", pd.Series([0.01, -0.02, 0.03], index=idx[:3]), {"k": "x"})
        two = led.record("fam2", "two", pd.Series([0.01, -0.02], index=idx[:2]), {"k": "y"})
        flat = led.record("fam2", "flat", pd.Series([0.01] * 5, index=idx[:5]), {"k": "z"})
        assert three["sharpe_per_period"] is not None and two["sharpe_per_period"] is None and flat["sharpe_per_period"] is None
        assert led.verify() == {"ok": True, "first_bad_line": None}
        R, names = led.returns_matrix("fam")
        assert R.shape == (60, 3) and all("#" in n and len(n.split("#")[1]) == 6 for n in names) and names[0].startswith("trial0#")
        _raises(lambda: led.returns_matrix("none"), "no trials")
        # tampering: editing a body line breaks the chain at that line; deleting a middle line breaks it at the next
        lines = led.file.read_text().splitlines()
        edited = [json.loads(x) for x in lines]; edited[1]["n_obs"] = 61
        led.file.write_text("\n".join(json.dumps(e, sort_keys=True) for e in edited) + "\n")
        assert led.verify() == {"ok": False, "first_bad_line": 2}
        led.file.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
        assert led.verify() == {"ok": False, "first_bad_line": 2}
        led.file.write_text("\n".join(lines) + "\n")
        dsr = led.deflated_sharpe("fam")
        assert dsr["verify"]["ok"] and dsr["best_name"] in names and dsr["trials"] == 3
    print("XV6 the ledger: hash length, Sharpe of a record, tamper detection at the right line, returns matrix  PASS")


def test_scenarios():
    p, _ = make_panel(n_days=800, n_stocks=60, seed=6)
    out = inject_delistings(p, annual_rate=0.10, seed=3)
    new = out.delist_after.to_numpy().sum()
    years = [y for y in sorted(set(p.dates.year)) if (p.dates.year == y).sum() >= 100]
    assert new == sum(int(round(0.10 * out.close.loc[p.dates[p.dates.year == y][0]].notna().sum())) for y in years), new
    for j, name in enumerate(p.tickers):
        flagged = np.flatnonzero(out.delist_after[name].to_numpy())
        for f in flagged:
            assert out.close[name].iloc[f] == p.close[name].iloc[f] or np.isnan(p.close[name].iloc[f])
            assert out.close[name].iloc[f + 1:].isna().all() and not out.eligible[name].iloc[f + 1:].any() and out.volume[name].iloc[f + 1:].isna().all()
            first = p.dates[p.dates.year == p.dates[f].year][0]
            assert (f - int(np.flatnonzero(p.dates == first)[0])) >= 60 // 3                      # not in the first weeks of the year
    assert int(inject_delistings(p, annual_rate=0.10, seed=3, min_age_days=90).delist_after.to_numpy().sum()) == new
    _raises(lambda: inject_delistings(p, hazard="x"), "hazard")
    _raises(lambda: inject_delistings(p, annual_rate=1.5), "annual_rate")
    _raises(lambda: inject_delistings(dataclasses.replace(p, volume=None), hazard="illiquid"), "volume")
    short = inject_delistings(make_panel(n_days=90, n_stocks=60, seed=6)[0], annual_rate=0.5)
    assert short.delist_after.to_numpy().sum() == 0                                           # a year with under 100 bars is skipped
    # a calendar year with exactly 100 bars is used; one with 99 is not (2021 here has 100 or 99 business days)
    ix = pd.bdate_range("2021-01-01", periods=100)
    pc = pd.DataFrame(100.0, index=ix, columns=[f"S{i:02d}" for i in range(60)])
    base100 = q.Panel(close=pc, eligible=pd.DataFrame(True, index=ix, columns=pc.columns), volume=pc * 10, market="T", entry_lag=1)
    assert inject_delistings(base100, annual_rate=0.5).delist_after.to_numpy().sum() == 30 and inject_delistings(dataclasses.replace(base100, close=pc.iloc[:99], eligible=base100.eligible.iloc[:99], volume=base100.volume.iloc[:99]), annual_rate=0.5).delist_after.to_numpy().sum() == 0
    # one delisting is a delisting (rate x names rounds to 1), and the earliest allowed date follows min_age_days
    assert inject_delistings(base100, annual_rate=0.02).delist_after.to_numpy().sum() == 1
    all_out = inject_delistings(base100, annual_rate=1.0, min_age_days=60)
    assert np.flatnonzero(all_out.delist_after.to_numpy().any(axis=1)).min() >= 20 and all_out.delist_after.to_numpy().sum() == 60
    # hazard weights
    names = p.tickers[:30]
    start = p.dates[300]
    assert np.allclose(_weights(p, start, names, "uniform"), 1.0) and np.allclose(_weights(p, p.dates[0], names, "volatile"), 1.0)
    vv = p.close.pct_change(fill_method=None).loc[p.dates < start].tail(60).std().reindex(names)
    assert np.allclose(_weights(p, start, names, "volatile"), 0.2 + vv.rank(pct=True).fillna(0.5).to_numpy() ** 2)
    adv = p.adv(30).loc[p.dates < start].tail(1).iloc[0].reindex(names)
    assert np.allclose(_weights(p, start, names, "illiquid"), 0.2 + (-adv).rank(pct=True).fillna(0.5).to_numpy() ** 2)
    assert _weights(p, start, names, "illiquid")[int(np.argmin(adv.to_numpy()))] > _weights(p, start, names, "illiquid")[int(np.argmax(adv.to_numpy()))]
    # the scenario table
    calls = []
    def ev(panel, dr):
        calls.append(dr); return float(panel.close.iloc[-1].sum()) * 0.01 + dr
    tbl = survivorship_scenarios(p, ev, rates=(0.05,), delist_returns=(-0.3,), hazards=("uniform",), n=12, seed=1)
    vals = [float(inject_delistings(p, annual_rate=0.05, hazard="uniform", seed=1 + i).close.iloc[-1].sum()) * 0.01 - 0.3 for i in range(12)]
    row = tbl.iloc[0]
    assert abs(row["baseline"] - float(p.close.iloc[-1].sum()) * 0.01) < 1e-12 and len(set(np.round(vals, 6))) > 3 and abs(row["mean"] - np.mean(vals)) < 1e-12 and abs(row["p5"] - np.percentile(vals, 5)) < 1e-12
    assert abs(row["p95"] - np.percentile(vals, 95)) < 1e-12 and abs(row["mean_change"] - (np.mean(vals) - row["baseline"])) < 1e-12 and len(calls) == 13
    assert len(survivorship_scenarios(p, ev, rates=(0.05,), delist_returns=(-0.3,), hazards=("uniform",), seed=1).index) == 1 and len(calls) == 13 + 1 + 10
    # survivors only: a name whose last price is exactly five days before the end is kept, six days before is not
    dd = pd.date_range("2020-01-01", periods=60, freq="D"); nm = [f"S{i:02d}" for i in range(12)]
    cl = pd.DataFrame(100.0, index=dd, columns=nm)
    cl.loc[dd[-1] - pd.Timedelta(days=5) + pd.Timedelta(days=1):, "S00"] = np.nan
    cl.loc[dd[-1] - pd.Timedelta(days=6) + pd.Timedelta(days=1):, "S01"] = np.nan
    so = survivors_only(q.Panel(close=cl, eligible=pd.DataFrame(True, index=dd, columns=nm), volume=cl * 10, market="T", entry_lag=1, periods_per_year=365))
    assert "S00" in so.tickers and "S01" not in so.tickers and len(so.tickers) == 11 and so.volume.shape[1] == 11
    print("XV7 injected delistings, hazard weights, the scenario table and survivors-only equal their definitions  PASS")


def test_export_and_bans():
    class Odd: pass
    assert jsonable(Odd()).startswith("<") and jsonable({1, 2}) == [1, 2]
    day = pd.bdate_range("2020-01-01", periods=3)
    assert iso_labels(day) == ["2020-01-01", "2020-01-02", "2020-01-03"]
    intraday = pd.date_range("2020-01-01 00:00", periods=3, freq="1h")
    assert iso_labels(intraday)[1] == "2020-01-01T01:00:00"
    assert iso_labels(pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-02 00:00:01"]))[0] == "2020-01-01T00:00:00"
    with tempfile.TemporaryDirectory(prefix="xv8-") as d:
        txt = dump_json({"a": 1, "b": "한글"})
        assert txt.splitlines()[1].startswith("  ") and not txt.splitlines()[1].startswith("   ") and "한글" in txt
        assert dump_json({"a": 1}, indent=None) == '{"a": 1}'
        path = dump_json({"a": 2}, Path(d) / "x.json")
        assert isinstance(path, Path) and json.loads(path.read_text(encoding="utf-8")) == {"a": 2}
        try:
            dump_json({"a": float("nan")})
        except ValueError:
            pass
        else:
            raise AssertionError("a NaN must not reach the JSON text")
    # bans
    dates = pd.bdate_range("2020-01-01", periods=10); tick = ["A", "B", "C"]
    one = shortable_from_bans(dates, tick, [(dates[3], dates[3])])
    assert not one.iloc[3].any() and one.iloc[2].all() and one.iloc[4].all()
    _raises(lambda: shortable_from_bans(dates, tick, [(dates[4], dates[3])]), "ends before it starts")
    _raises(lambda: shortable_from_bans(dates, tick, [(dates[1],)]), "(start, end) pair")
    ex = shortable_from_bans(dates, tick, [(dates[3], None)], exempt=["B"])
    assert ex.iloc[5, 1] and not ex.iloc[5, 0] and ex.iloc[2].all()
    frame = pd.DataFrame(True, index=dates[:6], columns=["A"])                                   # dates and tickers missing from the frame are not exempt
    fx = shortable_from_bans(dates, tick, [(dates[3], None)], exempt=frame)
    assert fx.iloc[4, 0] and not fx.iloc[7, 0] and not fx.iloc[4, 1]
    print("XV8/9 text fallback, intraday labels, the JSON writer, one-day bans and exemptions  PASS")


if __name__ == "__main__":
    for fn in (test_tradability, test_episodes, test_realize, test_costs, test_crypto_costs, test_ledger, test_scenarios, test_export_and_bans):
        fn()
    print("execution, costs, ledger, scenarios, export and ban value tests: all passed")
