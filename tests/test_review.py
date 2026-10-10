"""`review_portfolio`, `review_event`, `event_weights` and the review pages.

RV1 portfolio    every figure of the review equals the direct call it wraps; a planted factor reaches the smallest permutation p, noise does not; same seed, same page
RV2 grid         the settings table, the plateau, the walk-forward and the deflated Sharpe come from the same grid; bad arguments raise
RV3 event        the horizon table equals the engine's, the cost table is the win rate at each cost, the yearly trades add up, the permutation equals the direct call
RV4 weights      `event_weights` equals a hand-computed small case; held through `backtest_weights` the cost convention matches the event's round trip
RV6 fixes        a short sample does not break the event page, delisting reaches every table, the random-pick rate falls with the cost, signs decide the marks, one-parameter grids read right, the streak ignores column order
RV5 pages        both pages carry their sections, escape what they print, make no request, run no chart script on a chart without data, and parse as JSON
"""
from __future__ import annotations
import json, re, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import robustness as rb
from pitbacktest.review import _longest_run, event_weights, review_event, review_portfolio
from test_argument_checks import _raises
from test_synthetic import make_panel


def _q(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _world(n_days=900, n_stocks=60, rho=0.05, seed=0):
    p, rng = make_panel(n_days=n_days, n_stocks=n_stocks, seed=seed)
    z = (p.close.pct_change().shift(-2).rank(axis=1, pct=True) - .5) * np.sqrt(12)
    noise = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    return p, rng, (rho * z + noise).where(p.eligible), noise.where(p.eligible), z


def test_portfolio_review():
    p, rng, planted, noise, _ = _world()
    kw = dict(long_q=0.2, short_q=0.2, hold=1, spread_bp=5.0)
    rv = _q(lambda: review_portfolio(p, planted, n_permutations=30, seed=1, **kw))
    res = _q(lambda: q.backtest_portfolio(p, planted, **kw))
    assert np.array_equal(rv.result.net_returns.to_numpy(), res.net_returns.to_numpy())
    ref = rb.mean_tests(res.net_returns, periods_per_year=p.periods_per_year)
    assert rv.mean == ref and rv.subperiods["halves"] == rb.subperiods(res.net_returns)["halves"]
    mk = rb.market_relative(res.net_returns, res.benchmark_returns)
    assert rv.market == mk
    gross = _q(lambda: q.backtest_portfolio(p, planted, long_q=0.2, short_q=0.2, hold=1, spread_bp=0.0, benchmark=None, grid=False, funding=False)).metrics["Sharpe"]
    assert abs(rv.permutation["p"] - 1 / 31) < 1e-12 and rv.permutation["real_sharpe"] == gross and gross > res.metrics["Sharpe"]       # the permutation compares before costs
    assert rv.decomposition is not None and set(rv.decomposition["factors"]) >= {"market", "size"}
    assert list(rv.cost_table.index) == [0.0, 5.0, 10.0, 20.0, 40.0] and rv.cost_table.loc[5.0, "sharpe"] == res.metrics["Sharpe"]
    assert rv.cost_table["sharpe"].is_monotonic_decreasing                                                 # costs only subtract
    rn = _q(lambda: review_portfolio(p, noise, n_permutations=30, seed=1, **kw))
    assert rn.permutation["p"] > 0.05, rn.permutation["p"]
    a, b = rv.report(), _q(lambda: review_portfolio(p, planted, n_permutations=30, seed=1, **kw)).report()
    assert a == b                                                                                          # same seed, same page, byte for byte
    d = json.loads(rv.to_json(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert d["schema"] == "pitbacktest/portfolio-review" and d["permutation"]["p"] == rv.permutation["p"] and "null" not in d["permutation"]
    print(f"RV1 the review's figures equal the direct calls; the planted factor has p = {rv.permutation['p']:.3f} (the smallest possible), noise {rn.permutation['p']:.2f}; same seed gives the same page  PASS")


def test_grid():
    p, rng, planted, _, _ = _world()
    mk = lambda panel, lookback: -panel.close.pct_change(lookback)                                         # noqa: E731
    grid = {"lookback": [2, 3, 5, 10], "hold": [1, 5, 10]}
    rv = _q(lambda: review_portfolio(p, mk(p, 5), make_factor=mk, grid=grid, n_permutations=0, long_q=0.2, short_q=0.2, hold=5, spread_bp=5.0))
    t = rv.grid_table
    assert t.shape == (4, 3) and list(t.index) == [2, 3, 5, 10] and list(t.columns) == [1, 5, 10]
    ref = _q(lambda: q.backtest_portfolio(p, mk(p, 3), long_q=0.2, short_q=0.2, hold=10, spread_bp=5.0, benchmark=None, grid=False)).metrics["Sharpe"]
    assert abs(t.loc[3, 10] - ref) < 1e-12                                                                 # each cell is that backtest
    assert rv.plateau["best"] == np.nanmax(t.to_numpy()) and rv.plateau["best"] < 0 and rv.plateau["verdict"] == "undetermined"          # reversal on a random walk: no setting earns anything
    assert isinstance(rv.plateau["best_at"][0], int) and isinstance(rv.plateau["best_at"][1], int)         # plain Python numbers in what is printed
    assert rv.deflated["trials"] == 12 and rv.deflated["source"] == "grid" and rv.walk_forward["n_folds"] >= 3
    R = pd.DataFrame({f"lookback={a}, hold={b}": _q(lambda: q.backtest_portfolio(p, mk(p, a), long_q=0.2, short_q=0.2, hold=b, spread_bp=5.0, benchmark=None, grid=False)).net_returns
                      for a in grid["lookback"] for b in grid["hold"]})
    wf = rb.walk_forward(R, train=rv.walk_forward["folds"].shape[0] and int(0.4 * len(R)), test=int(0.1 * len(R)), embargo=5 + p.entry_lag)
    assert np.array_equal(wf["oos"].to_numpy(), rv.walk_forward["oos"].to_numpy())                         # the same walk-forward the library runs by hand
    pg = rv.report()
    assert "Is the setting a spike?" in pg and "Walk-forward" in pg and "np.int64" not in pg and "no setting has a positive Sharpe ratio" in pg
    # a grid with an edge in it: the signal's strength is the parameter, so the Sharpe rises with it and the best setting has strong neighbours
    _, rng2, _, noise2, z2 = _world()
    mk2 = lambda panel, rho: (rho * z2 + noise2).where(panel.eligible)                                       # noqa: E731
    rv2 = _q(lambda: review_portfolio(p, mk2(p, 0.08), make_factor=mk2, grid={"rho": [0.02, 0.04, 0.06, 0.08], "hold": [1, 2, 3]}, n_permutations=0, long_q=0.2, short_q=0.2, hold=1, spread_bp=5.0))
    assert rv2.plateau["best_at"][0] == 0.08 and rv2.plateau["best"] > 0 and rv2.plateau["verdict"] in ("plateau", "hill") and rv2.plateau["grid_positive"] > 0.8, rv2.plateau
    assert "a plateau" in rv2.report() or "a hill" in rv2.report()
    _raises(lambda: review_portfolio(p, planted, grid=grid), "go together")
    _raises(lambda: review_portfolio(p, planted, make_factor=mk), "go together")
    _raises(lambda: review_portfolio(p, planted, n_permutations=0, speed=3), "unknown argument")
    print("RV2 the settings table cells equal their own backtests, the walk-forward equals a hand run on the same matrix, 12 settings are 12 trials, bad arguments raise  PASS")


def test_event_review():
    p, rng, _, _, z = _world(n_days=900, n_stocks=60)
    sig = ((z + pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)) > 2.0) & p.eligible
    ev = _q(lambda: review_event(p, sig, horizons=(3, 5), cost_bp=10, n_permutations=60, seed=2))
    res = _q(lambda: q.backtest_event(p, sig, horizons=(3, 5), cost_bp=10))
    assert ev.horizon != 1                                                                                    # so that "always horizon 1" would be a different answer
    for h in (3, 5):
        assert ev.horizons_table.loc[h, "win_rate"] == res.per_horizon[h]["win_rate"] and ev.horizons_table.loc[h, "n_trades"] == res.per_horizon[h]["n_trades"]
        assert ev.horizons_table.loc[h, "win_lo"] < ev.horizons_table.loc[h, "win_rate"] < ev.horizons_table.loc[h, "win_hi"]
    h = ev.horizon
    assert h == res.spec["best_horizon"]
    row = ev.cost_table.loc[10.0]
    assert abs(row["win_rate"] - res.per_horizon[h]["win_rate"]) < 1e-9 and abs(row["mean_bp"] - res.per_horizon[h]["mean_bp"]) < 1e-9       # the table at the applied cost is the engine's figure
    assert ev.cost_table["win_rate"].is_monotonic_decreasing and ev.cost_table["mean_bp"].is_monotonic_decreasing
    assert int(ev.yearly["n_trades"].sum()) == res.per_horizon[h]["n_trades"]
    direct = rb.signal_permutation(p, sig, h, 10.0, n=60, seed=2)
    assert ev.permutation == direct
    assert ev.portfolio is not None and ev.portfolio_market is not None
    assert any("best of 2 horizons" in n for n in ev.notes)
    # the segment tables equal a plain loop: the third of the eligible securities (by 30-bar traded value) each fire belongs to, on its own date
    fwd = p.forward(h).to_numpy(np.float64)
    adv = p.adv(30)
    fire = sig.to_numpy(bool) & p.eligible.to_numpy(bool) & np.isfinite(fwd)
    rk = adv.where(p.eligible).rank(axis=1, pct=True).to_numpy(np.float64)
    wins, tot = np.zeros(3), np.zeros(3)
    for i, j in zip(*np.nonzero(fire)):
        if np.isfinite(rk[i, j]):
            b = min(2, max(0, int(np.ceil(rk[i, j] * 3)) - 1))
            tot[b] += 1
            wins[b] += (fwd[i, j] - 10 / 1e4) > 0
    sg = ev.segments["liquidity (30-bar average traded value)"]
    for k, lab in enumerate(("lowest third", "middle third", "highest third")):
        if tot[k] >= 20:
            assert sg.loc[lab, "n_trades"] == tot[k] and abs(sg.loc[lab, "win_rate"] - wins[k] / tot[k] * 100) < 1e-9, (lab, sg.loc[lab].to_dict(), wins[k], tot[k])
    assert "market capitalisation" in ev.segments and "price level" not in ev.segments            # no raw prices in this panel: nothing to split by price level
    assert _longest_run([True, True, False, True, True, True, False]) == 3 and _longest_run([]) == 0 and _longest_run([False]) == 0 and _longest_run([True] * 5) == 5
    d = json.loads(ev.to_json(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert d["schema"] == "pitbacktest/event-review" and d["horizon"] == h
    print(f"RV3 the horizon table and the cost table equal the engine's figures, yearly trades add up to {res.per_horizon[h]['n_trades']}, the permutation equals the direct call; the best of 2 horizons is flagged  PASS")


def test_event_weights():
    p, _ = make_panel(n_days=200, n_stocks=15)
    sig = pd.DataFrame(False, index=p.dates, columns=p.tickers)
    sig.iloc[100, [1, 2]] = True                                  # two names on date 100
    sig.iloc[101, [3]] = True                                     # one name the next date
    W = event_weights(p, sig, 3)
    lot = 1 / 3
    assert np.allclose(W.iloc[100, [1, 2]], lot / 2) and W.iloc[100, 3] == 0.0                                  # day 100: two names share 1/3
    assert np.allclose(W.iloc[101, [1, 2]], lot / 2) and abs(W.iloc[101, 3] - lot) < 1e-15                      # day 101: the first lot is still held, the second starts
    assert np.allclose(W.iloc[102, [1, 2]], lot / 2) and abs(W.iloc[102, 3] - lot) < 1e-15
    assert np.allclose(W.iloc[103, [1, 2]], 0.0) and abs(W.iloc[103, 3] - lot) < 1e-15                          # lot 1 ends after 3 bars
    assert W.iloc[104].abs().sum() == 0.0 and W.iloc[:100].abs().sum().sum() == 0.0
    assert abs(W.iloc[101].sum() - 2 * lot) < 1e-15
    one = pd.DataFrame(False, index=p.dates, columns=p.tickers); one.iloc[100, [1, 2]] = True                  # a single fire date: the total cost is the lot's round trip
    W1 = event_weights(p, one, 3)
    n0 = _q(lambda: q.backtest_weights(p, W1, spread_bp=0.0, benchmark=None, check_universe=False)).net_returns
    n10 = _q(lambda: q.backtest_weights(p, W1, spread_bp=10.0, benchmark=None, check_universe=False)).net_returns
    assert abs((n0 - n10).sum() - (1 / 3) * 10.0 / 1e4) < 1e-15, (n0 - n10).sum()                         # 1/3 of the capital, 10 bp in and out together
    _raises(lambda: event_weights(p, sig, 0), "at least 1")
    _raises(lambda: event_weights(p, sig.rename(columns=lambda c: "x" + c), 3), "in common")
    print("RV4 event_weights equals a hand-computed case (two names share 1/3, the next date's lot starts, a lot ends after 3 bars); bad input is refused  PASS")


def test_review_fixes():
    from dataclasses import replace
    from pitbacktest.event import _daily_excess
    from test_delisting_and_guards import _with_dead
    # a short sample: the event page used to fail on a missing 'vol'
    p, rng, _, _, z = _world(n_days=110, n_stocks=60)
    sg = (pd.DataFrame(rng.random(p.close.shape) < 0.05, index=p.dates, columns=p.tickers)) & p.eligible
    short = _q(lambda: review_event(p, sg, horizons=(5,), cost_bp=10, n_permutations=20, neutralize_check=False))
    assert "n/a" in short.report() and "Win rate" in short.report()
    # delisting reaches every table of the page, not only the engine's figures
    pd_, dead, first, spacing = _with_dead(n_dead=10, n_days=900, n_stocks=60, first=250, spacing=15)
    allfire = pd.DataFrame(True, index=pd_.dates, columns=pd_.tickers) & pd_.eligible
    ev = _q(lambda: review_event(pd_, allfire, horizons=(10,), cost_bp=10, n_permutations=20, delist_return=-0.9, neutralize_check=False))
    eng = ev.result.per_horizon[10]
    assert abs(ev.cost_table.loc[10.0, "win_rate"] - eng["win_rate"]) < 1e-9 and abs(ev.cost_table.loc[10.0, "mean_bp"] - eng["mean_bp"]) < 1e-9
    assert abs(ev.cost_table.loc[10.0, "base_rate"] - eng["base_rate"]) < 1e-9 and abs(ev.cost_table.loc[10.0, "lift_pp"] - eng["lift_pp"]) < 1e-9       # the random-pick rate is the engine's too
    assert ev.permutation == rb.signal_permutation(pd_, allfire, 10, 10.0, n=20, seed=0, delist_return=-0.9)
    plain = _q(lambda: review_event(pd_, allfire, horizons=(10,), cost_bp=10, n_permutations=20, neutralize_check=False))
    assert plain.cost_table.loc[10.0, "mean_bp"] > ev.cost_table.loc[10.0, "mean_bp"] + 5                                                        # a -90 percent delisting return is felt
    # the random-pick rate falls as the cost rises, and the lift column is the difference
    p2, rng2, _, _, z2 = _world()
    sig = ((z2 + pd.DataFrame(rng2.standard_normal(p2.close.shape), index=p2.dates, columns=p2.tickers)) > 2.0) & p2.eligible
    ev2 = _q(lambda: review_event(p2, sig, horizons=(3, 5), cost_bp=10, n_permutations=20, hold=7))
    ct = ev2.cost_table
    assert ct["base_rate"].is_monotonic_decreasing and ct["base_rate"].iloc[0] > ct["base_rate"].iloc[-1] + 1 and np.allclose(ct["lift_pp"], ct["win_rate"] - ct["base_rate"])
    assert ev2.hold == 7 and ev2.horizon != 7 and "share 1/7 of the capital for 7 bars" in ev2.report() and "Random pick" in ev2.report()
    # the streak and the day-level t do not depend on the order of the columns, and the t is the engine's own definition
    perm = np.random.default_rng(1).permutation(60)
    cols = [p2.tickers[i] for i in perm]
    p3 = q.Panel(close=p2.close[cols], eligible=p2.eligible[cols], volume=p2.volume[cols], mkt_cap=p2.mkt_cap[cols], market="TEST")
    ev3 = _q(lambda: review_event(p3, sig, horizons=(3, 5), cost_bp=10, n_permutations=20, hold=7))
    assert ev3.streak == ev2.streak and ev3.daily_excess == ev2.daily_excess, (ev3.streak, ev2.streak)
    h = ev2.horizon
    fw = p2.forward(h).to_numpy(np.float64)
    fire = sig.to_numpy(bool) & p2.eligible.to_numpy(bool) & np.isfinite(fw)
    ex = _daily_excess(fire, fw, p2.eligible.to_numpy(bool))
    m_, _, t_, _ = rb.newey_west_t(ex, max(h, 21))
    assert abs(ev2.daily_excess["t"] - t_) < 1e-12 and ev2.daily_excess["lag"] == max(h, 21) and abs(ev2.daily_excess["mean_excess_bp"] - m_ * 1e4) < 1e-9
    assert ev2.streak["unit"] == "fire-days"
    # signs decide the marks: a strategy that loses money beyond chance is not ticked
    pl, rngl, planted, _, _ = _world()
    lose = _q(lambda: review_portfolio(pl, -planted, n_permutations=0, long_q=0.2, short_q=0.2, hold=1, spread_bp=30.0, n_bootstrap=0))
    pg = lose.report()
    assert lose.mean["nw_t"] <= -2.5 and "significantly NEGATIVE" in pg and "the interval is entirely below 0" in pg and '<span class="rd">✕</span>the interval is entirely below 0' in pg and '<span class="rd">✕</span>significantly NEGATIVE' in pg and '<li class="bad">' in pg
    assert "above the 2.5 bar" not in pg.split("Is it luck?")[1].split("</table>")[0]
    # one-parameter grids: sorted axis, the parameter named, the best cell marked, a one-element best_at
    mk = lambda panel, lookback: -panel.close.pct_change(lookback)                                      # noqa: E731
    rv1 = _q(lambda: review_portfolio(pl, mk(pl, 5), make_factor=mk, grid={"lookback": [10, 2, 5, 3]}, n_permutations=0, n_bootstrap=0, long_q=0.2, short_q=0.2, hold=5, spread_bp=5.0))
    gt = rv1.grid_table
    assert isinstance(gt, pd.Series) and list(gt.index) == [2, 3, 5, 10] and gt.index.name == "lookback"
    assert len(rv1.plateau["best_at"]) == 1 and rv1.plateau["verdict"] == "undetermined" and rv1.plateau["reason"] in rv1.report()
    pg1 = rv1.report()
    assert "lookback" in pg1.split("Is the setting a spike?")[1] and 'class="best"' in pg1 and "Best setting " + str(rv1.plateau["best_at"][0]) + " (Sharpe" in pg1
    print("RV6 a short sample builds a page, delisting reaches the cost and permutation tables, the random-pick rate falls with the cost, the streak ignores column order, a loser is not ticked, a one-parameter grid reads right  PASS")


def test_return_basis():
    p, rng, planted, _, _ = _world()
    for basis, shown in (("price (adjusted for splits; dividends are not in the series)", True), ("total (adjusted close includes dividends and splits)", False),
                         ("no dividends (perpetual futures); funding is charged separately", False), (None, False)):
        pp = q.Panel(close=p.close, eligible=p.eligible, volume=p.volume, meta={} if basis is None else {"return_basis": basis}, market="T")
        r = _q(lambda: q.backtest_portfolio(pp, planted, long_q=0.2, short_q=0.2, hold=5, spread_bp=5.0))
        assert r.spec["return_basis"] == basis
        assert ("Price returns" in r.report()) == shown, basis
        w = pd.DataFrame(0.0, index=pp.dates, columns=pp.tickers); w.iloc[100:, 0] = 1.0
        assert _q(lambda: q.backtest_weights(pp, w, spread_bp=0.0, benchmark=None, check_universe=False)).spec["return_basis"] == basis
    print("RV7 the return basis travels from the panel to the result and a price-only basis is said on the page; total, no-dividend and unknown bases say nothing  PASS")


def test_new_sections():
    p, rng, planted, _, _ = _world(n_days=1100)
    idx_ret = pd.Series(np.random.default_rng(8).normal(0.0003, 0.01, len(p.dates)), index=p.dates)
    groups = pd.Series({t: ("tech" if i % 3 == 0 else "bank") for i, t in enumerate(p.tickers)})
    rv = _q(lambda: review_portfolio(p, planted, n_permutations=0, long_q=0.2, short_q=0.2, hold=1, spread_bp=5.0, benchmark_returns=idx_ret, split_date=p.dates[700], groups=groups, n_bootstrap=200))
    # the supplied benchmark replaces the panel's: the result carries it, on the signal dates
    assert rv.result.benchmark_returns.notna().sum() > 1000
    al = rb.align_benchmark(p, idx_ret).reindex(rv.result.net_returns.index)
    assert np.allclose(rv.result.benchmark_returns.dropna(), al.dropna(), atol=1e-15) and rv.market == rb.market_relative(rv.result.net_returns, al)
    assert rv.risk["n"] == len(rv.result.net_returns) and rv.drawdown_dist["n"] == 200 and rv.split["split"] == str(p.dates[700].date()) and rv.regimes is not None
    assert rv.holdings["avg_long"] > 0 and rv.brinson is None and any("Brinson attribution not computed" in n and "short" in n for n in rv.notes)         # a long-short book has no Brinson table, and says why
    pg = rv.report()
    for h in ("How bad can it get?", "What it held, and how much it traded", "What happened after the frozen date?", "In which markets did it work?", "the series you supplied"):
        assert h in pg or h in " ".join(rv.notes), h
    assert "Benchmark</th><th>Relative</th>" in pg                                                                 # the year table has the benchmark and the difference
    assert pg.count("<svg") >= 4                                                                                   # equity, drawdown, rolling Sharpe and the drawdown distribution
    # a long-only book against the cap-weighted benchmark gets the Brinson table
    w = (planted.rank(axis=1, pct=True) >= 0.8).astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).where(p.eligible, 0.0) * 0.9
    lo = _q(lambda: q.backtest_weights(p, w, spread_bp=5.0, check_universe=False))
    rvl = _q(lambda: review_portfolio(p, planted, n_permutations=0, long_q=0.2, short_q=None, hold=1, spread_bp=5.0, groups=groups, n_bootstrap=0))
    assert rvl.brinson is not None and rvl.brinson["identity_gap"] < 1e-12 and "Allocation or selection?" in rvl.report()
    # a benchmark that covers little of the sample is refused
    _raises(lambda: review_portfolio(p, planted, n_permutations=0, benchmark_returns=idx_ret.iloc[:50], long_q=0.2, short_q=0.2, hold=1, spread_bp=5.0), "at least 60")
    _raises(lambda: review_portfolio(p, planted, n_permutations=0, split_date=p.dates[5], long_q=0.2, short_q=0.2, hold=1, spread_bp=5.0), "at least 30")
    assert lo is not None
    print("RV8 an own benchmark is moved onto the signal dates, a frozen date, the risk, regime and holding sections, the year table with the benchmark and the Brinson table (long-only) are on the page  PASS")


def test_pages():
    p, rng, planted, _, z = _world()
    rv = _q(lambda: review_portfolio(p, planted, n_permutations=20, long_q=0.2, short_q=0.2, hold=1, spread_bp=5.0))
    sig = ((z + pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)) > 2.0) & p.eligible
    ev = _q(lambda: review_event(p, sig, horizons=(1, 5), cost_bp=10, n_permutations=40))
    for name, page in (("portfolio", rv.report(title='<img src=x onerror=1> "t"')), ("event", ev.report(title='<img src=x onerror=1> "t"'))):
        assert "<img src=x" not in page and "&lt;img src=x onerror=1&gt;" in page, name
        rest = (rv if name == "portfolio" else ev).report().replace("http://www.w3.org/2000/svg", "")          # an ordinary title: the hostile one above contains the text "src="
        for bad in ("http", "src=", "href=", "@import", "url(", "fetch(", "XMLHttpRequest"):
            assert bad not in rest, (name, bad)
        assert "var sc=fig.querySelector('script.cd');if(!sc)return;" in page                                  # bar and histogram figures have no hover data
        assert page.count("<script") == page.count("</script>")
    assert all(h in rv.report() for h in ("Is it luck?", "Against the market", "What explains the return?", "Does it hold over time?", "How much cost can it take?"))
    assert all(h in ev.report() for h in ("By holding period", "Is it luck?", "Year by year", "The same signal held as a portfolio", "The library's six gates", "What the signal picks"))
    assert "Against relabelled factors" in rv.report() and "Against the signal moved in time" in ev.report()
    assert "Where did the return come from?" in rv.report() and "From gross to net" in rv.report() and "Where does it work?" in ev.report() and "By liquidity" in rv.report()
    a = rv.attribution
    assert abs(a["long_annual"] + a["short_annual"] - a["gross_annual"]) < 1e-12 and abs(a["waterfall"]["residual_bp"]) < 1e-6
    out = ev.report(Path(__import__("tempfile").mkdtemp(prefix="rv-")) / "e.html")
    assert Path(out).read_text(encoding="utf-8") == ev.report()
    print("RV5 both pages carry their sections, escape titles, fetch nothing, skip the hover script on bar and histogram figures, and write a file  PASS")


if __name__ == "__main__":
    test_portfolio_review()
    test_grid()
    test_event_review()
    test_event_weights()
    test_review_fixes()
    test_return_basis()
    test_new_sections()
    test_pages()
    print("review tests: all passed")
