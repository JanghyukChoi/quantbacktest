"""A second automatic mutation run, on a fresh sample after the first round of fixes, still found these behaviours unpinned. One test each.

SR1 portfolio  short_q above 1 is refused with its own message; yearly figures chain the bars; the average number of positions skips empty bars; the ledger records cap_gross and the freeze;
               with an execution mask that blocks nothing, every execution statistic is zero
SR2 weights    a delisting counts only for a name that is held; the blocked share of a tiny book is 1
SR3 event      zero-return trades are neither wins nor losses; a signal that covers part of the dates and tickers is accepted
SR4 review     defaults (100 relabellings, the 20-bar test window), extra factors, no cost table, PBO at 128 bars, the note for several horizons only
SR5 panel      unknown delisting flags mean not delisted; adv() defaults to 20 bars; assert_causal's default seed; universe growth; audit edges; penny prices are eligible; ADV window of 20
SR6 analytics  relative constant-column rule, lag 0, the p-value, unknown flags in forward returns and IC, the bootstrap indices, the 30-bar floor, chunks
"""
from __future__ import annotations
import dataclasses, math, sys, tempfile, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import analytics as an
from pitbacktest import robustness as rb
from pitbacktest.core.panel import build_pit_eligible
from test_argument_checks import _raises
from test_synthetic import make_panel


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _factor(p, seed=1):
    return pd.DataFrame(np.random.default_rng(seed).standard_normal(p.close.shape), index=p.dates, columns=p.tickers)


def test_portfolio():
    p, _ = make_panel(n_days=300, n_stocks=24, seed=5)
    f = _factor(p)
    _raises(lambda: q.backtest_portfolio(p, f, long_q=0.1, short_q=1.5), "short_q must be in (0, 1]")
    r = _quiet(lambda: q.backtest_portfolio(p, f, long_q=0.2, short_q=None, benchmark=None, grid=False))
    for y, v in r.yearly.items():
        g = r.net_returns[r.net_returns.index.year == int(y)]
        assert abs(v - ((1 + g).prod() - 1)) < 1e-12
    npos = (np.asarray(r.holdings) > 0).sum(axis=1)
    assert (npos == 0).any() and abs(r.metrics["avg_positions"] - npos[npos > 0].mean()) < 1e-12        # the warm-up bars hold nothing and are not averaged in
    with tempfile.TemporaryDirectory(prefix="sr1-") as d:
        led = q.Ledger(d)
        _quiet(lambda: q.backtest_portfolio(p, f, ledger=led, family="x", name="x", cap_gross=True, freeze_days=4, freeze_return=-0.2, benchmark=None, grid=False))
        _quiet(lambda: q.backtest_portfolio(p, f, ledger=led, family="y", name="y", benchmark=None, grid=False))
        cx, cy = led.trials("x")[0]["config"], led.trials("y")[0]["config"]
    assert cx["cap_gross"] is True and cx["freeze_days"] == 4 and cx["freeze_return"] == -0.2 and "cap_gross" not in cy and "freeze_days" not in cy
    free = dataclasses.replace(p, can_buy=pd.DataFrame(True, index=p.dates, columns=p.tickers), can_sell=pd.DataFrame(True, index=p.dates, columns=p.tickers))
    m = _quiet(lambda: q.backtest_portfolio(free, f, long_q=0.2, short_q=0.2, benchmark=None, grid=False)).metrics
    assert m["blocked_trades"] == 0 and m["mean_stuck_weight"] == 0 and m["longest_freeze_days"] == 0, m
    assert m.get("cap_infeasible_days", 0) == 0 and m.get("mean_free_scale", 1.0) == 1.0, m
    print("SR1 portfolio: short_q message, yearly chain, positions average, ledger cap_gross and freeze, zero execution statistics when nothing is blocked  PASS")


def test_weights():
    p, _ = make_panel(n_days=260, n_stocks=24, seed=6)
    close = p.close.copy()
    da = pd.DataFrame(False, index=p.dates, columns=p.tickers)
    for j, d in ((5, 140), (6, 150)):                                                  # two names are delisted; only the first is held
        close.iloc[d + 1:, j] = np.nan
        da.iloc[d, j] = True
    pd_ = dataclasses.replace(p, close=close, delist_after=da)
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); W.iloc[70:, 5] = 0.3
    r = _quiet(lambda: q.backtest_weights(pd_, W, benchmark=None, check_universe=False))
    assert r.metrics["delist_events_held"] == 1, r.metrics["delist_events_held"]
    # a tiny book whose only trade is blocked: asked turnover under 1 and all of it blocked
    no_buy = pd.DataFrame(True, index=p.dates, columns=p.tickers); no_buy.iloc[:, 3] = False
    pb = dataclasses.replace(p, can_buy=no_buy)
    W2 = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); W2.iloc[80:81, 3] = 0.3
    r2 = _quiet(lambda: q.backtest_weights(pb, W2, benchmark=None, check_universe=False))
    assert r2.metrics["blocked_trades"] > 0 and abs(r2.metrics["blocked_turnover_share"] - 1.0) < 1e-12, r2.metrics
    print("SR2 weights: a delisting counts for a held name only; a fully blocked small book has a blocked share of 1  PASS")


def test_event():
    from pitbacktest.event import _trade_stats
    D, N = 50, 30
    cum = np.random.default_rng(1).normal(0.001, 0.03, (D, N))
    fire = np.zeros((D, N), bool); fire[:, :4] = True
    cum[:, 0] = 0.0                                                                      # 50 of the 200 trades earn exactly zero
    st = _trade_stats(fire, cum, np.ones((D, N), bool), 0.0)
    t = cum[fire]
    assert abs(st["avg_win_bp"] - t[t > 0].mean() * 1e4) < 1e-9 and abs(st["avg_loss_bp"] - t[t < 0].mean() * 1e4) < 1e-9
    assert abs(st["payoff"] - t[t > 0].mean() / abs(t[t < 0].mean())) < 1e-12 and abs(st["win_rate"] - (t > 0).mean() * 100) < 1e-12
    p, _ = make_panel(n_days=400, n_stocks=30, seed=2)
    sparse = pd.DataFrame(False, index=p.dates[100:300], columns=p.tickers[:20])         # covers only part of the dates and tickers
    sparse.iloc[:, :] = np.random.default_rng(0).random(sparse.shape) < 0.1
    res = _quiet(lambda: q.backtest_event(p, sparse, horizons=(5,), neutralize_check=False))
    assert res.per_horizon[5]["n_trades"] >= 100
    print("SR3 event: zero-return trades are neither wins nor losses; a partial signal frame is accepted  PASS")


def test_review():
    p, _ = make_panel(n_days=260, n_stocks=30, seed=7)
    f = _factor(p, 7)
    kw = dict(long_q=0.2, short_q=0.2, hold=5)
    r = _quiet(lambda: q.review_portfolio(p, f, **kw))
    assert r.permutation["n"] == 100
    extra = pd.DataFrame({"mine": np.random.default_rng(3).normal(0, 0.01, len(p.dates))}, index=p.dates)
    e = _quiet(lambda: q.review_portfolio(p, f, n_permutations=0, extra_factors=extra, style_factors=False, **kw))
    assert "mine" in e.decomposition["factors"]
    nc = _quiet(lambda: q.review_portfolio(p, f, n_permutations=0, spread_grid=None, **kw))
    assert nc.cost_table is None
    # the default test window is 20 bars even on a short sample (0.1 x T would be 19)
    ps, _ = make_panel(n_days=190, n_stocks=30, seed=7)
    fs = _factor(ps, 7)
    g = _quiet(lambda: q.review_portfolio(ps, fs, n_permutations=0, make_factor=lambda panel: fs, grid={"hold": [1, 3, 5]}, **kw))
    ts = g.walk_forward["folds"].iloc[0]
    assert int(((ps.dates >= ts["test_start"]) & (ps.dates <= ts["test_end"])).sum()) == 20
    # one horizon: no "best of several" note; several: the note
    sig = pd.DataFrame(np.random.default_rng(1).random(p.close.shape) < 0.08, index=p.dates, columns=p.tickers) & p.eligible
    one = _quiet(lambda: q.review_event(p, sig, horizons=(5,), n_permutations=0, as_portfolio=False))
    two = _quiet(lambda: q.review_event(p, sig, horizons=(3, 5), n_permutations=0, as_portfolio=False))
    assert not any("best of" in n for n in one.notes) and any("best of 2 horizons" in n for n in two.notes)
    print("SR4 review: 100 relabellings by default, extra factors, no cost table, the 20-bar test window, the several-horizons note  PASS")


def test_pbo_threshold():
    rng = np.random.default_rng(4)
    present = {}
    for n_days in range(125, 135):
        p, _ = make_panel(n_days=n_days, n_stocks=30, seed=7)
        f = _factor(p, 7)
        rows = len(pd.DataFrame({h: _quiet(lambda: q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=h, benchmark=None, grid=False)).net_returns for h in (1, 2, 3, 5)}).dropna())
        rv = _quiet(lambda: q.review_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5, n_permutations=0, style_factors=False, spread_grid=None, make_factor=lambda panel: f, grid={"hold": [1, 2, 3, 5]}))
        present[n_days] = (rows, rv.pbo is not None)
    for n_days, (rows, has) in present.items():
        assert has == (rows >= 128), (n_days, rows, has)
    assert any(r == 127 for r, _ in present.values()) and any(r == 128 for r, _ in present.values()), sorted(present.items())
    print("SR4b PBO is computed from 128 bars of 4 settings, not from 127  PASS")


def test_panel():
    p, _ = make_panel(n_days=260, n_stocks=24, seed=8)
    da = pd.DataFrame(np.nan, index=p.dates, columns=p.tickers)
    plain = p.forward(5)
    with_nan = dataclasses.replace(p, delist_after=da).forward(5)
    assert np.allclose(plain.to_numpy(), with_nan.to_numpy(), equal_nan=True)            # unknown flags mean "not delisted"
    assert p.adv().equals(p.adv(20)) and p.adv().equals((p.close * p.volume).rolling(20, min_periods=20).mean())
    big = _panel_long()
    c = big.assert_causal(lambda pn: pn.close)
    want = sorted(int(i) for i in np.random.default_rng(0).choice(np.arange(300, len(big.dates) - 1), size=6, replace=False))
    assert c["cuts"] == [str(big.dates[i].date()) for i in want]
    # universe growth: the mean eligible count of the last year over the first
    idx = pd.bdate_range("2018-01-01", periods=780); cols = [f"S{i:02d}" for i in range(30)]
    el = pd.DataFrame(False, index=idx, columns=cols)
    for yr, k in ((2018, 10), (2019, 20), (2020, 30)):
        el.loc[str(yr), cols[:k]] = True
    pp = q.Panel(close=pd.DataFrame(100.0, index=idx, columns=cols), eligible=el, market="T", entry_lag=1)
    assert abs(pp.audit()["universe_growth"] - 3.0) < 0.05
    el0 = el.copy(); el0.loc["2018", :] = False                                         # nobody eligible in the first year: the growth is unknown, not infinite
    nog = q.Panel(close=pd.DataFrame(100.0, index=idx, columns=cols), eligible=el0, market="T", entry_lag=1).audit()
    assert np.isnan(nog["universe_growth"])
    # the truncation check needs more than 60 rows
    for n_rows, key in ((60, False), (61, True)):
        ix = pd.bdate_range("2020-01-01", periods=n_rows); cl = pd.DataFrame(100.0, index=ix, columns=cols)
        pr = q.Panel(close=cl, eligible=pd.DataFrame(True, index=ix, columns=cols), market="T", entry_lag=1)
        assert ("truncation_suspected" in pr.audit()) == key, n_rows
    # "ended" means more than 30 calendar days before the end of the panel
    day = pd.date_range("2020-01-01", periods=100, freq="D")
    for stop_gap, n_ended in ((30, 0), (31, 1)):
        names = [f"N{i:02d}" for i in range(12)]
        cl = pd.DataFrame(100.0, index=day, columns=names)
        cl.loc[day[-1] - pd.Timedelta(days=stop_gap) + pd.Timedelta(days=1):, "N00"] = np.nan        # the first name's last price is `stop_gap` days before the end
        a = q.Panel(close=cl, eligible=pd.DataFrame(True, index=day, columns=names), market="T", entry_lag=1, periods_per_year=365).audit()
        assert abs(a["ended_before_end_share"] - n_ended / 12) < 1e-12, (stop_gap, a["ended_before_end_share"])
    penny = pd.DataFrame(0.5, index=idx[:30], columns=["A"])
    assert build_pit_eligible(penny)["A"].all()
    # an ADV window of 20 bars, not 21: a volume that grows by one a day
    vol = pd.DataFrame(np.arange(1, 61, dtype=float)[:, None], index=pd.bdate_range("2020-01-01", periods=60), columns=["A"])
    px = pd.DataFrame(1.0, index=vol.index, columns=["A"])
    ok = build_pit_eligible(px, volume=vol, min_adv=50.25)
    assert bool(ok["A"].iloc[-1]) and bool(ok["A"].iloc[59]), "the last 20 volumes average 50.5; the last 21 average 50"
    assert not build_pit_eligible(px, volume=vol, min_adv=50.75)["A"].iloc[-1]
    print("SR5 panel: unknown delisting flags, adv(), assert_causal's seed, universe growth, the 60-row rule, penny prices, the 20-bar ADV  PASS")


def _panel_long():
    p, _ = make_panel(n_days=420, n_stocks=12, seed=3)
    return p


def test_analytics():
    rng = np.random.default_rng(6)
    T = 300
    idx = pd.bdate_range("2019-01-01", periods=T)
    y = pd.Series(rng.normal(0, 0.01, T), index=idx)
    big = pd.DataFrame({"m": 1e6 + 1e-9 * rng.standard_normal(T)}, index=idx)                 # constant for all practical purposes, at a large level
    _raises(lambda: an.alpha_beta(y, big), "constant")
    F = pd.DataFrame({"m": rng.normal(0, 0.01, T)}, index=idx)
    r0 = an.alpha_beta(y, F, lag=0)
    assert r0["lag"] == 0
    assert abs(r0["alpha_p"] - math.erfc(abs(r0["alpha_t"]) / math.sqrt(2))) < 1e-12 and 0 < r0["alpha_p"] < 1
    # unknown delisting flags and unknown eligibility
    p, _ = make_panel(n_days=200, n_stocks=24, seed=9)
    plain = an.forward_returns(p, 5, delist_return=-0.3)
    nan_flags = an.forward_returns(dataclasses.replace(p, delist_after=pd.DataFrame(np.nan, index=p.dates, columns=p.tickers)), 5, delist_return=-0.3)
    assert np.allclose(plain.to_numpy(), nan_flags.to_numpy(), equal_nan=True)
    fac = _factor(p, 2); fwd = plain
    el = pd.DataFrame(True, index=p.dates, columns=p.tickers).astype(object); el.iloc[100, 10:] = np.nan                    # unknown eligibility: not eligible
    ic = an.information_coefficient(fac, fwd, el)
    assert np.isnan(ic.iloc[100]) and np.isfinite(ic.iloc[101])
    # the bootstrap indices: with a mean block of 1 every draw is fresh, so the indices are the raw uniform draws
    gen = np.random.default_rng(5)
    idx_ = an._stationary_indices(50, 7, 1.0, np.random.default_rng(5))
    start = gen.integers(0, 50, size=(7, 50))
    assert np.array_equal(idx_, start)
    x = pd.Series(rng.normal(0.001, 0.01, 60), index=pd.bdate_range("2020-01-01", periods=60))
    assert an.sharpe_ci(x[:30], n=100)["T"] == 30
    _raises(lambda: an.sharpe_ci(x[:29], n=100), "too few")
    # chunks of 400 draws: 401 is 400 and 1
    assert list(an._chunks(401)) == [400, 1] and list(an._chunks(800)) == [400, 400] and list(an._chunks(1)) == [1] and list(an._chunks(0)) == []
    print("SR6 analytics: relative constant rule, lag 0, the p-value, unknown flags, bootstrap indices, the 30-bar floor, chunks of 400  PASS")


if __name__ == "__main__":
    for fn in (test_portfolio, test_weights, test_event, test_review, test_pbo_threshold, test_panel, test_analytics):
        fn()
    print("second-round value tests: all passed")
