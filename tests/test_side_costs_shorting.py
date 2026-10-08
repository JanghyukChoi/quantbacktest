"""Side-specific costs (a sales tax, a commission) and short-selling restrictions.

S1 reference    the engine's buy and sell costs equal a plain loop written from the rule, for weights that change sign
S2 reconcile    with no overlapping tranches `backtest_portfolio` and `backtest_weights` agree exactly, side costs included
S3 schedule     a rate that changes on a date is read on the signal date; an unknown rate (before the first entry, NaN) raises
S4 shortable    the short leg is chosen among securities that can be sold short; with none, the strategy is long-only
S5 weights      opening or increasing a short where it cannot be done raises; holding or reducing one does not
S6 universe     turning a long into a smaller short in a name that is not eligible is now caught (it was not before)
S7 bans         `shortable_from_bans` marks ban periods, exemptions and open-ended bans; unknown securities are not shortable
S8 ledger       side costs reach the ledger configuration only when they are not zero
S10 residue     averaging overlapping tranches leaves no rounding residue that `avg_positions` would count as a position
S9 krx tax      `sell_tax_panel` gives each security the rate of its market on that day, including after a market move
"""
from __future__ import annotations
import sys, tempfile
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.adapters import krx
from test_reconcile import _panel


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"no ValueError for {needles}")


def _weights(panel, seed=1):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.05, panel.close.shape)
    W[rng.random(W.shape) < 0.5] = 0.0
    return pd.DataFrame(W, index=panel.dates, columns=panel.tickers)


def test_side_cost_matches_a_loop():
    p, _ = _panel()
    p = replace(p, eligible=p.eligible | True)
    W = _weights(p)
    buy, sell = 7.0, 23.0
    a = q.backtest_weights(p, W, benchmark=None)
    b = q.backtest_weights(p, W, buy_bp=buy, sell_bp=sell, benchmark=None)
    H = W.to_numpy()
    ref = np.zeros(len(H))
    for t in range(1, len(H)):
        for j in range(H.shape[1]):
            d = H[t, j] - H[t - 1, j]
            ref[t] += (d * buy if d > 0 else -d * sell) / 1e4
    cut = len(H) - (p.entry_lag + 1)
    got = (a.net_returns - b.net_returns).to_numpy()
    assert np.allclose(got, ref[:cut], atol=1e-13, rtol=0), np.abs(got - ref[:cut]).max()
    assert abs(b.metrics["side_cost_annual_bp"] - ref[:cut].mean() * p.periods_per_year * 1e4) < 1e-6
    assert a.metrics["side_cost_annual_bp"] == 0.0
    print(f"S1 buy and sell cost equal an independent loop (largest difference {np.abs(got - ref[:cut]).max():.1e}); side cost {b.metrics['side_cost_annual_bp']:.0f} bp a year  PASS")


def test_engines_agree_with_side_costs():
    p, f = _panel(seed=5, T=300, N=60)
    kw = dict(buy_bp=4.0, sell_bp=19.0)
    r = q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=1, spread_bp=10.0, benchmark=None, grid=False, **kw)
    w = q.backtest_weights(p, pd.DataFrame(r.holdings, index=p.dates, columns=p.tickers), spread_bp=10.0, benchmark=None, check_universe=False, **kw)
    assert np.allclose(r.net_returns.to_numpy(), w.net_returns.to_numpy(), atol=1e-13, rtol=0), np.abs(r.net_returns.to_numpy() - w.net_returns.to_numpy()).max()
    assert abs(r.metrics["side_cost_annual_bp"] - w.metrics["side_cost_annual_bp"]) < 1e-6
    z = q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=1, spread_bp=10.0, benchmark=None, grid=False)
    assert r.metrics["cost_annual_bp"] > z.metrics["cost_annual_bp"] + 1.0                     # cost_annual_bp holds the side cost too
    assert abs(r.metrics["cost_annual_bp"] - z.metrics["cost_annual_bp"] - r.metrics["side_cost_annual_bp"]) < 1e-6
    # With constant gross exposure a leg buys as much as it sells every day, so swapping the two rates changes nothing. Make the legs empty and
    # full again (no eligible names for a stretch; no name that can be sold short for another) so that the side of each trade matters.
    el = p.eligible.copy(); el.iloc[200:230] = False
    ok = pd.DataFrame(True, index=p.dates, columns=p.tickers); ok.iloc[100:160] = False
    g = replace(p, eligible=el, shortable=ok)
    rg = q.backtest_portfolio(g, f, long_q=0.2, short_q=0.2, hold=1, spread_bp=10.0, benchmark=None, grid=False, **kw)
    wg = q.backtest_weights(g, pd.DataFrame(rg.holdings, index=p.dates, columns=p.tickers), spread_bp=10.0, benchmark=None, check_universe=False, **kw)
    diff = np.abs(rg.net_returns.to_numpy() - wg.net_returns.to_numpy()).max()
    assert diff < 1e-13, diff
    swapped = q.backtest_weights(g, pd.DataFrame(rg.holdings, index=p.dates, columns=p.tickers), spread_bp=10.0, benchmark=None, check_universe=False, buy_bp=19.0, sell_bp=4.0)
    assert np.abs(rg.net_returns.to_numpy() - swapped.net_returns.to_numpy()).max() > 1e-6          # the side matters in this book
    print(f"S2 the two engines agree to {np.abs(r.net_returns.to_numpy() - w.net_returns.to_numpy()).max():.1e} with buy and sell costs, also when the legs empty and refill; cost_annual_bp includes them  PASS")


def test_schedule_and_unknown_rates():
    p, _ = _panel()
    p = replace(p, eligible=p.eligible | True)
    W = _weights(p, seed=2)
    switch = p.dates[120]
    sched = pd.Series([10.0, 40.0], index=[p.dates[0], switch])
    a = q.backtest_weights(p, W, benchmark=None)
    b = q.backtest_weights(p, W, sell_bp=sched, benchmark=None)
    H = W.to_numpy(); ref = np.zeros(len(H))
    for t in range(1, len(H)):
        rate = 40.0 if p.dates[t] >= switch else 10.0
        ref[t] = sum(max(-(H[t, j] - H[t - 1, j]), 0.0) for j in range(H.shape[1])) * rate / 1e4
    cut = len(H) - 2
    assert np.allclose((a.net_returns - b.net_returns).to_numpy(), ref[:cut], atol=1e-13, rtol=0)
    late = pd.Series([10.0], index=[p.dates[50]])
    _raises(lambda: q.backtest_weights(p, W, sell_bp=late, benchmark=None), "sell_bp", "unknown")
    nanp = pd.DataFrame(1.0, index=p.dates, columns=p.tickers); nanp.iloc[3, 3] = np.nan
    _raises(lambda: q.backtest_weights(p, W, buy_bp=nanp, benchmark=None), "buy_bp", "unknown")
    _raises(lambda: q.backtest_weights(p, W, buy_bp=-1.0, benchmark=None), "buy_bp")
    _raises(lambda: q.backtest_weights(p, W, sell_bp=float("nan"), benchmark=None), "sell_bp")
    _raises(lambda: q.backtest_weights(p, W, sell_bp=np.ones((3, 3)), benchmark=None), "shape")
    _raises(lambda: q.backtest_portfolio(p, _weights(p), sell_bp=pd.Series([1.0], index=[0]), benchmark=None), "indexed by date")
    print("S3 a rate that changes on a date is read per date and matches a loop; an unknown rate, a negative one and a wrong shape raise  PASS")


def test_shortable_in_portfolio():
    p, f = _panel(seed=7, T=300, N=60)
    p = replace(p, eligible=p.eligible | True)
    kw = dict(long_q=0.2, short_q=0.2, hold=1, spread_bp=0.0, benchmark=None, grid=False)
    base = q.backtest_portfolio(p, f, **kw)
    allsh = q.backtest_portfolio(replace(p, shortable=pd.DataFrame(True, index=p.dates, columns=p.tickers)), f, **kw)
    assert np.array_equal(base.net_returns.to_numpy(), allsh.net_returns.to_numpy())          # nothing restricted: bit for bit the same
    ok = np.zeros(p.close.shape, bool); ok[:, ::2] = True
    half = replace(p, shortable=pd.DataFrame(ok, index=p.dates, columns=p.tickers))
    r = q.backtest_portfolio(half, f, **kw)
    H = r.holdings
    assert (H[H < 0].size > 0) and not (H[:, ~ok[0]] < 0).any(), "a short sits on a name that cannot be sold short"
    # reference for one day: the short leg is the bottom q of the shortable, eligible names
    t = 100
    fv = f.to_numpy()[t]; cand = np.where(ok[t])[0]
    order = cand[np.argsort(fv[cand])]
    n = len(cand); rank = (np.arange(n) + 1) / n
    expect = set(order[rank <= 0.2])
    assert set(np.where(H[t] < 0)[0]) == expect, (set(np.where(H[t] < 0)[0]), expect)
    none = replace(p, shortable=pd.DataFrame(False, index=p.dates, columns=p.tickers))
    e = q.backtest_portfolio(none, f, **kw)
    lo = q.backtest_portfolio(p, f, **{**kw, "short_q": None})
    assert np.array_equal(e.net_returns.to_numpy(), lo.net_returns.to_numpy())               # no short possible: the long leg alone
    assert e.metrics["short_leg_empty_days"] == len(e.net_returns) - int((~p.eligible.any(axis=1)).iloc[:len(e.net_returns)].sum()) and "short_leg_empty_days" not in base.metrics
    assert not (e.holdings < 0).any()
    warm = p.eligible.copy(); warm.iloc[:20] = False                                         # no eligible names at all in a warm-up stretch
    w0 = q.backtest_portfolio(replace(p, eligible=warm, shortable=none.shortable), f, **kw)
    assert w0.metrics["short_leg_empty_days"] == len(w0.net_returns) - 20                   # those days are not blamed on the restriction
    print(f"S4 the short leg is picked among shortable names (day {t}: {len(expect)} names as in a loop); none shortable gives the long-only result  PASS")


def test_weights_short_checks():
    p, _ = _panel(seed=9, T=120, N=30)
    p = replace(p, eligible=pd.DataFrame(True, index=p.dates, columns=p.tickers))
    ok = pd.DataFrame(True, index=p.dates, columns=p.tickers)
    ok.iloc[40:, 0] = False                                                                 # name 0 cannot be shorted from day 40
    p = replace(p, shortable=ok)
    name = p.tickers[0]

    def book(rows):
        W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
        for (a, b), v in rows.items():
            W.iloc[a:b, 0] = v
        return W

    _raises(lambda: q.backtest_weights(p, book({(40, 60): -0.1}), benchmark=None), "cannot be sold short", name)        # open a short
    q.backtest_weights(p, book({(40, 60): -0.1}), benchmark=None, check_shortable=False)                              # the switch turns it off
    q.backtest_weights(p, book({(10, 60): -0.1}), benchmark=None)                          # opened while allowed, held through the restriction
    q.backtest_weights(p, book({(10, 40): -0.2, (40, 60): -0.1}), benchmark=None)          # reduced during the restriction: fine
    _raises(lambda: q.backtest_weights(p, book({(10, 40): -0.1, (40, 60): -0.2}), benchmark=None), "cannot be sold short")  # increased: not fine
    _raises(lambda: q.backtest_weights(p, book({(10, 60): 0.3, (40, 60): -0.1}), benchmark=None), "cannot be sold short")   # long -> smaller short
    q.backtest_weights(p, book({(40, 60): 0.1}), benchmark=None)                           # long where no short is allowed: fine
    print("S5 opening or increasing a short where it cannot be sold short raises (also long to a smaller short); holding, reducing or going long does not  PASS")


def test_universe_catches_flip():
    p, _ = _panel(seed=11, T=120, N=30)
    el = pd.DataFrame(True, index=p.dates, columns=p.tickers); el.iloc[50:, 0] = False
    p = replace(p, eligible=el)
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers)
    W.iloc[10:50, 0] = 0.3; W.iloc[50:80, 0] = -0.1                                         # |position| shrinks, but a short is opened out of the universe
    _raises(lambda: q.backtest_weights(p, W, benchmark=None), "not eligible")
    W2 = W.copy(); W2.iloc[50:80, 0] = 0.1                                                  # reducing a long outside the universe stays allowed
    q.backtest_weights(p, W2, benchmark=None)
    print("S6 turning a long into a smaller short in a name that left the universe raises; reducing a long there does not  PASS")


def test_bans():
    dates = pd.bdate_range("2021-01-04", periods=100); tk = pd.Index(["A", "B", "C"])
    out = q.shortable_from_bans(dates, tk, [(dates[10], dates[19])])
    assert not out.iloc[10:20].to_numpy().any() and out.iloc[:10].to_numpy().all() and out.iloc[20:].to_numpy().all()
    ex = q.shortable_from_bans(dates, tk, [(str(dates[10].date()), None)], exempt=["B", "Z"])
    assert ex.iloc[10:]["B"].all() and not ex.iloc[10:][["A", "C"]].to_numpy().any() and ex.iloc[:10].to_numpy().all()
    mask = pd.DataFrame(False, index=dates, columns=tk); mask.iloc[15:, 2] = True
    fr = q.shortable_from_bans(dates, tk, [(dates[10], dates[30]), (dates[25], dates[40])], exempt=mask)
    assert fr.iloc[10:15].sum().sum() == 0 and fr.iloc[15:41]["C"].sum() == 26 and not fr.iloc[10:41][["A", "B"]].to_numpy().any() and fr.iloc[41:].to_numpy().all()
    assert q.shortable_from_bans(dates, tk, []).to_numpy().all()
    _raises(lambda: q.shortable_from_bans(dates, tk, [(dates[20], dates[10])]), "ends before")
    _raises(lambda: q.shortable_from_bans(dates, tk, [(dates[1],)]), "pair")
    p, _ = _panel(T=60, N=20)
    part = pd.DataFrame(True, index=p.dates, columns=p.tickers[:10])
    pp = replace(p, shortable=part)
    assert not pp.shortable[p.tickers[10:]].to_numpy().any() and pp.shortable[p.tickers[:10]].to_numpy().all()      # an unknown is not permission
    assert replace(p, shortable=part).fingerprint() != p.fingerprint()
    print("S7 ban periods, exemptions (list or frame), open-ended and overlapping bans, and unknown securities behave as documented  PASS")


def test_ledger_config():
    p, f = _panel()
    with tempfile.TemporaryDirectory() as d:
        led = q.Ledger(d)
        kw = dict(long_q=0.2, short_q=0.2, hold=5, spread_bp=10.0, benchmark=None, grid=False, ledger=led, family="s8")
        q.backtest_portfolio(p, f, name="a", **kw)
        q.backtest_portfolio(p, f, name="a", sell_bp=20.0, **kw)
        q.backtest_portfolio(p, f, name="a", sell_bp=20.0, **kw)
        tr = led.trials("s8")
        assert len(tr) == 2, len(tr)                                                         # the same side cost twice is one trial
        assert "sell_bp" not in tr[0]["config"] and "buy_bp" not in tr[0]["config"] and tr[1]["config"]["sell_bp"] == 20.0
    print("S8 a run without side costs keeps its old configuration; a run with them is a different trial  PASS")


def _fake_krx(root: Path):
    days = pd.bdate_range("2021-01-04", periods=90)
    rng = np.random.default_rng(0)
    codes = {f"{(i + 1) * 10:06d}": ("KOSPI" if i < 6 else "KOSDAQ") for i in range(12)}
    px = {c: 10000.0 for c in codes}
    for t, d in enumerate(days):
        rows = []
        for c, m in codes.items():
            if c == "000010" and t >= 45:
                m = "KOSDAQ"                                                                # moves from KOSPI to KOSDAQ on day 45
            new = px[c] * (1 + rng.normal(0, 0.01))
            rows.append(dict(code=c, name=f"N{c}", market=m, close=int(new), change=int(new - px[c]), open=int(new), high=int(new * 1.01), low=int(new * 0.99),
                             volume=1_000_000, value=int(5e9), mktcap=int(new * 1e7), shares=10_000_000))
            px[c] = new
        pd.DataFrame(rows).to_pickle(root / f"{d.strftime('%Y%m%d')}.pkl")
    return days


def test_krx_sell_tax():
    with tempfile.TemporaryDirectory() as d:
        days = _fake_krx(Path(d))
        p = krx.build_krx_panel(d, min_age_days=5, adv_window=5, min_value_krw=1e6)
        sched = {"KOSPI": [(days[0], 30.0), (days[60], 20.0)], "KOSDAQ": [(days[0], 25.0)]}
        tax = krx.sell_tax_panel(p, sched)
        assert tax.shape == p.close.shape
        assert tax.loc[days[10], "000020"] == 30.0 and tax.loc[days[70], "000020"] == 20.0         # KOSPI changes on day 60
        assert tax.loc[days[10], "000100"] == 25.0 and tax.loc[days[70], "000100"] == 25.0         # KOSDAQ does not
        assert tax.loc[days[10], "000010"] == 30.0 and tax.loc[days[44], "000010"] == 30.0         # still KOSPI before the move
        assert tax.loc[days[45], "000010"] == 25.0 and tax.loc[days[70], "000010"] == 25.0         # KOSDAQ from day 45, whatever KOSPI does
        _raises(lambda: krx.sell_tax_panel(p, {"KOSPI": [(days[0], 30.0)]}), "KOSDAQ")
        _raises(lambda: krx.sell_tax_panel(p, {"KOSPI": [(days[5], 30.0)], "KOSDAQ": [(days[0], 25.0)]}), "no rate is known")
        r = q.backtest_weights(p, pd.DataFrame(0.05, index=p.dates, columns=p.tickers), sell_bp=tax, benchmark=None, check_universe=False)
        assert r.metrics["side_cost_annual_bp"] == 0.0                                          # a constant book never trades after the build
    print("S9 each security pays the rate of its market on that day, including after a move between markets; a missing market or date raises  PASS")


def test_tranche_leaves_no_residue():
    from pitbacktest.portfolio import _tranche
    rng = np.random.default_rng(0)
    w = np.zeros((600, 100))
    for t in range(600):
        w[t, rng.choice(100, 30, replace=False)] = 1 / 30                                        # 1/30 is not exact in binary: the residue shows
    out = _tranche(w, 5)
    ref = np.array([w[max(0, t - 4):t + 1].mean(axis=0) for t in range(600)])
    assert np.abs(out - ref).max() < 1e-15
    assert ((ref == 0) == (out == 0)).all(), "a rounding residue is counted as a position"
    assert ((out > 0).sum(axis=1) == (ref > 0).sum(axis=1)).all()
    p, f = _panel(seed=2, T=300, N=60)
    r = q.backtest_portfolio(p, f, long_q=0.2, short_q=None, hold=5, benchmark=None, grid=False)       # long only: holdings are the long leg
    exact = np.array([(r.holdings[t] > 0).sum() for t in range(len(r.holdings))])
    assert abs(r.metrics["avg_positions"] - exact[exact > 0].mean()) < 1e-9
    print("S10 averaging overlapping tranches leaves no rounding residue, so avg_positions counts real positions only  PASS")


if __name__ == "__main__":
    test_tranche_leaves_no_residue()
    test_side_cost_matches_a_loop()
    test_engines_agree_with_side_costs()
    test_schedule_and_unknown_rates()
    test_shortable_in_portfolio()
    test_weights_short_checks()
    test_universe_catches_flip()
    test_bans()
    test_ledger_config()
    test_krx_sell_tax()
    print("side cost and short-selling tests: all passed")
