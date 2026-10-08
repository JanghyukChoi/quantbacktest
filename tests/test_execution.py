"""What can actually be traded: halts, daily price limits, whole lots, a minimum trade value, and trading at another price.

X1 halt        a name that cannot be traded keeps its position; a loop written from the rule gives the same returns and costs
X2 sides       a name locked up cannot be bought but can be sold, one locked down the opposite; the mask is read on the execution day
X3 portfolio   `backtest_portfolio` with masks: nothing blocked is bit for bit the old result; blocked trades never change a position;
               with buys and sells blocked on different days it equals the net-position loop and the weights engine
X4 builder     `tradability` finds halts and limit locks as a loop does, with a limit that changed on a date;
               positions in a delisted name can be closed either way, never frozen
X5 lots        positions are whole lots at the real price; a loop gives the same weights and returns; a large capital approaches no rounding
X6 minimum     trades under the minimum value are skipped and counted; a high minimum leaves the book unchanged
X7 prices      `at_prices` enters and marks at the open: the result equals the engine run on a panel built from the open by hand
X8 checks      lots without capital, capital without price, a bad lot and a bad limit raise
X9 ledger      execution settings are a different trial; a run without them keeps its configuration
"""
from __future__ import annotations
import sys, tempfile
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import execution as ex
from test_reconcile import _panel


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"no ValueError for {needles}")


def _setup(seed=4, T=200, N=30):
    p, _ = _panel(seed=seed, T=T, N=N)
    p = replace(p, eligible=p.eligible | True)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.05, p.close.shape); W[rng.random(W.shape) < 0.4] = 0.0
    return p, pd.DataFrame(W, index=p.dates, columns=p.tickers)


def _loop(W, p, buy_ok, sell_ok, spread_bp):
    """Plain-Python reference: positions, daily cost and net return per signal date, written from the rule."""
    T, N = W.shape
    lag = p.entry_lag
    ret = p.close.pct_change(fill_method=None).fillna(0.0).to_numpy()
    pos = np.zeros((T, N)); prev = np.zeros(N)
    for t in range(T):
        for j in range(N):
            want = W[t, j]
            e = min(t + lag, T - 1) if t + lag < T else None
            b = True if e is None else buy_ok[e, j]
            s = True if e is None else sell_ok[e, j]
            if want > prev[j] and not b:
                want = prev[j]
            if want < prev[j] and not s:
                want = prev[j]
            pos[t, j] = want
        prev = pos[t].copy()
    net = np.zeros(T)
    for t in range(T - lag - 1):
        g = sum(pos[t, j] * ret[t + lag + 1, j] for j in range(N))
        c = 0.0 if t == 0 else sum(abs(pos[t, j] - pos[t - 1, j]) for j in range(N)) * spread_bp / 2 / 1e4
        net[t] = g - c
    return pos, net[:T - lag - 1]


def test_halt_matches_a_loop():
    p, W = _setup()
    rng = np.random.default_rng(1)
    ok = rng.random(p.close.shape) > 0.2                                     # 20% of name-days cannot be traded either way
    okf = pd.DataFrame(ok, index=p.dates, columns=p.tickers)
    pr = replace(p, can_buy=okf, can_sell=okf)
    r = q.backtest_weights(pr, W, spread_bp=20.0, benchmark=None, check_universe=False)
    pos, net = _loop(W.to_numpy(), p, ok, ok, 20.0)
    assert np.allclose(r.holdings, pos, atol=0, rtol=0), np.abs(r.holdings - pos).max()
    assert np.allclose(r.net_returns.to_numpy(), net, atol=1e-13, rtol=0), np.abs(r.net_returns.to_numpy() - net).max()
    free = q.backtest_weights(p, W, spread_bp=20.0, benchmark=None, check_universe=False)
    stuck = np.abs(pos - W.to_numpy()).sum(axis=1).mean()
    assert abs(r.metrics["mean_stuck_weight"] - stuck) < 1e-12, (r.metrics["mean_stuck_weight"], stuck)          # weight held where the target said otherwise
    assert r.metrics["blocked_trades"] > 0 and 0 < r.metrics["blocked_turnover_share"] < 1 and "blocked_trades" not in free.metrics
    assert r.metrics["turnover_daily"] < free.metrics["turnover_daily"]
    print(f"X1 positions and net returns equal an independent loop; {r.metrics['blocked_trades']} trades blocked ({100 * r.metrics['blocked_turnover_share']:.0f}% of the turnover asked)  PASS")


def test_freeze_is_measured():
    p, _ = _setup(seed=15, T=100, N=20)
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); W.iloc[10:22, 0] = 0.1                    # long 10% from day 10, wanted flat from day 22
    ok = pd.DataFrame(True, index=p.dates, columns=p.tickers); ok.iloc[21:30, 0] = False               # halted on execution days 21..29 (lag 1: signal rows 20..28)
    r = q.backtest_weights(replace(p, can_buy=ok, can_sell=ok), W, benchmark=None, check_universe=False)
    # the exit is wanted on signal row 22 (executes on 23, halted) and the halt lasts through execution row 29, i.e. signal rows 22..28: 7 days stuck
    assert r.metrics["longest_freeze_days"] == 7, r.metrics["longest_freeze_days"]
    assert abs(r.metrics["mean_stuck_weight"] - 7 * 0.1 / 100) < 1e-12, r.metrics["mean_stuck_weight"]
    assert r.holdings[28, 0] == 0.1 and r.holdings[29, 0] == 0.0
    print("X1b a position frozen by a halt is measured: 7 days stuck, 0.1 of capital, as counted by hand  PASS")


def test_sides_and_execution_day():
    p, W = _setup(seed=6)
    rng = np.random.default_rng(2)
    nb, ns = rng.random(p.close.shape) > 0.15, rng.random(p.close.shape) > 0.15
    for lag in (0, 1, 2):
        pl = replace(p, entry_lag=lag, can_buy=pd.DataFrame(nb, index=p.dates, columns=p.tickers), can_sell=pd.DataFrame(ns, index=p.dates, columns=p.tickers))
        r = q.backtest_weights(pl, W, spread_bp=10.0, benchmark=None, check_universe=False)
        pos, net = _loop(W.to_numpy(), replace(p, entry_lag=lag), nb, ns, 10.0)
        assert np.array_equal(r.holdings, pos), lag
        assert np.allclose(r.net_returns.to_numpy(), net, atol=1e-13, rtol=0), (lag, np.abs(r.net_returns.to_numpy() - net).max())
    only_buy = replace(p, can_buy=pd.DataFrame(nb, index=p.dates, columns=p.tickers))        # only buys can be blocked: sells always go through
    r = q.backtest_weights(only_buy, W, benchmark=None, check_universe=False)
    pos, _ = _loop(W.to_numpy(), p, nb, np.ones_like(nb), 0.0)
    assert np.array_equal(r.holdings, pos)
    print("X2 buys and sells are blocked separately and the mask is read on the execution day (lag 0, 1, 2 equal the loop)  PASS")


def test_portfolio_masks():
    p, f = _panel(seed=8, T=300, N=60)
    kw = dict(long_q=0.2, short_q=0.2, hold=3, spread_bp=10.0, benchmark=None)
    base = q.backtest_portfolio(p, f, **kw)
    allok = pd.DataFrame(True, index=p.dates, columns=p.tickers)
    same = q.backtest_portfolio(replace(p, can_buy=allok, can_sell=allok), f, **kw)
    assert np.array_equal(base.net_returns.to_numpy(), same.net_returns.to_numpy())
    assert same.metrics["blocked_trades"] == 0 and np.array_equal(base.holdings, same.holdings)
    assert set(base.grid) == set(same.grid) and all(base.grid[k]["Sharpe"] == same.grid[k]["Sharpe"] for k in base.grid)
    rng = np.random.default_rng(3)
    ok = pd.DataFrame(rng.random(p.close.shape) > 0.25, index=p.dates, columns=p.tickers)
    b = q.backtest_portfolio(replace(p, can_buy=ok, can_sell=ok), f, **kw)
    H = b.holdings                                                                           # hl - hs with each leg blocked on its own
    exec_ok = ok.shift(-p.entry_lag, fill_value=True).to_numpy()
    moved = np.abs(np.diff(H, axis=0)) > 1e-12
    assert not (moved & ~exec_ok[1:]).any(), "a blocked name changed its position"
    assert b.metrics["blocked_trades"] > 0 and 0 < b.metrics["blocked_turnover_share"] < 1
    print(f"X3 backtest_portfolio: all-True masks leave the result bit for bit unchanged (grid too); no blocked name moves ({b.metrics['blocked_trades']} trades blocked)  PASS")


def test_portfolio_asymmetric_masks():
    # With hold=1 a name is in one leg at a time, so blocking per leg equals blocking the net position: the net-position loop is the reference.
    # Asymmetric masks (buys and sells blocked on different days) are what tell a short leg's buy from its sell.
    p, f = _panel(seed=13, T=300, N=60)
    kw = dict(long_q=0.2, short_q=0.2, hold=1, spread_bp=10.0, benchmark=None, grid=False, buy_bp=3.0, sell_bp=17.0)
    free = q.backtest_portfolio(p, f, **kw)
    rng = np.random.default_rng(4)
    nb, ns = rng.random(p.close.shape) > 0.3, rng.random(p.close.shape) > 0.3
    r = q.backtest_portfolio(replace(p, can_buy=pd.DataFrame(nb, index=p.dates, columns=p.tickers),
                                     can_sell=pd.DataFrame(ns, index=p.dates, columns=p.tickers)), f, **kw)
    T, N = free.holdings.shape
    pos, _ = _loop(free.holdings, p, nb, ns, 0.0)
    assert np.allclose(r.holdings, pos, atol=1e-15), np.abs(r.holdings - pos).max()
    # the same positions through the weights engine (net costs equal the legs' costs when no name is in both legs)
    w = q.backtest_weights(replace(p, can_buy=None, can_sell=None), pd.DataFrame(pos, index=p.dates, columns=p.tickers), spread_bp=10.0, buy_bp=3.0, sell_bp=17.0,
                           benchmark=None, check_universe=False)
    assert np.allclose(r.net_returns.to_numpy(), w.net_returns.to_numpy(), atol=1e-13, rtol=0), np.abs(r.net_returns.to_numpy() - w.net_returns.to_numpy()).max()
    print(f"X3b with buys and sells blocked on different days the portfolio's positions equal the net-position loop and its net returns equal the weights engine's  PASS")


def test_delisted_positions_are_not_frozen():
    # First real-data run: a short in a delisted name could not be bought back (no price, so "cannot buy") and stayed in the book for ever.
    p, f = _panel(seed=14, T=320, N=60, delist=True)
    cb, cs = ex.tradability(p)
    pm = replace(p, can_buy=cb, can_sell=cs)
    kw = dict(long_q=0.2, short_q=0.2, hold=5, spread_bp=10.0, benchmark=None, grid=False)
    r = q.backtest_portfolio(pm, f, **kw)
    free = q.backtest_portfolio(p, f, **kw)
    last = {3: 150, 11: 200, 20: 90}
    for j, d in last.items():
        assert np.abs(r.holdings[d + 8:, j]).max() == 0.0, (j, np.abs(r.holdings[d + 8:, j]).max())                  # gone a week after the last bar
    assert np.allclose(r.holdings, free.holdings), "nothing but delisted names could be blocked here"
    assert r.metrics["blocked_turnover_share"] < 1e-9, r.metrics["blocked_turnover_share"]
    print("X4b a long or a short in a delisted name is closed after the last bar instead of being frozen (the case the first real-data run exposed)  PASS")


def test_tradability_builder():
    p, _ = _setup(seed=9, T=150, N=20)
    c = p.close.copy(); v = pd.DataFrame(1000.0, index=p.dates, columns=p.tickers)
    c.iloc[30, 0] = np.nan                                                                   # no price: halted
    v.iloc[40, 1] = 0.0                                                                      # no volume: halted
    v.iloc[41, 2] = np.nan                                                                   # unknown volume: not permission
    c.iloc[60, 3] = c.iloc[59, 3] * 1.295                                                    # +29.5% on a 30% limit, tolerance 1pt: locked up
    c.iloc[70, 4] = c.iloc[69, 4] * 0.70                                                     # -30%: locked down
    c.iloc[80, 5] = c.iloc[79, 5] * 1.20                                                     # +20%: not locked
    c.iloc[120, 6] = c.iloc[119, 6] * 1.17                                                   # +17%: locked under a 15% limit only
    pp = replace(p, close=c, volume=v)
    cb, cs = ex.tradability(pp, limit=0.30)
    assert not cb.iloc[30, 0] and not cs.iloc[30, 0]
    assert not cb.iloc[40, 1] and not cs.iloc[40, 1] and not cb.iloc[41, 2] and not cs.iloc[41, 2]
    assert not cb.iloc[60, 3] and cs.iloc[60, 3] and not cs.iloc[70, 4] and cb.iloc[70, 4]
    assert cb.iloc[80, 5] and cs.iloc[80, 5] and cb.iloc[120, 6]
    da = pd.DataFrame(False, index=p.dates, columns=p.tickers); da.iloc[90, 7] = True; c.iloc[91:, 7] = np.nan      # delisted: last bar at row 90
    cbd, csd = ex.tradability(replace(pp, close=c, delist_after=da), limit=0.30)
    assert cbd.iloc[90, 7] and cbd.iloc[91:, 7].all() and csd.iloc[91:, 7].all()                                  # after a delisting a position can be closed either way
    live = ex.tradability(pp, limit=0.30)[1]
    assert not live.iloc[30, 0] and live.iloc[31:, 0].all()                                                       # a gap that is not a delisting stays blocked only while it lasts
    c3 = c.copy(); c3.iloc[100:130, 8] = np.nan                                                                    # 30 days without a price
    c3.iloc[40:140, 9] = np.nan                                                                                    # 100 days without a price, then it trades again
    g = ex.tradability(replace(pp, close=c3), max_gap_days=60)
    assert not g[0].iloc[100:130, 8].any() and not g[1].iloc[100:130, 8].any()                                       # a short gap: halted throughout
    assert not g[0].iloc[40:100, 9].any() and g[0].iloc[100:140, 9].all() and g[1].iloc[100:140, 9].all()           # from day 61 of the gap on: settled, both ways
    assert g[0].iloc[140:, 9].all()
    off = ex.tradability(replace(pp, close=c3), max_gap_days=None)
    assert not off[0].iloc[40:140, 9].any()
    v0 = v.copy(); v0.iloc[40:200, 10] = 0.0                                                                       # a price but no volume for ever: stays frozen
    z = ex.tradability(replace(pp, volume=v0), max_gap_days=60)
    assert not z[0].iloc[40:, 10].any() and not z[1].iloc[40:, 10].any()
    sched = [(p.dates[0], 0.15), (p.dates[100], 0.30)]                                       # a limit that changed on a date
    cb2, _ = ex.tradability(pp, limit=sched)
    assert cb2.iloc[120, 6]                                                                  # +17% after the change to 30%: tradable
    c2 = c.copy(); c2.iloc[50, 6] = c2.iloc[49, 6] * 1.17                                    # +17% before the change: locked under 15%
    cb3, _ = ex.tradability(replace(pp, close=c2), limit=sched)
    assert not cb3.iloc[50, 6]
    # a panel without volume is judged by prices alone
    nv = ex.tradability(replace(pp, volume=None), limit=None)[0]
    assert nv.iloc[40, 1] and not nv.iloc[30, 0] and nv.iloc[41, 2]
    _raises(lambda: ex.tradability(pp, limit=1.5), "limit")
    _raises(lambda: ex.tradability(pp, limit=0.3, tol=0.5), "tol")
    print("X4 tradability marks no price, no or unknown volume and limit locks as a loop would, with a limit that changes on a date  PASS")


def test_lots():
    p, W = _setup(seed=5, T=160, N=25)
    rng = np.random.default_rng(5)
    price = pd.DataFrame(np.exp(rng.normal(4.5, 1.0, p.close.shape)), index=p.dates, columns=p.tickers)     # real price level, not the adjusted series
    cap, lot = 2e5, 10.0
    r = q.backtest_weights(p, W, capital=cap, price=price, lot=lot, spread_bp=10.0, benchmark=None, check_universe=False)
    H = r.holdings
    px = price.shift(-p.entry_lag).to_numpy()                                                        # row t is traded, so sized, at the price of day t + lag
    ok_rows = np.isfinite(px).all(axis=1)
    shares = H * cap / np.where(np.isfinite(px), px, 1.0)
    assert np.allclose((shares / lot)[ok_rows], np.round(shares / lot)[ok_rows], atol=1e-6), "a position is not a whole number of lots"
    ref = np.round(W.to_numpy() * cap / px / lot) * lot * px / cap
    assert np.allclose(H[ok_rows], ref[ok_rows], atol=1e-12)
    wrong = np.round(W.to_numpy() * cap / price.to_numpy() / lot) * lot * price.to_numpy() / cap          # sized at the signal-day price instead
    assert not np.allclose(H[ok_rows], wrong[ok_rows], atol=1e-6), "the test cannot tell the two days apart"
    assert r.metrics["mean_abs_rounding_gap"] > 0
    big = q.backtest_weights(p, W, capital=1e15, price=price, lot=1.0, spread_bp=10.0, benchmark=None, check_universe=False)
    plain = q.backtest_weights(p, W, spread_bp=10.0, benchmark=None, check_universe=False)
    assert np.abs(big.net_returns.to_numpy() - plain.net_returns.to_numpy()).max() < 1e-9           # enormous capital: rounding vanishes
    lots_series = pd.Series(np.where(np.arange(25) % 2 == 0, 10.0, 100.0), index=p.tickers)          # lot size by ticker
    rs = q.backtest_weights(p, W, capital=cap, price=price, lot=lots_series, benchmark=None, check_universe=False)
    sh = (rs.holdings * cap / px)[ok_rows]
    assert np.allclose(sh[:, 1] / 100.0, np.round(sh[:, 1] / 100.0), atol=1e-6) and np.allclose(sh[:, 0] / 10.0, np.round(sh[:, 0] / 10.0), atol=1e-6)
    base_pos = r.holdings
    t0, j0 = next((t, j) for t in range(60, 150) for j in range(25) if base_pos[t - 1, j] != 0 and base_pos[t, j] != base_pos[t - 1, j])   # a day the name would trade
    nanp = price.copy(); nanp.iloc[t0 + p.entry_lag, j0] = np.nan                                    # no price on the execution day: nothing to size, the position is kept
    rn = q.backtest_weights(p, W, capital=cap, price=nanp, lot=lot, benchmark=None, check_universe=False)
    assert rn.holdings[t0, j0] == rn.holdings[t0 - 1, j0] != 0
    # a flat target needs no price: a delisted name is closed even though it has no price any more (capital given)
    Wd = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); Wd.iloc[10:31, 2] = -0.1                    # short 10% until row 30, flat from row 31
    pd_ = price.copy(); pd_.iloc[32:, 2] = np.nan                                                    # no price from row 32 on (delisted)
    rd = q.backtest_weights(p, Wd, capital=cap, price=pd_, lot=lot, benchmark=None, check_universe=False)
    assert rd.holdings[10:31, 2].min() < 0 and rd.holdings[31:, 2].max() == 0 and rd.holdings[31:, 2].min() == 0, rd.holdings[28:40, 2]
    print(f"X5 positions are whole lots at the real price (equal to a loop); lot by ticker; no price keeps the position; capital 1e15 equals no rounding  PASS")


def test_min_trade():
    p, W = _setup(seed=7, T=160, N=25)
    rng = np.random.default_rng(7)
    price = pd.DataFrame(np.exp(rng.normal(3.0, 0.5, p.close.shape)), index=p.dates, columns=p.tickers)
    cap = 1e6
    a = q.backtest_weights(p, W, capital=cap, price=price, lot=1.0, benchmark=None, check_universe=False)
    b = q.backtest_weights(p, W, capital=cap, price=price, lot=1.0, min_trade_value=20000.0, benchmark=None, check_universe=False)
    assert b.metrics["min_trade_skipped"] > 0 and a.metrics["min_trade_skipped"] == 0
    assert b.metrics["turnover_daily"] < a.metrics["turnover_daily"]
    d = np.abs(np.diff(b.holdings, axis=0)) * cap
    moved = d[d > 1e-9]
    assert moved.min() >= 20000.0 - 1e-6, moved.min()                                               # nothing smaller than the minimum was traded
    huge = q.backtest_weights(p, W, capital=cap, price=price, lot=1.0, min_trade_value=1e12, benchmark=None, check_universe=False)
    assert np.abs(huge.holdings).sum() == 0.0                                                       # nothing ever reaches the minimum: no position at all
    print(f"X6 trades under the minimum are skipped ({b.metrics['min_trade_skipped']}), nothing smaller than it trades, an unreachable minimum leaves no position  PASS")


def test_at_prices():
    p, f = _panel(seed=10, T=250, N=40)
    rng = np.random.default_rng(10)
    op = p.close * (1 + rng.normal(0, 0.01, p.close.shape))
    po = replace(p, open=op)
    kw = dict(long_q=0.2, short_q=0.2, hold=1, spread_bp=10.0, benchmark=None, grid=False)
    r = q.backtest_portfolio(ex.at_prices(po, "open"), f, **kw)
    r2 = q.backtest_portfolio(replace(p, close=op), f, **kw)
    assert np.array_equal(r.net_returns.to_numpy(), r2.net_returns.to_numpy())
    c = q.backtest_portfolio(p, f, **kw)
    assert not np.allclose(r.net_returns.to_numpy(), c.net_returns.to_numpy())
    # reference: signal d, entry at open(d+1), mark at open(d+2), equal weight in the top and bottom fifth
    t = 120
    sig = f.to_numpy()[t]; el = p.eligible.to_numpy()[t]
    idx = np.where(el)[0]; order = idx[np.argsort(sig[idx])]; n = len(idx); rank = (np.arange(n) + 1) / n
    short, long = order[rank <= 0.2], order[rank >= 0.8]
    o = op.to_numpy()
    rt = (o[t + 2] / o[t + 1] - 1)
    gross = rt[long].mean() - rt[short].mean()
    sig0 = f.to_numpy()[t - 1]; el0 = p.eligible.to_numpy()[t - 1]
    idx0 = np.where(el0)[0]; ord0 = idx0[np.argsort(sig0[idx0])]; n0 = len(idx0); rk0 = (np.arange(n0) + 1) / n0
    s0, l0 = set(ord0[rk0 <= 0.2]), set(ord0[rk0 >= 0.8])
    cost = 0.0
    for names, old in ((set(long), l0), (set(short), s0)):
        w_new = {j: 1 / len(names) for j in names}; w_old = {j: 1 / len(old) for j in old}
        cost += sum(abs(w_new.get(j, 0) - w_old.get(j, 0)) for j in set(w_new) | set(w_old)) * 10.0 / 2 / 1e4
    assert abs((gross - cost) - r.net_returns.to_numpy()[t]) < 1e-12, (gross - cost, r.net_returns.to_numpy()[t])
    _raises(lambda: ex.at_prices(p, "open"), "no 'open'")
    print("X7 trading at the open equals the engine on a panel built from the open, and day 120 equals a hand computation  PASS")


def test_checks():
    p, W = _setup(seed=11, T=120, N=20)
    price = p.close * 10
    _raises(lambda: q.backtest_weights(p, W, lot=10.0, benchmark=None), "need capital")
    _raises(lambda: q.backtest_weights(p, W, min_trade_value=5.0, benchmark=None), "need capital")
    _raises(lambda: q.backtest_weights(p, W, capital=1e6, benchmark=None), "price")
    _raises(lambda: q.backtest_weights(p, W, capital=-1.0, price=price, benchmark=None), "capital")
    _raises(lambda: q.backtest_weights(p, W, capital=1e6, price=price, lot=0.0, benchmark=None), "lot")
    _raises(lambda: q.backtest_weights(p, W, capital=1e6, price=price, lot=pd.Series([1.0], index=[p.tickers[0]]), benchmark=None), "lot")
    _raises(lambda: q.backtest_weights(p, W, capital=1e6, price=price, min_trade_value=-1.0, benchmark=None), "min_trade_value")
    _raises(lambda: q.backtest_weights(p, W, capital=1e6, price=price.to_numpy(), benchmark=None), "frame")
    print("X8 lots without capital, capital without price, a bad lot or minimum raise  PASS")


def test_ledger_and_fingerprint():
    p, W = _setup(seed=12, T=160, N=20)
    price = p.close * 10
    with tempfile.TemporaryDirectory() as d:
        led = q.Ledger(d)
        kw = dict(spread_bp=10.0, benchmark=None, check_universe=False, ledger=led, family="x9", name="w")
        q.backtest_weights(p, W, **kw)
        q.backtest_weights(p, W, capital=1e6, price=price, **kw)
        q.backtest_weights(p, W, capital=1e6, price=price, **kw)
        q.backtest_weights(p, W, capital=2e6, price=price, **kw)
        tr = led.trials("x9")
        assert len(tr) == 3, len(tr)
        assert "capital" not in tr[0]["config"] and "execution" not in tr[0]["config"] and tr[1]["config"]["capital"] == 1e6
    ok = pd.DataFrame(True, index=p.dates, columns=p.tickers)
    assert replace(p, can_buy=ok).fingerprint() != p.fingerprint() and replace(p, can_sell=ok).fingerprint() != p.fingerprint()
    half = pd.DataFrame(True, index=p.dates, columns=p.tickers[:5])
    assert not replace(p, can_buy=half).can_buy[p.tickers[5:]].to_numpy().any()               # an unknown is not permission
    print("X9 execution settings are different trials, a run without them keeps its configuration, and the masks are in the data fingerprint  PASS")


if __name__ == "__main__":
    test_halt_matches_a_loop()
    test_freeze_is_measured()
    test_sides_and_execution_day()
    test_portfolio_masks()
    test_portfolio_asymmetric_masks()
    test_delisted_positions_are_not_frozen()
    test_tradability_builder()
    test_lots()
    test_min_trade()
    test_at_prices()
    test_checks()
    test_ledger_and_fingerprint()
    print("execution tests: all passed")
