"""The library against implementations written from its specification by assistants that had not seen its code (docs/blind_review/README.md).

BR1 portfolio   the net returns of 108 settings and the metrics agree; turnover_daily is one-way (the blind code did not halve it: the specification had not said)
BR2 events      15 figures of the event statistics agree for 72 settings, up to the single precision of the forward returns
BR3 gates       Newey-West, the firing structure (the figures that were defined), sub-periods, years and the holdout agree
BR4 statistics  deflated Sharpe, PBO, the sign test, Wilson, BH, Holm, mean tests, market-relative figures and the regression agree
BR5 closed holes  each place where the blind readers differed is pinned to the library's documented behaviour
"""
from __future__ import annotations
import itertools, math, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import analytics as an, robustness as rb, validation as vl
from pitbacktest.core import gates as G
from pitbacktest.core.estimators import newey_west_t
from pitbacktest.portfolio import metrics as lib_metrics
from pitbacktest.review import _wilson
import blind_event as BE
import blind_portfolio as BP
import blind_stats as BS


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _world(T, N, seed, p_fire=0.05, el_p=0.92, nan_p=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T); cols = [f"S{i:02d}" for i in range(N)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (T, N)), axis=0)), index=idx, columns=cols)
    el = pd.DataFrame(rng.random((T, N)) < el_p, index=idx, columns=cols); el.iloc[:30] = False
    fac = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols).mask(rng.random((T, N)) < nan_p)
    sig = pd.DataFrame(rng.random((T, N)) < p_fire, index=idx, columns=cols)
    return close, el, fac, sig, idx, cols


def test_portfolio():
    close, el, fac, _, idx, cols = _world(300, 40, 5, nan_p=0.05)
    n = 0
    for lag, hold, sp, w, sq in itertools.product((0, 1, 2), (1, 3, 5), (0.0, 10.0), ("equal", "signal", "rank"), (0.2, None)):
        p = q.Panel(close=close, eligible=el, volume=close * 1e5, mkt_cap=close * 1e7, market="T", entry_lag=lag)
        r = _quiet(lambda: q.backtest_portfolio(p, fac, long_q=0.2, short_q=sq, hold=hold, weighting=w, spread_bp=sp, benchmark=None, grid=False, funding=False))
        mine = BP.portfolio_net_returns(close, el, fac, 0.2, sq, hold, sp, entry_lag=lag, weighting=w)
        assert np.allclose(mine.reindex(r.net_returns.index).to_numpy(float), r.net_returns.to_numpy(float), atol=1e-10, equal_nan=True), (lag, hold, sp, w, sq)
        n += 1
    x = r.net_returns.to_numpy()
    L, M = lib_metrics(x, idx, 252), BP.metrics_from_returns(pd.Series(x, index=r.net_returns.index), 252)
    for k in ("CAGR", "MDD", "Sharpe", "Sortino", "Calmar", "vol", "pos_days"):
        assert np.isclose(L[k], M[k], equal_nan=True), k
    for lag, hold, sq in ((1, 5, 0.2), (1, 1, 0.2), (1, 5, None)):
        p = q.Panel(close=close, eligible=el, volume=close * 1e5, mkt_cap=close * 1e7, market="T", entry_lag=lag)
        r = _quiet(lambda: q.backtest_portfolio(p, fac, long_q=0.2, short_q=sq, hold=hold, spread_bp=0.0, benchmark=None, grid=False, funding=False))
        blind = BP.turnover_daily(close, el, fac, 0.2, sq, hold, entry_lag=lag)
        assert abs(r.metrics["turnover_daily"] - 0.5 * blind) / blind < 0.01, (r.metrics["turnover_daily"], blind)      # one-way is half of the sum of absolute changes (the mean differs by the one bar at the start)
    print(f"BR1 the net returns of {n} portfolio settings and the metrics equal the blind implementation; turnover is one-way  PASS")


def test_events():
    keys = ("n_trades", "win_rate", "base_rate", "lift_pp", "mean_bp", "median_bp", "avg_win_bp", "avg_loss_bp", "payoff", "p25_bp", "p75_bp", "excess_bp", "t", "n_days", "net_bp")
    n = 0
    for seed, lag, h, cost, pf in itertools.product((0, 1), (0, 1, 2), (1, 5, 21), (0.0, 20.0), (0.04, 0.12)):
        close, el, _, sig, idx, cols = _world(600, 40, seed, p_fire=pf)
        p = q.Panel(close=close, eligible=el, volume=close * 1e5, mkt_cap=close * 1e7, market="T", entry_lag=lag)
        try:
            lib = _quiet(lambda: q.backtest_event(p, sig, horizons=(h,), cost_bp=cost, neutralize_check=False)).per_horizon.get(h)
        except ValueError:
            lib = None
        mine = BE.event_stats(close, el, sig, h, cost, entry_lag=lag)
        if lib is None:
            assert mine.get("excluded", False), (seed, lag, h, cost, pf)
            continue
        for k in keys:
            assert np.isclose(float(lib[k]), float(mine[k]), rtol=1e-5, atol=1e-4, equal_nan=True), (seed, lag, h, cost, pf, k, lib[k], mine[k])       # the library keeps forward returns in single precision
        n += 1
    assert n >= 60
    print(f"BR2 15 event figures agree in {n} settings  PASS")


def test_gates():
    rng = np.random.default_rng(3)
    for lag in (0, 1, 5, 21, 63):
        for T in (60, 100, 800):
            x = rng.normal(0.001, 0.02, T) + np.convolve(rng.normal(0, 0.01, T), [1, .5, .3], mode="same")
            if lag < T - 1:
                assert np.allclose(np.array(newey_west_t(x, lag), float), np.array(BE.newey_west_t(x, lag), float), equal_nan=True, rtol=1e-9), (lag, T)
    for D, N, pf in ((100, 40, 0.1), (60, 10, 0.3), (250, 60, 0.02), (30, 40, 0.2), (200, 25, 0.5)):
        fire = rng.random((D, N)) < pf
        dates = pd.bdate_range("2020-01-01", periods=D); tick = pd.Index([f"S{i:02d}" for i in range(N)])
        a, b = G.fire_structure(fire, dates, tick), BE.fire_structure(fire, dates, tick)
        assert abs(a["max_name_share"] - b["top_name_share"]) < 1e-12 and a["max_consecutive"] == b["longest_run"]
    dates = pd.bdate_range("2015-01-01", periods=5 * 260)
    v = rng.normal(0.002, 0.01, len(dates))
    a, b = G.subperiod(v, dates, 21), BE.subperiod(v, dates, 21)
    assert abs(a["front"]["mean_bp"] - b["first_mean"] * 1e4) < 1e-8 and abs(a["back"]["t"] - b["second_t"]) < 1e-9 and a["sign_match"] == b["same_sign"]
    a, b = G.yearly(v, dates, 21), BE.yearly(v, dates, 21)
    assert a["total"] == len(b["years"]) and all(abs(a["by_year"][y["year"]]["t"] - y["t"]) < 1e-9 for y in b["years"])
    a, b = G.oos_holdout(v, dates, 12, 21), BE.oos_holdout(v, dates, 12, 21)
    assert abs(a["OOS"]["mean_bp"] - b["mean"] * 1e4) < 1e-8 and abs(a["IS"]["t"] - b["is_t"]) < 1e-9 and a["OOS"]["n"] == b["n_days"]
    print("BR3 Newey-West, firing structure, sub-periods, years and the holdout equal the blind implementation  PASS")


def test_statistics():
    rng = np.random.default_rng(11)
    x = rng.normal(0.0005, 0.01, 600); xs = pd.Series(x, index=pd.bdate_range("2019-01-01", periods=600))
    R = rng.normal(0.0004, 0.01, (500, 12)); R[:, 3] += 0.0008
    assert np.allclose(vl.sharpe(R, 252), BS.sharpe(R, 252))
    for kw in (dict(periods_per_year=252), dict(trials=50, periods_per_year=1)):
        a, b = vl.deflated_sharpe(R, **kw), BS.deflated_sharpe(R, **kw)
        assert all(np.isclose(float(a[k]), float(b[k]), rtol=1e-9) for k in a)
    for shape, blocks in (((320, 6), 16), ((333, 6), 8)):
        Rp = rng.normal(0.0003, 0.01, shape)
        a, b = vl.pbo_cscv(Rp, blocks=blocks), BS.pbo_cscv(Rp, blocks=blocks)
        assert all(np.isclose(float(a[k]), float(b[k]), rtol=1e-9) for k in a)
    for k, n in ((0, 10), (3, 10), (5, 10), (40, 100), (600, 1000)):
        assert np.isclose(rb._binom_two_sided(k, n), BS.binom_two_sided(k, n), rtol=1e-12) and np.allclose(_wilson(k, n), BS.wilson(k, n), atol=1e-12)
    p = rng.random(30) ** 2
    assert np.allclose(rb.bh_fdr(p), BS.bh_fdr(p)) and np.allclose(rb.holm(p), BS.holm(p))
    a, b = rb.mean_tests(xs), BS.mean_tests(xs)
    assert all(np.isclose(float(a[k]), float(b[k]), rtol=1e-8) for k in b if k in a and not isinstance(a[k], (tuple, list))) and np.allclose(a["win_ci"], b["win_ci"])
    bench = pd.Series(rng.normal(0.0003, 0.009, 600), index=xs.index)
    a, b = rb.market_relative(xs, bench), BS.market_relative(xs, bench)
    assert all(np.isclose(float(a[k]), float(b[k]), rtol=1e-8) for k in b if k in a)
    F = pd.DataFrame({"m": bench, "s": rng.normal(0, 0.005, 600)}, index=xs.index)
    la, lb = an.alpha_beta(xs, F), BS.alpha_beta(xs, F)
    for k in ("alpha", "alpha_annual", "alpha_t", "alpha_p", "r2", "n", "lag"):
        assert np.isclose(float(la[k]), float(lb[k]), rtol=1e-8), k
    assert all(abs(la["betas"][c]["beta"] - lb["betas"][c]["beta"]) < 1e-10 and abs(la["betas"][c]["t"] - lb["betas"][c]["t"]) < 1e-8 for c in la["betas"])
    rk, bk = rb.risk_profile(xs), BS.risk_profile(xs)
    for k in ("var_95", "var_99", "cvar_95", "cvar_99", "tail_ratio", "best_bar", "worst_bar", "max_dd_bars", "share_underwater", "ulcer_index", "gain_to_pain", "total_return", "var_95_normal"):
        assert np.isclose(float(rk[k]), float(bk[k]), rtol=1e-8), k
    print("BR4 deflated Sharpe, PBO, sign test, Wilson, BH, Holm, mean tests, benchmark-relative figures and the regression equal the blind implementation  PASS")


def test_closed_holes():
    rng = np.random.default_rng(2)
    # alpha_beta: the residual volatility divides by n - k (k = coefficients), not n - 1
    xs = pd.Series(rng.normal(0.0005, 0.01, 600), index=pd.bdate_range("2019-01-01", periods=600))
    F = pd.DataFrame({"m": rng.normal(0, 0.009, 600), "s": rng.normal(0, 0.005, 600)}, index=xs.index)
    la, lb = an.alpha_beta(xs, F), BS.alpha_beta(xs, F)
    assert abs(la["resid_vol_annual"] - lb["resid_vol_annual"] * math.sqrt((600 - 1) / (600 - 3))) < 1e-10
    # risk_profile: vol_now is annualised and the volatility estimate starts from the first 20 bars
    rk = rb.risk_profile(xs)
    r = xs.to_numpy(); lam = 0.94
    var = np.empty(len(r)); var[0] = np.var(r[:20], ddof=1)
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    nxt = math.sqrt(lam * var[-1] + (1 - lam) * r[-1] ** 2)
    assert abs(rk["vol_now"] - nxt * math.sqrt(252)) < 1e-12
    # Newey-West: nothing below 30 observations; a lag longer than the sample keeps its weights
    z = rng.normal(0, 1, 40)
    assert all(np.isnan(v) for v in newey_west_t(z[:29], 5)[:3]) and newey_west_t(z[:29], 5)[3] == 29
    xd = z - z.mean(); T = 40; lag = 63
    s = xd @ xd / T + sum(2 * (1 - l / (lag + 1)) * (xd[l:] @ xd[:-l]) / T for l in range(1, T))
    assert abs(newey_west_t(z, lag)[1] - math.sqrt(s / T)) < 1e-12
    # the firing set's turnover skips a pair when either day has no firing
    f = np.zeros((6, 3), bool); f[0, 0] = f[1, 0] = f[1, 1] = f[3, 0] = f[4, 0] = True
    d = G.fire_structure(f, pd.bdate_range("2020-01-01", periods=6), pd.Index(list("abc")))["daily_turnover"]
    assert abs(d - np.mean([1 - 1 / 2, 1 - 1 / 1])) < 1e-12, d                                           # days 0-1 and 3-4 only; the empty days 2 and 5 are not counted
    # the yearly gate skips a year with under 40 bars; a year without a firing figure at all does not count
    dates = pd.bdate_range("2018-01-01", periods=260 + 30)
    y = G.yearly(np.full(290, 0.001) + rng.normal(0, 0.0001, 290), dates)
    assert y["total"] == 1
    # G6 is evaluated only with fire and tickers
    base = G.run_gates(np.full(1300, 0.002) + rng.normal(0, 0.001, 1300), pd.bdate_range("2015-01-01", periods=1300), t_stat=4.0, net_bp=5.0, rho=0.8)
    assert len(base["steps"]) == 5
    print("BR5 each hole in the specification is pinned to the documented behaviour  PASS")


if __name__ == "__main__":
    test_portfolio(); test_events(); test_gates(); test_statistics(); test_closed_holes()
    print("blind-reference tests: all passed")
