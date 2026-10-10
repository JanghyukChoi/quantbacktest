"""Statistics in `robustness` that a second mutation run (a fresh sample, after the first round of fixes) still found unpinned.

RO1 market-relative   the 20-date floor; beta, correlation, tracking error and information ratio against numpy
RO2 sign test         the exact two-sided binomial p-value against exact fractions
RO3 walk-forward      the dates of the training and test windows with an embargo, the one-fold edge, the default annualisation
RO4 plateau           the share of neighbours within reach, and the exact edges 0.7 (plateau) and 0.3 (hill, not spike)
RO5 event null        the cost shifts the null by exactly the cost; ties count against the signal; a market that always rises gives p = 1
RO6 decompose         a factor needs more than 60 observations; the unexplained share is 1 - R2
RO7 drawdowns         the percentiles equal those of the same resampled histories rebuilt in the test
RO8 regimes           the win rate of a regime
RO9 holdings          a result with no holdings is refused
"""
from __future__ import annotations
import dataclasses, math, sys, warnings
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import robustness as rb
from test_argument_checks import _raises
from test_synthetic import make_panel


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def test_market_relative():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=300)
    b = pd.Series(rng.normal(0.0003, 0.01, 300), index=idx)
    s = pd.Series(0.0002 + 0.7 * b + rng.normal(0, 0.006, 300), index=idx)
    r = rb.market_relative(s, b)
    S, M = s.to_numpy(), b.to_numpy(); D = S - M
    assert abs(r["beta"] - np.cov(S, M, ddof=1)[0, 1] / np.var(M, ddof=1)) < 1e-12
    assert abs(r["correlation"] - np.corrcoef(S, M)[0, 1]) < 1e-12 and 0.5 < r["correlation"] < 1
    assert abs(r["tracking_error"] - D.std(ddof=1) * math.sqrt(252)) < 1e-12 and abs(r["information_ratio"] - D.mean() / D.std(ddof=1) * math.sqrt(252)) < 1e-12
    flat = rb.market_relative(pd.Series(0.001, index=idx[:100]), b[:100])                  # a strategy with no variation has no correlation
    assert np.isnan(flat["correlation"])
    assert rb.market_relative(s[:20], b[:20])["n"] == 20
    _raises(lambda: rb.market_relative(s[:19], b[:19]), "at least 20")
    print("RO1 market-relative: beta, correlation, tracking error, information ratio and the 20-date floor  PASS")


def test_sign_test():
    for n in (1, 2, 5, 10, 31, 64, 200):
        for k in range(0, n + 1):
            small = min(k, n - k)
            p = Fraction(2 * sum(math.comb(n, i) for i in range(small + 1)), 2 ** n)
            assert abs(rb._binom_two_sided(k, n) - min(1.0, float(p))) < 1e-15, (k, n)
    assert rb._binom_two_sided(0, 10) == 2 / 1024 and rb._binom_two_sided(5, 10) == 1.0 and rb._binom_two_sided(3, 10) == 2 * 176 / 1024
    assert 0.0 <= rb._binom_two_sided(1500, 3000) <= 1.0                                    # no overflow beyond 1023 tosses
    print("RO2 the exact two-sided sign test equals exact fractions for every k up to n = 200  PASS")


def test_walk_forward_windows():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2019-01-01", periods=200)
    R = pd.DataFrame(rng.normal(0.0005, 0.01, (200, 3)), index=idx, columns=["a", "b", "c"])
    wf = rb.walk_forward(R, train=60, test=20, embargo=5)
    f = wf["folds"]
    assert len(f) == (200 - 60 - 5) // 20                                                                   # folds that fit: 6 full test windows
    first = f.iloc[0]
    assert first["train_start"] == idx[0] and first["train_end"] == idx[59] and first["test_start"] == idx[65] and first["test_end"] == idx[84], first
    last = f.iloc[-1]
    assert last["test_start"] == idx[65 + 20 * 5] and last["test_end"] == idx[84 + 20 * 5] and last["train_end"] == idx[59 + 20 * 5]       # the window moves by `test`
    # exactly one fold fits, and one bar fewer is refused
    one = rb.walk_forward(R.iloc[:85], train=60, test=20, embargo=5)
    assert len(one["folds"]) == 1 and one["folds"].iloc[0]["test_end"] == idx[84]
    _raises(lambda: rb.walk_forward(R.iloc[:84], train=60, test=20, embargo=5), "one fold needs 85")
    # sliding window of fixed length
    sl = rb.walk_forward(R, train=60, test=20, expanding=False)
    assert all((r["train_end"] - r["train_start"]) == (idx[59] - idx[0]) for _, r in sl["folds"].iterrows())
    # default annualisation is 252
    ch = one["folds"].iloc[0]["chosen"]
    te = R[ch].iloc[65:85].to_numpy()
    assert abs(one["folds"].iloc[0]["oos_sharpe"] - te.mean() / te.std(ddof=1) * math.sqrt(252)) < 1e-12
    print("RO3 walk-forward: training and test dates with an embargo, the one-fold edge, the sliding window and the default annualisation  PASS")


def test_plateau_edges():
    def tbl(ratio_neighbours):
        T = np.zeros((5, 5)); T[2, 2] = 1.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    T[2 + di, 2 + dj] = ratio_neighbours
        return pd.DataFrame(T)
    assert rb.parameter_plateau(tbl(0.7))["verdict"] == "plateau"
    assert rb.parameter_plateau(tbl(0.69))["verdict"] == "hill"
    assert rb.parameter_plateau(tbl(0.3))["verdict"] == "hill"
    assert rb.parameter_plateau(tbl(0.29))["verdict"] == "spike"
    T = tbl(0.4).to_numpy().copy(); T[1, 1] = 0.6; T[3, 3] = 0.5      # of eight neighbours: 0.6 and 0.5 (exactly half of the best) are within reach; the six at 0.4 are not
    r = rb.parameter_plateau(pd.DataFrame(T))
    assert abs(r["share_within"] - 2 / 8) < 1e-12
    small = rb.parameter_plateau(pd.DataFrame(np.array([[0.5, 0.45, 0.2], [0.4, 0.3, 0.1]])))     # best below 1: the share is still defined
    assert np.isfinite(small["share_within"]) and np.isfinite(small["neighbour_ratio"])
    print("RO4 plateau: 0.7 is a plateau, 0.3 a hill, below 0.3 a spike; the share within reach counts neighbours; a best score under 1 is handled  PASS")


def test_event_null():
    p, _ = make_panel(n_days=520, n_stocks=40, seed=4)
    sig = pd.DataFrame(np.random.default_rng(11).random(p.close.shape) < 0.06, index=p.dates, columns=p.tickers) & p.eligible
    a = rb.signal_permutation(p, sig, 5, cost_bp=0.0, n=40, seed=7)
    b = rb.signal_permutation(p, sig, 5, cost_bp=25.0, n=40, seed=7)
    assert abs((a["null_mean_bp_mean"] - b["null_mean_bp_mean"]) - 25.0) < 1e-6 and abs((a["real_mean_bp"] - b["real_mean_bp"]) - 25.0) < 1e-6
    assert a["null_win_rate_p95"] >= a["null_win_rate_mean"] and a["null_mean_bp_p95"] >= a["null_mean_bp_mean"]
    # the null rebuilt: shifts of the whole table of fires, the same random draws
    fwd = p.forward(5).to_numpy(np.float64); el = p.eligible.to_numpy(bool); usable = el & np.isfinite(fwd); fire0 = sig.to_numpy(bool) & el
    gen = np.random.default_rng(7); T = len(fwd); lo = max(2 * 5, 21)
    wn, mn = [], []
    for _ in range(40):
        f = np.roll(fire0, int(gen.integers(lo, T - lo + 1)), axis=0) & usable
        v = fwd[f]
        wn.append((v > 0).mean()); mn.append(v.mean())
    assert abs(a["null_win_rate_p95"] - np.percentile(wn, 95)) < 1e-12 and abs(a["null_mean_bp_p95"] - np.percentile(mn, 95) * 1e4) < 1e-9
    assert abs(a["null_win_rate_mean"] - np.mean(wn)) < 1e-12 and abs(a["p_mean"] - (1 + (np.array(mn) >= a["real_mean_bp"] / 1e4).sum()) / 41) < 1e-12
    # a market that always rises: every fire wins however it is moved, so a tie counts against the signal and p = 1
    idx = pd.bdate_range("2020-01-01", periods=300); cols = [f"S{i:02d}" for i in range(15)]
    up = pd.DataFrame(np.tile(100 * 1.001 ** np.arange(300)[:, None], (1, 15)), index=idx, columns=cols)
    el = pd.DataFrame(True, index=idx, columns=cols); el.iloc[:25] = False
    pu = q.Panel(close=up, eligible=el, volume=up * 1e5, market="T", entry_lag=1)
    s2 = pd.DataFrame(np.random.default_rng(3).random(up.shape) < 0.08, index=idx, columns=cols) & el
    t = rb.signal_permutation(pu, s2, 3, cost_bp=0.0, n=30, seed=1)
    assert t["real_win_rate"] == 1.0 and t["p_win"] == 1.0, t
    print("RO5 the null moves by exactly the cost; ties count against the signal (p = 1 in a market that only rises)  PASS")


def test_decompose_and_shares():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=300)
    g = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    y = 0.5 * g + rng.normal(0, 0.006, 300)
    h60 = pd.Series(np.where(np.arange(300) < 60, rng.normal(0, 0.01, 300), np.nan), index=idx)       # exactly 60 observations: dropped
    h61 = pd.Series(np.where(np.arange(300) < 61, rng.normal(0, 0.01, 300), np.nan), index=idx)       # 61: kept
    d = rb.decompose(y, pd.DataFrame({"g": g, "h60": h60, "h61": h61}))
    assert set(d["factors"]) == {"g", "h61"}, set(d["factors"])
    assert abs(d["unexplained_share"] - (1.0 - d["r2"])) < 1e-15 and 0 < d["unexplained_share"] < 1
    print("RO6 a factor needs more than 60 observations; the unexplained share is 1 - R2  PASS")


def test_drawdown_percentiles_and_regimes():
    rng = np.random.default_rng(6)
    T = 300
    idx = pd.bdate_range("2019-01-01", periods=T)
    x = pd.Series(rng.normal(0.0004, 0.012, T), index=idx)
    d = rb.drawdown_distribution(x, n=500, seed=4)
    paths = rb._stationary_paths(x.to_numpy(), 500, float(round(T ** (1 / 3))) if d["mean_block"] == round(T ** (1 / 3)) else d["mean_block"], np.random.default_rng(4))
    eq = np.cumprod(1.0 + paths, axis=1)
    mdd = (eq / np.maximum(np.maximum.accumulate(eq, axis=1), 1.0) - 1.0).min(axis=1)
    pc = np.percentile(mdd, [5, 25, 50, 75, 95])
    assert all(abs(d[k] - v) < 1e-12 for k, v in zip(("p05", "p25", "p50", "p75", "p95"), pc))
    # regimes: the win rate of the strategy inside each regime
    b = pd.Series(rng.normal(0.0002, 0.01, T), index=idx)
    t = rb.regimes(x, b)["table"]
    mon = (1 + pd.concat([x.rename("s"), b.rename("b")], axis=1)).groupby(idx.to_period("M")).prod() - 1
    up = np.asarray(idx.to_period("M").isin(mon.index[mon["b"] > 0]))
    row = t.loc["benchmark rose in the month"]
    assert abs(row["win_rate"] - (x.to_numpy()[up] > 0).mean()) < 1e-12 and row["bars"] == up.sum()
    print("RO7/8 drawdown percentiles and a regime's win rate equal an independent rebuild  PASS")


def test_holdings_need_holdings():
    p, _ = make_panel(n_days=260, n_stocks=24, seed=5)
    f = pd.DataFrame(np.random.default_rng(1).standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    r = _quiet(lambda: q.backtest_portfolio(p, f, benchmark=None, grid=False))
    _raises(lambda: rb.holdings_summary(p, dataclasses.replace(r, holdings=None)), "no holdings")
    _raises(lambda: rb.brinson(p, dataclasses.replace(r, holdings=None), pd.Series("g", index=p.tickers)), "no holdings or net returns")
    print("RO9 a result with no holdings is refused  PASS")


if __name__ == "__main__":
    for fn in (test_market_relative, test_sign_test, test_walk_forward_windows, test_plateau_edges, test_event_null, test_decompose_and_shares, test_drawdown_percentiles_and_regimes, test_holdings_need_holdings):
        fn()
    print("robustness value tests: all passed")
