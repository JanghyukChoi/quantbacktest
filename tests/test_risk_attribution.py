"""Risk figures, the frozen-date check, regimes, holdings, Brinson and benchmark alignment, against worked answers.

RK1 risk profile   VaR and CVaR of a fixed list, the tail ratio, the longest time under water, the ulcer index and gain-to-pain by hand; the current-volatility VaR scales with volatility
RK2 drawdowns      the resampled maximum drawdowns: a series with a known deepest fall, the realised figure inside the distribution, and the share exceeding a level
RK3 split          the two sides equal direct computations; the Welch t equals scipy's; a bad split raises
RK4 regimes        up and down months, the bear-market flag and the volatility thirds on a constructed benchmark
RK5 holdings       counts, effective N, turnover and the implied holding period of a hand-built book
RK6 Brinson        a worked two-group example, and the identity allocation + selection + interaction = portfolio - benchmark on random books
RK7 benchmark      `align_benchmark` puts a daily index on the strategy's signal dates, so a strategy that holds the index has correlation 1 and beta 1
"""
from __future__ import annotations
import math, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import robustness as rb
from test_argument_checks import _raises
from test_synthetic import make_panel


def _q(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def test_risk_profile():
    idx = pd.bdate_range("2020-01-01", periods=200)
    r = np.tile([0.01, 0.005, -0.002, 0.004, -0.03], 40)                           # five-bar cycle: one big loss in every five
    x = pd.Series(r, index=idx)
    k = rb.risk_profile(x)
    assert abs(k["var_95"] - np.percentile(r, 5)) < 1e-15 and abs(k["var_99"] - np.percentile(r, 1)) < 1e-15
    assert abs(k["cvar_95"] - r[r <= np.percentile(r, 5)].mean()) < 1e-15 and k["cvar_95"] <= k["var_95"]
    assert abs(k["tail_ratio"] - np.percentile(r, 95) / abs(np.percentile(r, 5))) < 1e-15
    assert k["worst_bar"] == -0.03 and k["best_bar"] == 0.01
    eq = np.cumprod(1 + r)
    dd = eq / np.maximum(np.maximum.accumulate(eq), 1.0) - 1
    run = best = 0
    for v in dd < -1e-12:
        run = run + 1 if v else 0
        best = max(best, run)
    assert k["max_dd_bars"] == best and abs(k["share_underwater"] - (dd < -1e-12).mean()) < 1e-15
    assert abs(k["ulcer_index"] - math.sqrt((dd ** 2).mean())) < 1e-15
    assert abs(k["gain_to_pain"] - r.sum() / -r[r < 0].sum()) < 1e-15 and abs(k["total_return"] - (eq[-1] - 1)) < 1e-15
    z = (r - r.mean()) / r.std()
    assert abs(k["skew"] - (z ** 3).mean()) < 1e-12 and abs(k["kurtosis"] - (z ** 4).mean()) < 1e-12
    # current-volatility VaR: the same shape of returns at three times the volatility has three times the "now" VaR; the plain historical figure is by construction unchanged in shape
    rng = np.random.default_rng(0)
    base = rng.standard_t(5, 400) * 0.01
    calm = np.r_[base[:300], base[300:] * 0.3]
    storm = np.r_[base[:300], base[300:] * 3.0]
    kc, ks = rb.risk_profile(pd.Series(calm, index=pd.bdate_range("2020-01-01", periods=400))), rb.risk_profile(pd.Series(storm, index=pd.bdate_range("2020-01-01", periods=400)))
    assert ks["vol_now"] > 4 * kc["vol_now"] and ks["var_95_now"] < kc["var_95_now"] < 0, (ks["var_95_now"], kc["var_95_now"])      # after a storm today's risk is larger than after a calm
    rr = np.random.default_rng(9).standard_t(4, 1000) * 0.01
    kk = rb.risk_profile(pd.Series(rr, index=pd.bdate_range("2018-01-01", periods=1000)))
    lo5, lo1 = np.percentile(rr, 5), np.percentile(rr, 1)
    assert abs(kk["var_95"] - lo5) < 1e-15 and abs(kk["cvar_95"] - rr[rr <= lo5].mean()) < 1e-15 and kk["cvar_95"] < kk["var_95"]       # the shortfall is deeper than the quantile
    assert abs(kk["cvar_99"] - rr[rr <= lo1].mean()) < 1e-15 and kk["cvar_99"] < kk["var_99"] < kk["var_95"]
    assert kk["var_95_normal"] > kk["var_95"] - 0.01                                                                       # t(4) has the fatter tail: the normal figure is not more negative by much
    # the starting capital is a peak: a loss on the very first bar followed by a slow climb that never regains 1 is under water for the whole sample
    first = pd.Series(np.r_[-0.10, np.full(119, 0.0005)], index=pd.bdate_range("2020-01-01", periods=120))
    kf = rb.risk_profile(first)
    assert kf["max_dd_bars"] == 120 and kf["share_underwater"] == 1.0 and kf["ulcer_index"] > 0.05, kf
    # the current-volatility VaR by a plain loop: the volatility used for bar t is known before bar t
    lam = 0.94
    v = np.empty(len(rr)); v[0] = np.var(rr[:20], ddof=1)
    for t in range(1, len(rr)):
        v[t] = lam * v[t - 1] + (1 - lam) * rr[t - 1] ** 2
    zs = rr / np.sqrt(v)
    nxt = math.sqrt(lam * v[-1] + (1 - lam) * rr[-1] ** 2)
    assert abs(kk["var_95_now"] - nxt * np.percentile(zs, 5)) < 1e-15 and abs(kk["var_99_now"] - nxt * np.percentile(zs, 1)) < 1e-15 and abs(kk["vol_now"] - nxt * math.sqrt(252)) < 1e-15
    _raises(lambda: rb.risk_profile(x.iloc[:30]), "at least 60")
    _raises(lambda: rb.risk_profile(x, ewma_lambda=1.2), "ewma_lambda")
    print(f"RK1 VaR 95 {k['var_95']:.4f}, CVaR 95 {k['cvar_95']:.4f}, tail ratio, {k['max_dd_bars']} bars under water, ulcer index and gain-to-pain equal hand computations; today's VaR is larger after a storm than after a calm  PASS")


def test_drawdown_distribution():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=500)
    x = pd.Series(rng.normal(0.0004, 0.01, 500), index=idx)
    d = rb.drawdown_distribution(x, n=400, seed=2)
    assert abs(d["realized"] - rb._max_drawdown(x.to_numpy())) < 1e-15
    assert d["p05"] <= d["p25"] <= d["p50"] <= d["p75"] <= d["p95"] <= 0
    assert d["p05"] <= d["realized"] <= d["p95"] + 0.05                                          # one draw from the distribution it came from
    # a certain loss of 50 percent on one bar: every resampled history that contains the bar has that drawdown
    y = pd.Series(np.r_[np.zeros(99), -0.5, np.zeros(100)], index=pd.bdate_range("2020-01-01", periods=200))
    dy = rb.drawdown_distribution(y, n=200, seed=3, mean_block=1e12)                              # no jumps: every path is a rotation of the sample, so it contains the fall once
    assert abs(dy["realized"] + 0.5) < 1e-12 and abs(dy["p05"] + 0.5) < 1e-12 and dy["p_exceed_50"] == 0.0 and dy["p_exceed_30"] == 1.0
    assert rb.drawdown_distribution(x, n=400, seed=2) == d                                         # same seed, same answer
    _raises(lambda: rb.drawdown_distribution(x, n=10), "at least 100")
    _raises(lambda: rb.drawdown_distribution(x.iloc[:20]), "at least 60")
    print(f"RK2 the realised drawdown {d['realized']:.3f} lies inside the resampled distribution ({d['p05']:.3f} to {d['p95']:.3f}); a single certain 50 percent fall is found in every history  PASS")


def test_split():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2019-01-01", periods=600)
    x = pd.Series(np.r_[rng.normal(0.001, 0.01, 300), rng.normal(0.0, 0.012, 300)], index=idx)
    r = rb.in_out_of_sample(x, idx[300])
    a, b = x.iloc[:300], x.iloc[300:]
    assert r["in_sample"]["bars"] == 300 and r["out_of_sample"]["bars"] == 300 and r["split"] == str(idx[300].date())
    assert abs(r["in_sample"]["mean_annual"] - a.mean() * 252) < 1e-12 and abs(r["out_of_sample"]["sharpe"] - b.mean() / b.std(ddof=1) * math.sqrt(252)) < 1e-12
    t_hand = (b.mean() - a.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))                         # Welch, by hand
    assert abs(r["mean_diff_t"] - t_hand) < 1e-12 and abs(r["mean_diff_p"] - math.erfc(abs(t_hand) / math.sqrt(2))) < 1e-12
    try:
        from scipy import stats
        assert abs(r["mean_diff_t"] - stats.ttest_ind(b, a, equal_var=False).statistic) < 1e-12                                   # and scipy's, when installed
    except ImportError:
        pass
    assert abs(r["sharpe_ratio"] - r["out_of_sample"]["sharpe"] / r["in_sample"]["sharpe"]) < 1e-15
    _raises(lambda: rb.in_out_of_sample(x, idx[10]), "at least 30")
    _raises(lambda: rb.in_out_of_sample(x, idx[-5]), "at least 30")
    print(f"RK3 both sides equal direct computations and the Welch t equals scipy's ({r['mean_diff_t']:.2f}); a split leaving a thin side raises  PASS")


def test_regimes():
    idx = pd.bdate_range("2020-01-01", periods=400)
    b = np.r_[np.full(100, 0.002), np.full(100, -0.004), np.full(200, 0.001)]               # 100 up bars, then a long fall (about 33 percent), then a recovery
    s = np.r_[np.full(100, 0.001), np.full(100, 0.003), np.full(200, 0.0)]                  # the strategy gains most while the benchmark falls
    rg = rb.regimes(pd.Series(s, index=idx), pd.Series(b, index=idx), periods_per_year=252)
    t = rg["table"]
    assert rg["sample_has_bear"] is True and abs(rg["max_benchmark_drawdown"] - ((1.002 ** 100 * 0.996 ** 100) / 1.002 ** 100 - 1)) < 1e-9
    bear = [i for i in t.index if i.startswith("bear market")][0]
    assert t.loc[bear, "bars"] + t.loc["other bars", "bars"] == 400 and t.loc[bear, "bars"] > 100                      # the recovery to within 20 percent of the peak takes long: those bars still count
    assert abs(t.loc[bear, "mean_annual"] - s[np.cumprod(1 + b) / np.maximum(np.maximum.accumulate(np.cumprod(1 + b)), 1.0) - 1 <= -0.2].mean() * 252) < 1e-12
    calm = rb.regimes(pd.Series(0.0005, index=idx), pd.Series(np.tile([0.003, -0.002], 200), index=idx))
    assert calm["sample_has_bear"] is False
    _raises(lambda: rb.regimes(pd.Series(0.0, index=idx[:50]), pd.Series(0.0, index=idx[:50])), "at least 120")
    print("RK4 up and down months, the bear flag (-33 percent benchmark fall), the bear and other rows add to all bars, and a calm sample says it has no bear market  PASS")


def test_holdings():
    T, N = 120, 12
    idx = pd.bdate_range("2020-01-01", periods=T)
    names = [f"S{i:02d}" for i in range(N)]
    p = q.Panel(close=pd.DataFrame(100.0, index=idx, columns=names), eligible=pd.DataFrame(True, index=idx, columns=names))
    W = pd.DataFrame(0.0, index=idx, columns=names)
    W.iloc[10:, 0] = 0.5; W.iloc[10:, 1] = 0.5            # long two names, half each
    W.iloc[10:, 2] = -1.0                                 # short one name
    W.iloc[60:, 0] = 0.25; W.iloc[60:, 3] = 0.25          # at bar 60 half of name 0's weight moves to name 3
    r = _q(lambda: q.backtest_weights(p, W, spread_bp=0.0, benchmark=None, check_universe=False))
    hs = rb.holdings_summary(p, r)
    n_first, n_second = 50, len(r.net_returns) - 60
    assert abs(hs["avg_long"] - (n_first * 2 + n_second * 3) / (n_first + n_second)) < 1e-12 and hs["effective_n_short"] == 1.0 and hs["max_weight_short"] == 1.0
    n = len(r.net_returns)
    H = np.asarray(r.holdings)[:n]
    d = np.abs(np.diff(H, axis=0))
    assert abs(hs["turnover_per_bar"] - 0.5 * d.sum(axis=1).mean()) < 1e-15 and abs(hs["turnover_per_month"] - 21 * hs["turnover_per_bar"]) < 1e-15
    assert abs(hs["avg_trade"] - d[d > 1e-9].mean()) < 1e-15 and abs(hs["trades_per_year"] - (d > 1e-9).sum() / (n / 252)) < 1e-12
    assert abs(hs["implied_holding_bars"] - np.abs(H).sum(axis=1).mean() / (0.5 * d.sum(axis=1).mean())) < 1e-9
    assert list(hs["top_holdings"])[0] == "S02" and hs["top_holdings"]["S02"] < 0                    # the short is the largest average weight
    # effective N by hand: two equal long positions (0.5, 0.5) from bar 10 to 59 give 2; from bar 60 the weights are 0.25, 0.5, 0.25 of 1.0 give 1 / (0.0625 + 0.25 + 0.0625) = 2.667
    n_first = 50 - 0                                              # bars 10..59 in the first regime, within the first n rows
    n_second = n - 60
    assert abs(hs["effective_n_long"] - (n_first * 2.0 + n_second * (1 / 0.375)) / (n_first + n_second)) < 1e-9, hs["effective_n_long"]
    # a leg that does not sum to 1: weights 0.3 and 0.1 are 0.75 and 0.25 of the leg, effective N 1 / (0.5625 + 0.0625) = 1.6 (unnormalised it would be 10)
    W2 = pd.DataFrame(0.0, index=idx, columns=names); W2.iloc[10:, 0] = 0.3; W2.iloc[10:, 1] = 0.1
    h2 = rb.holdings_summary(p, _q(lambda: q.backtest_weights(p, W2, spread_bp=0.0, benchmark=None, check_universe=False)))
    assert abs(h2["effective_n_long"] - 1.6) < 1e-12 and abs(h2["max_weight_long"] - 0.75) < 1e-12 and h2["avg_short"] == 0.0
    print(f"RK5 positions, effective N, turnover per bar {hs['turnover_per_bar']:.4f}, trades per year and the implied holding period equal direct computations  PASS")


def test_brinson():
    T, N = 60, 10
    idx = pd.bdate_range("2020-01-01", periods=T)
    names = [f"S{i}" for i in range(N)]
    rng = np.random.default_rng(5)
    ret = pd.DataFrame(rng.normal(0.0005, 0.01, (T, N)), index=idx, columns=names)
    close = 100 * (1 + ret).cumprod()
    cap = pd.DataFrame(np.tile(np.arange(1, N + 1) * 1e6, (T, 1)), index=idx, columns=names)
    p = q.Panel(close=close, eligible=pd.DataFrame(True, index=idx, columns=names), mkt_cap=cap, market="T")
    groups = pd.Series({n: ("tech" if i < 4 else "bank") for i, n in enumerate(names)})
    # a long-only book with half in cash, over-weight tech
    W = pd.DataFrame(0.0, index=idx, columns=names); W.iloc[:, 0] = 0.3; W.iloc[:, 1] = 0.2; W.iloc[:, 7] = 0.1
    r = _q(lambda: q.backtest_weights(p, W, spread_bp=0.0, benchmark=None, check_universe=False))
    br = rb.brinson(p, r, groups)
    assert br["identity_gap"] < 1e-12, br["identity_gap"]
    n = len(r.net_returns)
    H = np.asarray(r.holdings)[:n]
    from pitbacktest.portfolio import _forward_arrays
    fwd = _forward_arrays(p, False, None)[0][:n]                                                          # the one-bar returns the engine used (panel.forward is float32)
    wb = cap.to_numpy(float)[:n]; wb = wb / wb.sum(axis=1, keepdims=True)
    gross_p, gross_b = (H * fwd).sum(axis=1), (wb * fwd).sum(axis=1)
    assert abs(br["excess_gross_annual"] - (gross_p - gross_b).sum() / (n / 252)) < 1e-10                    # the three effects add up to the gross return gap
    # one bar by hand: tech = S0..S3, bank = S4..S9; Wp tech 0.5, bank 0.1, cash 0.4; Wb tech 10/55, bank 45/55
    t = 20
    rr = fwd[t]
    Wb_t, Wb_b = wb[t, :4].sum(), wb[t, 4:].sum()
    Rb_t, Rb_b = (wb[t, :4] * rr[:4]).sum() / Wb_t, (wb[t, 4:] * rr[4:]).sum() / Wb_b
    R = Wb_t * Rb_t + Wb_b * Rb_b
    Rp_t, Rp_b = (H[t, :4] * rr[:4]).sum() / 0.5, (H[t, 4:] * rr[4:]).sum() / 0.1
    a_tech = (0.5 - Wb_t) * (Rb_t - R)
    s_tech = Wb_t * (Rp_t - Rb_t)
    assert abs(a_tech + s_tech + (0.5 - Wb_t) * (Rp_t - Rb_t) + (0.1 - Wb_b) * (Rb_b - R) + Wb_b * (Rp_b - Rb_b) + (0.1 - Wb_b) * (Rp_b - Rb_b) + 0.4 * (0 - R) * 1.0 - (gross_p[t] - R)) < 1e-12
    # every group's three effects against a plain loop over the bars
    tech, bank = np.arange(10) < 4, np.arange(10) >= 4
    ex = {k: {"tech": 0.0, "bank": 0.0, "(cash)": 0.0, "(no group)": 0.0} for k in ("a", "s", "i")}
    for t_ in range(n):
        Wp_ = {"tech": H[t_, tech].sum(), "bank": H[t_, bank].sum(), "(cash)": 1 - H[t_].sum()}
        Wb_ = {"tech": wb[t_, tech].sum(), "bank": wb[t_, bank].sum(), "(cash)": 0.0}
        Rb_ = {"tech": (wb[t_, tech] * fwd[t_, tech]).sum() / Wb_["tech"], "bank": (wb[t_, bank] * fwd[t_, bank]).sum() / Wb_["bank"], "(cash)": 0.0}
        Rp_ = {"tech": (H[t_, tech] * fwd[t_, tech]).sum() / Wp_["tech"] if Wp_["tech"] > 0 else 0.0, "bank": (H[t_, bank] * fwd[t_, bank]).sum() / Wp_["bank"] if Wp_["bank"] > 0 else 0.0, "(cash)": 0.0}
        Rtot = sum(Wb_[g] * Rb_[g] for g in Wb_)
        for g in Wp_:
            ex["a"][g] += (Wp_[g] - Wb_[g]) * (Rb_[g] - Rtot)
            ex["s"][g] += Wb_[g] * (Rp_[g] - Rb_[g])
            ex["i"][g] += (Wp_[g] - Wb_[g]) * (Rp_[g] - Rb_[g])
    for g in ("tech", "bank", "(cash)"):
        assert abs(br["allocation"][g] - ex["a"][g] / (n / 252)) < 1e-12 and abs(br["selection"][g] - ex["s"][g] / (n / 252)) < 1e-12 and abs(br["interaction"][g] - ex["i"][g] / (n / 252)) < 1e-12, g
    assert br["allocation"]["(cash)"] != 0.0                                                          # holding cash against a rising or falling benchmark is an allocation decision
    shorted = W.copy(); shorted.iloc[:, 5] = -0.1
    rs = _q(lambda: q.backtest_weights(p, shorted, spread_bp=0.0, benchmark=None, check_universe=False))
    _raises(lambda: rb.brinson(p, rs, groups), "short")
    lev = W * 3
    _raises(lambda: rb.brinson(p, _q(lambda: q.backtest_weights(p, lev, spread_bp=0.0, benchmark=None, check_universe=False)), groups), "more than 1")
    _raises(lambda: rb.brinson(p, r, ["a"]), "Series")
    _raises(lambda: rb.brinson(p, r, groups, benchmark="x"), "benchmark")
    print(f"RK6 allocation {br['total']['allocation']:+.4f} + selection {br['total']['selection']:+.4f} + interaction {br['total']['interaction']:+.4f} = {br['excess_gross_annual']:+.4f} a year equals the gross gap; one bar equals a hand computation  PASS")


def test_align_benchmark():
    p, rng = make_panel(n_days=500, n_stocks=40)
    daily = pd.Series(rng.normal(0.0003, 0.01, 500), index=p.dates)
    al = rb.align_benchmark(p, daily)
    assert abs(al.iloc[10] - daily.iloc[10 + p.entry_lag + 1]) < 1e-15 and al.iloc[-1] != al.iloc[-1] and al.iloc[-(p.entry_lag + 1):].isna().all()
    # a strategy that holds one name every day: its return series is the name's daily returns moved back by lag + 1 rows; aligned, correlation and beta against that name are 1
    name = p.tickers[0]
    W = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); W.iloc[60:, 0] = 1.0
    r = _q(lambda: q.backtest_weights(p, W, spread_bp=0.0, benchmark=None, check_universe=False))
    nm = p.close[name].pct_change()
    m = rb.market_relative(r.net_returns.iloc[60:], rb.align_benchmark(p, nm).reindex(r.net_returns.index).iloc[60:])
    assert abs(m["correlation"] - 1.0) < 1e-9 and abs(m["beta"] - 1.0) < 1e-9
    wrong = rb.market_relative(r.net_returns.iloc[60:], nm.reindex(r.net_returns.index).iloc[60:])     # not moved back: out of step by two bars
    assert abs(wrong["correlation"]) < 0.3
    _raises(lambda: rb.align_benchmark(p, daily.to_numpy()), "Series")
    print(f"RK7 an index moved onto the signal dates has correlation {m['correlation']:.3f} with the strategy that holds it; left in step with the dates it has {wrong['correlation']:.2f}  PASS")


if __name__ == "__main__":
    test_risk_profile()
    test_drawdown_distribution()
    test_split()
    test_regimes()
    test_holdings()
    test_brinson()
    test_align_benchmark()
    print("risk and attribution tests: all passed")
