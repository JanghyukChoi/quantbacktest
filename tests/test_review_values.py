"""The numbers in an event review, against a plain pandas computation of the same definitions.

An automatic mutation run found that most of the arithmetic in `review_event` (percent scalings, the win test, the streak, the Wilson interval, the buckets) could be changed without any test
noticing: the other tests look at structure, not at values. These recompute every table from the price panel and the signal with a loop-free but independent route.

RV9  cost table, yearly table, streak and segments equal the plain computation
RV10 Wilson interval equals the closed form and statsmodels
RV11 a missing value in a signal is no fire; the settings table and PBO switch on at their thresholds
RV12 ledger and grid together, the benchmark coverage floor, the event defaults, a pool of exactly twenty securities
"""
from __future__ import annotations
import math, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.review import _wilson, event_weights
from test_synthetic import make_panel

Z = 1.959963984540054


def _wil(k, n):
    p = k / n
    c = (k + Z * Z / 2) / (n + Z * Z)
    h = Z * math.sqrt(n) / (n + Z * Z) * math.sqrt(p * (1 - p) + Z * Z / (4 * n))
    return c - h, c + h


def _longest(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def test_event_review_values():
    p, _ = make_panel(n_days=520, n_stocks=40, seed=4)
    rng = np.random.default_rng(11)
    sig = pd.DataFrame(rng.random(p.close.shape) < 0.03, index=p.dates, columns=p.tickers) & p.eligible
    h, cost, lag = 5, 20.0, p.entry_lag
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rv = q.review_event(p, sig, horizons=(h,), cost_bp=cost, cost_grid=(0.0, 20.0, 60.0), n_permutations=0, as_portfolio=False)
    fwd = (p.close.shift(-(lag + h)) / p.close.shift(-lag) - 1.0)
    fire = sig & p.eligible & np.isfinite(fwd)
    ii, jj = np.nonzero(fire.to_numpy())
    raw = fwd.to_numpy()[ii, jj]
    dates = p.dates[ii]
    assert rv.horizon == h and len(raw) > 100
    pool = (p.eligible & np.isfinite(fwd)).to_numpy()
    for c, row in rv.cost_table.iterrows():
        r = raw - c / 1e4
        assert abs(row["win_rate"] - (r > 0).mean() * 100) < 1e-9 and abs(row["mean_bp"] - r.mean() * 1e4) < 1e-6 and abs(row["median_bp"] - np.median(r) * 1e4) < 1e-6
        days = [i for i in np.unique(ii) if pool[i].sum() >= 20]
        kd = {i: int(fire.to_numpy()[i].sum()) for i in days}
        base = sum(kd[i] * ((fwd.to_numpy()[i][pool[i]] - c / 1e4) > 0).mean() for i in days) / sum(kd.values())
        assert abs(row["base_rate"] - base * 100) < 1e-9 and abs(row["lift_pp"] - (row["win_rate"] - base * 100)) < 1e-9
    r_net = raw - cost / 1e4
    day = pd.Series(r_net).groupby(ii).mean()
    assert rv.streak["longest_losing_streak"] == _longest((day <= 0).tolist()) and rv.streak["n_trades"] == len(r_net)
    assert abs(rv.streak["win_rate_pct"] - (r_net > 0).mean() * 100) < 1e-9
    for y, row in rv.yearly.iterrows():
        m = np.asarray(dates.year == y)
        k, n = int((r_net[m] > 0).sum()), int(m.sum())
        lo, hi = _wil(k, n)
        assert row["n_trades"] == n and abs(row["win_rate"] - k / n * 100) < 1e-9 and abs(row["mean_bp"] - r_net[m].mean() * 1e4) < 1e-6
        assert abs(row["win_lo"] - lo * 100) < 1e-6 and abs(row["win_hi"] - hi * 100) < 1e-6, (y, row["win_lo"], lo * 100)
    for label, vals in (("liquidity (30-bar average traded value)", p.adv(30)), ("market capitalisation", p.mkt_cap.where(p.mkt_cap > 0))):
        rk = vals.where(p.eligible).rank(axis=1, pct=True).to_numpy()
        rr_ = rk[ii, jj]
        terc = np.where(np.isfinite(rr_), np.clip(np.ceil(np.nan_to_num(rr_) * 3).astype(int) - 1, 0, 2), -1)          # unknown characteristic: in no third
        seg = rv.segments[label]
        for k, name in enumerate(("lowest third", "middle third", "highest third")):
            m = terc == k
            if m.sum() < 20:
                assert name not in seg.index
                continue
            rr = r_net[m]
            row = seg.loc[name]
            lo, hi = _wil(int((rr > 0).sum()), int(m.sum()))
            assert row["n_trades"] == m.sum() and abs(row["share_of_fires"] - m.mean() * 100) < 1e-9 and abs(row["win_rate"] - (rr > 0).mean() * 100) < 1e-9
            got = (row["win_lo"], row["win_hi"], row["mean_bp"], row["median_bp"])
            want = (lo * 100, hi * 100, rr.mean() * 1e4, np.median(rr) * 1e4)
            assert all(abs(g - w) < 1e-4 for g, w in zip(got, want)), (label, name, got, want)
    ht = rv.horizons_table.loc[h]
    lo, hi = _wil(int(round(ht["win_rate"] / 100 * ht["n_trades"])), int(ht["n_trades"]))
    assert abs(ht["win_lo"] - lo * 100) < 1e-6 and abs(ht["win_hi"] - hi * 100) < 1e-6
    txt = rv.to_json()
    assert txt.splitlines()[1].startswith("  ") and not txt.splitlines()[1].startswith("   ")
    print("RV9 cost table, yearly table, streak, segments and horizon intervals equal a plain recomputation  PASS")


def test_wilson():
    for k, n in ((0, 1), (1, 1), (8, 10), (3, 7), (50, 100), (99, 100), (0, 30), (30, 30), (1234, 5000)):
        lo, hi = _wilson(k, n)
        elo, ehi = _wil(k, n)
        assert abs(lo - max(elo, 0.0)) < 1e-12 and abs(hi - min(ehi, 1.0)) < 1e-12 or (k in (0, n)), (k, n, lo, hi, elo, ehi)
    lo, hi = _wilson(8, 10)
    assert abs(lo - 0.4901625) < 1e-6 and abs(hi - 0.9433178) < 1e-6                      # the usual textbook values for 8 of 10
    try:
        from statsmodels.stats.proportion import proportion_confint
    except ImportError:
        print("RV10 (statsmodels not installed: closed form and textbook values only)  PASS")
        return
    for k, n in ((3, 7), (50, 100), (99, 100), (1234, 5000)):
        a, b = proportion_confint(k, n, alpha=0.05, method="wilson")
        lo, hi = _wilson(k, n)
        assert abs(lo - a) < 1e-9 and abs(hi - b) < 1e-9
    print("RV10 the Wilson interval equals the closed form, the textbook values and statsmodels  PASS")


def test_thresholds_and_missing():
    p, _ = make_panel(n_days=300, n_stocks=30, seed=6)
    sig = pd.DataFrame(False, index=p.dates, columns=p.tickers)
    sig.iloc[100, :4] = True
    sig = sig.astype(object)
    sig.iloc[110, 5] = np.nan                                                     # unknown is not a fire
    sig.iloc[120, 6] = None
    W = event_weights(p, sig, 5)
    assert W.iloc[110].abs().sum() == 0 and W.iloc[120].abs().sum() == 0 and abs(W.iloc[100].sum() - 1.0 / 5) < 1e-12 and (W.iloc[100, :4] > 0).all()

    rng = np.random.default_rng(2)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = q.review_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5, n_permutations=0, make_factor=lambda panel: f, grid={"hold": [1, 3, 5]})
        assert a.plateau is not None and len(a.grid_table) == 3, "a settings table of exactly three cells gets a plateau"
        assert a.pbo is None and a.deflated is not None and a.deflated["source"] == "grid", "three settings: no PBO"
        b = q.review_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5, n_permutations=0, make_factor=lambda panel: f, grid={"hold": [1, 2, 3, 5]})
        assert b.pbo is not None, "four settings and enough bars: PBO"
        c = q.review_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5, n_permutations=0)
        assert not any("ledger" in n for n in c.notes), c.notes
    print("RV11 a missing signal value is no fire; three settings get a plateau but no PBO, four get both; no ledger means no ledger note  PASS")


def test_ledger_benchmark_and_defaults():
    import json, tempfile
    p, _ = make_panel(n_days=300, n_stocks=30, seed=6)
    rng = np.random.default_rng(2)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    kw = dict(long_q=0.2, short_q=0.2, hold=5, n_permutations=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with tempfile.TemporaryDirectory(prefix="rv12-") as d:
            led = q.Ledger(d)
            g = q.review_portfolio(p, f, make_factor=lambda panel: f, grid={"hold": [1, 3, 5]}, ledger=led, family="fam", **kw)
            assert g.deflated["source"] == "grid", "the grid's own deflated Sharpe is kept when a ledger is also given"
        with tempfile.TemporaryDirectory(prefix="rv12b-") as d:
            led = q.Ledger(d)
            for h in (1, 3, 5):
                q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=h, ledger=led, family="fam2", name=f"h{h}")
            l = q.review_portfolio(p, f, ledger=led, family="fam2", **kw)
            assert l.deflated is not None and l.deflated["source"] == "ledger"
        assert json.loads(g.to_json())["schema_version"] == 1
        # a benchmark must cover at least 60 of the strategy's dates
        def bench(k):
            b = pd.Series(np.nan, index=p.dates)
            b.iloc[102:102 + k] = rng.normal(0.0003, 0.01, k)
            return b
        base = q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5)
        q.review_portfolio(p, f, benchmark_returns=bench(60), **kw)
        try:
            q.review_portfolio(p, f, benchmark_returns=bench(59), **kw)
        except ValueError as e:
            assert "at least 60" in str(e), e
        else:
            raise AssertionError("a benchmark covering 59 dates must be refused")
        # the defaults of the event review: 200 shifted copies, the portfolio view reported with schema 1
        sig = pd.DataFrame(rng.random(p.close.shape) < 0.03, index=p.dates, columns=p.tickers) & p.eligible
        e = q.review_event(p, sig, horizons=(5,))
        assert e.permutation["n"] == 200 and json.loads(e.to_json())["schema_version"] == 1
        # exactly twenty securities: the pool for the random-pick rate is big enough (the engine's rule is 20 or more)
        p20, _ = make_panel(n_days=300, n_stocks=20, seed=8)
        s20 = pd.DataFrame(np.random.default_rng(1).random(p20.close.shape) < 0.05, index=p20.dates, columns=p20.tickers) & p20.eligible
        e20 = q.review_event(p20, s20, horizons=(5,), n_permutations=0, as_portfolio=False)
        assert e20.cost_table["base_rate"].iloc[0] > 0 and np.isfinite(e20.cost_table["lift_pp"]).all()
    print("RV12 a grid keeps its deflated Sharpe next to a ledger, a ledger alone supplies it, 60 benchmark dates pass and 59 do not, the event defaults, a pool of exactly 20  PASS")


if __name__ == "__main__":
    test_event_review_values(); test_wilson(); test_thresholds_and_missing(); test_ledger_benchmark_and_defaults()
    print("review value tests: all passed")
