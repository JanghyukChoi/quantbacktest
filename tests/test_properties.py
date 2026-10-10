"""Property-based tests (Hypothesis): the invariants of the metamorphic tests and of the statistics, over inputs a person would not think to try.

Hypothesis draws the shape of the market, the holding period, the entry lag, the costs, the quantiles, whether securities are delisted and whether funding exists; builds a random
panel from a seed; and checks relations that must hold for every one of them. When it finds a counter-example it shrinks it to the smallest one. The number of examples per property is
PITBT_EXAMPLES (default 25: a minute or two; a deep run is 300 or more). `derandomize` makes a given count reproducible.

PP1 portfolio   returns are finite and above -100 percent, a higher spread never raises a bar, the legs never hold more than their capital, gross minus costs is net
PP2 causality   cutting the data after a date, and scrambling the factor and the eligibility after it, change nothing before it, for any lag and holding period
PP3 order       the order of the securities changes nothing
PP4 weights     a blend of two books at zero cost is the weighted sum of their returns; a ruin stops the account for good
PP5 statistics  p-value corrections, risk figures, market-relative figures, the plateau, the walk-forward count and the JSON conversion obey their bounds and identities
PP6 events      the weights an event signal makes sum to what was fired, the longest run and the intervals are what the plain definition says
Skipped with a note when `hypothesis` is not installed."""
from __future__ import annotations
import json, math, os, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import robustness as rb
from pitbacktest.export import jsonable
from pitbacktest.portfolio import stop_at_ruin
from pitbacktest.review import _longest_run, _wilson, event_weights

try:
    from hypothesis import HealthCheck, given, settings, strategies as st
except ImportError:                                                       # pragma: no cover
    given = None

EXAMPLES = int(os.environ.get("PITBT_EXAMPLES", "25"))
FAST = os.environ.get("PITBT_FAST") == "1"


def _world(seed, T, N, lag, funding, delist, vol=0.02):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T)
    cols = [f"S{i:02d}" for i in range(N)]
    ret = rng.normal(0.0003, 0.01, (T, 1)) + rng.normal(0, vol, (T, N))
    close = pd.DataFrame(100 * np.exp(ret.cumsum(0)), index=idx, columns=cols)
    el = pd.DataFrame(rng.random((T, N)) < 0.9, index=idx, columns=cols)
    el.iloc[:25] = False
    da = None
    if delist:
        da = pd.DataFrame(False, index=idx, columns=cols)
        for j in rng.choice(N, max(1, N // 6), replace=False):
            k = int(rng.integers(T // 3, 2 * T // 3))
            close.iloc[k + 1:, j] = np.nan
            da.iloc[k, j] = True
    fund = pd.DataFrame(rng.normal(0.0001, 0.0003, (T, N)), index=idx, columns=cols) if funding else None
    p = q.Panel(close=close, eligible=el, volume=pd.DataFrame(rng.lognormal(12, 1, (T, N)), index=idx, columns=cols), mkt_cap=close * 1e7, funding=fund, delist_after=da, entry_lag=lag, market="T")
    f = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)               # a raw factor: values also where a security is not eligible
    return p, f, rng


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


if given is not None:
    CFG = settings(max_examples=EXAMPLES, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))
    world = st.tuples(st.integers(0, 10 ** 6), st.integers(110, 230), st.integers(12, 26), st.integers(0, 2), st.booleans(), st.booleans())
    params = st.tuples(st.sampled_from([0.1, 0.2, 0.3, 0.5]), st.integers(1, 8), st.floats(0.0, 60.0), st.floats(0.0, 30.0))

    @CFG
    @given(world, params)
    def prop_portfolio(w, pr):
        seed, T, N, lag, fund, dl = w
        qq, hold, spread, sell = pr
        p, f, _ = _world(seed, T, N, lag, fund, dl)
        r = _quiet(lambda: q.backtest_portfolio(p, f, long_q=qq, short_q=qq, hold=hold, spread_bp=spread, sell_bp=sell, benchmark=None, grid=False))
        net = r.net_returns.to_numpy()
        assert np.isfinite(net).all() and (net >= -1.0 - 1e-12).all()
        H = np.asarray(r.holdings)
        assert (np.where(H > 0, H, 0).sum(axis=1) <= 1.0 + 1e-9).all() and (np.where(H < 0, -H, 0).sum(axis=1) <= 1.0 + 1e-9).all()      # a leg never holds more than its capital
        hi = _quiet(lambda: q.backtest_portfolio(p, f, long_q=qq, short_q=qq, hold=hold, spread_bp=spread + 25.0, sell_bp=sell, benchmark=None, grid=False)).net_returns.to_numpy()
        assert (hi <= net + 1e-14).all()                                                      # a higher spread never raises a bar
        at = rb.attribution(p, r)
        if not r.metrics["ruined"] and "freeze_markdown_annual_bp" not in r.metrics:
            assert abs(at["waterfall"]["residual_bp"]) < 1e-5 and abs(at["long_annual"] + at["short_annual"] - at["gross_annual"]) < 1e-10
        assert 0.0 <= r.metrics["turnover_daily"] <= 2.0 + 1e-9

    @CFG
    @given(world, params, st.integers(0, 10 ** 6))
    def prop_causality(w, pr, s2):
        seed, T, N, lag, fund, dl = w
        qq, hold, spread, sell = pr
        p, f, rng = _world(seed, T, N, lag, fund, dl)
        cut = int(np.random.default_rng(s2).integers(T // 2, T - 5))
        kw = dict(long_q=qq, short_q=qq, hold=hold, spread_bp=spread, sell_bp=sell, benchmark=None, grid=False)
        full = _quiet(lambda: q.backtest_portfolio(p, f, **kw))
        part = _quiet(lambda: q.backtest_portfolio(p.truncate(p.dates[cut]), f.iloc[:cut + 1], **kw))
        n = len(part.net_returns)
        assert np.allclose(full.net_returns.iloc[:n].to_numpy(), part.net_returns.to_numpy(), atol=1e-13, rtol=0)
        f2 = f.copy()
        f2.iloc[cut + 1:] = rng.standard_normal(f2.iloc[cut + 1:].shape)
        el2 = p.eligible.copy()
        el2.iloc[cut + 1:] = (rng.random(el2.iloc[cut + 1:].shape) < 0.5) | (np.arange(N) < 10)         # scrambled, but the panel still has the ten securities it insists on
        p2 = q.Panel(close=p.close, eligible=el2, volume=p.volume, mkt_cap=p.mkt_cap, funding=p.funding, delist_after=p.delist_after, entry_lag=lag, market="T")
        b = _quiet(lambda: q.backtest_portfolio(p2, f2, **kw))
        assert np.allclose(np.asarray(full.holdings)[:cut + 1], np.asarray(b.holdings)[:cut + 1], atol=1e-15, rtol=0)             # no position held up to the date saw what came after it

    @CFG
    @given(world, params, st.integers(0, 10 ** 6))
    def prop_order(w, pr, s2):
        seed, T, N, lag, fund, dl = w
        qq, hold, spread, sell = pr
        p, f, _ = _world(seed, T, N, lag, fund, dl)
        cols = list(p.tickers)
        np.random.default_rng(s2).shuffle(cols)
        r_ = lambda d: None if d is None else d[cols]                                          # noqa: E731
        p2 = q.Panel(close=p.close[cols], eligible=p.eligible[cols], volume=r_(p.volume), mkt_cap=r_(p.mkt_cap), funding=r_(p.funding), delist_after=r_(p.delist_after), entry_lag=lag, market="T")
        kw = dict(long_q=qq, short_q=qq, hold=hold, spread_bp=spread, sell_bp=sell, benchmark=None, grid=False)
        a = _quiet(lambda: q.backtest_portfolio(p, f, **kw)).net_returns.to_numpy()
        b = _quiet(lambda: q.backtest_portfolio(p2, f[cols], **kw)).net_returns.to_numpy()
        assert np.allclose(a, b, atol=1e-13, rtol=0)

    @CFG
    @given(world, st.floats(0.05, 0.95), st.floats(0.5, 40.0))
    def prop_weights_blend_and_ruin(w, mix, lev):
        seed, T, N, lag, fund, dl = w
        p, f, rng = _world(seed, T, N, lag, False, dl)
        def book():
            W = pd.DataFrame(np.where(rng.random((T, N)) < 0.2, rng.standard_normal((T, N)), 0.0), index=p.dates, columns=p.tickers).where(p.eligible, 0.0)
            return W / np.maximum(1.0, W.abs().sum(axis=1)).to_numpy()[:, None]
        A, B = book(), book()
        g = lambda W: _quiet(lambda: q.backtest_weights(p, W, spread_bp=0.0, benchmark=None, check_universe=False, check_shortable=False))        # noqa: E731
        ra, rb_ = g(A), g(B)
        mixed = g(mix * A + (1 - mix) * B)
        assert np.allclose(mixed.net_returns.to_numpy(), mix * ra.net_returns.to_numpy() + (1 - mix) * rb_.net_returns.to_numpy(), atol=1e-12, rtol=0)          # linear at zero cost
        big = _quiet(lambda: q.backtest_weights(p, A * lev, spread_bp=0.0, benchmark=None, check_universe=False, check_shortable=False))
        n = big.net_returns.to_numpy()
        assert (n >= -1.0 - 1e-12).all() and np.isfinite(n).all()
        if big.metrics["ruined"]:
            i = int(np.flatnonzero(n <= -1.0 + 1e-12)[0])
            assert (n[i + 1:] == 0.0).all()                                                       # once wiped out, for good
            if len(n) >= 126:                                                                       # under half a year the annualised figures are NaN by design
                assert abs(big.metrics["MDD"] + 1.0) < 1e-9

    @CFG
    @given(st.lists(st.floats(-5.0, 5.0, allow_nan=False), min_size=1, max_size=40), st.integers(1, 30))
    def prop_ruin_helper(vals, cut):
        x = np.array(vals)
        out, i = stop_at_ruin(x, cut)
        bad = np.flatnonzero(x[:cut] <= -1.0)
        if bad.size == 0:
            assert i is None and out is x
        else:
            assert i == int(bad[0]) and np.array_equal(out[:i], x[:i]) and out[i] == -1.0 and (out[i + 1:] == 0.0).all()

    @CFG
    @given(st.lists(st.floats(0.0, 1.0), min_size=1, max_size=60))
    def prop_multiple_testing(ps):
        p = np.array(ps)
        bh, hm = rb.bh_fdr(p), rb.holm(p)
        assert (bh >= p - 1e-15).all() and (bh <= 1.0).all() and (hm >= bh - 1e-12).all() and (hm <= 1.0).all()
        o = np.argsort(p, kind="stable")
        assert (np.diff(bh[o]) >= -1e-12).all() and (np.diff(hm[o]) >= -1e-12).all()                           # an adjusted p-value never falls as the raw one rises

    @CFG
    @given(st.integers(0, 10 ** 6), st.integers(80, 400), st.floats(0.0001, 0.05), st.floats(-0.002, 0.004))
    def prop_statistics(seed, n, sd, mu):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2018-01-01", periods=n)
        x = pd.Series(rng.standard_t(5, n) * sd + mu, index=idx).clip(-0.9, 0.9)
        b = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
        k = rb.risk_profile(x)
        assert k["worst_bar"] <= k["var_99"] + 1e-15 <= k["var_95"] + 2e-15 and k["cvar_99"] <= k["var_99"] + 1e-15 and k["cvar_95"] <= k["var_95"] + 1e-15 and k["cvar_95"] >= k["worst_bar"] - 1e-15
        assert 0.0 <= k["share_underwater"] <= 1.0 and k["ulcer_index"] >= 0.0 and 0 <= k["max_dd_bars"] <= n and k["best_bar"] >= k["worst_bar"]
        dd = rb.drawdown_distribution(x, n=100, seed=seed)
        assert -1.0 <= dd["p05"] <= dd["p25"] <= dd["p50"] <= dd["p75"] <= dd["p95"] <= 0.0 and -1.0 <= dd["realized"] <= 0.0 and 0.0 <= dd["p_worse"] <= 1.0
        mt = rb.mean_tests(x)
        lo, hi = mt["win_ci"]
        assert 0.0 <= lo <= mt["win_rate"] <= hi <= 1.0 and 0.0 <= mt["sign_p"] <= 1.0 and 0.0 <= mt["nw_p"] <= 1.0 and mt["n_zero"] >= 0
        mr = rb.market_relative(x, b)
        assert 0.0 <= mr["hit_rate_month"] <= 1.0 and 0.0 <= mr["hit_rate_year"] <= 1.0 and -1.0 - 1e-12 <= mr["correlation"] <= 1.0 + 1e-12 and mr["max_relative_drawdown"] <= 0.0
        sp = rb.in_out_of_sample(x, idx[n // 2])
        assert 0.0 <= sp["mean_diff_p"] <= 1.0 and sp["in_sample"]["bars"] + sp["out_of_sample"]["bars"] == n
        sub = rb.subperiods(x)
        assert abs((1 + sub["years"]["ret"]).prod() - (1 + x).prod()) < 1e-9                                  # the years chain to the whole
        R = pd.DataFrame(rng.normal(0, 0.01, (n, 5)), index=idx)
        tr, te = max(20, n // 4), max(10, n // 8)
        if tr + te <= n:
            wf = rb.walk_forward(R, train=tr, test=te)
            assert wf["n_folds"] == (n - tr) // te and len(wf["oos"]) == wf["n_folds"] * te and set(wf["folds"]["chosen"]) <= set(R.columns)

    @CFG
    @given(st.lists(st.lists(st.one_of(st.floats(-3, 3), st.just(float("nan"))), min_size=1, max_size=6), min_size=1, max_size=6))
    def prop_plateau(rows):
        w = max(len(r) for r in rows)
        T = pd.DataFrame([r + [float("nan")] * (w - len(r)) for r in rows])
        if T.size < 3 or not np.isfinite(T.to_numpy()).any():
            return
        r = rb.parameter_plateau(T)
        assert r["best"] == np.nanmax(T.to_numpy()) and 0 <= r["n_neighbours"] <= 8 and 0.0 <= r["grid_positive"] <= 1.0
        assert r["verdict"] in ("plateau", "hill", "spike", "undetermined") and (r["reason"] == "") == math.isfinite(r["neighbour_ratio"])

    def _json_values():
        leaf = st.one_of(st.none(), st.booleans(), st.integers(-10 ** 9, 10 ** 9), st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=8),
                         st.floats(allow_nan=False).map(np.float64), st.integers(-100, 100).map(np.int64), st.just(pd.NaT), st.just(np.nan), st.just(pd.Timestamp("2020-01-02")))
        return st.recursive(leaf, lambda c: st.one_of(st.lists(c, max_size=4), st.dictionaries(st.text(min_size=1, max_size=5), c, max_size=4)), max_leaves=12)

    @CFG
    @given(_json_values())
    def prop_json(x):
        text = json.dumps(jsonable(x), allow_nan=False)                                                       # never a NaN or Infinity token, never an exception
        json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))

    @CFG
    @given(st.integers(0, 10 ** 6), st.integers(1, 8), st.floats(0.0, 0.2))
    def prop_events(seed, hold, density):
        p, f, rng = _world(seed, 150, 16, 1, False, False)
        sig = pd.DataFrame(rng.random(p.close.shape) < density, index=p.dates, columns=p.tickers)
        W = event_weights(p, sig, hold)
        assert (W.sum(axis=1) <= 1.0 + 1e-12).all() and (W.to_numpy() >= -1e-15).all()
        body = sig.copy(); body.iloc[len(body) - hold:] = False                                                # lots that end inside the sample
        W2 = event_weights(p, body, hold)
        fired_days = int(((body & p.eligible).any(axis=1)).sum())
        assert abs(W2.to_numpy().sum() - fired_days) < 1e-9                                                    # each date with a fire invests 1 in total (1/hold for hold bars)
        m = list(rng.random(40) < 0.5)
        assert _longest_run(m) == max([len(s) for s in "".join("1" if v else "0" for v in m).split("0")] + [0])
        k = int(rng.integers(0, 50)); nn = k + int(rng.integers(0, 50))
        if nn:
            lo_, hi_ = _wilson(k, nn)
            assert 0.0 <= lo_ <= k / nn <= hi_ <= 1.0

    PROPS = [prop_portfolio, prop_causality, prop_order, prop_weights_blend_and_ruin, prop_ruin_helper, prop_multiple_testing, prop_statistics, prop_plateau, prop_json, prop_events]
else:
    PROPS = []


if __name__ == "__main__":
    if not PROPS:
        print("property tests skipped: hypothesis is not installed")
    else:
        for fn in PROPS:
            fn()
            print(f"PP {fn.__name__:28s} {EXAMPLES} examples  PASS")
        print("property tests: all passed")
