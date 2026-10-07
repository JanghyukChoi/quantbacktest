"""Behaviour that had no test: spread estimators, the crypto cost model, event signals (including the dummy-regression neutralisation),
Fama-MacBeth with missing returns, the paired difference, panel validation and the point-in-time mask helper.

CE1 Roll         a bid-ask bounce of known size is recovered; trending prices give NaN
CE2 Corwin-Schultz  a world with only a spread recovers it; the output is bounded and zero without a range
CE3 spread model the log-log fit recovers its coefficients; the panel is clipped, monotone and overridable
CE4 crypto costs the fixed model is exact, the thin-contract penalty applies, the old estimator warns; participation has a known answer
CE5 event dummy  a signal that only echoes a control loses its effect after controls; a real, independent effect keeps it
CE6 event stats  skewed trades raise the warning, a rare signal raises, a small universe warns
CE7 Fama-MacBeth with missing returns equals an independent per-day least squares
CE8 paired diff  recovers a known contribution and refuses a short sample
CE9 panel        invalid inputs raise; the audit flags inconsistent adjusted opens; the mask helper follows its documentation
CE10 chars       the five firm characteristics follow their definitions and the missing inputs are left out
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.core import costs
from pitbacktest.core.controls import build_chars_from_financials, build_controls
from pitbacktest.core.estimators import fama_macbeth, paired_diff
from pitbacktest.core.panel import build_pit_eligible
from pitbacktest.crypto import costs as ccosts
from pitbacktest import event


def _panel(T=500, N=60, seed=0, vol=True, open_gaps=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T)
    cols = [f"S{i:02d}" for i in range(N)]
    close = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0, 0.015, (T, N)), axis=0)), index=idx, columns=cols)
    el = pd.DataFrame(True, index=idx, columns=cols); el.iloc[:60] = False
    kw = {}
    if vol:
        kw["volume"] = pd.DataFrame(rng.lognormal(12, 0.3, (T, N)), index=idx, columns=cols)
        kw["mkt_cap"] = close * rng.lognormal(15, 0.8, N)
    if open_gaps is not None:
        o = close.shift(1) * (1 + rng.normal(0, 0.003, (T, N)))
        m = rng.random((T, N)) < open_gaps
        kw["open"] = o.where(~m, o * 1.4)
    return q.Panel(close=close, eligible=el, market="TEST", entry_lag=1, **kw), rng


# ------------------------------------------------------------------------------------------------ CE1
def test_roll_known_answer():
    rng = np.random.default_rng(1)
    n, s = 400_000, 0.02                                          # true quoted spread in price units
    m = 100 + np.cumsum(rng.normal(0, 0.01, n))                   # efficient price
    p = m + (s / 2) * rng.choice([-1.0, 1.0], n)                  # trades bounce between bid and ask
    est = costs.roll_spread(p)
    assert abs(est - s) / s < 0.05, est
    d = np.zeros(5000); e = rng.normal(0, 0.01, 5000)
    for t in range(1, 5000):
        d[t] = 0.6 * d[t - 1] + e[t]                              # momentum in the price changes: positive autocovariance
    assert np.isnan(costs.roll_spread(100 + np.cumsum(d)))
    assert np.isnan(costs.roll_spread(p[:20])) and np.isnan(costs.roll_spread(np.full(100, np.nan)))
    print(f"CE1 Roll recovers a 2.00 cent spread as {est * 100:.2f} cents; momentum and short samples give NaN  PASS")


# ------------------------------------------------------------------------------------------------ CE2
def test_corwin_schultz():
    idx = pd.bdate_range("2020-01-01", periods=300)
    s = 0.004
    px = pd.DataFrame(100.0, index=idx, columns=["A"])
    cs = costs.corwin_schultz(px * (1 + s / 2), px * (1 - s / 2))               # every day's range is exactly the spread
    assert abs(cs["A"].iloc[5:].mean() - s) / s < 0.02, cs["A"].iloc[5:].mean()
    flat = costs.corwin_schultz(px, px)
    assert (flat["A"].iloc[1:] == 0).all()                                      # no range, no spread
    rng = np.random.default_rng(2)
    h = pd.DataFrame(100 * (1 + rng.uniform(0, 0.5, (300, 4))), index=idx, columns=list("ABCD"))
    l = pd.DataFrame(100 * (1 - rng.uniform(0, 0.5, (300, 4))), index=idx, columns=list("ABCD"))
    out = costs.corwin_schultz(h, l).iloc[1:]
    assert out.min().min() >= 0 and out.max().max() <= 0.2                      # clipped at both ends
    print(f"CE2 Corwin-Schultz recovers a 40 bp range-only spread as {cs['A'].iloc[5:].mean() * 1e4:.1f} bp; no range gives 0; output in [0, 0.2]  PASS")


# ------------------------------------------------------------------------------------------------ CE3
def test_spread_model_and_panel():
    rng = np.random.default_rng(3)
    dv = np.exp(rng.uniform(np.log(0.5), np.log(500), 4000))
    sp = np.exp(2.0 - 0.45 * np.log(dv) + rng.normal(0, 0.1, dv.size))
    a, b = costs.fit_spread_model(dv, sp)
    assert abs(a - 2.0) < 0.02 and abs(b + 0.45) < 0.01, (a, b)
    assert all(np.isnan(x) for x in costs.fit_spread_model(dv[:10], sp[:10]))
    adv = pd.DataFrame({"big": 5e9, "mid": 5e7, "tiny": 1e3, "meas": 5e7}, index=pd.bdate_range("2020-01-01", periods=3))
    sp_panel = costs.spread_panel(adv, a=a, b=b, measured={"meas": 7.0, "tiny": float("nan")})
    r = sp_panel.iloc[0]
    assert r["big"] < r["mid"] < r["tiny"] and r["meas"] == 7.0                 # liquid names are cheaper; a measurement wins
    assert sp_panel.min().min() >= 0.05 and sp_panel.max().max() <= 50.0
    print(f"CE3 spread model recovers a={a:.2f}, b={b:.3f}; panel is monotone in liquidity, clipped, and measured values override  PASS")


# ------------------------------------------------------------------------------------------------ CE4
def test_crypto_costs_and_participation():
    p, _ = _panel(T=200, N=12)
    p.volume.iloc[:, 0] = 1.0                                                   # contract 0 is thin: turnover about 50 dollars
    fixed = ccosts.liquidity_cost_bp(p, taker_fee_bp=5.0, half_spread_bp=2.0, window=30, thin_usd=1e8, thin_extra_bp=5.0)
    adv = p.adv(30)
    thin = adv < 1e8
    have = adv.notna().to_numpy()
    fx = fixed.to_numpy()
    assert np.allclose(fx[thin.to_numpy() & have], 12.0) and np.allclose(fx[~thin.to_numpy() & have], 7.0)      # (no stack(): pandas 3 keeps NaN)
    for bad in ("nonsense",):
        try:
            ccosts.liquidity_cost_bp(p, spread_estimator=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown estimator must raise")
    pc = replace_hl(p)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old = ccosts.liquidity_cost_bp(pc, spread_estimator="corwin_schultz")
    ov = old.to_numpy()
    assert any("Corwin-Schultz overstates" in str(x.message) for x in w) and (ov[np.isfinite(ov)] >= 5.0 + 1.0).all()   # fee + the 1 bp floor
    # participation: weights jump 0 -> 1 -> 0 on one name; turnover (ADV) 1e7; account 1e6 -> every trade is 10% of ADV
    p2, _ = _panel(T=120, N=12)
    p2.volume.loc[:, :] = 1e7 / p2.close
    w_ = np.zeros(p2.close.shape); w_[40:60, 0] = 1.0
    rep = ccosts.participation_report(p2, w_, 1e6, window=30)
    assert abs(rep["participation_p50"] - 0.1) < 1e-9 and abs(rep["participation_p99"] - 0.1) < 1e-9 and rep["n_trades"] == 2
    assert abs(rep["aum_at_1pct_p95"] - 1e5) < 1e-3
    assert ccosts.participation_report(p2, np.zeros(p2.close.shape), 1e6)["n_trades"] == 0
    print("CE4 fixed cost = fee + half spread (+5 bp when thin), the old estimator warns and floors at 1 bp, participation is 10% and 100 000 dollars at 1%  PASS")


def replace_hl(p):
    from dataclasses import replace
    return replace(p, high=p.close * 1.01, low=p.close * 0.99)


# ------------------------------------------------------------------------------------------------ CE5
def test_event_dummy_regression():
    p, rng = _panel(T=500, N=80, seed=4)
    ctrl = build_controls(p, include_chars=False)
    x = ctrl["mom21"].to_numpy(np.float64)
    T, N = x.shape
    xc = np.nan_to_num(x, nan=0.0)
    cum_noise = rng.normal(0, 0.02, (T, N))
    # 1) the return is explained by the control; the signal fires on the control's top decile and adds nothing of its own
    cum1 = 0.01 * xc + cum_noise
    fire1 = (pd.DataFrame(x).rank(axis=1, pct=True) >= 0.9).to_numpy() & p.eligible.to_numpy()
    r1 = event._dummy_neutralized(p, fire1, {5: cum1}, (5,))[5]
    assert r1["raw_bp"] > 40 and r1["raw_t"] > 5, r1
    assert abs(r1["neutral_bp"]) < 0.25 * r1["raw_bp"] and r1["survival_pct"] < 30, r1
    # 2) a real effect of the signal itself, independent of the controls: it must survive the controls
    fire2 = (rng.random((T, N)) < 0.1) & p.eligible.to_numpy()
    cum2 = 0.01 * xc + cum_noise + 0.01 * fire2
    r2 = event._dummy_neutralized(p, fire2, {5: cum2}, (5,))[5]
    assert 0.7 < r2["survival_pct"] / 100 < 1.3 and abs(r2["neutral_bp"] - 100) < 15 and r2["neutral_t"] > 5, r2
    print(f"CE5 event dummy regression: a signal echoing a control keeps {r1['survival_pct']:.0f}% of its effect; a real 100 bp effect keeps {r2['survival_pct']:.0f}%  PASS")


# ------------------------------------------------------------------------------------------------ CE6
def test_event_stats_and_errors():
    rng = np.random.default_rng(5)
    T, N = 400, 40
    el = np.ones((T, N), dtype=bool)
    cum = rng.normal(0, 0.001, (T, N)) - 0.0005                     # most trades lose a little
    cum[rng.random((T, N)) < 0.08] += 0.02                           # a few win big
    fire = rng.random((T, N)) < 0.2
    st = event._trade_stats(fire, cum, el, 0.0)
    assert st["mean_bp"] > 0 > st["median_bp"] and st["skew_warning"], st
    assert st["n_trades"] == int((fire & np.isfinite(cum)).sum()) and st["payoff"] > 1
    assert event._trade_stats(rng.random((T, N)) < 0.0001, cum, el, 0.0) is None     # fewer than 100 trades
    p, _ = _panel(T=400, N=60, seed=6)
    sig = pd.DataFrame(False, index=p.dates, columns=p.tickers)
    sig.iloc[100, 0] = True
    try:
        event.backtest_event(p, sig, horizons=(5,), neutralize_check=False)
    except ValueError as e:
        assert "No horizon can be tested" in str(e)
    else:
        raise AssertionError("a signal that fires once cannot be tested")
    # the pool threshold adapts: a 20-name universe is fine over a long sample (min pool 10), so no warning ...
    p20, rng2 = _panel(T=600, N=20, seed=7)
    sig20 = pd.DataFrame(rng2.random(p20.close.shape) < 0.2, index=p20.dates, columns=p20.tickers)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = event.backtest_event(p20, sig20, horizons=(5,), neutralize_check=False)
    assert not any("days of daily excess return" in str(x.message) for x in w) and res.per_horizon[5]["n_days"] >= 100
    # ... but a short sample leaves fewer than 100 usable days, and the gates say they cannot be trusted
    pshort, rng3 = _panel(T=140, N=60, seed=7)
    sigs = pd.DataFrame(rng3.random(pshort.close.shape) < 0.2, index=pshort.dates, columns=pshort.tickers)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        event.backtest_event(pshort, sigs, horizons=(5,), neutralize_check=False)
    assert any("days of daily excess return" in str(x.message) and "pool of at least" in str(x.message) for x in w), [str(x.message)[:60] for x in w]
    print("CE6 positive mean with a negative median raises skew_warning; a signal that fires once raises; a short sample warns and a 20-name universe over a long one does not  PASS")


# ------------------------------------------------------------------------------------------------ CE7
def test_fama_macbeth_matches_least_squares_with_missing_returns():
    p, rng = _panel(T=520, N=70, seed=8)
    ctrl = build_controls(p, include_chars=False)
    f = pd.DataFrame(rng.normal(size=p.close.shape), index=p.dates, columns=p.tickers) + 0.5 * ctrl["mom21"].fillna(0)
    fwd = p.forward(5).astype(float)
    fwd = fwd.mask(rng.random(fwd.shape) < 0.15)                                  # missing forward returns on 15% of cells
    out = fama_macbeth(f, {5: fwd}, ctrl, p.eligible, min_stocks=30)[5]
    coefs, raws = [], []
    names = list(ctrl)
    for i in range(len(p.dates)):
        ok = p.eligible.iloc[i].to_numpy() & np.isfinite(f.iloc[i].to_numpy()) & np.isfinite(fwd.iloc[i].to_numpy())
        for k in names:
            ok &= np.isfinite(ctrl[k].iloc[i].to_numpy(np.float64))
        if ok.sum() < 30:
            continue
        X = np.column_stack([np.ones(ok.sum())] + [ctrl[k].iloc[i].to_numpy(np.float64)[ok] for k in names] + [f.iloc[i].to_numpy()[ok]])
        y = fwd.iloc[i].to_numpy(np.float64)[ok]
        coefs.append(np.linalg.lstsq(X, y, rcond=None)[0][-1])
        fx = f.iloc[i].to_numpy()[ok]
        raws.append(np.cov(fx, y, ddof=0)[0, 1] / fx.var())
    assert abs(out["coef_bp"] - np.mean(coefs) * 1e4) < 1e-6 * max(1.0, abs(out["coef_bp"])) + 1e-6, (out["coef_bp"], np.mean(coefs) * 1e4)
    assert abs(out["coef_bp_raw"] - np.mean(raws) * 1e4) < 1e-6 * max(1.0, abs(out["coef_bp_raw"])) + 1e-6
    assert out["n_days"] == len(coefs)
    print(f"CE7 Fama-MacBeth with 15% missing returns equals a per-day least squares ({out['coef_bp']:.4f} bp, {out['n_days']} days)  PASS")


# ------------------------------------------------------------------------------------------------ CE8
def test_paired_diff():
    rng = np.random.default_rng(9)
    T, N = 300, 50
    cum = rng.normal(0, 0.02, (T, N))
    base = np.ones((T, N), dtype=bool)
    alt = rng.random((T, N)) < 0.3
    cum = cum + 0.001 * alt                                                      # the condition adds exactly 10 bp
    d = paired_diff(base, alt, cum)
    expected = float(np.mean([cum[i][alt[i]].mean() - cum[i].mean() for i in range(T) if alt[i].sum() >= 5]) * 1e4)
    assert abs(d["diff_bp"] - expected) < 1e-9 and d["n_days"] == sum(alt[i].sum() >= 5 for i in range(T))
    assert d["diff_bp"] > 0 and d["t"] > 3
    short = paired_diff(base[:20], alt[:20], cum[:20])
    assert np.isnan(short["diff_bp"]) and short["n_days"] <= 20
    print(f"CE8 paired difference equals its loop ({d['diff_bp']:.2f} bp, t {d['t']:.1f}); under 30 days it refuses  PASS")


# ------------------------------------------------------------------------------------------------ CE9
def test_panel_validation_audit_and_mask():
    idx = pd.bdate_range("2020-01-01", periods=100)
    cols = [f"S{i}" for i in range(20)]
    close = pd.DataFrame(10.0, index=idx, columns=cols)
    ok = pd.DataFrame(True, index=idx, columns=cols)
    def expect(msg, **kw):
        try:
            q.Panel(**{"close": close, "eligible": ok, **kw})
        except ValueError as e:
            assert msg in str(e), (msg, str(e))
        else:
            raise AssertionError(f"no error for {msg}")
    expect("duplicate dates", close=pd.concat([close, close.iloc[:3]]), eligible=pd.concat([ok, ok.iloc[:3]]))
    expect("all False", eligible=ok & False)
    expect("entry_lag", entry_lag=-1)
    few = ok.copy(); few.iloc[:, 5:] = False
    expect("fewer than 10", eligible=few)
    good = q.Panel(close=close, eligible=ok)
    good.close = good.close.iloc[::-1]
    try:
        good.validate()
    except ValueError as e:
        assert "ascending" in str(e)
    else:
        raise AssertionError("a reversed index must fail validate()")
    clean, _ = _panel(T=400, N=40, seed=10, open_gaps=0.0)
    a0 = clean.audit()
    gappy, _ = _panel(T=400, N=40, seed=10, open_gaps=0.05)
    a1 = gappy.audit()
    assert a0["adj_price_consistent"] is True and a0["gap_vs_daily_ratio"] < 3.0, a0
    assert a1["adj_price_consistent"] is False and a1["gap_vs_daily_ratio"] > 3.0, a1
    # the mask helper, by its documentation
    c = pd.DataFrame({"A": np.r_[np.nan, np.full(39, 10.0)], "B": np.full(40, 10.0), "C": np.full(40, 10.0), "D": np.full(40, 10.0)},
                     index=pd.bdate_range("2021-01-01", periods=40))
    vol = pd.DataFrame({"A": 1e6, "B": 1e6, "C": 1.0, "D": 1e6}, index=c.index)
    listed = pd.DataFrame(True, index=c.index, columns=c.columns); listed["D"] = False
    m = build_pit_eligible(c, listed=listed, min_adv=1e6, volume=vol, exclude=pd.DataFrame({"B": [True] * 40}, index=c.index))
    last = m.iloc[-1]
    assert not last["B"] and not last["C"] and not last["D"] and last["A"] and not m["A"].iloc[0]       # excluded, illiquid, unlisted; NaN price is out
    assert not m.iloc[:19].any().any()                                                                   # 20 days of history for the average
    print("CE9 invalid inputs raise (duplicates, all False, entry_lag, too few, reversed); inconsistent adjusted opens are flagged; the mask follows its doc  PASS")


# ------------------------------------------------------------------------------------------------ CE10
def test_chars_definitions():
    idx = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(11)
    close = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0, 0.01, (600, 3)), axis=0)), index=idx, columns=list("ABC"))
    mcap = close * 1e7
    be, ni, ta = mcap * 0.5, mcap * 0.05, mcap * 2.0
    ch = build_chars_from_financials(close, mcap, be, ni, ta)
    assert set(ch) == {"log_size", "momentum", "log_bm", "roa", "asset_growth"}
    assert np.allclose(ch["log_size"], np.log(mcap)) and np.allclose(ch["log_bm"].dropna(), np.log(0.5))
    assert np.allclose(ch["roa"].dropna(), 0.025) and np.allclose(ch["momentum"].dropna(), (close.pct_change(252) - close.pct_change(21)).dropna())
    assert np.allclose(ch["asset_growth"].dropna(), ta.pct_change(252).dropna())
    assert set(build_chars_from_financials(close, mcap)) == {"log_size", "momentum"}
    assert set(build_chars_from_financials(close, mcap, book_equity=be)) == {"log_size", "momentum", "log_bm"}
    assert set(build_chars_from_financials(close, mcap, net_income=ni)) == {"log_size", "momentum"}          # ROA needs total assets too
    print("CE10 the five characteristics follow their definitions; inputs that are not given are left out, not invented  PASS")


if __name__ == "__main__":
    test_roll_known_answer()
    test_corwin_schultz()
    test_spread_model_and_panel()
    test_crypto_costs_and_participation()
    test_event_dummy_regression()
    test_event_stats_and_errors()
    test_fama_macbeth_matches_least_squares_with_missing_returns()
    test_paired_diff()
    test_panel_validation_audit_and_mask()
    test_chars_definitions()
    print("costs and events tests: all passed")
