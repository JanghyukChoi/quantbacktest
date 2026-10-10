"""Core pieces whose thresholds and formulas a mutation run changed without any test failing: the panel (fingerprint, validity, spacing note, causality check, audit, universe builder),
the impact model and the analytics helpers (alpha and beta, IC report, bootstrap intervals).

CV1 impact        cost = trade x min(y x sigma x sqrt(trade x aum / ADV), cap) worked by hand; no volume history is charged the cap; participation is NaN where nothing traded; the model is frozen
CV2 panel         fingerprint, the half-of-the-dates rule, the spacing note at a 12-hour gap
CV3 assert_causal the relative tolerance, the row floor and the cuts
CV4 audit         extreme-return share, truncation and survivorship flags at their edges
CV5 universe      listing, minimum average traded value (equal passes), exclusions, bad prices
CV6 alpha/beta    appraisal ratio, the default lag rule, the observation floor
CV7 IC            the IC of a day against Spearman, the minimum names, the report's standard deviation and horizons
CV8 bootstrap     Sharpe interval and the paired difference against the same bootstrap rebuilt in the test
CV9 look-ahead    the heuristic's spreads and tolerance against an independent computation
"""
from __future__ import annotations
import dataclasses, math, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import analytics as an
from pitbacktest.core.impact import ImpactModel, _impact_cost
from pitbacktest.core.panel import build_pit_eligible
from test_argument_checks import _raises


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _panel(T=60, N=12, seed=0, **kw):
    rng = np.random.default_rng(seed)
    idx = kw.pop("index", pd.bdate_range("2020-01-01", periods=T))
    cols = [f"S{i:02d}" for i in range(N)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), N)), axis=0)), index=idx, columns=cols)
    el = kw.pop("eligible", pd.DataFrame(True, index=idx, columns=cols))
    return q.Panel(close=close, eligible=el, volume=kw.pop("volume", pd.DataFrame(1e5, index=idx, columns=cols)), market="T", entry_lag=1, **kw)


def test_impact():
    p = _panel(T=80, N=12, seed=1)
    p.volume.iloc[:, 3] = 0.0                                                     # no traded value at all: untradable, charged the cap
    m = ImpactModel(aum=5e6, y=1.0, vol_window=20, adv_window=30, max_cost_bp=50.0)
    H = np.zeros(p.close.shape)
    r_big, r_small = 70, 15
    H[r_big, 0] = 0.3; H[r_big, 3] = 0.2                                          # a trade on a late row (full windows) in a normal and in a zero-volume name
    H[r_small, 1] = 0.2                                                           # a trade early, when only part of the volatility window exists
    cost, part = _impact_cost(p, H, m)
    ret = p.ret1().to_numpy(float)
    px = p.close.to_numpy(float)
    vol = p.volume.to_numpy(float)
    cap = 50.0 / 1e4

    def expected(r, j, delta):
        w = ret[max(1, r - 19):r + 1, j]                                          # the last 20 returns (row 0 has none)
        sig = np.std(w, ddof=1) if len(w) >= 10 else np.nan
        k = min(r + 1, 30)
        adv = (px[r - 29:r + 1, j] * vol[r - 29:r + 1, j]).mean() if r + 1 >= 30 else np.nan
        pa = delta * 5e6 / adv if (np.isfinite(adv) and adv > 0) else np.inf
        u = 1.0 * sig * math.sqrt(pa) if np.isfinite(sig) else np.nan
        return min(u, cap) if np.isfinite(u) else cap
    # the book is built from cash at row 0, so the trade at row r is H[r] - H[r-1]: a position opened at r then closed at r + 1 costs twice
    want_big = 0.3 * expected(r_big, 0, 0.3) + 0.2 * cap
    want_big_close = 0.3 * expected(r_big + 1, 0, 0.3) + 0.2 * expected(r_big + 1, 3, 0.2)
    assert abs(cost[r_big] - want_big) < 1e-12, (cost[r_big], want_big)
    assert abs(cost[r_big + 1] - 0.3 * expected(r_big + 1, 0, 0.3) - 0.2 * cap) < 1e-12 and want_big_close > 0
    assert abs(cost[r_small] - 0.2 * cap) < 1e-12                                 # ADV needs 30 rows, so a trade at row 15 is charged the cap
    assert np.isnan(part[10]).all() and np.isnan(part[r_big, 5]) and np.isfinite(part[r_big, 0]) and np.isinf(part[r_big, 3])
    # volatility from a part-filled window: 10 of 20 returns are enough, 9 are not
    p2 = _panel(T=80, N=12, seed=2)
    p2.volume.iloc[:, :] = 1e6
    for row, finite in ((10, True), (9, False)):                                  # row r has r returns (row 0 has none)
        H2 = np.zeros(p2.close.shape); H2[row, 0] = 0.1
        c2, _ = _impact_cost(p2, H2, ImpactModel(aum=1e6, adv_window=1, max_cost_bp=1e9))
        r_ = p2.ret1().to_numpy(float)[1:row + 1, 0]
        exp_ = 0.1 * 1.0 * np.std(r_, ddof=1) * math.sqrt(0.1 * 1e6 / (p2.close.iloc[row, 0] * 1e6)) if finite else 0.1 * 1e9 / 1e4
        assert abs(c2[row] - exp_) < 1e-9 * max(1.0, abs(exp_)), (row, c2[row], exp_)
    assert dataclasses.is_dataclass(m)
    try:
        m.aum = 1.0
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("an ImpactModel must be immutable: it is part of a recorded run")
    ImpactModel(aum=1.0, vol_window=2, adv_window=1)
    _raises(lambda: ImpactModel(aum=1.0, vol_window=1), "vol_window")
    _raises(lambda: ImpactModel(aum=1.0, adv_window=0), "adv_window")
    print("CV1 impact cost, cap, part-filled volatility window, participation and validation equal the worked values  PASS")


def test_panel_basics():
    p = _panel(T=40, N=12)
    fp = p.fingerprint()
    assert len(fp) == 16 and int(fp, 16) >= 0 and fp == _panel(T=40, N=12).fingerprint()
    assert fp != _panel(T=40, N=12, seed=9).fingerprint()
    # fewer than ten eligible securities on more than half of the dates is refused; exactly half is allowed
    idx = pd.bdate_range("2020-01-01", periods=40); cols = [f"S{i:02d}" for i in range(12)]
    for n_thin, ok in ((20, True), (21, False)):
        el = pd.DataFrame(True, index=idx, columns=cols); el.iloc[:n_thin, 5:] = False       # 5 eligible on the first n_thin dates
        try:
            _panel(T=40, N=12, eligible=el)
            raised = False
        except ValueError:
            raised = True
        assert raised == (not ok), (n_thin, raised)
    empty = pd.DataFrame(False, index=idx, columns=cols); empty.iloc[30:, :] = True                       # 30 dates with nobody eligible, 10 with everybody
    _panel(T=40, N=12, eligible=empty)                                                                     # dates with no eligible security do not count as thin
    # the spacing note: bars 12 hours apart annualised with 252 are out by a factor of 2.9
    hh = pd.date_range("2021-01-01", periods=60, freq="12h")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pp = _panel(index=hh)
    assert any("periods_per_year" in n for n in pp.meta.get("construction_notes", [])), pp.meta
    h6 = pd.date_range("2021-01-01", periods=60, freq="6h")
    assert not any("periods_per_year" in n for n in _panel(index=h6).meta.get("construction_notes", []))
    print("CV2 fingerprint, the half-of-the-dates rule and the spacing note at a 12-hour gap  PASS")


def test_assert_causal():
    p = _panel(T=420, N=12, seed=4)
    ok = p.assert_causal(lambda pn: pn.close.rolling(20).mean())
    assert ok["pass"] and len(ok["cuts"]) == 6 and len(set(ok["cuts"])) == 6                                # six distinct cuts by default
    tight = _panel(T=310, N=12, seed=4).assert_causal(lambda pn: pn.close, n_cuts=8)
    assert len(tight["cuts"]) == len(set(tight["cuts"])) == 8                                              # drawn without replacement
    bad = p.assert_causal(lambda pn: pn.close.shift(-1), n_cuts=6)
    assert not bad["pass"] and bad["mismatched_cuts"]
    # relative tolerance: a difference of 1e-10 passes at 1e-9, 1e-8 does not, however large the level
    def drifting(pn, eps):
        s = pn.close * 1e6
        if len(pn.dates) < 420:
            s.iloc[-1] = s.iloc[-1] * (1 + eps)                                      # the truncated rebuild differs from the full signal by eps, relative
        return s
    assert p.assert_causal(lambda pn: drifting(pn, 1e-10))["pass"]
    assert not p.assert_causal(lambda pn: drifting(pn, 1e-8))["pass"]
    short = _panel(T=302, N=12, seed=4)
    _raises(lambda: _panel(T=301, N=12).assert_causal(lambda pn: pn.close), "at least 302")
    assert len(short.assert_causal(lambda pn: pn.close, n_cuts=6)["cuts"]) == 1                # 302 rows: cuts can only fall on row 300
    assert len(p.assert_causal(lambda pn: pn.close, n_cuts=3)["cuts"]) == 3
    assert p.assert_causal(lambda pn: pn.close, min_history=100)["cuts"][0] >= str(p.dates[100].date())
    print("CV3 assert_causal: relative tolerance, the 300-row floor, the number of cuts, a look-ahead is caught  PASS")


def test_audit():
    rng = np.random.default_rng(5)
    T, N = 800, 40
    idx = pd.bdate_range("2018-01-01", periods=T); cols = [f"S{i:02d}" for i in range(N)]
    ret = rng.normal(0, 0.01, (T, N))
    ret[100, :10] = 0.5                                                           # 10 of 40 names jump on one day: 10 / (T * N) of the cells
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=cols)
    p = q.Panel(close=close, eligible=pd.DataFrame(True, index=idx, columns=cols), market="T", entry_lag=1)
    a = p.audit()
    got_cells = (np.abs(close.pct_change(fill_method=None).to_numpy()) > 0.30).sum()
    assert got_cells >= 10 and abs(a["ret_extreme_pct"] - got_cells / (T * N) * 100) < 1e-9, (a["ret_extreme_pct"], got_cells / (T * N) * 100)
    assert a["tickers"] == N and a["dates"] == T
    # survivorship: nothing ends before the end, 40 names, 4 years: suspected. 29 names: not. One of forty ending: not (2.5 percent).
    assert a["survivorship_suspected"] is True and a["ended_before_end_share"] == 0.0
    few = q.Panel(close=close.iloc[:, :29], eligible=pd.DataFrame(True, index=idx, columns=cols[:29]), market="T", entry_lag=1).audit()
    assert few["survivorship_suspected"] is False
    c2 = close.copy(); c2.iloc[400:, 0] = np.nan
    one = q.Panel(close=c2, eligible=pd.DataFrame(True, index=idx, columns=cols), market="T", entry_lag=1).audit()
    assert one["survivorship_suspected"] is False and abs(one["ended_before_end_share"] - 1 / 40) < 1e-12
    for n_names, expect in ((30, True), (29, False)):
        sub = close.iloc[:, :n_names]
        assert q.Panel(close=sub, eligible=pd.DataFrame(True, index=idx, columns=cols[:n_names]), market="T", entry_lag=1).audit()["survivorship_suspected"] is expect
    for days, expect in ((1096, True), (1095, False)):                                                      # the span must be 3 years: 3 x 365.25 = 1095.75 days
        ix = pd.DatetimeIndex(pd.date_range("2018-01-01", periods=120).tolist()[:119] + [pd.Timestamp("2018-01-01") + pd.Timedelta(days=days)])
        cl = pd.DataFrame(np.tile(100.0, (120, 30)), index=ix, columns=cols[:30])
        assert q.Panel(close=cl, eligible=pd.DataFrame(True, index=ix, columns=cols[:30]), market="T", entry_lag=1).audit()["survivorship_suspected"] is expect, days
    shortspan = q.Panel(close=close.iloc[:600], eligible=pd.DataFrame(True, index=idx[:600], columns=cols), market="T", entry_lag=1).audit()     # 600 bdays = 2.3 years
    assert shortspan["survivorship_suspected"] is False
    c3 = close.copy(); c3.iloc[-5:, 0] = np.nan                                   # ended 5 days before the end: not 30 days
    assert q.Panel(close=c3, eligible=pd.DataFrame(True, index=idx, columns=cols), market="T", entry_lag=1).audit()["ended_before_end_share"] == 0.0
    c4 = close.copy(); c4.iloc[-23:, 0] = np.nan                                  # 23 business days = 31+ calendar days before the end
    assert abs(q.Panel(close=c4, eligible=pd.DataFrame(True, index=idx, columns=cols), market="T", entry_lag=1).audit()["ended_before_end_share"] - 1 / 40) < 1e-12
    # truncation: coverage collapsing to under half in the last 20 rows
    c5 = close.copy(); c5.iloc[-20:, :25] = np.nan
    t5 = q.Panel(close=c5, eligible=pd.DataFrame(True, index=idx, columns=cols), market="T", entry_lag=1).audit()
    assert t5["truncation_suspected"] is True and abs(t5["tail_coverage_ratio"] - 15 / 40) < 1e-12
    c6 = close.copy(); c6.iloc[-20:, :20] = np.nan                                # exactly half: not "under half"
    assert q.Panel(close=c6, eligible=pd.DataFrame(True, index=idx, columns=cols), market="T", entry_lag=1).audit()["truncation_suspected"] is False
    print("CV4 audit: extreme-return share, survivorship flag (names, span, share ended, 30-day rule) and truncation flag at their edges  PASS")


def test_universe_builder():
    idx = pd.bdate_range("2020-01-01", periods=60); cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(10.0, index=idx, columns=cols)
    close.loc[idx[30]:, "D"] = np.nan; close.iloc[:, 2] = np.where(np.arange(60) % 2 == 0, 10.0, -1.0)           # D vanishes, C has bad prices on odd rows
    volume = pd.DataFrame(100.0, index=idx, columns=cols); volume["B"] = 50.0
    listed = pd.DataFrame(True, index=idx, columns=cols); listed.loc[:idx[9], "A"] = False
    excl = pd.DataFrame(False, index=idx, columns=cols); excl.loc[idx[40]:, "B"] = True
    ok = build_pit_eligible(close, listed=listed, exclude=excl)
    assert not ok.loc[:idx[9], "A"].any() and ok.loc[idx[10]:, "A"].all()
    assert ok.loc[:idx[29], "D"].all() and not ok.loc[idx[30]:, "D"].any()
    assert ok.loc[idx[0], "C"] and not ok.loc[idx[1], "C"]
    assert ok.loc[:idx[39], "B"].all() and not ok.loc[idx[40]:, "B"].any()
    adv = build_pit_eligible(close, volume=volume, min_adv=1000.0)               # A: 10 x 100 = 1000 exactly; B: 500
    assert adv["A"].iloc[19:].all() and not adv["A"].iloc[:19].any() and not adv["B"].any(), "equal to the minimum passes; fewer than 20 days of history do not"
    unlisted = listed.drop(columns=["B"])                                                                  # no listing record for B: not listed
    assert not build_pit_eligible(close, listed=unlisted)["B"].any()
    assert build_pit_eligible(close, volume=volume)["A"].all(), "no minimum: no history needed"
    tiny = volume.copy(); tiny["A"] = 0.01                                                                 # average traded value 0.1
    assert not build_pit_eligible(close, volume=tiny, min_adv=0.5)["A"].any() and build_pit_eligible(close, volume=tiny, min_adv=0.05)["A"].iloc[19:].all()
    print("CV5 universe builder: listing, exclusions, bad prices, the minimum average traded value (equal passes, 20 days needed)  PASS")


def test_alpha_beta():
    rng = np.random.default_rng(6)
    T = 400
    idx = pd.bdate_range("2019-01-01", periods=T)
    F = pd.DataFrame({"m": rng.normal(0, 0.01, T), "s": rng.normal(0, 0.005, T)}, index=idx)
    y = pd.Series(0.0004 + 0.8 * F["m"] - 0.3 * F["s"] + rng.normal(0, 0.006, T), index=idx)
    r = an.alpha_beta(y, F)
    X = np.column_stack([np.ones(T), F.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    e = y.to_numpy() - X @ coef
    rv = e.std(ddof=3) * math.sqrt(252)
    assert abs(r["resid_vol_annual"] - rv) < 1e-12 and abs(r["appraisal"] - coef[0] * 252 / rv) < 1e-9 and 0.01 < abs(r["appraisal"]) < 50
    assert r["lag"] == int(4 * (T / 100) ** (2 / 9)) == 5 and an.alpha_beta(y, F, lag=5)["alpha_t"] == r["alpha_t"]
    assert an.alpha_beta(y, F, lag=0)["alpha_t"] != r["alpha_t"] and an.alpha_beta(y, F, lag=20)["lag"] == 20
    T2 = 273                                                                                             # 4 x (T/100)^(2/9) = 5.0003: the rule's constants decide between 4 and 5
    F2, y2 = F.iloc[:T2], y.iloc[:T2]
    assert an.alpha_beta(y2, F2)["lag"] == int(4 * (T2 / 100) ** (2 / 9)) == 5
    # observation floor: three per coefficient and min_obs
    f1 = F[["m"]]
    n6 = idx[:6]
    assert an.alpha_beta(y[n6], f1.loc[n6], min_obs=1)["n"] == 6
    _raises(lambda: an.alpha_beta(y[idx[:5]], f1.loc[idx[:5]], min_obs=1), "usable observations")
    _raises(lambda: an.alpha_beta(y[:59], f1[:59]), "usable observations")
    assert an.alpha_beta(y[:60], f1[:60])["n"] == 60
    print("CV6 appraisal ratio, the default lag rule and the observation floor  PASS")


def test_ic():
    rng = np.random.default_rng(7)
    T, N = 120, 30
    idx = pd.bdate_range("2020-01-01", periods=T); cols = [f"S{i:02d}" for i in range(N)]
    fac = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)
    fwd = fac * 0.3 + pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)
    el = pd.DataFrame(True, index=idx, columns=cols)
    ic = an.information_coefficient(fac, fwd, el)
    for i in (0, 50, 119):
        assert abs(ic.iloc[i] - fac.iloc[i].rank().corr(fwd.iloc[i].rank())) < 1e-12           # Spearman: the Pearson correlation of the ranks
    el20 = el.copy(); el20.iloc[7, 20:] = False; el20.iloc[8, 19:] = False          # day 7 keeps 20 names, day 8 keeps 19
    ic20 = an.information_coefficient(fac, fwd, el20)
    assert np.isfinite(ic20.iloc[7]) and np.isnan(ic20.iloc[8])
    const = fac.copy(); const.iloc[9] = 1.0
    assert np.isnan(an.information_coefficient(const, fwd, el).iloc[9])             # no variation: undefined, not zero
    p = _panel(T=T, N=N, seed=8)
    p = dataclasses.replace(p, close=p.close.set_axis(idx), eligible=el)
    rep = an.ic_report(p, fac)
    assert sorted(rep) == [1, 5, 20]
    from pitbacktest.core.estimators import newey_west_t
    for h, row in rep.items():
        s = an.information_coefficient(fac, an.forward_returns(p, h), el).dropna()
        mu, _, t, n = newey_west_t(s.to_numpy(), lag=max(1, h))
        assert row["n_days"] == n and abs(row["mean_ic"] - mu) < 1e-12 and abs(row["t_nw"] - t) < 1e-9
        assert abs(row["std_ic"] - s.std(ddof=1)) < 1e-12 and abs(row["icir"] - mu / s.std(ddof=1)) < 1e-9 and abs(row["hit_rate"] - (s > 0).mean()) < 1e-12
    print("CV7 information coefficient against Spearman, the 20-name floor, and the report's std, ICIR, hit rate, t and default horizons  PASS")


def test_bootstrap_intervals():
    rng = np.random.default_rng(9)
    T = 250
    idx = pd.bdate_range("2020-01-01", periods=T)
    x = pd.Series(rng.normal(0.0006, 0.01, T), index=idx)
    r = an.sharpe_ci(x, n=800, seed=3)
    assert r["n_boot"] == 800 and r["mean_block"] == round(T ** (1 / 3)) and r["level"] == 0.95 and r["T"] == T
    assert abs(r["sharpe"] - x.mean() / x.std(ddof=1) * math.sqrt(252)) < 1e-12
    gen = np.random.default_rng(3)
    sr = np.concatenate([an._sharpe_rows(x.to_numpy()[an._stationary_indices(T, c, float(round(T ** (1 / 3))), gen)], 252) for c in an._chunks(800)])
    assert abs(r["lo"] - np.quantile(sr, 0.025)) < 1e-12 and abs(r["hi"] - np.quantile(sr, 0.975)) < 1e-12 and abs(r["se"] - sr.std(ddof=1)) < 1e-12
    assert abs(an._sharpe_rows(np.array([[1.0, 2.0, 4.0]]), 1)[0] - 7 / 3 / np.std([1, 2, 4], ddof=1)) < 1e-12
    n90 = an.sharpe_ci(x, n=800, seed=3, level=0.90)
    assert n90["lo"] > r["lo"] and n90["hi"] < r["hi"]
    assert an.sharpe_ci(x, seed=1)["n_boot"] == 2000 and an.sharpe_diff_ci(x, x * 0.5)["n_boot"] == 2000
    # paired difference
    same = an.sharpe_diff_ci(x, x, n=400)
    assert same["diff"] == 0.0 and same["p_boot"] == 1.0 and same["lo"] == 0.0 and same["hi"] == 0.0
    y = pd.Series(rng.normal(0.0, 0.01, T), index=idx)
    d = an.sharpe_diff_ci(x, y, n=800, seed=5)
    gen = np.random.default_rng(5)
    xa, xb = x.to_numpy(), y.to_numpy()
    dd = []
    for c in an._chunks(800):
        idx_ = an._stationary_indices(T, c, float(round(T ** (1 / 3))), gen)
        dd.append(an._sharpe_rows(xb[idx_], 252) - an._sharpe_rows(xa[idx_], 252))
    dd = np.concatenate(dd)
    assert abs(d["lo"] - np.quantile(dd, 0.025)) < 1e-12 and abs(d["hi"] - np.quantile(dd, 0.975)) < 1e-12 and abs(d["se"] - dd.std(ddof=1)) < 1e-12
    assert abs(d["p_boot"] - min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))) < 1e-12
    assert abs(d["diff"] - (y.mean() / y.std(ddof=1) - x.mean() / x.std(ddof=1)) * math.sqrt(252)) < 1e-12
    _raises(lambda: an.sharpe_ci(x[:29]), "too few")
    _raises(lambda: an.sharpe_diff_ci(x[:29], y[:29]), "too few")
    print("CV8 Sharpe interval and paired difference equal the same bootstrap rebuilt in the test; default 2000 draws, 95 percent  PASS")


def test_lookahead_heuristic():
    p = _panel(T=300, N=40, seed=11)
    rng = np.random.default_rng(12)
    sig = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    h = 5
    r = p.assert_no_lookahead(sig, h=h)
    fwd = p.forward(h).to_numpy(float)
    el = p.eligible.to_numpy(bool)

    def spread(s_):
        rk = s_.where(p.eligible).rank(axis=1, pct=True, na_option="keep").to_numpy(float)
        out = []
        for i in range(len(rk)):
            a, b = (rk[i] >= 0.9) & el[i] & np.isfinite(fwd[i]), (rk[i] <= 0.1) & el[i] & np.isfinite(fwd[i])
            if a.sum() >= 3 and b.sum() >= 3:
                out.append(fwd[i][a].mean() - fwd[i][b].mean())
        return float(np.mean(out))
    assert abs(r["base_bp"] - spread(sig) * 1e4) < 1e-9 and abs(r["lagged_bp"] - spread(sig.shift(1)) * 1e4) < 1e-9
    gen = np.random.default_rng(0)
    nulls = []
    for _ in range(8):                                                                                     # eight shuffles within each date's eligible names
        sh = sig.to_numpy().copy()
        for i in range(sh.shape[0]):
            j = np.where(el[i])[0]
            if len(j) > 1:
                sh[i, j] = sh[i, gen.permutation(j)]
        nulls.append(spread(pd.DataFrame(sh, index=sig.index, columns=sig.columns)))
    assert abs(r["tol_bp"] - 2 * np.std(nulls) * 1e4) < 1e-9 and r["pass"] is True
    print("CV9 the look-ahead heuristic's spreads and noise tolerance equal an independent computation  PASS")


if __name__ == "__main__":
    for fn in (test_impact, test_panel_basics, test_assert_causal, test_audit, test_universe_builder, test_alpha_beta, test_ic, test_bootstrap_intervals, test_lookahead_heuristic):
        fn()
    print("core value tests: all passed")
