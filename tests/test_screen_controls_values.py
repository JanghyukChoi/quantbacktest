"""`screen` and the controls it regresses on, against rebuilds from the definitions. A mutation run caught 29 and 38 percent of the changes to these two files.

SCR1 controls     each control's definition (reversals, momenta, volatility, 12-1 momentum and its history rule, size, turnover), the rank scaling, the warnings
SCR2 neutralise   the residual is orthogonal to the controls; the 50-observation floor; a factor inside the controls' span stays missing
SCR3 characteristics  the five firm characteristics from financials
SCR4 deploy       the screen's day-level excess return: pool floor, the 100-day floor, net of cost
SCR5 screen       the table, the funnel at its strict inequalities, the neutralised survival, the null's size, argument checks
"""
from __future__ import annotations
import dataclasses, json, math, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.core.controls import EPS, build_chars_from_financials, build_controls, neutralize, xs_norm, xs_rank
from pitbacktest.core.estimators import newey_west_t
from pitbacktest.core.gates import GateConfig, run_gates
from pitbacktest.screen import _deploy
from test_argument_checks import _raises
from test_synthetic import make_panel


def _caught(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(x.message) for x in w]


def _norm(v, el):
    return ((v.where(el).rank(axis=1, pct=True) - 0.5) * math.sqrt(12.0)).astype(np.float32)


def test_controls():
    p, _ = make_panel(n_days=400, n_stocks=30, seed=3)
    ctrl, msgs = _caught(lambda: build_controls(p))
    assert any("Firm characteristics" in m for m in msgs) and not any("12-1" in m for m in msgs)
    close, el = p.close, p.eligible
    ret = close.pct_change()
    want = {"rev1": ret, "rev5": close.pct_change(5), "mom21": close.pct_change(21), "mom63": close.pct_change(63), "vol21": ret.rolling(21, min_periods=21).std(),
            "mom252_21": close.pct_change(252) - close.pct_change(21), "logsize": np.log(p.mkt_cap.replace(0, np.nan)),
            "turnover": p.adv(20) / (p.mkt_cap + EPS)}
    assert set(ctrl) == set(want), set(ctrl) ^ set(want)
    for k, v in want.items():
        a, b = ctrl[k].to_numpy(np.float64), _norm(v, el).to_numpy(np.float64)
        assert np.allclose(a, b, atol=1e-6, equal_nan=True), k
        assert ctrl[k].dtypes.eq(np.float32).all()
    # scaling: a rank scaled to mean 0, variance 1, NaN outside the eligible set
    v = pd.DataFrame(np.random.default_rng(1).standard_normal((5, 40)))
    e = pd.DataFrame(True, index=v.index, columns=v.columns); e.iloc[2, 30:] = False
    r = xs_rank(v, e)
    assert r.iloc[2, 30:].isna().all() and abs(r.iloc[0].max() - 1.0) < 1e-12 and abs(r.iloc[0].min() - 1 / 40) < 1e-12
    n = xs_norm(v, e)
    assert abs(float(n.iloc[0].mean()) - ((41 / 80) - 0.5) * math.sqrt(12.0)) < 1e-5 and abs(float(n.iloc[0].var(ddof=0)) - 1.0) < 0.01 and n.dtypes.eq(np.float32).all()
    # characteristics are added with a prefix, and no warning is raised when they exist or are declined
    chars = {"roa": pd.DataFrame(np.random.default_rng(2).standard_normal(p.close.shape), index=p.dates, columns=p.tickers)}
    pc = dataclasses.replace(p, chars=chars)
    c2, m2 = _caught(lambda: build_controls(pc))
    assert "char_roa" in c2 and not any("Firm characteristics" in m for m in m2)
    c3, m3 = _caught(lambda: build_controls(pc, include_chars=False))
    assert "char_roa" not in c3
    c4, m4 = _caught(lambda: build_controls(p, include_chars=False))
    assert not any("Firm characteristics" in m for m in m4)
    # the 12-1 momentum needs more than a year of bars plus 21: with exactly 252 + 21 rows it is left out
    for n_rows, kept in ((273, False), (274, True)):
        pp, _ = make_panel(n_days=n_rows, n_stocks=30, seed=3)
        cc, mm = _caught(lambda: build_controls(pp, include_chars=False))
        assert ("mom252_21" in cc) == kept and (any("12-1 momentum" in m for m in mm)) == (not kept), n_rows
    zero_cap = p.mkt_cap.copy(); zero_cap.iloc[100, 3] = 0.0                                # a market value of zero is unknown, not the smallest firm
    zc = build_controls(dataclasses.replace(p, mkt_cap=zero_cap), include_chars=False)
    assert np.isnan(zc["logsize"].iloc[100, 3]) and np.isfinite(zc["logsize"].iloc[100, 4])
    # the lookback follows periods_per_year
    short, _ = make_panel(n_days=100, n_stocks=30, seed=3)
    short = dataclasses.replace(short, periods_per_year=50)
    assert "mom252_21" in build_controls(short, include_chars=False)
    print("SCR1 each control equals its definition; the history rule for 12-1 momentum, the prefix and the warnings are as documented  PASS")


def test_neutralize():
    rng = np.random.default_rng(4)
    T, N = 30, 80
    idx = pd.bdate_range("2020-01-01", periods=T); cols = [f"S{i:02d}" for i in range(N)]
    c1 = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols); c2 = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)
    f = 0.7 * c1 + 0.3 * c2 + pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)
    el = pd.DataFrame(True, index=idx, columns=cols)
    res = neutralize(f, {"a": c1, "b": c2}, el)
    for i in (0, 10, 29):
        X = np.column_stack([np.ones(N), c1.iloc[i], c2.iloc[i]]); y = f.iloc[i].to_numpy()
        want = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        assert np.allclose(res.iloc[i].to_numpy(), want, atol=1e-9)
        assert abs(res.iloc[i].to_numpy() @ c1.iloc[i].to_numpy()) < 1e-8 and abs(res.iloc[i].sum()) < 1e-8
    # 50 observations are enough, 49 are not
    e50 = el.copy(); e50.iloc[0, 50:] = False; e50.iloc[1, 49:] = False
    r50 = neutralize(f, {"a": c1, "b": c2}, e50)
    assert r50.iloc[0].notna().sum() == 50 and r50.iloc[1].isna().all()
    assert neutralize(f, {"a": c1, "b": c2}, e50, min_obs=49).iloc[1].notna().sum() == 49
    # a factor inside the controls' span leaves nothing: NaN; a factor with a small but real remainder stays
    inside = 0.5 * c1 + 0.5 * c2 + 1e-7 * pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)
    assert neutralize(inside, {"a": c1, "b": c2}, el).isna().all().all()
    outside = 0.5 * c1 + 0.5 * c2 + 1e-3 * pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)
    assert neutralize(outside, {"a": c1, "b": c2}, el).notna().all().all()
    scaled = 1000.0 * (0.5 * c1 + 0.5 * c2) + 1e-3 * pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=cols)       # the remainder is 1e-6 of the factor's size
    assert neutralize(scaled, {"a": c1, "b": c2}, el).isna().all().all(), "the span rule is relative to the factor's size, not absolute"
    print("SCR2 the residual equals per-date OLS and is orthogonal to the controls; the 50-observation floor and the span rule  PASS")


def test_chars():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2018-01-01", periods=600); cols = ["A", "B", "C"]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (600, 3)), axis=0)), index=idx, columns=cols)
    mc = close * 1e6; mc.iloc[0, 0] = 0.2                                                  # under 1: clipped to 1
    be = pd.DataFrame(rng.uniform(1e5, 1e7, (600, 3)), index=idx, columns=cols); be.iloc[5, 1] = 1e-9                  # a book value far below the floor of 1e-6
    ni = pd.DataFrame(rng.normal(1e5, 1e4, (600, 3)), index=idx, columns=cols); ta = pd.DataFrame(rng.uniform(1e6, 2e6, (600, 3)), index=idx, columns=cols); ta.iloc[3, 2] = 0
    out = build_chars_from_financials(close, mc, be, ni, ta)
    assert set(out) == {"log_size", "momentum", "log_bm", "roa", "asset_growth"}
    assert np.allclose(out["log_size"].to_numpy(), np.log(mc.clip(lower=1)).to_numpy(), equal_nan=True) and out["log_size"].iloc[0, 0] == 0.0
    assert np.allclose(out["momentum"].to_numpy(), (close.pct_change(252) - close.pct_change(21)).to_numpy(), equal_nan=True)
    assert np.allclose(out["log_bm"].to_numpy(), np.log((be / mc.replace(0, np.nan)).clip(lower=1e-6)).to_numpy(), equal_nan=True) and abs(out["log_bm"].iloc[5, 1] - math.log(1e-6)) < 1e-12
    assert np.isnan(out["roa"].iloc[3, 2]) and abs(out["roa"].iloc[0, 0] - ni.iloc[0, 0] / ta.iloc[0, 0]) < 1e-15
    assert np.allclose(out["asset_growth"].to_numpy(), ta.pct_change(252).to_numpy(), equal_nan=True)
    assert set(build_chars_from_financials(close, mc)) == {"log_size", "momentum"} and "asset_growth" in build_chars_from_financials(close, mc, total_assets=ta) and "roa" not in build_chars_from_financials(close, mc, total_assets=ta)
    print("SCR3 the five characteristics equal their definitions and appear only when their inputs are given  PASS")


def test_deploy():
    rng = np.random.default_rng(6)
    D, N = 120, 40
    cum = rng.normal(0.0005, 0.02, (D, N)); ev = np.ones((D, N), bool)
    fire = rng.random((D, N)) < 0.1
    fire[5] = False                                                                         # a day with no fire adds no observation
    a, st = _deploy(fire, cum, ev, 12.0)
    exp = np.array([cum[i][fire[i]].mean() - cum[i].mean() for i in range(D) if fire[i].any()])
    assert np.allclose(a, exp) and len(a) == (fire.any(axis=1)).sum()
    mu, _, t, _ = newey_west_t(exp, 21)
    assert abs(st["excess_bp"] - mu * 1e4) < 1e-9 and abs(st["net_bp"] - (mu * 1e4 - 12.0)) < 1e-9 and abs(st["t"] - t) < 1e-12
    # 100 days are needed: with 99 qualifying days nothing is reported
    f100 = np.ones((100, N), bool); a100, s100 = _deploy(f100 & (rng.random((100, N)) < 0.2) | np.eye(100, N, dtype=bool), cum[:100], ev[:100], 0.0)
    assert len(a100) == 100 and np.isfinite(s100["excess_bp"])
    a99, s99 = _deploy(f100[:99], cum[:99], ev[:99], 0.0)
    assert len(a99) == 99 and np.isnan(s99["excess_bp"]) and np.isnan(s99["net_bp"]) and np.isnan(s99["t"])
    # pool floor: half the median universe, at least 5 and at most 30
    big = np.ones((120, 100), bool); cum_b = rng.normal(0, 0.02, (120, 100)); cum_b[:60, 30:] = np.nan              # 30 usable names on half the days
    fb = np.zeros((120, 100), bool); fb[:, :3] = True
    ab, _ = _deploy(fb, cum_b, big, 0.0)
    assert len(ab) == 120                                                                    # median universe 100: floor min(30, 50) = 30; 30 qualifies
    cum_c = cum_b.copy(); cum_c[:60, 29:] = np.nan
    assert len(_deploy(fb, cum_c, big, 0.0)[0]) == 60                                        # 29 does not
    print("SCR4 the deployed excess return equals its definition; 100 days and the pool floor are exact  PASS")


def test_screen():
    p, _ = make_panel(n_days=700, n_stocks=60, seed=9)
    rng = np.random.default_rng(9)
    mk = lambda: pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)       # noqa: E731
    factors = {"noise1": mk(), "noise2": mk(), "mom": p.close.pct_change(20)}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = q.screen(p, factors, horizons=(5, 20), primary_h=20, n_null=10)
    df = r.summary
    assert list(df["net_bp"]) == sorted(df["net_bp"], reverse=True) or df["net_bp"].isna().any()
    thr = r.null["p95"]
    assert r.null["n"] == 10
    f = {k: v for k, v in factors.items()}
    for name in factors:
        d = r.factors[name]
        fm20 = d["fm"][20]
        assert abs(df.set_index("factor").loc[name, "t"] - fm20["t"]) < 1e-12 or (np.isnan(fm20["t"]) and np.isnan(df.set_index("factor").loc[name, "t"]))
        rk = factors[name].where(p.eligible).rank(axis=1, pct=True).to_numpy()
        fire = (rk >= 0.9) & p.eligible.to_numpy()
        cum = p.forward(20).to_numpy(np.float64)
        a, dep = _deploy(fire, cum, p.eligible.to_numpy(), 20.0)
        assert abs(d["deploy"]["excess_bp"] - dep["excess_bp"]) < 1e-9 and abs(d["deploy"]["net_bp"] - dep["net_bp"]) < 1e-9
        g = run_gates(a, p.dates, t_stat=fm20["t"], net_bp=dep["net_bp"], rho=d["decile"]["monotonicity_rho"], fire=fire, tickers=p.tickers, cfg=GateConfig(null_threshold=thr), lag=21)
        assert g["subperiod"]["front"]["t"] == d["gates"]["subperiod"]["front"]["t"] or (np.isnan(g["subperiod"]["front"]["t"]) and np.isnan(d["gates"]["subperiod"]["front"]["t"]))
        assert g["passed"] == d["gates"]["passed"]
        nu = d["neutralized"]
        if np.isfinite(dep["excess_bp"]) and abs(dep["excess_bp"]) > 1e-9:
            assert abs(nu["survival_pct"] - nu["excess_bp"] / dep["excess_bp"] * 100) < 1e-9
    fun = r.funnel
    assert fun["all"] == 3
    assert fun[f"G1 |t|>{thr:.2f}"] == int((df.t.abs() > thr).sum()) and fun["G2 net of costs > 0"] == int(((df.t.abs() > thr) & (df.net_bp > 0)).sum())
    assert fun["passed all gates"] == len(r.survivors) == int(df.passed.sum()) and fun["survival after neutralisation > 50%"] == int((df["neu_survival_%"] > 50).sum())
    assert q.screen(p, {"x": factors["noise1"]}, horizons=(20,), neutralize_all=False).null["n"] == 20          # twenty shuffles by default
    assert json.loads(r.to_json()) and r.to_json().splitlines()[1].startswith("  ") and not r.to_json().splitlines()[1].startswith("   ")
    # without the neutralisation the last funnel entry is empty and the columns are NaN
    nn = q.screen(p, {"x": factors["noise1"]}, horizons=(20,), n_null=10, neutralize_all=False)
    assert nn.funnel["survival after neutralisation > 50%"] is None and nn.summary["neu_t"].isna().all()
    # argument rules
    _raises(lambda: q.screen(p, factors, n_null=0), "n_null must be at least 1")
    _raises(lambda: q.screen(p, factors, horizons=(5, 20), primary_h=10), "primary_h")
    _raises(lambda: q.screen(p, factors, cost_bp=-1.0), "cost_bp")
    _raises(lambda: q.screen(p, factors, fire_q=0.0), "fire_q")
    _raises(lambda: q.screen(p, factors, fire_q=1.5), "fire_q")
    _raises(lambda: q.screen(p, {"b": factors["noise1"] > 0}), "boolean")
    with warnings.catch_warnings(record=True) as w5:
        warnings.simplefilter("always")
        q.screen(p, {"x": factors["noise1"]}, horizons=(20,), n_null=5, neutralize_all=False)
    assert any("n_null=5" in str(x.message) for x in w5)
    q.screen(p, {"x": factors["noise1"]}, horizons=(20,), n_null=10, cost_bp=0.5, fire_q=1.0, neutralize_all=False)       # a half basis point and the whole universe are allowed
    # no measurable null: the threshold falls back to 3
    nanf = pd.DataFrame(np.nan, index=p.dates, columns=p.tickers)
    fb = q.screen(p, {"nan": nanf}, horizons=(20,), n_null=10, neutralize_all=False)
    assert "G1 |t|>3.00" in fb.funnel
    print("SCR5 the summary, the gates, the funnel and the neutralised survival equal rebuilds; arguments and the fallback threshold  PASS")


if __name__ == "__main__":
    for fn in (test_controls, test_neutralize, test_chars, test_deploy, test_screen):
        fn()
    print("screen and controls value tests: all passed")
