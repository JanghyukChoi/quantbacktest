"""The gates that judge an event signal, on hand-made inputs. A mutation run found that 22 of 24 changes to this file went unnoticed.

GT1 structure   fires per day, active days, name counts, shares, the longest run (among the 30 most frequent names), turnover, the eight top names; and the empty case
GT2 subperiod   the two halves, their means in bp and t, and whether the signs match
GT3 yearly      years with under 40 bars are skipped; a year counts as positive only when its mean is above zero
GT4 holdout     the cut is the last date minus the months; the cut date itself is in-sample; a side under 30 bars is not measured
GT5 gates       each of the six gates at its edge, the order of failure, and the switches
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from pitbacktest.core.estimators import newey_west_t
from pitbacktest.core.gates import GateConfig, fire_structure, oos_holdout, run_gates, subperiod, yearly


def test_structure():
    D, N = 100, 40
    fire = np.zeros((D, N), bool)
    for j in range(30):                                                        # the thirty most frequent names fire on alternate days: counts 40, 39, ... 11, runs of 1
        fire[0:2 * (40 - j):2, j] = True
    for j in range(30, 40):                                                    # the rest fire in one block each: counts 10, 9, ... 1, runs as long as the counts
        fire[:10 - (j - 30), j] = True
    dates = pd.bdate_range("2020-01-01", periods=D); tickers = pd.Index([f"S{j:02d}" for j in range(N)])
    st = fire_structure(fire, dates, tickers)
    cnt = fire.sum(axis=0)
    order = np.argsort(-cnt, kind="stable")
    tot = int(fire.sum()); n_day = fire.sum(axis=1); act = n_day[n_day > 0]
    assert st["n_names"] == N and abs(st["fires_per_day"] - act.mean()) < 1e-12 and abs(st["active_days_pct"] - len(act) / D * 100) < 1e-12
    assert abs(st["top10_share"] - np.sort(cnt)[::-1][:10].sum() / tot) < 1e-12 and abs(st["max_name_share"] - cnt.max() / D) < 1e-12
    assert list(st["top_names"]) == [tickers[i] for i in order[:8]] and len(st["top_names"]) == 8
    top30 = order[:30]
    def run(col):
        best = cur = 0
        for x in fire[:, col]:
            cur = cur + 1 if x else 0; best = max(best, cur)
        return best
    assert st["max_consecutive"] == max(run(c) for c in top30)
    assert max(run(c) for c in range(N)) > st["max_consecutive"], "the planted long run belongs to a name outside the top thirty"
    turn = []
    for i in range(1, D):
        a, b = fire[i - 1], fire[i]
        if a.sum() and b.sum():
            turn.append(1 - (a & b).sum() / (a | b).sum())
    assert abs(st["daily_turnover"] - np.mean(turn)) < 1e-12
    empty = fire_structure(np.zeros((10, 4), bool), dates[:10], tickers[:4])
    assert empty["n_names"] == 0 and empty["fires_per_day"] == 0.0 and empty["active_days_pct"] == 0.0 and empty["max_consecutive"] == 0 and empty["top_names"] == {} and empty["note"] == "no firings"
    assert np.isnan(empty["top10_share"]) and np.isnan(empty["max_name_share"]) and np.isnan(empty["daily_turnover"])
    # fewer days than names: the turnover runs over the days
    f3 = np.random.default_rng(8).random((30, 40)) < 0.2
    tn = []
    for i in range(1, 30):
        a, b = f3[i - 1], f3[i]
        tn.append(1 - (a & b).sum() / (a | b).sum())
    assert abs(fire_structure(f3, dates[:30], tickers)["daily_turnover"] - np.mean(tn)) < 1e-12
    one = np.zeros((10, 4), bool); one[3, 2] = True                                     # a single fire in the whole sample is still a signal with one name
    s1 = fire_structure(one, dates[:10], tickers[:4])
    assert s1["n_names"] == 1 and s1["max_consecutive"] == 1 and s1["top_names"] == {"S02": 1} and s1["top10_share"] == 1.0
    # a day with no fires between two days with fires is skipped in the turnover, not counted as 100 percent
    f2 = np.zeros((5, 3), bool); f2[0, 0] = f2[1, 0] = f2[3, 0] = f2[4, 0] = True
    assert fire_structure(f2, dates[:5], tickers[:3])["daily_turnover"] == 0.0
    print("GT1 fire structure equals the definitions; only the 30 most frequent names count for runs; turnover skips empty days  PASS")


def test_subperiod_yearly_holdout():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2018-01-01", periods=1000)
    v = rng.normal(0.003, 0.01, 1000)
    sp = subperiod(v, dates)
    for lab, part in (("front", v[:500]), ("back", v[500:])):
        mu, _, t, n = newey_west_t(part, lag=21)
        assert abs(sp[lab]["mean_bp"] - mu * 1e4) < 1e-9 and abs(sp[lab]["t"] - t) < 1e-12 and sp[lab]["n"] == n == 500
    assert sp["sign_match"] is True
    flip = np.concatenate([np.full(500, 0.001), np.full(500, -0.001)]) + rng.normal(0, 0.0001, 1000)
    assert subperiod(flip, dates)["sign_match"] is False
    both_neg = subperiod(-np.abs(rng.normal(0.001, 0.0001, 1000)), dates)
    assert both_neg["sign_match"] is True                                              # two negative halves agree
    empty = subperiod(np.array([]), dates)
    assert empty["sign_match"] is False and empty["front"]["n"] == 0
    # yearly: 2018 has 260 bars, the last year only 30
    d2 = pd.bdate_range("2018-01-01", periods=260 * 2 + 30)
    v2 = np.concatenate([np.full(260, 0.001), np.full(260, -0.001), np.full(30, 0.001)]) + rng.normal(0, 0.0005, 550)
    y = yearly(v2, d2)
    assert y["total"] == 2 and y["positive"] == 1 and y["positive_ratio"] == 0.5, y            # the 30-bar year is skipped
    zero = yearly(np.zeros(260), pd.bdate_range("2018-01-01", periods=260))
    assert zero["total"] == 1 and zero["positive"] == 0                                         # a mean of exactly zero is not positive
    d3 = pd.bdate_range("2018-01-01", periods=300)
    just = yearly(np.full(300, 0.001), d3)
    sizes = pd.Series(1, index=d3).groupby(d3.year).sum()
    assert just["total"] == int((sizes >= 40).sum())
    assert yearly(np.array([]), d3)["total"] == 0 and np.isnan(yearly(np.array([]), d3)["positive_ratio"])
    # holdout
    ho = oos_holdout(v, dates, months=12)
    cut = dates[999] - pd.DateOffset(months=12)
    assert ho["cutoff"] == str(cut.date())
    isv, oosv = v[dates[:1000] <= cut], v[dates[:1000] > cut]
    for lab, part in (("IS", isv), ("OOS", oosv)):
        mu, _, t, n = newey_west_t(part, lag=21)
        assert ho[lab]["n"] == len(part) == n and abs(ho[lab]["mean_bp"] - mu * 1e4) < 1e-9 and abs(ho[lab]["t"] - t) < 1e-12
    d4 = pd.bdate_range("2020-01-01", periods=300)
    c4 = d4[-1] - pd.DateOffset(months=3)
    assert c4 in d4, "the cut date is itself a trading date in this construction"
    h4 = oos_holdout(np.arange(300, dtype=float), d4, months=3)
    assert h4["IS"]["n"] == int((d4 <= c4).sum()) and h4["OOS"]["n"] == int((d4 > c4).sum())       # the cut date is in-sample
    thin = oos_holdout(rng.normal(0, 0.01, 300), d4, months=1)
    assert thin["OOS"]["n"] < 30 and np.isnan(thin["OOS"]["mean_bp"]) and np.isnan(thin["OOS"]["t"])
    print("GT2/3/4 subperiod, yearly and holdout equal their definitions at the edges  PASS")


def _base_args(**kw):
    dates = pd.bdate_range("2015-01-01", periods=5 * 260)
    rng = np.random.default_rng(3)
    v = rng.normal(0.002, 0.003, len(dates))
    args = dict(daily_excess=v, dates=dates, t_stat=4.0, net_bp=5.0, rho=0.8)
    args.update(kw)
    return args


def test_gates():
    base = run_gates(**_base_args())
    assert base["passed"] is True and base["failed_at"] is None and len(base["steps"]) == 5
    cfg = GateConfig()
    # G1: strictly above the threshold
    assert run_gates(**_base_args(t_stat=3.0))["failed_at"] == "G1 statistical threshold"
    assert run_gates(**_base_args(t_stat=3.0001))["passed"] is True and run_gates(**_base_args(t_stat=-3.5))["passed"] is True
    assert run_gates(**_base_args(t_stat=2.0, cfg=GateConfig(null_threshold=1.5)))["passed"] is True
    # G2: net profit above zero; unknown is not profit unless the switch is off
    assert run_gates(**_base_args(net_bp=0.0))["failed_at"] == "G2 net profit after costs"
    assert run_gates(**_base_args(net_bp=float("nan")))["failed_at"] == "G2 net profit after costs"
    assert run_gates(**_base_args(net_bp=-5.0, cfg=GateConfig(require_net_positive=False)))["passed"] is True
    assert run_gates(**_base_args(net_bp=float("nan"), cfg=GateConfig(require_net_positive=False)))["passed"] is True
    # G3: signs of the halves
    v = np.concatenate([np.full(650, 0.002), np.full(650, -0.002)]) + np.random.default_rng(1).normal(0, 0.0003, 1300)
    r = run_gates(**_base_args(daily_excess=v))
    assert r["failed_at"] == "G3 sub-period sign" and run_gates(**_base_args(daily_excess=v, cfg=GateConfig(require_subperiod_sign=False)))["failed_at"] != "G3 sub-period sign"
    # G4: |rho| at least the minimum, unknown fails
    assert run_gates(**_base_args(rho=0.5))["passed"] is True and run_gates(**_base_args(rho=-0.5))["passed"] is True
    assert run_gates(**_base_args(rho=0.49))["failed_at"] == "G4 decile monotonicity" and run_gates(**_base_args(rho=float("nan")))["failed_at"] == "G4 decile monotonicity"
    assert run_gates(**_base_args(rho=0.2, cfg=GateConfig(min_monotonicity=0.1)))["passed"] is True
    # G5: share of positive years at least 0.6: three of five years
    d = pd.bdate_range("2015-01-01", periods=5 * 260)
    yrs = np.array(d.year)
    def by_year(signs):
        out = np.zeros(len(d))
        for k, y in enumerate(sorted(set(yrs))):
            out[yrs == y] = signs[k] * 0.002
        return out + np.random.default_rng(2).normal(0, 0.0004, len(d))
    three = run_gates(**_base_args(daily_excess=by_year([1, -1, 1, -1, 1]), dates=d, cfg=GateConfig(require_subperiod_sign=False)))
    assert three["yearly"]["positive_ratio"] == 0.6 and three["passed"] is True
    two = run_gates(**_base_args(daily_excess=by_year([1, -1, -1, -1, 1]), dates=d, cfg=GateConfig(require_subperiod_sign=False)))
    assert two["failed_at"] == "G5 yearly consistency"
    # G6: firing structure at its edges (<= for the top-name share, >= for the turnover)
    Dn = len(d)
    rng = np.random.default_rng(4)
    fire = rng.random((Dn, 30)) < 0.1
    tickers = pd.Index([f"S{j:02d}" for j in range(30)])
    st = fire_structure(fire, d, tickers)
    ok = run_gates(**_base_args(fire=fire, tickers=tickers, cfg=GateConfig(max_single_name_share=st["max_name_share"], min_turnover=st["daily_turnover"])))
    assert ok["passed"] is True and len(ok["steps"]) == 6
    over = run_gates(**_base_args(fire=fire, tickers=tickers, cfg=GateConfig(max_single_name_share=st["max_name_share"] - 1e-9, min_turnover=0.0)))
    under = run_gates(**_base_args(fire=fire, tickers=tickers, cfg=GateConfig(max_single_name_share=1.0, min_turnover=st["daily_turnover"] + 1e-9)))
    assert over["failed_at"] == under["failed_at"] == "G6 firing structure"
    # the first failure is reported, later gates still run
    both = run_gates(**_base_args(t_stat=1.0, net_bp=-1.0))
    assert both["failed_at"] == "G1 statistical threshold" and [s["pass"] for s in both["steps"]][:2] == [False, False] and both["passed"] is False
    assert abs(cfg.fallback_t - 3.0) < 1e-12 and base["threshold"] == 3.0
    print("GT5 each gate passes and fails at its edge; switches; the first failure is reported  PASS")


if __name__ == "__main__":
    test_structure(); test_subperiod_yearly_holdout(); test_gates()
    print("gate value tests: all passed")
