"""Our figures against another library's (empyrical), on many random return series.

The point is an outside anchor: `empyrical` is widely used, was written by other people, and defines the usual figures. Where we agree, we agree to the last digit; where a definition
differs, the difference is stated and pinned here (an exact relation, not a vague "close").

DF1 agree      Sharpe, Sortino, CAGR, volatility, maximum drawdown (the starting capital is a peak in both), value at risk, expected shortfall, tail ratio, beta, information ratio
DF2 differ     skewness (we use the population moment, pandas the sample-corrected one: an exact factor), up and down capture (a ratio of mean returns here, of annualised growth there)
Skipped with a note when `empyrical` is not installed."""
from __future__ import annotations
import math, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest import robustness as rb
from pitbacktest.portfolio import metrics

try:
    import empyrical as ep
except ImportError:                                                       # pragma: no cover
    ep = None


def _series(seed, n, kind):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    x = {"normal": rng.normal(0.0004, 0.01, n), "t4": rng.standard_t(4, n) * 0.008 + 0.0003, "skewed": -np.abs(rng.normal(0, 0.01, n)) + 0.006,
         "drawdown_first": np.r_[-0.12, rng.normal(0.0003, 0.008, n - 1)]}[kind]
    return pd.Series(x, index=idx), pd.Series(rng.normal(0.0003, 0.009, n), index=idx)


def test_agreement():
    if ep is None:
        print("DF1 skipped: empyrical is not installed")
        return
    worst = {}
    for seed in range(24):
        for kind in ("normal", "t4", "skewed", "drawdown_first"):
            n = (300, 777, 1500)[seed % 3]
            r, b = _series(seed, n, kind)
            m = metrics(r.to_numpy(), r.index, 252)
            k = rb.risk_profile(r)
            pairs = {"Sharpe": (m["Sharpe"], ep.sharpe_ratio(r)), "Sortino": (m["Sortino"], ep.sortino_ratio(r)), "CAGR": (m["CAGR"], ep.annual_return(r)),
                     "vol": (m["vol"], ep.annual_volatility(r)), "MDD": (m["MDD"], ep.max_drawdown(r)), "VaR95": (k["var_95"], ep.value_at_risk(r)),
                     "CVaR95": (k["cvar_95"], ep.conditional_value_at_risk(r)), "tail ratio": (k["tail_ratio"], ep.tail_ratio(r))}
            ab = q.analytics.alpha_beta(r, b)
            pairs["beta"] = (list(ab["betas"].values())[0]["beta"], ep.alpha_beta(r, b)[1])
            mr = rb.market_relative(r, b)
            pairs["information ratio"] = (mr["information_ratio"], ep.excess_sharpe(r, b) * math.sqrt(252))
            for name, (a, e) in pairs.items():
                d = abs(a - e) / max(1.0, abs(e))
                worst[name] = max(worst.get(name, 0.0), d)
                assert d < 1e-9, (name, seed, kind, n, a, e)
    print("DF1 24 x 4 random series: " + ", ".join(f"{k} {v:.0e}" for k, v in worst.items()) + " (largest relative differences from empyrical)  PASS")


def test_stated_differences():
    if ep is None:
        print("DF2 skipped: empyrical is not installed")
        return
    for seed in range(10):
        r, b = _series(50 + seed, 800, "t4")
        k = rb.risk_profile(r)
        n = len(r)
        assert abs(r.skew() - k["skew"] * math.sqrt(n * (n - 1)) / (n - 2)) < 1e-12                 # pandas' skew is the sample-corrected moment: an exact factor of ours
        mr = rb.market_relative(r, b)
        up, dn = b > 0, b < 0
        assert abs(mr["up_capture"] - r[up].mean() / b[up].mean()) < 1e-12 and abs(mr["down_capture"] - r[dn].mean() / b[dn].mean()) < 1e-12      # ratio of mean returns
        assert abs(ep.up_capture(r, b) - ep.annual_return(r[up]) / ep.annual_return(b[up])) < 1e-9 and abs(ep.down_capture(r, b) - ep.annual_return(r[dn]) / ep.annual_return(b[dn])) < 1e-9
    print("DF2 skewness differs from pandas' by exactly sqrt(n(n-1))/(n-2); capture ratios are means here and annualised growth in empyrical (both formulas pinned)  PASS")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    test_agreement()
    test_stated_differences()
    print("differential tests: all passed")
