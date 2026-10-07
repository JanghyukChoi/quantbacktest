"""Equity survivorship tools, tested on data where the right answer is known.

E1 inject_delistings flags the right number of names per year and removes their later prices
E2 known answer: injecting a 5% annual delisting rate at -30% lowers an equal-weight long book by about 5% x 30% = 1.5% a year
E3 universe_coverage counts listed, stopped and in-panel names correctly
E4 survivorship_scenarios reports a lower mean outcome than the baseline when the delisting return is negative
E5 panel_from_long: strict checks, delisting return compounded into the last close, ticker reuse found
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.adapters.long_format import panel_from_long
from pitbacktest.equity import inject_delistings, survivorship_scenarios, survivors_only, universe_coverage
from pitbacktest.equity.master import _clean


def _panel(n=100, T=1300, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=T)
    tick = [f"S{i:03d}" for i in range(n)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.015, (T, n)), axis=0)), index=dates, columns=tick)
    vol = pd.DataFrame(rng.lognormal(12, 0.3, (T, n)), index=dates, columns=tick)
    return q.Panel(close=close, eligible=pd.DataFrame(True, index=dates, columns=tick), volume=vol, market="US")


def test_inject_counts():
    p = _panel()
    s = inject_delistings(p, annual_rate=0.05, seed=3)
    n_flag = int(s.delist_after.values.sum())
    assert 15 <= n_flag <= 25, n_flag                                                  # 5 names a year over 4 full years, about
    for name in s.delist_after.columns[s.delist_after.any()]:
        last = s.delist_after[name][s.delist_after[name]].index[0]
        assert s.close.loc[:last, name].notna().all() and s.close.loc[last + pd.Timedelta(days=1):, name].isna().all()
    print(f"E1 inject_delistings  {n_flag} names flagged, prices removed after the last bar  PASS")


def test_known_answer_bias():
    base_means, inj_means = [], []
    for seed in range(6):
        p = _panel(seed=seed)
        kw = dict(long_q=1.0, short_q=None, hold=1, spread_bp=0.0, benchmark=None, grid=False)
        f = pd.DataFrame(np.random.default_rng(seed).normal(size=p.close.shape), index=p.dates, columns=p.tickers)
        base_means.append(float(q.backtest_portfolio(p, f, **kw).net_returns.mean()) * 252)
        s = inject_delistings(p, annual_rate=0.05, seed=seed)
        inj_means.append(float(q.backtest_portfolio(s, f, delist_return=-0.30, **kw).net_returns.mean()) * 252)
    diff = float(np.mean(inj_means) - np.mean(base_means))
    assert -0.021 < diff < -0.009, diff                                                 # expected about -0.015
    print(f"E2 known answer  5% a year delisted at -30% lowers the long book by {diff * 100:.2f}% a year (expected about -1.5%)  PASS")


def test_coverage():
    raw = pd.DataFrame({"ticker": ["A", "B", "C", "D"], "exchange": ["NYSE"] * 4, "assetType": ["Stock"] * 4,
                        "startDate": ["2005-01-03"] * 4, "endDate": ["2012-06-01", "2026-10-06", "2014-03-03", "2026-10-06"]})
    m = _clean(raw)
    cov = universe_coverage(["B", "D"], m, years=[2012, 2014])
    assert cov.loc[2012, "listed"] == 4 and cov.loc[2012, "stopped_trading"] == 1 and cov.loc[2012, "stopped_in_panel"] == 0
    assert cov.loc[2014, "listed"] == 3 and abs(cov.loc[2014, "share_listed_in_panel"] - 2 / 3) < 1e-9
    assert cov.loc[2014, "stopped_in_panel"] == 0
    print("E3 universe_coverage  counts listed, stopped and in-panel names  PASS")


def test_scenarios_report():
    p = _panel(n=60, T=1000)
    f = pd.DataFrame(np.random.default_rng(0).normal(size=p.close.shape), index=p.dates, columns=p.tickers)

    def evaluate(pan, dr):
        return float(q.backtest_portfolio(pan, f, long_q=1.0, short_q=None, hold=1, spread_bp=0.0, benchmark=None, grid=False,
                                          delist_return=dr).net_returns.mean()) * 252
    out = survivorship_scenarios(p, evaluate, rates=(0.05,), delist_returns=(-0.3,), hazards=("uniform",), n=4)
    assert list(out.columns) == ["hazard", "annual_rate", "delist_return", "baseline", "mean", "p5", "p95", "mean_change"]
    assert float(out["mean_change"].iloc[0]) < 0
    print("E4 survivorship_scenarios  mean outcome below baseline for a negative delisting return  PASS")


def test_long_format():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2020-01-01", periods=300)
    rows = []
    for i in range(12):
        end = 250 if i < 3 else 299
        for t in range(end + 1):
            rows.append({"id": f"P{i:03d}", "ticker": "AAA" if i in (0, 1) else f"T{i:03d}", "date": dates[t],
                         "close": 50.0 + i + 0.1 * t, "volume": 1e6, "dl": -0.3 if (i < 3 and t == end) else np.nan})
    df = pd.DataFrame(rows)
    p = panel_from_long(df, volume_col="volume", delist_return_col="dl", ticker_col="ticker", min_age_days=20)
    assert p.meta["ended_before_end"] == 3 and p.meta["delist_returns_applied"] == 3
    last = dates[250]
    r = p.ret1().loc[last, "P000"]
    assert -0.33 < r < -0.27, r                                                          # the last-day return carries the -30%
    assert p.meta["reused_tickers"] == ["AAA"]
    for bad in (pd.concat([df, df.iloc[:1]]), df.assign(close=-1.0)):
        try:
            panel_from_long(bad)
        except ValueError:
            continue
        raise AssertionError("bad input was accepted")
    print("E5 panel_from_long  delisting return compounded, ticker reuse found, bad input refused  PASS")


def test_survivors_only():
    p = _panel(n=40)
    s = inject_delistings(p, annual_rate=0.1, seed=1)
    keep = survivors_only(s)
    assert 0 < keep.close.shape[1] < s.close.shape[1]
    assert keep.close.notna().iloc[-1].all() and int(keep.delist_after.values.sum()) == 0
    print(f"E6 survivors_only  {s.close.shape[1]} names -> {keep.close.shape[1]} that still trade on the last day  PASS")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    test_inject_counts(); test_known_answer_bias(); test_coverage(); test_scenarios_report(); test_long_format(); test_survivors_only()
    print("equity tests: all passed")
