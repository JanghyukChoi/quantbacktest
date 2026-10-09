"""The exact look-ahead check, `Panel.assert_causal`, on signals whose answer is known.

C1 causal      factors built only from the past pass: a return over a past window, a rolling mean and volatility, a cross-sectional rank, a rolling z-score, an expanding mean
C2 leaks       factors that use the future fail: a negative shift, a centred window, a mean over the whole history, a rank across time, a full-sample z-score, a window that ends tomorrow
C3 heuristic   the weak `assert_no_lookahead` misses the blatant leak and flags a legitimate factor with a negative edge: the reason `assert_causal` exists (pinned, so a change is noticed)
C4 truncate    a truncated panel holds exactly the rows up to the cut in every field and keeps the optional ones
C5 arguments   a function that does not return a frame, a panel too short to cut, and the same cuts for the same seed
"""
from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q
from test_reconcile import _panel


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"no ValueError for {needles}")


def _p():
    p, _ = _panel(seed=17, T=520, N=30)
    rng = np.random.default_rng(1)
    return replace(p, volume=p.close * 0 + rng.lognormal(10, 0.3, p.close.shape), mkt_cap=p.close * 1e6)


def test_causal_signals_pass():
    p = _p()
    causal = {
        "past 5-day return": lambda x: -x.close.pct_change(5),
        "past 60-day return skipping 5": lambda x: x.close.shift(5) / x.close.shift(60) - 1,
        "rolling volatility": lambda x: -x.close.pct_change().rolling(20, min_periods=15).std(),
        "cross-sectional rank of a rolling mean": lambda x: x.close.pct_change().rolling(10).mean().rank(axis=1, pct=True),
        "rolling z-score": lambda x: (x.close - x.close.rolling(60).mean()) / x.close.rolling(60).std(),
        "expanding mean deviation": lambda x: x.close / x.close.expanding(min_periods=50).mean() - 1,
        "size and turnover": lambda x: -np.log(x.mkt_cap) + x.volume.rolling(20).mean() / x.mkt_cap,
    }
    for name, fn in causal.items():
        r = p.assert_causal(fn, n_cuts=5)
        assert r["pass"] and r["max_abs_diff"] <= 1e-9 and len(r["cuts"]) == 5, (name, r)
    print(f"C1 {len(causal)} causal factors (past returns, rolling and expanding windows, ranks, z-scores, size and turnover) pass  PASS")


def test_leaks_fail():
    p = _p()
    leaks = {
        "future return (negative shift)": lambda x: x.close.shift(-3) / x.close - 1,
        "centred window": lambda x: x.close.pct_change().rolling(5, center=True).mean(),
        "mean over the whole history": lambda x: x.close - x.close.mean(),
        "full-sample z-score": lambda x: (x.close.pct_change() - x.close.pct_change().stack().mean()) / x.close.pct_change().stack().std(),
        "rank across time": lambda x: x.close.rank(axis=0, pct=True),
        "window that ends tomorrow": lambda x: x.close.pct_change().shift(-1).rolling(4).mean(),
        "a standardisation by the last row's cross-section of the future": lambda x: x.close / x.close.iloc[-1],
    }
    for name, fn in leaks.items():
        r = p.assert_causal(fn, n_cuts=5)
        assert not r["pass"] and r["mismatched_cuts"], (name, r)
    print(f"C2 {len(leaks)} leaking factors (negative shift, centred window, whole-history mean, full-sample z-score, rank across time, last row) are caught  PASS")


def test_heuristic_blind_spots_are_pinned():
    p = _p()
    fut = p.close.shift(-5) / p.close - 1
    neg = -p.close.pct_change(5)
    a = p.assert_no_lookahead(fut, h=5)
    assert p.assert_causal(lambda x: x.close.shift(-5) / x.close - 1)["pass"] is False                # the exact check catches what the heuristic can miss
    assert "pass" in a and "base_bp" in a
    n = p.assert_no_lookahead(neg, h=5)
    assert p.assert_causal(lambda x: -x.close.pct_change(5))["pass"] is True                          # and does not accuse a causal factor, whatever its sign
    print(f"C3 the exact check settles what the heuristic cannot: future-return factor heuristic pass={a['pass']} (base {a['base_bp']:.0f} bp), exact check fails it; causal factor heuristic pass={n['pass']}, exact check passes it  PASS")


def test_truncate():
    p = _p()
    t = p.dates[300]
    tp = p.truncate(t)
    assert tp.dates[-1] == t and len(tp.dates) == 301
    for f in ("close", "eligible", "volume", "mkt_cap"):
        assert getattr(tp, f).equals(getattr(p, f).loc[:t]), f
    pm = replace(p, can_buy=pd.DataFrame(True, index=p.dates, columns=p.tickers), shortable=pd.DataFrame(True, index=p.dates, columns=p.tickers))
    tm = pm.truncate(t)
    assert tm.can_buy.index[-1] == t and tm.shortable.index[-1] == t and tm.can_sell is None
    assert len(p.dates) == 520                                                                       # the original is untouched
    print("C4 a truncated panel holds exactly the rows up to the cut in every field, optional ones included  PASS")


def test_arguments():
    p = _p()
    _raises(lambda: p.assert_causal(lambda x: x.close.to_numpy()), "DataFrame")
    short, _ = _panel(seed=18, T=120, N=30)
    _raises(lambda: short.assert_causal(lambda x: x.close.pct_change(5)), "at least")
    a = p.assert_causal(lambda x: x.close.pct_change(5), seed=4); b = p.assert_causal(lambda x: x.close.pct_change(5), seed=4); c = p.assert_causal(lambda x: x.close.pct_change(5), seed=5)
    assert a["cuts"] == b["cuts"] and a["cuts"] != c["cuts"]
    print("C5 a function that does not return a frame, a panel too short to cut raise; the same seed gives the same cuts  PASS")


if __name__ == "__main__":
    test_causal_signals_pass()
    test_leaks_fail()
    test_heuristic_blind_spots_are_pinned()
    test_truncate()
    test_arguments()
    print("causality tests: all passed")
