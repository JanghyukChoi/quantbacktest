"""The two measurement scripts in docs/, on their pure functions (no network).

D1 slippage   a book of known depth gives the hand-computed average slippage; an order larger than the book is NaN
D2 implied Y  the implied coefficient equals slippage / (sigma * sqrt(share of turnover)), with the sides averaged
D3 spread     the time-weighted half spread of a tiny hand-made quote file, with a gap capped, a crossed quote counted and chunking not changing it
D4 power      the exact binomial interval of the gate power script equals closed forms, and the planted factor has the rank correlation it was built for
"""
from __future__ import annotations
import importlib.util, io, sys, tempfile, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_slippage_and_implied_y():
    m = _load("cis", "docs/crypto_impact_study.py")
    cum = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
    assert abs(m.average_slippage(cum, 0.5e6) - 0.0025) < 1e-12                       # inside the first band: half of 0.5% of the way = 0.25%
    assert abs(m.average_slippage(cum, 2e6) - 0.01) < 1e-9                            # constant depth 1M per 1%: average over 0 to 2M is 1%
    assert np.isnan(m.average_slippage(cum, 6e6)) and np.isnan(m.average_slippage(cum, 0.0))
    assert np.isnan(m.average_slippage(np.array([1e6, 1e6, 3e6, 4e6, 5e6]), 1e5))     # a cumulative depth that does not grow is refused, not guessed
    prof = pd.DataFrame({"notional": [5e6, 4e6, 3e6, 2e6, 1e6, 1e6, 2e6, 3e6, 4e6, 5e6]}, index=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    y = m.implied_y(prof, adv=1e9, sigma_daily=0.04, fractions=(1e-3,))[1e-3]
    assert abs(y - 0.005 / (0.04 * np.sqrt(1e-3))) < 1e-9, y                          # q = 1M, slippage 0.5%
    asym = prof.copy(); asym.loc[[1, 2, 3, 4, 5], "notional"] = [0.5e6, 1e6, 1.5e6, 2e6, 2.5e6]      # the ask side is half as deep: twice the slippage
    ya = m.implied_y(asym, adv=1e9, sigma_daily=0.04, fractions=(1e-3,))[1e-3]
    assert abs(ya - 1.5 * y) < 1e-9, (ya, y)                                          # the sides are averaged: (0.5% + 1.0%) / 2 = 0.75%
    print("D1/D2 average slippage and the implied coefficient equal hand computations; a larger order than the book and a flat book give NaN  PASS")


def test_spread_measure():
    m = _load("csp", "docs/crypto_spread_probe.py")
    # quotes: bid/ask 99.99/100.01 for 10 s (half spread 1.0 bp), then 99.98/100.02 for 20 s (2.0 bp), then a 500 s feed gap capped at 60 s, then the last
    rows = [(0, 99.99, 100.01), (10_000, 99.98, 100.02), (30_000, 99.99, 100.01), (530_000, 99.99, 100.01), (531_000, 100.02, 100.00)]
    df = pd.DataFrame(rows, columns=["event_time", "best_bid_price", "best_ask_price"])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.zip"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("x.csv", df.to_csv(index=False))
        r = m.measure(str(p))
        # intervals: 10 s at 1.0 bp, 20 s at 2.0 bp, 60 s (capped) at 1.0 bp, 1 s at 1.0 bp; the crossed last quote has no following interval
        want = (10 * 1.0 + 20 * 2.0 + 60 * 1.0 + 1 * 1.0) / 91
        assert abs(r["half_mean_bp"] - want) < 1e-6, (r["half_mean_bp"], want)
        assert r["seconds"] == 91 and r["crossed"] == 1
        assert abs(r["half_median_bp"] - 1.0) < 0.011                                 # 71 of 91 seconds are at 1 bp: the median is 1 bp (to the histogram's 0.01)
        # the answer must not depend on how the file is cut into chunks
        big = pd.concat([df] * 1, ignore_index=True)
        orig = pd.read_csv
        pd.read_csv = lambda f, **kw: orig(f, **{**kw, "chunksize": 2})
        try:
            r2 = m.measure(str(p))
        finally:
            pd.read_csv = orig
        assert abs(r2["half_mean_bp"] - r["half_mean_bp"]) < 1e-9 and r2["seconds"] == r["seconds"]
    print("D3 the time-weighted half spread equals a hand computation (gap capped, crossed quote counted), whatever the chunking  PASS")


def test_gate_power_helpers():
    m = _load("gp", "docs/gate_power.py")
    lo, up = m.interval(0, 20)
    assert lo == 0.0 and abs(up - (1 - 0.025 ** (1 / 20))) < 1e-9, up                # zero of n: the upper limit solves (1-p)^n = alpha/2
    lo, up = m.interval(20, 20)
    assert up == 1.0 and abs(lo - 0.025 ** (1 / 20)) < 1e-9, lo                       # all of n: the lower limit solves p^n = alpha/2
    lo, up = m.interval(10, 20)
    assert abs(lo + up - 1.0) < 1e-9 and 0.27 < lo < 0.28                             # symmetric around one half
    rng = np.random.default_rng(0)
    idx = pd.RangeIndex(60); cols = [f"s{i}" for i in range(400)]
    fwd = pd.DataFrame(rng.standard_normal((60, 400)), index=idx, columns=cols)
    elig = pd.DataFrame(True, index=idx, columns=cols)
    z = m.rank_score(fwd, elig)
    assert np.allclose(z.mean(axis=1), np.sqrt(12.0) / (2 * 400), atol=1e-9) and np.allclose(z.std(axis=1, ddof=0), 1.0, atol=0.01)   # mean of pct ranks is (n+1)/2n: a constant per day, which changes no ranking
    for rho in (0.0, 0.1, 0.5):
        noise = pd.DataFrame(rng.standard_normal((60, 400)), index=idx, columns=cols)
        f = rho * z + np.sqrt(1 - rho ** 2) * noise
        assert abs(m.mean_rank_ic(f, fwd, step=1) - rho) < 0.03, (rho, m.mean_rank_ic(f, fwd, step=1))
    print("D4 the exact interval equals closed forms; the planted factor has the rank correlation it was built for  PASS")


if __name__ == "__main__":
    test_slippage_and_implied_y()
    test_spread_measure()
    test_gate_power_helpers()
    print("docs script tests: all passed")
