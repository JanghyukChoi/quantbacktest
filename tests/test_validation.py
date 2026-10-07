"""Known-answer tests for deflated Sharpe, PBO and the permutation test.

Noise must not look like skill; a real edge must survive. Fixed seeds, so the results are reproducible.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from quantbt import validation as v


def _noise(T=2000, N=60, seed=1):
    return np.random.default_rng(seed).normal(0, 0.01, (T, N))


def test_dsr_noise_fails_and_real_edge_passes():
    R = _noise()
    best_noise = v.deflated_sharpe(R)
    assert best_noise["dsr"] < 0.95, best_noise          # best of 60 random strategies is luck
    R2 = R.copy()
    R2[:, 7] += 0.003                                    # one strategy with a real edge (per-period Sharpe about 0.3)
    real = v.deflated_sharpe(R2)
    assert real["best"] == 7 and real["dsr"] > 0.95, real
    print(f"T1 DSR  noise {best_noise['dsr']:.3f} (<0.95)  real edge {real['dsr']:.3f} (>0.95)  PASS")


def test_pbo_noise_near_half_and_real_edge_near_zero():
    # One PBO value is noisy (about 0.16 standard deviation across random datasets), so check the average.
    p_noise = float(np.mean([v.pbo_cscv(_noise(seed=k), blocks=12)["pbo"] for k in range(20)]))
    assert 0.4 < p_noise < 0.62, p_noise
    R = _noise(); R[:, 7] += 0.003
    p_real = v.pbo_cscv(R, blocks=12)["pbo"]
    assert p_real < 0.1, p_real
    print(f"T2 PBO  noise mean {p_noise:.2f} (about 0.5)  real edge {p_real:.2f} (<0.1)  PASS")


def test_permutation_noise_not_significant_and_edge_significant():
    rng = np.random.default_rng(5)
    # search: best Sharpe over 20 lookbacks of a trend rule on a single return stream
    def search(r):
        best = -9.0
        for w in range(2, 42, 2):
            sig = np.sign(np.convolve(r, np.ones(w) / w, mode="full")[: len(r)])
            pos = np.concatenate([[0.0], sig[:-1]])
            x = pos * r
            best = max(best, x.mean() / (x.std() + 1e-12))
        return best
    # Under the null, p-values are spread out. A single p can be small by chance, so look at several noise series.
    ps = [v.permutation_test(np.random.default_rng(100 + k).normal(0, 0.01, 1500), search, n=100, seed=k)["p"] for k in range(10)]
    assert 0.3 < float(np.mean(ps)) < 0.7 and sum(p < 0.05 for p in ps) <= 2, ps
    p_noise = float(np.mean(ps))
    noise = rng.normal(0, 0.01, 1500)
    regime = np.repeat(rng.choice([-1, 1], 30), 50) * 0.004      # persistent drift regimes: a trend rule has an edge
    p_edge = v.permutation_test(noise + regime, search, n=200, seed=1)["p"]
    assert p_edge < 0.05, p_edge
    print(f"T3 permutation  noise mean p={p_noise:.2f} (about 0.5)  trending p={p_edge:.3f} (<0.05)  PASS")


if __name__ == "__main__":
    test_dsr_noise_fails_and_real_edge_passes()
    test_pbo_noise_near_half_and_real_edge_near_zero()
    test_permutation_noise_not_significant_and_edge_significant()
    print("validation tests: all passed")
