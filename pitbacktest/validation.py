"""Overfitting checks for a set of tried strategies: deflated Sharpe, PBO (CSCV) and a permutation test.

All three answer the same question from different angles: *given that I tried many variants and kept the best one,
is the result still believable?* None of them proves a strategy works; each can only fail to reject.

Inputs are returns matrices of shape (T, N): T periods, N tried variants (parameter sets, factors, ...).
Everything is per-period unless `periods_per_year` is given.

References
  Bailey, Lopez de Prado (2014), The Deflated Sharpe Ratio.
  Bailey, Borwein, Lopez de Prado, Zhu (2015), The Probability of Backtest Overfitting (CSCV).
  Masters, Monte Carlo permutation of the whole search.
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Callable

import numpy as np

_N = NormalDist()
_EULER = 0.5772156649015329


def sharpe(R: np.ndarray, periods_per_year: int = 1) -> np.ndarray:
    """Sharpe ratio of each column (risk-free 0), annualised by sqrt(periods_per_year)."""
    R = np.asarray(R, dtype=np.float64)
    return R.mean(axis=0) / R.std(axis=0, ddof=1) * math.sqrt(periods_per_year)


def deflated_sharpe(R: np.ndarray, *, trials: int | None = None, periods_per_year: int = 1) -> dict:
    """Probability that the best column's true Sharpe is above what the best of `trials` random strategies
    would show by luck. `trials` defaults to the number of columns; pass a larger number if you also tried
    variants you did not keep in R (and say so honestly)."""
    R = np.asarray(R, dtype=np.float64)
    T, n = R.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        sr_all = R.mean(axis=0) / R.std(axis=0, ddof=1)
    if not np.isfinite(sr_all).any():
        raise ValueError("no column of R has a usable Sharpe ratio: every column contains NaN or inf, or has zero variance (a strategy that never traded). "
                         "Columns with a NaN are ignored, so drop or fill the NaN rows if all of them have one")
    j = int(np.nanargmax(np.where(np.isfinite(sr_all), sr_all, np.nan)))
    x = R[:, j]
    sr = float(sr_all[j])
    z = (x - x.mean()) / x.std(ddof=0)
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    k = trials or n
    var = float(np.nanvar(sr_all, ddof=1)) if n > 1 else 0.0
    sr0 = math.sqrt(var) * ((1 - _EULER) * _N.inv_cdf(1 - 1 / k) + _EULER * _N.inv_cdf(1 - 1 / (k * math.e))) if k > 1 else 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    dsr = _N.cdf((sr - sr0) * math.sqrt(T - 1) / denom)
    a = math.sqrt(periods_per_year)
    return {"best": j, "sharpe": sr * a, "sharpe_luck_benchmark": sr0 * a, "trials": k, "skew": skew,
            "kurtosis": kurt, "T": T, "dsr": float(dsr)}


def pbo_cscv(R: np.ndarray, *, blocks: int = 16, periods_per_year: int = 1, max_splits: int | None = None,
             seed: int = 0) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    The sample is cut into `blocks` contiguous blocks; every way of choosing half of them as in-sample is tried.
    PBO is the share of splits where the in-sample best column ranks at or below the median out of sample.
    Around 0.5 means the in-sample winner carries no information about out-of-sample rank."""
    R = np.asarray(R, dtype=np.float64)
    T = (len(R) // blocks) * blocks
    R = R[len(R) - T:]
    L = T // blocks
    S = np.stack([R[i * L:(i + 1) * L].sum(0) for i in range(blocks)])
    Q = np.stack([(R[i * L:(i + 1) * L] ** 2).sum(0) for i in range(blocks)])

    def sr(idx):
        c = L * len(idx)
        s, q = S[list(idx)].sum(0), Q[list(idx)].sum(0)
        return (s / c) / np.sqrt((q - s * s / c) / (c - 1))

    combos = list(itertools.combinations(range(blocks), blocks // 2))
    if max_splits and len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]
    logits, is_best, oos_best = [], [], []
    a = math.sqrt(periods_per_year)
    for isb in combos:
        osb = tuple(i for i in range(blocks) if i not in isb)
        x, y = sr(isb), sr(osb)
        j = int(np.nanargmax(x))
        w = (1 + int((y < y[j]).sum())) / (len(y) + 1)
        logits.append(math.log(w / (1 - w)))
        is_best.append(float(x[j] * a))
        oos_best.append(float(y[j] * a))
    logits = np.asarray(logits)
    return {"pbo": float((logits <= 0).mean()), "splits": len(logits), "is_best_sharpe_mean": float(np.mean(is_best)),
            "oos_of_best_sharpe_mean": float(np.mean(oos_best))}


def permutation_test(returns: np.ndarray, search: Callable[[np.ndarray], float], *, n: int = 1000, seed: int = 0,
                     block: int = 1) -> dict:
    """Monte Carlo permutation of the **whole search**.

    `search(returns)` must run your entire selection procedure on a returns (or price-change) array and return
    the best statistic it found (for example the best Sharpe over a parameter grid). The returns are shuffled
    (in blocks of `block` periods, to keep some autocorrelation if you need it), the whole search is repeated, and
    p = (1 + number of shuffles with a best statistic >= the real one) / (n + 1).
    The null is "the timing has no value". Note that shuffling also destroys volatility clustering."""
    r = np.asarray(returns, dtype=np.float64)
    real = float(search(r))
    rng = np.random.default_rng(seed)
    nb = len(r) // block
    cnt = 0
    nulls = []
    for _ in range(n):
        if block == 1:
            rs = r[rng.permutation(len(r))]
        else:
            order = rng.permutation(nb)
            rs = np.concatenate([r[i * block:(i + 1) * block] for i in order] + [r[nb * block:]])
        v = float(search(rs))
        nulls.append(v)
        cnt += v >= real
    return {"real": real, "p": (1 + cnt) / (n + 1), "n": n, "null_mean": float(np.mean(nulls)),
            "null_p95": float(np.percentile(nulls, 95))}
