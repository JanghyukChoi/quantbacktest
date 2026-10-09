"""How often do the six gates pass a signal that is really there? The other half of `gate_false_positives.py`.

    python docs/gate_power.py crypto|us|krx [N]      # N planted factors and N planted event signals per strength (default 20)

A rejection rate of zero on noise says nothing if the gates reject everything, so this plants signals of known strength into the real panel and counts how many get through.

* `screen`: the factor is rho x (cross-sectional rank score of the 5-day forward return) + sqrt(1 - rho^2) x noise, so its daily rank correlation with the outcome is about rho.
  rho = 0 is a control (pure noise, same code path, the default 20 shuffles of the null). Realistic cross-sectional rank correlations are in the range 0.02 to 0.05.
* `backtest_event`: a yes/no signal that fires with probability 2 percent x (1 + lift) before a 5-day move above the day's median and 2 percent x (1 - lift) before one below it.

The forward return is used on purpose: it is how the signal is planted, not how a strategy would see it. The result is the share that survives all six gates (screen) or passes (event),
with the exact 95 percent interval. It is the power at this cost level (20 bp) and these data, not for every design."""
from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import pitbacktest as q                                          # noqa: E402

RHOS = (0.0, 0.005, 0.01, 0.02, 0.05)
LIFTS = (0.0, 0.05, 0.1, 0.2, 0.3)
H = 5


def interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval by bisection."""
    from math import comb
    def cdf(p, kk): return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk + 1))
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return lo
    a = (1 - level) / 2
    up = 1.0 if k == n else solve(lambda p: cdf(p, k) > a)
    low = 0.0 if k == 0 else solve(lambda p: 1 - cdf(p, k - 1) < a)
    return low, up


def rank_score(x: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """Per-day cross-sectional score: the centred rank scaled to unit variance (NaN outside the eligible set)."""
    r = x.where(elig).rank(axis=1, pct=True)
    return (r - 0.5) * np.sqrt(12.0)


def mean_rank_ic(f: pd.DataFrame, fwd: pd.DataFrame, step: int = 5) -> float:
    a, b = f.iloc[::step].rank(axis=1), fwd.iloc[::step].rank(axis=1)
    return float(a.corrwith(b, axis=1).mean())


if __name__ == "__main__":
    market = sys.argv[1] if len(sys.argv) > 1 else "crypto"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    spec = importlib.util.spec_from_file_location("mv", HERE / "market_validation.py")
    mv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mv)
    P, cfg = mv.build(market)
    c = P.close
    fwd = P.forward(H)
    z = rank_score(fwd, P.eligible)
    t0 = time.time()
    print(f"{cfg['label']}: {n} planted factors and {n} planted event signals per strength", flush=True)
    for rho in RHOS:
        facs, ics = {}, []
        for i in range(n):
            rng = np.random.default_rng(7000 + int(rho * 1000) * 100 + i)
            noise = pd.DataFrame(rng.standard_normal(c.shape), index=c.index, columns=c.columns)
            f = (rho * z + np.sqrt(1 - rho ** 2) * noise).where(P.eligible)
            facs[f"p{i}"] = f
            if i < 3: ics.append(mean_rank_ic(f, fwd))
        k = g1 = 0
        for b in range(0, n, 10):
            sr = q.screen(P, {kk: facs[kk] for kk in list(facs)[b:b + 10]}, horizons=(H,), primary_h=H, neutralize_all=False, cost_bp=20.0)
            k += len(sr.survivors)
            g1 += int(sr.funnel.get(next(kk for kk in sr.funnel if kk.startswith("G1")), 0))
        lo, up = interval(k, n)
        print(f"  screen  rho {rho:.3f} (measured rank IC {np.mean(ics):.3f}): {k}/{n} survive all six gates ({lo:.0%}-{up:.0%}); first gate alone {g1}/{n}  [{time.time() - t0:.0f} s]", flush=True)
    up_move = fwd.gt(fwd.median(axis=1), axis=0)
    for lift in LIFTS:
        ok, means = 0, []
        for i in range(n):
            rng = np.random.default_rng(9000 + int(lift * 100) * 100 + i)
            p = 0.02 * (1 + lift * np.where(up_move.to_numpy(), 1.0, -1.0))
            sig = pd.DataFrame(rng.random(c.shape) < p, index=c.index, columns=c.columns) & P.eligible
            r = q.backtest_event(P, sig, horizons=(H,), cost_bp=20.0, neutralize_check=False)
            ok += int(bool(r.gates["passed"]))
            means.append(r.per_horizon[H]["mean_bp"])
        lo, up = interval(ok, n)
        print(f"  event   lift {lift:.2f} (mean trade {np.mean(means):.0f} bp after costs): {ok}/{n} pass all gates ({lo:.0%}-{up:.0%})  [{time.time() - t0:.0f} s]", flush=True)
