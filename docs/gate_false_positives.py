"""How often do the six gates let pure noise through? A larger measurement than the 8 factors and 6 signals of `market_validation.py`.

    python docs/gate_false_positives.py crypto|us|krx [N]      # N shuffled factors and N random event signals (default 60)

Factors shuffled across securities each day (so they carry no information) go through `screen`, and random yes/no signals through `backtest_event`. Every one that survives all six
gates is a false discovery. With zero survivors out of N the 95 percent upper bound on the false-discovery rate is about 3 / N (the rule of three); with k survivors it is the
exact binomial bound. This measures the gates on these data and this cost level, not on every possible research design."""
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
from pitbacktest.core.estimators import shuffle_columns        # noqa: E402


def upper_bound(k: int, n: int, level: float = 0.95) -> float:
    """Clopper-Pearson upper bound for k successes in n trials, by bisection (no scipy needed)."""
    from math import comb
    def cdf(p): return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > 1 - level else (lo, mid)
    return hi


if __name__ == "__main__":
    market = sys.argv[1] if len(sys.argv) > 1 else "crypto"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    spec = importlib.util.spec_from_file_location("mv", HERE / "market_validation.py")
    mv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mv)
    P, cfg = mv.build(market)
    c = P.close
    rev = -c.pct_change(5, fill_method=None)
    t0 = time.time()
    surv = g1 = 0
    for b in range(0, n, 10):
        fac = {f"sh{b + i}": shuffle_columns(rev, P.eligible, np.random.default_rng(5000 + b + i)) for i in range(min(10, n - b))}
        sr = q.screen(P, fac, horizons=(5,), primary_h=5, n_null=10, neutralize_all=False, cost_bp=20.0)
        surv += len(sr.survivors)
        g1 += int(sr.funnel.get(next(k for k in sr.funnel if k.startswith("G1")), 0))
        print(f"  screen {b + len(fac)}/{n}: {surv} survived, {g1} passed the first gate", flush=True)
    ev_pass = 0
    rng = np.random.default_rng(99)
    for i in range(n):
        sig = pd.DataFrame(rng.random(c.shape) < 0.02, index=c.index, columns=c.columns) & P.eligible
        ev_pass += int(bool(q.backtest_event(P, sig, horizons=(5,), cost_bp=20.0, neutralize_check=False).gates["passed"]))
    print(f"{cfg['label']}: shuffled factors through `screen`: {surv} of {n} survived all six gates (95% upper bound on the rate {upper_bound(surv, n):.1%}); "
          f"{g1} of {n} passed the first gate alone; random event signals through `backtest_event`: {ev_pass} of {n} passed (upper bound {upper_bound(ev_pass, n):.1%}); {time.time() - t0:.0f} s", flush=True)
