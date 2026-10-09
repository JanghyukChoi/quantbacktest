"""How strict is the deflated Sharpe, measured against the usual practice? A simulation: no data needed.

    python docs/dsr_power.py

One strategy among 40 has a true annual Sharpe S, the other 39 have none; returns are normal, 252 days a year. The question is how often the tool picks the true one AND calls it real
(deflated Sharpe above 0.95), for several sample lengths. Two yardsticks are printed beside it: a plain t-test on the same strategy with no allowance for having tried 40 (t above 1.65), and
the hurdle that Harvey, Liu and Zhu (2016) suggest for a new factor after the multiple testing that the literature has done (t above 3.0, where t = Sharpe x sqrt(years)). With S = 0 the
share of runs that call the best of the 40 real is the false-discovery rate (a test run at the 5 percent level should show about 5 percent or less)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pitbacktest as q                                        # noqa: E402

PPY, TRIALS, REPS = 252, 40, 150


def one(years: float, S: float, rng) -> tuple[bool, bool, bool, bool]:
    T = int(years * PPY)
    sd = 0.01
    R = rng.normal(0.0, sd, (T, TRIALS))
    R[:, 0] += S / np.sqrt(PPY) * sd                            # the strategy with a true edge
    d = q.validation.deflated_sharpe(R, periods_per_year=PPY)
    sr0 = R[:, 0].mean() / R[:, 0].std(ddof=1) * np.sqrt(PPY)
    t0 = sr0 * np.sqrt(years)
    tbest = (R.mean(0) / R.std(0, ddof=1)).max() * np.sqrt(PPY) * np.sqrt(years)      # the best of the 40, what a researcher would report
    return d["best"] == 0 and d["dsr"] > 0.95, d["dsr"] > 0.95, t0 > 1.65, t0 > 3.0, tbest > 1.65, tbest > 3.0


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    print(f"{TRIALS} strategies tried, {REPS} runs per cell. Share of runs in which the strategy with the true edge is picked and called real:")
    print("years  true Sharpe |  deflated Sharpe | plain t > 1.65 (no allowance for 40 tries) | t > 3.0 (Harvey-Liu hurdle)")
    for years in (3, 5, 10, 20):
        for S in (0.5, 1.0, 1.5, 2.0):
            r = np.array([one(years, S, rng) for _ in range(REPS)])
            print(f"{years:5d}  {S:11.1f} | {r[:, 0].mean():16.0%} | {r[:, 2].mean():42.0%} | {r[:, 3].mean():27.0%}")
    r0 = np.array([one(10, 0.0, rng) for _ in range(REPS * 2)])
    print(f"no edge at all (10 years, {REPS * 2} runs), the best of the 40 is called real by: deflated Sharpe {r0[:, 1].mean():.1%} | "
          f"plain t > 1.65 {r0[:, 4].mean():.1%} | t > 3.0 {r0[:, 5].mean():.1%}")
