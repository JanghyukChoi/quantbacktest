# Amendment 1: how precise is the difference, and is it exposure?

**Written and committed before the analyses below were run.** The original preregistration, its code (`run_study.py`)
and its results (`results.json`, `REPORT.md`) are not changed. Everything here is **post hoc**: it was added after the
first result was seen, so it is exploratory and carries no gates. It answers two questions the first report left open.

## Why
The first report gave the difference in Sharpe between A (every stock listed each day) and B (only today's survivors)
as a single number per trial, with no sense of its precision, and could not say whether the difference reflects
survivorship itself or only a different exposure of the two universes (for example to the market).

## Part I: precision of the difference
For each of the 8 original trials and both cost levels (16 comparisons), a paired stationary block bootstrap of
`Sharpe(B) - Sharpe(A)` on the two net return series (`quantbt.analytics.sharpe_diff_ci`): 2000 resamples,
mean block length `round(T ** (1/3))`, seed 0, 95% percentile interval, bootstrap p-value (two-sided, approximate).
With 16 comparisons, about one interval is expected to exclude zero by chance alone if there were no real difference.
All 16 are reported; none is singled out.

## Part II: exposure
**(a) Market regression.** The net return series of the A portfolio and of the B portfolio are each regressed on the same
regressor, the equal-weight return of the eligible stocks of universe A (`PortfolioResult.alpha_beta`, Newey-West).
Reported: beta and annualised alpha with t-statistics, for A and for B, per trial and cost level.

**(b) Neutralised factors.** For each factor the daily cross-section is residualised (`quantbt.core.controls.neutralize`,
the factor and the controls both rank-normalised each day within the universe being tested) against the **other three** style exposures and
not against itself:

| factor | controls removed |
|---|---|
| K1 reversal_5d | momentum 240-20 return, 60-day volatility, log market cap |
| K2 momentum_240_20 | 5-day return, 60-day volatility, log market cap |
| K3 low_vol_60 | 5-day return, momentum 240-20 return, log market cap |
| K4 small_size | 5-day return, momentum 240-20 return, 60-day volatility |

The neutralised factor is run through the same long-short portfolio (same 8 trials, both cost levels, both universes)
and compared the same way. A and B are neutralised separately, each within its own cross-section.

**Pre-declared reading rule** (for each factor, holding period and cost level): if the neutralised `B - A` has the same
sign as the raw one and at least half its size, the survivorship effect is **not explained** by the other exposures;
if it is smaller than half or has the opposite sign, it is **largely exposure**. The counts of each are reported. The
rule is a convention for reading, not a test.

## Reproduction check
The 8 original trials are recomputed in the same run; their Sharpe and CAGR must equal `results.json` to 1e-9, otherwise
the new numbers are not comparable and the discrepancy is reported instead of the new analysis.

## Limits (written before the result)
- Neutralising against volatility, size and momentum removes part of what defines K3 and K4; a neutralised factor is a
  different object, not a cleaner version of the same one.
- The market regression uses one regressor; it does not remove industry or style exposures.
- Same sample, same period, same data as the original; nothing here is out of sample.
- The bootstrap interval is somewhat too narrow in finite samples (about 93% coverage at a nominal 95% in
  `tests/test_analytics.py` B1).
