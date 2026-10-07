# Report: Amendment 1 (post hoc)

Generated from `results_amendment1.json`. Read `AMENDMENT_1.md` first: everything here is exploratory and was added after the first result was seen. Bootstrap: 2000 paired stationary resamples, seed 0.

Reproduction check: the 8 original trials were recomputed and differ from `results.json` by at most 0.0e+00 (Sharpe and CAGR).

## 15bp one-way cost

| trial | A Sharpe | B Sharpe | B - A | 95% interval | p (boot) | neutral B - A | 95% interval | reading |
|---|---|---|---|---|---|---|---|---|
| K1_reversal_5d_h5 | -1.18 | -1.11 | +0.08 | [+0.01, +0.16] | 0.028 | +0.04 | [-0.05, +0.13] | not explained |
| K1_reversal_5d_h20 | -0.65 | -0.53 | +0.12 | [+0.04, +0.19] | 0.000 | +0.04 | [-0.05, +0.14] | largely exposure |
| K2_momentum_240_20_h5 | 0.16 | 0.09 | -0.07 | [-0.16, +0.02] | 0.103 | -0.15 | [-0.27, -0.04] | not explained |
| K2_momentum_240_20_h20 | 0.29 | 0.21 | -0.08 | [-0.17, +0.00] | 0.056 | -0.17 | [-0.28, -0.06] | not explained |
| K3_low_vol_60_h5 | 0.80 | 0.59 | -0.21 | [-0.30, -0.13] | 0.000 | -0.25 | [-0.35, -0.16] | not explained |
| K3_low_vol_60_h20 | 0.76 | 0.53 | -0.23 | [-0.32, -0.15] | 0.000 | -0.28 | [-0.37, -0.19] | not explained |
| K4_small_size_h5 | -0.77 | -0.57 | +0.20 | [+0.11, +0.29] | 0.000 | +0.03 | [-0.06, +0.11] | largely exposure |
| K4_small_size_h20 | -0.60 | -0.41 | +0.19 | [+0.09, +0.28] | 0.000 | +0.01 | [-0.07, +0.09] | largely exposure |

## 0bp one-way cost

| trial | A Sharpe | B Sharpe | B - A | 95% interval | p (boot) | neutral B - A | 95% interval | reading |
|---|---|---|---|---|---|---|---|---|
| K1_reversal_5d_h5 | 0.43 | 0.49 | +0.06 | [-0.01, +0.14] | 0.083 | +0.02 | [-0.07, +0.12] | largely exposure |
| K1_reversal_5d_h20 | 0.04 | 0.15 | +0.11 | [+0.04, +0.19] | 0.002 | +0.03 | [-0.06, +0.12] | largely exposure |
| K2_momentum_240_20_h5 | 0.42 | 0.35 | -0.08 | [-0.16, +0.01] | 0.074 | -0.15 | [-0.27, -0.04] | not explained |
| K2_momentum_240_20_h20 | 0.43 | 0.34 | -0.09 | [-0.18, -0.00] | 0.044 | -0.17 | [-0.28, -0.06] | not explained |
| K3_low_vol_60_h5 | 0.96 | 0.75 | -0.21 | [-0.30, -0.13] | 0.000 | -0.26 | [-0.35, -0.16] | not explained |
| K3_low_vol_60_h20 | 0.87 | 0.63 | -0.23 | [-0.32, -0.16] | 0.000 | -0.28 | [-0.38, -0.19] | not explained |
| K4_small_size_h5 | -0.62 | -0.42 | +0.20 | [+0.10, +0.29] | 0.000 | +0.02 | [-0.07, +0.11] | largely exposure |
| K4_small_size_h20 | -0.49 | -0.31 | +0.19 | [+0.09, +0.28] | 0.000 | +0.01 | [-0.07, +0.09] | largely exposure |

## Summary

- Of 16 comparisons, **12** have a 95% interval for `B - A` that excludes zero (about 0.8 would by chance alone if there were no real difference).
- Reading rule on the neutralised factors: **9** not explained by the other exposures, **7** largely exposure.

## Market exposure of the long-short portfolios (15 bp, raw factors)

Regression of each net return series on the equal-weight return of the eligible stocks of universe A (Newey-West).

| trial | beta A | beta B | alpha A (annual, t) | alpha B (annual, t) |
|---|---|---|---|---|
| K1_reversal_5d_h5 | +0.22 | +0.23 | -17.4% (-5.0) | -16.4% (-4.7) |
| K1_reversal_5d_h20 | +0.13 | +0.13 | -5.9% (-2.8) | -5.0% (-2.4) |
| K2_momentum_240_20_h5 | +0.22 | +0.24 | +2.2% (+0.5) | +1.0% (+0.2) |
| K2_momentum_240_20_h20 | +0.22 | +0.23 | +4.2% (+0.9) | +2.9% (+0.6) |
| K3_low_vol_60_h5 | -0.57 | -0.58 | +18.9% (+4.1) | +14.7% (+3.1) |
| K3_low_vol_60_h20 | -0.56 | -0.58 | +17.3% (+3.8) | +12.8% (+2.8) |
| K4_small_size_h5 | +0.37 | +0.39 | -14.6% (-3.4) | -11.2% (-2.6) |
| K4_small_size_h20 | +0.36 | +0.37 | -11.3% (-2.7) | -8.1% (-1.9) |
