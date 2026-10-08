# Report: a lower bound on survivorship bias in US equity factors, from free data

Generated from `results.json`. Read `PREREGISTRATION.md` and `AMENDMENT_1.md` first. **Part 1 and 2 are measured. Part 3 is a scenario: what is assumed, not what is known.** The measured difference is a lower bound, because free data lacks most bankruptcies.

## Part 4: the sample

- Frame: 10,688 tickers, 39.6% of them delisted by the last date.
- Drawn (first 468 of the seeded random order): **466 with prices**, 2 with none (KEG1 (delisted), VREXV (delisted)).
- Delisted share: frame 39.6%, drawn 34.8%, obtained 34.5% (161 securities). Flag for missing data concentrated in delisted names (below 70% of the frame's share): **no**. Inconclusive (fewer than 300 with prices): **no**.
- Securities in A_T: 473 (B: 303); ever eligible 308; eligible per day: median 120, range 94 to 160.
- **How many distress-type delistings the free data holds:** 16 of the 170 securities flagged as delisted fell by more than 50% in their last 252 bars (9%).
- Listing windows cut for ticker reuse: 25 tickers. Daily returns above +1000% anywhere in the panel: 22 (data errors are not audited beyond that). Largest absolute daily returns among eligible securities: SDOT 2026-06-26 +247%; ACRS 2021-01-19 +220%; MFA 2020-03-25 +217%; VERU 2022-04-11 +182%; RXDX@20210312 2022-12-07 +166%.
- Securities with a bar, by year: 2013: 216, 2014: 221, 2015: 223, 2016: 235, 2017: 233, 2018: 226, 2019: 230, 2020: 243, 2021: 282, 2022: 285, 2023: 276, 2024: 273, 2025: 282, 2026: 319.

## Parts 1 and 2: 10bp one-way cost

| trial | A_T Sharpe | B Sharpe | B - A_T | 95% interval | p (boot) | neutral B - A_T | 95% interval | reading |
|---|---|---|---|---|---|---|---|---|
| U1_reversal_5d_h5 | -0.23 | -0.28 | -0.04 | [-0.18, +0.09] | 0.513 | -0.07 | [-0.24, +0.10] | not explained |
| U1_reversal_5d_h20 | 0.18 | 0.16 | -0.02 | [-0.17, +0.13] | 0.829 | -0.05 | [-0.23, +0.10] | not explained |
| U2_momentum_240_20_h5 | 0.18 | 0.22 | +0.04 | [-0.09, +0.15] | 0.550 | +0.00 | [-0.15, +0.14] | largely exposure |
| U2_momentum_240_20_h20 | 0.25 | 0.26 | +0.01 | [-0.11, +0.12] | 0.881 | -0.01 | [-0.16, +0.12] | largely exposure |
| U3_low_vol_60_h5 | -0.05 | -0.05 | +0.00 | [-0.13, +0.13] | 0.968 | -0.00 | [-0.13, +0.13] | largely exposure |
| U3_low_vol_60_h20 | 0.01 | -0.01 | -0.02 | [-0.15, +0.10] | 0.669 | -0.04 | [-0.17, +0.09] | not explained |
| U4_low_dollar_volume_h5 | -0.42 | -0.40 | +0.02 | [-0.17, +0.23] | 0.843 | -0.05 | [-0.23, +0.14] | largely exposure |
| U4_low_dollar_volume_h20 | -0.40 | -0.38 | +0.02 | [-0.16, +0.22] | 0.835 | -0.02 | [-0.18, +0.14] | largely exposure |

## Parts 1 and 2: 0bp one-way cost

| trial | A_T Sharpe | B Sharpe | B - A_T | 95% interval | p (boot) | neutral B - A_T | 95% interval | reading |
|---|---|---|---|---|---|---|---|---|
| U1_reversal_5d_h5 | 0.59 | 0.53 | -0.06 | [-0.19, +0.07] | 0.362 | -0.09 | [-0.25, +0.08] | not explained |
| U1_reversal_5d_h20 | 0.54 | 0.52 | -0.02 | [-0.17, +0.13] | 0.761 | -0.06 | [-0.23, +0.10] | not explained |
| U2_momentum_240_20_h5 | 0.28 | 0.31 | +0.04 | [-0.09, +0.15] | 0.599 | -0.00 | [-0.15, +0.14] | largely exposure |
| U2_momentum_240_20_h20 | 0.30 | 0.31 | +0.01 | [-0.11, +0.12] | 0.905 | -0.02 | [-0.16, +0.12] | largely exposure |
| U3_low_vol_60_h5 | 0.01 | 0.01 | -0.00 | [-0.14, +0.12] | 0.935 | +0.00 | [-0.13, +0.13] | largely exposure |
| U3_low_vol_60_h20 | 0.05 | 0.02 | -0.03 | [-0.16, +0.09] | 0.642 | -0.04 | [-0.17, +0.09] | not explained |
| U4_low_dollar_volume_h5 | -0.36 | -0.35 | +0.01 | [-0.18, +0.22] | 0.903 | -0.05 | [-0.23, +0.13] | largely exposure |
| U4_low_dollar_volume_h20 | -0.36 | -0.35 | +0.01 | [-0.16, +0.21] | 0.873 | -0.03 | [-0.18, +0.14] | largely exposure |

**Summary of the measured part.**

- 0 of 16 intervals for B - A_T exclude zero (about 0.8 would by chance alone); they are not independent (two holds and two cost levels of four factors).
- The 95% intervals are about 0.29 Sharpe wide (average over the trials at 10 bp). A difference smaller than that cannot be told from zero in this sample, and that is **not** evidence that the bias is zero.
- Reading rule on the neutralised factors: 6 not explained by the other exposures, 10 largely exposure (a convention for reading; when the raw difference is itself indistinguishable from zero it carries little).

### Market exposure (10 bp, raw factors)

Regression of each net return series on the equal-weight return of A_T's eligible securities (Newey-West).

| trial | beta A | beta B | alpha A (annual, t) | alpha B (annual, t) |
|---|---|---|---|---|
| U1_reversal_5d_h5 | +0.25 | +0.25 | -7.7% (-1.6) | -8.7% (-1.7) |
| U1_reversal_5d_h20 | +0.16 | +0.16 | -0.3% (-0.1) | -0.5% (-0.2) |
| U2_momentum_240_20_h5 | -0.29 | -0.31 | +8.9% (+1.4) | +10.3% (+1.6) |
| U2_momentum_240_20_h20 | -0.28 | -0.30 | +10.4% (+1.7) | +11.1% (+1.8) |
| U3_low_vol_60_h5 | -0.97 | -0.98 | +11.7% (+1.8) | +11.8% (+1.8) |
| U3_low_vol_60_h20 | -0.93 | -0.94 | +13.1% (+2.0) | +12.5% (+1.9) |
| U4_low_dollar_volume_h5 | +0.19 | +0.21 | -9.4% (-2.1) | -9.6% (-2.0) |
| U4_low_dollar_volume_h20 | +0.18 | +0.21 | -8.7% (-2.0) | -9.1% (-2.1) |

## Part 3: a scenario, not a measurement

Delistings are put back into A_T at annual rates [0.01, 0.02, 0.03] of the names alive at the start of each year, with a delisting return of [-0.3, -0.55], drawn uniform or volatile, 10 random draws per cell, at 10bp. The grid was fixed before the result. The columns are `Sharpe(B) - Sharpe(A_scenario)` over the grid cells. **They depend on the assumed rates and returns, and some of Tiingo's delistings are already distress cases, so the scenario partly double counts.**

| trial | measured B - A_T | scenario: lowest cell | mean over cells | highest cell | cells where B - A_scenario > 0 |
|---|---|---|---|---|---|
| U1_reversal_5d_h20 | -0.02 | -0.08 | -0.02 | +0.02 | 4/12 |
| U1_reversal_5d_h5 | -0.04 | -0.17 | -0.07 | -0.01 | 0/12 |
| U2_momentum_240_20_h20 | +0.01 | +0.03 | +0.06 | +0.09 | 12/12 |
| U2_momentum_240_20_h5 | +0.04 | +0.03 | +0.06 | +0.09 | 12/12 |
| U3_low_vol_60_h20 | -0.02 | -0.10 | -0.05 | -0.01 | 0/12 |
| U3_low_vol_60_h5 | +0.00 | -0.10 | -0.03 | +0.00 | 1/12 |
| U4_low_dollar_volume_h20 | +0.02 | +0.06 | +0.11 | +0.17 | 12/12 |
| U4_low_dollar_volume_h5 | +0.02 | +0.04 | +0.08 | +0.14 | 12/12 |

By delisting return (mean over all cells and trials of `Sharpe(B) - Sharpe(A_scenario)`):

- delisting return -30%: +0.009
- delisting return -55%: +0.024
- annual rate 1%: +0.025
- annual rate 2%: +0.019
- annual rate 3%: +0.006
