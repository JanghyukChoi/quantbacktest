# Report: cross-sectional factors on Binance perpetuals

Generated from `results_preregistered.json` and `results_amended.json` by `make_report.py`.
Read `PREREGISTRATION.md` first, then `AMENDMENT_1.md` (the preregistered cost model turned out to be invalid).

Signal dates 2021-01-01 to 2026-10-06 (2105 days). Universe A (point in time, delisted
included): 893 contracts, 373 of them delisted, median 95 eligible per day.
Universe B (survivors only): 520 contracts, median 77 eligible per day.

## Verdict

| run | best trial | verdict |
|---|---|---|
| preregistered (invalid cost model) | F4_illiquidity_h10 | not validated |
| amended (fixed 2 bp half spread) | F2_momentum_30_5_h5 | **not validated** |

### Amended run, gates on the best trial
| gate | criterion | value | result |
|---|---|---|---|
| G1 | net Sharpe >= 0.5 | 0.667 | pass |
| G2 | deflated Sharpe >= 0.95 | 0.156 | **fail** |
| G3 | PBO < 0.20 | 0.063 | pass |
| G4 | positive in >= 60% of calendar years | 0.667 | pass |
| G5 | net Sharpe > 0 under all stresses | 0.601 | pass |

### Preregistered run, gates on its best trial
| gate | criterion | value | result |
|---|---|---|---|
| G1 | net Sharpe >= 0.5 | -0.749 | **fail** |
| G2 | deflated Sharpe >= 0.95 | 0.000 | **fail** |
| G3 | PBO < 0.20 | 0.010 | pass |
| G4 | positive in >= 60% of calendar years | 0.167 | **fail** |
| G5 | net Sharpe > 0 under all stresses | -0.783 | **fail** |

## Best trial of the amended run: F2_momentum_30_5_h5
- Net Sharpe 0.67, CAGR 23%, max drawdown -61%, daily one-way turnover 0.15.
- Deflated Sharpe 0.156: the Sharpe a lucky best-of-12 would show is 1.09, above the observed 0.67.
- PBO 0.063 (in-sample best Sharpe 0.82, out-of-sample Sharpe of that same column 0.53).
- Calendar years: 2021: +41%, 2022: -43%, 2023: +26%, 2024: -3%, 2025: +92%, 2026: +76% (277 days).
- Funding paid: -1618 bp per year (negative = received). Costs: 1098 bp per year.
- Stress (net Sharpe): delist_return_-0.30 0.73, taker_fee_8bp 0.60, entry_lag_2 0.63.
- **Cost sensitivity** (flat one-way cost, with funding): 0bp 0.89, 2bp 0.84, 5bp 0.78, 10bp 0.67, 20bp 0.44, 40bp -0.00. Break-even on this grid: 20 bp one way.
- Capacity: with 10M per leg the 95th percentile trade is 0.40% of that contract's daily turnover; the account size at which it reaches 1% is about 25M.

## All 12 trials, net Sharpe (amended run)
| trial | A primary | B survivors only | C no funding | D flat 2 bp, no funding | cost bp/yr | funding bp/yr | delistings held |
|---|---|---|---|---|---|---|---|
| F1_reversal_5d_h1 | -1.51 | -1.53 | -1.48 | -0.78 | 5124 | +221 | 53 |
| F1_reversal_5d_h5 | -0.98 | -1.05 | -0.94 | -0.52 | 2217 | +200 | 109 |
| F1_reversal_5d_h10 | -0.75 | -0.77 | -0.69 | -0.43 | 1127 | +201 | 121 |
| F2_momentum_30_5_h1 | 0.39 | 0.33 | 0.09 | 0.45 | 2423 | -1632 | 58 |
| F2_momentum_30_5_h5 | 0.67 | 0.55 | 0.34 | 0.52 | 1098 | -1618 | 112 |
| F2_momentum_30_5_h10 | 0.49 | 0.40 | 0.15 | 0.28 | 777 | -1570 | 115 |
| F3_low_vol_30_h1 | -0.68 | -0.48 | -0.27 | -0.14 | 1043 | +2642 | 52 |
| F3_low_vol_30_h5 | -0.43 | -0.35 | -0.04 | 0.04 | 601 | +2401 | 83 |
| F3_low_vol_30_h10 | -0.35 | -0.27 | 0.02 | 0.08 | 473 | +2197 | 88 |
| F4_illiquidity_h1 | -0.10 | 0.19 | 0.19 | 0.36 | 810 | +1186 | 52 |
| F4_illiquidity_h5 | 0.05 | 0.30 | 0.31 | 0.43 | 542 | +1015 | 85 |
| F4_illiquidity_h10 | 0.19 | 0.32 | 0.43 | 0.52 | 439 | +878 | 92 |

## How much does each shortcut change the answer? (amended run, difference in net Sharpe versus A, over 12 trials)
| shortcut | mean | smallest | largest |
|---|---|---|---|
| survivors only | +0.05 | -0.12 | +0.30 |
| no funding | +0.10 | -0.33 | +0.41 |
| flat 2 bp cost, no funding | +0.32 | -0.20 | +0.74 |

## Preregistered run, all 12 trials (invalid cost model, kept for the record)
| trial | A primary | B survivors only | C no funding | D flat 2 bp, no funding | cost bp/yr | funding bp/yr | delistings held |
|---|---|---|---|---|---|---|---|
| F1_reversal_5d_h1 | -9.36 | -8.83 | -9.34 | -0.78 | 51498 | +221 | 53 |
| F1_reversal_5d_h5 | -5.74 | -5.36 | -5.70 | -0.52 | 22387 | +200 | 109 |
| F1_reversal_5d_h10 | -3.82 | -3.58 | -3.76 | -0.43 | 11434 | +201 | 121 |
| F2_momentum_30_5_h1 | -3.59 | -3.46 | -3.90 | 0.45 | 23834 | -1632 | 58 |
| F2_momentum_30_5_h5 | -1.31 | -1.30 | -1.64 | 0.52 | 10898 | -1618 | 112 |
| F2_momentum_30_5_h10 | -1.00 | -0.99 | -1.33 | 0.28 | 7762 | -1570 | 115 |
| F3_low_vol_30_h1 | -2.07 | -1.76 | -1.66 | -0.14 | 10036 | +2642 | 52 |
| F3_low_vol_30_h5 | -1.28 | -1.14 | -0.90 | 0.04 | 5890 | +2401 | 83 |
| F3_low_vol_30_h10 | -1.05 | -0.92 | -0.69 | 0.08 | 4699 | +2197 | 88 |
| F4_illiquidity_h1 | -1.67 | -1.16 | -1.37 | 0.36 | 7096 | +1186 | 52 |
| F4_illiquidity_h5 | -1.05 | -0.64 | -0.79 | 0.43 | 4749 | +1015 | 85 |
| F4_illiquidity_h10 | -0.75 | -0.49 | -0.51 | 0.52 | 3879 | +878 | 92 |
