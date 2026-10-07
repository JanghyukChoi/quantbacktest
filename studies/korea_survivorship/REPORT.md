# Report: survivorship bias in Korean equity factors

Generated from `results.json` by `make_report.py`. Read `PREREGISTRATION.md` first. Days 2013-01-02 to 2026-10-06 (3377 days).

A = every stock listed each day (point in time, delisted included): 3426 securities, median 807 eligible per day.
B = only stocks still listed on the last day: 2766 securities, median 751 eligible per day.

Share of eligible security-days that belong to securities missing from B, by year: 2013: 13%, 2014: 12%, 2015: 13%, 2016: 12%, 2017: 10%, 2018: 10%, 2019: 9%, 2020: 6%, 2021: 6%, 2022: 4%, 2023: 3%, 2024: 1%, 2025: 1%, 2026: 0%.

## 15 bp one-way cost
| trial | A Sharpe | B Sharpe | B - A | A CAGR | B CAGR | B - A (pp) |
|---|---|---|---|---|---|---|
| K1_reversal_5d_h5 | -1.18 | -1.11 | +0.08 | -16.3% | -15.5% | +0.8 |
| K1_reversal_5d_h20 | -0.65 | -0.53 | +0.12 | -5.8% | -4.9% | +0.9 |
| K2_momentum_240_20_h5 | 0.16 | 0.09 | -0.07 | 1.3% | 0.0% | -1.2 |
| K2_momentum_240_20_h20 | 0.29 | 0.21 | -0.08 | 3.5% | 2.1% | -1.4 |
| K3_low_vol_60_h5 | 0.80 | 0.59 | -0.21 | 16.2% | 11.3% | -4.9 |
| K3_low_vol_60_h20 | 0.76 | 0.53 | -0.23 | 14.5% | 9.3% | -5.2 |
| K4_small_size_h5 | -0.77 | -0.57 | +0.20 | -14.1% | -11.2% | +3.0 |
| K4_small_size_h20 | -0.60 | -0.41 | +0.19 | -11.1% | -8.3% | +2.8 |

Mean difference in Sharpe: -0.00 (range -0.23 to +0.20); mean difference in CAGR: -0.7 pp (range -5.2 to +3.0).

## 0 bp cost
| trial | A Sharpe | B Sharpe | B - A | A CAGR | B CAGR | B - A (pp) |
|---|---|---|---|---|---|---|
| K1_reversal_5d_h5 | 0.43 | 0.49 | +0.06 | 5.2% | 6.2% | +1.0 |
| K1_reversal_5d_h20 | 0.04 | 0.15 | +0.11 | -0.1% | 0.9% | +1.0 |
| K2_momentum_240_20_h5 | 0.42 | 0.35 | -0.08 | 6.0% | 4.7% | -1.3 |
| K2_momentum_240_20_h20 | 0.43 | 0.34 | -0.09 | 6.0% | 4.5% | -1.5 |
| K3_low_vol_60_h5 | 0.96 | 0.75 | -0.21 | 20.1% | 15.0% | -5.1 |
| K3_low_vol_60_h20 | 0.87 | 0.63 | -0.23 | 17.2% | 11.9% | -5.4 |
| K4_small_size_h5 | -0.62 | -0.42 | +0.20 | -11.8% | -8.8% | +3.0 |
| K4_small_size_h20 | -0.49 | -0.31 | +0.19 | -9.5% | -6.6% | +2.9 |

Mean difference in Sharpe: -0.01 (range -0.23 to +0.20); mean difference in CAGR: -0.7 pp (range -5.4 to +3.0).
