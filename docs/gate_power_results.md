# Do the six gates pass a signal that is really there? (`docs/gate_power.py`, 2026-10-09)

The false-discovery measurement (`gate_false_positives.py`) shows that noise does not get through. That alone would also be true of a tool that rejects everything, so this
measures the other side: signals of known strength are planted into the real panels and counted.

* `screen`: factor = rho x (rank score of the 5-day forward return) + sqrt(1 - rho^2) x noise. Its daily rank correlation with the outcome (the "measured" column) is about rho.
  rho = 0 is a control. Realistic cross-sectional rank correlations are 0.02 to 0.05.
* `backtest_event`: a yes/no signal that fires with probability 2 percent x (1 + lift) before a 5-day move above the day's median and 2 percent x (1 - lift) before one below it.
  The mean trade (after 20 bp cost) is what the planted signal is worth per trade.
* 20 signals per cell, cost 20 bp, horizon 5 days, default `n_null`. Cells show how many of 20 survive all six gates; the 95 percent interval of a cell with 20 of 20 is 83 to 100 percent,
  of 0 of 20 is 0 to 17 percent, so single cells are coarse.

## `screen`: factors that survive all six gates (of 20)

| rho (measured rank IC) | Binance USDT-M 2021- | US sample 2013- | KRX 2016- |
|---|---|---|---|
| 0 (control, noise) | 0 (-0.000) | 0 (0.001) | 0 (-0.001) |
| 0.005 | 0 (0.002) | 0 (0.007) | 0 (0.004) |
| 0.01 | 5 (0.011) | 0 (0.009) | 0 (0.009) |
| 0.02 | 18 (0.021) | 3 (0.022) | 20 (0.019) |
| 0.05 | 20 (0.049) | 20 (0.048) | 20 (0.048) |

The first gate alone passes 18 to 20 of 20 from rho 0.01 on (and 6 to 20 of 20 at 0.005), so the later gates are what separate a weak signal from noise: at rho 0.005 to 0.01 the
first gate lets signals through that the later ones reject.

## `backtest_event`: signals that pass every gate (of 20); mean trade in brackets

| lift | Binance USDT-M | US sample | KRX |
|---|---|---|---|
| 0 (control) | 0 (-22 bp) | 0 (11 bp) | 0 (-16 bp) |
| 0.05 | 3 (26 bp) | 5 (27 bp) | 17 (12 bp) |
| 0.10 | 14 (59 bp) | 15 (46 bp) | 20 (38 bp) |
| 0.20 | 20 (123 bp) | 20 (85 bp) | 20 (95 bp) |
| 0.30 | 20 (190 bp) | 20 (127 bp) | 20 (151 bp) |

## What this says

* The gates do not reject everything. All six controls (pure noise, 120 signals in all) are rejected, and planted signals pass from a rank correlation of about 0.02 (a typical
  published size) on in Korea and in crypto, and from about 0.05 in the US sample.
* Between 0.01 and 0.02 the outcome depends on the market: the same rank correlation passes 20 of 20 in Korea (about 2,000 names), 18 of 20 in crypto and 3 of 20 in the US sample
  (465 names). The more names and days, the weaker the signal that can be told from noise. At rho 0.005 none is found in any market.
* The event gates pass most signals once the mean trade is about 45 to 60 bp after costs in crypto and the US sample (at 26 and 27 bp: 3 and 5 of 20), and already at 12 bp in Korea
  (17 of 20), where there are far more trades. What the gates see is the mean trade against its noise, so the number of trades matters as much as the size of the edge.

## What this does not say

* These are **upper bounds on real power**. The planted signal is stationary, equally strong every day and in every name, and its error is independent from day to day. Real signals
  decay, come in regimes, and are clustered across related names, which makes them harder to tell from noise than the plant.
* One cost level (20 bp), one horizon (5 days), one factor construction, 20 signals per cell: the cells are coarse, and 18 of 20 against 20 of 20 is not a difference.
* It does not show that a signal that passes is tradable: the gates test for evidence against noise after costs, not for capacity or for survival of the effect out of sample.
* Labels in the first runs of `gate_power.py` printed rho 0.005 as "0.01" and lift 0.05 as "0.1" (the format had too few digits); the tables above use the true values, in the order
  the script runs them. The Korean log has the corrected labels. The crypto and US numbers are the runs of that day.

Raw logs: `gate_power_results.txt`.
