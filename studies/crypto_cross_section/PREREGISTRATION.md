# Preregistration: cross-sectional factors on Binance perpetuals, and how much the biases matter

**Written and committed before any factor result was computed.** The commit time of this file precedes the
analysis run. Whatever comes out is judged by the criteria below, the criteria are not changed afterwards, and the
result is published even if no factor passes.

## Questions
1. Do four simple, price- and volume-only factors survive honest checks on crypto perpetuals?
2. How much do survivorship, funding and flat-cost assumptions change the answer? The biases are *measured*, not
   just listed.

## Data
- Binance USDT-margined perpetuals from the public archive (`data.binance.vision`), topped up from the public API
  for contracts that still trade. **Every contract that ever traded is included, delisted ones too.**
  Pegged coins, fiat pairs and index perpetuals are excluded (the regex in `quantbt/crypto/panel.py`).
- Daily UTC bars. Zero-volume bars are dropped (frozen prices after a halt).
- Signal dates: 2021-01-01 to the last complete UTC day available when the analysis is run (the date is recorded
  in `results.json`). Earlier data are used only for warm-up.

## Universe (point in time)
On day t a contract is eligible only if, using data up to and including day t: it has at least 90 daily bars, its
trailing 30-day **median** USDT turnover is at least 20 million, and day t's bar is valid. No top-N cap.

## Factors (direction fixed in advance; higher value = long)
| id | definition |
|---|---|
| F1 reversal_5d | minus the 5-day return |
| F2 momentum_30_5 | return from t-30 to t-5 (skips the latest 5 days) |
| F3 low_vol_30 | minus the 30-day standard deviation of daily returns |
| F4 illiquidity | minus the log of the trailing 30-day mean USDT turnover (long the less liquid) |

Holding periods h in {1, 5, 10} days. **Trials N = 12** (4 factors x 3 holding periods). Nothing else is tried.

## Portfolio
Rank eligible contracts each day; long the top 20% and short the bottom 20%, equal weight inside each leg, each leg
sums to 1 (dollar neutral). Overlapping tranches of h days. Signal at the UTC close of day t, entry at the close of
day t+1 (`entry_lag = 1`, deliberately conservative).

## Costs and frictions (primary specification)
- One-way cost = 5 bp taker fee + half the Corwin-Schultz spread (30-day median, floor 1 bp) + 5 bp if trailing
  turnover is below 100 million (`liquidity_cost_bp(panel, taker_fee_bp=5, thin_usd=1e8)`).
- Funding is charged from the archive (long pays a positive rate, short receives).
- Delisting: a contract that stops trading is closed at its last real bar; `delist_return = 0` in the primary
  specification (the favourable case). The number of delisting events held is reported.

## Bias measurement (reported for all 12 trials, not part of the verdict)
Net Sharpe under: (A) primary; (B) **survivors only** (contracts that still trade today); (C) A without funding;
(D) A with a flat 2 bp one-way cost and no funding. The difference B - A, C - A and D - A is the measured inflation
from each shortcut.

## Verdict gates (applied to the best of the 12 trials by full-sample net Sharpe in A)
| | criterion |
|---|---|
| G1 | net Sharpe >= 0.5 |
| G2 | deflated Sharpe >= 0.95, with 12 trials |
| G3 | PBO < 0.20 (CSCV, 16 blocks, all 12,870 splits, 12 trials) |
| G4 | positive net return in at least 60% of calendar years (a partial year counts if it has 180+ days) |
| G5 | net Sharpe stays > 0 under all of: delist_return = -0.30; taker fee 8 bp; entry_lag = 2 |

**All five must pass to be "validated".** If any fails, the failed item is reported as it is.

## Known limits (written before the result)
- Price and volume factors only, one exchange, daily bars.
- Costs are estimates; the archive has no order book. Capacity is reported as participation, not modelled as impact.
- The settlement price of a delisted contract after its last bar is unknown; the primary case is optimistic.
- The 12 trials are correlated, which makes DSR and PBO approximate. PBO from one dataset is noisy (about 0.16
  standard deviation under pure noise in `tests/test_validation.py`), so a PBO near the 0.20 line is not decisive.
- Crypto regimes change quickly; a pass or a fail describes this sample only.
