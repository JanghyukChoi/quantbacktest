# Preregistration: how much does survivorship bias change Korean equity factor results?

**Written and committed before any factor result was computed on this data.** This is a measurement of the bias, not a
search for a strategy: there are no pass or fail gates, only a comparison of the same analysis on two universes.

## Question
Take the same factor portfolios on (A) every Korean stock that was listed on each day, including the ones delisted or
merged later, and (B) only the stocks that are still listed on the last day. How much does B overstate or understate A?

## Data
KRX OpenAPI daily trade data for KOSPI and KOSDAQ, one request per market per trading day, from 2013-01-01 to the last
available day. Adjusted returns come from the change versus the reference price, `close / (close - change) - 1`, which
already reflects splits and rights issues (see `quantbt/adapters/krx.py`). Price return only, no dividends.

## Universe (point in time)
On day t a security is eligible only if, using data up to day t: at least 60 bars of history, trailing 30-day median
traded value of at least 1 billion won, positive volume that day, common shares only (code ends in 0), and not a SPAC,
REIT or fund-like listing (name filter in the adapter). Securities that stopped trading are kept for all days they traded.
**B is the same panel restricted to securities that still have a price on the last day** (`survivors_only`).

## Factors (higher = long) and holding periods
| id | definition |
|---|---|
| K1 reversal_5d | minus the 5-day return |
| K2 momentum_240_20 | return from t-240 to t-20 |
| K3 low_vol_60 | minus the 60-day standard deviation of daily returns |
| K4 small_size | minus the log of market capitalisation |

Holding periods in {5, 20} trading days: **8 trials**. Long the top 20%, short the bottom 20% by cross-sectional rank,
equal weight, dollar neutral, overlapping tranches, signal at the close of t and entry at the close of t+1.
Short selling is not generally available in Korea; the long-short form is used because it isolates the factor, and the
long-only legs can be read from the same runs.

## Costs and delisting
One-way cost of 15 bp as a flat assumption (commission plus the sales tax averaged over both sides; it varies over time,
so a 0 bp run is reported next to it). `delist_return = 0`: the final trading days before a Korean delisting (the
liquidation-trading period) are in the data as real returns, so nothing extra is assumed.

## What will be reported
For each of the 8 trials: net Sharpe and CAGR on A and on B, the difference B minus A, and the mean and range of that
difference over trials; the share of eligible security-days in A that belong to securities missing from B, by year.
Reported whatever the sign is.

## Known limits (written before the result)
- One market, one period; price return only; costs are an assumption; no short-selling constraints modelled.
- B drops securities that were *delisted*, including mergers where the target's holders were paid a premium, so some
  delistings are good outcomes. The bias can have either sign depending on the factor.
- Code reuse is handled by a gap rule (120 days); a few corporate restructurings that change the code look like a
  delisting plus a new listing, which slightly overstates both.
