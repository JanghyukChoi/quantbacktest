# Preregistration: a lower bound on survivorship bias in US equity factors, from free data

**Written and committed before any price was downloaded for this study.** The only data looked at so far is the list of
tickers (no prices) and the 41-stock probe in `docs/tiingo_probe.py`. This is a measurement of bias, not a search for a
strategy: there are no pass or fail gates.

## Question
How much does restricting a US stock universe to today's survivors change factor results, **as far as free data can show**?

Free data cannot show all of it. The probe found that Tiingo returns the history of takeover targets (23 of 29) but of
**none** of 12 bankruptcies and rescue sales. So the universe built here (A_T) still lacks the worst delistings, and the
survivors-only universe (B) is compared against an A_T that is itself too kind. The measured difference B - A_T is therefore a
**lower bound**. The report keeps two things apart: what is **measured** (B - A_T) and what is **assumed** (a scenario that
puts distress delistings back at stated rates). Only the first is evidence.

## Population and sample
- Source of tickers: Tiingo `supported_tickers.zip` (no key), asset type Stock, exchanges NYSE, NASDAQ, NYSE MKT, AMEX, NYSE ARCA.
- Frame: tickers with a listing window ending on or after 2013-01-01, excluding any ticker containing `-` or `.` and any
  five-letter ticker ending in W, U, R or P (warrants, units, rights, preferreds). **10,688 tickers, 40% of them delisted by
  2026-10-06**, 496 with more than one listing window (ticker reuse).
- Sample: a uniformly random permutation of the frame (`numpy.random.default_rng(0)`), downloaded in that order. Any prefix of
  the order is itself a random sample, so stopping early (a quota) does not bias it.
- **Primary sample: the first 500 tickers for which Tiingo returns prices.** Secondary: all obtained, if the free tier allows
  more without payment. A ticker for which Tiingo returns nothing is counted and reported by alive or delisted, not replaced
  silently.
- The sample is of tickers; a ticker with several listing windows contributes each window as a separate security, cut at the
  window boundaries in the ticker list.

## Data
Tiingo end-of-day API, one request per ticker, bars from 2012-01-01 (one year of warm-up) to 2026-10-06. Returns use `adjClose`
(split and dividend adjusted, so total return). Eligibility uses raw `close` times raw `volume`. **Market capitalisation is not
available**, so no factor here uses it.

## Panel and universe (point in time)
Trading calendar: dates on which at least 100 securities of the sample have a bar. On day t a security is eligible only if,
using data up to day t: at least 60 bars since its first bar in its window; trailing 30-bar median dollar volume of at least
2,000,000 USD; raw close of at least 1 USD; positive volume that day. Securities that stopped trading are kept for all days they
traded; the last bar of one that ends more than 5 days before the last calendar date is flagged in `delist_after`.
**B is the same panel restricted to securities that still have a bar within 5 days of the last date** (`survivors_only`).

## Factors (higher = long) and holding periods
| id | definition |
|---|---|
| U1 reversal_5d | minus the 5-day return |
| U2 momentum_240_20 | return from t-240 to t-20 |
| U3 low_vol_60 | minus the 60-day standard deviation of daily returns |
| U4 low_dollar_volume | minus the log of the trailing 60-bar median dollar volume (an illiquidity and size proxy) |

Holding periods {5, 20}: 8 trials. Long the top 20%, short the bottom 20% by cross-sectional rank among eligible securities,
equal weight, dollar neutral, overlapping tranches, signal at the close of t, entry at the close of t+1 (the same engine and
convention as the Korean study).

## Costs and delisting
One-way cost of 10 bp as a flat assumption and a 0 bp run next to it (scalar `spread_bp` is a round trip, so 20 and 0 are
passed). **`delist_return = 0` in the measured part**: a security that stops trading is closed at its last price, which is
right for a takeover and optimistic for a distress delisting.

## What will be reported
**Part 1, measured.** For each of the 8 trials and both cost levels: net Sharpe and CAGR on A_T and on B, B - A_T, and its 95%
interval from a paired stationary block bootstrap (2000 resamples, mean block `round(T ** (1/3))`, seed 0).

**Part 2, exposure.** (a) Market regression: each portfolio's net returns on the equal-weight return of A_T's eligible
securities (Newey-West), reporting beta and annualised alpha for A_T and B. (b) Neutralised factors: each factor residualised
daily (factor and controls rank-normalised within the universe) against the other three styles, run through the same
portfolios, with the same reading rule as the Korean Amendment: if the neutralised B - A_T has the same sign as the raw one and
at least half its size the difference is **not explained** by the other exposures, otherwise **largely exposure**. The
controls for each factor are the styles U1 to U4 other than itself, with U4's style being the log dollar volume.

**Part 3, assumed (a scenario, not a measurement).** The library's `survivorship_scenarios` puts delistings into A_T at annual
rates {1%, 2%, 3%} of names alive at the start of each year, with a delisting return of {-30%, -55%} (Shumway 1997 for NYSE and
AMEX, Shumway and Warther 1999 for Nasdaq, as a range and not as an estimate for this sample), drawn uniformly or tilted toward
volatile names, 10 random draws per cell, at 10 bp. For each trial the report gives the mean and range over the grid cells of
`Sharpe(B) - Sharpe(A_scenario)`. The grid is fixed here and is not tuned to the results. Some of Tiingo's delistings are already
distress cases, so the scenario partly double counts.

**Part 4, audit of the sample.** Sample size and counts; tickers with no data, by alive or delisted; the delisted share of the
obtained sample against that of the frame; for obtained delisted securities, the share whose last year fell by more than 50%
(how many distress-type delistings the free data actually contains); eligible securities per day (median and range); number of
securities per year; windows cut at ticker reuse.

## Conditions under which the study is reported as inconclusive
- fewer than 300 tickers with prices (only Part 4 is reported);
- the delisted share of the obtained sample is below 70% of the frame's 40% (that is, below 28%): missing data is then
  concentrated among delisted names and even the measured part leans toward survivors. The results are reported with that
  flagged.

## Reading rules
- Part 1 is a **lower bound**. No sentence in the report may state the US bias as a level; at most "at least this much, from
  the part of the problem free data can see".
- Part 3 is conditional on the stated assumptions and is labelled as a what-if wherever it appears.
- Comparison with Korea is descriptive only; the two studies have different universes, factors (no market capitalisation here)
  and data, and nothing is pooled.
- Everything is reported whatever the sign. There are 16 comparisons from four factors; read it as four factors, not sixteen.

## Known limits (written before the result)
- A random 500-ticker sample of a population dominated by small names leaves few eligible securities per day (the liquidity
  filter removes most of them), so the long and short legs hold few names and the intervals will be wide. A difference
  smaller than the interval cannot be distinguished from zero, and that is not evidence of no bias.
- Distress delistings are largely absent (see above). Tiingo's last bar for a delisted name may precede further falls.
- Ticker reuse is handled only by the listing windows in the ticker list; a stitched history inside one window (for example a
  renamed company) is kept as one security.
- Total returns here (dividends included) versus price returns in the Korean study; no market capitalisation.
- Price data is vendor data; its errors are not audited beyond the checks in `quantbt.adapters.tiingo`.
- One market, one period, one draw. No out-of-sample data.
