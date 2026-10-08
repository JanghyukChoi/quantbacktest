# US survivorship study (free data)

How much does restricting a US stock universe to today's survivors change factor results, **as far as free data can show**? A random
sample of 466 US-listed tickers (2013 to 2026) is run twice: with the securities that were delisted since (A_T, from a free Tiingo
account) and with only the ones still listed (B). Four price and volume factors, two holding periods, the same engine as the Korean study.

Free data lacks most bankruptcies (a probe found none of 12; `docs/survivorship.md`), so **A_T is itself too kind and the measured
difference is a lower bound**. The report keeps what is measured apart from what is assumed.

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | question, sample, universe rules, factors, the four parts, reading rules, limits: committed before any price was downloaded |
| `AMENDMENT_1.md` | the free tier's 500-symbol monthly limit stopped the download at 466 tickers; written before any analysis |
| `fetch_sample.py` | downloads the seeded random sample (resumable, waits out the hourly limit) |
| `run_study.py`, `results.json` | the analysis (about 2 minutes on 12 cores) and its raw output |
| `make_report.py`, `REPORT.md` | the tables, generated from `results.json` |

## What the numbers say

- **The measured difference is indistinguishable from zero in this sample.** For all 16 comparisons (4 factors x 2 holding periods x 2 cost
  levels) `Sharpe(B) - Sharpe(A_T)` lies between -0.07 and +0.04, and every 95% interval includes zero. The intervals are about 0.29 Sharpe wide,
  so a difference smaller than roughly 0.15 cannot be seen here. **That is not evidence that the bias is zero.**
- **Why not: A_T holds few distress delistings.** Of 170 securities flagged as delisted, 16 (9%) fell by more than half in their last year. The
  takeovers that make up most of the rest tend to leave at a premium, which does not hurt a survivors-only result the way a bankruptcy does.
- **The scenario (assumed, not measured)** puts distress delistings back at 1%, 2% and 3% a year with returns of -30% and -55%. Then the survivors-only
  shortcut overstates momentum and the illiquidity factor by 0.03 to 0.17 Sharpe and understates reversal and low volatility by up to 0.17 in some cells.
  The sign depends on the factor, as it did for Korean stocks, and the size depends on rates and returns nobody measured for this sample.
- Exposure: beta to the equal-weight market is almost identical for A_T and B (differences of 0.01 to 0.03); the neutralised comparison is read with the rule
  fixed in advance (6 not explained, 10 largely exposure), which carries little when the raw difference is itself indistinguishable from zero.

## Limits (see also the preregistration)
- 466 tickers, a median of 120 eligible securities a day: the legs hold few names and the intervals are wide. The 500-ticker extension planned after
  1 November 2026 (`AMENDMENT_1.md`) will narrow them only slightly.
- Distress delistings are largely absent and Tiingo's last bar of a delisted name may precede further falls (`delist_return = 0` in the measured part).
- Total returns here (dividends included), no market capitalisation, so the size factor is a dollar-volume proxy; not comparable one to one with the Korean study.
- Vendor price data is not audited beyond the adapter's checks. The report lists the largest daily returns among eligible securities (up to +247%) and
  22 daily returns above +1000% in the whole panel; a sensitivity analysis without them was not part of the plan and was not run.
- One market, one period, one draw. No out-of-sample data. Nothing here validates a factor.

## Reproduce
```
export TIINGO_API_KEY=...          # free key; the free tier allows 50 requests an hour and 500 distinct symbols a month
python fetch_sample.py             # about 10 hours for 500 tickers because of the hourly limit; resumable
python run_study.py && python make_report.py
```
