# Korea survivorship study

Same four factor portfolios, run twice on Korean equities (KOSPI and KOSDAQ, 2013-01-02 to 2026-10-06, 3,377 trading days):

- **A**: every stock listed on each day, including the ones delisted later (point in time, from the official KRX OpenAPI).
- **B**: only the stocks still listed on the last day. This is what a universe taken from today's listing gives you.

The question is how much B differs from A. It is a measurement of the bias, not a search for a strategy, so there are no
pass or fail gates. `PREREGISTRATION.md` was committed before any factor result existed.

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | question, universe rules, 4 factors x 2 holding periods = 8 trials, costs, what will be reported |
| `run_study.py` | the analysis, written from the preregistration (about 3.5 minutes on cached data) |
| `results.json` | raw output |
| `make_report.py`, `REPORT.md` | the tables, generated from `results.json` so no number is typed by hand |

## What the numbers say

Difference in net Sharpe, B minus A, at 15 bp one-way cost (positive means the survivors-only shortcut flatters the result):

| factor | 5-day hold | 20-day hold |
|---|---|---|
| K1 5-day reversal | +0.08 | +0.12 |
| K2 momentum (240 to 20 days) | -0.07 | -0.08 |
| K3 low volatility (60 days) | -0.21 | -0.23 |
| K4 small size | +0.20 | +0.19 |

- **The average is about zero (mean -0.00, range -0.23 to +0.20), and that is the misleading part.** Survivorship bias is
  usually described as "it inflates returns". Here the net effect over the four factors cancels, while each factor moves
  by up to 0.23 Sharpe and 5.2 percentage points of CAGR, in a direction that depends on the factor.
- **Low volatility is understated on the survivors-only universe** (Sharpe 0.80 on A, 0.59 on B for the 5-day hold;
  CAGR 16.2% against 11.3%). **Small size is overstated** (-0.77 on A, -0.57 on B). The sign is the same at both holding
  periods and at 0 bp cost (`REPORT.md`), so it is not noise in one cell.
- The share of eligible security-days that belong to stocks missing from B is 13% in 2013 and falls to under 1% by 2025.
  B is a poor picture of the early years and a good one of recent years, as expected.

## Why the sign differs (a hypothesis, not tested here)

Not tested; this is my reading and the study does not isolate it. Delisted Korean stocks are disproportionately small,
volatile and distressed. A long-short portfolio's short leg then holds many stocks that later fall to delisting, and
dropping them removes some of its best short positions. For low volatility the short leg is the high-volatility names,
so B loses profit. For small size the long leg is the small names, so dropping the ones that collapsed makes the factor
look less bad. A test would split the A portfolio into legs and remove delisted names leg by leg.

## Limits

- One market, one period, one run. There is no significance test and no confidence interval on the differences; the
  reader should not treat 0.2 Sharpe as a precise number.
- Price return only, no dividends. Costs are a flat assumption (15 bp and 0 bp are both reported).
- Shorting is not generally available in Korea. The long-short form isolates the factor; it is not a tradable strategy,
  and **nothing here validates a factor**. Two of the four lose money at 15 bp on both universes.
- B removes names that were *delisted*, which includes mergers where holders were paid a premium, so the sign can differ
  by factor (as it does). Code changes from corporate restructurings can look like a delisting plus a new listing
  (120-day gap rule, see `quantbt/adapters/krx.py`), which slightly overstates how many names disappear.
- Eligibility needs a trailing median traded value of 1 billion won, so most micro-caps are excluded from both A and B.
  The bias in the full market is probably larger than measured here.

## Reproduce

```
export KRX_OPENAPI_KEY=...   # free key from openapi.krx.co.kr
python -c "from quantbt.adapters.krx import fetch_days; fetch_days('2013-01-01','2026-10-06','~/.cache/quantbt/krx')"
python run_study.py && python make_report.py
```

The download is resumable (about two calls per trading day, some 6,800 calls in total) and stops cleanly at the daily quota.
