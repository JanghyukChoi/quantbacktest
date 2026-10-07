# Crypto cross-section study

Four simple factors on Binance USDT-margined perpetuals, tested with the biases that usually flatter crypto backtests
removed or measured: survivorship (delisted contracts included), universe look-ahead, funding, costs and delisting.

**Result: not validated.** Four of the five preregistered gates pass; the deflated Sharpe does not.

Files, in the order they were written:

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | factors, rules and gates, committed before any factor result existed |
| `run_study.py` | the analysis, written from the preregistration |
| `results_preregistered.json` | the first run, unchanged. Its cost model was invalid (see below) |
| `AMENDMENT_1.md` | what went wrong with the costs, the one change made, and why it is only a post-hoc fix |
| `results_amended.json` | the amended run |
| `make_report.py`, `REPORT.md` | the report, generated from the two result files so no number is typed by hand |

## What the numbers say

- The best of the 12 trials is a 30-day momentum factor that skips the latest 5 days, held 5 days (`F2_momentum_30_5_h5`):
  net Sharpe 0.67, CAGR 23%, maximum drawdown -61%.
- It does not clear the deflated Sharpe gate (0.16 against 0.95 needed). The best of 12 tries would show a Sharpe of about
  1.09 by luck alone with this much dispersion across trials, more than the 0.67 observed.
- It is not stable through time: +41%, -43%, +26%, -3%, +92% and +76% (partial year) in 2021 to 2026.
- About half of the Sharpe is **funding carry**, not price prediction: without funding it is 0.34. The strategy
  received roughly 16% a year in funding, and the sign of funding flips with the market regime.
- Costs are not what kills it. The break-even one-way cost is about 20 bp on the grid tested, against 7 to 12 bp assumed.
- The other factors do not work: 5-day reversal loses money at every holding period, low volatility is negative after costs
  and funding, and the illiquidity factor is about zero.

## How much each shortcut changes the answer (difference in net Sharpe, 12 trials)

| shortcut | mean | smallest | largest |
|---|---|---|---|
| survivors only | +0.05 | -0.12 | +0.30 |
| no funding | +0.10 | -0.33 | +0.41 |
| flat 2 bp cost and no funding | +0.32 | -0.20 | +0.74 |

On average survivorship inflates less than I expected, but it is large for a specific factor: the illiquidity factor
looks +0.30 better on survivors only, because long positions in small contracts that later died disappear from the
sample. Ignoring funding changes the answer by up to 0.4 in either direction, depending on the factor: it hides the
carry that momentum collects and the funding bill that low volatility pays.

## What went wrong, in order

1. I preregistered a cost model based on the Corwin-Schultz spread estimator from daily highs and lows.
2. The first run gave Sharpe ratios far below zero for every trial. The size of the gap to a flat 2 bp cost made me check
   the cost numbers, which I should have done before running: the median spread came out at 1.5% and BTC cost 38 bp
   one way. That is orders of magnitude too high, so the cost term, not the factors, drove the first result.
3. I kept that result unchanged, wrote `AMENDMENT_1.md` and committed it before running again, replaced the estimator with
   a fixed 2 bp half spread, and added a cost-sensitivity curve so the conclusion does not rest on one assumed number.

Because the change was made after seeing the first result, the amended run is a post-hoc correction and a pass in it
would be weaker evidence than a pass in the original design. Here nothing passes completely, so the question does not arise.

## Limits

- Price and volume factors, one exchange, daily bars, one sample (2021 to 2026) with a few distinct regimes.
- Costs are assumptions; the archive has no order book. Capacity is reported as participation (the 95th percentile trade is
  0.40% of a contract's daily turnover at 10 million per leg).
- The settlement price of a delisted contract after its last bar is unknown. The primary case closes it at the last bar.
  In a long-short book the preregistered stress of -30% on delisting helps rather than hurts, because delistings fall mostly
  on the short leg, so that stress is not adverse for this strategy.
- The deflated Sharpe uses the dispersion of Sharpe ratios across all 12 trials. Trials that differ in real quality
  (reversal loses, momentum wins) inflate that dispersion, which makes the gate stricter than a test of 12 equivalent tries.
- PBO from one dataset is noisy (about 0.16 standard deviation under pure noise, see `tests/test_validation.py`).
  0.06 is a good sign, not a proof.
- Reproduce with `python -c "import quantbt.crypto as c; c.fetch_all()"` then `python run_study.py amended`
  (about 25 minutes of downloading, a few minutes of analysis). Re-running later uses more recent data and will differ.
