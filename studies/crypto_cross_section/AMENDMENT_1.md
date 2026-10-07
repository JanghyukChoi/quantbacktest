# Amendment 1: the preregistered cost model was invalid

Written after the preregistered run and before the amended run. The original result is kept unchanged in
`results_preregistered.json` (commit `29522e4`, 2026-10-07 12:24 KST) and is reported next to the amended one.

## What happened
The preregistered run produced net Sharpe ratios far below zero for all 12 trials (best: -0.75; the worst, F1 with
1-day holding: -9.4), while the same portfolios under a flat 2 bp cost with no funding sat near zero. That gap was too
large to believe, so I looked at the cost numbers, which I should have done **before** running. The check only
happened because the result was extremely negative; that is a forking-paths risk and it is stated here on purpose.

## Evidence that the cost model was wrong
Specification as preregistered: one-way cost = 5 bp fee + half the Corwin-Schultz spread (30-day median, floor 1 bp)
+ 5 bp if turnover < 100 million.
- Corwin-Schultz median spread over eligible contract-days: **1.48%** (the median daily high-low range is 8.2%).
- Implied one-way cost for **BTCUSDT: 38 bp** (median). Liquid perpetuals quote well under 1 bp of spread.
- Annual cost drag for F1 / 1-day holding: **51,498 bp** (daily turnover 0.7).

These are orders of magnitude away from real trading costs on liquid Binance perpetuals, so the cost term, not the
factors, drove specification A. I did not establish *why* the estimator fails; daily crypto ranges are dominated by
jumps and volatility clustering, which its assumptions do not cover.

## The change (and nothing else)
One-way cost = 5 bp taker fee + **2 bp fixed half spread** + 5 bp if trailing turnover < 100 million.
Factors, universe, holding periods, portfolio rules, funding, delisting treatment, trials (12) and the five gates are
unchanged. The 2 bp is an assumption, not a measurement, so two cost-independent outputs are added:
- a **cost-sensitivity curve** for the best trial: net Sharpe at flat one-way costs of 0, 2, 5, 10, 20 and 40 bp (with
  funding);
- the **break-even one-way cost**: the largest cost on that grid at which net Sharpe is still positive.

## How to read the two runs
The preregistered run is the only one whose rules were fixed before seeing any result. The amended run is a
**post-hoc correction**: its rules were chosen after I saw the first results, so a pass would be weaker evidence than a
pass in the original design. Both are published; the amended one is the better description of what these factors
would have cost to trade.
