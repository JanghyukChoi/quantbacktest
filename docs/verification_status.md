# What has been verified, and what has not

Written for anyone about to rely on this library or to review it. It states the evidence behind each part and, more important, what is **not** established. It reports
no strategy's performance.

## How things were checked

| Angle | What was done |
|---|---|
| Independent implementation | The engines were written a second time as plain Python loops from the documented rules, sharing no code, and compared (portfolio engine to 1e-17; costs, side costs, trade masks, lot sizes, gross cap likewise) |
| Other libraries | `alpha_beta` against statsmodels (coefficients 1e-12), rank IC against scipy |
| Known answers and planted bugs | 128 tests with answers known by construction. For the newer features each test was also run against deliberately broken code to confirm that it fails; several tests were weak and were rewritten after a planted bug survived |
| Statistical calibration | A factor shuffled across securities is rejected about as often as claimed; Sharpe intervals cover about 93 percent; the Newey-West alpha t over-rejects in finite samples (9.4 percent against 5) |
| Real data, three markets | `docs/market_validation.py` (report in `docs/market_validation.md`): calibration, timing canaries, cost monotonicity, determinism, identity of the realism options with the plain result, data sanity, on KRX, a US sample and Binance |
| Environments | Python 3.10 to 3.14, pandas 2.0.3 to 3.0.6; CI runs six environments |
| Independent review | Several rounds of review by separate AI agents reading the code and the documents. They found more than seventy documentation errors and over a dozen code defects, all fixed. **This is not human review.** |
| Preregistered studies | Korean and US survivorship, crypto cross-section (`studies/`) |

Defects found only by running on real data (synthetic tests had missed them): delisted positions frozen for ever by a trade mask; a contract absent for years that came back
and was treated as one security; false returns of thousands of percent when a consolidation falls inside a trading suspension (Korea).

## Status by part

| Part | Evidence | Status |
|---|---|---|
| `backtest_portfolio`, `backtest_weights`, costs, funding | all angles above | strong |
| `alpha_beta`, IC, Sharpe intervals and paired differences | other libraries, calibration | strong; known finite-sample over-rejection of the alpha t |
| Trade masks, shorting limits, lots and minimum orders, open execution, suspension markdown, gross cap | loops, planted bugs, real data in three markets, review | moderate to strong; see the assumptions below |
| Impact model | hand computations; the visible order book bounds Y (`docs/crypto_impact_check.md`) | moderate: not calibrated against executions |
| Deflated Sharpe, PBO, permutation test, trial ledger | noise fails, a real edge passes, tampering is detected | moderate: little use on real research |
| `screen`, `backtest_event`, the six gates | synthetic tests and shuffled-null thresholds | **weak to moderate**: not part of the three-market validation. In the documentation's own example the event signal passes every gate on synthetic data with no signal |
| Korean adapter | offline fake API, full real history, defects found and fixed | strong |
| Binance adapter, intraday | fake network; minute bars equal Binance's own daily files for two contract-months | moderate |
| US adapter (Tiingo sample) | offline; a 466-ticker random sample | **weak to moderate**: bankruptcies are almost absent (0 of 12 checked); the open, high, low and dividend path has been run on real data only if the line below says so |
| `adapters.long_format` (CRSP, Sharadar, Norgate files) | tests only | **weak**: never run on real paid data |
| `adapters.yfinance` | offline | weak by design: not point in time, and it warns |

## Not verified

1. **No human review and no outside users.** Every review so far was done by AI agents; they can share blind spots with the author.
2. **Korean dividends are not in the prices**; Korean price limits (30 percent assumed), transaction-tax schedule and short-selling ban calendar are not shipped and must be supplied.
3. **`backtest_portfolio` has no impact model** (`backtest_weights` does).
4. **Suspended names:** the markdown is a scenario, not a measurement, and it is asymmetric (gains on a short frozen in a suspended name are not credited, but its gains before the suspension are kept).
   A stuck position frees no capital unless `cap_gross` is used, and then only by scaling the free names.
5. **Statistics:** heavy tails, regime changes and strongly correlated sectors were not tested against the shuffled null; the null is one factor family at one cost level.
6. **Impact coefficient Y** is not calibrated against real executions, only bracketed by the visible book (the rise of Y with size in that note is built in by the interpolation).
7. **The US data path with open, high, low and dividends** (`fetch_symbols(full=True)`) was exercised by tests; its first run on the full real sample is in the report of the next
   validation. Until that report says it passed, treat it as not verified on real data.
8. **Order book, queues, partial fills, liquidation, margin, borrow availability and per-name borrow fees** are not simulated.
9. **Survivorship:** the US free data cannot show bankruptcies; any US result is a lower bound on survivorship bias. Korean data are survivorship-free; Binance keeps delisted contracts.
10. **Operating system:** Linux only.

## If you review

The fastest way to find what the author missed: run `python docs/market_validation.py all`, read `docs/market_validation.md`, then try to break one of
`tests/test_execution.py`, `tests/test_side_costs_shorting.py` or `tests/test_reconcile.py` by changing the code under test; each should fail.
