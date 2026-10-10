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
| Metamorphic relations | `tests/test_metamorphic.py`: with no expected number written anywhere, 12 random panels each are put through input changes whose effect is known: the order or the names of the securities, a monotone transform of the factor, adding a never-eligible security, a price-level change, a positive scaling of returns, a higher cost, **cutting the data after a date** and **scrambling the factor and the eligibility after a date** (no position held up to the date may change). They found no defect in the final code; they do catch a one-bar lookahead in the return, in the eligibility and in the factor when it is put in on purpose |
| Calibration by simulation | the false-discovery rate of each permutation test is measured on information-free signals with persistent fires or fixed baskets (60 samples each): the earlier nulls called 17 and 20 percent significant at the nominal 5, the ones in the code 3 and 5 percent; `docs/gate_false_positives.py`, `docs/gate_power.py`, `docs/dsr_power.py` do the same for the six gates and the deflated Sharpe |
| Independent review | Several rounds of review by separate AI agents reading the code and the documents. They found more than seventy documentation errors and over a dozen code defects, all fixed. **This is not human review, and it is not independent in the strong sense: the reviewing agents are the same kind of model as the author, so they can share blind spots.** Checks that do not depend on anyone's judgement (worked answers, simulations with a known truth, metamorphic relations, deliberately broken code) carry more weight than the reviews, and none of them replaces someone reading the assumptions: a reviewer should be asked to read `docs/verification_status.md` and try to break it, not to confirm it. |
| Preregistered studies | Korean and US survivorship, crypto cross-section (`studies/`) |

Defects found only by running on real data (synthetic tests had missed them): delisted positions frozen for ever by a trade mask; a contract absent for years that came back
and was treated as one security; false returns of thousands of percent when a consolidation falls inside a trading suspension (Korea).

## Known simplifications, with their size

- **Costs are charged on target changes, not on drifting back to target.** The engines return each bar to the target weights but charge trading cost only on the change of the target. For a
  5-bar-hold reversal portfolio, measured on Binance and on KRX, returning to target after prices moved would add 2 to 3 percent to the one-way turnover (about 10 to 14 bp a year at a 10 bp
  round-trip spread). For a book whose targets rarely change it is a larger share of a small number.
- **Dividends, taxes and interest.** The KRX series is a price series (no dividends); a dividend tax, a capital-gains tax and interest on idle cash are not modelled. The review page says
  when returns are price only.
- **What a permutation does and does not say.** The factor permutation gives each security another security's factor history and compares gross Sharpe ratios: it asks whether the factor's
  link to the securities' own returns carries information. The event permutation moves the whole table of fires in time: it asks whether *when* the signal fires carries information. Both were
  calibrated on information-free persistent signals (60 and 60 simulated samples: about 5 percent called significant at the nominal 5, against 17 and 20 percent for the earlier nulls). Neither says
  whether the strategy is tradable, and a signal whose only edge is *which* securities it picks (a persistent tilt to high-drift names) is not distinguished from picking those names at any time.
- **The walk-forward runs on a matrix of whole-sample returns** of each setting, which is exact because a setting's signal uses only the past; it does not re-estimate anything inside a fold.
- **Brinson attribution is arithmetic and gross of costs**, for long-only books against a market-capitalisation or equal-weighted benchmark; it does not link over time geometrically.

## Status by part

| Part | Evidence | Status |
|---|---|---|
| `backtest_portfolio`, `backtest_weights`, costs, funding | all angles above | strong |
| `alpha_beta`, IC, Sharpe intervals and paired differences | other libraries, calibration | strong; known finite-sample over-rejection of the alpha t |
| Trade masks, shorting limits, lots and minimum orders, open execution, suspension markdown, gross cap | loops, planted bugs, real data in three markets, review | moderate to strong; see the assumptions below |
| Impact model | hand computations; the visible order book bounds Y (`docs/crypto_impact_check.md`) | moderate: not calibrated against executions |
| Deflated Sharpe, PBO, permutation test, trial ledger | noise fails, a real edge passes, tampering is detected | moderate: little use on real research |
| `screen`, `backtest_event`, the six gates | synthetic tests and shuffled-null thresholds; on real data, shuffled factors and random event signals do not survive the gates: 8 and 6 per market in the validation, and 60 and 60 in crypto and in the US sample and 40 and 40 in Korea with none surviving (`docs/gate_false_positives.py`: 95 percent upper bound on the false-discovery rate 4.9, 4.9 and 7.2 percent). The first gate alone let 14 of 160 shuffled factors through (8.8 percent against a nominal 5): the threshold was built from 10 shuffles, and the documentation already warns that so few shuffles bias it low (the default is 20); the later gates removed every one of them. Power was measured by planting signals of known strength into the same real panels (`docs/gate_power.py`, `docs/gate_power_results.md`): 20 of 20 factors with a rank correlation of 0.05 survive all six gates in all three markets, 20 of 20 with 0.02 in Korea, 18 of 20 in crypto and 3 of 20 in the US sample, none at 0.005, and all six pure-noise controls are rejected; event signals pass 20 of 20 from a mean trade of about 85 bp after costs (17 of 20 at 12 bp in Korea). The plant is stationary with independent errors, so this is an upper bound on real power; use the default `n_null` (20) | **moderate**: this is noise from one factor family at one cost level, so it bounds the false-discovery rate for that, not for a researcher who tries many designs. In the documentation's own example the event signal still passes every gate on synthetic data with no signal |
| Deflated Sharpe, PBO | on real return distributions: the best of 40 shuffled strategies is not called real, a planted Sharpe 3 edge is (V8a); a planted Sharpe 1.5 edge is called real in Korea (about 10 years) and the US sample, not in crypto (5.7 years) | **moderate**: strict by construction, so weak edges are missed when the sample is short. Simulated (`docs/dsr_power.py`, 40 strategies tried, normal returns, 150 runs per cell): with a true Sharpe of 1.0 the true strategy is called real in 3 percent of runs at 5 years, 15 percent at 10 and 52 percent at 20; with 1.5, in 19, 67 and 99 percent; with no edge the best of 40 is called real in 0 of 600 runs (a plain t above 1.65 on the best: 85 percent; t above 3.0: 4.7 percent). The deflated Sharpe therefore runs well below its nominal 5 percent false-discovery level and pays for it in power. Heavy tails (crypto) were not simulated |
| `Panel.assert_causal` (exact look-ahead check) | known-answer tests with 7 causal and 7 leaking factors, planted bugs; on real data a past-only signal passes and a future return fails (V8d) | strong for what it checks: only the signal function, not the data |
| `Panel.assert_no_lookahead` (heuristic) | measured on real data | **weak**: misses a future-return signal, flags a legitimate one with a negative edge. Use `assert_causal` |
| Korean adapter | offline fake API, full real history, defects found and fixed | strong |
| Binance adapter, intraday | fake network; minute bars equal Binance's own daily files for two contract-months | moderate |
| US adapter (Tiingo sample) | offline; a 466-ticker random sample; the full path with open, high, low and dividends ran on real data in `docs/market_validation.md` (465 tickers with data, 813,344 bars, all checks pass) | **moderate**: bankruptcies are almost absent (0 of 12 checked), so survivorship bias is only bounded from below |
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
7. **The US free sample is small and has almost no bankruptcies.** The path with open, high, low and dividends (`fetch_symbols(full=True)`) passes on the 465 tickers with data;
   a result on a different or larger sample is not covered. One ticker (an expired warrant) returned nothing and is recorded as such.
8. **Order book, queues, partial fills, liquidation, margin, borrow availability and per-name borrow fees** are not simulated.
9. **Survivorship:** the US free data cannot show bankruptcies; any US result is a lower bound on survivorship bias. Korean data are survivorship-free; Binance keeps delisted contracts.
10. **Operating system:** Linux only.

## If you review

The fastest way to find what the author missed: run `python docs/market_validation.py all`, read `docs/market_validation.md`, then try to break one of
`tests/test_execution.py`, `tests/test_side_costs_shorting.py` or `tests/test_reconcile.py` by changing the code under test; each should fail.
