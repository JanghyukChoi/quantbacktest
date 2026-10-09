# Changelog

## 0.3.0
- **Results as JSON.** `result.to_dict()` and `result.to_json(path)` for `backtest_portfolio`, `backtest_weights`, `backtest_event` and `screen`: figures, the unit of each, the
  run's settings and warnings, as plain JSON with no NaN (a figure that could not be computed is `null`). Layout versioned (`schema_version`), described in `pitbacktest.export`.
  Meant for notebooks, other programs, and assistants that call the library as a tool.
- **Warnings kept on the result.** `result.notes` holds the text of every warning the run raised; the console and `-W error` behave as before, and the warning now points at the
  caller's line.
- **A one-page report.** `result.report("report.html")`: what limits the number first (warnings, Sharpe interval, deflated Sharpe with a ledger, alpha against the benchmark),
  then figures, equity, drawdown and rolling Sharpe with hover values, a monthly heatmap, a year table, costs and exposure, and the capacity table from `capacity_curve`.
  One self-contained file, no dependency, light and dark. Thinning for drawing keeps each stretch's lowest and highest point.
- **Market impact in `backtest_portfolio`.** `impact=ImpactModel(aum=..., y=...)`; the same holdings cost the same through `backtest_weights` (checked to 1e-12). `aum` is the money
  in each leg and in the currency of `close x volume`. The 25-cell `grid` is left out when an impact model is given. `ImpactModel` now lives in `pitbacktest.core.impact`
  (still importable from `pitbacktest.weights` and from `pitbacktest`).

## 0.2.1
Found by running the engines on hostile inputs (zero and infinite prices, misaligned tables, leverage that wipes the account).
- **Fixed: a zero or infinite price on a held security gave a daily return of 3.6e307** (the infinite return was turned into the largest float) and a Sharpe of 0.
  A price that is zero, negative or infinite is now a missing price, with a warning. A vendor that writes a halted day as 0 hits this.
- **Fixed: an account that lost 100 percent or more in a day went on compounding with negative equity** (a 30x long that falls 4 percent; CAGR, drawdown and Sharpe
  described an account that could not exist). `backtest_portfolio` and `backtest_weights` now set that day to -100 percent and every later day to 0, warn, and report
  `ruined` and `ruin_date` in `metrics`. This is not a margin or liquidation model: there is still no leverage interest and no maximum leverage.
- **Fixed: a signal that did not line up with the panel gave a Sharpe of 0.0 and no error** (other ticker names, transposed, a time zone on one side only). `backtest_portfolio`,
  `backtest_weights`, `backtest_event` and `screen` now refuse a table with no date or no ticker in common with the panel, say what to check, and warn when under half of
  the dates or tickers match. A numpy array or a Series is refused with the type named.
- `backtest_portfolio` refuses `long_q + short_q > 1` (the same security in both legs); duplicate ticker names in `Panel` raise.
- New warnings: a daily move above +1000 percent on an eligible security (an unadjusted split or a bad price; the numbers are not changed), `periods_per_year` that
  does not fit the date spacing (weekly dates with 252), and a result in which no position was ever held (an all-NaN or constant factor, zero weights, a capital below one lot).
- `deflated_sharpe` says so when no column has a usable Sharpe ratio instead of "All-NaN slice encountered".
- `ImpactModel.aum` is money in the currency of `close x volume` (it said dollars; for Korean stocks it is won).
- Measured: `docs/dsr_power.py` (power of the deflated Sharpe on simulated edges) and `docs/gate_power.py` (power of the six gates on signals planted in the real crypto, US and
  Korean panels), with the results in `docs/verification_status.md` and `docs/gate_power_results.md`.

## 0.2.0
- **Renamed: the package and the import are now `pitbacktest`** (they were `quantbt`). The name `quantbt` on PyPI belongs to an unrelated
  project, and two distributions that share a top-level package would overwrite each other's files. The data cache directory
  keeps its old name, `~/.cache/quantbt`, so downloads already on disk are still found.
- `pitbacktest.ledger.Ledger`: hash-chained record of every run; `backtest_portfolio(..., ledger=...)` and the deflated Sharpe take the
  trial count from the record.
- `pitbacktest.analytics`: alpha and beta with Newey-West errors, rank IC with delisted names kept, stationary-bootstrap Sharpe intervals
  and a paired difference. Cross-checked against statsmodels and scipy.
- `pitbacktest.weights`: `backtest_weights` (spread, borrow, square-root market impact), `capacity_curve`, with a point-in-time universe guard.
- `pitbacktest.crypto.intraday`: Binance 1-minute bars aggregated to any bar that divides a day, a point-in-time panel (rows labelled by bar end,
  eligibility one day behind, funding at the settlement instant), `latency_sweep` and `breakeven_cost`, with a measured memory guard.
  Design in `docs/intraday_design.md`.
- Annualised metrics from a sample shorter than half a year now **warn** instead of returning NaN silently.
- `pitbacktest.adapters.tiingo`: random-sample downloader and panel builder that keeps delisted securities (free account).
- `pitbacktest.adapters.krx`: official KRX OpenAPI adapter; Korean survivorship study in `studies/korea_survivorship`.
- `pitbacktest.equity`: delisting scenarios, `survivors_only`, coverage report, strict long-format adapter.
- Docstrings, comments, warnings and example output are now in English (the Korean left is the security-name filter and the quota message of the
  KRX adapter, which must match Korean text). Gate names and the `screen()` funnel keys are English too.
- **Fixed**: the yfinance probe was described as 42 stocks (no data 37) in the README and docs; the list has 41 and a re-run gives no data 36, another company 5, correct history 0.
- **Fixed (found by independent review of the specification)**: `screen` and `backtest_event` measured returns with `Panel.forward`, which was NaN when the
  exit price was missing, so **a trade that ran into a delisting dropped out of their statistics**: the losses of delisted securities were missing, a
  survivorship leak in the two tools that had no `delist_return`. `Panel.forward` now carries a security flagged in `delist_after` at its last price (or at
  `last x (1 + delist_return)`), and `screen` and `backtest_event` take `delist_return`. A halt that is not a delisting still gives NaN.
- **Fixed**: the README, the `screen` docstring and the package docstring said `screen` reports post-control results only. Only `t` and `neu_*` are
  computed with controls; `excess_bp`, `net_bp` and `rho` come from the uncontrolled top-decile firing. The wording is corrected everywhere, and the
  `Panel.forward` docstring (which described a log accumulation that never existed) is rewritten.
- **Side-specific costs.** `backtest_portfolio` and `backtest_weights` take `buy_bp` and `sell_bp` (a float, a Series indexed by date for a rate that
  changes on a date, or a date x ticker frame), charged on top of the spread. A weight going down is a sell, so opening a short pays `sell_bp`. An unknown
  value (NaN, a date before the first entry) raises. `metrics` gain `side_cost_annual_bp`; `cost_annual_bp` includes it. Runs recorded without them keep their
  ledger fingerprint.
- **Short-selling limits.** `Panel.shortable` (bool frame; what is missing counts as not shortable). `backtest_portfolio` picks the short leg among shortable
  names (`short_leg_empty_days` counts days with none); `backtest_weights` raises when a short is opened or increased where it cannot be sold short
  (`check_shortable`). `shortable_from_bans` builds the frame from ban periods and exemptions. **No ban calendar and no tax table ship with the library.**
- `adapters.krx.build_krx_panel` records the market of every security on every day in `meta`; `adapters.krx.sell_tax_panel` turns a rate schedule per market
  into a date x ticker panel for `sell_bp`.
- **Fixed**: averaging overlapping tranches (`hold > 1`) left rounding residue of about 1e-18 where the exact weight is 0, and `avg_positions` and
  `delist_events_held` counted every residue as a held position (on a synthetic panel `avg_positions` read 148.7 where the true figure is 100.9). The returns were
  not affected (the residue is 1e-18). Residue is now snapped to 0.
- **Execution realism.** `Panel.can_buy` / `can_sell` (bool frames; what is missing counts as not possible) stop trades that cannot be done on the execution day, in both
  engines. `execution.tradability` builds them from prices and volume (no price, no volume, a daily-limit lock; a position is settled after a flagged delisting and
  after `max_gap_days` without any price). `execution.at_prices` trades at another price (the open). `backtest_weights` takes `capital`, `price`, `lot` and
  `min_trade_value` for whole-lot sizes. New metrics: `blocked_trades`, `blocked_turnover_share`, `mean_stuck_weight`, `longest_freeze_days`, `min_trade_skipped`,
  `mean_abs_rounding_gap`. The KRX and Tiingo panels record the real price level in `meta["raw_close"]`. No limit, lot size or minimum order ships with the library.
  Found by running it on real data: refusing the exit from a delisted name froze a large part of one real short leg for ever, and a Binance contract absent for years and then listed again kept a position alive.
- **Binance trading rules.** `ArchiveStore.symbol_rules()` reads the quantity step, minimum quantity and minimum order value of each contract from the exchange-info
  endpoint (no key; today's values only, delisted contracts absent) and `execution_rules()` turns them into `lot` and `min_trade_value` for `backtest_weights`, which now
  takes `min_trade_value` per ticker. An unknown ticker raises or is NaN: no rule is invented.
- **Crypto spread against the order book.** Measured the quoted half spread of 12 USDT-M contracts (36 contract-days, 105 million quote updates) from the public `bookTicker` files
  (`docs/crypto_spread_check.md`, `docs/crypto_spread_probe.py`). The default flat 2 bp (7 bp for thin contracts) was far above the quote for the liquid ones (100 to 200 times for BTC and
  ETH) and above the default for only 1 of 12. The quote is one price tick for 9 of 12; for the others the exchange has cut the tick since (10, 10 and 100 times). New
  `liquidity_cost_bp(spread_estimator="tick", tick_size=...)`. The default is unchanged because the preregistered study used it.
- **Impact coefficient against the order book.** `docs/crypto_impact_check.md`, `docs/crypto_impact_study.py`: from the public `bookDepth` files (30 contracts, 10 days),
  the Y the visible book implies is about 0.1 to 0.8 for orders up to 0.3 percent of a day's turnover and 1.2 to 2.7 at 1 percent, so `Y = 1` overstates small orders and understates
  large ones; `capacity_curve(y_values=(0.5, 1.0, 2.0))` brackets it. The size dependence is built into the interpolation and is not a finding. `ImpactModel` is unchanged.
- **Frozen positions.** `execution.freeze_episodes` finds trading suspensions (a price but no volume) of at least N days and classifies how each ended (resumed, resumed then
  delisted, ended in halt, ongoing) with the return to the end of it. `backtest_portfolio` and `backtest_weights` take `freeze_days` and `freeze_return`: a long position in a security whose
  suspension reaches `freeze_days` is marked down once (shorts are not credited), and `cap_gross`, which scales the free names by the largest k that keeps the gross exposure at the target's
  (bisection on k; days on which the stuck positions alone exceed the target are counted). Defaults change nothing. Measured on KRX since mid-2015: a suspension that has lasted 20, 60, 120 days
  ends in a delisting 25, 35, 48 percent of the time and the mean return to the end of it is -23, -35, -49 percent (median -7, -14, -30).
- **Fixed (KRX adapter)**: `close / (close - change) - 1` gives a move of thousands of percent when a consolidation happens during a suspension (one case: +29,948%). They are now listed in
  `meta["suspect_returns"]` and `build_krx_panel(drop_suspect_above=1.0)` can cut them; the default is unchanged so earlier results reproduce. None of the repository's strategies held such a
  name. The first measurement of suspension outcomes was distorted by exactly this: one false jump made the mean look like zero (+29,948% in one day; +15,285% over the whole suspension-to-delisting episode in the uncleaned panel).
  In the whole cache 38 daily returns beyond +-100% are listed (14 right after zero or missing volume); 29 of them, 10 after zero volume, since June 2015.
- **One validation on real data in three markets.** `docs/market_validation.py` runs the same checks on KRX, the US Tiingo sample and Binance USDT-M (shuffled-factor calibration, timing canaries,
  cost monotonicity, determinism, identities of the realism options with plain results, realism metrics, data sanity) and writes `docs/market_validation.md`; the exit code is non-zero if a check
  fails. Method only: no factor's performance is reported. All checks pass at the commit it names.
- **An exact look-ahead check.** `Panel.assert_causal(make_signal)` rebuilds a signal from the panel cut at random dates and requires the last row to equal the full-data row: any use of
  the future (negative shift, centred window, whole-history mean or z-score, rank across time) is caught, a past-only signal always passes. `Panel.truncate(date)` is the cut. The old
  `assert_no_lookahead` heuristic is kept but its docstring now states its blind spots, found on real data: it **passes** a signal that is the future return itself and **flags** a legitimate
  signal whose own edge is negative. Its own test (T3) had been written to accept a miss.
- **Fixed (KRX adapter)**: the raw files write open, high and low as 0 on a day without trades; the panel now has NaN there (it had 0, which made trading at the open produce infinite
  returns). `execution.at_prices` treats any non-positive price as missing. Found by the real-data check that open, high and low are positive.
- **More real-data validation** (`docs/market_validation.py`, V8): the deflated Sharpe does not call the best of 40 shuffled strategies real and does call a planted edge of Sharpe 3 real; the
  probability of backtest overfitting is lower with the edge; shuffled factors and random event signals do not survive the six gates; the exact causality check. Measured limit: an edge of Sharpe 1.5
  is not called real in the crypto sample (5.7 years).
- **Gate false-discovery rate on real data.** `docs/gate_false_positives.py`: 60 shuffled factors through `screen` and 60 random yes/no signals through `backtest_event` in crypto and in the US
- **Deflated Sharpe power.** `docs/dsr_power.py` simulates how often a planted edge among 40 tries is called real (Sharpe 1.0: 3, 15 and 52 percent at 5, 10 and 20 years; Sharpe 1.5: 19, 67 and 99 percent) and shows the rule is far stricter than its nominal 5 percent level (0 false discoveries in 600 runs). Documented in `docs/verification_status.md`, with the note that gate power on planted real signals is not measured.
- **Gate power.** `docs/gate_power.py` plants factors (rank correlation 0 to 0.05) and event signals into the real crypto, US and Korean panels and counts how many pass the six gates: noise controls 0 of 120, rank correlation 0.05 passes 20 of 20 everywhere, 0.02 passes 20, 18 and 3 of 20 in Korea, crypto and the US sample. An upper bound on real power. Results in `docs/gate_power_results.md`.
  sample (and 40 and 40 in Korea): none survived the six gates (95 percent upper bound on the rate 4.9 and 7.2 percent). The first gate alone passed 14 of 160 (8.8 percent against a nominal 5), because the
  null threshold was built from 10 shuffles, which biases it low; the later gates removed all of them.
- **Tiingo full downloads.** `fetch_symbols(full=True)` keeps open, high, low (adjusted), the dividend and the split factor; `build_tiingo_panel` then fills `Panel.open/high/low`
  and `meta["div_cash"]`. Stores made without it read as before. The free plan allows 500 distinct symbols a month and 50 requests an hour; re-requesting a symbol already looked up
  did not hit the monthly limit.
- **Changed**: `backtest_weights(check_universe=True)` now looks at long and short exposure separately, so turning a long into a smaller short in a name that is
  not eligible raises (it used to pass because the absolute weight shrank).
- **Fixed (second independent review)**: `inject_delistings(hazard="illiquid")` raised `IndexError` on any panel whose first calendar year had 100 or more bars
  (there is nothing earlier to tilt on), and `hazard="volatile"` silently became uniform there; the first year is now drawn uniformly. `Panel.fingerprint` left out
  `volume`, `mkt_cap`, `open`, `high`, `low` and `chars`, so two impact-model runs that differed only in volume counted as one trial; they are now hashed.
  `screen` ignored the shuffled-null threshold when given a `GateConfig()` with `null_threshold=None` (it used 3.0), and accepted `fire_q` outside (0, 1] and boolean factors; it now fills the measured threshold and rejects both.
- **Fixed**: an intraday backtest accepted `entry_lag=0` through a hand-built panel or `latency_sweep(lags=(0, ...))`; a scalar NaN cost passed the cost check and
  produced NaN metrics; `inject_delistings` checked `hazard` only when it drew a name, and `hazard="illiquid"` without volume silently became uniform;
  on a panel shorter than a year the 12-1 momentum control was NaN everywhere and every date dropped out of the regressions (it is now left out with a warning).
- **Fixed (found while writing the specification)**: the `screen` gate G1 threshold is the 95th percentile of shuffled |t| values, and the default
  `n_null=2` made it the percentile of two numbers: biased far below the level of pure noise (about 0.9 on average against 1.96, 1.8 with 20 repetitions), so G1
  passed too easily. The default is now 20, fewer than 10 warns and fewer than 1 raises.
- **Fixed (found while writing the specification)**: bad arguments used to run something else without a word. A mistyped `weighting` or `benchmark`
  was read as another option, `long_q` outside (0, 1] ran, `short_q=0` silently meant long-only, `hold=0` gave a Sharpe of 0, a **negative cost paid the
  strategy for trading** (a Sharpe of -1.01 became +0.96), and a boolean factor was documented as supported but was not. All of these now raise a
  `ValueError` naming the argument; the same for `backtest_weights` (benchmark, negative costs, a nonsensical `ImpactModel`), `screen` (`primary_h`
  not in `horizons`) and `backtest_event` (`horizons`, `cost_bp`).
- **Fixed**: duplicate dates in `close` now raise the library's own clear message; before, pandas failed first with an opaque `reindex` error.
- **Fixed**: the `backtest_event` warning about too few days described an old rule; it now states the adaptive pool threshold the code uses.
- Test coverage 74% -> 85% (spread estimators, crypto costs, event signals and their dummy-regression neutralisation, the yfinance adapter's
  warnings, the Binance downloader, panel validation).
- **Fixed**: Sortino used the standard deviation of negative returns instead of the downside deviation.
- **Fixed**: `neutralize` returned rounding noise when a factor lay inside the span of the controls; after a rank-normalisation that noise
  became a full-scale signal and the result depended on the BLAS build (seen as a failure on Python 3.10 only). Such days are now NaN.
- `backtest_portfolio` documents that scalar `spread_bp` is a round trip while a panel is one way.
- Tested on Python 3.10 to 3.14 and pandas 2.0.3 to 3.0.6.

## 0.1.0
- First public version: crypto, Korean and US-equity panels, deflated Sharpe, PBO, permutation test, preregistered studies.
