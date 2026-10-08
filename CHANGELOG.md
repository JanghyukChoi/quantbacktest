# Changelog

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
  Found by running it on real data: refusing the exit from a delisted name froze 38% of one short leg for ever, and a Binance contract absent for 894 days kept a position alive.
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
