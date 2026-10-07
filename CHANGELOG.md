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
