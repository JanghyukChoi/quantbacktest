# Changelog

## 0.2.0
- `quantbt.ledger.Ledger`: hash-chained record of every run; `backtest_portfolio(..., ledger=...)` and the deflated Sharpe take the
  trial count from the record.
- `quantbt.analytics`: alpha and beta with Newey-West errors, rank IC with delisted names kept, stationary-bootstrap Sharpe intervals
  and a paired difference. Cross-checked against statsmodels and scipy.
- `quantbt.weights`: `backtest_weights` (spread, borrow, square-root market impact), `capacity_curve`, with a point-in-time universe guard.
- `quantbt.adapters.tiingo`: random-sample downloader and panel builder that keeps delisted securities (free account).
- `quantbt.adapters.krx`: official KRX OpenAPI adapter; Korean survivorship study in `studies/korea_survivorship`.
- `quantbt.equity`: delisting scenarios, `survivors_only`, coverage report, strict long-format adapter.
- **Fixed**: Sortino used the standard deviation of negative returns instead of the downside deviation.
- **Fixed**: `neutralize` returned rounding noise when a factor lay inside the span of the controls; after a rank-normalisation that noise
  became a full-scale signal and the result depended on the BLAS build (seen as a failure on Python 3.10 only). Such days are now NaN.
- `backtest_portfolio` documents that scalar `spread_bp` is a round trip while a panel is one way.
- Tested on Python 3.10 to 3.14 and pandas 2.0.3 to 3.0.6.

## 0.1.0
- First public version: crypto, Korean and US-equity panels, deflated Sharpe, PBO, permutation test, preregistered studies.
