# quantbacktest

[![CI](https://github.com/JanghyukChoi/quantbacktest/actions/workflows/ci.yml/badge.svg)](https://github.com/JanghyukChoi/quantbacktest/actions/workflows/ci.yml)

The Python package is named `pitbacktest` (`import pitbacktest`); the repository is `quantbacktest`.

**A backtest toolkit that measures its own biases.** Most backtest libraries compute a return and leave you to wonder how much
of it is survivorship, look-ahead, a flattering cost assumption, or luck from trying many variants. Here each of those is either a
measured number or a refused input, for US stocks, Korean stocks and crypto perpetuals. Core dependencies are `pandas` and `numpy`.
It is not a strategy and ships no data.

What is different:

- **Survivorship is measured, not assumed.** A free yfinance download returned a correct history for 0 of 42 well-known delisted or
  acquired US stocks; a free Tiingo key returned 23 of 29 takeover and rename cases but **none of 12 bankruptcies and rescue sales**
  (`docs/survivorship.md`). On Korean stocks, where the official KRX data is survivorship-free, restricting to today's survivors moved
  the net Sharpe of four factors by -0.23 to +0.20 depending on the factor, averaging about zero, and the understatement of low
  volatility survived neutralisation against the other styles (`studies/korea_survivorship`).
- **Preregistered studies, with their corrections kept.** Rules are committed before results; when a cost model turned out wrong, the
  original results stayed, an amendment explained it, and both are reported.
- **Overfitting is counted, not recalled.** Deflated Sharpe, PBO and a permutation test, plus a hash-chained trial ledger that records
  every run so the number of variants you tried cannot be understated from memory.
- **One timing convention, asserted in code**, point-in-time universes, firm-characteristic controls by default, and reporting that
  refuses to show an impressive raw number before the controls are applied.
- **Costs and capacity:** spread, borrow and square-root market impact for weights you supply, a capacity curve, and an intraday layer
  (1-minute Binance bars) that shows how fast an edge decays with latency and at what cost it stops paying.
- **Checked against independent implementations**, not just against itself: the portfolio engine equals a plain-loop reimplementation to
  1e-17, alpha/beta equal statsmodels' Newey-West regression and IC equals scipy's Spearman, the intraday parser reproduces Binance's own
  daily files, and over 40 deliberately planted bugs are each caught by the tests.

What it cannot do, stated up front: it has no order book (so no queue, partial fills or impact below a bar), it does not ship an
optimiser or point-in-time fundamentals, and only Linux is tested. See **Limits**.

## Markets

The engine works on any date x security panel. What differs between markets is the **data path** and whether it is
free of survivorship bias.

| Market | Data path | Survivorship-free? | Status |
|---|---|---|---|
| Crypto perpetuals (Binance) | `pitbacktest.crypto`, public archive | **Yes**: 900 contracts ever listed, 376 of them delisted or halted. Funding charged, delistings explicit | Tested; preregistered study in `studies/crypto_cross_section` |
| Korean stocks | `pitbacktest.adapters.krx`, official KRX OpenAPI (free key) | **Yes**: the API returns every stock listed on each day, so later delistings are inside the history. Adjusted returns come from the change versus the reference price, no price-adjustment table needed | Return formula checked on live data (all 953 KOSPI names on 2024-01-02, Samsung's 50:1 split day); panel build tested offline. Measured on 13 years of data: the survivors-only shortcut moves a factor's Sharpe by up to 0.23 in either direction, averaging about zero (`studies/korea_survivorship`) |
| US stocks | `adapters.yfinance`, or your own point-in-time data through `adapters.long_format` | **Not with yfinance**: it returned a correct history for 0 of 42 well-known delisted or acquired stocks (`docs/survivorship.md`). Yes if you bring CRSP, Sharadar or Norgate data | Detector, coverage report and delisting scenarios; no free fix exists |

## Three entry points

| Function | Question it answers |
|---|---|
| `screen(panel, factors)` | Does a factor survive costs, sub-periods, monotonicity and neutralisation? (research stage) |
| `backtest_portfolio(panel, factor)` | What does the factor do as a long-short portfolio? CAGR, Sharpe, max drawdown, turnover |
| `backtest_event(panel, signal)` | Does a boolean event signal work as an individual alert? Win rate vs base rate, mean vs median |

## What it enforces

| Rule | Why |
|---|---|
| One timing convention, checked by `assert_timing()` | An entry that is one day late (or early) silently changes results, most of all for 1-day mean reversion |
| Firm-characteristic controls are the default (size, book-to-market, momentum, ROA, asset growth) | Without them a "new alpha" is often a known factor in disguise. It warns when characteristics are missing |
| Uncontrolled results are returned but left out of the default summary | Reporting only after controls stops a large raw number from setting expectations |
| Win rate is reported with its base rate: `lift = win rate - base rate` | With a longer holding period both rise together; only the lift says anything about the signal |
| Mean and median are both reported (`skew_warning`) | If the signs differ, a few big winners hide many small losses. Fine for a portfolio, bad for an alert |
| Multiple-testing thresholds come from a shuffled null, not Bonferroni | Bonferroni ignores the correlation between tests |
| Parameter grids return the whole distribution | The user can see whether the best point is a plateau or a spike |
| `backtest_portfolio(..., ledger=Ledger(dir), family=..., name=...)` records every run; `ledger.deflated_sharpe(family)` takes the trial count from the record | People under-report how many variants they tried. A rerun of the same configuration counts once; any change is a new trial. The file is hash-chained, so editing or deleting a line in the middle is detected. It sees only runs that go through it |

## Reading a result

```python
r = q.backtest_portfolio(panel, factor, long_q=0.2, short_q=0.2, hold=5, spread_bp=10)
r.alpha_beta()        # regress net returns on the benchmark: alpha, beta, R2, Newey-West t
r.sharpe_ci()         # stationary block bootstrap interval for the Sharpe ratio
q.analytics.ic_report(panel, factor, horizons=(1, 5, 20), delist_return=-0.3)   # rank IC, ICIR, NW t, hit rate
q.analytics.sharpe_diff_ci(result_a.net_returns, result_b.net_returns)          # paired: is B really different from A?
```

| Tool | Default that avoids a common mistake |
|---|---|
| `alpha_beta` | Newey-West errors with the plug-in lag; refuses constant or collinear factors; dates aligned on the intersection. Its t-statistics still over-reject in finite samples (9% instead of 5% in the A4 test), so read |t| below about 2.5 as no evidence |
| `information_coefficient`, `ic_report` | A name that stops trading earns 0 after its last bar, or `delist_return` if given, instead of a NaN that silently removes the losers from the test |
| `sharpe_ci` | Block bootstrap keeps autocorrelation (an AR(1) of 0.3 widens the standard error 1.36x, matching theory). Intervals cover about 93% at a nominal 95% on 500 days |
| `sharpe_diff_ci` | Both series are resampled on the same dates, so a small real difference is detectable (standard error 0.04 against 0.57 unpaired in the B3 test) |

## Your own weights, costs and capacity

`backtest_portfolio` builds quantile portfolios from a factor. Institutions separate the steps: signal, portfolio
construction (an optimiser with risk and turnover limits), simulation. `backtest_weights` is the third step: give it the
weights your optimiser produced. The library does not ship an optimiser on purpose.

```python
w = q.backtest_weights(panel, weights, spread_bp=10, borrow_bp=300,
                       impact=q.ImpactModel(aum=50e6, y=1.0))        # square-root impact: Y * sigma * sqrt(|trade| * AUM / ADV)
q.capacity_curve(panel, weights, aums=[1e6, 5e6, 25e6, 100e6], y_values=(0.5, 1.0, 2.0), spread_bp=10)
```

| Choice | Why |
|---|---|
| Opening or increasing a position in a name that is not eligible that day raises | An optimiser that buys outside the point-in-time universe is using information it should not have. Holding a name that left is allowed |
| The impact coefficient is a parameter and the capacity function takes a list of them | Y is of order 1 in the literature but unknown for a given market; one capacity number would be false precision |
| Unknown volatility or volume is charged the cap (100 bp per unit traded by default), never zero | A name you cannot size is not free to trade |
| It reports participation (p99, max, share of trades above 10% of ADV) next to the cost | The square-root law is least reliable at high participation; look at both |
| Costs are on the net trade per name | `backtest_portfolio` charges its two legs as separate sleeves; with overlapping tranches the two can differ by the netting saving |

## Intraday bars (Binance, 1 minute and up)

```python
from pitbacktest.crypto import intraday as ib, ArchiveStore
store = ArchiveStore()
ib.fetch_minutes(store, ["BTCUSDT", "ETHUSDT", ...], "2024-01-01", "2024-06-30")          # monthly zips, resumable, no key
panel = ib.build_intraday_panel(store, symbols, "2024-01-01", "2024-06-30", bar="5min", top_n=30)
ib.latency_sweep(panel, factor, lags=(1, 2, 5, 15), one_way_bp=4)      # the same signal entered k bars late
ib.breakeven_cost(panel, factor)                                       # one-way bp at which the net mean is zero
```

What it answers: how fast an edge decays with latency, and at what cost it stops paying. What it cannot: whether the signal can
be traded. The archive has bars, not an order book, so there is no queue, no partial fill and no impact below the bar. See
`docs/intraday_design.md`.

| Choice | Why |
|---|---|
| Rows are labelled by bar **end**, and `entry_lag` below 1 is refused | A bar's close is only known when it ends; trading on it is look-ahead |
| Eligibility for every bar of day D uses days up to D - 1 | Day D's own volume must not decide whether D's bars are in the universe |
| Only the terminal run of zero-volume bars is removed; a no-trade stretch inside a contract's life keeps its price | Deleting it leaves a hole whose crossing return the engine would drop |
| Funding keeps its settlement instant (daily sums lose it) | A position held through the settlement pays it, and a flat one does not |
| `one_way_bp` instead of the engine's round-trip scalar | One unit trap fewer |
| A memory estimate, measured and not assumed, and a refusal with a bar length that fits | A year of 1-minute bars for 40 contracts needs about 3.4 GB |
| Annualised metrics from under half a year warn instead of returning NaN silently | Intraday studies are often a few months |

## Install

Not on PyPI yet (the planned name is `pitbacktest`; import it as `pitbacktest`). From a clone:

```bash
pip install -e .                  # pandas and numpy are the only requirements
pip install -e ".[yfinance]"      # the public-data adapter
pip install -e ".[test]"          # pytest, plus statsmodels and scipy for the cross-check tests
```

Python 3.10 to 3.14, pandas 2.0 to 3.0. CI runs every test file on each of them, including the oldest pandas and numpy the
package declares. Only Linux is tested.

## Example (public data)

```python
import pitbacktest as q
from pitbacktest.adapters.yfinance import load_panel

panel = load_panel(["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "JNJ"],
                   "2018-01-01", "2024-12-31")

print(panel.audit())            # data integrity audit
print(q.assert_timing(panel))   # timing self-check -> {'pass': True, ...}

# 5-day reversal as a long-short portfolio
r = q.backtest_portfolio(panel, -panel.close.pct_change(5), long_q=0.2, short_q=0.2, hold=5, spread_bp=5)
print(r.metrics["CAGR"], r.metrics["Sharpe"], r.metrics["MDD"])
```

This is a demonstration of the API on ten of today's large caps, not a result. The yfinance adapter is **not
point-in-time**: only tickers that exist today are returned, so delisted names are missing and performance is
biased upward. The adapter says so when it loads.

`examples/quickstart.py` runs all three entry points on synthetic data. It uses random factors on purpose; they
should fail every gate, and they do.

## Crypto

Two layers. `market="CRYPTO"` in the yfinance adapter (or `periods_per_year=365` on a `Panel`) fixes the annualisation:
with 252 days the CAGR, Sharpe and volatility of a 24/7 market come out wrong (on a 4.6-year sample, 252 reported 6.6
years of data). On top of that, `pitbacktest.crypto` builds a panel for Binance USDT-margined perpetuals from the public
archive and removes or measures the biases that usually flatter a crypto backtest:

| bias | what is done |
|---|---|
| Survivorship | the universe is every contract that ever traded (900, of which 376 are delisted or halted), not today's survivors. `survivors_only=True` reproduces the shortcut so its effect can be measured |
| Universe look-ahead | eligibility on day t uses only data up to t: minimum age, trailing median turnover, a valid and non-zero-volume bar |
| Stale prices | zero-volume bars (frozen prices after a halt) are dropped |
| Funding | the daily funding rate is charged by `backtest_portfolio` (long pays a positive rate, short receives) |
| Delisting | the last real bar is marked; `delist_return` sets the explicit return on the next day, and the events held are counted |
| Costs | taker fee plus a fixed half spread plus a thin-contract penalty, and a participation report instead of an invented impact model |

```python
import pitbacktest as q
from pitbacktest.crypto import fetch_all, build_panel, liquidity_cost_bp, participation_report

fetch_all()                                                     # once: ~900 contracts into ~/.cache/quantbt
panel = build_panel(start="2020-08-01", min_adv_usd=2e7, min_age_days=90)
cost = liquidity_cost_bp(panel)                                 # one-way bp, date x ticker
res = q.backtest_portfolio(panel, factor, long_q=0.2, short_q=0.2, hold=5, spread_bp=cost, delist_return=-0.3)
print(res.metrics["funding_annual_bp"], res.metrics["delist_events_held"])
print(participation_report(panel, res.holdings, aum_usd=10e6))
```

`pitbacktest.validation` has the deflated Sharpe, PBO (CSCV) and a permutation test for the selection you ran.
`studies/crypto_cross_section/` is a worked, preregistered example: four factors, 12 trials, five gates. It is **not
validated**, and the README there explains what the biases changed and what went wrong along the way.

Still not modelled for crypto: market impact beyond the thin-contract penalty (use `backtest_weights` with `ImpactModel` for a square-root estimate), the settlement price of a
delisted contract after its last bar, borrow limits and margin, and anything outside Binance.

## Equities: Korea and the US

**Korea.** `pitbacktest.adapters.krx` builds a point-in-time panel from the cached daily files of the official KRX OpenAPI
(`fetch_days`, resumable, about two calls per trading day). Eligibility on day t uses only data up to t; securities that stop
trading are kept for the days they traded and flagged in `delist_after`. A code that vanishes for 120+ days and returns is
treated as a different security. Prices are adjusted by compounding the reference-price returns (price return only,
no dividends). You need a KRX OpenAPI key (`KRX_OPENAPI_KEY`).

```python
from pitbacktest.adapters.krx import fetch_days, build_krx_panel, load_key
fetch_days("2013-01-01", "2026-10-06", "~/.cache/quantbt/krx", key=load_key(".env"))
panel = build_krx_panel("~/.cache/quantbt/krx", start="2013-01-01")
```

**US.** Free data cannot remove the bias, so the package detects it, measures it, and shows what it could be worth:

| tool | what it does |
|---|---|
| `Panel.audit()` | `survivorship_suspected` is true when a panel of 30+ names over 3+ years has almost nothing that stops trading. The yfinance adapter warns |
| `equity.universe_coverage` | how many US stocks were listed each year and how many of the ones that stopped trading are in your panel (uses a free Tiingo ticker list; lower bound, it is thin before 2013) |
| `equity.survivorship_scenarios` | puts delistings back at random (uniform, or tilted to volatile or illiquid names) with an explicit delisting return, and reports the spread of your result. A what-if, not a correction |
| `equity.survivors_only` | the usual shortcut made explicit, so a result can be compared with and without it |
| `adapters.long_format.panel_from_long` | a strict door for point-in-time data from CRSP, Sharadar, Norgate and the like: permanent ids, duplicate checks, delisting returns compounded into the last close, ticker-reuse detection |

A ticker is not an identifier: in the probe, five tickers returned the history of a *different* company that later reused the
symbol, with no error. See `docs/survivorship.md`.

## Tests: known answers, not real data

Every test uses data where the right answer is known, so none of them needs a network or real prices.

```bash
python tests/test_synthetic.py      # core engine: look-ahead, neutralisation, base rates (10 tests)
python tests/test_annualization.py  # 252 versus 365 days
python tests/test_validation.py     # deflated Sharpe, PBO, permutation test: noise must fail, a real edge must pass
python tests/test_crypto.py         # survivorship, point-in-time eligibility, stale bars, delisting, funding sign, costs
python tests/test_krx.py            # Korea: delisted names kept, split-day return, listing day, code reuse, resumable fetch
python tests/test_equity.py         # equity tools: delisting scenarios (known answer), coverage, survivors_only, long-format checks
python tests/test_reconcile.py      # the engine against an independent loop implementation (agrees to 1e-17), and DSR/permutation false-positive rates on noise
python tests/test_ledger.py         # trial ledger: distinct configurations, DSR count from the record, tamper detection
python tests/test_costs_events.py    # spread estimators, crypto costs, event signals and their neutralisation, FM with missing returns, panel checks
python tests/test_yfinance_adapter.py # the free-data adapter against a fake yfinance: when it warns about survivorship, market-cap paths, errors
python tests/test_binance_archive.py # the downloader on a fake network: retries, 404, listings, daily bars, funding sums, cache
python tests/test_intraday.py        # intraday layer on a fake archive: parsing, aggregation, point in time, latency, funding, break-even, memory, download
python tests/test_packaging.py       # version, Python floor, CI matrix and classifiers agree
python tests/smoke_installed.py     # run from outside the repo against an installed wheel (what the CI package job does)
python tests/test_weights.py        # weights: equals the engine, plain-loop reference with all costs, known answers, guard rails, capacity curve
python tests/test_tiingo.py         # Tiingo adapter offline: delisting kept, windows cut, resumable, quota stop, point in time, loud failures
python tests/test_analytics.py      # alpha/beta, IC, bootstrap: against statsmodels and scipy when installed, known answers, error rates
```

| Test | Expectation |
|---|---|
| T1 perfect-foresight factor | large positive result (timing is aligned) |
| T2 random factor | near zero (no false positives) |
| T3 look-ahead factor | caught by the look-ahead check |
| T4 random factor vs look-ahead check | passes (no false alarm) |
| T5 shuffled-null threshold | produced and sensible |
| T6 neutralisation | a control used as the factor disappears after neutralising |
| T7 base rate | random firing gives a lift near zero |
| T8 to T10 | small universes, small Fama-MacBeth regressions, perfectly collinear controls |
| T11 | annualisation scales by exactly sqrt(365/252) |
| V1 to V3 | deflated Sharpe, PBO and permutation p-value: noise looks like noise, a real edge is caught |
| K1 to K5 | Korean adapter: a stock gone by the end is in the panel and flagged, a 50:1 split leaves the return unchanged, the listing-day move is dropped, a returning code becomes a new security, fetching resumes and stops cleanly on a quota error |
| E1 to E6 | injecting a 5% yearly delisting rate at -30% lowers an equal-weight long book by 1.5% a year (the known answer), coverage counts, scenarios, strict long-format input |
| R1 to R7 | portfolio returns agree with a separate plain-loop implementation (long-short, long-only, delisting, funding), CAGR, Sharpe, drawdown and Sortino match textbook definitions, cost units are pinned, DSR and the permutation test do not reject noise more than they claim |
| L1 to L4 | the ledger counts a repeated run once and any change as a new trial, its DSR equals the direct computation, editing or deleting a line breaks the hash chain, recording changes no number |
| CE1 to CE10 | Roll recovers a 2 cent spread, Corwin-Schultz a 40 bp one, the spread model its coefficients, the crypto cost model is exact, an event signal that only echoes a control keeps -8% of its effect after controls while a real 100 bp effect keeps 102%, Fama-MacBeth with missing returns equals a per-day least squares, the paired difference, panel validation and the point-in-time mask follow their documentation |
| Y1 to Y5 | the yfinance adapter warns that it is not point in time every time, warns about survivors-only only with 30+ tickers none of which end early, reports partial market-cap coverage with counts, and refuses nothing-eligible and one-ticker panels |
| B1 to B7 | the Binance downloader retries a 503 and not a 403, treats 404 as missing, follows paginated listings, sums 8-hour funding settlements per day in both timestamp units, drops today's unfinished bar, caches, and survives one failing symbol |
| I1 to I10, I11 | 1-minute parsing in both timestamp units, aggregation equal to an independent loop, eligibility one day behind (first eligible bar is the listing day + 5 whole days), a knows-one-bar-ahead signal earns at lag 1 and nothing at lag 2, funding paid exactly once per settlement, break-even cost makes the net mean zero, memory guard, resumable download with recorded 404s, halts and no-trade stretches. Planted bugs (shifted labels, same-day eligibility, wrong funding bar, dropped partial bars and others) are caught |
| P1, P2 | the version is the same in pyproject, `__version__` and the changelog; the declared Python floor, the classifiers and the CI matrix agree, and the matrix tests the oldest declared pandas and numpy |
| R8 | `neutralize` equals a least-squares solution; a factor inside the controls' span gives NaN even with rounding noise (found because the old behaviour made one test fail on Python 3.10 only) |
| W1 to W6 | `backtest_weights` equals the engine on the engine's own holdings (exactly, except for a netting saving it documents), equals a plain-loop implementation with spread, borrow and impact, impact matches a hand calculation and scales as sqrt(AUM) and linearly in Y, borrow follows its formula, the untradable cap is exact, guard rails, capacity curve shape. Ten planted bugs are all caught |
| U1 to U8 | Tiingo adapter against a fake API: a delisted security is kept and flagged, a split does not move the return, a sub-dollar stock is never eligible, a reused ticker becomes two securities, a quota stops cleanly and resumes, eligibility ignores the future, the sample draw is seeded, malformed answers leave no file. Nine planted bugs are all caught |
| A1 to A8, B1 to B3 | alpha and beta equal statsmodels' Newey-West regression to 1e-12 and IC equals scipy's Spearman (when installed), known answers and invariances, rejection rates on noise, forward returns equal a loop implementation with and without delisting returns, bootstrap coverage and standard errors against theory, a paired difference of identical series is exactly zero. Planting eight bugs in the module (wrong taper, shifted window, ignored delisting, unpaired resampling and others) is caught by these tests every time |
| C1 to C8 | delisted contracts included, eligibility unchanged by future data, frozen bars dropped, new listings wait `min_age_days`, funding sign and size, explicit delisting return, bounded costs |

## Limits

- No bundled data. The yfinance adapter is not point-in-time; the Binance archive and KRX paths are. US stocks need data you bring.
- Fundamentals (`chars`) must be supplied by the user for the controls to be complete; without them it warns.
- Everything is in English except the Korean security-name patterns and the quota message the KRX adapter has to match.
- For research and education. Nothing here is investment advice.

## License

MIT
