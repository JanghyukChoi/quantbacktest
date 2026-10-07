# quantbacktest

The Python package is named `quantbt` (`import quantbt`); the repository is `quantbacktest`.

A small backtest harness for factor screening, event signals and portfolio alphas. Core dependencies are
`pandas` and `numpy` only.

It is not a strategy and ships no data. Its purpose is to make the usual backtest mistakes hard to make:
one timing convention that is asserted in code, controls for firm characteristics by default, and reporting
that refuses to show an impressive raw number before the controls are applied.

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

## Install

```bash
pip install pandas numpy
pip install yfinance          # only for the public-data adapter
pip install -e .              # from this directory
```

## Example (public data)

```python
import quantbt as q
from quantbt.adapters.yfinance import load_panel

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
years of data). On top of that, `quantbt.crypto` builds a panel for Binance USDT-margined perpetuals from the public
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
import quantbt as q
from quantbt.crypto import fetch_all, build_panel, liquidity_cost_bp, participation_report

fetch_all()                                                     # once: ~900 contracts into ~/.cache/quantbt
panel = build_panel(start="2020-08-01", min_adv_usd=2e7, min_age_days=90)
cost = liquidity_cost_bp(panel)                                 # one-way bp, date x ticker
res = q.backtest_portfolio(panel, factor, long_q=0.2, short_q=0.2, hold=5, spread_bp=cost, delist_return=-0.3)
print(res.metrics["funding_annual_bp"], res.metrics["delist_events_held"])
print(participation_report(panel, res.holdings, aum_usd=10e6))
```

`quantbt.validation` has the deflated Sharpe, PBO (CSCV) and a permutation test for the selection you ran.
`studies/crypto_cross_section/` is a worked, preregistered example: four factors, 12 trials, five gates. It is **not
validated**, and the README there explains what the biases changed and what went wrong along the way.

Still not modelled: market impact beyond the thin-contract penalty (no order book in the archive), the settlement price of a
delisted contract after its last bar, borrow limits and margin, and anything outside Binance.

## Tests: known answers, not real data

Every test uses data where the right answer is known, so none of them needs a network or real prices.

```bash
python tests/test_synthetic.py      # core engine: look-ahead, neutralisation, base rates (10 tests)
python tests/test_annualization.py  # 252 versus 365 days
python tests/test_validation.py     # deflated Sharpe, PBO, permutation test: noise must fail, a real edge must pass
python tests/test_crypto.py         # survivorship, point-in-time eligibility, stale bars, delisting, funding sign, costs
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
| C1 to C7 | delisted contracts included, eligibility unchanged by future data, frozen bars dropped, new listings wait `min_age_days`, funding sign and size, explicit delisting return, bounded costs |

## Limits

- No bundled data. The yfinance adapter is not point-in-time; the Binance archive path is.
- Fundamentals (`chars`) must be supplied by the user for the controls to be complete; without them it warns.
- Docstrings and some code comments are in Korean. This README is the English documentation.
- For research and education. Nothing here is investment advice.

## License

MIT
