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

## Crypto and other 24/7 markets

Set `market="CRYPTO"` in the yfinance adapter (or `periods_per_year=365` on a `Panel` you build yourself). The
annualisation uses 365 days instead of 252; with 252 the CAGR, Sharpe and volatility of a 24/7 market come out wrong
(on a 4.6-year crypto sample, 252 reported 6.6 years of data). `tests/test_annualization.py` checks the scaling.

```python
panel = load_panel(["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"], "2020-06-01", "2024-12-31", market="CRYPTO")
```

What is **not** modelled for crypto, so treat the numbers as optimistic:

- No perpetual-futures funding or borrow cost for the short leg; shorting is assumed free apart from the spread.
- The yfinance universe is today's survivors (delisted coins are missing), and crypto survivorship bias is larger
  than in equities.
- The daily bar is a UTC midnight close; there is no market open or close, so the entry-lag convention is a modelling
  choice, not an exchange fact.
- Spread and cost estimators were written for equities; check them on thin tokens before relying on them.

## Tests: known answers, not real data

`python tests/test_synthetic.py` checks that the harness answers correctly on data where the answer is known.
All ten pass, plus `python tests/test_annualization.py` for the 252 versus 365 scaling.

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

## Limits

- No bundled data. The only adapter uses yfinance, which is not point-in-time.
- Fundamentals (`chars`) must be supplied by the user for the controls to be complete; without them it warns.
- Docstrings and some code comments are in Korean. This README is the English documentation.
- For research and education. Nothing here is investment advice.

## License

MIT
