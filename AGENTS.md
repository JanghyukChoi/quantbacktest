# Using pitbacktest from an AI assistant

Read this before writing code with the library. It is short on purpose.

`pitbacktest` judges whether a backtest result is real. Your job as an assistant is to run the checks and report **what limits the result**, not to find a strategy that looks good.

## The one workflow

**1. Your data into a panel.** The input is a table with one row per security and day. `panel_from_long` takes it (column names are arguments) and refuses ambiguous data.

```python
import numpy as np, pandas as pd, pitbacktest as q
from pitbacktest import recipes
from pitbacktest.adapters.long_format import panel_from_long

d = pd.read_csv("prices.csv")                                     # for example: Date, Symbol, Adj Close, Volume (one row per security and day)
d["Date"] = pd.to_datetime(d["Date"], format="%m/%d/%Y")
d = d.drop_duplicates(["Symbol", "Date"])                         # exact duplicates are an error otherwise
d = d[d["Adj Close"] > 0]                                         # a zero or negative close is a bad print: drop the row (or carry the previous close one day) and SAY which
panel = panel_from_long(d, id_col="Symbol", date_col="Date", close_col="Adj Close", volume_col="Volume", market="US")
print(panel.meta)                                                 # rows, securities, how many ended early, how many delisting returns were applied
print(panel.audit())                                              # look at it: extreme returns, truncation, survivorship, and the equal-weighted market (worst day, deepest fall)
```

Check what the audit says about the **market** of the sample against what you know about the period, and the spread of the securities' own volatility (`stock_volatility_p05` and `_p95`: real stocks differ by a
factor of two or more, a universe whose volatilities are all alike is likely generated): a stretch of several years whose equal-weighted market never fell more than a few percent
(and whose worst day is mild) is synthetic, a calm window or a selection of survivors, and any conclusion about regimes or crashes is unsupported. If a security's last row is long before the end of the
panel it is flagged as delisted; without a delisting return the library closes it at its last price, which is optimistic (see the delisting stress in the review).

**2. A factor, from a recipe** (higher score = long). `recipes.describe()` lists them.

**3. A review with a ledger**, which counts every variant tried; keep every try inside the same family.

```python
print(list(recipes.describe()))                                   # what exists, its parameters and what it needs from the panel
factor = recipes.recipe(panel, "reversal", lookback=5)
ledger = q.Ledger("ledger_dir")
rv = q.review_portfolio(panel, factor, long_q=0.2, short_q=0.2, hold=5, spread_bp=10, ledger=ledger, family="reversal", name="reversal-5")
```

**4. Read it back, starting with `summary()`.**

```python
s = rv.summary()                                                  # the headline in one flat dictionary; everything is in rv.to_dict() and rv.report("review.html")
print(s["notes"])                                                 # the warnings the run raised: read every one
print(s["figures"], s["sharpe_interval_95"], s["variants"])
print(*s["reading"], sep="\n")                                    # sentences that follow from the numbers
```

| Report this | Where |
|---|---|
| What limits the number | `s["notes"]`, `s["reading"]`, and the first box of `review.html` |
| Sharpe and its interval | `s["figures"]["sharpe"]`, `s["sharpe_interval_95"]` |
| Variants tried and the deflated Sharpe | `s["variants"]`: `trials`, `best_deflated_sharpe` (of the best trial) and `reviewed_deflated_sharpe` (of the strategy you were asked about); about 0.95 or more is the usual bar, and a strategy whose Sharpe is not positive has no edge whatever the probability says |
| Luck tests | `s["luck"]` (the permutation p-value compares Sharpe ratios **before costs** and does not know you chose among several, so it never says that a strategy pays; Newey-West; sign test) |
| CAGR, drawdown, turnover, cost | `s["figures"]`; `s["figures"]["break_even_spread_bp"]` is the round-trip spread above which the Sharpe ratio is not positive: compare it with what the strategy would really pay. There is no universal realistic cost: say what you assumed and why (the quoted spread of the market and the size of the names), and show the cost table. The delisting stress is `s["delisting_stress"]` |
| Bear market in the sample? | `s["market"]["sample_has_bear"]` |

## What to tell the user

Report in this order, in plain words, and do not skip the first two:

1. **What limits the number.** `notes` and `reading`. If the Sharpe interval includes 0, say there is no evidence of an edge in this sample.
2. **How many variants were tried.** Every recipe, parameter value, universe, data cleaning or factor you tried and **looked at the result of** is a trial: the ledger counts the ones run through it, and
   `extra_trials=` declares the rest. State the count.
3. Then the figures: CAGR, Sharpe, maximum drawdown, turnover, cost sensitivity, and whether the sample contains a bear market.

## Rules

- **Never write a factor from scratch if a recipe does the job.** If you must, run `panel.assert_causal(make_signal)` on it first; a factor that fails it uses the future, and its result is worthless.
- **One panel, one timing convention.** A signal on date d is traded at the close of d + `entry_lag` (default 1). Do not shift the factor yourself.
- **Never choose settings on the whole sample and then report that sample.** Put the choice in a grid (`review_portfolio(..., make_factor=..., grid=...)`), or fix it before looking (`primary_horizon=` for events,
  `split_date=` for a frozen out-of-sample stretch).
- **Do not report a Sharpe ratio without its interval, or a win rate without its base rate.**
- **A missing number is `null` in the JSON and NaN in Python. It means unknown, not zero.** Do not fill it.
- **Costs are an assumption you must state.** `spread_bp` is a round trip for a scalar; say the number you used and why. Without `Panel.shortable` and a `borrow_bp` the short side is assumed free and borrowable.
- **Do not claim more than the library does.** Read `docs/verification_status.md` (what was checked, what was not), `docs/assumptions.md` and the "Limits" section of the README. It has no order book, no margin, and no
  point-in-time fundamentals; US data from yfinance is not survivorship-free (the adapter says so).
- **This is research tooling, not investment advice.** Do not recommend buying or selling a security on the strength of a review.

## Where things are

| You want | Use |
|---|---|
| Whether a factor survives costs and controls | `q.screen(panel, {name: factor}, ledger=ledger)` |
| A factor as a long-short portfolio | `q.backtest_portfolio(panel, factor, ...)` |
| An on/off signal | `q.backtest_event(panel, signal, primary_horizon=5)` and `q.review_event` |
| Weights you already have | `q.backtest_weights(panel, weights, ...)` |
| The whole battery of tests, one page | `q.review_portfolio` / `q.review_event`, then `.summary()`, `.report()` or `.to_json()` |
| Several strategies side by side | `q.compare({...}, "compare.html")` |
| Data | `adapters.long_format.panel_from_long` (your own table), `pitbacktest.crypto` (Binance public archive, no key), `adapters.krx` (free key), `adapters.yfinance` |
