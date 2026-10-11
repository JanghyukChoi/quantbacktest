# Using pitbacktest from an AI assistant

Read this before writing code with the library. It is short on purpose.

`pitbacktest` judges whether a backtest result is real. Your job as an assistant is to run the checks and report **what limits the result**, not to find a strategy that looks good.

## The one workflow

```python
import numpy as np, pandas as pd, pitbacktest as q
from pitbacktest import recipes

# 1. A panel: close, eligible (bool), and volume / mkt_cap when you have them; all (date x ticker), the same index and columns.
rng = np.random.default_rng(1)
dates = pd.bdate_range("2019-01-01", periods=700); names = [f"S{i:02d}" for i in range(40)]
close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (700, 40)), axis=0)), index=dates, columns=names)
eligible = pd.DataFrame(True, index=dates, columns=names); eligible.iloc[:300] = False       # nothing to trade until a year of history exists
panel = q.Panel(close=close, eligible=eligible, volume=close * rng.lognormal(10, 1, close.shape), mkt_cap=close * 1e7, market="DEMO")

# 2. A factor: take a recipe instead of writing one (higher score = long).
print(list(recipes.describe()))                       # what exists, its parameters and what it needs from the panel
factor = recipes.recipe(panel, "reversal", lookback=5)

# 3. Review it with a ledger, which counts every variant tried; keep trying inside the same family.
ledger = q.Ledger("ledger_dir")
rv = q.review_portfolio(panel, factor, long_q=0.2, short_q=0.2, hold=5, spread_bp=10, ledger=ledger, family="reversal", name="reversal-5")

# 4. Read it back as data.
data = rv.to_dict()                                    # or rv.to_json("review.json"); rv.report("review.html") for a page
print(data["notes"])                                   # the warnings the run raised: read every one
print(rv.sharpe_ci, rv.deflated)
```

## What to tell the user

Report in this order, in plain words, and do not skip the first two:

1. **What limits the number.** The `notes` and the first box of `review.html` ("Read this first"). If the Sharpe interval (`sharpe_ci`) includes 0, say there is no evidence of an edge in this sample.
2. **How many variants were tried.** Every recipe, parameter value or factor you tried is a trial. State the count (`rv.deflated["trials"]`) and the deflated Sharpe, which is a probability: about 0.95 or more is the usual bar.
3. Then the figures: CAGR, Sharpe, maximum drawdown, turnover, cost sensitivity (`rv.cost_table`), and whether the sample contains a bear market (`rv.regimes["sample_has_bear"]`).

## Rules

- **Never write a factor from scratch if a recipe does the job** (`recipes.describe()`). If you must, run `panel.assert_causal(make_signal)` on it first; a factor that fails it uses the future, and its result is worthless.
- **One panel, one timing convention.** A signal on date d is traded at the close of d + `entry_lag` (default 1). Do not shift the factor yourself.
- **Never choose settings on the whole sample and then report that sample.** Put the choice in a grid (`review_portfolio(..., make_factor=..., grid=...)`) and read the walk-forward and deflated-Sharpe parts, or fix a `split_date` before looking.
- **Do not report a Sharpe ratio without its interval, or a win rate without its base rate.**
- **A missing number is `null` in the JSON and NaN in Python. It means unknown, not zero.** Do not fill it.
- **Costs are an assumption you must state.** `spread_bp` is a round trip for a scalar; say the number you used and why.
- **Do not claim more than the library does.** Read `docs/verification_status.md` (what was checked, what was not) and the "Limits" section of the README. It has no order book, no margin, and no point-in-time fundamentals; US data from yfinance is not survivorship-free (the adapter says so).
- **This is research tooling, not investment advice.** Do not recommend buying or selling a security on the strength of a review.

## Where things are

| You want | Use |
|---|---|
| Whether a factor survives costs and controls | `q.screen(panel, {name: factor})` |
| A factor as a long-short portfolio | `q.backtest_portfolio(panel, factor, ...)` |
| An on/off signal | `q.backtest_event(panel, signal)` and `q.review_event` |
| Weights you already have | `q.backtest_weights(panel, weights, ...)` |
| The whole battery of tests, one page | `q.review_portfolio` / `q.review_event`, then `.report()` or `.to_json()` |
| Several strategies side by side | `q.compare({...}, "compare.html")` |
| Data | `pitbacktest.crypto` (Binance public archive, no key), `adapters.krx` (free key), `adapters.yfinance`, `adapters.long_format` (your own data) |
