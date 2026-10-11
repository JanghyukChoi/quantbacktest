# Blind re-implementation: does the specification determine the library?

Three assistants that had **not seen the library's code** were each given part of the specification (the document built from the code's own signatures, defaults and docstrings, plus the
explanation of the concepts) and asked to implement the same functions from it. They were told to write down every place the specification did not decide their code. Their
implementations are in `tests/blind_portfolio.py`, `tests/blind_event.py` and `tests/blind_stats.py`; `tests/test_blind_reference.py` compares them with the library on random inputs.

What this can find: places where the specification is silent or ambiguous (two careful readers would write different code), and places where the library does not do what its
description says. What it cannot find: a definition that the specification and the library state in the same wrong way; the assistants are the same kind of model as the author.

## Result

| Part | Compared | Outcome |
|---|---|---|
| Portfolio engine (`blind_portfolio.py`) | the net-return series of 108 combinations (entry lag 0 to 2, holding 1/3/5, spread 0/10, three weightings, long-short and long-only, with unknown scores) and the metrics | **identical** to 1e-10 |
| Event statistics (`blind_event.py`) | 15 figures of 108 combinations (lag, horizon, cost, firing density) | **identical** (the library keeps forward returns in single precision: relative 6e-8) |
| Statistics (`blind_stats.py`) | Newey-West, Sharpe, deflated Sharpe, PBO (also when the length is not a multiple of the blocks), exact sign test, Wilson interval, BH and Holm, mean tests, relative-to-benchmark figures, alpha and betas | **identical** |

Where the two **differed**, the specification was incomplete, not wrong. Each was closed in the documentation of the library (version 0.4.0) and pinned by a test:

| Hole | Blind reading | Library | Now stated in |
|---|---|---|---|
| `turnover_daily` was not defined | the sum of absolute weight changes | **one-way**: half of that | `backtest_portfolio` |
| Newey-West below 30 observations; a lag longer than the sample | computed anyway, lag capped at T - 1 | NaN below 30; the weights keep the requested lag | `newey_west_t` |
| Turnover of the firing set when one day has no firing | counted as 1 | the pair is skipped | `fire_structure` |
| Years with few bars in the yearly gate; short sides of the holdout | every year counts | a year needs 40 bars, a holdout side 30 | `yearly`, `oos_holdout` |
| Gate G6 without `fire` | a failure | not evaluated (five gates) | `run_gates` |
| `vol_now` and the start of the volatility estimate in `risk_profile` | per bar; started from the full-sample variance | annualised; started from the first 20 bars | `risk_profile` |
| Residual volatility in `alpha_beta` | divides by n - 1 | divides by n minus the number of coefficients | `alpha_beta` |
| The base rate of an event: net of cost? Which days? Which lag? | the same as the library (all three guessed alike) | now written down | `backtest_event` |
| Forward returns | double precision | single precision | `Panel.forward` |

One more finding in the other direction: the assistants' own sign-test code overflowed above 1,023 observations (`2.0 ** n`), the defect this library had and fixed after a
run on real data. The ambiguity files (`AMBIGUITIES_A.md`, `_B.md`, `_C.md`) are the assistants' own lists, kept as written.
