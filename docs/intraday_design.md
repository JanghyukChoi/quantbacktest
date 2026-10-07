# Intraday layer: design (written before the code)

## What it is for, and what it is not
**For.** Answering the questions that only exist at high frequency, with the same point-in-time and cost discipline as the
daily engine:
- how fast does an edge decay with **latency** (the same signal entered 1, 2, 5, 15 bars late)?
- at what **one-way cost** does a signal stop paying, given how often it turns over?
- does a result survive moving from daily to 5-minute bars at all?

**Not for.** Order-book simulation, queue position, partial fills, limit orders or market making. The archive has trades
aggregated into bars and no book, so none of that can be tested honestly here. A latency or cost result from this layer
says how fragile a signal is; it does not say it can be traded. The documentation states this next to every function.

## Data
Binance public archive, USDT-margined perpetuals, 1-minute klines: `data/futures/um/monthly/klines/<SYM>/1m/` for complete
months and `.../daily/klines/<SYM>/1m/` for the current partial month. No key is needed. Delisted contracts are in the archive,
so the universe is not limited to survivors (as for the daily crypto panel). Files are cached as pickles under
`~/.cache/quantbt/binance_um/minutes/<SYM>/<YYYY-MM>.pkl`; downloads are resumable and a missing file (404) is recorded, not retried.

Timestamps are milliseconds in older files and microseconds in newer ones; the unit is detected from the magnitude of each file.
Header rows and duplicate rows are tolerated; a file that does not parse raises, and nothing half-read is cached.

## Time convention (the part that goes wrong)
Rows are indexed by **bar end time** (UTC, naive): `open_time + 1 minute` for 1-minute bars. A row therefore holds only information
available at that instant, so a signal computed from row t may be entered at row t + lag with `lag >= 1`, never at t itself.
(The daily engine allows `entry_lag = 0` for a market-on-close entry; for bars that would trade at a price you only know once
the bar is finished, so the intraday builder refuses `entry_lag < 1`.)

Coarser bars (`bar="5min"`, `"15min"`, `"1h"`) are aggregated from 1-minute bars: open first, high max, low min, close last,
volume and quote volume summed, labelled by their end. A last bar that would end after the data ends is dropped. A minute with no
trade is missing (NaN), not zero; a bar with no minute at all is NaN, and the engine's rule for a missing price (cash, return 0)
applies. Bars with zero quote volume (halts, the tail before a delisting) are removed as in the daily panel. The number of partial
bars (fewer minutes than the bar length) is reported in `panel.meta`.

## Point-in-time universe
Daily liquidity is the sum of the day's quote volume. A contract is eligible on day D only if, **using days up to D - 1**,
it has at least `min_age_days` of history and a trailing median daily turnover of at least `min_adv_usd`; `top_n` keeps the most
liquid of those. Every bar of day D uses that flag. Nothing from day D itself enters its own eligibility, which is stricter than the
daily panel (whose day-t flag may use day t because the daily signal is formed after the close).
Warm-up: the builder loads `max(adv_window, min_age_days) + 1` days before `start`, and fails with a clear message if those days were
not downloaded, instead of silently using a shorter window.

## Funding
Settlements keep their timestamps (the existing daily store sums them per day, which loses the instant). A rate is placed on the bar
whose end equals the settlement time rounded to the nearest minute (the archive stamps them a few milliseconds late), or on the next
bar end if the bar length does not divide it. A position held through that bar pays or receives it, which is the engine's
convention (funding at the end of the holding bar).

## Costs
`backtest_intraday(panel, factor, one_way_bp=...)` takes the **one-way** cost and converts it for `backtest_portfolio`
(whose scalar argument is a round trip), so this unit trap does not exist here. A flat taker fee plus half spread is an assumption, not a
measurement; use `breakeven_cost` rather than trusting one number. Net returns are linear in the cost per unit traded, so the
break-even cost is computed exactly from two runs:
`c* = mean(net at 0 bp) / mean(net at 0 bp - net at 1 bp)` in basis points, and a third run checks that the net mean at `c*` is zero.

## Latency
`latency_sweep(panel, factor, lags=(1, 2, 5, 15))` reruns the same portfolio with `entry_lag = lag` bars and reports net and gross
Sharpe, CAGR and turnover per lag, with the delay in minutes. A real edge decays smoothly; a result that is large at lag 1 and
gone at lag 2 is a bar-alignment artefact or a signal nobody can trade.

## Size
One year of 1-minute bars is about 525,600 rows; for 40 contracts that is 21 million cells per matrix and the engine holds around eight
of them. `estimate_memory` is called before building and the builder raises if the estimate is above `max_gb` (default 3), naming the
bar length that would fit. Practical limits: 1-minute bars for a few months and a few dozen contracts, 5-minute bars for a year.

## Annualisation and short samples
`periods_per_year = 365 * 1440 / bar_minutes` (the market never closes). `backtest_portfolio` already returns NaN metrics for a
sample shorter than half a year; for intraday studies, which are often a few months, that silence is a trap, so it now **warns**
with the length it found instead of returning NaN without a word.

## Tests, written from this document
Fake archive, no network. Both timestamp units give the same frame; end-time labelling; 1m to 5m aggregation equals an independent
loop, including gaps and an incomplete last bar; eligibility on day D is unchanged when day D or later is altered; a signal that
knows only the next bar's return earns at lag 1 and nothing at lag 2 (a known-answer latency curve); a position held through a
funding instant pays it exactly once and a flat one pays nothing; the 0.003-second stamp lands on the right bar; periods per year;
break-even cost makes the net mean zero in an independent rerun; the memory guard; resumable download and a recorded 404; and a
halted or missing stretch becomes NaN and a flagged delisting. Planted bugs (a shifted label, a signal that sees its own bar, a
wrong funding bar, a dropped partial bar) must be caught.

## Known limits (written before the code)
- Bars, not trades or a book: no market impact below the bar, no queue, no partial fills. The square-root impact model in
  `backtest_weights` is the only size effect available, and it is a daily-scale model applied to bar volumes.
- Quote volume is the archive's own, including wash trading, which is not filtered.
- Binance only, perpetuals only, and only since about 2020 for most contracts.
- A signal that works at one bar size and not at another is information about the signal, but nothing here tells you which is right.
