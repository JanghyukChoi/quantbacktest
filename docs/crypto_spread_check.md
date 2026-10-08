# What the library assumes for crypto spreads, against the order book

`pitbacktest.crypto.costs.liquidity_cost_bp` has always said that its spread is an **assumption**: a flat half spread of 2 bp, and 5 bp more for contracts
with a trailing turnover under $100 million a day. This note measures the quoted spread from the public top-of-book files and compares.

**Data.** Binance USDT-M perpetuals, `bookTicker` daily files from `data.binance.vision` (best bid and ask with every update; public, no key). Daily files exist
from 2023-05-16 to 2024-03-30, so that is the only period measured. Twelve contracts chosen by rank of trailing turnover from the top to the bottom of the eligible
universe (158 contracts eligible on all three days), three days each (2023-09-13, 2023-12-06, 2024-02-14): 36 contract-days, 105 million quote updates, no crossed
quote. Each quote counts for the time until the next update (at most 60 seconds). Script: `docs/crypto_spread_probe.py`.

## Result

| Contract | 30-day turnover ($M/day) | Measured half spread, median (bp) | 90th pct (bp) | Library default (bp) |
|---|---:|---:|---:|---:|
| BTCUSDT | 10,491 | 0.010 | 0.010 | 2.0 |
| ETHUSDT | 4,816 | 0.020 | 0.020 | 2.0 |
| XRPUSDT | 579 | 0.923 | 0.930 | 2.0 |
| AVAXUSDT | 386 | 0.283 | 0.283 | 3.7 |
| TRBUSDT | 296 | 0.093 | 0.393 | 2.0 |
| RUNEUSDT | 200 | 1.627 | 1.700 | 3.7 |
| CFXUSDT | 136 | 3.040 | 3.093 | 2.0 |
| FETUSDT | 81 | 1.303 | 1.317 | 5.3 |
| XLMUSDT | 55 | 0.417 | 0.420 | 5.3 |
| GTCUSDT | 34 | 4.503 | 4.637 | 7.0 |
| JASMYUSDT | 20 | 1.040 | 1.050 | 7.0 |
| QNTUSDT | 8 | 0.490 | 1.127 | 7.0 |

The quoted half spread was **below the default for 11 of 12 contracts** and above it for one (CFX: 3.0 bp against 2.0). For the two largest contracts the default is
100 to 200 times the quote. Across all 36 contract-days the median measured half spread was 0.19 bp for contracts above $100M turnover (default 2 bp) and
0.82 bp below it (default 7 bp).

## The spread is one tick

For 9 of the 12 contracts the measured median was within 0.8 to 1.25 times **half a price tick over the price** (`0.5 * tick_size / price`), using the tick size the
exchange reports today (`ArchiveStore.symbol_rules()`). The three others (RUNE, CFX, GTC) were 9.9, 9.9 and 100 times wider than that prediction: the exchange has
since cut their tick by a factor of 10, 10 and 100. So the tick-based estimate is a **floor** where the tick has been cut, because the rules are today's values and
the endpoint has no history. It is available as `liquidity_cost_bp(panel, spread_estimator="tick", tick_size=...)`.

## What this does not say

- It is the cost of a **small order at the best quote**. An order larger than the size quoted there walks the book and pays more; impact is not in it. The default
  2 bp and 5 bp may be a reasonable stand-in for the total cost of a real trade, but they are not the spread, and they were not derived from a measurement.
- 36 contract-days in a ten-month window of 2023 to 2024, one exchange, and quiet days were not selected for or against. Spreads widen in stress.
- The default has **not** been changed: the preregistered crypto study (`studies/crypto_cross_section`) was run with it, and changing it would change that study.
  Run your own work with `spread_estimator="tick"` and a grid of impact assumptions instead of one flat number.
