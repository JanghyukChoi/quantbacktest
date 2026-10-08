# Amendment 1: the free tier stopped at 466 tickers

**Written and committed before any analysis was run on the downloaded prices.** Only counts have been looked at.

## What happened
The preregistered primary sample was "the first 500 tickers for which Tiingo returns prices". The free account has a limit of **500
distinct symbols per month**, and the 46 symbols used by the earlier probe (`docs/tiingo_probe.py`, 41 stocks, and a few checks) counted
against it. The limit was reached after the first **468** tickers of the seeded random order: **466 returned prices** and 2 returned
nothing (`KEG1`, `VREXV`). The next request (`BFS`) returned HTTP 200 with the message "You have run over your 500 symbol look up for this
month". Nothing was stored for it, and nothing beyond position 468 was requested.

## What changes, and what does not
- **The primary sample is the 466 tickers with prices among the first 468 of the random order.** This is the only deviation. The
  preregistration said that stopping early at a quota does not bias the sample (any prefix of a random order is a random sample) and set
  the "inconclusive" threshold at 300 tickers with prices; 466 is above it.
- Everything else in `PREREGISTRATION.md` stands: universe rules, factors, costs, the four parts, the reading rules and the limits.
- The two tickers with no data are reported by alive or delisted, as preregistered (Part 4).
- **Extension.** When the monthly counter resets (1 November 2026) the same random order will be continued to the preregistered 500
  with prices, and the unchanged analysis will be run again. The 500-ticker result is reported **next to** the 466-ticker one, and if the
  two differ in a way that changes a conclusion, both are reported and the difference is stated. The 466 result is not replaced.

## One more correction found on the way
The downloader treated the monthly-limit message (a dict in a 200 response) as an unexpected format and raised, which is why the run
stopped with an error and not quietly. The adapter now recognises that message as a quota stop (`tests/test_tiingo.py` U9). No data was
affected: a response that does not parse is never stored.
