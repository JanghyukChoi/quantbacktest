# Survivorship bias with free data: what was measured

**Measured on 2026-10-07 with yfinance 1.7.0** (`docs/survivorship_probe.py` reruns it; Yahoo's data changes, so
yours may differ).

I looked up 42 well-known US stocks that were delisted, acquired, went bankrupt or changed ticker between 2008 and
2023 (Lehman, Bear Stearns, Silicon Valley Bank, Twitter, Time Warner, Celgene and so on). This is a convenience
sample chosen from memory, not a statistical sample, so no rate should be read from it.

| yfinance returned | count |
|---|---|
| no data at all | 37 |
| the price history of a **different company** that later reused the ticker (WM, GM, WB, SBNY, SHLD) | 5 |
| the correct history up to the delisting | **0** |

Two separate problems, and the second is the worse one:
1. **Missing names.** A universe built from yfinance contains only stocks that exist today. Every bankruptcy and
   every takeover target is gone, so returns are biased upward and drawdowns are understated.
2. **Wrong names.** A ticker is not a stable identifier. If a backtest stores `WM` for 2007 and gets Waste Management's
   prices from yfinance, it has silently replaced Washington Mutual, which went bankrupt, with a company that did not.
   There is no error and no NaN.

## What a free Tiingo key changes (measured 2026-10-07)

`docs/tiingo_probe.py` repeats the 41-stock probe against Tiingo's end-of-day API (free account, key in
`TIINGO_API_KEY`). Same caveat: a convenience sample from memory, so read the pattern, not a rate.

| Tiingo returned | count |
|---|---|
| the stock's own history up to its delisting | 23 |
| no data | 7 |
| data that runs past the delisting date (a later company reusing the ticker, or a rename) | 11 |

The split by **reason for leaving** matters more than the total. I sorted the 41 by hand into 12 distress cases (bankruptcy
or rescue sale: LEH, WB, BSC, CFC, WM, GM, CIT, SIVB, FRC, SBNY, SHLD, JCP) and 29 others (takeovers, renames). In the first
group **none** of the 12 came back with the failed company's own history: seven had no data or a later company's data
under the same ticker, and for GM and CIT the list only starts after the restructuring. In the second group 23 of 29 came back correct
(the rest: two ticker reuses, a rename with no data, a rename that continues under the new ticker, one with no data).

So Tiingo removes most of the takeover half of survivorship bias for free, and **not the bankruptcy half**, which is the
half with the large negative delisting returns. A universe built from it is less biased, not unbiased, and a
survivors-versus-Tiingo comparison would understate the bias. Ticker reuse is also present (WM returns Waste Management
from 1991; MON returns two different companies with a gap), so key securities by ticker *and* listing dates and split
at long gaps, as the KRX adapter does.

I did not check how many requests the free tier allows in total; the probe stops at the first HTTP 429.

## What `pitbacktest` does about it

- `Panel.audit()` reports `ended_before_end_share` and `survivorship_suspected`. In a real market a few percent of
  names disappear each year; a panel of 30+ names over 3+ years where almost none end early is a survivors-only panel.
  The yfinance adapter raises a warning when that happens.
- The crypto path (`pitbacktest.crypto`) does not have the problem: the Binance archive keeps delisted contracts, and 42%
  of the USDT perpetuals ever listed are delisted or halted. `survivors_only=True` reproduces the shortcut so the
  bias can be measured (`studies/crypto_cross_section`).
- Korea can be done without bias for free: the official KRX OpenAPI returns every stock listed on each day. On
  2013-2026 data (`studies/korea_survivorship`), running four factors on survivors only instead changes the net
  Sharpe by -0.23 to +0.20 depending on the factor (low volatility is understated, small size overstated), and by
  about zero on average. 13% of eligible security-days in 2013 belong to stocks no longer listed. The sign differs by
  factor, so "survivorship inflates returns" is not a safe assumption to correct for. The low-volatility
  understatement is the robust part (paired bootstrap interval clear of zero, and it survives neutralising against the
  other styles); the small-size overstatement mostly disappears once the other styles are neutralised.
- `backtest_portfolio(..., delist_return=...)` takes an explicit return for a name that leaves the sample. For US equities
  the literature gives sensible sensitivity values: about -30% for performance-related NYSE/AMEX delistings
  (Shumway 1997) and about -55% for Nasdaq (Shumway and Warther 1999).

## What a real fix needs

A point-in-time dataset keyed by a **permanent identifier** (not the ticker), with delisted securities, delisting
returns and dated index membership. The usual sources are CRSP (through WRDS), Sharadar, Norgate and Polygon; none is
free. The framework's contract (`Panel`: a date x security matrix plus a point-in-time `eligible` mask) is meant to
take such data as input; what it cannot do is create it.
