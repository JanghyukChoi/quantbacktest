# Assumptions that can change a result

Collected in one place after a reader asked what the specification leaves unsaid. "Stated" means it is also in `docs/verification_status.md` or the documentation of the function; "gathered here" means it was
true and documented only in the code or spread over several places.

| # | Assumption | Effect | Where |
|---|---|---|---|
| 1 | A signal seen at the close of day d is traded at the close of d + `entry_lag` (default 1). Slippage is a fixed spread and the square-root impact model; no intraday path, no order book | Optimistic for fast signals, more for illiquid names | gathered here |
| 2 | A delisted security is closed at its last price (return 0) unless `delist_return` is given | Optimistic; try -0.3 to -0.5 | stated |
| 3 | Returns are per unit of capital **in each leg**: a long-short book has gross exposure 2. No margin, no margin call, no borrowing limit; an account that loses 100 percent stops (`stop_at_ruin`) | Return on actual capital is lower; leverage risk is not measured | gathered here |
| 4 | Idle cash earns nothing; short proceeds earn no rebate; KRX series have no dividends; taxes only through `buy_bp` and `sell_bp` | Small for short holdings, large for dividend strategies | stated (dividends, taxes) |
| 5 | The risk-free rate is 0 in every Sharpe and Sortino ratio | Overstates Sharpe when rates are high | gathered here |
| 6 | A scalar `spread_bp` is a **round trip**, a (date x ticker) panel is **one way**; a scalar 20 equals a panel of 10 | A factor of 2 in cost if confused | stated |
| 7 | Cost is charged on changes of the target weights, not on the drift back to target | 2 to 3 percent of the turnover (10 to 14 bp a year at a 10 bp spread) | stated |
| 8 | Borrow cost is one number or a panel you supply; whether a name can be borrowed is not modelled unless `Panel.shortable` is given | Shorts in hard-to-borrow names are cheaper here than in life | stated |
| 9 | The number of variants tried is only what the ledger (or the grid of a review) recorded | The deflated Sharpe cannot count what was not written down | gathered here |
| 10 | Bootstrap intervals and the drawdown distribution resample blocks of one history (stationary bootstrap) | Optimistic if the regime changes | gathered here |
| 11 | The deflated Sharpe estimates the variance of the trials' Sharpe ratios from the recorded trials, and corrects for skew and kurtosis; similar variants make it count more trials than there are independent ideas | Conservative for near-duplicate variants | gathered here |
| 12 | The factor permutation asks whether the factor's link to the securities' own returns carries information, before cost; it does not ask whether the strategy is tradable | A significant permutation is not a tradable edge | stated |
| 13 | `eligible` (the point-in-time universe) is taken as given; the library cannot check that it was known on the day | Look-ahead in the universe is invisible | gathered here |
| 14 | Adjusted prices are taken as the provider's; a wrong split or dividend adjustment is not detected beyond the audit's extreme-return count | Spurious returns | gathered here |
| 15 | US free data has almost no bankruptcies, so a US result is a lower bound on survivorship bias | Optimistic | stated |
| 16 | The default benchmark is the market-capitalisation-weighted (or equal-weighted) return of the same universe, not a published index; give `benchmark_returns` for another | Alpha against the wrong index | gathered here |
| 17 | Overlapping holdings split capital equally among the tranches opened on different days | Differs from a book rebalanced at once | gathered here |
| 18 | Funding: a positive rate means longs pay and shorts receive | Sign errors in a data feed flip the result | gathered here |
| 19 | Borrow fees and buy/sell rates that vary with the date are read on the execution day (signal date + `entry_lag`); an unknown borrow fee on a held short raises; borrow is charged on the short leg | Pre-0.4.0 the rates were read on the signal date | gathered here |
| 20 | A held position whose price is missing earns 0 and is counted (`held_missing_price_days`, a warning); a halt (a price, no volume) and a flagged delisting are not counted | A vendor gap understates volatility and drawdown | gathered here |
| 21 | `entry_lag=0` trades at the same close the signal was computed on; the panel says so when it is chosen | Look-ahead unless the order is a market-on-close | gathered here |
| 22 | The engines are research simulators: targets, not shares; drift back to target is free; tranches start empty and the first build is not charged | Turnover and cost are estimates | gathered here |
| 23 | Forward returns are held in single precision (relative error about 6e-8) | Negligible | gathered here |
