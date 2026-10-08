# What the order book implies for the impact coefficient Y

`ImpactModel` charges a trade of notional `Q` a unit cost `Y * sigma * sqrt(Q / ADV)`. `Y` is "of order 1 in the literature" and has never been
measured here, which is why `capacity_curve` takes a list of values. This note asks what the **visible order book** implies for `Y`, as a bracket for that
list. It is not a calibration against executions.

**Data.** Binance USDT-M `bookDepth` daily files (public, no key): every 30 seconds the cumulative resting notional within 1 to 5 percent of the mid, on
each side. 30 contracts chosen by rank of trailing turnover (from $5 million to $22 billion a day), 10 days between 2023-02-15 and 2024-04-17: 300
contract-days. For an order of a given share of the day's turnover (`ADV`, trailing 30 days) we walk the median book, linear inside each 1 percent band, and
take the average price paid against the mid. `Y` is that slippage divided by `sigma * sqrt(share)` with `sigma` the trailing 20-day daily volatility (median
4.1 percent). Code: `docs/crypto_impact_study.py`; per-contract results: `docs/crypto_impact_study_results.csv`.

## Result: median implied Y

| Trailing turnover | order = 0.01% | 0.03% | 0.1% | 0.3% | 1% of the day's turnover |
|---|---:|---:|---:|---:|---:|
| < $50M | 0.12 | 0.20 | 0.37 | 0.64 | 1.19 |
| $50 to 200M | 0.14 | 0.25 | 0.45 | 0.78 | 1.48 |
| $200M to 1B | 0.21 | 0.36 | 0.65 | 1.13 | 2.18 |
| > $1B | 0.29 | 0.51 | 0.93 | 1.60 | 2.71 |
| all | 0.14 | 0.24 | 0.43 | 0.75 | 1.40 |

The 10th to 90th percentile of `Y` at 0.1 percent of turnover is 0.27 to 0.78. The book within 1 percent of the mid holds a median of 0.93 percent of a
day's turnover (10th to 90th percentile 0.41 to 1.92 percent). At 1 percent of turnover, 7 percent of the contract-days have an order larger than the 5
percent book shows (those are NaN, not counted).

At an order of 0.1 percent of turnover, the model with `Y = 1` charges a median 13.1 bp per unit traded against 5.4 bp implied by the book (0.43 times). At 1
percent of turnover it charges 41.4 bp against 53.3 bp (1.3 times).

## How to read it

- **`Y = 1` overstates the cost of small orders and understates the cost of large ones.** For orders up to about 0.3 percent of a day's turnover, the book
  implies `Y` of 0.1 to 0.8; at 1 percent, 1.2 to 2.7. `capacity_curve(..., y_values=(0.5, 1.0, 2.0))` brackets that range for orders up to 1 percent of
  turnover. The default `ImpactModel(y=1.0)` is unchanged.
- **The rise of `Y` with size is built in; it is not a finding.** Inside the first 1 percent band the book is interpolated linearly, so slippage is exactly
  proportional to `Q` and `Y` rises as the square root of the size (a log-log slope of 0.5). The data cannot tell a square-root law from a linear one at
  these sizes, because only the 1 to 5 percent bands are given and almost every order falls inside the first.
- **It is the cost of taking the visible book at once.** Resting depth refills, real orders are split over time (cheaper), and a trade carries information
  that the book does not (dearer). The sign of the net error is not known.
- **The depth near the mid is not observed.** Uniform depth inside the 1 percent band is an assumption. If the book is thinner near the mid, small orders cost
  more than shown here; if thicker, less.
- 30 contracts, 10 days, one exchange, 2023 to 2024. A stressed day would look different, and so would a contract in the first weeks after listing.
