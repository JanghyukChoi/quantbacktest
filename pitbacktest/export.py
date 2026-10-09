"""Results as plain JSON.

`result.to_dict()` and `result.to_json()` give a dictionary of numbers, dates and text with nothing from pandas or numpy in it, so that a report writer, a notebook,
or a program that calls the library as a tool (for example an assistant over MCP) can read a result without knowing the library's classes. Every figure carries its unit
in `units`, and the warnings the run raised are in `notes`. `null` means not available or not computable (NaN or infinity in Python); it is never a zero.

The layout is versioned (`schema_version`). Fields are added, not renamed, within a version."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

# The unit of each figure, written from the code that computes it. Only figures whose definition was checked are listed.
PORTFOLIO_UNITS = {
    "CAGR": "annualised compound growth rate of the net returns, as a fraction (0.10 = 10 percent a year)",
    "MDD": "largest fall of the net equity curve from a peak, as a negative fraction (-0.25 = -25 percent)",
    "Sharpe": "annualised mean over standard deviation of the net returns (no risk-free rate subtracted)",
    "Sortino": "annualised mean over the downside deviation (target 0) of the net returns",
    "Calmar": "CAGR divided by the absolute maximum drawdown",
    "vol": "annualised standard deviation of the net returns, as a fraction",
    "years": "number of bars divided by periods_per_year",
    "pos_days": "share of bars with a positive net return, as a fraction",
    "mdd_peak": "date of the equity peak before the largest drawdown",
    "mdd_trough": "date of the lowest point of the largest drawdown",
    "mdd_recovered": "date the equity regained the earlier peak, or the text 'not recovered'",
    "ruined": "true when the account lost 100 percent or more in one bar; the return is then -100 percent on that bar and 0 afterwards",
    "ruin_date": "signal date of that bar",
    "gross_CAGR": "CAGR before trading costs and funding",
    "cost_annual_bp": "trading costs (spread and buy/sell side costs) in basis points of capital per year: mean cost per bar x periods_per_year x 10,000",
    "side_cost_annual_bp": "the part of the costs that comes from buy_bp and sell_bp, in basis points of capital per year",
    "spread_annual_bp": "spread cost in basis points of capital per year (backtest_weights)",
    "borrow_annual_bp": "short borrow cost in basis points of capital per year (backtest_weights)",
    "impact_annual_bp": "market impact cost in basis points of capital per year",
    "funding_annual_bp": "funding paid (positive) or received (negative) in basis points of capital per year",
    "turnover_daily": "one-way traded share of the portfolio weights per bar (half the sum of absolute weight changes)",
    "avg_positions": "mean number of securities held on bars with a position (backtest_portfolio)",
    "avg_gross_exposure": "mean sum of absolute weights (backtest_weights)",
    "avg_net_exposure": "mean sum of signed weights (backtest_weights)",
    "participation_p99": "99th percentile of a trade's size as a share of the security's average daily traded value, with the AUM given to the impact model",
    "trades_over_10pct_adv": "share of trades above 10 percent of the average daily traded value",
}
EVENT_UNITS = {
    "n_trades": "number of fired signals with a measured outcome",
    "win_rate": "percent of trades with a positive return after the round-trip cost",
    "base_rate": "percent of all eligible securities on the same days with a positive return after the cost (what a random pick would show)",
    "lift_pp": "win_rate minus base_rate, in percentage points",
    "mean_bp": "mean trade return after the round-trip cost, in basis points",
    "median_bp": "median trade return after the cost, in basis points",
    "avg_win_bp": "mean of the winning trades, in basis points",
    "avg_loss_bp": "mean of the losing trades, in basis points",
    "payoff": "average win divided by the absolute average loss",
    "p25_bp": "25th percentile of the trade return, in basis points",
    "p75_bp": "75th percentile of the trade return, in basis points",
    "skew_warning": "true when the mean is positive and the median negative: a few large winners carry many small losses",
}


def jsonable(x):
    """Convert numpy, pandas and Python objects to JSON types. NaN and infinity become None; mapping keys become text; timestamps become ISO text."""
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(x).isoformat()
    if isinstance(x, pd.DataFrame):
        return {"columns": [str(c) for c in x.columns], "index": [jsonable(i) for i in x.index], "data": [[jsonable(v) for v in row] for row in x.to_numpy(dtype=object)]}
    if isinstance(x, pd.Series):
        return {"index": [jsonable(i) for i in x.index], "data": [jsonable(v) for v in x.to_numpy(dtype=object)]}
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in x]
    return str(x)


def _header(kind: str, notes) -> dict:
    from . import __version__
    return {"schema": f"pitbacktest/{kind}", "schema_version": SCHEMA_VERSION, "library_version": __version__,
            "null_means": "not available or not computable (NaN or infinity), never zero", "notes": list(notes or [])}


def _check_series_arg(series: str) -> str:
    if series not in ("none", "monthly", "full"):
        raise ValueError(f"series must be 'none', 'monthly' or 'full', not {series!r}")
    return series


def portfolio_to_dict(res, series: str = "monthly") -> dict:
    """`PortfolioResult` as a JSON-ready dictionary. `series`: 'none' leaves the return series out, 'monthly' gives compounded monthly returns and the month-end equity
    (small), 'full' gives every bar's net return. The weights (`holdings`) are never included: they are large; take them from the result object."""
    series = _check_series_arg(series)
    out = _header("portfolio-result", getattr(res, "notes", []))
    out.update({
        "spec": jsonable(res.spec), "metrics": jsonable(res.metrics),
        "units": {k: PORTFOLIO_UNITS[k] for k in res.metrics if k in PORTFOLIO_UNITS},
        "benchmark": jsonable(res.benchmark), "excess": jsonable(res.excess), "yearly": jsonable(res.yearly), "grid": jsonable(res.grid),
        "conventions": "returns are net of costs and funding, per bar, per unit of capital in each leg for backtest_portfolio (long and short legs each sum to 1) "
                       "and per unit of capital for backtest_weights; a signal on date d enters at close(d + entry_lag) and earns close(d + entry_lag) to close(d + entry_lag + 1)",
    })
    s = res.net_returns
    if series != "none" and s is not None:
        if series == "full":
            out["series"] = {"dates": [d.date().isoformat() for d in s.index], "net_returns": jsonable(s.to_numpy())}
            if res.benchmark_returns is not None:
                out["series"]["benchmark_returns"] = jsonable(res.benchmark_returns.reindex(s.index).to_numpy())
        else:
            g = (1.0 + s).groupby(s.index.to_period("M")).prod() - 1.0
            eq = (1.0 + s).cumprod().groupby(s.index.to_period("M")).last()
            out["series"] = {"months": [str(p) for p in g.index], "net_return": jsonable(g.to_numpy()), "equity_at_month_end": jsonable(eq.to_numpy())}
            if res.benchmark_returns is not None:
                b = res.benchmark_returns.reindex(s.index)
                out["series"]["benchmark_return"] = jsonable(((1.0 + b.fillna(0.0)).groupby(b.index.to_period("M")).prod() - 1.0).to_numpy())
    return out


def event_to_dict(res) -> dict:
    out = _header("event-result", getattr(res, "notes", []))
    out.update({"spec": jsonable(res.spec), "per_horizon": jsonable(res.per_horizon), "units": dict(EVENT_UNITS), "gates": jsonable(res.gates),
                "structure": jsonable(res.structure), "neutralized": jsonable(res.neutralized), "lookahead": jsonable(res.lookahead)})
    return out


def screen_to_dict(res) -> dict:
    out = _header("screen-result", getattr(res, "notes", []))
    out.update({"survivors": list(res.survivors), "funnel": jsonable(res.funnel), "null": jsonable(res.null), "summary": jsonable(res.summary),
                "factors": jsonable(res.factors),
                "reading": "`summary` has one row per factor; `failed_at` names the first gate it failed; t, neu_t and neu_survival_% are computed with controls, "
                           "excess_bp, net_bp and rho without them (see the documentation of `screen`)"})
    return out


def dump_json(d: dict, path=None, indent: int | None = 2) -> str | Path:
    """Write `d` as JSON text. Without `path` the text is returned; with one the file is written and its path returned. NaN cannot appear (it was turned into null)."""
    text = json.dumps(d, indent=indent, allow_nan=False, ensure_ascii=False)
    if path is None:
        return text
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    return p
