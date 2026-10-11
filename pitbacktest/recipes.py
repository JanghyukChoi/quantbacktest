"""Factor recipes: named, causal, documented factor definitions.

A factor written from scratch, by a person or by an assistant, is where most look-ahead enters: a negative shift, a centred window, a z-score over the whole history. A recipe is a
definition that has already been checked: it uses only data up to the row it is written on (`Panel.assert_causal` passes on every one, and `tests/test_recipes.py` runs it),
leaves a cell empty (NaN) where there is not enough history instead of filling it, and says which way it points. In every recipe a **higher score means the security goes into the
long leg**, so the result of `recipe(panel, name)` can be handed to `backtest_portfolio`, `review_portfolio` or `screen` as it is.

    from pitbacktest import recipes
    recipes.describe()                       # every recipe: what it is, its parameters and defaults, what it needs from the panel
    f = recipes.recipe(panel, "momentum")    # a (date x ticker) frame; recipes.recipe(panel, "reversal", lookback=3) changes a parameter

A recipe is a hypothesis, not a finding: that low volatility or short-term losers earn more is something to test, not something the library believes. Trying several recipes or
several parameter values is trying variants: record every run with a `Ledger` so that the count is kept (`review_portfolio(..., ledger=...)`).
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

__all__ = ["recipe", "describe", "names"]


def _int(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be a whole number of bars, at least {minimum}, got {value!r}")
    return int(value)


def _need(panel, field: str, recipe_name: str):
    value = getattr(panel, field)
    if value is None:
        raise ValueError(f"recipe {recipe_name!r} needs panel.{field}, which this panel does not have")
    return value


def momentum(panel, lookback: int | None = None, skip: int | None = None) -> pd.DataFrame:
    """The return from `lookback` bars ago to `skip` bars ago (the classic 12-1 month momentum: skipping the latest month, which tends to reverse). Higher = stronger past winner.
    Defaults: `lookback` is one year of bars (`panel.periods_per_year`), `skip` one month (a twelfth of it). NaN until `lookback` bars exist."""
    ppy = int(round(panel.periods_per_year))
    lookback = _int(ppy if lookback is None else lookback, "lookback", 2)
    skip = _int(max(1, round(ppy / 12)) if skip is None else skip, "skip", 1)
    if skip >= lookback:
        raise ValueError(f"skip ({skip}) must be smaller than lookback ({lookback})")
    c = panel.close
    return c.shift(skip) / c.shift(lookback) - 1.0


def reversal(panel, lookback: int = 5) -> pd.DataFrame:
    """Minus the return of the last `lookback` bars: last period's losers score high. Short-term reversal is a hypothesis, and it is also what an expensive trading cost eats first."""
    lookback = _int(lookback, "lookback")
    return -panel.close.pct_change(lookback, fill_method=None)


def low_volatility(panel, window: int = 60) -> pd.DataFrame:
    """Minus the standard deviation of one-bar returns over `window` bars (at least two thirds of them present): the calmest securities score high."""
    window = _int(window, "window", 5)
    r = panel.close.pct_change(fill_method=None)
    return -r.rolling(window, min_periods=max(3, (2 * window) // 3)).std()


def size(panel) -> pd.DataFrame:
    """Minus the market capitalisation (zero or negative values are unknown): small securities score high. Needs `panel.mkt_cap`."""
    cap = _need(panel, "mkt_cap", "size")
    return -cap.where(cap > 0)


def illiquidity(panel, window: int = 30) -> pd.DataFrame:
    """Minus the average traded value (close x volume) over `window` bars (at least two thirds of them present): thinly traded securities score high. Needs `panel.volume`.
    The securities that score high here are also the costly ones to trade: read the cost table of a review."""
    window = _int(window, "window", 3)
    vol = _need(panel, "volume", "illiquidity")
    dv = (panel.close * vol).rolling(window, min_periods=max(2, (2 * window) // 3)).mean()
    return -dv.where(dv > 0)


def volume_shock(panel, short: int = 5, window: int = 60) -> pd.DataFrame:
    """The average volume of the last `short` bars over the average of the last `window` bars, minus 1: securities trading unusually heavily score high. Needs `panel.volume`.
    Heavy volume is neither good nor bad news on its own; this ranks it, it does not say what it means."""
    short, window = _int(short, "short"), _int(window, "window", 2)
    if short >= window:
        raise ValueError(f"short ({short}) must be smaller than window ({window})")
    vol = _need(panel, "volume", "volume_shock").where(lambda v: v >= 0)
    base = vol.rolling(window, min_periods=window).mean()
    return vol.rolling(short, min_periods=short).mean() / base.where(base > 0) - 1.0


def distance_to_high(panel, window: int | None = None) -> pd.DataFrame:
    """The price over its highest close of the last `window` bars, minus 1 (zero at a new high, negative below it): securities near their high score high (the 52-week-high idea).
    Default `window`: one year of bars. NaN until `window` bars exist."""
    window = _int(int(round(panel.periods_per_year)) if window is None else window, "window", 5)
    c = panel.close
    return c / c.rolling(window, min_periods=window).max() - 1.0


_RECIPES = {
    "momentum": (momentum, ["close"]),
    "reversal": (reversal, ["close"]),
    "low_volatility": (low_volatility, ["close"]),
    "size": (size, ["mkt_cap"]),
    "illiquidity": (illiquidity, ["close", "volume"]),
    "volume_shock": (volume_shock, ["volume"]),
    "distance_to_high": (distance_to_high, ["close"]),
}


def names() -> list[str]:
    """The recipe names."""
    return list(_RECIPES)


def describe(name: str | None = None) -> dict:
    """What each recipe is: `{name: {"definition", "parameters": {parameter: default}, "needs": [panel fields], "higher_is": "long"}}`, or just one entry when `name` is given.
    A parameter whose default is None takes a value from the panel (for example one year of bars)."""
    def one(key):
        fn, needs = _RECIPES[key]
        params = {p.name: (None if p.default is inspect.Parameter.empty else p.default) for p in list(inspect.signature(fn).parameters.values())[1:]}
        return {"definition": " ".join((fn.__doc__ or "").split()), "parameters": params, "needs": needs, "higher_is": "long"}
    if name is None:
        return {k: one(k) for k in _RECIPES}
    _check_name(name)
    return one(name)


def _check_name(name: str) -> None:
    if name not in _RECIPES:
        raise ValueError(f"unknown recipe {name!r}; the recipes are {', '.join(_RECIPES)} (see recipes.describe())")


def recipe(panel, name: str, **params) -> pd.DataFrame:
    """The factor of a recipe on `panel`: a (date x ticker) frame on the panel's own dates and tickers, higher = long. NaN where the history is too short or an input is missing.
    Unknown names and parameters raise a ValueError that lists what is available."""
    _check_name(name)
    fn, _ = _RECIPES[name]
    allowed = [p for p in inspect.signature(fn).parameters if p != "panel"]
    bad = [k for k in params if k not in allowed]
    if bad:
        raise ValueError(f"recipe {name!r} has no parameter {bad[0]!r}; its parameters are {allowed or 'none'}")
    out = fn(panel, **params)
    return out.reindex(index=panel.close.index, columns=panel.close.columns).astype(float)
