"""Square-root market impact: the cost of trading a share of a security's daily volume.

A trade of `|dw|` of the book, with `aum` of money, is a fraction `part = |dw| x aum / ADV` of the security's average daily traded value, and costs `y x sigma x sqrt(part)` per
unit traded (sigma: the daily volatility; the cost is capped at `max_cost_bp`, which is also what a security with no volatility or volume history is charged). The coefficient
`y` is of order 1 in the literature and unknown for a given market; `capacity_curve` reads a list of them. Used by `backtest_weights` and `backtest_portfolio`."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .panel import Panel


@dataclass(frozen=True)
class ImpactModel:
    aum: float                 # money, in the currency of close x volume (dollars for US stocks and USDT contracts, won for KRX); participation = trade x aum / ADV
    y: float = 1.0             # coefficient of the square-root law
    vol_window: int = 20
    adv_window: int = 30
    max_cost_bp: float = 100.0  # cap per unit traded; also what a name without volatility or volume history is charged

    def __post_init__(self) -> None:
        if not (self.aum >= 0 and np.isfinite(self.aum)):
            raise ValueError(f"aum must be a finite amount of money (same currency as close x volume), not negative, got {self.aum!r}")
        if not (self.y >= 0 and np.isfinite(self.y)):
            raise ValueError(f"y must be finite and not negative, got {self.y!r}")
        if self.vol_window < 2 or self.adv_window < 1:
            raise ValueError("vol_window must be at least 2 and adv_window at least 1")
        if not (self.max_cost_bp >= 0):
            raise ValueError(f"max_cost_bp must not be negative, got {self.max_cost_bp!r}")


def _impact_cost(panel: Panel, H: np.ndarray, m: ImpactModel) -> tuple[np.ndarray, np.ndarray]:
    """(cost per signal date, participation |trade| * AUM / ADV per date and name)."""
    if panel.volume is None:
        raise ValueError("market impact needs panel.volume (dollar volume = close * volume)")
    sigma = panel.ret1().rolling(m.vol_window, min_periods=m.vol_window // 2).std().to_numpy(float)
    adv = panel.adv(m.adv_window).to_numpy(float)
    trade = np.zeros_like(H)
    trade[1:] = np.abs(np.diff(H, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        part = np.where(adv > 0, trade * m.aum / adv, np.inf)
        unit = m.y * sigma * np.sqrt(part)
    cap = m.max_cost_bp / 1e4
    unit = np.where(np.isfinite(unit), np.minimum(unit, cap), cap)       # unknown or untradable: charged the cap
    unit = np.where(trade > 0, unit, 0.0)
    return (trade * unit).sum(axis=1), np.where(trade > 0, part, np.nan)
