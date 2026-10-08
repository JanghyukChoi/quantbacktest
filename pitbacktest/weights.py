"""Simulate portfolio weights you supply, with spread, borrow and market-impact costs, and read off capacity.

`backtest_portfolio` turns a factor into quantile portfolios. Institutions separate the steps: a signal, then a portfolio
construction step (an optimiser with risk and turnover limits), then a simulation. This module is the third step. It does
not build portfolios and does not ship an optimiser; give it the weights your optimiser produced.

Timing (the same as `backtest_portfolio`)
  weights.loc[d] are target weights, as a fraction of capital, chosen with data up to the close of day d. They are entered at
  close(d + lag), earn close(d + lag) -> close(d + lag + 1), and are replaced by the next row. Long is positive, short negative.
  Gross exposure may exceed 1. Rows with no change cost nothing; the first row, a build from cash, is not charged (as in
  `backtest_portfolio`, so the two agree exactly).

Costs, all charged on the signal date of the trade, in return per unit of capital
  spread_bp   scalar: round-trip spread, each unit traded pays half of it. Panel (date x ticker): one-way cost in bp.
              The same units as `backtest_portfolio`.
  borrow_bp   annual borrow fee in bp (scalar or panel) on the short market value, charged per period.
  impact      square-root law, per unit traded:  Y * sigma * sqrt(|trade| * AUM / ADV)
              sigma the trailing daily volatility, ADV the trailing average dollar volume, both known at the signal date.
              Y is of order 1 in the literature (Torre 1997; Almgren et al. 2005; Toth et al. 2011) but it is not known for a
              given market, so run `capacity_curve` over several values instead of trusting one. Capital is taken to equal
              AUM throughout (no compounding of the base).

You cannot open or increase a position in a security that is not eligible that day: an optimiser that buys names outside
the point-in-time universe is using information it should not have. A position already held may stay after the name leaves the
universe until you reduce it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core.costs import apply_turnover_cost
from .core.panel import Panel
from .portfolio import PortfolioResult, _benchmark_returns, _check_benchmark, _check_cost, _forward_arrays, metrics


@dataclass(frozen=True)
class ImpactModel:
    aum: float                 # dollars of capital
    y: float = 1.0             # coefficient of the square-root law
    vol_window: int = 20
    adv_window: int = 30
    max_cost_bp: float = 100.0  # cap per unit traded; also what a name without volatility or volume history is charged

    def __post_init__(self) -> None:
        if not (self.aum >= 0 and np.isfinite(self.aum)):
            raise ValueError(f"aum must be a finite number of dollars, not negative, got {self.aum!r}")
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


def backtest_weights(panel: Panel, weights: pd.DataFrame, *, spread_bp=0.0, borrow_bp=0.0,
                     impact: ImpactModel | None = None, funding: bool = True, delist_return: float | None = None,
                     benchmark: str | None = "cap", ledger=None, family: str = "default",
                     name: str | None = None, check_universe: bool = True) -> PortfolioResult:
    """Net returns of the weights you supply. See the module docstring for timing and costs.

    `check_universe` raises if a position is opened or increased in a security that is not eligible that day. It looks at net
    weights, so a name that is long from one signal and short from another (the two cancel) can show an increase when the
    short expires; holdings built from overlapping long and short tranches (`PortfolioResult.holdings`) can trip it, and
    then `check_universe=False` is the honest setting. Weights from an optimiser that nets positions are not affected.

    Costs are charged on the **net** trade per security. `backtest_portfolio` charges its long and short legs as separate sleeves,
    so feeding it its own overlapping-tranche holdings can come out slightly cheaper here (a name long in one tranche and short in
    another is netted); without such overlap the two agree exactly."""
    _check_benchmark(benchmark)
    _check_cost(spread_bp, "spread_bp")
    _check_cost(borrow_bp, "borrow_bp")
    W = weights.reindex(index=panel.dates, columns=panel.tickers)
    raw = W.to_numpy(float)
    if np.isinf(raw).any():
        raise ValueError("weights contain inf")
    n_nan = int(np.isnan(raw).sum())
    H = np.nan_to_num(raw, nan=0.0)
    prev = np.vstack([np.zeros((1, H.shape[1])), H[:-1]])
    bad = (np.abs(H) > np.abs(prev) + 1e-12) & ~panel.eligible.to_numpy(bool)      # opening or increasing outside the universe
    if check_universe and bad.any():
        i, j = np.argwhere(bad)[0]
        raise ValueError(f"{int(bad.sum())} positions opened or increased on securities that are not eligible that day "
                         f"(first: {panel.tickers[j]} on {panel.dates[i].date()}); the universe is point in time")
    fwd, fwdf, hit_next = _forward_arrays(panel, funding, delist_return)
    cut = len(panel.dates) - (panel.entry_lag + 1)
    gross = (H * fwd).sum(axis=1)
    spread = apply_turnover_cost(H, spread_bp)                          # the first row, a build from cash, is not charged
    short = np.maximum(-H, 0.0)
    b = np.asarray(borrow_bp, float)
    borrow = (short * (b / 1e4 / panel.periods_per_year)).sum(axis=1) if b.ndim == 0 else \
        (short * (np.nan_to_num(b, nan=0.0) / 1e4 / panel.periods_per_year)).sum(axis=1)
    fcost = (H * fwdf).sum(axis=1)
    imp, part = (np.zeros(len(H)), None)
    if impact is not None:
        imp, part = _impact_cost(panel, H, impact)
    net = gross - spread - borrow - imp - fcost

    ppy = panel.periods_per_year
    m = metrics(net[:cut], panel.dates, ppy)
    tr = np.zeros_like(H)
    tr[1:] = np.abs(np.diff(H, axis=0))
    m.update({"turnover_daily": float(0.5 * tr[:cut].sum(axis=1).mean()), "gross_CAGR": metrics(gross[:cut], panel.dates, ppy)["CAGR"],
              "spread_annual_bp": float(spread[:cut].mean() * ppy * 1e4), "borrow_annual_bp": float(borrow[:cut].mean() * ppy * 1e4),
              "impact_annual_bp": float(imp[:cut].mean() * ppy * 1e4), "funding_annual_bp": float(fcost[:cut].mean() * ppy * 1e4),
              "avg_gross_exposure": float(np.abs(H[:cut]).sum(axis=1).mean()), "avg_net_exposure": float(H[:cut].sum(axis=1).mean()),
              "delist_events_held": int(((H != 0) & hit_next)[:cut].sum()), "nan_weights_treated_as_zero": n_nan})
    if part is not None:
        p = part[:cut][np.isfinite(part[:cut])]
        m["participation_p99"] = float(np.percentile(p, 99)) if len(p) else float("nan")
        m["participation_max"] = float(p.max()) if len(p) else float("nan")
        m["trades_over_10pct_adv"] = float((p > 0.10).mean()) if len(p) else float("nan")
    bench = bexc = bret = None
    if benchmark:
        br = _benchmark_returns(panel, fwd, benchmark)
        bench = metrics(br[:cut], panel.dates, ppy)
        bexc = metrics((net - br)[:cut], panel.dates, ppy)
        bret = pd.Series(br[:cut], index=panel.dates[:cut], name="benchmark")
    s = pd.Series(net[:cut], index=panel.dates[:cut])
    yr = s.groupby(s.index.year).apply(lambda g: float((1 + g).prod() - 1))
    spec = {"kind": "weights", "entry_lag": panel.entry_lag, "market": panel.market, "periods_per_year": ppy,
            "spread": "panel" if np.ndim(spread_bp) else f"{spread_bp}bp round trip", "borrow_bp": "panel" if b.ndim else float(b),
            "impact": None if impact is None else {"aum": impact.aum, "y": impact.y, "vol_window": impact.vol_window,
                                                   "adv_window": impact.adv_window, "max_cost_bp": impact.max_cost_bp},
            "funding": bool(funding and panel.funding is not None), "delist_return": delist_return}
    if ledger is not None:
        from .ledger import array_fingerprint
        ledger.record(family, name or "weights", s, {**{k: v for k, v in spec.items()},
                                                    "spread_bp": spread_bp if np.ndim(spread_bp) == 0 else array_fingerprint(spread_bp),
                                                    "weights": array_fingerprint(H), "data": panel.fingerprint()})
    return PortfolioResult(spec=spec, metrics=m, benchmark=bench, excess=bexc, yearly={str(k): v for k, v in yr.items()}, grid=None,
                           holdings=H, net_returns=s, benchmark_returns=bret)


def capacity_curve(panel: Panel, weights: pd.DataFrame, aums, *, y_values=(1.0,), max_cost_bp: float = 100.0,
                   **kw) -> pd.DataFrame:
    """Net Sharpe, CAGR and costs for each AUM (and each impact coefficient `y`). Extra keyword arguments go to
    `backtest_weights` (spread, borrow, funding, ...). Read the AUM where the net CAGR or Sharpe falls to the level you can
    accept; do not read a single capacity number, because Y is uncertain by a factor of a few and costs grow with sqrt(AUM).
    `max_cost_bp` caps the impact per unit traded (default 100 bp); where it binds, cost grows more slowly than the law says,
    which flatters very large AUM, so check `participation_p99` and `trades_over_10pct_adv` next to the cost."""
    rows = []
    for y in y_values:
        for a in aums:
            r = backtest_weights(panel, weights, impact=ImpactModel(aum=float(a), y=float(y), max_cost_bp=max_cost_bp), benchmark=None, **kw)
            m = r.metrics
            rows.append({"y": y, "aum": a, "sharpe": m["Sharpe"], "cagr": m["CAGR"], "gross_cagr": m["gross_CAGR"],
                         "impact_annual_bp": m["impact_annual_bp"], "spread_annual_bp": m["spread_annual_bp"],
                         "participation_p99": m["participation_p99"], "trades_over_10pct_adv": m["trades_over_10pct_adv"]})
    return pd.DataFrame(rows)
