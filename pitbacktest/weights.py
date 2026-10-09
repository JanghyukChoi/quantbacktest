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
  buy_bp, sell_bp  one-way cost per unit of weight bought / sold (bp, on top of the spread): a float, a Series by date (a rate that
              changes over time) or a (date x ticker) frame. A weight going down is a sell, so opening a short pays sell_bp. A missing
              value raises. The same as in `backtest_portfolio`.
  borrow_bp   annual borrow fee in bp (scalar or panel) on the short market value, charged per period.
  impact      square-root law, per unit traded:  Y * sigma * sqrt(|trade| * AUM / ADV)
              sigma the trailing daily volatility, ADV the trailing average dollar volume, both known at the signal date.
              Y is of order 1 in the literature (Torre 1997; Almgren et al. 2005; Toth et al. 2011) but it is not known for a
              given market, so run `capacity_curve` over several values instead of trusting one. Capital is taken to equal
              AUM throughout (no compounding of the base).

You cannot open or increase a position in a security that is not eligible that day: an optimiser that buys names outside
the point-in-time universe is using information it should not have. A position already held may stay after the name leaves the
universe until you reduce it. Long and short exposure are checked separately, so turning a long into a smaller short in a name that is
not eligible is an increase of its short and raises.

Execution (what can actually be traded)
  `panel.can_buy` / `panel.can_sell` (see `pitbacktest.execution`): a trade that increases a position on a day when it cannot be bought, or
  decreases one where it cannot be sold, does not happen and the position stays. They are read on the execution day, `lag` after the signal.
  `capital`, `price`, `lot`, `min_trade_value`: with `capital` given, each day's target is rounded to whole lots at the real prices and trades
  worth less than `min_trade_value` are skipped, so that a small account does not hold 0.3 of a share or trade a few dollars.
  The returned `holdings` are the positions after all of that.
If `panel.shortable` is given, opening or increasing a short in a security that cannot be sold short on the execution day (signal date +
`entry_lag`) raises as well.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core.costs import apply_side_cost, apply_turnover_cost, side_cost_input
from .core.panel import Panel, check_alignment
from .execution import check_freeze_return, exec_masks, freeze_hits, realize
from .portfolio import (PortfolioResult, _benchmark_returns, _check_benchmark, _check_cost, _forward_arrays, _side_config, stop_at_ruin,
                        _side_label, metrics)


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


def backtest_weights(panel: Panel, weights: pd.DataFrame, *, spread_bp=0.0, buy_bp=0.0, sell_bp=0.0, borrow_bp=0.0,
                     impact: ImpactModel | None = None, funding: bool = True, delist_return: float | None = None,
                     benchmark: str | None = "cap", ledger=None, family: str = "default",
                     name: str | None = None, check_universe: bool = True, check_shortable: bool = True,
                     capital: float | None = None, price=None, lot=1.0, min_trade_value: float = 0.0,
                     freeze_days: int | None = None, freeze_return: float = 0.0, cap_gross: bool = False) -> PortfolioResult:
    """Net returns of the weights you supply. See the module docstring for timing and costs.

    `check_universe` raises if a long or a short position is opened or increased in a security that is not eligible that day. It looks at
    net weights, so a name that is long from one signal and short from another (the two cancel) can show an increase when the
    short expires; holdings built from overlapping long and short tranches (`PortfolioResult.holdings`) can trip it, and
    then `check_universe=False` is the honest setting. Weights from an optimiser that nets positions are not affected.
    `capital` (money, same currency as `price` and `ImpactModel.aum`), `price` (date x ticker **real** price level on the day of the row, as for the weights:
    a back-adjusted series has an arbitrary level and gives wrong share counts; the engine sizes row t at `price` of day t + `entry_lag`, the day it is traded), `lot` (shares per lot: a number, a Series by ticker or a date x ticker frame) and
    `min_trade_value` (money; a number or a Series by ticker) switch on whole-lot sizes; without `capital` they must be left alone.
    `freeze_days`, `freeze_return`: a long position in a security whose suspension (a price but no volume) reaches `freeze_days` days is marked down once by
    `freeze_return` (between -1 and 0): the loss is taken by the close of the `freeze_days`-th suspended day (booked on the signal row `entry_lag` days earlier); shorts are not credited. A scenario, not a measurement (see `execution.freeze_episodes`).
    `cap_gross`: positions that cannot be traded tie up capital; the free names are scaled down so that the gross exposure stays at the target's (see `execution.realize`).
    `check_shortable` (only with `panel.shortable`) raises if a short is opened or increased where the security cannot be sold short.

    Costs are charged on the **net** trade per security. `backtest_portfolio` charges its long and short legs as separate sleeves,
    so feeding it its own overlapping-tranche holdings can come out slightly cheaper here (a name long in one tranche and short in
    another is netted); without such overlap the two agree exactly."""
    _check_benchmark(benchmark)
    _check_cost(spread_bp, "spread_bp")
    _check_cost(borrow_bp, "borrow_bp")
    fz = freeze_hits(panel, freeze_days)
    freeze_return = check_freeze_return(freeze_return)
    bb = side_cost_input(buy_bp, panel.dates, panel.tickers, "buy_bp")
    sb = side_cost_input(sell_bp, panel.dates, panel.tickers, "sell_bp")
    check_alignment(panel, weights, "weights")
    W = weights.reindex(index=panel.dates, columns=panel.tickers)
    raw = W.to_numpy(float)
    if np.isinf(raw).any():
        raise ValueError("weights contain inf")
    n_nan = int(np.isnan(raw).sum())
    H = np.nan_to_num(raw, nan=0.0)
    prev = np.vstack([np.zeros((1, H.shape[1])), H[:-1]])
    more_long = np.maximum(H, 0.0) > np.maximum(prev, 0.0) + 1e-12
    more_short = np.maximum(-H, 0.0) > np.maximum(-prev, 0.0) + 1e-12
    bad = (more_long | more_short) & ~panel.eligible.to_numpy(bool)               # opening or increasing outside the universe
    if check_universe and bad.any():
        i, j = np.argwhere(bad)[0]
        raise ValueError(f"{int(bad.sum())} positions opened or increased on securities that are not eligible that day "
                         f"(first: {panel.tickers[j]} on {panel.dates[i].date()}); the universe is point in time")
    if check_shortable and panel.shortable is not None:
        nos = more_short & ~panel.shortable.shift(-panel.entry_lag, fill_value=True).to_numpy(bool)       # read on the execution day, like can_sell
        if nos.any():
            i, j = np.argwhere(nos)[0]
            raise ValueError(f"{int(nos.sum())} short positions opened or increased on securities that cannot be sold short on the execution day "
                             f"(first: {panel.tickers[j]} on {panel.dates[i].date()}); see Panel.shortable")
    bo, so = exec_masks(panel)
    ex = None
    if isinstance(min_trade_value, pd.Series):
        mtv = min_trade_value.reindex(panel.tickers).to_numpy(float)
        if np.isnan(mtv).any():
            raise ValueError("min_trade_value must be known for every security when it is given per security")
    else:
        mtv = float(min_trade_value)
    if capital is None and (price is not None or np.any(np.asarray(mtv) > 0) or (np.ndim(lot) or float(lot) != 1.0)):
        raise ValueError("price, lot and min_trade_value need capital: whole-lot sizes depend on how much money there is")
    if capital is not None:
        if not (np.isfinite(capital) and capital > 0):
            raise ValueError(f"capital must be a positive number, got {capital!r}")
        if price is None:
            raise ValueError("capital needs price, the real price level in the same currency (not a back-adjusted series)")
        if not (np.all(np.isfinite(mtv)) and np.all(np.asarray(mtv) >= 0)):
            raise ValueError(f"min_trade_value must be at least 0, got {min_trade_value!r}")
        if not isinstance(price, pd.DataFrame):
            raise ValueError("price must be a (date x ticker) frame")
        # Row t of the weights is traded at close(t + lag), so it is sized at that day's price. The last `lag` rows have no such day: unpriced, kept.
        px = price.reindex(index=panel.dates, columns=panel.tickers).shift(-panel.entry_lag).to_numpy(float)
        if isinstance(lot, pd.DataFrame):
            lot_ = lot.reindex(index=panel.dates, columns=panel.tickers).to_numpy(float)
        elif isinstance(lot, pd.Series):
            lot_ = lot.reindex(panel.tickers).to_numpy(float)
        else:
            lot_ = float(lot)
        if np.isnan(lot_).any() or not (np.asarray(lot_) > 0).all():
            raise ValueError("lot must be positive and known for every security")
    H_in = H                                                            # what was asked for; `H` becomes what could be held
    if bo is not None or capital is not None:
        H, ex = realize(H, bo, so, capital=capital, price=px if capital is not None else None,
                        lot=lot_ if capital is not None else 1.0, min_trade_value=mtv, cap_gross=cap_gross)
    fwd, fwdf, hit_next = _forward_arrays(panel, funding, delist_return)
    cut = len(panel.dates) - (panel.entry_lag + 1)
    gross = (H * fwd).sum(axis=1)
    mark = (np.maximum(H, 0.0) * fz).sum(axis=1) * freeze_return if fz is not None else np.zeros(len(gross))
    gross = gross + mark
    spread = apply_turnover_cost(H, spread_bp)                          # the first row, a build from cash, is not charged
    side = apply_side_cost(H, bb, sb)
    short = np.maximum(-H, 0.0)
    b = np.asarray(borrow_bp, float)
    borrow = (short * (b / 1e4 / panel.periods_per_year)).sum(axis=1) if b.ndim == 0 else \
        (short * (np.nan_to_num(b, nan=0.0) / 1e4 / panel.periods_per_year)).sum(axis=1)
    fcost = (H * fwdf).sum(axis=1)
    imp, part = (np.zeros(len(H)), None)
    if impact is not None:
        imp, part = _impact_cost(panel, H, impact)
    net = gross - spread - side - borrow - imp - fcost

    ppy = panel.periods_per_year
    net, ruin = stop_at_ruin(net, cut)
    if ruin is not None:
        warnings.warn(f"the account lost 100% or more on {panel.dates[ruin].date()}: from that day the return is -100% and then 0 (a leveraged account would have been liquidated)", stacklevel=2)
    m = metrics(net[:cut], panel.dates, ppy)
    m["ruined"] = ruin is not None
    if float(np.abs(H[:cut]).sum()) == 0.0:
        warnings.warn("no position was held on any day: the weights are all zero or NaN, or `capital` is too small to buy one lot of anything; "
                      "the zero return is not a result", stacklevel=2)
    m["ruin_date"] = None if ruin is None else str(panel.dates[ruin].date())
    tr = np.zeros_like(H)
    tr[1:] = np.abs(np.diff(H, axis=0))
    m.update({"turnover_daily": float(0.5 * tr[:cut].sum(axis=1).mean()), "gross_CAGR": metrics(stop_at_ruin(gross, cut)[0][:cut], panel.dates, ppy)["CAGR"],
              "spread_annual_bp": float(spread[:cut].mean() * ppy * 1e4), "borrow_annual_bp": float(borrow[:cut].mean() * ppy * 1e4),
              "side_cost_annual_bp": float(side[:cut].mean() * ppy * 1e4),
              "impact_annual_bp": float(imp[:cut].mean() * ppy * 1e4), "funding_annual_bp": float(fcost[:cut].mean() * ppy * 1e4),
              "avg_gross_exposure": float(np.abs(H[:cut]).sum(axis=1).mean()), "avg_net_exposure": float(H[:cut].sum(axis=1).mean()),
              "delist_events_held": int(((H != 0) & hit_next)[:cut].sum()), "nan_weights_treated_as_zero": n_nan})
    if fz is not None:
        m["freeze_markdown_annual_bp"] = float(mark[:cut].mean() * ppy * 1e4)
        m["freeze_markdown_events"] = int(((H > 0) & fz)[:cut].sum())
    if ex is not None:
        m["blocked_trades"] = int(ex["blocked_trades"])
        m["blocked_turnover_share"] = float(ex["blocked_turnover"] / ex["asked_turnover"]) if ex["asked_turnover"] > 0 else 0.0
        m["mean_stuck_weight"] = float(ex["mean_stuck_weight"])
        m["longest_freeze_days"] = int(ex["longest_freeze_days"])
        if cap_gross:
            m["mean_free_scale"] = float(ex["mean_free_scale"])
            m["cap_infeasible_days"] = int(ex["infeasible_days"])
        if capital is not None:
            m.update({"min_trade_skipped": int(ex["min_trade_skipped"]), "min_trade_skipped_share": float(ex["min_trade_skipped_share"]),
                      "mean_abs_rounding_gap": float(ex["mean_abs_rounding_gap"])})
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
            "spread": "panel" if np.ndim(spread_bp) else f"{spread_bp}bp round trip",
            "buy_bp": _side_label(bb), "sell_bp": _side_label(sb), "borrow_bp": "panel" if b.ndim else float(b),
            "impact": None if impact is None else {"aum": impact.aum, "y": impact.y, "vol_window": impact.vol_window,
                                                   "adv_window": impact.adv_window, "max_cost_bp": impact.max_cost_bp},
            "funding": bool(funding and panel.funding is not None), "delist_return": delist_return,
            "freeze": None if freeze_days is None else {"days": int(freeze_days), "return": freeze_return},
            "execution": None if (bo is None and capital is None) else {"blocked": bo is not None, "capital": capital,
                                                                                    "min_trade_value": float(mtv) if np.ndim(mtv) == 0 else "by security"}}
    if ledger is not None:
        from .ledger import array_fingerprint
        ledger.record(family, name or "weights", s, {**{k: v for k, v in spec.items() if k not in ("buy_bp", "sell_bp", "execution", "freeze")}, **_side_config(bb, sb), **_exec_config(capital, price, lot, mtv),
                                                    **({"freeze_days": int(freeze_days), "freeze_return": freeze_return} if freeze_days is not None else {}), **({"cap_gross": True} if cap_gross else {}),
                                                    "spread_bp": spread_bp if np.ndim(spread_bp) == 0 else array_fingerprint(spread_bp),
                                                    "weights": array_fingerprint(H_in), "data": panel.fingerprint()})
    return PortfolioResult(spec=spec, metrics=m, benchmark=bench, excess=bexc, yearly={str(k): v for k, v in yr.items()}, grid=None,
                           holdings=H, net_returns=s, benchmark_returns=bret)


def _exec_config(capital, price, lot, mtv) -> dict:
    """Ledger configuration of the execution settings; empty when none is used, so older runs keep their fingerprint."""
    if capital is None:
        return {}
    from .ledger import array_fingerprint
    return {"capital": float(capital), "min_trade_value": float(mtv) if np.ndim(mtv) == 0 else array_fingerprint(np.asarray(mtv)),
            "price": array_fingerprint(price.to_numpy(float)),
            "lot": float(lot) if np.ndim(lot) == 0 else array_fingerprint(np.asarray(lot, dtype=float))}


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
