"""B. Portfolio alpha backtest: from a signal to a result.

Timing convention (stated because this is where the bug was)
    holdings[d] = the average of the tranches from the signals of days t = d-hold+1 ... d. A signal is known at close(d),
    entered at close(d+entry_lag), and earns close(d+entry_lag) -> close(d+entry_lag+1).
    An entry that is one day late (off by one) wipes out most of the profit of a 1-day mean-reversion strategy.
    assert_timing() catches it.

Costs
    Proportional to turnover. **Use a per-security measured spread panel.**
    A flat assumption can inflate the result of a high-turnover strategy a lot.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core.costs import apply_side_cost, apply_turnover_cost, side_cost_input, turnover
from .core.panel import Panel
from .execution import check_freeze_return, exec_masks, freeze_hits, realize

ANN = 252


@dataclass
class PortfolioResult:
    spec: dict
    metrics: dict
    benchmark: dict | None
    excess: dict | None
    yearly: dict
    grid: dict | None
    holdings: np.ndarray | None = None   # (date x ticker) net target weights, per unit of capital in each leg
    net_returns: pd.Series | None = None  # daily net return series (after costs and funding), for DSR / PBO
    benchmark_returns: pd.Series | None = None  # return of the benchmark on the same dates (no costs), when one was asked for

    def alpha_beta(self, factors=None, **kw) -> dict:
        """Regress the net returns on `factors` (default: the benchmark, a market proxy). See `analytics.alpha_beta`."""
        from .analytics import alpha_beta
        if factors is None:
            if self.benchmark_returns is None:
                raise ValueError("no benchmark was computed; pass factors=... or run with benchmark='cap' or 'equal'")
            factors = self.benchmark_returns.rename("benchmark")
        kw.setdefault("periods_per_year", self.spec.get("periods_per_year", ANN))
        return alpha_beta(self.net_returns, factors, **kw)

    def sharpe_ci(self, **kw) -> dict:
        """Bootstrap interval for the Sharpe ratio of `net_returns`. Keyword arguments go to `analytics.sharpe_ci`."""
        from .analytics import sharpe_ci
        kw.setdefault("periods_per_year", self.spec.get("periods_per_year", ANN))
        return sharpe_ci(self.net_returns, **kw)


def _tranche(weights: np.ndarray, hold: int) -> np.ndarray:
    """Overlapping tranches: the position on day d is the average of the signals of days t = d-hold+1 ... d (each day 1/hold of capital enters)."""
    T, N = weights.shape
    out = np.zeros((T, N))
    run = np.zeros(N)
    cnt = 0
    for d in range(T):
        run += weights[d]
        cnt += 1
        if d - hold >= 0:
            run -= weights[d - hold]
            cnt -= 1
        if cnt > 0:
            out[d] = run / cnt
    # Adding a tranche and subtracting it later leaves rounding residue (about 1e-18) where the exact weight is 0. A residue is not a position,
    # but `held` and `avg_positions` count anything above 0, so snap it to 0. Real weights are many orders of magnitude larger than the tolerance.
    tol = 1e-12 * float(np.abs(weights).max()) if weights.size else 0.0
    out[np.abs(out) < tol] = 0.0
    return out


def _normalize(mask_or_w: np.ndarray) -> np.ndarray:
    w = np.asarray(mask_or_w, dtype=np.float64)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = np.nan
    return np.where(np.isfinite(s), w / s, 0.0)


def metrics(net: np.ndarray, dates: pd.DatetimeIndex, ann: int = ANN) -> dict:
    """Annualised figures of a net return series: CAGR, MDD, Sharpe, Sortino (downside deviation, target 0), Calmar, vol, years, the share of
    positive days and the drawdown dates. With fewer than `ann // 2` observations it warns and returns NaN for CAGR, MDD and Sharpe only."""
    s = pd.Series(net, index=dates[: len(net)]).dropna()
    if len(s) < ann // 2:
        # Annualised figures from under half a year are not reported. Say so: a NaN with no explanation looks like a bug, and
        # intraday studies are often a few months long.
        warnings.warn(f"only {len(s):,} observations = {len(s) / ann:.2f} years (< 0.5): CAGR, MDD and Sharpe are NaN", stacklevel=2)
        return {"CAGR": np.nan, "MDD": np.nan, "Sharpe": np.nan}
    eq = (1 + s).cumprod()
    yrs = len(s) / ann
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:]
    recov = rec[rec >= eq.loc[peak]].index
    return {"CAGR": float(cagr), "MDD": float(mdd),
            "Sharpe": float(s.mean() / (s.std() + 1e-12) * np.sqrt(ann)),
            "Sortino": float(s.mean() / (np.sqrt(np.mean(np.minimum(s.to_numpy(), 0.0) ** 2)) + 1e-12) * np.sqrt(ann)),  # downside deviation, target 0
            "Calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
            "vol": float(s.std() * np.sqrt(ann)),
            "years": float(yrs), "pos_days": float((s > 0).mean()),
            "mdd_peak": str(peak.date()), "mdd_trough": str(trough.date()),
            "mdd_recovered": str(recov[0].date()) if len(recov) else "not recovered"}


def _forward_arrays(panel: Panel, funding: bool, delist_return: float | None):
    """The three (date x ticker) arrays every simulation needs, indexed by the **signal date** d:
    fwd[d]       return from close(d + lag) to close(d + lag + 1); a missing price earns 0 (cash), and the day after a
                 delisting flagged in `panel.delist_after` earns `delist_return` when it is given
    fwdf[d]      funding rate over the same day (0 unless `funding` and the panel has it); positive means longs pay
    hit_next[d]  True where that day is the delisting day
    The last lag + 1 rows of fwd and fwdf are 0: their holding period is not in the sample."""
    ret = np.nan_to_num(panel.ret1().values.astype(np.float64), nan=0.0)
    nxt = np.zeros(ret.shape, dtype=bool)
    if panel.delist_after is not None:
        da = panel.delist_after.reindex(index=panel.dates, columns=panel.tickers).fillna(False).values.astype(bool)
        nxt[1:] = da[:-1]                                   # the day after the last real bar
        if delist_return is not None:
            ret = np.where(nxt, float(delist_return), ret)
    k = panel.entry_lag + 1
    fwd = np.roll(ret, -k, axis=0)
    fwd[-k:] = 0.0
    fwdf = np.zeros(ret.shape)
    if funding and panel.funding is not None:
        fd = np.nan_to_num(panel.funding.reindex(index=panel.dates, columns=panel.tickers).values.astype(np.float64), nan=0.0)
        fwdf = np.roll(fd, -k, axis=0)
        fwdf[-k:] = 0.0
    return fwd, fwdf, np.roll(nxt, -k, axis=0)


def _benchmark_returns(panel: Panel, fwd: np.ndarray, kind: str) -> np.ndarray:
    """Daily return (indexed by signal date, like `fwd`) of the benchmark: market-cap weighted when `kind` is "cap" and the
    panel has market caps, otherwise equal weighted, over the securities eligible that day. No costs."""
    ev = panel.eligible.values
    if kind == "cap" and panel.mkt_cap is not None:
        w = np.where(ev, np.nan_to_num(panel.mkt_cap.values, nan=0.0), 0.0)
    else:
        w = ev.astype(float)
    return (_normalize(w) * fwd).sum(axis=1)


def _check_cost(value, name: str) -> None:
    """A cost must be a finite, non-negative number (or an array of them). A negative cost would pay you to trade."""
    a = np.asarray(value, dtype=float)
    if a.ndim == 0 and not np.isfinite(a):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if a.size and (np.nanmin(a, initial=np.inf) < 0 or np.isinf(a).any()):
        raise ValueError(f"{name} must be finite and not negative (a negative cost would pay you for trading), got {np.nanmin(a):g}")


def _check_benchmark(benchmark) -> None:
    if benchmark not in ("cap", "equal", None):
        raise ValueError(f"benchmark must be 'cap', 'equal' or None, not {benchmark!r}")


def _check_portfolio_args(factor, long_q, short_q, hold, weighting, spread_bp, benchmark) -> None:
    if weighting not in ("equal", "signal", "rank"):
        raise ValueError(f"weighting must be 'equal', 'signal' or 'rank', not {weighting!r}")
    _check_benchmark(benchmark)
    if not (0 < long_q <= 1):
        raise ValueError(f"long_q must be in (0, 1], got {long_q!r}")
    if short_q is not None and not (0 < short_q <= 1):
        raise ValueError(f"short_q must be in (0, 1], or None for a long-only portfolio, got {short_q!r}")
    if isinstance(hold, bool) or not isinstance(hold, (int, np.integer)) or hold < 1:
        raise ValueError(f"hold must be a whole number of periods, at least 1, got {hold!r}")
    _check_cost(spread_bp, "spread_bp")
    if isinstance(factor, pd.DataFrame) and len(factor.columns) and (factor.dtypes == bool).all():
        raise ValueError("a boolean factor is not a ranking: pass a numeric score, or use backtest_event for a yes/no signal")


def _side_label(v) -> float | str:
    return float(v) if not isinstance(v, np.ndarray) else ("schedule" if v.shape[1] == 1 else "panel")


def _side_config(buy_bp, sell_bp) -> dict:
    """Ledger configuration of the side costs: left out when both are zero so that runs recorded without them keep their fingerprint."""
    from .ledger import array_fingerprint
    out = {}
    for k, v in (("buy_bp", buy_bp), ("sell_bp", sell_bp)):
        if v is None or (np.ndim(v) == 0 and float(v) == 0.0):
            continue
        out[k] = float(v) if np.ndim(v) == 0 else array_fingerprint(np.asarray(v, dtype=float))
    return out


def backtest_portfolio(panel: Panel, factor: pd.DataFrame, *,
                       long_q: float = 0.10, short_q: float | None = 0.10,
                       hold: int = 5, weighting: str = "equal",
                       spread_bp=20.0, buy_bp=0.0, sell_bp=0.0, benchmark: str | None = "cap",
                       grid: bool = True, funding: bool = True,
                       delist_return: float | None = None, freeze_days: int | None = None, freeze_return: float = 0.0,
                       cap_gross: bool = False, ledger=None, family: str = "default", name: str | None = None) -> PortfolioResult:
    """Factor -> portfolio result.

    factor      numeric score (higher = long). A boolean frame is refused: use backtest_event for yes/no signals.
    long_q      long quantile (top q), in (0, 1]. short_q is the short quantile; None means long-only (0 is refused).
    weighting   equal | signal (proportional to signal strength) | rank
    spread_bp   **Two units.** A scalar is a round-trip spread: each unit of weight traded pays spread_bp/2
                (to use a one-way cost of c bp, pass 2*c). A (date x ticker) panel is a one-way cost in bp, multiplied
                by the weight traded as it is. A scalar 20 equals a panel of 10 (pinned by tests/test_reconcile.py).
    buy_bp, sell_bp  One-way cost per unit of weight **bought** / **sold**, in bp, charged on top of the spread; a float, a Series
                indexed by date (a rate that changes over time, for example a transaction tax) or a (date x ticker) frame. A weight going
                down is a sell, so opening a short pays `sell_bp` and covering it pays `buy_bp` (a sales tax is charged when a short is
                opened). The rate is read on the signal date. A missing value raises: an unknown rate is not zero.
    benchmark   cap (market-cap weighted) | equal (equal weighted) | None
    funding     if panel.funding exists, longs pay and shorts receive it (futures). False ignores it
    ledger      pitbacktest.ledger.Ledger. If given, this run is recorded in it, so the number of trials reaches the deflated Sharpe.
    freeze_days, freeze_return  A long position in a security whose suspension (a price but no volume) reaches `freeze_days` days is marked down once
                by `freeze_return` (between -1 and 0), on the day it reaches that length. Short positions are not credited: the gain cannot be taken while the
                security is frozen. A scenario, not a measurement: `execution.freeze_episodes` measures how suspensions ended in your data. None: no markdown.
    cap_gross   True: a position that cannot be traded (see `Panel.can_buy`) ties up capital, so the free names of that leg are scaled down to keep the leg's gross
                exposure at its target instead of piling new positions on top (see `execution.realize`). Only matters where something is blocked. Off by default.
    family      name that groups runs of one research question in the ledger. name labels this trial (the same label with other settings is another trial).
    delist_return  assumed return on the day **after** the last real bar of a security flagged in panel.delist_after.
                None means 0 (closed at the last price, which can be optimistic). For example -0.5 shows the sensitivity.
    """
    _check_portfolio_args(factor, long_q, short_q, hold, weighting, spread_bp, benchmark)
    fz = freeze_hits(panel, freeze_days)
    freeze_return = check_freeze_return(freeze_return)
    bb = side_cost_input(buy_bp, panel.dates, panel.tickers, "buy_bp")
    sb = side_cost_input(sell_bp, panel.dates, panel.tickers, "sell_bp")
    f = factor.reindex(index=panel.dates, columns=panel.tickers)
    el = panel.eligible
    rk = f.where(el).rank(axis=1, pct=True, na_option="keep")
    # The short leg is chosen among securities that can be sold short on the day the short is opened, the execution day (signal date + lag): a ban that
    # starts on that day stops a short that would be opened then, though the signal was given the day before.
    sh_exec = None if panel.shortable is None else panel.shortable.shift(-panel.entry_lag, fill_value=True)
    el_s = el if sh_exec is None else el & sh_exec
    rk_s = rk if sh_exec is None else f.where(el_s).rank(axis=1, pct=True, na_option="keep")
    fwd, fwdf, hit_next = _forward_arrays(panel, funding, delist_return)
    ev = el.values

    def leg(q: float, top: bool) -> np.ndarray:
        rk_, el_ = (rk, el) if top else (rk_s, el_s)
        m = ((rk_ >= 1 - q) if top else (rk_ <= q)) & el_
        mv = m.values
        if weighting == "equal":
            w = mv.astype(float)
        elif weighting == "signal":
            w = np.where(mv, np.abs(np.nan_to_num(f.values, nan=0.0)), 0.0)
        else:  # rank
            r = np.nan_to_num(rk_.values, nan=0.0)
            w = np.where(mv, (r - (1 - q)) if top else (q - r), 0.0)
        return _normalize(w)

    bo, so = exec_masks(panel)                                           # can the trade be done on the execution day? (None: always)
    ex = {"asked_turnover": 0.0, "blocked_turnover": 0.0, "blocked_trades": 0, "mean_stuck_weight": 0.0, "longest_freeze_days": 0, "mean_free_scale": 0.0, "infeasible_days": 0, "legs": 0}

    def execute(h: np.ndarray, sign: float, record: bool) -> np.ndarray:
        """The leg's positions after blocked trades. A short leg is a negative position, so covering it is a buy."""
        if bo is None:
            return h
        pos, st = realize(sign * h, bo, so, cap_gross=cap_gross)
        if record:
            for k in st.keys() & ex.keys():
                ex[k] = max(ex[k], st[k]) if k == "longest_freeze_days" else ex[k] + st[k]
            ex["legs"] += 1
        return sign * pos

    hl = execute(_tranche(leg(long_q, True), hold), 1.0, True)
    gross = (hl * fwd).sum(axis=1)
    mark = (hl * fz).sum(axis=1) * freeze_return if fz is not None else np.zeros(len(gross))          # the one-off markdown of frozen longs
    gross = gross + mark
    cost = apply_turnover_cost(hl, spread_bp)
    side = apply_side_cost(hl, bb, sb)
    turn = turnover(hl)
    fcost = (hl * fwdf).sum(axis=1)                     # longs pay a positive funding rate
    held = hl > 0
    hs = None
    if short_q:
        hs = execute(_tranche(leg(short_q, False), hold), -1.0, True)
        gross = gross - (hs * fwd).sum(axis=1)
        cost = cost + apply_turnover_cost(hs, spread_bp)
        side = side + apply_side_cost(-hs, bb, sb)                  # a short is a negative weight: opening it is a sell
        turn = turn + turnover(hs)
        fcost = fcost - (hs * fwdf).sum(axis=1)         # shorts receive it
        held = held | (hs > 0)
    cost = cost + side                                  # `cost_annual_bp` is every trading cost: spread and side costs
    net = gross - cost - fcost

    cut = len(panel.dates) - (panel.entry_lag + 1)
    m = metrics(net[:cut], panel.dates, panel.periods_per_year)
    m["turnover_daily"] = float(turn[:cut].mean())
    m["cost_annual_bp"] = float(cost[:cut].mean() * panel.periods_per_year * 1e4)
    m["gross_CAGR"] = metrics(gross[:cut], panel.dates, panel.periods_per_year)["CAGR"]
    if fz is not None:
        m["freeze_markdown_annual_bp"] = float(mark[:cut].mean() * panel.periods_per_year * 1e4)        # negative: the return given up
        m["freeze_markdown_events"] = int(((hl > 0) & fz)[:cut].sum())
    m["side_cost_annual_bp"] = float(side[:cut].mean() * panel.periods_per_year * 1e4)   # the part of cost_annual_bp from buy_bp and sell_bp
    if bo is not None:
        m["blocked_trades"] = int(ex["blocked_trades"])                  # name-days on which a wanted trade could not be done
        m["blocked_turnover_share"] = float(ex["blocked_turnover"] / ex["asked_turnover"]) if ex["asked_turnover"] > 0 else 0.0
        m["mean_stuck_weight"] = float(ex["mean_stuck_weight"])             # both legs, in units of capital
        m["longest_freeze_days"] = int(ex["longest_freeze_days"])
        if cap_gross:
            m["mean_free_scale"] = float(ex["mean_free_scale"] / max(ex["legs"], 1))                  # 1 means nothing had to be scaled down
            m["cap_infeasible_days"] = int(ex["infeasible_days"])               # leg-days on which the stuck positions alone exceeded the target
    if panel.shortable is not None and short_q:
        m["short_leg_empty_days"] = int((el.to_numpy(bool).any(axis=1) & ~el_s.to_numpy(bool).any(axis=1))[:cut].sum())   # eligible names exist, none can be sold short
    m["funding_annual_bp"] = float(fcost[:cut].mean() * panel.periods_per_year * 1e4)   # positive = a cost
    m["delist_events_held"] = int((held & hit_next)[:cut].sum())
    m["avg_positions"] = float((hl > 0).sum(axis=1)[(hl > 0).sum(axis=1) > 0].mean())

    bench = bexc = None
    bret = None
    if benchmark:
        br = _benchmark_returns(panel, fwd, benchmark)
        bench = metrics(br[:cut], panel.dates, panel.periods_per_year)
        bexc = metrics((net - br)[:cut], panel.dates, panel.periods_per_year)
        bret = pd.Series(br[:cut], index=panel.dates[:cut], name="benchmark")

    s = pd.Series(net[:cut], index=panel.dates[:cut])
    yr = s.groupby(s.index.year).apply(lambda g: float((1 + g).prod() - 1))

    g = None
    if grid:
        g = {}
        for h in (1, 2, 5, 10, 21):
            hl2 = execute(_tranche(leg(long_q, True), h), 1.0, False)
            hs2 = execute(_tranche(leg(short_q, False), h), -1.0, False) if short_q else None
            for c in (0, 2, 5, 10, 20):
                gr = (hl2 * fwd).sum(axis=1) + ((hl2 * fz).sum(axis=1) * freeze_return if fz is not None else 0.0)
                co = apply_turnover_cost(hl2, c) + apply_side_cost(hl2, bb, sb) + (hl2 * fwdf).sum(axis=1)
                if short_q:
                    gr = gr - (hs2 * fwd).sum(axis=1)
                    co = co + apply_turnover_cost(hs2, c) + apply_side_cost(-hs2, bb, sb) - (hs2 * fwdf).sum(axis=1)
                mm = metrics((gr - co)[:cut], panel.dates, panel.periods_per_year)
                g[f"h{h}_c{c}"] = {"CAGR": mm["CAGR"], "Sharpe": mm["Sharpe"], "MDD": mm["MDD"]}

    if ledger is not None:
        from .ledger import array_fingerprint
        ledger.record(family, name or "unnamed", s, {
            "long_q": long_q, "short_q": short_q, "hold": hold, "weighting": weighting,
            "spread_bp": spread_bp if np.isscalar(spread_bp) else array_fingerprint(spread_bp),
            "funding": bool(funding and panel.funding is not None), "delist_return": delist_return,
            **_side_config(bb, sb), **({"freeze_days": int(freeze_days), "freeze_return": freeze_return} if freeze_days is not None else {}), **({"cap_gross": True} if cap_gross else {}),
            "factor": array_fingerprint(f.to_numpy()), "data": panel.fingerprint()})
    return PortfolioResult(
        spec={"long_q": long_q, "short_q": short_q, "hold": hold, "weighting": weighting,
              "entry_lag": panel.entry_lag, "market": panel.market, "periods_per_year": panel.periods_per_year,
              "spread": "panel" if not np.isscalar(spread_bp) else f"{spread_bp}bp flat",
              "buy_bp": _side_label(bb), "sell_bp": _side_label(sb),
              "freeze": None if freeze_days is None else {"days": int(freeze_days), "return": freeze_return},
              "funding": bool(funding and panel.funding is not None), "delist_return": delist_return},
        metrics=m, benchmark=bench, excess=bexc,
        yearly={str(k): v for k, v in yr.items()}, grid=g,
        holdings=(hl - hs) if hs is not None else hl, net_returns=s, benchmark_returns=bret)


def assert_timing(panel: Panel) -> dict:
    """Timing self-check: feed a perfect-foresight factor and see whether it earns money.

    Using the future return as the factor must give a clearly positive CAGR. If not, the timing is misaligned.
    (A regression test for entry-lag bugs.)
    """
    lag = panel.entry_lag
    oracle = panel.close.shift(-(lag + 1)) / panel.close.shift(-lag) - 1.0
    r = backtest_portfolio(panel, oracle, long_q=0.10, short_q=0.10, hold=1,
                           spread_bp=0.0, benchmark=None, grid=False)
    ok = np.isfinite(r.metrics["CAGR"]) and r.metrics["CAGR"] > 0.5
    return {"oracle_CAGR": r.metrics["CAGR"], "oracle_Sharpe": r.metrics["Sharpe"],
            "pass": bool(ok),
            "note": "A perfect-foresight factor must give a large positive number. If it fails, entry_lag or the fwd alignment is wrong."}
