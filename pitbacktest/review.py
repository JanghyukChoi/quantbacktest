"""One call that runs a strategy through the tests an institution would ask for, and a page that shows them.

    rv = q.review_portfolio(panel, factor, long_q=0.2, short_q=0.2, hold=5, spread_bp=10)          # a factor held as a long-short portfolio
    rv.report("review.html")

    rv = q.review_event(panel, signal, horizons=(1, 5, 20), cost_bp=20)                              # a yes/no signal, judged by its win rate
    rv.report("review.html")

Both put the same questions in the same order, so two strategies of different kinds can be read side by side:

    1. Is it luck?              t-tests, the Sharpe interval, and above all a permutation: the whole thing rerun with the information destroyed (each security given another's
                                factor history, or the table of fires moved in time) and how often noise does as well
    2. Against the market       excess growth, how often it beat the benchmark, up and down capture, beta
    3. What explains it         size, momentum, reversal, volatility, liquidity and market exposure (or factors you supply), and what is left
    4. Does it hold over time   year by year, the two halves, and a walk-forward of the choice of setting
    5. Is the setting a spike   the neighbours of the best setting, and the cost needed to kill it
    6. Capacity                 the cost of the money that has to be traded (portfolio), the cost at which the edge ends (event)

Only the performance section differs: a portfolio is a curve of returns (CAGR, Sharpe, drawdown), an event signal is a list of independent trades (win rate against the win rate of a
random pick on the same days, mean and median, payoff). An event signal can also be held as a portfolio (`event_weights`) to put it on the same footing as a factor.

The functions are slow on purpose where a permutation is involved: `n_permutations` backtests. Nothing here recommends a strategy."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import robustness as rb
from .core.notes import muted, warn
from .export import jsonable

_BACKTEST_KEYS = {"long_q", "short_q", "hold", "weighting", "spread_bp", "buy_bp", "sell_bp", "benchmark", "funding", "delist_return", "freeze_days", "freeze_return",
                  "cap_gross", "impact"}


def event_weights(panel, signal: pd.DataFrame, hold: int) -> pd.DataFrame:
    """An event signal held as a portfolio: each date's fires share 1/`hold` of the capital equally, each lot is held `hold` bars (the convention of `backtest_portfolio`'s
    overlapping tranches), and the capital of dates with no fire stays in cash. A row is the weights set on the signal date, to be run with `backtest_weights`."""
    from .core.panel import check_alignment
    if isinstance(hold, bool) or not isinstance(hold, (int, np.integer)) or hold < 1:
        raise ValueError(f"hold must be a whole number of at least 1, got {hold!r}")
    check_alignment(panel, signal, "signal", sparse=True)
    fire = signal.reindex(index=panel.dates, columns=panel.tickers).fillna(False).astype(bool) & panel.eligible
    k = fire.sum(axis=1).replace(0, np.nan)
    lot = fire.astype(float).div(k, axis=0).fillna(0.0) / hold
    return lot.rolling(hold, min_periods=1).sum()


def _ordered(values) -> list:
    """The values of one parameter in their natural order: numbers ascending (neighbours in a table must be neighbours in value); anything else as given."""
    vals = list(values)
    try:
        return sorted(vals, key=float)
    except (TypeError, ValueError):
        return vals


def _grid_configs(grid: dict) -> list[dict]:
    grid = {k: _ordered(v) for k, v in grid.items()}
    keys = list(grid)
    out = [{}]
    for k in keys:
        out = [{**c, k: v} for c in out for v in grid[k]]
    return out


def _tbl(df: pd.DataFrame | None):
    return None if df is None else jsonable(df)


@dataclass
class PortfolioReview:
    result: object
    mean: dict
    subperiods: dict
    market: dict | None = None
    decomposition: dict | None = None
    style_returns: pd.DataFrame | None = None
    permutation: dict | None = None
    sharpe_ci: dict | None = None
    deflated: dict | None = None
    pbo: dict | None = None                         # probability of backtest overfitting over the settings of the grid (CSCV)
    grid_table: pd.DataFrame | None = None          # Sharpe per setting (rows: first parameter, columns: second), when `grid` was given
    plateau: dict | None = None
    walk_forward: dict | None = None
    cost_table: pd.DataFrame | None = None          # Sharpe and CAGR at several spreads
    delist_table: pd.DataFrame | None = None        # Sharpe and CAGR if a delisting loses 0, 30, 50 or 100 percent (only when the panel flags delistings)
    attribution: dict | None = None                 # where the return came from (see `robustness.attribution`)
    risk: dict | None = None                        # VaR, CVaR, tails, drawdown length (`robustness.risk_profile`)
    drawdown_dist: dict | None = None               # the maximum drawdown as one draw (`robustness.drawdown_distribution`)
    regimes: dict | None = None                     # bull, bear, calm, stormy (`robustness.regimes`)
    holdings: dict | None = None                    # positions, concentration, trading (`robustness.holdings_summary`)
    split: dict | None = None                       # before and after a frozen date (`robustness.in_out_of_sample`)
    brinson: dict | None = None                     # allocation against selection by group (`robustness.brinson`)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """The whole review as a JSON-ready dictionary: the run's result, every test and its inputs (the permutation's null is left out: it is long)."""
        wf = None if self.walk_forward is None else {k: v for k, v in self.walk_forward.items() if k not in ("oos",)}
        perm = None if self.permutation is None else {k: v for k, v in self.permutation.items() if k != "null"}
        return jsonable({"schema": "pitbacktest/portfolio-review", "schema_version": 1, "result": self.result.to_dict(series="monthly"), "mean_tests": self.mean,
                         "subperiods": self.subperiods, "market_relative": self.market, "decomposition": self.decomposition, "permutation": perm, "sharpe_ci": self.sharpe_ci,
                         "deflated_sharpe": self.deflated, "pbo": self.pbo, "attribution": self.attribution, "risk": self.risk, "drawdown_distribution": self.drawdown_dist,
                         "regimes": None if self.regimes is None else {**self.regimes, "table": _tbl(self.regimes["table"])}, "holdings": self.holdings, "in_out_of_sample": self.split,
                         "brinson": self.brinson, "grid": _tbl(self.grid_table), "plateau": self.plateau, "walk_forward": wf, "cost_table": _tbl(self.cost_table), "delisting_stress": _tbl(self.delist_table),
                         "notes": self.notes, "summary": self.summary()})

    def summary(self) -> dict:
        """The headline of the review in one flat dictionary, in the order a reader should take it: `notes` (what limits the numbers), the figures, the Sharpe interval, the luck tests, the variants
        tried and the deflated Sharpe (of the best trial and of the strategy reviewed), the market and style views, the delisting stress, and `reading`: plain sentences that follow from them.
        Everything is in `to_dict()` too; this is where to start. Figures that could not be computed are `None`."""
        m = self.result.metrics or {}
        d, ci, pm, mt = self.deflated, self.sharpe_ci, self.permutation, self.mean
        dec, rg = self.decomposition, self.regimes
        var = None
        if d:
            var = {"trials": d.get("trials"), "source": d.get("source"), "best_name": d.get("best_name"), "best_sharpe": d.get("sharpe"), "best_deflated_sharpe": d.get("dsr"),
                   "reviewed_sharpe": d.get("reference_sharpe"), "reviewed_deflated_sharpe": d.get("reference_dsr"), "luck_benchmark_sharpe": d.get("sharpe_luck_benchmark")}
        reading = []
        if ci and ci.get("lo") is not None and math.isfinite(ci["lo"]) and ci["lo"] <= 0 <= ci["hi"]:
            reading.append(f"The Sharpe ratio's 95 percent interval ({ci['lo']:.2f} to {ci['hi']:.2f}) includes 0: this sample shows no evidence of an edge.")
        if d:
            rs = d.get("reference_sharpe", d.get("sharpe"))
            if rs is not None and math.isfinite(rs) and rs <= 0:
                reading.append("The strategy's Sharpe ratio is not positive: a deflated Sharpe has nothing to deflate, and its probability is not evidence of anything. No edge is shown.")
            else:
                dv = d.get("reference_dsr", d.get("dsr"))
                if dv is not None and math.isfinite(dv):
                    reading.append(f"Deflated Sharpe {dv:.2f} against {d.get('trials')} trials (about 0.95 or more is the usual bar).")
        if pm and pm.get("p") is not None and math.isfinite(pm["p"]) and d and (d.get("trials") or 1) > 1:
            reading.append("The permutation test does not know that you chose this strategy among several; the deflated Sharpe does.")
        be = None
        ct = self.cost_table
        if ct is not None and len(ct) > 1:
            xs_, ys_ = [float(x) for x in ct.index], [float(v) for v in ct["sharpe"]]
            if all(math.isfinite(v) for v in ys_) and ys_[0] > 0 >= ys_[-1]:
                for (x0, y0), (x1, y1) in zip(zip(xs_, ys_), zip(xs_[1:], ys_[1:])):
                    if y0 > 0 >= y1:
                        be = x0 + (x1 - x0) * y0 / (y0 - y1)
                        break
                if be is not None:
                    reading.append(f"The Sharpe ratio is positive only below about {be:.0f} bp of round-trip spread; compare that with what this strategy would really pay (its turnover is {m.get('turnover_daily', float('nan')):.2f} a day).")
            elif ys_[-1] > 0:
                be = float("inf")
        if pm and pm.get("real_sharpe") is not None:
            reading.append("The permutation test compares Sharpe ratios before costs: a small p-value there says the factor beats shuffled copies of itself, not that the strategy pays after costs.")
        if rg and rg.get("sample_has_bear") is False:
            reading.append("The sample contains no bear market: nothing here says how the strategy behaves in one.")
        if m.get("held_missing_price_days"):
            reading.append(f"{m['held_missing_price_days']} position-days were held with no price and earned 0: check the data.")
        out = {"notes": list(self.notes),
               "figures": {"sharpe": m.get("Sharpe"), "cagr": m.get("CAGR"), "max_drawdown": m.get("MDD"), "volatility": m.get("vol"), "turnover_daily": m.get("turnover_daily"),
                           "cost_annual_bp": m.get("cost_annual_bp"), "borrow_annual_bp": m.get("borrow_annual_bp"), "held_missing_price_days": m.get("held_missing_price_days"),
                           "break_even_spread_bp": None if be is None or be == float("inf") else be},
               "sharpe_interval_95": None if not ci else [ci.get("lo"), ci.get("hi")],
               "luck": {"permutation_p": None if not pm else pm.get("p"), "permutation_gross_sharpe": None if not pm else pm.get("real_sharpe"), "newey_west_t": None if not mt else mt.get("nw_t"), "newey_west_p": None if not mt else mt.get("nw_p"),
                        "sign_test_p": None if not mt else mt.get("sign_p")},
               "variants": var,
               "market": {"sample_has_bear": None if not rg else rg.get("sample_has_bear"), "excess_cagr": None if not self.market else self.market.get("excess_cagr"),
                          "beta": None if not self.market else self.market.get("beta")},
               "style": None if not dec else {"alpha_annual": dec.get("alpha_annual"), "alpha_t": dec.get("alpha_t"), "r2": dec.get("r2")},
               "settings": None if not (self.plateau or self.pbo) else {"plateau_verdict": None if not self.plateau else self.plateau.get("verdict"), "pbo": None if not self.pbo else self.pbo.get("pbo")},
               "walk_forward_efficiency": None if not self.walk_forward else self.walk_forward.get("efficiency"),
               "delisting_stress": None if self.delist_table is None else jsonable(self.delist_table),
               "reading": reading}
        return jsonable(out)

    def to_json(self, path=None, indent: int | None = 2):
        """`to_dict()` as JSON text, or written to `path`."""
        from .export import dump_json
        return dump_json(self.to_dict(), path, indent)

    def report(self, path=None, **kw):
        """The review page (HTML text, or written to `path`); `title` names it and the portfolio report's own keywords (`ledger`, `family`, `capacity`) are accepted."""
        from .report_review import review_html, write_text
        text = review_html(self, **kw)
        return text if path is None else write_text(text, path)


def review_portfolio(panel, factor: pd.DataFrame, *, make_factor=None, grid: dict | None = None, walk_forward: dict | None = None, n_permutations: int = 100,
                     style_factors: bool = True, extra_factors: pd.DataFrame | None = None, groups: pd.Series | None = None, benchmark_returns: pd.Series | None = None,
                     split_date=None, n_bootstrap: int = 1000, spread_grid=(0.0, 5.0, 10.0, 20.0, 40.0), ledger=None,
                     family: str | None = None, extra_trials: int = 0,
                     name: str | None = None, seed: int = 0, **backtest_kwargs) -> PortfolioReview:
    """Run a factor portfolio and every test of the list at the top of this module.

    `factor` and `backtest_kwargs` (long_q, short_q, hold, spread_bp, ...) are the strategy under review. Optional:

    grid, make_factor   a dictionary of settings to try, for the settings and walk-forward questions: for example `{"lookback": [3, 5, 10, 20], "hold": [1, 5, 10]}` with
                        `make_factor(panel, lookback=...)` returning the factor of one setting. Keys that are arguments of `backtest_portfolio` (hold, long_q, short_q,
                        weighting, spread_bp) go there, the others to `make_factor`. Every setting is backtested once over the whole sample (this is what the walk-forward
                        and the deflated Sharpe read) and counted as a trial. Put the reviewed strategy's own setting in the grid.
    walk_forward        `dict(train=..., test=..., expanding=True, embargo=...)` in bars; default: train 40 percent of the sample, test 10 percent, embargo hold + entry lag.
    n_permutations      relabellings of the factor (each security gets another's factor history) for the luck test (0 skips it; at 100 the smallest p is about 0.01)
    style_factors       explain the return with the panel's own size, momentum, reversal, low-volatility, illiquidity and market factors; `extra_factors` adds your own
                        (a DataFrame of factor returns indexed by signal date, for example published Fama-French factors aligned to the panel's convention)
    groups              a Series (ticker -> label, for example the sector) to split the return by in the attribution (and, for a long-only portfolio, in the Brinson table)
    benchmark_returns   an index's *daily returns by date* to use as the benchmark instead of the panel's market-cap or equal-weight one; moved onto the signal dates for you
                        (`robustness.align_benchmark`). Choose it to match what the strategy holds: a tech strategy against a broad index mostly measures tech.
    split_date          the first date of a frozen out-of-sample stretch, fixed before looking at what followed: adds the before-and-after comparison
    n_bootstrap         resampled histories for the maximum-drawdown distribution (0 skips it)
    spread_grid         round-trip spreads in bp for the cost table (skipped when `spread_bp` is a panel)
    ledger, family      record the reviewed run (and each grid setting) in a trial ledger and add its deflated Sharpe
    extra_trials        variants you tried and did not record (see `Ledger.deflated_sharpe`): added to the trial count of the deflated Sharpe, from the grid and from the ledger alike. Count anything whose
                        result you looked at: another cleaning of the data, another universe, another formula, another horizon
    """
    from .portfolio import backtest_portfolio
    bad = set(backtest_kwargs) - _BACKTEST_KEYS
    if bad:
        raise ValueError(f"unknown argument(s) {sorted(bad)}; the strategy's settings are {sorted(_BACKTEST_KEYS)}")
    if (grid is None) != (make_factor is None):
        raise ValueError("grid and make_factor go together: give both or neither")
    ppy = panel.periods_per_year
    notes: list[str] = []
    res = backtest_portfolio(panel, factor, ledger=ledger, family=family or "review", name=name or "reviewed", **backtest_kwargs)
    if benchmark_returns is not None:
        from dataclasses import replace as _replace
        from .portfolio import metrics as _metrics
        al = rb.align_benchmark(panel, benchmark_returns).reindex(res.net_returns.index)
        ok = al.notna()
        if ok.sum() < 60:
            raise ValueError(f"the benchmark covers only {int(ok.sum())} of the strategy's dates: at least 60 are needed")
        bm = _metrics(al[ok].to_numpy(), res.net_returns.index[ok], ppy)
        ex = _metrics((res.net_returns[ok] - al[ok]).to_numpy(), res.net_returns.index[ok], ppy)
        res = _replace(res, benchmark_returns=al, benchmark=bm, excess=ex)
        notes.append("the benchmark is the series you supplied (moved onto the signal dates)")
    s = res.net_returns
    out = PortfolioReview(result=res, mean=rb.mean_tests(s, periods_per_year=ppy), subperiods=rb.subperiods(s, periods_per_year=ppy))
    try:
        out.sharpe_ci = res.sharpe_ci(seed=seed)
    except ValueError as e:
        notes.append(f"Sharpe interval not computed: {e}")
    if res.benchmark_returns is not None:
        out.market = rb.market_relative(s, res.benchmark_returns, periods_per_year=ppy)
    try:
        out.attribution = rb.attribution(panel, res, groups=groups)
    except ValueError as e:
        notes.append(f"attribution not computed: {e}")
    for key, fn in (("risk", lambda: rb.risk_profile(s, periods_per_year=ppy)), ("holdings", lambda: rb.holdings_summary(panel, res))):
        try:
            setattr(out, key, fn())
        except ValueError as e:
            notes.append(f"{key} not computed: {e}")
    if n_bootstrap:
        try:
            out.drawdown_dist = rb.drawdown_distribution(s, n=int(n_bootstrap), seed=seed)
        except ValueError as e:
            notes.append(f"drawdown distribution not computed: {e}")
    if res.benchmark_returns is not None:
        try:
            out.regimes = rb.regimes(s, res.benchmark_returns, periods_per_year=ppy)
        except ValueError as e:
            notes.append(f"regimes not computed: {e}")
    if split_date is not None:
        out.split = rb.in_out_of_sample(s, split_date, periods_per_year=ppy)
    if groups is not None:
        try:
            out.brinson = rb.brinson(panel, res, groups)
        except ValueError as e:
            notes.append(f"Brinson attribution not computed: {e}")
    if style_factors or extra_factors is not None:
        try:
            with muted():
                F = rb.style_factor_returns(panel) if style_factors else pd.DataFrame(index=s.index)
            if extra_factors is not None:
                F = pd.concat([F, extra_factors], axis=1)
            keep = [c for c in F.columns if F[c].reindex(s.index).notna().mean() >= 0.8]
            for c in set(F.columns) - set(keep):
                notes.append(f"factor {c!r} left out of the decomposition: it covers under 80 percent of the strategy's dates")
            out.style_returns = F[keep]
            out.decomposition = rb.decompose(s, F[keep], periods_per_year=ppy)
        except ValueError as e:
            notes.append(f"decomposition not computed: {e}")
    if n_permutations:
        pk = {k: v for k, v in backtest_kwargs.items() if k not in ("benchmark",)}
        try:
            with muted():
                out.permutation = rb.factor_permutation(panel, factor, n=n_permutations, seed=seed, **pk)
        except ValueError as e:
            notes.append(f"permutation test not computed: {e}")
    if spread_grid is not None and np.ndim(backtest_kwargs.get("spread_bp", 20.0)) == 0:
        rows = []
        for c in spread_grid:
            with muted():
                r = backtest_portfolio(panel, factor, **{**backtest_kwargs, "spread_bp": float(c), "benchmark": None, "grid": False})
            rows.append({"spread_bp": float(c), "sharpe": r.metrics["Sharpe"], "cagr": r.metrics["CAGR"]})
        out.cost_table = pd.DataFrame(rows).set_index("spread_bp")
    if panel.delist_after is not None and bool(np.asarray(panel.delist_after.values).any()):
        rows = []
        for dr in (0.0, -0.3, -0.5, -1.0):
            with muted():
                r = backtest_portfolio(panel, factor, **{**backtest_kwargs, "delist_return": dr, "benchmark": None, "grid": False})
            rows.append({"delist_return": dr, "sharpe": r.metrics["Sharpe"], "cagr": r.metrics["CAGR"], "delist_events_held": r.metrics["delist_events_held"]})
        out.delist_table = pd.DataFrame(rows).set_index("delist_return")
        ev_held = int(out.delist_table["delist_events_held"].iloc[0])
        if ev_held == 0:
            notes.append("delisting stress: no held position was delisted in this run, so the assumed loss on a delisting changes nothing here")
        else:
            s0, s50 = float(out.delist_table["sharpe"].loc[0.0]), float(out.delist_table["sharpe"].loc[-0.5])
            line = f"delisting stress: {ev_held} held positions were delisted; the Sharpe ratio is {s0:.2f} if a delisting loses nothing and {s50:.2f} if it loses 50 percent"
            notes.append(line + ("; the verdict depends on what a delisting costs" if (math.isfinite(s0) and math.isfinite(s50) and (s50 < 0.5 * s0 or s50 < 0 <= s0)) else ""))
    if grid is not None:
        grid = {k: _ordered(v) for k, v in grid.items()}
        cfgs = _grid_configs(grid)
        cols, sharpes = {}, {}
        for cfg in cfgs:
            bt = {k: v for k, v in cfg.items() if k in _BACKTEST_KEYS}
            fp = {k: v for k, v in cfg.items() if k not in _BACKTEST_KEYS}
            label = ", ".join(f"{k}={v}" for k, v in cfg.items())
            with muted():
                r = backtest_portfolio(panel, make_factor(panel, **fp), **{**backtest_kwargs, **bt, "benchmark": None, "grid": False}, ledger=ledger, family=(family or "review") + "-grid",
                                       name=label)
            cols[label] = r.net_returns
            sharpes[tuple(cfg.values())] = r.metrics["Sharpe"]
        R = pd.DataFrame(cols)
        keys = list(grid)
        if len(keys) in (1, 2):
            if len(keys) == 1:
                out.grid_table = pd.Series({k[0]: v for k, v in sharpes.items()}, name="Sharpe").rename_axis(keys[0])
            else:
                out.grid_table = pd.DataFrame({b: {a: sharpes[(a, b)] for a in grid[keys[0]]} for b in grid[keys[1]]})
                out.grid_table.index.name, out.grid_table.columns.name = keys[0], keys[1]
            if out.grid_table.size >= 3 and np.isfinite(out.grid_table.to_numpy(float)).any():                 # a Series (one parameter) or a table (two)
                out.plateau = rb.parameter_plateau(out.grid_table)
        else:
            notes.append("the settings table is drawn for one or two parameters; with more, only the walk-forward and the deflated Sharpe are computed")
        hold = backtest_kwargs.get("hold", 5)
        wf = dict(walk_forward or {})
        T = len(R)
        wf.setdefault("train", max(60, int(0.4 * T)))
        wf.setdefault("test", max(20, int(0.1 * T)))
        wf.setdefault("embargo", int(hold) + panel.entry_lag)
        try:
            out.walk_forward = rb.walk_forward(R, periods_per_year=ppy, **wf)
        except ValueError as e:
            notes.append(f"walk-forward not computed: {e}")
        from .validation import deflated_sharpe
        Rm = R.dropna().to_numpy()
        if Rm.shape[1] >= 2 and len(Rm) > 30:
            out.deflated = deflated_sharpe(Rm, trials=Rm.shape[1] + int(extra_trials), periods_per_year=ppy, reference=s.to_numpy())
            out.deflated["best_name"] = str(R.columns[int(out.deflated["best"])])
            out.deflated["source"] = "grid"
            if Rm.shape[1] >= 4 and len(Rm) >= 16 * 8:                                    # CSCV needs enough bars per block to rank the settings
                from .validation import pbo_cscv
                out.pbo = pbo_cscv(Rm, periods_per_year=ppy, seed=seed)
    if ledger is not None and out.deflated is None:
        try:
            out.deflated = {**ledger.deflated_sharpe(family or "review", periods_per_year=ppy, extra_trials=extra_trials, reference=s.to_numpy()), "source": "ledger"}
        except Exception as e:                                                          # a ledger with one trial has nothing to deflate against
            notes.append(f"deflated Sharpe from the ledger not computed: {str(e)[:80]}")
    out.notes = list(res.notes) + notes
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------- events
@dataclass
class EventReview:
    result: object
    horizon: int
    cost_bp: float
    horizons_table: pd.DataFrame
    cost_table: pd.DataFrame                         # win rate, mean and median at several round-trip costs, at the best horizon
    yearly: pd.DataFrame
    streak: dict
    daily_excess: dict
    hold: int = 1                                    # the holding period of the portfolio view
    segments: dict = field(default_factory=dict)      # win rate and mean by third of liquidity, market capitalisation and price level
    permutation: dict | None = None
    portfolio: object | None = None
    portfolio_market: dict | None = None
    portfolio_mean: dict | None = None
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        """The headline of the event review in one flat dictionary: `notes`, the horizon and whether it was fixed in advance, the trade statistics, the day-level excess return, the luck test, the gates
        and `reading`. Everything is in `to_dict()` too. Figures that could not be computed are `None`."""
        ht = self.horizons_table
        row = ht.loc[self.horizon] if ht is not None and self.horizon in ht.index else None
        g = self.result.gates if hasattr(self.result, "gates") else {}
        de, pm = self.daily_excess or {}, self.permutation
        fixed = bool((self.result.spec or {}).get("horizon_fixed_in_advance"))
        reading = []
        if not fixed and ht is not None and len(ht) > 1:
            reading.append(f"The horizon ({self.horizon} bars) was chosen as the best of {len(ht)} by its lift on this sample: the figures are optimistic. Fix it in advance with primary_horizon.")
        if row is not None and math.isfinite(float(row["lift_pp"])):
            reading.append(f"Win rate {float(row['win_rate']):.1f} percent against {float(row['base_rate']):.1f} percent for a random pick on the same days: lift {float(row['lift_pp']):+.2f} points.")
        if g and not g.get("passed", True):
            reading.append(f"It failed gate {g.get('failed_at')}.")
        out = {"notes": list(self.notes), "horizon": int(self.horizon), "horizon_fixed_in_advance": fixed, "cost_bp": self.cost_bp,
               "trades": None if row is None else {"n_trades": int(row["n_trades"]), "win_rate": float(row["win_rate"]), "base_rate": float(row["base_rate"]), "lift_pp": float(row["lift_pp"]),
                                                   "mean_bp": float(row["mean_bp"]), "median_bp": float(row["median_bp"])},
               "daily_excess": {k: de.get(k) for k in ("days", "mean_excess_bp", "t", "p")},
               "luck": None if not pm else {"p_win": pm.get("p_win"), "p_mean": pm.get("p_mean"), "method": pm.get("method")},
               "gates": {"passed": g.get("passed"), "failed_at": g.get("failed_at")} if g else None,
               "longest_losing_streak": (self.streak or {}).get("longest_losing_streak"), "reading": reading}
        return jsonable(out)

    def to_dict(self) -> dict:
        """The whole review as a JSON-ready dictionary: the engine's result, the horizons, the cost and year tables, the segments, the permutation and the portfolio view."""
        return jsonable({"schema": "pitbacktest/event-review", "schema_version": 1, "result": self.result.to_dict(), "horizon": self.horizon, "cost_bp": self.cost_bp,
                         "horizons": _tbl(self.horizons_table), "cost_table": _tbl(self.cost_table), "yearly": _tbl(self.yearly), "streak": self.streak,
                         "daily_excess": self.daily_excess, "segments": {k: _tbl(v) for k, v in self.segments.items()}, "permutation": self.permutation, "portfolio": None if self.portfolio is None else self.portfolio.to_dict("monthly"),
                         "portfolio_market_relative": self.portfolio_market, "summary": self.summary(), "portfolio_mean_tests": self.portfolio_mean, "notes": self.notes})

    def to_json(self, path=None, indent: int | None = 2):
        """`to_dict()` as JSON text, or written to `path`."""
        from .export import dump_json
        return dump_json(self.to_dict(), path, indent)

    def report(self, path=None, **kw):
        """The review page (HTML text, or written to `path`); `title` names it."""
        from .report_review import review_html, write_text
        text = review_html(self, **kw)
        return text if path is None else write_text(text, path)


def _longest_run(mask) -> int:
    """The longest run of consecutive True in a boolean sequence."""
    run = best = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z, p = 1.959963984540054, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return min(max(c - h, 0.0), p), max(min(c + h, 1.0), p)                      # rounding must not put the point estimate outside its own interval (win rate 1 gave 0.9999999999999999)


def review_event(panel, signal: pd.DataFrame, *, horizons=(1, 5, 20), cost_bp: float = 20.0, cost_grid=(0.0, 5.0, 10.0, 20.0, 40.0, 80.0), n_permutations: int = 200,
                 as_portfolio: bool = True, hold: int | None = None, seed: int = 0, **event_kwargs) -> EventReview:
    """Run an event signal and every test of the list at the top of this module.

    The headline is the **win rate against the win rate of a random pick on the same days** (`base_rate`, `lift_pp`): with a longer holding period both rise together and
    a rising market lifts every pick, so only the difference says anything about the signal. `cost_bp` is the round-trip cost per trade. `event_kwargs` go to `backtest_event`.

    horizons        holding periods in bars; the best by lift is the one the permutation, the cost table, the yearly table and the portfolio use (`spec["best_horizon"]`, chosen
                    among the horizons you give: the more you give the more optimistic it is, and the report says so)
    cost_grid       round-trip costs in bp for the cost table (at which cost does the win rate fall to a coin toss)
    n_permutations  copies of the signal moved in time for the luck test (see `robustness.signal_permutation`); 0 skips it
    as_portfolio    also hold the signal as a portfolio (`event_weights`, `hold` bars, default the best horizon) and judge it as one: CAGR, Sharpe, drawdown, against the market
    """
    from .event import backtest_event
    from .weights import backtest_weights
    res = backtest_event(panel, signal, horizons=horizons, cost_bp=cost_bp, **event_kwargs)
    h = int(res.spec["best_horizon"])
    notes = list(res.notes)
    rows = []
    for hh, st in sorted(res.per_horizon.items()):
        if st:
            lo, hi = _wilson(int(round(st["win_rate"] / 100 * st["n_trades"])), int(st["n_trades"]))
            rows.append({"horizon": hh, "n_trades": st["n_trades"], "win_rate": st["win_rate"], "win_lo": lo * 100, "win_hi": hi * 100, "base_rate": st["base_rate"], "lift_pp": st["lift_pp"],
                         "mean_bp": st["mean_bp"], "median_bp": st["median_bp"], "avg_win_bp": st["avg_win_bp"], "avg_loss_bp": st["avg_loss_bp"], "payoff": st["payoff"]})
    horizons_table = pd.DataFrame(rows).set_index("horizon")

    dr = event_kwargs.get("delist_return")                                                    # the engine's own figures use it, so do these tables
    fwd = panel.forward(h, dr).to_numpy(np.float64)
    elig = panel.eligible.to_numpy(bool)
    fire = signal.reindex(index=panel.dates, columns=panel.tickers).fillna(False).to_numpy(bool) & elig & np.isfinite(fwd)
    ii, jj = np.nonzero(fire)                                                          # date-major order: chronological
    raw = fwd[ii, jj]
    pool = elig & np.isfinite(fwd)
    fire_days = [i for i in np.unique(ii) if pool[i].sum() >= 20]                              # the engine's rule for the random-pick rate: a pool of at least 20
    kd = {i: int(fire[i].sum()) for i in fire_days}
    cost_rows = []
    for c in cost_grid:
        r = raw - c / 1e4
        base = sum(kd[i] * float(((fwd[i][pool[i]] - c / 1e4) > 0).mean()) for i in fire_days) / max(1, sum(kd.values()))
        wr = float((r > 0).mean() * 100)
        cost_rows.append({"cost_bp": float(c), "win_rate": wr, "base_rate": base * 100, "lift_pp": wr - base * 100, "mean_bp": float(r.mean() * 1e4), "median_bp": float(np.median(r) * 1e4)})
    r_net = raw - cost_bp / 1e4
    day_ret = pd.Series(r_net).groupby(ii).mean()                                              # one number per fire-day: the order of names inside a day does not matter
    streak = {"longest_losing_streak": int(_longest_run((day_ret <= 0).to_numpy())), "unit": "fire-days", "n_trades": int(len(r_net)), "win_rate_pct": float((r_net > 0).mean() * 100)}
    dates = panel.dates[ii]
    yrows = []
    for y in sorted(set(dates.year)):
        m = np.asarray(dates.year == y)
        k, n_ = int((r_net[m] > 0).sum()), int(m.sum())
        lo, hi = _wilson(k, n_)
        yrows.append({"year": int(y), "n_trades": n_, "win_rate": k / n_ * 100, "win_lo": lo * 100, "win_hi": hi * 100, "mean_bp": float(r_net[m].mean() * 1e4)})
    yearly = pd.DataFrame(yrows).set_index("year")
    # day-level excess over the random pick of the same day: trades of one day are correlated, so the unit is the day. The engine's own definition (its pool rule, its Newey-West lag)
    from .event import _daily_excess
    ex = _daily_excess(fire, fwd, elig)
    lag_ = max(h, 21)
    if len(ex) >= 30:
        m_, se_, t_, _ = rb.newey_west_t(ex, lag_)
        dex = {"days": int(len(ex)), "mean_excess_bp": float(m_ * 1e4), "t": float(t_), "p": float(rb._norm_two_sided(t_)) if math.isfinite(t_) else float("nan"), "lag": int(lag_)}
    else:
        dex = {"days": int(len(ex)), "mean_excess_bp": float("nan"), "t": float("nan"), "p": float("nan"), "lag": int(lag_)}
    segs = {}
    for label, values in (("liquidity (30-bar average traded value)", panel.adv(30) if panel.volume is not None else None),
                          ("market capitalisation", panel.mkt_cap.where(panel.mkt_cap > 0) if panel.mkt_cap is not None else None),
                          ("price level", pd.DataFrame(panel.meta["raw_close"]).reindex(index=panel.dates, columns=panel.tickers) if panel.meta.get("raw_close") is not None else None)):
        if values is None or len(ii) == 0:
            continue
        terc = rb._terciles(values, panel.eligible)[ii, jj]
        rows_ = []
        for k, lab in enumerate(("lowest third", "middle third", "highest third")):
            m_ = terc == k
            if m_.sum() >= 20:
                rr = r_net[m_]
                lo_, hi_ = _wilson(int((rr > 0).sum()), int(m_.sum()))
                rows_.append({"bucket": lab, "n_trades": int(m_.sum()), "share_of_fires": float(m_.mean() * 100), "win_rate": float((rr > 0).mean() * 100), "win_lo": lo_ * 100, "win_hi": hi_ * 100,
                              "mean_bp": float(rr.mean() * 1e4), "median_bp": float(np.median(rr) * 1e4)})
        if rows_:
            segs[label] = pd.DataFrame(rows_).set_index("bucket")
    rv = EventReview(result=res, horizon=h, segments=segs, cost_bp=float(cost_bp), horizons_table=horizons_table, cost_table=pd.DataFrame(cost_rows).set_index("cost_bp"), yearly=yearly, streak=streak,
                     daily_excess=dex, notes=notes)
    if n_permutations:
        try:
            rv.permutation = rb.signal_permutation(panel, signal, h, cost_bp, n=n_permutations, seed=seed, delist_return=dr)
        except ValueError as e:
            rv.notes.append(f"permutation test not computed: {e}")
    if as_portfolio:
        try:
            rv.hold = int(hold or h)
            W = event_weights(panel, signal, rv.hold)
            with muted():
                pr = backtest_weights(panel, W, spread_bp=float(cost_bp), benchmark="cap", check_universe=False)
            rv.portfolio = pr
            rv.portfolio_mean = rb.mean_tests(pr.net_returns, periods_per_year=panel.periods_per_year)
            if pr.benchmark_returns is not None:
                rv.portfolio_market = rb.market_relative(pr.net_returns, pr.benchmark_returns, periods_per_year=panel.periods_per_year)
        except ValueError as e:
            rv.notes.append(f"portfolio view not computed: {str(e)[:100]}")
    return rv
