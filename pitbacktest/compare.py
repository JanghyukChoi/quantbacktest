"""Strategies side by side, and strategies mixed.

    q.compare({"reversal": r1, "momentum": r2, "my event": ev}, "compare.html", panel=panel, benchmark_returns=index_daily_returns)
    q.strategy_correlation({"reversal": r1, "momentum": r2})
    mix = q.blend(panel, {"reversal": r1, "momentum": r2}, weights={"reversal": 0.6, "momentum": 0.4}, spread_bp=10)

`compare` puts up to six results on one page: a table of the usual figures with the benchmark's beside them, the growth of 1 of each, their risk and return as a scatter,
the correlation of their returns, and year by year. A factor result (`backtest_portfolio`, `backtest_weights`), a `PortfolioReview` and an event review (taken as the portfolio view
of its signal, see `review_event`) can be mixed, which is the point: an event signal held as a portfolio and a factor are on the same footing. Everything is computed on the dates all of
them share, from each strategy's net returns.

`blend` mixes weights, not returns: the combined target weights are the weighted sum of the parts' holdings and the result is a `backtest_weights` run of them. Netting is the reason:
where one part buys what another sells, only the difference trades, so a blend costs less than the weighted costs of its parts (and with zero costs it equals the weighted sum of their
returns exactly). Give the `spread_bp` and other costs explicitly: the parts were run with their own and the blend cannot know which to use."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import robustness as rb
from .export import naive_index
from .report import _chart, _date_fmt, _esc, _nice_ticks, _num, _pct, _unsigned_zero, _year_fmt, _year_ticks, wide_range
from .report_review import _EXTRA_CSS, _review_page

_COLORS = ("var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)")
MAX_STRATEGIES = 6


def _result_of(x):
    """The `PortfolioResult` behind whatever was given: a result, a `PortfolioReview`, or an `EventReview` (its portfolio view)."""
    if hasattr(x, "horizons_table"):                                   # EventReview
        if x.portfolio is None:
            raise ValueError("this event review has no portfolio view (it was run with as_portfolio=False)")
        return x.portfolio
    if hasattr(x, "result") and hasattr(x, "mean"):                    # PortfolioReview
        return x.result
    if hasattr(x, "net_returns") and hasattr(x, "metrics"):
        return x
    raise ValueError(f"cannot compare a {type(x).__name__}: give a backtest result, a review, or an event review")


def _nets(parts: dict) -> tuple[dict, int]:
    if not isinstance(parts, dict) or len(parts) < 1:
        raise ValueError("parts must be a non-empty dict of name -> result")
    if len(parts) > MAX_STRATEGIES:
        raise ValueError(f"at most {MAX_STRATEGIES} strategies fit on one page (the colours stay distinguishable), got {len(parts)}")
    res = {k: _result_of(v) for k, v in parts.items()}
    ppys = {int(r.spec.get("periods_per_year", 252)) for r in res.values()}
    if len(ppys) != 1:
        raise ValueError(f"the strategies use different periods_per_year ({sorted(ppys)}): they are not on the same bars")
    return res, ppys.pop()


def strategy_correlation(parts: dict) -> pd.DataFrame:
    """Correlation of the strategies' net returns on the dates they all share (a matrix with the names on both axes)."""
    res, _ = _nets(parts)
    df = pd.concat({k: r.net_returns.astype(float) for k, r in res.items()}, axis=1, join="inner").dropna()
    if len(df) < 30:
        raise ValueError(f"the strategies share only {len(df)} dates: at least 30 are needed")
    return df.corr()


def blend(panel, parts: dict, weights: dict | None = None, *, normalize: bool = True, **backtest_kwargs):
    """Mix strategies by their holdings and run the mixture (see the module docstring). `weights`: name -> share of the capital (default equal); with `normalize` they are divided by
    their sum so that the blend uses the capital once. `backtest_kwargs` go to `backtest_weights`; `spread_bp` must be given."""
    from .weights import backtest_weights
    if "spread_bp" not in backtest_kwargs:
        raise ValueError("give spread_bp (and any other cost) explicitly: the parts were run with their own costs and the blend cannot know which to use")
    res, _ = _nets(parts)
    names = list(res)
    w = {k: 1.0 / len(names) for k in names} if weights is None else dict(weights)
    if set(w) != set(names):
        raise ValueError(f"weights must name exactly the strategies {names}, got {sorted(w)}")
    tot = float(sum(w.values()))
    if not math.isfinite(tot) or abs(tot) < 1e-12:
        raise ValueError("the weights sum to zero")
    if normalize:
        w = {k: v / tot for k, v in w.items()}
    H = None
    for k, r in res.items():
        if r.holdings is None:
            raise ValueError(f"strategy {k!r} has no holdings")
        h = np.asarray(r.holdings, dtype=float)
        if h.shape != (len(panel.dates), len(panel.tickers)):
            raise ValueError(f"strategy {k!r} has holdings of shape {h.shape}, the panel is {(len(panel.dates), len(panel.tickers))}: it was run on another panel")
        H = h * w[k] if H is None else H + h * w[k]
    backtest_kwargs.setdefault("benchmark", "cap")
    backtest_kwargs.setdefault("check_universe", False)                    # each part was checked on its own; a sum of lawful books is lawful
    return backtest_weights(panel, pd.DataFrame(H, index=panel.dates, columns=panel.tickers), **backtest_kwargs)


def _scatter(cid: str, caption: str, pts: list, *, width: int = 760, height: int = 260) -> str:
    """Volatility (x) against CAGR (y), one dot per strategy with its name beside it."""
    xs = np.array([p[1] for p in pts], dtype=float)
    ys = np.array([p[2] for p in pts], dtype=float)
    x0, x1 = max(0.0, float(xs.min()) * 0.8), float(xs.max()) * 1.2 + 1e-9
    y0, y1 = min(0.0, float(ys.min()) * 1.2), max(0.0, float(ys.max()) * 1.2) + 1e-9
    if y1 - y0 < 1e-9:
        y1 = y0 + 0.1
    L, R, T, B = 52, 90, 12, 34
    px = lambda v: L + (v - x0) / (x1 - x0) * (width - L - R)                # noqa: E731
    py = lambda v: T + (height - T - B) * (1 - (v - y0) / (y1 - y0))         # noqa: E731
    g = []
    for t in _nice_ticks(y0, y1, 4):
        g.append(f'<line class="grid" x1="{L}" x2="{width - R}" y1="{py(t):.1f}" y2="{py(t):.1f}"/><text x="{L - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{_esc(_unsigned_zero(f"{t * 100:.0f}"))}%</text>')
    for t in _nice_ticks(x0, x1, 5):
        g.append(f'<text x="{px(t):.1f}" y="{height - 16}" text-anchor="middle">{_esc(f"{t * 100:.0f}")}%</text>')
    g.append(f'<text x="{(L + width - R) / 2:.1f}" y="{height - 2}" text-anchor="middle">volatility, annualised</text>')
    if y0 < 0 < y1:
        g.append(f'<line class="axis" x1="{L}" x2="{width - R}" y1="{py(0):.1f}" y2="{py(0):.1f}"/>')
    dots = "".join(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="6" style="fill:{col};stroke:var(--surface);stroke-width:2"><title>{_esc(lab)}: CAGR {_pct(y)}, volatility {_pct(x)}</title></circle>'
                   f'<text x="{px(x) + 10:.1f}" y="{py(y) + 4:.1f}" style="fill:var(--ink)">{_esc(lab)}</text>' for lab, x, y, col in pts)
    return (f'<figure class="chart" id="{_esc(cid)}"><figcaption>{_esc(caption)}</figcaption><svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(caption)}">{"".join(g)}{dots}</svg></figure>')


def compare(parts: dict, path=None, *, panel=None, benchmark_returns: pd.Series | None = None, title: str | None = None, max_points: int = 900):
    """One page with up to six strategies side by side. Returns the HTML text, or writes it to `path` and returns the path.

    `benchmark_returns` (an index's daily returns by date, with the `panel` it is to be put on the signal dates of) is shown beside the strategies and used for beta; without it the first
    strategy that has a benchmark of its own supplies it, and the page says so."""
    from .portfolio import metrics as _metrics
    res, ppy = _nets(parts)
    names = list(res)
    df = pd.concat({k: r.net_returns.astype(float) for k, r in res.items()}, axis=1, join="inner").dropna()
    if len(df) < 60:
        raise ValueError(f"the strategies share only {len(df)} dates: at least 60 are needed")
    bench, bsrc = None, ""
    if benchmark_returns is not None:
        if panel is None:
            raise ValueError("benchmark_returns needs the panel (to put the index on the signal dates)")
        bench, bsrc = rb.align_benchmark(panel, benchmark_returns).reindex(df.index), "the index you supplied"
    else:
        for k, r in res.items():
            if r.benchmark_returns is not None:
                bench, bsrc = r.benchmark_returns.reindex(df.index), f"the benchmark of {k!r}"
                break
    if bench is not None and bench.notna().sum() < 60:
        bench, bsrc = None, ""
    d = df.copy()
    d.index = naive_index(d.index)
    bn = None if bench is None else bench.set_axis(naive_index(bench.index))
    cols = {n: _COLORS[i] for i, n in enumerate(names)}
    rows, risk = {}, {}
    for n in names:
        x = d[n].to_numpy()
        m = _metrics(x, d.index, ppy)
        rk = rb.risk_profile(d[n], periods_per_year=ppy)
        risk[n] = rk
        tm = res[n].metrics
        rows[n] = {"Total return": _pct(rk["total_return"]), "CAGR": _pct(m["CAGR"]), "Volatility": _pct(m["vol"]), "Sharpe": _num(m["Sharpe"]), "Sortino": _num(m["Sortino"]),
                   "Max drawdown": _pct(m["MDD"]), "Skewness": _num(rk["skew"]), "Kurtosis": _num(rk["kurtosis"]), "VaR 95% (one bar)": _pct(rk["var_95"], 2),
                   "Expected shortfall 95%": _pct(rk["cvar_95"], 2),
                   "Beta to the benchmark": _num(rb.market_relative(d[n], bn, periods_per_year=ppy)["beta"]) if bn is not None else "n/a",
                   "Turnover per 21 bars": _pct(tm.get("turnover_daily", float("nan")) * 21 if isinstance(tm.get("turnover_daily"), (int, float)) else None, 0),
                   "Cost per year": (f"{tm['cost_annual_bp']:,.0f} bp" if isinstance(tm.get("cost_annual_bp"), (int, float)) else (f"{sum(tm[k] for k in ('spread_annual_bp', 'side_cost_annual_bp', 'borrow_annual_bp', 'impact_annual_bp') if isinstance(tm.get(k), (int, float))):,.0f} bp" if "spread_annual_bp" in tm else "n/a"))}
    brow = None
    if bn is not None:
        bm = _metrics(bn.dropna().to_numpy(), bn.dropna().index, ppy)
        brk = rb.risk_profile(bn.dropna(), periods_per_year=ppy)
        brow = {"Total return": _pct(brk["total_return"]), "CAGR": _pct(bm["CAGR"]), "Volatility": _pct(bm["vol"]), "Sharpe": _num(bm["Sharpe"]), "Sortino": _num(bm["Sortino"]), "Max drawdown": _pct(bm["MDD"]),
                "Skewness": _num(brk["skew"]), "Kurtosis": _num(brk["kurtosis"]), "VaR 95% (one bar)": _pct(brk["var_95"], 2), "Expected shortfall 95%": _pct(brk["cvar_95"], 2), "Beta to the benchmark": "1.00",
                "Turnover per 21 bars": "", "Cost per year": ""}
    metrics_rows = list(rows[names[0]])
    head = "<tr><th>Figure</th>" + "".join(f'<th><span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;background:{cols[n]}"></span>{_esc(n)}</th>' for n in names) + ("<th>Benchmark</th>" if brow else "") + "</tr>"
    body = "".join("<tr><td>" + _esc(k) + "</td>" + "".join(f"<td>{rows[n][k]}</td>" for n in names) + (f"<td>{brow[k]}</td>" if brow else "") + "</tr>" for k in metrics_rows)
    eq_series = [{"name": n, "y": (1.0 + d[n]).cumprod().to_numpy(), "color": cols[n], "fmt": "eq"} for n in names]
    if bn is not None:
        eq_series.append({"name": "Benchmark (no costs)", "y": (1.0 + bn.fillna(0.0)).cumprod().to_numpy(), "color": "var(--muted)", "fmt": "eq"})
    xs = np.array([t.value // 10 ** 6 for t in d.index], dtype=np.float64)
    tk = _year_ticks(d.index[0], d.index[-1]) if (d.index[-1] - d.index[0]).days > 700 else None
    wide = wide_range([sr["y"] for sr in eq_series])
    chart = _chart("cmp", "Growth of 1 (net of costs)" + (", log scale" if wide else ""), xs, eq_series, yfmt="num", xfmt=_date_fmt, height=280, max_points=max_points, xticks=tk,
                   xtickfmt=_year_fmt if tk else None, logy=wide)
    vc = {n: rows_f(d, n, ppy) for n in names}
    scat = _scatter("risk", "Risk and return of each strategy (and the benchmark)", [(n, vc[n][0], vc[n][1], cols[n]) for n in names] +
                    ([("Benchmark", float(bm["vol"]), float(bm["CAGR"]), "var(--muted)")] if brow else []))
    corr = d.corr()
    chead = "<tr><th></th>" + "".join(f"<th>{_esc(n)}</th>" for n in names) + "</tr>"
    crows = ""
    for a in names:
        cells = ""
        for b in names:
            v = float(corr.loc[a, b])
            arm, amt = ("pos", v) if v >= 0 else ("neg", -v)
            bg = f"color-mix(in srgb, var(--{arm}) {min(55.0, 8.0 + 47.0 * min(1.0, amt)):.0f}%, var(--mid))"
            cells += f'<td style="background:{bg}" title="{_esc(a)} and {_esc(b)}: {v:.2f}">{_unsigned_zero(f"{v:.2f}")}</td>'
        crows += f"<tr><td>{_esc(a)}</td>{cells}</tr>"
    yrows = ""
    for y, g in d.groupby(d.index.year):
        rets = {n: float((1.0 + g[n]).prod() - 1.0) for n in names}
        br = float((1.0 + bn.reindex(g.index).fillna(0.0)).prod() - 1.0) if bn is not None else None
        yrows += f"<tr><td>{y}</td>" + "".join(f"<td>{_pct(rets[n])}</td>" for n in names) + (f"<td>{_pct(br)}</td>" if br is not None else "") + f"<td>{len(g)}</td></tr>"
    yhead = "<tr><th>Year</th>" + "".join(f"<th>{_esc(n)}</th>" for n in names) + ("<th>Benchmark</th>" if brow else "") + "<th>Bars</th></tr>"
    ttl = title or "Strategies side by side"
    sub = f"{d.index[0].date()} to {d.index[-1].date()} · {len(d):,} bars in common · " + (f"benchmark: {bsrc}" if brow else "no benchmark")
    body_html = (f'<h2>Figures</h2><div class="card wrap"><table>{head}{body}</table></div><h2>Growth</h2><div class="card">{chart}</div>'
                 f'<h2>Risk and return</h2><div class="card">{scat}</div>'
                 f'<h2>Do they move together?</h2><div class="card wrap"><table class="heat">{chead}{crows}</table>'
                 '<p class="note">Correlation of the strategies\' net returns on the shared dates. Strategies that are not correlated improve each other when mixed (<code>q.blend</code>).</p></div>'
                 f'<h2>Year by year</h2><div class="card wrap"><table>{yhead}{yrows}</table></div>')
    page = _review_page(f"<h1>{_esc(ttl)}</h1><p class=\"sub\">{_esc(sub)}</p>", body_html, ttl)
    page = page.replace("</style>", _EXTRA_CSS + "</style>", 1)
    if path is None:
        return page
    p = Path(path)
    p.write_text(page, encoding="utf-8")
    return p


def rows_f(d, n, ppy):
    """(volatility, CAGR) of strategy n on the shared dates."""
    from .portfolio import metrics as _metrics
    m = _metrics(d[n].to_numpy(), d.index, ppy)
    return float(m["vol"]), float(m["CAGR"])


__all__ = ["compare", "blend", "strategy_correlation", "MAX_STRATEGIES"]
