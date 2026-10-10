"""The review page: `PortfolioReview.report()` and `EventReview.report()` (see `pitbacktest.review`).

A portfolio review is the ordinary report (`report_html`) with the evidence sections added; an event review is a page of its own, led by the win rate against the win rate of a
random pick. Same look, same colours, same rule: the page states what a number is and how uncertain it is, and does not recommend anything."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .export import naive_index
from .report import (_CSS, _JS, _chart, wide_range, _date_fmt, _datetime_fmt, _esc, _num, _pct, _tile, _unsigned_zero, _year_fmt, _year_ticks, report_html)

_EXTRA_CSS = """
.rd{display:inline-block;width:18px;font-weight:700;text-align:center}.t{width:100%}.t td:last-child,.t th:last-child{text-align:left}.t td.n{white-space:nowrap}
.bars{display:grid;grid-template-columns:max-content 1fr max-content;gap:3px 10px;font-size:13px;align-items:center}.bars .b{height:14px;position:relative;background:transparent}
.bars .b i{position:absolute;top:0;height:14px;border-radius:0 4px 4px 0}.bars .b i.neg{border-radius:4px 0 0 4px}.bars .b u{position:absolute;left:50%;top:-2px;bottom:-2px;border-left:1px solid var(--axis)}
.grid2 td{text-align:center;border:2px solid var(--surface);padding:5px 6px;min-width:46px}.grid2 td.best{outline:2px solid var(--ink);outline-offset:-2px}
.note{color:var(--ink2);font-size:13px;margin:6px 0 0}.mut{color:var(--muted)}
"""


def write_text(text: str, path) -> Path:
    """Write the page as UTF-8 text and return the path."""
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    return p


def _thin(ticks: list[float], gap: float = 0.12) -> list[float]:
    """Keep ticks at least `gap` of the range apart, so that labels do not run into each other."""
    if len(ticks) < 3:
        return ticks
    span = max(ticks) - min(ticks)
    out = [ticks[0]]
    for t in ticks[1:]:
        if t - out[-1] >= gap * span:
            out.append(t)
    return out


def _ok_(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _pv(p) -> str:
    return "n/a" if p is None or not math.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def _rd(kind: str) -> str:
    return {"ok": '<span class="rd">✓</span>', "w": '<span class="rd">!</span>', "bad": '<span class="rd">✕</span>'}[kind]


def _row(label: str, value: str, kind: str, reading: str) -> str:
    return f'<tr><td>{_esc(label)}</td><td class="n">{value}</td><td>{_rd(kind)}{_esc(reading)}</td></tr>'


def _hist(cid: str, caption: str, values, marker: float, marker_label: str, *, xfmt=lambda v: f"{v:.2f}", bins: int = 24, width: int = 760, height: int = 200) -> str:
    """A histogram of the null with the real value as a vertical line. Bars are muted; the line carries the colour."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = float(min(v.min(), marker)), float(max(v.max(), marker))
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    cnt, edges = np.histogram(v, bins=bins, range=(lo, hi))
    L, R, T, B = 40, 16, 12, 28
    W, H = width, height
    px = lambda x: L + (x - lo) / (hi - lo) * (W - L - R)                    # noqa: E731
    mx = max(1, int(cnt.max()))
    py = lambda c: T + (H - T - B) * (1 - c / mx)                            # noqa: E731
    bars = []
    for c, a, b in zip(cnt, edges[:-1], edges[1:]):
        if c:
            x0, x1 = px(a) + 1, px(b) - 1
            bars.append(f'<rect x="{x0:.1f}" y="{py(c):.1f}" width="{max(1.0, x1 - x0):.1f}" height="{py(0) - py(c):.1f}" rx="3" style="fill:var(--muted);fill-opacity:.55">'
                        f'<title>{xfmt(a)} to {xfmt(b)}: {int(c)} of {len(v)}</title></rect>')
    ticks = "".join(f'<text x="{px(t):.1f}" y="{H - 8}" text-anchor="middle">{_esc(xfmt(t))}</text>' for t in np.linspace(lo, hi, 6))
    mk = px(marker)
    anchor = "end" if mk > W * 0.7 else "start"
    off = -6 if anchor == "end" else 6
    return (f'<figure class="chart" id="{_esc(cid)}"><figcaption>{_esc(caption)}</figcaption>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(caption)}"><line class="axis" x1="{L}" x2="{W - R}" y1="{py(0):.1f}" y2="{py(0):.1f}"/>{"".join(bars)}'
            f'<line x1="{mk:.1f}" x2="{mk:.1f}" y1="{T}" y2="{py(0):.1f}" style="stroke:var(--s2);stroke-width:2"/>'
            f'<text x="{mk + off:.1f}" y="{T + 10}" text-anchor="{anchor}" style="fill:var(--ink);font-weight:600">{_esc(marker_label)}</text>{ticks}</svg></figure>')


def _barchart(cid: str, caption: str, labels, values, *, refs=(), yfmt=lambda v: f"{v:.0f}", ymin: float | None = None, ymax: float | None = None, width: int = 760, height: int = 220,
              tips=None) -> str:
    """Columns from a baseline, at most 24px wide, 4px rounded at the data end, the value above each one; `refs` are horizontal reference lines (value, label)."""
    vals = np.asarray(values, dtype=float)
    lo = min(0.0, float(np.nanmin(vals))) if ymin is None else ymin
    hi = float(np.nanmax(vals)) * 1.12 if ymax is None else ymax
    for r, _ in refs:
        hi = max(hi, r * 1.05)
        lo = min(lo, r)
    if hi - lo < 1e-12:
        hi = lo + 1
    L, R, T, B = 48, 16, 14, 28
    W, H = width, height
    py = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))              # noqa: E731
    n = len(vals)
    slot = (W - L - R) / max(1, n)
    bw = min(24.0, slot * 0.7)
    g, b = [], []
    from .report import _nice_ticks
    for t in _nice_ticks(lo, hi, 4):
        g.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{py(t):.1f}" y2="{py(t):.1f}"/><text x="{L - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{_esc(yfmt(t))}</text>')
    for i, (lab, v) in enumerate(zip(labels, vals)):
        if not np.isfinite(v):
            continue
        cx = L + slot * (i + 0.5)
        top, base = py(max(v, lo)), py(max(lo, 0.0) if lo < 0 < hi or lo >= 0 else 0.0)
        y0, y1 = (min(top, base), max(top, base))
        tip = (tips[i] if tips else f"{lab}: {yfmt(v)}")
        b.append(f'<rect x="{cx - bw / 2:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(1.0, y1 - y0):.1f}" rx="4" style="fill:var(--s1)"><title>{_esc(tip)}</title></rect>'
                 f'<text x="{cx:.1f}" y="{y0 - 4:.1f}" text-anchor="middle" style="fill:var(--ink2)">{_esc(yfmt(v))}</text><text x="{cx:.1f}" y="{H - 8}" text-anchor="middle">{_esc(str(lab))}</text>')
    rl = "".join(f'<line x1="{L}" x2="{W - R}" y1="{py(r):.1f}" y2="{py(r):.1f}" style="stroke:{"var(--s2)" if k == 0 else "var(--muted)"};stroke-width:1.5"/>'
                  f'<text x="{(W - R) if k == 0 else (L + 4)}" y="{py(r) - 4 if k == 0 else py(r) + 13:.1f}" text-anchor="{"end" if k == 0 else "start"}" style="fill:var(--ink2)">{_esc(lab)}</text>' for k, (r, lab) in enumerate(refs))
    return (f'<figure class="chart" id="{_esc(cid)}"><figcaption>{_esc(caption)}</figcaption><svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(caption)}">'
            f'{"".join(g)}{"".join(b)}{rl}</svg></figure>')


def _heat_grid(tbl: pd.DataFrame, best_at) -> str:
    v = tbl.astype(float).to_numpy()
    fin = v[np.isfinite(v)]
    top = float(np.max(np.abs(fin))) if len(fin) else 1.0
    head = "<tr><th>" + _esc(str(tbl.index.name or "")) + " \\ " + _esc(str(tbl.columns.name or "")) + "</th>" + "".join(f"<th>{_esc(str(c))}</th>" for c in tbl.columns) + "</tr>"
    rows = []
    for i, r in enumerate(tbl.index):
        cells = []
        for j, c in enumerate(tbl.columns):
            x = v[i, j]
            if not np.isfinite(x):
                cells.append("<td></td>")
                continue
            arm, a = ("pos", x / top) if x >= 0 else ("neg", -x / top)
            bg = f"color-mix(in srgb, var(--{arm}) {min(55.0, 8.0 + 47.0 * a):.0f}%, var(--mid))"
            best = " best" if best_at == (r, c) else ""
            cells.append(f'<td class="{best.strip()}" style="background:{bg}" title="{_esc(str(r))}, {_esc(str(c))}: Sharpe {_unsigned_zero(f"{x:.2f}")}">{_unsigned_zero(f"{x:.2f}")}</td>')
        rows.append(f"<tr><td>{_esc(str(r))}</td>{''.join(cells)}</tr>")
    return f'<div class="wrap"><table class="grid2">{head}{"".join(rows)}</table></div>'


def _contrib_bars(dec: dict) -> str:
    names = list(dec["factors"])
    vals = [dec["factors"][k]["contribution_annual"] for k in names] + [dec["alpha_annual"]]
    labs = names + ["alpha (unexplained)"]
    top = max(1e-9, max(abs(x) for x in vals))
    rows = []
    for lab, x in zip(labs, vals):
        w = abs(x) / top * 48
        side = f"left:50%;width:{w:.1f}%" if x >= 0 else f"left:{50 - w:.1f}%;width:{w:.1f}%"
        cls = "" if x >= 0 else "neg"
        col = "var(--s1)" if lab != "alpha (unexplained)" else "var(--s2)"
        rows.append(f'<span>{_esc(lab)}</span><span class="b"><u></u><i class="{cls}" style="{side};background:{col}"></i></span><span>{_pct(x, 1)}</span>')
    return f'<div class="bars">{"".join(rows)}</div>'


# ----------------------------------------------------------------------------------------------------------------------------------------- portfolio review
def _portfolio_parts(rv):
    items, luck, over_time = [], [], []
    m, t = rv.result.metrics, rv.mean
    # --- luck
    rows = [
        _row("Mean return is not zero (Newey-West t)", f"t = {_num(t['nw_t'])}, p = {_pv(t['nw_p'])}", "ok" if t["nw_t"] >= 2.5 else ("bad" if t["nw_t"] <= -2.5 else "w"),
             "above the 2.5 bar" if t["nw_t"] >= 2.5 else ("significantly NEGATIVE: the mean return is below zero beyond what chance explains" if t["nw_t"] <= -2.5
                                                           else "within 2.5 of zero: no evidence that the mean differs from zero")),
        _row("Sharpe t-statistic (Sharpe x sqrt(years))", _num(t["sharpe_t"]), "ok" if _ok_(t["sharpe_t"]) and t["sharpe_t"] >= 3.0 else ("bad" if _ok_(t["sharpe_t"]) and t["sharpe_t"] <= -3.0 else "w"),
             "at or above 3, the bar Harvey, Liu and Zhu suggest for a new factor" if _ok_(t["sharpe_t"]) and t["sharpe_t"] >= 3.0 else
             ("at or below -3: significantly negative" if _ok_(t["sharpe_t"]) and t["sharpe_t"] <= -3.0 else "between -3 and 3, the bar Harvey, Liu and Zhu suggest for a new factor")),
        _row("Bars with a positive return (zero bars left out)", f"{_pct(t['win_rate'])} (95% {_pct(t['win_ci'][0])} to {_pct(t['win_ci'][1])})",
             "ok" if t["win_ci"][0] > 0.5 else ("bad" if t["win_ci"][1] < 0.5 else "w"),
             f"sign test p = {_pv(t['sign_p'])}" + (f"; {t['n_zero']} bars with exactly zero return" if t.get("n_zero") else "") +
             ("; fewer than half the bars won" if t["win_ci"][1] < 0.5 else "")),
        _row("Fat tails", f"skew {_num(t['skew'])}, kurtosis {_num(t['kurtosis'])}", "w" if t["kurtosis"] > 5 else "ok", "kurtosis 3 is normal; fat tails make the t-values above too optimistic" if t["kurtosis"] > 5 else "close to normal"),
    ]
    if rv.sharpe_ci is not None and all(math.isfinite(rv.sharpe_ci[k]) for k in ("lo", "hi")):
        _lo, _hi = rv.sharpe_ci["lo"], rv.sharpe_ci["hi"]
        rows.append(_row("Sharpe, bootstrap interval", f"{_num(rv.sharpe_ci['sharpe'])} ({_num(_lo)} to {_num(_hi)})", "ok" if _lo > 0 else ("bad" if _hi < 0 else "w"),
                         "the interval is above 0" if _lo > 0 else ("the interval is entirely below 0" if _hi < 0 else "the interval includes 0")))
    perm_html = ""
    if rv.permutation is not None:
        pm = rv.permutation
        beat = int(round((pm["p"] * (pm["n"] + 1)) - 1))
        ok = pm["p"] <= 0.05
        rows.append(_row("Each security given another's factor history, rerun (before costs)", f"real gross Sharpe {_num(pm['real_sharpe'])} vs relabelled mean {_num(pm['null_mean'])}, 95th {_num(pm['null_p95'])}", "ok" if ok else "w",
                         f"{beat} of {pm['n']} relabellings did as well (p = {_pv(pm['p'])})"))
        perm_html = _hist("perm", f"Gross Sharpe of the same backtest with each security given another security's factor history ({pm['n']} times, before costs); the line is the real strategy", pm["null"], pm["real_sharpe"], f"real {_num(pm['real_sharpe'])}")
        items.append(("ok" if ok else "w", "Against relabelled factors", f"{beat} of {pm['n']} relabellings of the factor matched or beat the real gross Sharpe ratio (p = {_pv(pm['p'])}); the comparison is before costs, which are judged separately."))
    if rv.deflated is not None:
        d = rv.deflated
        ok = math.isfinite(d["dsr"]) and d["dsr"] >= 0.95
        src = "settings tried in the grid" if d.get("source") == "grid" else "trials in the ledger"
        rows.append(_row("Deflated Sharpe", f"{d['trials']} {src}", "ok" if ok else "w", f"probability the best beats luck: {_pct(d['dsr'], 0)}; best {d.get('best_name', '')}, luck level {_num(d['sharpe_luck_benchmark'])}"))
        items.append(("ok" if ok else "w", "Deflated Sharpe", f"{d['trials']} {src}; the best has Sharpe {_num(d['sharpe'])} and the best of {d['trials']} random strategies would show about "
                      f"{_num(d['sharpe_luck_benchmark'])}: probability it beats luck {_pct(d['dsr'], 0)}."))
    if rv.pbo is not None:
        pb = rv.pbo
        rows.append(_row("Probability of backtest overfitting (CSCV)", _pct(pb["pbo"], 0), "ok" if pb["pbo"] < 0.3 else "w",
                         f"in {pb['splits']:,} splits of the history, the setting that was best in one half ranked in the lower half of the other half in {_pct(pb['pbo'], 0)} of them; "
                         f"best-in-sample Sharpe {_num(pb['is_best_sharpe_mean'])} became {_num(pb['oos_of_best_sharpe_mean'])} out of sample. This is about whether the ranking of the settings persists, not whether they make money"))
        items.append(("ok" if pb["pbo"] < 0.3 else "w", "Overfitting probability", f"{_pct(pb['pbo'], 0)}: how often the best of the {rv.deflated['trials']} settings in one half of the history falls below the median in the other half."))
    luck.append('<h2>Is it luck?</h2><div class="card wrap"><table class="t"><tr><th>Test</th><th>Result</th><th>Reading</th></tr>' + "".join(rows) + "</table>" + perm_html +
                '<p class="note">Newey-West and sign tests treat the bars as the unit; a permutation asks the question that matters for a ranking: does the ranking carry information.</p></div>')
    # --- market
    if rv.market is not None:
        k = rv.market
        mrows = [
            _row("Excess CAGR", _pct(k["excess_cagr"]), "ok" if k["excess_cagr"] > 0 else "w", f"strategy {_pct(k['strategy_cagr'])} against benchmark {_pct(k['benchmark_cagr'])}"),
            _row("Months it beat the benchmark", _pct(k["hit_rate_month"], 0), "ok" if k["hit_rate_month"] > 0.5 else "w", f"years: {_pct(k['hit_rate_year'], 0)}"),
            _row("Up capture / down capture", f"{_num(k['up_capture'])} / {_num(k['down_capture'])}", "ok" if (k["up_capture"] > k["down_capture"]) else "w", "mean return on the benchmark's up days and down days relative to the benchmark's"),
            _row("Beta, correlation", f"{_num(k['beta'])}, {_num(k['correlation'])}", "ok", "market exposure of the daily returns"),
            _row("Information ratio, tracking error", f"{_num(k['information_ratio'])}, {_pct(k['tracking_error'])}", "ok" if (k["information_ratio"] or 0) > 0 else "w", "annualised mean difference over its standard deviation; its standard deviation"),
            _row("Worst fall against the benchmark", _pct(k["max_relative_drawdown"]), "ok" if k["max_relative_drawdown"] > -0.3 else "w", "of the ratio of the two equity curves"),
        ]
        luck.append('<h2>Against the market</h2><div class="card wrap"><table class="t"><tr><th>Measure</th><th>Result</th><th>Reading</th></tr>' + "".join(mrows) + "</table></div>")
    # --- decomposition
    if rv.decomposition is not None:
        d = rv.decomposition
        drows = "".join(f"<tr><td>{_esc(n)}</td><td>{_num(f['beta'])}</td><td>{_num(f['t'])}</td><td>{_pct(f['contribution_annual'])}</td></tr>" for n, f in d["factors"].items())
        luck.append('<h2>What explains the return?</h2><div class="card wrap"><table><tr><th>Factor</th><th>Beta</th><th>t</th><th>Return from it, per year</th></tr>' + drows +
                    f'<tr><td><b>Left over (alpha)</b></td><td></td><td>{_num(d["alpha_t"])}</td><td>{_pct(d["alpha_annual"])}</td></tr></table>'
                    + _contrib_bars(d) + f'<p class="note">R-squared {_num(d["r2"])}: {_pct(1 - d["unexplained_share"], 0)} of the variance is explained by these factors. These are the panel\'s own simple factors '
                    "(long-short top and bottom 30 percent, equal weight, before costs), not published factor returns.</p></div>")
        a_ok = d["alpha_t"] >= 2.5
        items.append(("ok" if a_ok else "w", "Alpha after the panel's own factors", f"{_pct(d['alpha_annual'])} a year left over, t = {_num(d['alpha_t'])}: " +
                      ("above the 2.5 bar" if a_ok else "significantly negative" if d["alpha_t"] <= -2.5 else "below the 2.5 bar")))
    # --- attribution
    attr = []
    if rv.attribution is not None:
        a = rv.attribution
        nm = a["names"]
        wfa = a["waterfall"]
        trow = "".join(f"<tr><td>{_esc(k)}</td><td>{_pct(v, 2)}</td></tr>" for k, v in list(nm["top"].items())[:5])
        brow = "".join(f"<tr><td>{_esc(k)}</td><td>{_pct(v, 2)}</td></tr>" for k, v in list(nm["bottom"].items())[:5])
        attr.append('<h2>Where did the return come from?</h2><div class="card wrap"><table class="t"><tr><th>Piece</th><th>Per year, per unit of capital in each leg</th><th>Reading</th></tr>'
                    + _row("Gross return (before every cost)", _pct(a["gross_annual"]), "ok", "mean gross return per bar x bars a year")
                    + _row("Long positions", _pct(a["long_annual"]), "ok" if a["long_annual"] > 0 else "w", "contribution of the securities held long")
                    + _row("Short positions", _pct(a["short_annual"]), "ok" if a["short_annual"] > 0 else "w", "contribution of the securities held short (positive when they fell)")
                    + _row("Securities with a positive total", f"{nm['n_positive']} of {nm['n_held']}", "ok" if nm["n_positive"] >= 0.5 * nm["n_held"] else "w", "")
                    + _row("The 10 best securities' share of the positive contributions", _pct(nm["top_share"], 0), "ok" if nm["top_share"] < 0.5 else "w",
                           "a return that rests on a few names is fragile" if nm["top_share"] >= 0.5 else "spread over many names")
                    + _row("The five best bars' share of the total", _pct(nm["best_days_share"], 0), "ok" if (nm["best_days_share"] or 1) < 0.5 else "w",
                           "a return that rests on a few days is fragile" if (nm["best_days_share"] or 0) >= 0.5 else "not dependent on a few days") + "</table>"
                    f'<div class="two"><div class="wrap"><table><tr><th>Best five securities</th><th>per year</th></tr>{trow}</table></div>'
                    f'<div class="wrap"><table><tr><th>Worst five securities</th><th>per year</th></tr>{brow}</table></div></div>')
        for key, ttl_ in (("by_size", "By market capitalisation"), ("by_liquidity", "By liquidity"), ("by_group", "By group")):
            if key in a:
                rws = "".join(f"<tr><td>{_esc(k)}</td><td>{_pct(v['contribution_annual'], 2)}</td><td>{_pct(v['exposure_share'], 0)}</td></tr>" for k, v in a[key].items())
                attr.append(f'<div class="wrap" style="margin-top:10px"><table><tr><th>{ttl_}</th><th>Return per year</th><th>Share of the exposure</th></tr>{rws}</table></div>')
        cw = "".join(f"<tr><td>minus {_esc(lb)}</td><td>{-v:+,.0f} bp</td></tr>" for lb, v in wfa["costs"])
        attr.append(f'<div class="wrap" style="margin-top:10px"><table><tr><th>From gross to net (per year)</th><th>basis points</th></tr><tr><td>Gross return</td><td>{wfa["gross_bp"]:+,.0f} bp</td></tr>{cw}'
                    + (f'<tr><td>other (a freeze markdown, a ruin)</td><td>{wfa["residual_bp"]:+,.0f} bp</td></tr>' if abs(wfa["residual_bp"]) >= 0.5 else "")
                    + f'<tr><td><b>Net return</b></td><td><b>{wfa["net_bp"]:+,.0f} bp</b></td></tr></table></div></div>')
    # --- regimes, risk, what it held, Brinson
    extra = []
    if rv.regimes is not None:
        rg = rv.regimes
        rrows = "".join(f"<tr><td>{_esc(i)}</td><td>{int(r.bars):,} ({_pct(r.share, 0)})</td><td>{_pct(r.mean_annual)}</td><td>{_num(r.sharpe)}</td><td>{_pct(r.win_rate, 0)}</td></tr>" for i, r in zip(rg["table"].index, rg["table"].itertuples()))
        extra.append('<h2>In which markets did it work?</h2><div class="card wrap"><table><tr><th>Regime (set by the benchmark alone)</th><th>Bars</th><th>Mean return per year</th><th>Sharpe</th><th>Bars up</th></tr>'
                     + rrows + "</table>" + f'<p class="note">The benchmark\'s worst fall in the sample was {_pct(rg["max_benchmark_drawdown"])}. ' +
                     ("The sample contains a bear market." if rg["sample_has_bear"] else "The sample contains **no** bear market: nothing here says how the strategy behaves in one.").replace("**", "") + "</p></div>")
        if not rg["sample_has_bear"]:
            items.append(("w", "No bear market", f"the benchmark never fell {rg['bear']:.0%} below its peak in this sample (worst {_pct(rg['max_benchmark_drawdown'])}): the strategy has not been tested in one."))
    if rv.risk is not None:
        k = rv.risk
        dd = rv.drawdown_dist
        rrows2 = (_row("Value at risk, 95% / 99% (historical)", f"{_pct(k['var_95'], 2)} / {_pct(k['var_99'], 2)}", "ok", "the return that 5 and 1 percent of bars fell below")
                  + _row("Expected shortfall, 95% / 99%", f"{_pct(k['cvar_95'], 2)} / {_pct(k['cvar_99'], 2)}", "ok", "the mean return of the bars below the value at risk")
                  + _row("Value at risk at today's volatility, 95% / 99%", f"{_pct(k['var_95_now'], 2)} / {_pct(k['var_99_now'], 2)}", "ok", f"the same shape of returns scaled to the current volatility ({_pct(k['vol_now'])} a year)")
                  + _row("Normal-distribution value at risk, 95%", _pct(k["var_95_normal"], 2), "w" if k["var_95_normal"] > k["var_95"] + 0.002 else "ok", "what a normal distribution would give: the historical tail is fatter" if k["var_95_normal"] > k["var_95"] + 0.002 else "close to the historical figure")
                  + _row("Worst bar, worst month", f"{_pct(k['worst_bar'], 1)}, {_pct(k['worst_month'], 1)}", "ok", f"best bar {_pct(k['best_bar'], 1)}, best month {_pct(k['best_month'], 1)}; tail ratio {_num(k['tail_ratio'])}")
                  + _row("Longest time below a previous peak", f"{k['max_dd_bars']:,} bars", "ok", f"{_pct(k['share_underwater'], 0)} of the bars were spent below a peak; ulcer index {_pct(k['ulcer_index'], 1)}")
                  + _row("Skew, kurtosis", f"{_num(k['skew'])}, {_num(k['kurtosis'])}", "w" if k["skew"] < -0.5 else "ok", "negative skew: more large losses than large gains" if k["skew"] < -0.5 else "3 is normal kurtosis"))
        ddhtml = ""
        if dd is not None:
            lab = ["5th percentile", "25th", "median", "75th", "95th"]
            vals = [dd["p05"], dd["p25"], dd["p50"], dd["p75"], dd["p95"]]
            ddhtml = _barchart("ddd", f"Maximum drawdown of {dd['n']:,} resampled histories of the same returns (the line is the realised one)", lab, vals, refs=[(dd["realized"], f"realised {_pct(dd['realized'], 0)}")],
                               yfmt=lambda v: f"{v * 100:.0f}%", ymin=min(min(vals), dd["realized"]) * 1.15, ymax=0.0,
                               tips=[f"{a}: {_pct(v, 1)}" for a, v in zip(lab, vals)])
            ddhtml += (f'<p class="note">{_pct(dd["p_worse"], 0)} of the resampled histories fell deeper than the realised {_pct(dd["realized"], 1)}; {_pct(dd["p_exceed_20"], 0)} fell more than 20 percent, '
                       f'{_pct(dd["p_exceed_30"], 0)} more than 30 and {_pct(dd["p_exceed_50"], 0)} more than 50. The realised drawdown is one draw.</p>')
            if dd["p_worse"] >= 0.75:
                items.append(("w", "A kind history", f"{_pct(dd['p_worse'], 0)} of resampled histories of the same returns fell deeper than the realised {_pct(dd['realized'], 1)}; the median was {_pct(dd['p50'], 1)}."))
        extra.append('<h2>How bad can it get?</h2><div class="card wrap"><table class="t"><tr><th>Measure</th><th>Result</th><th>Reading</th></tr>' + rrows2 + "</table>" + ddhtml + "</div>")
    # --- over time
    s_ = rv.subperiods
    ot = [f'<h2>Does it hold over time?</h2><div class="card wrap"><table class="t"><tr><th>Measure</th><th>Result</th><th>Reading</th></tr>'
          + _row("Years with a positive return", _pct(s_["share_positive"], 0), "ok" if s_["share_positive"] >= 0.6 else "w", f"worst year {_pct(s_['worst_year'])}, best {_pct(s_['best_year'])}")
          + _row("Sharpe, first half / second half", f"{_num(s_['halves'][0])} / {_num(s_['halves'][1])}", "ok" if min(s_["halves"]) > 0 else "w",
                 "the edge shows in both halves" if min(s_["halves"]) > 0 else "the edge is in one half only: a regime, not a property")]
    if rv.walk_forward is not None:
        w = rv.walk_forward
        eff_ok = math.isfinite(w["efficiency"]) and w["efficiency"] >= 0.5 and w["oos_sharpe"] > 0
        ot.append(_row("Walk-forward of the choice of setting", f"out-of-sample Sharpe {_num(w['oos_sharpe'])} vs in-sample {_num(w['mean_is_sharpe'])}", "ok" if eff_ok else "w",
                       f"efficiency {_num(w['efficiency'])} over {w['n_folds']} folds; {_pct(w['share_folds_positive'], 0)} of folds positive"))
        items.append(("ok" if eff_ok else "w", "Walk-forward", f"choosing the best setting on the past and running it on the next stretch gave an out-of-sample Sharpe of {_num(w['oos_sharpe'])} "
                      f"against {_num(w['mean_is_sharpe'])} in-sample (efficiency {_num(w['efficiency'])})."))
    ot.append("</table>")
    if rv.walk_forward is not None:
        oos = rv.walk_forward["oos"]
        oi = naive_index(oos.index)
        xs = np.array([t.value // 10 ** 6 for t in oi], dtype=np.float64)
        ticks = _year_ticks(oi[0], oi[-1]) if (oi[-1] - oi[0]).days > 700 else None
        ot.append(_chart("wf", "Walk-forward: the stitched out-of-sample growth of 1", xs, [{"name": "out-of-sample", "y": (1 + oos.to_numpy()).cumprod(), "color": "var(--s1)", "fmt": "eq"}],
                         yfmt="num", xfmt=_date_fmt, height=200, xticks=ticks, xtickfmt=_year_fmt if ticks else None))
        fd = rv.walk_forward["folds"]
        ot.append('<details><summary>Folds</summary><div class="wrap"><table><tr><th>Test period</th><th>Chosen setting</th><th>In-sample Sharpe</th><th>Out-of-sample Sharpe</th></tr>' +
                  "".join(f"<tr><td>{r.test_start.date()} to {r.test_end.date()}</td><td>{_esc(str(r.chosen))}</td><td>{_num(r.is_sharpe)}</td><td>{_num(r.oos_sharpe)}</td></tr>" for r in fd.itertuples()) + "</table></div></details>")
    ot.append("</div>")
    settings = []
    if rv.grid_table is not None:
        pl = rv.plateau
        verdict = {"plateau": ("ok", "the neighbours of the best setting keep most of its Sharpe: a plateau"), "hill": ("w", "the neighbours keep part of it: a hill"),
                   "spike": ("w", "the neighbours lose almost all of it: a spike, the sign of a search rather than a property"), "undetermined": ("w", (pl["reason"] if pl else "") + ", so there is nothing to call a plateau or a spike")}[pl["verdict"]] if pl else ("w", "")
        gt_ = rv.grid_table
        if isinstance(gt_, pd.Series):                                                       # one parameter: a single row, the parameter named on the left
            show = gt_.to_frame().T.rename(index={gt_.name: str(gt_.index.name)}).rename_axis(columns=None)
            best_cell = (show.index[0], pl["best_at"][0]) if pl else None
        else:
            show, best_cell = gt_, (pl["best_at"] if pl else None)
        settings.append('<h2>Is the setting a spike?</h2><div class="card">' + _heat_grid(show, best_cell) +
                        (f'<p class="note">{_rd(verdict[0])}Best setting {_esc(", ".join(str(x) for x in pl["best_at"]))} (Sharpe {_num(pl["best"])}); its {pl["n_neighbours"]} neighbours have an average Sharpe of {_num(pl["neighbour_mean"])}. {_esc(verdict[1])}. '
                         f'{_pct(pl["grid_positive"], 0)} of all settings have a positive Sharpe.</p>' if pl else "") + "</div>")
        if pl:
            items.append(verdict[:1] + ("Setting", f"the best setting has Sharpe {_num(pl['best'])} and its neighbours average {_num(pl['neighbour_mean'])} ({pl['verdict']})."))
    if rv.cost_table is not None:
        ct = rv.cost_table
        rowsc = "".join(f"<tr><td>{c:.0f} bp</td><td>{_num(r.sharpe)}</td><td>{_pct(r.cagr)}</td></tr>" for c, r in zip(ct.index, ct.itertuples()))
        pos = [c for c, r in zip(ct.index, ct.itertuples()) if math.isfinite(r.sharpe) and r.sharpe > 0]
        settings.append('<h2>How much cost can it take?</h2><div class="card wrap"><table><tr><th>Round-trip spread</th><th>Sharpe</th><th>CAGR</th></tr>' + rowsc + "</table>"
                        f'<p class="note">The Sharpe ratio stays above 0 up to {max(pos):.0f} bp of the spreads tried.</p></div>' if pos else
                        '<h2>How much cost can it take?</h2><div class="card wrap"><table><tr><th>Round-trip spread</th><th>Sharpe</th><th>CAGR</th></tr>' + rowsc +
                        '</table><p class="note">The Sharpe ratio is not above 0 at any of the spreads tried.</p></div>')
    hold = []
    if rv.holdings is not None:
        h = rv.holdings
        trows = "".join(f"<tr><td>{_esc(k_)}</td><td>{_pct(v, 2)}</td></tr>" for k_, v in h["top_holdings"].items())
        hold.append('<h2>What it held, and how much it traded</h2><div class="card wrap"><table class="t"><tr><th>Measure</th><th>Result</th><th>Reading</th></tr>'
                    + _row("Positions held, long / short", f"{_num(h['avg_long'], 0)} / {_num(h['avg_short'], 0)}", "ok", "mean number of securities on bars with a position")
                    + _row("Effective number of equal positions, long / short", f"{_num(h['effective_n_long'], 1)} / {_num(h['effective_n_short'], 1)}", "ok" if h["effective_n_long"] >= 0.5 * max(h["avg_long"], 1) else "w",
                           "1 over the sum of squared weights: far below the count means a few names dominate")
                    + _row("Largest weight in a leg (mean)", f"{_pct(h['max_weight_long'], 0)} / {_pct(h['max_weight_short'], 0)}", "ok", "as a share of the leg")
                    + _row("Turnover, per bar / per 21 bars", f"{_pct(h['turnover_per_bar'], 1)} / {_pct(h['turnover_per_month'], 0)}", "w" if h["turnover_per_month"] > 1.0 else "ok",
                           "one-way share of the book; above 100 percent a month the costs dominate the question" if h["turnover_per_month"] > 1.0 else "one-way share of the book")
                    + _row("Trades per year, mean trade", f"{h['trades_per_year']:,.0f}, {_pct(h['avg_trade'], 2)}", "ok", "security-bars on which the position changed; the mean size of the change")
                    + _row("Implied holding period", f"{_num(h['implied_holding_bars'], 1)} bars", "ok", "average gross exposure over one-way turnover")
                    + f'</table><div class="wrap" style="margin-top:10px"><table><tr><th>Largest average weights</th><th>weight</th></tr>{trows}</table></div></div>')
    bri = []
    if rv.brinson is not None:
        b = rv.brinson
        rows_b = "".join(f"<tr><td>{_esc(g)}</td><td>{_pct(b['allocation'][g], 2)}</td><td>{_pct(b['selection'][g], 2)}</td><td>{_pct(b['interaction'][g], 2)}</td></tr>" for g in b["groups"])
        tt = b["total"]
        bri.append('<h2>Allocation or selection?</h2><div class="card wrap"><table><tr><th>Group</th><th>Allocation</th><th>Selection</th><th>Interaction</th></tr>' + rows_b +
                   f'<tr><td><b>Total</b></td><td><b>{_pct(tt["allocation"], 2)}</b></td><td><b>{_pct(tt["selection"], 2)}</b></td><td><b>{_pct(tt["interaction"], 2)}</b></td></tr></table>'
                   f'<p class="note">Per year, gross of costs, long-only against the {b["benchmark"]}-weighted benchmark: the three add up to {_pct(b["excess_gross_annual"], 2)}, the portfolio\'s gross return minus the benchmark\'s. '
                   "Allocation: being over- or under-weight the right groups. Selection: picking the right securities inside a group.</p></div>")
    return items, "".join(luck) + "".join(extra) + "".join(attr) + "".join(hold) + "".join(bri), "".join(ot) + "".join(split_html_parts(rv, items)) + "".join(settings)


def split_html_parts(rv, items) -> list:
    """The frozen-date comparison, when a `split_date` was given."""
    if rv.split is None:
        return []
    sp = rv.split
    a, b = sp["in_sample"], sp["out_of_sample"]
    keep = math.isfinite(sp["sharpe_ratio"]) and sp["sharpe_ratio"] >= 0.5 and b["sharpe"] > 0
    rows = "".join(f"<tr><td>{lab}</td><td>{_esc(side['start'])} to {_esc(side['end'])}</td><td>{side['bars']:,}</td><td>{_pct(side['mean_annual'])}</td><td>{_pct(side['vol_annual'])}</td><td>{_num(side['sharpe'])}</td><td>{_pct(side['win_rate'], 0)}</td></tr>"
                   for lab, side in (("Before", a), ("After", b)))
    items.append(("ok" if keep else "w", "Frozen date", f"Sharpe {_num(a['sharpe'])} before {sp['split']} and {_num(b['sharpe'])} after it (ratio {_num(sp['sharpe_ratio'])}); the mean return changed with t = {_num(sp['mean_diff_t'])}."))
    return ['<h2>What happened after the frozen date?</h2><div class="card wrap"><table><tr><th></th><th>Period</th><th>Bars</th><th>Mean return per year</th><th>Volatility</th><th>Sharpe</th><th>Bars up</th></tr>' + rows + "</table>"
            f'<p class="note">Change in the mean after the date: t = {_num(sp["mean_diff_t"])}, p = {_pv(sp["mean_diff_p"])} (Welch, bars treated as independent). Out-of-sample over in-sample Sharpe: {_num(sp["sharpe_ratio"])}. '
            "This means something only if the date was fixed before the later results were seen.</p></div>"]


def _review_page(inner_open: str, inner: str, title: str) -> str:
    from ._version import __version__
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_esc(title)}</title><style>{_CSS}{_EXTRA_CSS}</style></head><body><main>{inner_open}{inner}"
            f"<p class=\"foot\">pitbacktest {__version__}. Self-contained page: no network access, no scripts beyond the hover tooltips. This page describes a simulation; it is not advice.</p>"
            f"</main><script>{_JS}</script></body></html>")


def review_html(rv, **kw) -> str:
    """The review page for a `PortfolioReview` or an `EventReview`."""
    if hasattr(rv, "horizons_table"):                                      # an EventReview (checked by its content: `review` imports this module, not the other way round)
        return _event_page(rv, **kw)
    items, after_curves, after_months = _portfolio_parts(rv)
    page = report_html(rv.result, title=kw.pop("title", "Strategy review"), extra_items=items, after_curves=after_curves, after_months=after_months, trials_item=rv.deflated is None, **kw)
    return page.replace("</style>", _EXTRA_CSS + "</style>", 1)


# ----------------------------------------------------------------------------------------------------------------------------------------- event review
def _event_page(rv, *, title: str | None = None, max_points: int = 900) -> str:
    res, ht, h = rv.result, rv.horizons_table, rv.horizon
    st = res.per_horizon[h]
    items = []
    for n in rv.notes:
        items.append(("w", "Note", n))
    lift_ok = st["lift_pp"] > 0
    lo, hi = float(ht.loc[h, "win_lo"]), float(ht.loc[h, "win_hi"])
    items.append(("ok" if lo > st["base_rate"] else "w", "Win rate against a random pick", f"{st['win_rate']:.1f}% of {st['n_trades']:,} trades won after the {rv.cost_bp:.0f} bp round-trip cost "
                  f"(95% {lo:.1f}% to {hi:.1f}%); a random pick on the same days wins {st['base_rate']:.1f}%, a lift of {st['lift_pp']:+.1f} points."
                  + ("" if lo > st["base_rate"] else " The interval includes the random-pick rate.")))
    if rv.permutation is not None:
        pm = rv.permutation
        ok = pm["p_win"] <= 0.05
        items.append(("ok" if ok else "w", "Against the signal moved in time", f"{pm['n']} copies of the signal, each moved in time by a random amount (how often and on which securities it fires are kept), won "
                      f"{pm['null_win_rate_mean'] * 100:.1f}% on average (95th percentile {pm['null_win_rate_p95'] * 100:.1f}%); the signal won {pm['real_win_rate'] * 100:.1f}% (p = {_pv(pm['p_win'])}). "
                      f"Mean return: signal {pm['real_mean_bp']:+.0f} bp, moved copies {pm['null_mean_bp_mean']:+.0f} bp (p = {_pv(pm['p_mean'])})."))
    if st["skew_warning"]:
        items.append(("w", "Mean and median disagree", "the mean trade is positive and the median negative: a few large winners carry many small losses. Fine for a portfolio, bad for an alert."))
    de = rv.daily_excess
    if math.isfinite(de["t"]):
        items.append(("ok" if abs(de["t"]) >= 2.5 and de["t"] > 0 else "w", "Day-level excess over a random pick", f"{de['mean_excess_bp']:+.0f} bp on {de['days']:,} days, Newey-West t = {_num(de['t'])} (lag {de['lag']}); trades on one day are correlated, so the day is the unit."))
    cls = "ok" if all(k == "ok" for k, _, _ in items) else ""
    icon = {"w": "!", "ok": "✓", "bad": "✕"}
    lis = "".join(f'<li class="{k}"><span class="ic">{icon[k]}</span><span class="lab">{_esc(t)}.</span> {_esc(x)}</li>' for k, t, x in items)
    read = f'<section class="card read {cls}"><div class="lab">Read this first: what limits these numbers</div><ul>{lis}</ul></section>'
    tiles = "".join([_tile("Win rate", f"{st['win_rate']:.1f}%", f"random pick {st['base_rate']:.1f}%, lift {st['lift_pp']:+.1f} points"),
                     _tile("Trades", f"{st['n_trades']:,}", f"{rv.streak['longest_losing_streak']} losing fire-days in a row at most"),
                     _tile("Mean trade", f"{st['mean_bp']:+.0f} bp", f"median {st['median_bp']:+.0f} bp, after {rv.cost_bp:.0f} bp"),
                     _tile("Payoff", _num(st["payoff"]), f"average win {st['avg_win_bp']:+.0f}, loss {st['avg_loss_bp']:+.0f} bp"),
                     _tile("Horizon", f"{h} bars", f"the best by lift of {len(ht)} tried" if len(ht) > 1 else "")])
    hrows = "".join(f"<tr><td>{i}</td><td>{int(r.n_trades):,}</td><td>{r.win_rate:.1f}% ({r.win_lo:.1f}-{r.win_hi:.1f})</td><td>{r.base_rate:.1f}%</td><td>{r.lift_pp:+.1f}</td>"
                     f"<td>{r.mean_bp:+.0f}</td><td>{r.median_bp:+.0f}</td><td>{_num(r.payoff)}</td></tr>" for i, r in zip(ht.index, ht.itertuples()))
    htab = ('<div class="card wrap"><table><tr><th>Hold (bars)</th><th>Trades</th><th>Win rate (95% interval)</th><th>Random pick</th><th>Lift (points)</th><th>Mean bp</th><th>Median bp</th><th>Payoff</th></tr>'
            + hrows + "</table></div>")
    ct = rv.cost_table
    cost_chart = _chart("cost", f"Win rate against the round-trip cost (holding {h} bars)", ct.index.to_numpy(float),
                        [{"name": "Win rate", "y": (ct["win_rate"] / 100).to_numpy(), "color": "var(--s1)", "fmt": "pct"},
                         {"name": "Random pick on the same days (at each cost)", "y": (ct["base_rate"] / 100).to_numpy(), "color": "var(--muted)", "fmt": "pct"}],
                        yfmt="pct", xfmt=lambda v: f"{v:.0f} bp", height=220, xticks=_thin([float(v) for v in ct.index]))
    crow = "".join(f"<tr><td>{c:.0f} bp</td><td>{r.win_rate:.1f}%</td><td>{r.base_rate:.1f}%</td><td>{r.lift_pp:+.1f}</td><td>{r.mean_bp:+.0f}</td><td>{r.median_bp:+.0f}</td></tr>" for c, r in zip(ct.index, ct.itertuples()))
    above = [c for c, r in zip(ct.index, ct.itertuples()) if r.lift_pp > 0.0]
    yr = rv.yearly
    ybar = _barchart("yearly", "Win rate by year (the line is 50 percent; the other is the random pick's rate over all years)", [str(i) for i in yr.index], yr["win_rate"].to_numpy(),
                     refs=[(st["base_rate"], f"random pick {st['base_rate']:.0f}%"), (50.0, "50%")], yfmt=lambda v: f"{v:.0f}%", ymin=0.0, ymax=100.0,
                     tips=[f"{i}: {r.win_rate:.1f}% of {int(r.n_trades):,} trades (95% {r.win_lo:.1f}-{r.win_hi:.1f}), mean {r.mean_bp:+.0f} bp" for i, r in zip(yr.index, yr.itertuples())])
    n_up = int((yr["win_rate"] > st["base_rate"]).sum())
    perm_html = ""
    if rv.permutation is not None:
        pm = rv.permutation
        perm_html = ('<h2>Is it luck?</h2><div class="card"><div class="wrap"><table class="t"><tr><th>Measure</th><th>Signal</th><th>The signal moved in time</th><th>p</th></tr>'
                     f"<tr><td>Win rate</td><td>{pm['real_win_rate'] * 100:.1f}%</td><td>{pm['null_win_rate_mean'] * 100:.1f}% (95th {pm['null_win_rate_p95'] * 100:.1f}%)</td><td>{_pv(pm['p_win'])}</td></tr>"
                     f"<tr><td>Mean trade</td><td>{pm['real_mean_bp']:+.0f} bp</td><td>{pm['null_mean_bp_mean']:+.0f} bp (95th {pm['null_mean_bp_p95']:+.0f})</td><td>{_pv(pm['p_mean'])}</td></tr></table></div>"
                     f'<p class="note">{pm["n"]} copies of the signal, each moved in time by a random amount: how often it fires, for how long and on which securities are kept, only the alignment with the returns is '
                     "destroyed. p counts the copies at least as good as the signal, plus one. A test against zero cannot answer this: in a rising market almost any pick has a positive mean.</p></div>")
    port = ""
    if rv.portfolio is not None:
        pr = rv.portfolio
        pm_ = pr.metrics
        s_ = pr.net_returns.copy()
        s_.index = naive_index(s_.index)
        eq = (1 + s_).cumprod()
        sers = [{"name": "Signal held as a portfolio (net)", "y": eq.to_numpy(), "color": "var(--s1)", "fmt": "eq"}]
        if pr.benchmark_returns is not None:
            b = pr.benchmark_returns.copy(); b.index = naive_index(b.index)
            sers.append({"name": "Benchmark (no costs)", "y": (1 + b.reindex(s_.index).fillna(0.0)).cumprod().to_numpy(), "color": "var(--muted)", "fmt": "eq"})
        xs = np.array([t.value // 10 ** 6 for t in s_.index], dtype=np.float64)
        tk = _year_ticks(s_.index[0], s_.index[-1]) if (s_.index[-1] - s_.index[0]).days > 700 else None
        mk = rv.portfolio_market
        mt = rv.portfolio_mean
        prow = (f"<tr><td>CAGR</td><td>{_pct(pm_.get('CAGR'))}</td></tr><tr><td>Sharpe</td><td>{_num(pm_.get('Sharpe'))}"
                + (f" (Newey-West t of the mean {_num(mt['nw_t'])})" if mt else "") + f"</td></tr><tr><td>Max drawdown</td><td>{_pct(pm_.get('MDD'))}</td></tr>"
                f"<tr><td>Volatility</td><td>{_pct(pm_.get('vol'))}</td></tr><tr><td>Turnover per bar</td><td>{_num(pm_.get('turnover_daily'), 3)}</td></tr>"
                + (f"<tr><td>Excess CAGR against the benchmark</td><td>{_pct(mk['excess_cagr'])}</td></tr><tr><td>Months it beat the benchmark</td><td>{_pct(mk['hit_rate_month'], 0)}</td></tr>"
                   f"<tr><td>Up / down capture, beta</td><td>{_num(mk['up_capture'])} / {_num(mk['down_capture'])}, {_num(mk['beta'])}</td></tr>" if mk else ""))
        port = ("<h2>The same signal held as a portfolio</h2><div class=\"card\">"
                + _chart("pf", f"Growth of 1: each date's fires share 1/{rv.hold} of the capital for {rv.hold} bars, cash on dates with no fire", xs, sers, yfmt="num", xfmt=_date_fmt, height=230, max_points=max_points,
                         xticks=tk, xtickfmt=_year_fmt if tk else None, logy=wide_range([sr["y"] for sr in sers]))
                + f'<div class="wrap"><table class="t">{prow}</table></div><p class="note">This puts an event signal on the same footing as a factor portfolio: a win rate says how often, '
                  "the curve says how much and how bumpy.</p></div>")
    stc = res.structure
    srow = (f"<tr><td>Fires per day (mean)</td><td>{_num(stc['fires_per_day'])}</td></tr><tr><td>Days with a fire</td><td>{_num(stc['active_days_pct'], 0)}%</td></tr>"
            f"<tr><td>Different securities</td><td>{int(stc['n_names']):,}</td></tr><tr><td>Share of fires in the 10 most frequent securities</td><td>{_pct(stc['top10_share'], 0)}</td></tr>"
            f"<tr><td>Most frequent security</td><td>{_pct(stc['max_name_share'], 1)} of all fires</td></tr><tr><td>Longest run of consecutive days with a fire</td><td>{int(stc['max_consecutive'])}</td></tr>")
    segs_html = ""
    for lab, tb in rv.segments.items():
        rws = "".join(f"<tr><td>{_esc(i)}</td><td>{int(r.n_trades):,} ({r.share_of_fires:.0f}%)</td><td>{r.win_rate:.1f}% ({r.win_lo:.1f}-{r.win_hi:.1f})</td><td>{r.mean_bp:+.0f}</td><td>{r.median_bp:+.0f}</td></tr>"
                      for i, r in zip(tb.index, tb.itertuples()))
        segs_html += f'<div class="wrap" style="margin-top:10px"><table><tr><th>By {_esc(lab)}</th><th>Trades</th><th>Win rate (95% interval)</th><th>Mean bp</th><th>Median bp</th></tr>{rws}</table></div>'
    if segs_html:
        segs_html = ('<h2>Where does it work?</h2><div class="card">' + segs_html + '<p class="note">Thirds are taken among the eligible securities on each signal date. '
                     "A win rate that holds only in the least liquid third is one that is hard to trade at size.</p></div>")
    gt = res.gates
    grow = "".join(f'<tr><td>{_esc(g["gate"])}</td><td class="n">{_rd("ok" if g["pass"] else "bad")}{"passed" if g["pass"] else "failed"}</td><td>{_esc(str(g["detail"]))}</td></tr>' for g in gt["steps"])
    gates = ('<h2>The library\'s six gates</h2><div class="card wrap"><table class="t"><tr><th>Gate</th><th>Result</th><th>Detail</th></tr>' + grow + "</table>"
             '<p class="note">The gates are a screen against noise, not a certificate: one pass is not evidence of an edge (see the limits in docs/verification_status.md).</p></div>')
    spec_rows = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in res.spec.items() if k != "decile")      # built here: quotes inside an f-string need Python 3.12
    ttl = title or "Event signal review"
    sub = f"{res.spec.get('market', '')} · cost {rv.cost_bp:.0f} bp round trip · {len(ht)} horizon(s) tried"
    body = (f"{read}<h2>Figures</h2><div class=\"tiles\">{tiles}</div><h2>By holding period</h2>{htab}"
            f"{perm_html}<h2>How much cost can it take?</h2><div class=\"card\">{cost_chart}<div class=\"wrap\"><table><tr><th>Round-trip cost</th><th>Win rate</th><th>Random pick</th><th>Lift (points)</th><th>Mean bp</th><th>Median bp</th></tr>{crow}</table></div>"
            f"<p class=\"note\">{'The win rate stays above a random pick on the same days up to ' + format(max(above), '.0f') + ' bp of the costs tried (a random pick also wins less as the cost rises).' if above else 'The win rate is not above a random pick on the same days at any of the costs tried.'}</p></div>"
            f"<h2>Year by year</h2><div class=\"card\">{ybar}<p class=\"note\">{n_up} of {len(yr)} years have a win rate above the random pick's {st['base_rate']:.0f}%. "
            f"Years with few trades have wide intervals (hover a bar).</p></div>{port}"
            f"{segs_html}{gates}<h2>What the signal picks</h2><div class=\"card wrap\"><table class=\"t\">{srow}</table></div>"
            f"<h2>How it was run</h2><div class=\"card\"><dl>{spec_rows}</dl>"
            "<p class=\"foot\" style=\"margin-bottom:0\">Win rates and trade returns are after the round-trip cost, over the holding period, from the close after the signal (entry lag as in the spec). "
            "The engine has no order book, queue or partial fills.</p></div>")
    return _review_page(f"<h1>{_esc(ttl)}</h1><p class=\"sub\">{_esc(sub)}</p>", body, ttl)


__all__ = ["review_html", "write_text"]
