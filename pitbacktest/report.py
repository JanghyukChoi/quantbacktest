"""A one-page HTML report of a backtest, for a person to read.

    r = q.backtest_portfolio(panel, factor, ...)
    r.report("report.html")                         # or: html_text = r.report()

The file is self-contained (no network, no script libraries, no extra install): the charts are inline SVG, they react to the mouse, and the colours follow the
reader's light or dark setting. It opens with what limits the number (the warnings the run raised, the Sharpe interval, the trial count when a ledger is given),
then the figures, the curves, the months and years, the costs, and how the run was set up. Nothing in it is a recommendation, and a strategy's Sharpe ratio is shown
together with how uncertain it is.

Figures come from the result: the same numbers as `result.metrics` and `result.net_returns`, nothing is re-estimated except the rolling Sharpe, the monthly and yearly
tables, and the bootstrap interval (seeded, so the same result gives the same page byte for byte)."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--pos:#2a78d6;--neg:#e34948;--mid:#f0efec;--warn:#fab219;--crit:#d03b3b;--good:#006300}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--pos:#3987e5;--neg:#e66767;--mid:#383835;--good:#0ca30c}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--pos:#3987e5;--neg:#e66767;--mid:#383835;--good:#0ca30c}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 56px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}
.sub{color:var(--ink2);margin:0 0 18px;font-size:14px}.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.read{border-left:4px solid var(--warn)}.read.ok{border-left-color:var(--muted)}.read.bad{border-left-color:var(--crit)}
.read ul{margin:6px 0 0;padding-left:0;list-style:none}.read li{margin:4px 0;padding-left:26px;position:relative;font-size:14px}
.read li .ic{position:absolute;left:0;top:0;font-weight:700;width:20px;text-align:center}.read li.w .ic{color:var(--ink)}.lab{font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 12px}
.tile .l{font-size:13px;color:var(--ink2)}.tile .v{font-size:24px;font-weight:600;line-height:1.25}.tile .d{font-size:12px;color:var(--ink2)}
figure.chart{margin:0;position:relative}figure.chart figcaption{font-size:13px;color:var(--ink2);margin:0 0 4px}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:2px 0 4px}
.legend i{display:inline-block;width:16px;height:0;border-top:2px solid;vertical-align:middle;margin-right:6px}
svg{width:100%;height:auto;display:block;touch-action:pan-y}svg text{fill:var(--muted);font:11px system-ui,sans-serif}svg .grid{stroke:var(--grid);stroke-width:1}svg .axis{stroke:var(--axis);stroke-width:1}
svg .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}svg .cross{stroke:var(--axis);stroke-width:1}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:6px 9px;font-size:12px;line-height:1.4;box-shadow:0 2px 8px rgba(0,0,0,.12);white-space:nowrap;display:none;z-index:2}
.tip b{font-weight:600}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:4px 8px;text-align:right;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:500}.heat td{border:2px solid var(--surface);padding:5px 2px;text-align:center;font-size:12px;min-width:44px}
.heat td.na{background:transparent}.heat td.y{color:var(--ink2)}.wrap{overflow-x:auto}.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:640px){.two{grid-template-columns:1fr}}dl{margin:0;display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;font-size:13px}dt{color:var(--ink2)}dd{margin:0;overflow-wrap:anywhere}
.badge{display:inline-block;border:1px solid var(--crit);color:var(--crit);border-radius:999px;padding:0 8px;font-size:12px;font-weight:600;margin-left:8px;vertical-align:middle}
details summary{cursor:pointer;color:var(--ink2);font-size:13px}.foot{color:var(--muted);font-size:12px;margin-top:28px}
"""

_JS = """
(function(){document.querySelectorAll('figure.chart').forEach(function(fig){
var d=JSON.parse(fig.querySelector('script.cd').textContent),svg=fig.querySelector('svg'),cross=svg.querySelector('.cross'),tip=fig.querySelector('.tip'),dots=[];
var NS='http://www.w3.org/2000/svg';
d.series.forEach(function(s,i){var c=document.createElementNS(NS,'circle');c.setAttribute('r',4);c.style.fill=s.color;c.style.stroke='var(--surface)';c.style.strokeWidth='2';c.style.display='none';svg.appendChild(c);dots.push(c);});
function fy(v){if(d.log)v=Math.log(v);return d.t+(d.h-d.t-d.b)*(1-(v-d.ymin)/(d.ymax-d.ymin));}
function fmt(k,v){if(v===null)return 'n/a';if(k==='pct')return (v*100).toFixed(2)+'%';if(k==='eq')return v.toFixed(3);return v.toFixed(2);}
function near(px){var lo=0,hi=d.xs.length-1;while(hi-lo>1){var m=(lo+hi)>>1;if(d.xs[m]<px)lo=m;else hi=m;}return Math.abs(d.xs[lo]-px)<=Math.abs(d.xs[hi]-px)?lo:hi;}
svg.addEventListener('pointermove',function(e){var r=svg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*d.w;if(x<d.l||x>d.w-d.r){hide();return;}
var xv=d.x0+(x-d.l)/(d.w-d.l-d.r)*(d.x1-d.x0),i=near(xv),px=d.l+(d.xs[i]-d.x0)/(d.x1-d.x0)*(d.w-d.l-d.r);
cross.setAttribute('x1',px);cross.setAttribute('x2',px);cross.style.display='';var h='<b>'+d.labels[i]+'</b>';
d.series.forEach(function(s,k){var v=s.y[i];h+='<br>'+s.name+': '+fmt(s.fmt,v);if(v===null){dots[k].style.display='none';}else{dots[k].setAttribute('cx',px);dots[k].setAttribute('cy',fy(v));dots[k].style.display='';}});
tip.innerHTML=h;tip.style.display='block';var fr=fig.getBoundingClientRect(),tx=e.clientX-fr.left+12;if(tx+tip.offsetWidth>fr.width)tx=e.clientX-fr.left-tip.offsetWidth-12;tip.style.left=Math.max(0,tx)+'px';tip.style.top=(e.clientY-fr.top+12)+'px';});
function hide(){cross.style.display='none';tip.style.display='none';dots.forEach(function(c){c.style.display='none';});}
svg.addEventListener('pointerleave',hide);});})();
"""


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _unsigned_zero(text: str) -> str:
    return text[1:] if text.startswith("-") and float(text.rstrip("%")) == 0 else text


def _pct(v, d: int = 1) -> str:
    return _unsigned_zero(f"{float(v) * 100:.{d}f}") + "%" if _ok(v) else "n/a"


def _num(v, d: int = 2) -> str:
    return _unsigned_zero(f"{float(v):.{d}f}") if _ok(v) else "n/a"


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-12 * step:
        out.append(round(v, 12))
        v += step
    return out


def _pick(ys: list[np.ndarray], k: int) -> np.ndarray:
    """Indices to draw when a series is longer than about k points: the first and last, an even spread, and in every stretch the lowest and the highest point of every
    series, so that thinning never hides the trough of a drawdown or a spike (a chart that skipped the worst day would disagree with the number printed above it)."""
    n = len(ys[0])
    if n <= k:
        return np.arange(n)
    keep = {0, n - 1, *np.linspace(0, n - 1, k // 2).round().astype(int).tolist()}
    edges = np.linspace(0, n, k // 4 + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        for y in ys:
            seg = y[a:b]
            if len(seg) and np.isfinite(seg).any():
                keep.add(a + int(np.nanargmin(seg)))
                keep.add(a + int(np.nanargmax(seg)))
    return np.array(sorted(keep))


def _log_ticks(lo: float, hi: float) -> list[float]:
    """Ticks 1, 2, 5 times a power of ten between lo and hi (positive numbers)."""
    cand = [m * 10.0 ** e for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1) for m in (1, 2, 5)]
    out = [t for t in cand if lo * 0.999 <= t <= hi * 1.001]
    return out if len(out) >= 3 else sorted(set(out + [m * 10.0 ** e for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1) for m in (1, 1.5, 3)]))


def _chart(cid: str, caption: str, x, series: list[dict], *, yfmt: str, xfmt, height: int = 230, area: bool = False, zero_line: bool = False,
           max_points: int = 900, xticks: list | None = None, xtickfmt=None, width: int = 760, logy: bool = False) -> str:
    """One line chart. `x` is a numeric array (milliseconds for dates); each series is {name, y, color, fmt}. `xfmt(value)` writes an x value as text."""
    W, H, L, R, T, B = width, height, 54, 18, 10, 24
    x = np.asarray(x, dtype=np.float64)
    full = [np.asarray(s["y"], dtype=np.float64) for s in series]
    idx = _pick(full, max_points)
    xs = x[idx]
    ys = [y[idx] for y in full]
    if logy:                                                              # the caller guarantees positive values; geometry is drawn on the log scale, values stay as they are
        gys = [np.log(np.where(y > 0, y, np.nan)) for y in ys]
    else:
        gys = ys
    allv = np.concatenate([y[np.isfinite(y)] for y in gys]) if any(np.isfinite(y).any() for y in gys) else np.array([0.0, 1.0])
    lo, hi = float(allv.min()), float(allv.max())
    if zero_line:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    if logy:
        tick_vals = _log_ticks(math.exp(lo), math.exp(hi))
        yt = [math.log(t) for t in tick_vals]
        ylab = [f"{t:g}" for t in tick_vals]
    else:
        yt = _nice_ticks(lo, hi, 5)
        ylab = None
    lo, hi = min(lo, yt[0]), max(hi, yt[-1])
    x0, x1 = float(xs[0]), float(xs[-1])
    if x1 <= x0:
        x1 = x0 + 1.0
    px = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)          # noqa: E731
    py = lambda v: T + (H - T - B) * (1.0 - (v - lo) / (hi - lo))    # noqa: E731
    g = []
    for i_t, t in enumerate(yt):
        t0 = 0.0 if abs(t) < 1e-9 else t                                  # never write -0%
        label = ylab[i_t] if ylab else (f"{t0 * 100:.0f}%" if yfmt == "pct" else (f"{t0:.2f}" if abs(t0) < 10 else f"{t0:.0f}"))
        g.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{py(t):.1f}" y2="{py(t):.1f}"/><text x="{L - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{_esc(label)}</text>')
    for t in (xticks if xticks is not None else [x0 + (x1 - x0) * k / 5 for k in range(6)]):
        if x0 <= t <= x1:
            g.append(f'<text x="{px(t):.1f}" y="{H - 6}" text-anchor="middle">{_esc((xtickfmt or xfmt)(t))}</text>')
    g.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{py(lo):.1f}" y2="{py(lo):.1f}"/>')
    if zero_line and lo < 0 < hi:
        g.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{py(0):.1f}" y2="{py(0):.1f}"/>')
    paths = []
    for s, y in zip(series, gys):
        segs, cur = [], []
        for xv, yv in zip(xs, y):
            if np.isfinite(yv):
                cur.append((px(xv), py(yv)))
            elif cur:
                segs.append(cur); cur = []
        if cur:
            segs.append(cur)
        for seg in segs:
            d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in seg)
            if area and s is series[0]:
                base = py(0.0 if lo < 0 < hi else lo)
                paths.append(f'<path d="{d} L{seg[-1][0]:.1f},{base:.1f} L{seg[0][0]:.1f},{base:.1f} Z" style="fill:{s["color"]};fill-opacity:.10;stroke:none"/>')
            paths.append(f'<path class="ln" d="{d}" style="stroke:{s["color"]}"/>')
    data = {"w": W, "h": H, "l": L, "r": R, "t": T, "b": B, "x0": x0, "x1": x1, "ymin": lo, "ymax": hi, "log": bool(logy), "xs": [round(float(v), 3) for v in xs],
            "labels": [xfmt(v) for v in xs],
            "series": [{"name": s["name"], "color": s["color"], "fmt": s["fmt"], "y": [None if not np.isfinite(v) else round(float(v), 6) for v in y]}
                       for s, y in zip(series, ys)]}
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    legend = ""
    if len(series) > 1:
        legend = '<div class="legend">' + "".join(f'<span><i style="border-color:{s["color"]}"></i>{_esc(s["name"])}</span>' for s in series) + "</div>"
    return (f'<figure class="chart" id="{_esc(cid)}"><figcaption>{_esc(caption)}</figcaption>{legend}'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(caption)}">{"".join(g)}{"".join(paths)}'
            f'<line class="cross" x1="0" x2="0" y1="{T}" y2="{H - B}" style="display:none"/></svg><div class="tip"></div>'
            f'<script type="application/json" class="cd">{blob}</script></figure>')


def _date_fmt(v: float) -> str:
    return pd.Timestamp(int(v), unit="ms").strftime("%Y-%m-%d")


def _year_fmt(v: float) -> str:
    return pd.Timestamp(int(v), unit="ms").strftime("%Y")


def _year_ticks(first: pd.Timestamp, last: pd.Timestamp) -> list[float]:
    years = pd.date_range(first.normalize() + pd.offsets.YearBegin(0), last, freq="YS")
    step = max(1, math.ceil(len(years) / 7))
    return [float(t.value // 10 ** 6) for t in years[::step]]


def _monthly(s: pd.Series) -> pd.Series:
    return (1.0 + s).groupby(s.index.to_period("M")).prod() - 1.0


def _tile(label: str, value: str, detail: str = "") -> str:
    return f'<div class="tile"><div class="l">{_esc(label)}</div><div class="v">{_esc(value)}</div><div class="d">{_esc(detail)}</div></div>'


def _heat(monthly: pd.Series) -> str:
    years = sorted({p.year for p in monthly.index})
    m = float(np.nanmax(np.abs(monthly.to_numpy()))) if len(monthly) else 0.0
    m = m if m > 0 else 1.0
    head = "<tr><th>Year</th>" + "".join(f"<th>{_esc(a)}</th>" for a in _MONTHS) + "<th>Year</th></tr>"
    rows = []
    for y in years:
        cells, ys = [], []
        for mo in range(1, 13):
            p = pd.Period(year=y, month=mo, freq="M")
            if p in monthly.index and math.isfinite(monthly[p]):
                v = float(monthly[p]); ys.append(v)
                arm, a = ("pos", v / m) if v >= 0 else ("neg", -v / m)
                bg = f"color-mix(in srgb, var(--{arm}) {min(55.0, 8.0 + 47.0 * a):.0f}%, var(--mid))"
                cells.append(f'<td style="background:{bg}" title="{_esc(p)}: {v * 100:.2f}%">{v * 100:.1f}</td>')
            else:
                cells.append('<td class="na"></td>')
        tot = float(np.prod([1.0 + v for v in ys]) - 1.0) if ys else float("nan")
        rows.append(f'<tr><td class="y">{y}</td>{"".join(cells)}<td class="y">{_pct(tot)}</td></tr>')
    return f'<div class="wrap"><table class="heat">{head}{"".join(rows)}</table></div>'


def _year_table(s: pd.Series, ppy: int) -> str:
    rows = []
    for y, g in s.groupby(s.index.year):
        eq = (1.0 + g).cumprod()
        sh = g.mean() / g.std() * math.sqrt(ppy) if len(g) > 2 and g.std() > 0 else float("nan")
        rows.append(f"<tr><td>{y}</td><td>{_pct(eq.iloc[-1] - 1)}</td><td>{_pct(g.std() * math.sqrt(ppy))}</td><td>{_num(sh)}</td><td>{_pct((eq / eq.cummax() - 1).min())}</td><td>{len(g)}</td></tr>")
    return ('<div class="wrap"><table><tr><th>Year</th><th>Return</th><th>Volatility</th><th>Sharpe</th><th>Worst drawdown</th><th>Bars</th></tr>'
            + "".join(rows) + "</table></div>")


def report_html(res, *, title: str | None = None, ledger=None, family: str | None = None, capacity: pd.DataFrame | None = None, max_points: int = 900) -> str:
    """The report as one HTML string. `ledger` and `family` add the deflated Sharpe of the family's trials; `capacity` is the table from `capacity_curve`."""
    s = res.net_returns
    if s is None or len(s) == 0:
        raise ValueError("the result has no net_returns to report")
    s = s.astype(float)
    ppy = int(res.spec.get("periods_per_year", 252))
    m = res.metrics
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    xms = np.array([t.value // 10 ** 6 for t in s.index], dtype=np.float64)
    ticks = _year_ticks(s.index[0], s.index[-1]) if (s.index[-1] - s.index[0]).days > 700 else None
    win = max(20, ppy // 2)
    roll = s.rolling(win).mean() / s.rolling(win).std() * math.sqrt(ppy)

    ci = None
    try:
        ci = res.sharpe_ci(seed=0) if len(s) >= 30 else None
    except Exception:                                                  # the interval is an extra: a report must not fail because it cannot be computed
        ci = None
    ab = None
    if res.benchmark_returns is not None:
        try:
            ab = res.alpha_beta()
        except Exception:
            ab = None

    # ---- what limits the number
    items = []
    if m.get("ruined"):
        items.append(("bad", "Account wiped out", f"the account lost 100% or more on {m.get('ruin_date')}; the series is -100% on that bar and 0 afterwards. Read CAGR and drawdown, not Sharpe."))
    for n in getattr(res, "notes", []):
        items.append(("w", "Warning", n))
    if ledger is not None and family is not None:
        try:
            d = ledger.deflated_sharpe(family, periods_per_year=ppy)
            ok = _ok(d.get("dsr")) and d["dsr"] >= 0.95
            items.append(("ok" if ok else "w", "Deflated Sharpe",
                          f"{d['trials']} trials recorded in family '{family}'. The best trial in the family ('{d.get('best_name', '?')}', not necessarily this run) has Sharpe {_num(d['sharpe'])}; the best of {d['trials']} random strategies would show about {_num(d['sharpe_luck_benchmark'])} "
                          f"by luck. Probability the best one beats luck: {_pct(d['dsr'], 0)} ({'above' if ok else 'below'} the usual 95% bar)."))
        except Exception as ex:
            items.append(("w", "Deflated Sharpe", f"could not be computed from the ledger ({ex})"))
    else:
        items.append(("w", "Trials not counted", "No ledger was given, so this page cannot say how many variants were tried before this one. A Sharpe ratio picked as the best of many tries is "
                      "overstated; pass ledger= and family= to include the deflated Sharpe."))
    if ci is not None:
        items.append(("w" if ci["lo"] <= 0 else "ok", "Sharpe interval", f"{_num(ci['sharpe'])}, 95% interval {_num(ci['lo'])} to {_num(ci['hi'])} "
                      f"(stationary block bootstrap, {len(s):,} bars); {'it includes 0: no evidence of an edge from this sample alone.' if ci['lo'] <= 0 else 'it excludes 0.'}"))
    if ab is not None and _ok(ab.get("alpha_t")):
        t_a = ab["alpha_t"]
        verdict = ("the alpha is significantly negative: the strategy lost against the benchmark beyond what chance explains" if t_a <= -2.5
                   else "positive and above the 2.5 bar" if t_a >= 2.5 else "no evidence of alpha (|t| below 2.5)")
        items.append(("ok" if t_a >= 2.5 else "w", "Alpha against the benchmark", f"{_pct(ab.get('alpha_annual'))} a year, t = {_num(t_a)}, beta {_num(next(iter(ab['betas'].values()), {}).get('beta'))}: {verdict}. "
                      "Newey-West t-values over-reject in finite samples (about 9% instead of 5%), so 2.5 rather than 2 is the bar."))
    cls = "bad" if m.get("ruined") else ("ok" if all(k == "ok" for k, _, _ in items) else "")
    icon = {"w": "!", "ok": "✓", "bad": "✕"}
    lis = "".join(f'<li class="{k}"><span class="ic">{icon[k]}</span><span class="lab">{_esc(t)}.</span> {_esc(x)}</li>' for k, t, x in items)
    read = f'<section class="card read {cls}"><div class="lab">Read this first: what limits these numbers</div><ul>{lis}</ul></section>'

    tiles = "".join([
        _tile("CAGR", _pct(m.get("CAGR")), f"before costs {_pct(m.get('gross_CAGR'))}" if "gross_CAGR" in m else ""),
        _tile("Sharpe", _num(m.get("Sharpe")), f"95% interval {_num(ci['lo'])} to {_num(ci['hi'])}" if ci else "no risk-free rate subtracted"),
        _tile("Max drawdown", _pct(m.get("MDD")), f"{m.get('mdd_peak', '')} to {m.get('mdd_trough', '')}" if m.get("mdd_peak") else ""),
        _tile("Volatility", _pct(m.get("vol")), "annualised"),
        _tile("Turnover", _num(m.get("turnover_daily"), 3), "one-way share of the book per bar"),
        _tile("Years", _num(m.get("years")), f"{len(s):,} bars"),
    ])
    sers = [{"name": "Strategy (net)", "y": eq.to_numpy(), "color": "var(--s1)", "fmt": "eq"}]
    if res.benchmark_returns is not None:
        b = res.benchmark_returns.reindex(s.index).fillna(0.0)
        sers.append({"name": "Benchmark (no costs)", "y": (1.0 + b).cumprod().to_numpy(), "color": "var(--muted)", "fmt": "eq"})
    allpos = np.concatenate([x_[np.isfinite(x_)] for x_ in (np.asarray(sr["y"], dtype=float) for sr in sers)])
    wide = bool(len(allpos) and allpos.min() > 0 and allpos.max() / allpos.min() > 8.0)       # a fall from 1 to 0.1 is unreadable on a straight axis
    charts = [
        _chart("equity", "Growth of 1 (net of costs and funding)" + (", log scale" if wide else ""), xms, sers, yfmt="num", xfmt=_date_fmt, height=260, max_points=max_points, xticks=ticks,
               xtickfmt=_year_fmt, logy=wide),
        _chart("drawdown", "Drawdown from the previous peak", xms, [{"name": "Drawdown", "y": dd.to_numpy(), "color": "var(--s2)", "fmt": "pct"}], yfmt="pct",
               xfmt=_date_fmt, area=True, zero_line=True, max_points=max_points, xticks=ticks, xtickfmt=_year_fmt, width=470, height=220),
        _chart("rolling", f"Rolling Sharpe ({win} bars, annualised)", xms, [{"name": "Rolling Sharpe", "y": roll.to_numpy(), "color": "var(--s1)", "fmt": "num"}], yfmt="num",
               xfmt=_date_fmt, zero_line=True, max_points=max_points, xticks=ticks, xtickfmt=_year_fmt, width=470, height=220),
    ]
    mon = _monthly(s)

    cost_rows = []
    for k, label in (("cost_annual_bp", "Trading costs"), ("spread_annual_bp", "Spread"), ("side_cost_annual_bp", "Buy/sell side costs"), ("borrow_annual_bp", "Short borrow"),
                     ("impact_annual_bp", "Market impact"), ("funding_annual_bp", "Funding paid (negative: received)")):
        if k in m and _ok(m[k]):
            cost_rows.append(f"<tr><td>{label}</td><td>{m[k]:,.0f} bp a year</td></tr>")
    if "gross_CAGR" in m and _ok(m.get("gross_CAGR")) and _ok(m.get("CAGR")):
        cost_rows.append(f"<tr><td>CAGR given up to costs and funding</td><td>{_pct(m['gross_CAGR'] - m['CAGR'])} a year</td></tr>")
    H = res.holdings
    if H is not None and getattr(H, "size", 0):
        H = np.asarray(H, dtype=float)
        cost_rows.append(f"<tr><td>Average long exposure</td><td>{np.nanmean(np.where(H > 0, H, 0).sum(axis=1)):.2f}</td></tr>"
                         f"<tr><td>Average short exposure</td><td>{np.nanmean(np.where(H < 0, -H, 0).sum(axis=1)):.2f}</td></tr>")
    costs = f'<div class="card wrap"><table>{"".join(cost_rows)}</table></div>' if cost_rows else ""

    cap = ""
    if capacity is not None and len(capacity):
        c = capacity.sort_values(["y", "aum"])
        ys_ = list(dict.fromkeys(c["y"].tolist()))[:3]
        aums = sorted(c["aum"].unique())
        cs = [("var(--s1)", "var(--s2)", "var(--s3)")[i] for i in range(len(ys_))]
        series = []
        for yv, col in zip(ys_, cs):
            sub = c[c["y"] == yv].set_index("aum").reindex(aums)
            series.append({"name": f"Impact coefficient Y = {yv:g}", "y": sub["cagr"].to_numpy(dtype=float), "color": col, "fmt": "pct"})
        lx = np.log10(np.array(aums, dtype=float))
        cap = ("<h2>Capacity</h2><div class='card'>"
               + _chart("capacity", "Net CAGR by the amount of money run (log scale)", lx, series, yfmt="pct", xfmt=lambda v: f"{10 ** v:,.3g}", zero_line=True, max_points=max_points,
                        xticks=[float(v) for v in lx])
               + "<div class='wrap'><table><tr><th>Money</th>" + "".join(f"<th>Sharpe, Y={yv:g}</th>" for yv in ys_) + "<th>Largest trade vs daily volume (99th pct)</th></tr>"
               + "".join(f"<tr><td>{a:,.4g}</td>" + "".join(f"<td>{_num(c[(c['y'] == yv) & (c['aum'] == a)]['sharpe'].iloc[0])}</td>" for yv in ys_)
                         + f"<td>{_pct(c[(c['y'] == ys_[0]) & (c['aum'] == a)]['participation_p99'].iloc[0], 1) if 'participation_p99' in c else 'n/a'}</td></tr>" for a in aums)
               + "</table></div></div>")

    spec_rows = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in res.spec.items())
    conv = ("Returns are net of trading costs, borrow and funding, per bar. A signal on date d enters at close(d + entry_lag) and earns close(d + entry_lag) to close(d + entry_lag + 1). "
            "Costs are only as good as the inputs given; the engine has no order book, queue, partial fills or margin model. This page describes a simulation; it is not advice.")
    from ._version import __version__
    ttl = title or "Backtest report"
    badge = '<span class="badge">ruined</span>' if m.get("ruined") else ""         # built outside the f-string: a backslash in one needs Python 3.12
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_esc(ttl)}</title><style>{_CSS}</style></head><body><main>"
            f"<h1>{_esc(ttl)}{badge}</h1>"
            f"<p class=\"sub\">{s.index[0].date()} to {s.index[-1].date()} · {len(s):,} bars · pitbacktest {__version__}</p>"
            f"{read}<h2>Figures</h2><div class=\"tiles\">{tiles}</div>"
            f"<h2>Curves</h2><div class=\"card\">{charts[0]}</div><div class=\"two\" style=\"margin-top:12px\"><div class=\"card\">{charts[1]}</div><div class=\"card\">{charts[2]}</div></div>"
            f"<h2>Months and years</h2><div class=\"card\">{_heat(mon)}</div><div class=\"card\" style=\"margin-top:12px\">{_year_table(s, ppy)}</div>"
            f"<h2>Costs and exposure</h2>{costs}{cap}"
            f"<h2>How it was run</h2><div class=\"card\"><dl>{spec_rows}</dl><p class=\"foot\" style=\"margin-bottom:0\">{_esc(conv)}</p></div>"
            f"<p class=\"foot\">Self-contained page: no network access, no scripts beyond the hover tooltips.</p></main><script>{_JS}</script></body></html>")


def write_report(res, path, **kw) -> Path:
    """Write `report_html(res, **kw)` to `path` (UTF-8) and return the path."""
    p = Path(path)
    p.write_text(report_html(res, **kw), encoding="utf-8")
    return p
