"""The one-page HTML report.

RP1 numbers   the page shows the result's figures; the same result gives the same page byte for byte; no address of any kind (nothing is fetched)
RP2 thinning  a long series is thinned for drawing but the trough of the drawdown and the end of the equity curve are still in the chart data
RP3 escaping  a title or a warning with HTML in it is shown as text, never run
RP4 limits    the warnings, a wiped-out account, the missing ledger, the deflated Sharpe with a ledger, and the capacity table are on the page
RP5 shapes    no benchmark, a sample too short for annual figures, and a result with no returns all behave
"""
from __future__ import annotations
import json, re, sys, tempfile, warnings
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.report import _pct, _num, _pick
from test_argument_checks import _raises
from test_synthetic import make_panel


def _result(n_days=900, **kw):
    p, rng = make_panel(n_days=n_days, n_stocks=50)
    f = pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p, f, q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5, spread_bp=10, **kw)


def _charts(text):
    return {m.group(1): json.loads(m.group(2).replace("\\u003c", "<")) for m in re.finditer(r'<figure class="chart" id="([^"]+)".*?<script type="application/json" class="cd">(.*?)</script>', text, re.S)}


def test_numbers_and_determinism():
    p, f, r = _result()
    page = r.report(title="Reversal")
    assert page.startswith("<!doctype html>") and "<title>Reversal</title>" in page
    for want in (_pct(r.metrics["CAGR"]), _num(r.metrics["Sharpe"]), _pct(r.metrics["MDD"]), _pct(r.metrics["vol"]), _num(r.metrics["turnover_daily"], 3)):
        assert want in page, want
    for h in ("Read this first", "Figures", "Curves", "Months and years", "Costs and exposure", "How it was run"):
        assert h in page, h
    assert page == r.report(title="Reversal")                                         # no clock, no randomness
    rest = page.replace("http://www.w3.org/2000/svg", "")                                # the SVG namespace is a name, not an address that is fetched
    for bad in ("http", "src=", "href=", "@import", "url(", "fetch(", "XMLHttpRequest", "import("):
        assert bad not in rest, bad
    out = r.report(Path(tempfile.mkdtemp(prefix="report-")) / "r.html", title="Reversal")
    assert Path(out).read_text(encoding="utf-8") == page
    c = _charts(page)
    assert set(c) == {"equity", "drawdown", "rolling"}
    print("RP1 the page shows the result's figures, is identical on a second call, writes a file, and contains no address  PASS")


def test_thinning_keeps_extremes():
    p, f, r = _result(n_days=2500)
    c = _charts(r.report(max_points=300))
    eq = (1 + r.net_returns).cumprod()
    dd = eq / eq.cummax() - 1
    ys = [v for v in c["drawdown"]["series"][0]["y"] if v is not None]
    assert len(c["drawdown"]["xs"]) < len(dd) // 2, "the series was thinned"
    assert abs(min(ys) - dd.min()) < 1e-6, (min(ys), dd.min())                            # the worst point is drawn
    e = c["equity"]["series"][0]["y"]
    assert abs(e[-1] - eq.iloc[-1]) < 1e-6 and abs(max(v for v in e if v is not None) - eq.max()) < 1e-6 and abs(min(v for v in e if v is not None) - eq.min()) < 1e-6
    a = np.array([0.0, 5, 1, 9, 2, 0, 4, 3, -7, 3, 8, 1])
    idx = _pick([a], 4)
    assert idx.tolist() == [0, 3, 8, 11], idx                                            # first, the maximum (index 3), the minimum (index 8), last
    print("RP2 a long series is thinned but the drawdown trough, the equity peak and low, and the last point are still in the chart  PASS")


def test_escaping():
    p, f, r = _result()
    r.notes = ['<img src=x onerror=alert(1)> & "quoted"']
    page = r.report(title='<script>alert("x")</script>')
    assert "<script>alert(" not in page and "&lt;script&gt;alert(" in page
    assert "<img src=x" not in page and "&lt;img src=x onerror=alert(1)&gt; &amp; &quot;quoted&quot;" in page
    assert page.count("<script") == page.count("</script>") == 4                          # three chart data blocks and the hover code, nothing else
    print("RP3 HTML in a title or a warning is shown as text and nothing extra runs  PASS")


def test_limits_on_the_page():
    p, rng = make_panel(n_days=300, n_stocks=12)
    c = p.close.copy(); c.iloc[:, 0] = 100.0; c.iloc[150:, 0] = 50.0
    w = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); w.iloc[100:, 0] = 30.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = q.backtest_weights(replace(p, close=c), w, spread_bp=0, benchmark=None)
    page = r.report()
    assert 'class="badge">ruined' in page and "Account wiped out" in page and "lost 100%" in page and "Trials not counted" in page
    p2, f2, r2 = _result()
    led = q.Ledger(tempfile.mkdtemp(prefix="rep-ledger-"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in (3, 5, 10):
            q.backtest_portfolio(p2, -p2.close.pct_change(k), hold=5, spread_bp=10, ledger=led, family="rev", name=f"k{k}")
        cap = q.capacity_curve(p2, pd.DataFrame(r2.holdings * 0.5, index=p2.dates, columns=p2.tickers), aums=[1e6, 1e8], y_values=(0.5, 1.0), spread_bp=10)
    page2 = r2.report(ledger=led, family="rev", capacity=cap)
    assert "3 trials recorded in family" in page2 and "Trials not counted" not in page2 and "Capacity" in page2 and "Impact coefficient Y = 0.5" in page2
    assert "-0%" not in page2 and "-0.00<" not in page2
    assert "Capacity" not in r2.report() and 'class="badge"' not in r2.report()
    print("RP4 warnings, a wiped-out account, the missing ledger, the deflated Sharpe with a ledger and the capacity table are on the page  PASS")


def test_shapes():
    p, f, r = _result(benchmark=None)
    page = r.report()
    assert "Benchmark" not in page and "Alpha against the benchmark" not in page and "Growth of 1" in page
    p3, _ = make_panel(n_days=100, n_stocks=30)
    f3 = pd.DataFrame(np.random.default_rng(0).standard_normal(p3.close.shape), index=p3.dates, columns=p3.tickers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        short = q.backtest_portfolio(p3, f3, hold=5, spread_bp=10)
    page3 = short.report()
    assert "n/a" in page3 and "Read this first" in page3                                  # the annual figures are unavailable, the page still stands
    empty = replace(r, net_returns=None)
    _raises(lambda: empty.report(), "no net_returns")
    assert _pct(-0.00001) == "0.0%" and _num(-0.001) == "0.00" and _pct(float("nan")) == "n/a" and _num(None) == "n/a"
    print("RP5 no benchmark, a sample too short for annual figures, and a result with no returns behave; -0 is never printed  PASS")


if __name__ == "__main__":
    test_numbers_and_determinism()
    test_thinning_keeps_extremes()
    test_escaping()
    test_limits_on_the_page()
    test_shapes()
    print("report tests: all passed")
