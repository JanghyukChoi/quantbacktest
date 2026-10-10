"""`compare`, `blend` and `strategy_correlation`.

CP1 blend         with zero costs the blend's net returns are the weighted sum of the parts' (a linear identity); with costs it is cheaper than the weighted costs of its parts; bad input raises
CP2 correlation   equals numpy's on the shared dates
CP3 compare       the page's figures equal the metrics, an event review is accepted through its portfolio view, limits and mismatches raise, names are escaped, the page is reproducible
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.report import _num, _pct
from test_argument_checks import _raises
from test_synthetic import make_panel


def _q(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _parts(n_days=900, n_stocks=60):
    p, rng = make_panel(n_days=n_days, n_stocks=n_stocks)
    z = (p.close.pct_change().shift(-2).rank(axis=1, pct=True) - .5) * np.sqrt(12)
    noise = lambda: pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)       # noqa: E731
    fa = (0.05 * z + noise()).where(p.eligible)
    fb = (0.03 * z + noise()).where(p.eligible)
    kw = dict(long_q=0.2, short_q=0.2, hold=5, spread_bp=10.0, funding=False)
    ra = _q(lambda: q.backtest_portfolio(p, fa, **kw))
    rb_ = _q(lambda: q.backtest_portfolio(p, fb, **kw))
    return p, rng, z, ra, rb_


def test_blend():
    p, rng, z, ra, rb_ = _parts()
    parts = {"A": ra, "B": rb_}
    m0 = _q(lambda: q.blend(p, parts, {"A": 0.7, "B": 0.3}, spread_bp=0.0, benchmark=None))
    g = lambda r: _q(lambda: q.backtest_weights(p, pd.DataFrame(r.holdings, index=p.dates, columns=p.tickers), spread_bp=0.0, benchmark=None, check_universe=False)).net_returns   # noqa: E731
    want = 0.7 * g(ra) + 0.3 * g(rb_)
    assert float(np.abs(m0.net_returns - want).max()) < 1e-12, "zero cost: the blend is the weighted sum of the parts' returns"
    eq = _q(lambda: q.blend(p, parts, spread_bp=0.0, benchmark=None))
    assert float(np.abs(eq.net_returns - 0.5 * g(ra) - 0.5 * g(rb_)).max()) < 1e-12                                      # default: equal weights
    un = _q(lambda: q.blend(p, parts, {"A": 7, "B": 3}, spread_bp=0.0, benchmark=None))
    assert float(np.abs(un.net_returns - m0.net_returns).max()) < 1e-15                                                  # weights are normalised to sum to 1
    # with costs the blend nets what the parts trade against each other: it is cheaper than the weighted costs of its parts
    cost = lambda r, c: float((g(r) - _q(lambda: q.backtest_weights(p, pd.DataFrame(r.holdings, index=p.dates, columns=p.tickers), spread_bp=c, benchmark=None, check_universe=False)).net_returns).sum())   # noqa: E731
    bl = _q(lambda: q.blend(p, parts, spread_bp=10.0, benchmark=None))
    blend_cost = float((m0.net_returns * 0 + (0.5 * g(ra) + 0.5 * g(rb_)) - bl.net_returns).sum())
    assert 0 < blend_cost < 0.5 * cost(ra, 10.0) + 0.5 * cost(rb_, 10.0), (blend_cost, cost(ra, 10.0), cost(rb_, 10.0))
    one = _q(lambda: q.blend(p, {"A": ra}, spread_bp=0.0, benchmark=None))
    assert float(np.abs(one.net_returns - g(ra)).max()) < 1e-12
    _raises(lambda: q.blend(p, parts), "spread_bp")
    _raises(lambda: q.blend(p, parts, {"A": 1.0}, spread_bp=0.0), "exactly")
    _raises(lambda: q.blend(p, parts, {"A": 1.0, "B": -1.0}, spread_bp=0.0), "sum to zero")
    other, _ = make_panel(n_days=500, n_stocks=60)
    _raises(lambda: q.blend(other, parts, spread_bp=0.0), "another panel")
    print("CP1 with zero costs the blend is the weighted sum of its parts (1e-12), weights are normalised, with costs it is cheaper than the parts' weighted costs, bad input raises  PASS")


def test_correlation():
    p, rng, z, ra, rb_ = _parts()
    c = q.strategy_correlation({"A": ra, "B": rb_})
    both = pd.concat([ra.net_returns, rb_.net_returns], axis=1, join="inner").dropna()
    assert abs(c.loc["A", "B"] - np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]) < 1e-12 and c.loc["A", "B"] == c.loc["B", "A"] and c.loc["A", "A"] == 1.0
    import dataclasses
    short_b = dataclasses.replace(rb_, net_returns=rb_.net_returns.iloc[300:])                                         # a strategy that starts later: only the shared dates count
    c2 = q.strategy_correlation({"A": ra, "B": short_b})
    sh = pd.concat([ra.net_returns, short_b.net_returns], axis=1, join="inner").dropna()
    assert abs(c2.loc["A", "B"] - np.corrcoef(sh.iloc[:, 0], sh.iloc[:, 1])[0, 1]) < 1e-12 and len(sh) == len(short_b.net_returns)
    _raises(lambda: q.strategy_correlation({}), "non-empty")
    print(f"CP2 the correlation of the two strategies ({c.loc['A', 'B']:.3f}) equals numpy's on their shared dates  PASS")


def test_compare():
    import dataclasses
    p, rng, z, ra, rb_ = _parts()
    sig = ((z + pd.DataFrame(rng.standard_normal(p.close.shape), index=p.dates, columns=p.tickers)) > 2.0) & p.eligible
    ev = _q(lambda: q.review_event(p, sig, horizons=(1, 5), cost_bp=10, n_permutations=20))
    parts = {"Reversal <b>A</b>": ra, "B": rb_, "Event": ev}
    page = _q(lambda: q.compare(parts, title="T"))
    assert "Reversal &lt;b&gt;A&lt;/b&gt;" in page and "Reversal <b>A</b>" not in page
    for h in ("Figures", "Growth", "Risk and return", "Do they move together?", "Year by year", "Skewness", "Expected shortfall 95%", "Beta to the benchmark"):
        assert h in page, h
    assert page == _q(lambda: q.compare(parts, title="T"))                                                              # reproducible
    for bad in ("fetch(", "XMLHttpRequest", "@import", "url(", "href=", "src="):
        assert bad not in page, bad
    # the figures on the page are the metrics of each strategy on the shared dates
    common = pd.concat({k: r.net_returns for k, r in (("A", ra), ("B", rb_), ("E", ev.portfolio))}, axis=1, join="inner").dropna()
    from pitbacktest.portfolio import metrics
    mA = metrics(common["A"].to_numpy(), common.index, 252)
    assert _pct(mA["CAGR"]) in page and _num(mA["Sharpe"]) in page and _pct(mA["MDD"]) in page
    assert "benchmark: the benchmark of" in page and "Benchmark</th>" in page
    # an index of your own: it must be put on the panel's signal dates
    idx = pd.Series(np.random.default_rng(3).normal(0.0003, 0.01, len(p.dates)), index=p.dates)
    own = _q(lambda: q.compare({"A": ra, "B": rb_}, panel=p, benchmark_returns=idx))
    assert "benchmark: the index you supplied" in own
    # a strategy that holds the index: aligned onto the signal dates, its beta to it is exactly 1; not moved, it would be about 0
    from pitbacktest import robustness as rbm
    held = dataclasses.replace(ra, net_returns=rbm.align_benchmark(p, idx).reindex(ra.net_returns.index).dropna())
    own2 = _q(lambda: q.compare({"holds the index": held, "B": rb_}, panel=p, benchmark_returns=idx))
    beta_row = [r for r in own2.split("<tr>") if r.startswith("<td>Beta to the benchmark")][0]
    import re
    cells = re.findall(r"<td>(.*?)</td>", beta_row)
    assert cells[1] == "1.00" and cells[2] != "1.00", cells                                                            # the strategy that holds the index; the other one; (the last cell is the benchmark's own 1.00)
    _raises(lambda: q.compare({"A": ra}, benchmark_returns=idx), "needs the panel")
    many = {f"S{i}": ra for i in range(7)}
    _raises(lambda: q.compare(many), "at most 6")
    wk = _q(lambda: q.backtest_portfolio(p, (0.05 * z).where(p.eligible) + 0, long_q=0.2, short_q=0.2, hold=5, spread_bp=10.0))
    odd = dataclasses.replace(wk, spec={**wk.spec, "periods_per_year": 365})
    _raises(lambda: q.compare({"A": ra, "odd": odd}), "periods_per_year")
    noport = _q(lambda: q.review_event(p, sig, horizons=(5,), cost_bp=10, n_permutations=20, as_portfolio=False))
    _raises(lambda: q.compare({"E": noport}), "no portfolio view")
    _raises(lambda: q.compare({"x": 3}), "cannot compare")
    out = q.compare({"A": ra, "B": rb_}, Path(__import__("tempfile").mkdtemp(prefix="cmp-")) / "c.html")
    assert Path(out).read_text(encoding="utf-8").startswith("<!doctype html>")
    six = _q(lambda: q.compare({f"S{i}": (ra if i % 2 == 0 else rb_) for i in range(6)}))
    assert all(f"var(--s{i})" in six for i in range(1, 7))
    print("CP3 the page's figures equal the metrics on the shared dates, an event review joins through its portfolio view, names are escaped, limits and mismatches raise, six colours appear  PASS")


if __name__ == "__main__":
    test_blend()
    test_correlation()
    test_compare()
    print("compare tests: all passed")
