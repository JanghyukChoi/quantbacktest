"""The yfinance adapter against a fake `yfinance` module (no network). Its main job is to be loud about what free data cannot do.

Y1 point in time   every panel carries the not-point-in-time warning
Y2 survivorship    with 30+ tickers and none ending early the survivorship warning fires; with delisted names present it does not;
                   with fewer than 30 tickers the size rule keeps it quiet
Y3 market cap      shares x close; partial coverage warns with the counts; a manual share count is the fallback; none switches the controls off
Y4 errors          no data, no Close column, nothing eligible, and a missing yfinance each raise a clear error
Y5 shapes          one ticker is read and then refused (no cross-section); the 365-day year for crypto
"""
from __future__ import annotations
import sys, types, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pitbacktest.adapters import yfinance as yfa


def _fake_yf(n=40, T=300, end_early=0, empty=False, no_close=False, shares_fail=(), seed=0, flat=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=T)
    cols = [f"T{i:02d}" for i in range(n)]
    close = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, n)), axis=0)), index=idx, columns=cols)
    for c in cols[:end_early]:
        close.loc[close.index[T // 2]:, c] = np.nan                               # delisted halfway
    vol = pd.DataFrame(1e5, index=idx, columns=cols)
    fields = {"Close": close, "Open": close, "High": close * 1.01, "Low": close * 0.99, "Volume": vol}
    if no_close:
        fields.pop("Close")
    if flat:
        raw = pd.DataFrame({k: v[cols[0]] for k, v in fields.items()})
    else:
        raw = pd.concat(fields, axis=1)

    class Ticker:
        def __init__(self, t):
            self.t = t

        def get_shares_full(self, start=None):
            if self.t in shares_fail:
                raise RuntimeError("no shares")
            return pd.Series(1e7, index=pd.to_datetime(pd.bdate_range(start or "2023-01-02", periods=10)))

    m = types.ModuleType("yfinance")
    m.download = lambda *a, **k: pd.DataFrame() if empty else raw
    m.Ticker = Ticker
    return m, cols, close


def _load(fake, cols, **kw):
    old = sys.modules.get("yfinance")
    sys.modules["yfinance"] = fake
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = yfa.load_panel(cols, "2023-01-02", "2024-03-01", **kw)
        return p, [str(x.message) for x in w]
    finally:
        if old is None:
            sys.modules.pop("yfinance", None)
        else:
            sys.modules["yfinance"] = old


def test_warnings_for_survivorship():
    fake, cols, _ = _fake_yf(n=40)
    p, w = _load(fake, cols)
    assert any("not a point-in-time universe" in m for m in w), w                                   # Y1
    assert any("holds only securities alive today" in m and "docs/survivorship.md" in m for m in w), w
    fake, cols, _ = _fake_yf(n=40, end_early=8)
    p2, w2 = _load(fake, cols)
    assert any("not a point-in-time universe" in m for m in w2) and not any("alive today" in m for m in w2), w2
    fake, cols, _ = _fake_yf(n=12)
    _, w3 = _load(fake, cols)
    assert not any("alive today" in m for m in w3)                                                  # fewer than 30 tickers: the rule stays quiet
    print("Y1/Y2 the point-in-time warning is always there; the survivors-only warning needs 30+ tickers and none ending early  PASS")


def test_market_cap_paths():
    fake, cols, close = _fake_yf(n=40)
    p, w = _load(fake, cols)
    assert p.mkt_cap is not None and np.allclose((p.mkt_cap / p.close).iloc[-1].dropna(), 1e7)
    assert not any("Received shares" in m or "Market cap could not" in m for m in w)
    fake, cols, _ = _fake_yf(n=40, shares_fail=("T03", "T07"))
    p, w = _load(fake, cols)
    assert any("Received shares for only 38/40" in m for m in w), w
    assert p.mkt_cap["T03"].isna().all() and p.mkt_cap["T00"].notna().any()
    fake, cols, _ = _fake_yf(n=40, shares_fail=tuple(f"T{i:02d}" for i in range(40)))
    p, w = _load(fake, cols)
    assert p.mkt_cap is None and any("Market cap could not be built" in m for m in w)
    p, w = _load(fake, cols, shares_outstanding={"T00": 2e6, "T01": 3e6})                           # the manual fallback
    assert p.mkt_cap is not None and abs((p.mkt_cap["T00"] / p.close["T00"]).iloc[5] - 2e6) < 1e-6 and p.mkt_cap["T05"].isna().all()
    p, w = _load(fake, cols, with_market_cap=False)
    assert p.mkt_cap is None and any("Market cap could not be built" in m for m in w)
    p, w = _load(fake, cols, with_market_cap=False, with_chars=True)
    assert any("no market cap" in m.lower() for m in w) and p.chars == {}
    print("Y3 market cap = shares x close; 38/40 coverage warns with the counts; the manual fallback and the no-cap paths are explicit  PASS")


def test_errors_and_shapes():
    for kw, msg in ((dict(empty=True), "returned no data"), (dict(no_close=True), "no Close column")):
        fake, cols, _ = _fake_yf(**kw)
        try:
            _load(fake, cols)
        except ValueError as e:
            assert msg in str(e), (msg, str(e))
        else:
            raise AssertionError(msg)
    fake, cols, _ = _fake_yf(n=40)
    try:
        _load(fake, cols, min_dollar_volume=1e12)
    except ValueError as e:
        assert "lower min_dollar_volume" in str(e)
    else:
        raise AssertionError("nothing eligible must raise")
    old = sys.modules.get("yfinance")
    sys.modules["yfinance"] = None                                                                  # `import yfinance` now raises ImportError
    try:
        yfa.load_panel(["A"], "2023-01-02", "2024-03-01")
    except ImportError as e:
        assert "pip install yfinance" in str(e)
    else:
        raise AssertionError("a missing yfinance must say how to install it")
    finally:
        sys.modules.pop("yfinance", None) if old is None else sys.modules.__setitem__("yfinance", old)
    fake, cols, _ = _fake_yf(n=1, flat=True)                      # one ticker is read (flat columns) and then refused: no cross-section to test
    try:
        _load(fake, cols, with_market_cap=False)
    except ValueError as e:
        assert "fewer than 10 securities" in str(e), str(e)
    else:
        raise AssertionError("a one-ticker panel must be refused")
    fake, cols, _ = _fake_yf(n=12)
    p, _ = _load(fake, cols, with_market_cap=False)
    assert list(p.close.columns) == cols and p.periods_per_year == 252
    fake, cols, _ = _fake_yf(n=40)
    p, _ = _load(fake, cols, market="CRYPTO", with_market_cap=False)
    assert p.periods_per_year == 365
    print("Y4/Y5 no data, no Close, nothing eligible and a missing yfinance each raise a clear error; one ticker is refused for lack of a cross-section; crypto uses 365 days  PASS")


if __name__ == "__main__":
    test_warnings_for_survivorship()
    test_market_cap_paths()
    test_errors_and_shapes()
    print("yfinance adapter tests: all passed")
