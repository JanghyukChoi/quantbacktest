"""Trading rules of Binance contracts, on a fake network.

R1 parse      step, minimum quantity, minimum order value, tick and dates come out per symbol; contracts without the filters are skipped
R2 cache      the second call makes no request and carries the day it was fetched; refresh asks again
R3 rules      lot is the larger of the step and the minimum quantity; an unknown ticker raises, or is NaN so it can be left out
R4 engine     `backtest_weights` takes a minimum order value per ticker: each ticker skips the trades under its own value
"""
from __future__ import annotations
import json, sys, tempfile
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.crypto import binance_archive as ba
from pitbacktest.crypto.binance_archive import ArchiveStore, execution_rules
from test_reconcile import _panel

INFO = {"symbols": [
    {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "onboardDate": 1567965300000, "deliveryDate": 4133404800000,
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"}, {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                 {"filterType": "MIN_NOTIONAL", "notional": "50"}]},
    {"symbol": "DOGEUSDT", "status": "TRADING", "contractType": "PERPETUAL", "onboardDate": 1, "deliveryDate": 4133404800000,
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.00001"}, {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "10"},
                 {"filterType": "MIN_NOTIONAL", "notional": "5"}]},
    {"symbol": "ODDUSDT", "status": "SETTLING", "contractType": "PERPETUAL", "onboardDate": 2, "deliveryDate": 1790000000000, "filters": []},
]}


def test_parse_cache_and_rules():
    calls = []
    real = ba._get
    ba._get = lambda url, *a, **k: (calls.append(url), json.dumps(INFO).encode())[1]
    try:
        with tempfile.TemporaryDirectory(prefix="rulestest-") as d:
            st = ArchiveStore(d)
            r = st.symbol_rules()
            assert list(r.index) == ["BTCUSDT", "DOGEUSDT"], list(r.index)                       # ODDUSDT has no LOT_SIZE or MIN_NOTIONAL: skipped
            assert r.loc["BTCUSDT", "min_notional"] == 50.0 and r.loc["BTCUSDT", "step_size"] == 0.001 and r.loc["DOGEUSDT", "min_qty"] == 10.0
            assert r.loc["BTCUSDT", "tick_size"] == 0.1 and r.loc["BTCUSDT", "status"] == "TRADING"
            assert r.attrs["fetched"] == pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
            n = len(calls)
            st.symbol_rules()
            assert len(calls) == n, "the cached rules must not be fetched again"
            st.symbol_rules(refresh=True)
            assert len(calls) == n + 1
            lot, mn = execution_rules(r, ["BTCUSDT", "DOGEUSDT"])
            assert lot["BTCUSDT"] == 0.001 and lot["DOGEUSDT"] == 10.0 and mn["BTCUSDT"] == 50.0 and mn["DOGEUSDT"] == 5.0    # DOGE: min quantity 10 > step 1
            try:
                execution_rules(r, ["BTCUSDT", "GONEUSDT"])
            except ValueError as e:
                assert "GONEUSDT" in str(e) and "nan" in str(e)
            else:
                raise AssertionError("an unknown ticker must raise by default")
            lot2, mn2 = execution_rules(r, ["BTCUSDT", "GONEUSDT"], missing="nan")
            assert np.isnan(lot2["GONEUSDT"]) and np.isnan(mn2["GONEUSDT"]) and lot2["BTCUSDT"] == 0.001      # no rule is invented
            try:
                execution_rules(r, ["BTCUSDT"], missing="guess")
            except ValueError as e:
                assert "missing" in str(e)
            else:
                raise AssertionError("a bad `missing` must raise")
    finally:
        ba._get = real
    print("R1 to R3 rules are parsed per symbol, cached, and turned into lot and minimum order value; an unknown ticker raises or is NaN, never a made-up rule  PASS")


def test_engine_takes_a_minimum_per_ticker():
    p, _ = _panel(seed=21, T=160, N=20)
    p = replace(p, eligible=p.eligible | True)
    rng = np.random.default_rng(21)
    W = pd.DataFrame(rng.normal(0, 0.05, p.close.shape), index=p.dates, columns=p.tickers)
    price = pd.DataFrame(np.exp(rng.normal(3.0, 0.4, p.close.shape)), index=p.dates, columns=p.tickers)
    cap = 1e6
    mtv = pd.Series(np.where(np.arange(20) < 10, 0.0, 30000.0), index=p.tickers)                   # the first ten have no minimum, the rest 30,000
    r = q.backtest_weights(p, W, capital=cap, price=price, lot=1.0, min_trade_value=mtv, benchmark=None, check_universe=False)
    d = np.abs(np.diff(r.holdings, axis=0)) * cap
    big = d[:, 10:]; moved = big[big > 1e-9]
    assert moved.min() >= 30000.0 - 1e-6, moved.min()                                               # those ten never trade under their minimum
    assert (d[:, :10][d[:, :10] > 1e-9]).min() < 30000.0                                            # the others still make small trades
    assert r.metrics["min_trade_skipped"] > 0
    flat = q.backtest_weights(p, W, capital=cap, price=price, lot=1.0, min_trade_value=0.0, benchmark=None, check_universe=False)
    same = q.backtest_weights(p, W, capital=cap, price=price, lot=1.0, min_trade_value=pd.Series(0.0, index=p.tickers), benchmark=None, check_universe=False)
    assert np.array_equal(flat.holdings, same.holdings)                                             # a Series of zeros is the same as no minimum
    try:
        q.backtest_weights(p, W, capital=cap, price=price, min_trade_value=pd.Series([1.0], index=[p.tickers[0]]), benchmark=None, check_universe=False)
    except ValueError as e:
        assert "every security" in str(e)
    else:
        raise AssertionError("a minimum that is unknown for some securities must raise")
    print("R4 a minimum order value per ticker: each ticker skips only trades under its own value; zeros equal none; an unknown value raises  PASS")


if __name__ == "__main__":
    test_parse_cache_and_rules()
    test_engine_takes_a_minimum_per_ticker()
    print("symbol rule tests: all passed")
