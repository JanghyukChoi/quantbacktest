"""Trial ledger: counts distinct configurations, feeds the deflated Sharpe, and notices tampering."""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pitbacktest as q
from pitbacktest import validation as v
from test_reconcile import _panel


def _run(led, p, f, name, **kw):
    return q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, benchmark=None, grid=False, ledger=led,
                                family="fam", name=name, **kw)


def test_counts_distinct_configs_only():
    with tempfile.TemporaryDirectory(prefix="ledgertest-") as d:
        led = q.Ledger(d)
        p, f = _panel()
        _run(led, p, f, "a", hold=5, spread_bp=4.0)
        _run(led, p, f, "a", hold=5, spread_bp=4.0)               # identical rerun: still one trial
        assert led.n_trials("fam") == 1, led.n_trials("fam")
        _run(led, p, f, "a", hold=10, spread_bp=4.0)              # one parameter changed
        _run(led, p, f, "a", hold=5, spread_bp=8.0)               # cost changed
        assert led.n_trials("fam") == 3
        _run(led, p, -f, "a", hold=5, spread_bp=4.0)              # same label, different factor values
        assert led.n_trials("fam") == 4
        p2, _ = _panel(seed=99)
        _run(led, p2, f, "a", hold=5, spread_bp=4.0)              # different data
        assert led.n_trials("fam") == 5
        assert led.n_trials("other") == 0
        print("L1 distinct configs counted once, any change is a new trial  PASS")


def test_dsr_uses_ledger_count_not_columns():
    with tempfile.TemporaryDirectory(prefix="ledgertest-") as d:
        led = q.Ledger(d)
        p, f = _panel()
        for h in (1, 2, 3, 5, 8, 10, 15, 20):
            _run(led, p, f, f"hold{h}", hold=h, spread_bp=0.0)
        out = led.deflated_sharpe("fam")
        R, _ = led.returns_matrix("fam")
        direct = v.deflated_sharpe(R, trials=8)
        assert out["trials"] == 8 and abs(out["dsr"] - direct["dsr"]) < 1e-12
        assert out["verify"]["ok"]
        print(f"L2 ledger DSR equals direct DSR with trials=8 (dsr {out['dsr']:.3f})  PASS")


def test_chain_detects_edit_and_deletion():
    with tempfile.TemporaryDirectory(prefix="ledgertest-") as d:
        led = q.Ledger(d)
        p, f = _panel()
        for h in (1, 2, 3, 4):
            _run(led, p, f, f"h{h}", hold=h, spread_bp=0.0)
        assert led.verify()["ok"]
        lines = led.file.read_text().splitlines()
        led.file.write_text("\n".join(lines[:1] + lines[2:]) + "\n")           # drop line 2
        bad = led.verify()
        assert not bad["ok"] and bad["first_bad_line"] == 2, bad
        led.file.write_text("\n".join(lines) + "\n")
        assert led.verify()["ok"]
        edited = lines[:]
        edited[1] = edited[1].replace('"hold": 2', '"hold": 9')
        led.file.write_text("\n".join(edited) + "\n")
        assert not led.verify()["ok"]
        print("L3 deleting or editing a middle line breaks the chain  PASS")


def test_ledger_does_not_change_results():
    with tempfile.TemporaryDirectory(prefix="ledgertest-") as d:
        p, f = _panel()
        a = q.backtest_portfolio(p, f, long_q=0.2, short_q=0.2, hold=5, spread_bp=4.0, benchmark=None, grid=False)
        b = _run(q.Ledger(d), p, f, "x", hold=5, spread_bp=4.0)
        assert np.array_equal(a.net_returns.to_numpy(), b.net_returns.to_numpy())
        print("L4 recording does not change the numbers  PASS")


def test_fingerprint_covers_volume_and_market_cap():
    from dataclasses import replace
    p = _panel()[0] if isinstance(_panel(), tuple) else _panel()
    vol = p.close * 1000.0
    a = replace(p, volume=vol, mkt_cap=p.close * 1e6)
    assert a.fingerprint() == replace(p, volume=vol.copy(), mkt_cap=p.close * 1e6).fingerprint()
    assert a.fingerprint() != replace(a, volume=vol * 100).fingerprint()        # the impact model reads volume
    assert a.fingerprint() != replace(a, mkt_cap=p.close * 2e6).fingerprint()
    assert a.fingerprint() != p.fingerprint()
    print("L5 the data fingerprint changes when volume or market cap changes  PASS")


if __name__ == "__main__":
    test_fingerprint_covers_volume_and_market_cap()
    test_counts_distinct_configs_only()
    test_dsr_uses_ledger_count_not_columns()
    test_chain_detects_edit_and_deletion()
    test_ledger_does_not_change_results()
    print("ledger tests: all passed")
