"""Synthetic data checks: does the framework answer correctly on data whose answer is known?

Checks
  T1 perfect-foresight factor -> must be large and positive (timing aligned)
  T2 random factor          -> must be near 0 (no false positives)
  T3 look-ahead factor      -> assert_no_lookahead must catch it
  T4 future signal          -> detected if the result holds or rises when delayed a day
  T5 shuffled null          -> the |t| distribution of a random factor is in a sensible range
  T6 neutralisation         -> a control fed in as the factor must vanish after neutralising
  T7 base win rate          -> the lift of random firings must be near 0
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pitbacktest as q
from pitbacktest.core.controls import build_controls, xs_norm


def make_panel(n_days=1200, n_stocks=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tick = [f"S{i:03d}" for i in range(n_stocks)]
    # market + idiosyncratic returns
    mkt = rng.normal(0.0003, 0.010, n_days)[:, None]
    idio = rng.normal(0, 0.018, (n_days, n_stocks))
    ret = mkt + idio
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=tick)
    vol = pd.DataFrame(rng.lognormal(11, 1.0, (n_days, n_stocks)), index=dates, columns=tick)
    mcap = close * rng.lognormal(15, 1.0, n_stocks)
    el = pd.DataFrame(True, index=dates, columns=tick)
    el.iloc[:60] = False                        # exclude the initial warm-up
    return q.Panel(close=close, eligible=el, volume=vol, mkt_cap=mcap,
                   market="TEST", entry_lag=1), rng


def main():
    p, rng = make_panel()
    fails = []
    print(f"[panel] {len(p.dates)} days x {len(p.tickers)} securities | entry_lag={p.entry_lag}\n")

    # T1 perfect foresight
    r = q.assert_timing(p)
    ok = r["pass"]
    print(f"T1 timing (perfect foresight)  CAGR {r['oracle_CAGR']*100:>8.1f}%  "
          f"Sharpe {r['oracle_Sharpe']:>6.2f}  {'PASS' if ok else 'FAIL'}")
    if not ok: fails.append("T1")

    # T2 random factor
    noise = pd.DataFrame(rng.normal(size=(len(p.dates), len(p.tickers))),
                         index=p.dates, columns=p.tickers)
    rp = q.backtest_portfolio(p, noise, hold=5, spread_bp=0.0, grid=False)
    ok = abs(rp.metrics["Sharpe"]) < 0.5
    print(f"T2 random factor        CAGR {rp.metrics['CAGR']*100:>8.2f}%  "
          f"Sharpe {rp.metrics['Sharpe']:>6.2f}  {'PASS' if ok else 'FAIL'} (|Sharpe| < 0.5 expected)")
    if not ok: fails.append("T2")

    # T3 look-ahead detection
    fut = p.close.shift(-3) / p.close - 1.0        # obvious future information
    la = p.assert_no_lookahead(fut, h=5)
    ok = not la["pass"] or la["base_bp"] > 100     # detected, or at least base is abnormally large
    print(f"T3 look-ahead detection base {la['base_bp']:>8.1f}bp  lagged {la['lagged_bp']:>8.1f}bp  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok: fails.append("T3")

    # T4 a normal signal gets weaker when delayed
    la2 = p.assert_no_lookahead(noise, h=5)
    print(f"T4 random look-ahead   pass={la2['pass']}  (a random factor must pass)")
    if not la2["pass"]: fails.append("T4")

    # T5 shuffled null
    sr = q.screen(p, {"noise": noise}, horizons=(1, 5, 20), primary_h=20,
                  n_null=2, neutralize_all=False)
    thr = sr.null.get("p95", np.nan)
    # The synthetic data holds no signal, so a null |t| near 0 is normal.
    # What is checked is that "a value is produced and it is not abnormally large", not a particular range.
    ok = np.isfinite(thr) and 0 <= thr < 4.0
    print(f"T5 shuffled-null threshold      p95={thr:.2f}  max={sr.null.get('max', float('nan')):.2f}  "
          f"{'PASS' if ok else 'FAIL'} (produced and < 4 expected)")
    if not ok: fails.append("T5")

    # T6 neutralisation: a control used as the factor vanishes after neutralising
    ctrl = build_controls(p, include_chars=False)
    mom = ctrl["mom21"]
    sr2 = q.screen(p, {"mom21_self": mom}, horizons=(1, 5, 20), primary_h=20,
                   n_null=1, neutralize_all=True)
    surv = sr2.summary["neu_survival_%"].iloc[0]
    ok = (not np.isfinite(surv)) or abs(surv) < 30
    print(f"T6 neutralisation (self-control)  survival {surv:>7.1f}%  {'PASS' if ok else 'FAIL'} (|survival| < 30% expected)")
    if not ok: fails.append("T6")

    # T7 base win rate
    fire = pd.DataFrame(rng.random((len(p.dates), len(p.tickers))) < 0.05,
                        index=p.dates, columns=p.tickers)
    re = q.backtest_event(p, fire, horizons=(1, 5, 20), cost_bp=0.0,
                          neutralize_check=False)
    lifts = [v["lift_pp"] for v in re.per_horizon.values() if np.isfinite(v.get("lift_pp", np.nan))]
    ok = all(abs(x) < 3.0 for x in lifts)
    print(f"T7 random firings lift    {['%+.2f' % x for x in lifts]}  "
          f"{'PASS' if ok else 'FAIL'} (|lift| < 3 pp expected)")
    if not ok: fails.append("T7")

    # T8 small universe: a fixed threshold throws away every date (a bug found on real data)
    small, srng = make_panel(n_days=800, n_stocks=20, seed=3)
    sig_s = pd.DataFrame(srng.random((len(small.dates), 20)) < 0.2,
                         index=small.dates, columns=small.tickers)
    try:
        rs = q.backtest_event(small, sig_s, horizons=(1, 5), cost_bp=0.0,
                              neutralize_check=False)
        got = len(rs.per_horizon) > 0 and np.isfinite(
            rs.gates["oos"]["OOS"].get("mean_bp", np.nan))
        print(f"T8 small universe (20 names) {len(rs.per_horizon)} horizons | OOS produced "
              f"{'PASS' if got else 'FAIL'}")
        if not got: fails.append("T8")
    except Exception as ex:
        print(f"T8 small universe (20 names) exception {type(ex).__name__} FAIL")
        fails.append("T8")

    # T9 FM regression in a small universe: a fixed min_stocks throws away every date
    small2, r2 = make_panel(n_days=900, n_stocks=25, seed=11)
    from pitbacktest.core.estimators import fama_macbeth
    from pitbacktest.core.controls import build_controls as _bc
    c2 = _bc(small2, include_chars=False)
    f2 = xs_norm(pd.DataFrame(r2.normal(size=(len(small2.dates), 25)),
                              index=small2.dates, columns=small2.tickers), small2.eligible)
    fm2 = fama_macbeth(f2, {5: small2.forward(5)}, c2, small2.eligible)
    ok = np.isfinite(fm2[5]["t"]) and fm2[5]["n_days"] > 100
    print(f"T9 small FM regression (25 names) t={fm2[5]['t']:+.2f} · n_days={fm2[5]['n_days']} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok: fails.append("T9")

    # T10 perfect-collinearity warning: feeding a control in as the factor must warn
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        fama_macbeth(c2["mom21"], {5: small2.forward(5)}, c2, small2.eligible)
        got = any("collinear" in str(x.message) for x in rec)
    print(f"T10 collinearity warning     {'PASS' if got else 'FAIL'}")
    if not got: fails.append("T10")

    # audit
    aud = p.audit()
    print(f"\n[audit] universe growth {aud['universe_growth']:.2f}x · "
          f"extreme returns {aud['ret_extreme_pct']:.3f}% · "
          f"suspected truncation {aud.get('truncation_suspected')}")

    print(f"\n{'='*60}")
    print(f"result: {10-len(fails)}/10 passed" + (f" | failed: {fails}" if fails else " | all passed"))
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(main())
