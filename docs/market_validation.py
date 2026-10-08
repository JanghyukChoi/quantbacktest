"""The same validation of the engines on real data in three markets: Korea (KRX), the US (Tiingo sample) and Binance USDT-M perpetuals.

    python docs/market_validation.py krx|us|crypto      # one market (about 3.5 GB of memory for KRX; run them one at a time)
    python docs/market_validation.py all                # the three, one process each, then writes docs/market_validation.md
    python docs/market_validation.py report             # rewrite the markdown from the saved results

It needs the data caches on disk (see the adapters) and no network. It checks the **method**, not any strategy: every check has a stated pass rule and the exit code is
non-zero if one fails. The performance level of the test factor (a 5-day reversal) is deliberately not printed: only how much each extra spread takes off it.

  V1 null        a factor shuffled across securities each day, zero cost: the share of Sharpe intervals that exclude 0 is near its nominal level and the mean is near 0
  V2 timing      a perfect-foresight factor explodes; the return from the signal day to the entry day does not pay at an entry lag of 1 and does at a lag of 0
  V3 costs       the net Sharpe falls as the spread rises
  V4 determinism the same inputs give the same result to the last bit; the data fingerprint is stable
  V5 identities  masks that are all True, a gross cap with nothing blocked and no freeze option give exactly the plain result (on real data)
  V6 realism     the trade masks, the suspension markdown and the gross cap run, and their metrics are finite and consistent
  V7 data        prices are positive and returns finite, nothing is duplicated; in the cleaned KRX series no return beyond +100% is left (the others only report the count)

The shuffled null uses 40 seeds: with a nominal 5 percent and the bootstrap's known 93 percent coverage the number of rejections should be 0 to 8 (a binomial band
with a false-alarm rate under 1 percent)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

import pitbacktest as q                                     # noqa: E402
from pitbacktest import execution as ex                       # noqa: E402
from pitbacktest.core.estimators import shuffle_columns       # noqa: E402

N_SEEDS = 40
REJECT_BAND = (0, 8)


def build(market: str):
    """(panel, settings): the panel for a market and the knobs its checks use."""
    if market == "krx":
        from pitbacktest.adapters.krx import build_krx_panel
        P = build_krx_panel(os.path.expanduser("~/.cache/quantbt/krx"), start="2016-01-01", drop_suspect_above=1.0)
        return P, dict(limit=0.30, label="KRX 2016- (price limit 30% assumed)")
    if market == "us":
        from pitbacktest.adapters import tiingo
        from pitbacktest.equity import load_us_master
        full = os.path.expanduser("~/.cache/quantbt/tiingo_full")
        plain = os.path.expanduser("~/.cache/quantbt/tiingo")
        store = full if (os.path.isdir(full) and len(os.listdir(full)) >= len(os.listdir(plain))) else plain
        names = sorted(f[:-5] for f in os.listdir(plain) if f.endswith(".json"))
        P = tiingo.build_tiingo_panel(store, load_us_master(), names, start="2013-01-01")
        return P, dict(limit=None, label=f"US Tiingo sample 2013- ({'full fields' if store == full else 'close only'}, {len(names)} tickers)")
    if market == "crypto":
        from pitbacktest.crypto import ArchiveStore, panel as cp
        root = os.path.expanduser("~/.cache/quantbt/binance_um")
        names = sorted(f[:-4] for f in os.listdir(root + "/daily") if f.endswith(".pkl"))
        P = cp.build_panel(ArchiveStore(root), symbols=names, start="2021-01-01", min_adv_usd=5e6, min_age_days=60, live=set(names))
        return P, dict(limit=None, label="Binance USDT-M 2021-")
    raise SystemExit(f"unknown market {market!r}")


def run_market(market: str) -> dict:
    t0 = time.time()
    P, cfg = build(market)
    c = P.close
    ppy = P.periods_per_year
    out = {"market": market, "label": cfg["label"], "shape": list(c.shape), "periods_per_year": ppy, "checks": [], "numbers": {}}

    def check(name: str, ok: bool, detail: str) -> None:
        out["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        print(("  ok   " if ok else "  FAIL ") + name + "  [" + detail + "]", flush=True)

    kw = dict(long_q=0.2, short_q=0.2, hold=5, benchmark=None, grid=False)
    rev = -c.pct_change(5, fill_method=None)

    def run(p, f, spread=20.0, **extra):
        return q.backtest_portfolio(p, f, spread_bp=spread, **{**kw, **extra})

    # V1 null
    rej, shs = 0, []
    for s in range(N_SEEDS):
        r = run(P, shuffle_columns(rev, P.eligible, np.random.default_rng(s)), spread=0.0)
        ci = r.sharpe_ci(n=300)
        rej += not (ci["lo"] <= 0 <= ci["hi"])
        shs.append(r.metrics["Sharpe"])
    se = np.std(shs, ddof=1) / np.sqrt(N_SEEDS)
    check("V1 null: Sharpe intervals that exclude 0", REJECT_BAND[0] <= rej <= REJECT_BAND[1], f"{rej} of {N_SEEDS} (band {REJECT_BAND[0]} to {REJECT_BAND[1]})")
    check("V1 null: mean Sharpe near 0", abs(np.mean(shs)) <= 3 * se, f"mean {np.mean(shs):+.3f}, standard error {se:.3f}")
    out["numbers"]["null_rejections"] = int(rej)

    # V2 timing
    lag = P.entry_lag
    fut = c.shift(-(lag + 1)) / c.shift(-lag) - 1
    leak = c.shift(-1) / c - 1
    sh_fore = run(P, fut, spread=0.0, hold=1).metrics["Sharpe"]
    sh_leak1 = run(P, leak, spread=0.0, hold=1).metrics["Sharpe"]
    sh_leak0 = run(replace(P, entry_lag=0), leak, spread=0.0, hold=1).metrics["Sharpe"]
    check("V2 timing: perfect foresight explodes", sh_fore > 20, f"Sharpe {sh_fore:.1f}")
    check("V2 timing: the day before entry does not pay at lag 1", abs(sh_leak1) < 3, f"Sharpe {sh_leak1:+.2f}")
    check("V2 timing: the same signal pays at lag 0", sh_leak0 > 20, f"Sharpe {sh_leak0:.1f}")

    # V3 costs
    sp = {x: run(P, rev, spread=x).metrics["Sharpe"] for x in (0, 10, 30, 60)}
    v3 = list(sp.values())
    drops = [a - b for a, b in zip(v3[:-1], v3[1:])]
    # only the drops are printed, not the Sharpe levels: a drop measures what a spread costs the turnover of the test factor, a level would be that factor's performance
    check("V3 costs: Sharpe falls as the spread rises", all(d > 0 for d in drops), "falls at each of 4 spreads (0, 10, 30, 60 bp): by " + ", ".join(f"{d:.2f}" for d in drops))

    # V4 determinism
    a, b = run(P, rev), run(P, rev)
    check("V4 determinism: two runs are identical", np.array_equal(a.net_returns.to_numpy(), b.net_returns.to_numpy()), "net returns equal to the last bit")
    check("V4 determinism: the data fingerprint is stable", P.fingerprint() == replace(P).fingerprint(), P.fingerprint())

    # V5 identities
    allok = pd.DataFrame(True, index=P.dates, columns=P.tickers)
    Pt = replace(P, can_buy=allok, can_sell=allok)
    base = run(P, rev)
    same = run(Pt, rev, cap_gross=True, freeze_days=None)
    check("V5 identities: all-True masks, an idle gross cap and no freeze option equal the plain result", np.array_equal(base.net_returns.to_numpy(), same.net_returns.to_numpy()) and same.metrics["blocked_trades"] == 0,
          f"max difference {np.abs(base.net_returns.to_numpy() - same.net_returns.to_numpy()).max():.1e}")

    # V6 realism
    cb, cs = ex.tradability(P, limit=cfg["limit"])
    Pm = replace(P, can_buy=cb, can_sell=cs)
    elig = P.eligible.to_numpy()
    share_buy = 1 - cb.to_numpy()[elig].mean()
    r_mask = run(Pm, rev)
    r_all = run(Pm, rev, freeze_days=60, freeze_return=-0.35, cap_gross=True)
    keys = ("blocked_trades", "blocked_turnover_share", "mean_stuck_weight", "longest_freeze_days")
    finite = all(np.isfinite(r_mask.metrics[k]) for k in keys) and all(np.isfinite(r_all.metrics[k]) for k in ("freeze_markdown_annual_bp", "mean_free_scale", "gross_CAGR"))
    check("V6 realism: masks, markdown and gross cap run with finite metrics", finite, f"blocked share of turnover {r_mask.metrics['blocked_turnover_share']:.3%}, mean stuck weight {r_mask.metrics['mean_stuck_weight']:.4f}, longest freeze {r_mask.metrics['longest_freeze_days']} days")
    g_mask, g_cap = np.abs(r_mask.holdings).sum(axis=1).mean(), np.abs(r_all.holdings).sum(axis=1).mean()
    check("V6 realism: the gross cap never raises the gross exposure", g_cap <= g_mask + 1e-9, f"{g_mask:.3f} -> {g_cap:.3f}")
    check("V6 realism: a markdown never adds return", r_all.metrics["freeze_markdown_annual_bp"] <= 0, f"{r_all.metrics['freeze_markdown_annual_bp']:+.0f} bp a year, {r_all.metrics['freeze_markdown_events']} events")
    out["numbers"].update(blocked_buy_share_of_eligible=float(share_buy), blocked_turnover_share=float(r_mask.metrics["blocked_turnover_share"]),
                          mean_stuck_weight=float(r_mask.metrics["mean_stuck_weight"]), longest_freeze_days=int(r_mask.metrics["longest_freeze_days"]),
                          gross_without_cap=float(g_mask), gross_with_cap=float(g_cap))
    if P.open is not None:
        r_open = run(ex.at_prices(P, "open"), rev)
        check("V6 realism: trading at the open runs and differs from the close", np.isfinite(r_open.metrics["Sharpe"]) and not np.array_equal(r_open.net_returns.to_numpy(), base.net_returns.to_numpy()), "open and close results differ")

    # V7 data
    r1 = c.pct_change(fill_method=None)
    big = int((r1 > 1.0).sum().sum())
    dup = bool(c.index.has_duplicates or c.columns.has_duplicates)
    check("V7 data: no duplicated dates or tickers, panel not empty", (not dup) and c.shape[0] > 100 and c.shape[1] > 10, f"{c.shape[0]} dates x {c.shape[1]} tickers")
    sus = P.meta.get("suspect_returns")
    px = c.to_numpy(float)
    rr = r1.to_numpy(float)
    check("V7 data: prices are positive and returns are finite", bool((px[np.isfinite(px)] > 0).all() and np.isfinite(rr[~np.isnan(rr)]).all()), "no zero or negative price, no infinite return")
    if market == "krx":
        check("V7 data: the cleaned KRX series has no daily return beyond +100%", big == 0,
              f"{big} left; {len(sus)} were listed as suspect in this panel's raw chain, {int(sus.after_suspension.sum())} of them after zero or missing volume")
    else:
        print(f"  info V7 data: {big} of {int(r1.notna().sum().sum()):,} observed daily returns are beyond +100% (not a pass rule: real pumps exist)", flush=True)
    out["numbers"].update(returns_beyond_100pct=big, delist_flags=int(P.delist_after.to_numpy().sum()) if P.delist_after is not None else None)
    out["seconds"] = round(time.time() - t0)
    out["passed"] = all(x["ok"] for x in out["checks"])
    return out


def write_report(results: list[dict]) -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE.parent, capture_output=True, text=True).stdout.strip()
    lines = ["# Market validation: the same checks on real data in three markets", "",
             f"Generated by `docs/market_validation.py` at commit `{commit}`, {time.strftime('%Y-%m-%d')}, Python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}.",
             "It checks the **method** (calibration, timing, costs, determinism, identities, realism, data), not any strategy; the Sharpe ratios printed are those of a shuffled factor (which should be about 0) and of oracle factors that look into the future, and for costs only the drop between spreads, never the level. "
             "Pass rules are in the docstring of the script. The data are local caches (KRX daily files, a 466-ticker Tiingo sample, Binance USDT-M daily bars).", "",
             "| Check | " + " | ".join(r["label"] for r in results) + " |", "|---|" + "---|" * len(results)]
    names = [c["name"] for c in results[0]["checks"]]
    for n in names:
        row = []
        for r in results:
            c = next((x for x in r["checks"] if x["name"] == n), None)
            row.append("n/a" if c is None else ("pass: " if c["ok"] else "**FAIL**: ") + c["detail"])
        lines.append(f"| {n} | " + " | ".join(row) + " |")
    lines += ["", "## What the realism numbers say", "", "| | " + " | ".join(r["label"] for r in results) + " |", "|---|" + "---|" * len(results)]
    def num(r, k, fmt):
        v = r["numbers"].get(k)
        return "n/a" if v is None else format(v, fmt)
    for k, lab, fmt in (("blocked_buy_share_of_eligible", "Eligible name-days on which a buy was blocked (halt or limit lock)", ".3%"), ("blocked_turnover_share", "Share of the turnover asked for that was blocked", ".1%"),
                        ("mean_stuck_weight", "Mean weight held where the target said otherwise (units of capital, both legs)", ".3f"), ("longest_freeze_days", "Longest run of days a wanted trade in one name was blocked", "d"),
                        ("gross_without_cap", "Mean gross exposure with trade masks, no cap", ".3f"), ("gross_with_cap", "Mean gross exposure with the gross cap", ".3f"),
                        ("returns_beyond_100pct", "Daily returns beyond +100% in the panel", "d"), ("delist_flags", "Securities flagged as delisted", "d")):
        lines.append(f"| {lab} | " + " | ".join(num(r, k, fmt) for r in results) + " |")
    lines += ["", "## Limits", "",
              "- The KRX price limit (30%) is an assumption, not read from the data; the US and crypto runs assume none.",
              "- The US panel is a 466-ticker random sample from a free plan; its bankruptcies are almost absent (see `docs/survivorship.md`).",
              "- The US adjusted close is rounded: names whose adjusted price falls below a cent give daily returns of thousands of times (234 beyond +100% in the sample, 19 beyond 10x). "
              "The eligibility rule (raw close of at least one dollar) keeps them out of every strategy here, but a study of penny stocks on this series would be wrong.",
              "- The shuffled null checks one factor family on one cost level; it says nothing about heavy tails, regimes or strongly correlated sectors.",
              "- Passing these checks does not say a strategy works. It says the engines measure what they claim to on these data.", ""]
    (HERE / "market_validation.md").write_text("\n".join(lines))
    (HERE / "market_validation_results.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "report":                                  # rewrite the markdown from the saved results without rerunning anything
        write_report(json.loads((HERE / "market_validation_results.json").read_text()))
        sys.exit(0)
    if arg == "all":
        results = []
        for m in ("krx", "us", "crypto"):
            print(f"== {m}", flush=True)
            p = subprocess.run([sys.executable, __file__, m], capture_output=False)
            f = HERE / f".market_validation_{m}.json"
            if f.exists():
                results.append(json.loads(f.read_text()))
                f.unlink()
        write_report(results)
        sys.exit(0 if results and all(r["passed"] for r in results) and len(results) == 3 else 1)
    res = run_market(arg)
    (HERE / f".market_validation_{arg}.json").write_text(json.dumps(res, default=str))
    sys.exit(0 if res["passed"] else 1)
