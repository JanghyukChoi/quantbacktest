"""Build REPORT.md from results_preregistered.json and results_amended.json. No number is typed by hand."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
P = json.load(open(H / "results_preregistered.json"))
A = json.load(open(H / "results_amended.json"))


def f(x, n=2):
    return "n/a" if x is None else f"{x:.{n}f}"


def gates(d):
    return "\n".join(f"| {k} | {g['label']} | {f(g['value'], 3)} | {'pass' if g['pass'] else '**fail**'} |" for k, g in d["gates"].items())


def trial_table(d):
    rows = ["| trial | A primary | B survivors only | C no funding | D flat 2 bp, no funding | cost bp/yr | funding bp/yr | delistings held |",
            "|---|---|---|---|---|---|---|---|"]
    for t, r in d["trials"].items():
        rows.append(f"| {t} | {f(r['A'])} | {f(r['B'])} | {f(r['C'])} | {f(r['D'])} | {r['A_cost_annual_bp']:.0f} | {r['A_funding_annual_bp']:+.0f} | {r['A_delist_events_held']} |")
    return "\n".join(rows)


def years(d):
    return ", ".join(f"{y}: {v['ret'] * 100:+.0f}%" + ("" if v["days"] >= 360 else f" ({v['days']} days)") for y, v in d["year_stats"].items())


def bias_rows(d):
    t = d["trials"]
    out = []
    for k, name in (("B", "survivors only"), ("C", "no funding"), ("D", "flat 2 bp cost, no funding")):
        diffs = [r[k] - r["A"] for r in t.values()]
        out.append(f"| {name} | {sum(diffs) / len(diffs):+.2f} | {min(diffs):+.2f} | {max(diffs):+.2f} |")
    return "\n".join(out)


u, ub = A["universe_A"], A["universe_B"]
b = A["trials"][A["best_trial"]]
part = A["participation"]["10M"]
md = f"""# Report: cross-sectional factors on Binance perpetuals

Generated from `results_preregistered.json` and `results_amended.json` by `make_report.py`.
Read `PREREGISTRATION.md` first, then `AMENDMENT_1.md` (the preregistered cost model turned out to be invalid).

Signal dates {A['first_signal_date']} to {A['last_signal_date']} ({A['days']} days). Universe A (point in time, delisted
included): {u['symbols_used']} contracts, {u['delisted_in_panel']} of them delisted, median {u['eligible_median']:.0f} eligible per day.
Universe B (survivors only): {ub['symbols_used']} contracts, median {ub['eligible_median']:.0f} eligible per day.

## Verdict

| run | best trial | verdict |
|---|---|---|
| preregistered (invalid cost model) | {P['best_trial']} | {'validated' if P['verdict'] else 'not validated'} |
| amended (fixed 2 bp half spread) | {A['best_trial']} | {'validated' if A['verdict'] else '**not validated**'} |

### Amended run, gates on the best trial
| gate | criterion | value | result |
|---|---|---|---|
{gates(A)}

### Preregistered run, gates on its best trial
| gate | criterion | value | result |
|---|---|---|---|
{gates(P)}

## Best trial of the amended run: {A['best_trial']}
- Net Sharpe {f(b['A'])}, CAGR {b['A_CAGR'] * 100:.0f}%, max drawdown {b['A_MDD'] * 100:.0f}%, daily one-way turnover {b['A_turnover_daily']:.2f}.
- Deflated Sharpe {f(A['dsr']['dsr'], 3)}: the Sharpe a lucky best-of-{A['dsr']['trials']} would show is {f(A['dsr']['sharpe_luck_benchmark'])}, above the observed {f(A['dsr']['sharpe'])}.
- PBO {f(A['pbo']['pbo'], 3)} (in-sample best Sharpe {f(A['pbo']['is_best_sharpe_mean'])}, out-of-sample Sharpe of that same column {f(A['pbo']['oos_of_best_sharpe_mean'])}).
- Calendar years: {years(A)}.
- Funding paid: {b['A_funding_annual_bp']:+.0f} bp per year (negative = received). Costs: {b['A_cost_annual_bp']:.0f} bp per year.
- Stress (net Sharpe): {', '.join(f'{k} {f(v)}' for k, v in A['stress'].items())}.
- **Cost sensitivity** (flat one-way cost, with funding): {', '.join(f'{k.replace("_oneway", "")} {f(v)}' for k, v in A['cost_sensitivity_best'].items())}. Break-even on this grid: {A['breakeven_oneway_bp']:g} bp one way.
- Capacity: with {part['aum_usd'] / 1e6:.0f}M per leg the 95th percentile trade is {part['participation_p95'] * 100:.2f}% of that contract's daily turnover; the account size at which it reaches 1% is about {part['aum_at_1pct_p95'] / 1e6:.0f}M.

## All 12 trials, net Sharpe (amended run)
{trial_table(A)}

## How much does each shortcut change the answer? (amended run, difference in net Sharpe versus A, over 12 trials)
| shortcut | mean | smallest | largest |
|---|---|---|---|
{bias_rows(A)}

## Preregistered run, all 12 trials (invalid cost model, kept for the record)
{trial_table(P)}
"""
(H / "REPORT.md").write_text(md, encoding="utf-8")
print("REPORT.md", len(md), "chars")
