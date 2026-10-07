"""REPORT.md from results.json. No number is typed by hand."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
R = json.load(open(H / "results.json"))


def row(t, c):
    r = R["trials"][t][c]
    return f"| {t} | {r['A_sharpe']:.2f} | {r['B_sharpe']:.2f} | {r['B_sharpe'] - r['A_sharpe']:+.2f} | {r['A_cagr'] * 100:.1f}% | {r['B_cagr'] * 100:.1f}% | {(r['B_cagr'] - r['A_cagr']) * 100:+.1f} |"


def block(c):
    d = [R["trials"][t][c]["B_sharpe"] - R["trials"][t][c]["A_sharpe"] for t in R["trials"]]
    dc = [(R["trials"][t][c]["B_cagr"] - R["trials"][t][c]["A_cagr"]) * 100 for t in R["trials"]]
    head = "| trial | A Sharpe | B Sharpe | B - A | A CAGR | B CAGR | B - A (pp) |\n|---|---|---|---|---|---|---|\n"
    return head + "\n".join(row(t, c) for t in R["trials"]) + \
        f"\n\nMean difference in Sharpe: {sum(d) / len(d):+.2f} (range {min(d):+.2f} to {max(d):+.2f}); mean difference in CAGR: {sum(dc) / len(dc):+.1f} pp (range {min(dc):+.1f} to {max(dc):+.1f})."


miss = ", ".join(f"{y}: {v * 100:.0f}%" for y, v in R["missing_share_by_year"].items())
md = f"""# Report: survivorship bias in Korean equity factors

Generated from `results.json` by `make_report.py`. Read `PREREGISTRATION.md` first. Days {R['first_day']} to {R['last_day']} ({R['days']} days).

A = every stock listed each day (point in time, delisted included): {R['securities_A']} securities, median {R['eligible_median_A']:.0f} eligible per day.
B = only stocks still listed on the last day: {R['securities_B']} securities, median {R['eligible_median_B']:.0f} eligible per day.

Share of eligible security-days that belong to securities missing from B, by year: {miss}.

## 15 bp one-way cost
{block('15bp')}

## 0 bp cost
{block('0bp')}
"""
(H / "REPORT.md").write_text(md, encoding="utf-8")
print("REPORT.md", len(md))
