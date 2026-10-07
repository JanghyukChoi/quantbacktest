"""Turn results_amendment1.json into REPORT_AMENDMENT1.md. No number is typed by hand."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "results_amendment1.json").read_text())
T = r["trials"]


def reading(d: float, n: float) -> str:
    """The convention fixed in AMENDMENT_1.md: same sign and at least half the size means not explained."""
    return "not explained" if (d * n > 0 and abs(n) >= 0.5 * abs(d)) else "largely exposure"


L = ["# Report: Amendment 1 (post hoc)", "",
     f"Generated from `results_amendment1.json`. Read `AMENDMENT_1.md` first: everything here is exploratory and was added after "
     f"the first result was seen. Bootstrap: {r['n_boot']} paired stationary resamples, seed {r['seed']}.", "",
     f"Reproduction check: the 8 original trials were recomputed and differ from `results.json` by at most "
     f"{r['reproduction_max_abs_diff']:.1e} (Sharpe and CAGR).", ""]
counts = {"excl0": 0, "total": 0, "not explained": 0, "largely exposure": 0}
for cname in ("15bp", "0bp"):
    L += [f"## {cname} one-way cost", "",
          "| trial | A Sharpe | B Sharpe | B - A | 95% interval | p (boot) | neutral B - A | 95% interval | reading |",
          "|---|---|---|---|---|---|---|---|---|"]
    for tid, v in T.items():
        a, n = v[cname]["raw"], v[cname]["neutral"]
        rd = reading(a["diff"], n["diff"])
        counts["total"] += 1
        counts["excl0"] += (a["diff_lo"] > 0) or (a["diff_hi"] < 0)
        counts[rd] += 1
        L.append(f"| {tid} | {a['A_sharpe']:.2f} | {a['B_sharpe']:.2f} | {a['diff']:+.2f} | [{a['diff_lo']:+.2f}, {a['diff_hi']:+.2f}] | "
                 f"{a['p_boot']:.3f} | {n['diff']:+.2f} | [{n['diff_lo']:+.2f}, {n['diff_hi']:+.2f}] | {rd} |")
    L.append("")
L += ["## Summary", "",
      f"- Of {counts['total']} comparisons, **{counts['excl0']}** have a 95% interval for `B - A` that excludes zero "
      f"(about {0.05 * counts['total']:.1f} would by chance alone if there were no real difference).",
      f"- Reading rule on the neutralised factors: **{counts['not explained']}** not explained by the other exposures, "
      f"**{counts['largely exposure']}** largely exposure.", "",
      "## Market exposure of the long-short portfolios (15 bp, raw factors)", "",
      "Regression of each net return series on the equal-weight return of the eligible stocks of universe A (Newey-West).", "",
      "| trial | beta A | beta B | alpha A (annual, t) | alpha B (annual, t) |", "|---|---|---|---|---|"]
for tid, v in T.items():
    a = v["15bp"]["raw"]
    L.append(f"| {tid} | {a['A_beta']:+.2f} | {a['B_beta']:+.2f} | {a['A_alpha_annual']*100:+.1f}% ({a['A_alpha_t']:+.1f}) | "
             f"{a['B_alpha_annual']*100:+.1f}% ({a['B_alpha_t']:+.1f}) |")
L.append("")
(HERE / "REPORT_AMENDMENT1.md").write_text("\n".join(L), encoding="utf-8")
print("REPORT_AMENDMENT1.md", len("\n".join(L)))
