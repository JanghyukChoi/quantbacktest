"""Turn results.json into REPORT.md. No number is typed by hand."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "results.json").read_text())
T, A, S, C = r["trials"], r["audit"], r["scenario"], r["config"]


def reading(d: float, n: float) -> str:
    """The convention fixed in PREREGISTRATION.md: same sign and at least half the size means not explained."""
    return "not explained" if (d * n > 0 and abs(n) >= 0.5 * abs(d)) else "largely exposure"


L = ["# Report: a lower bound on survivorship bias in US equity factors, from free data", "",
     "Generated from `results.json`. Read `PREREGISTRATION.md` and `AMENDMENT_1.md` first. **Part 1 and 2 are measured. Part 3 is a scenario: "
     "what is assumed, not what is known.** The measured difference is a lower bound, because free data lacks most bankruptcies.", ""]

L += ["## Part 4: the sample", "",
      f"- Frame: {A['frame_tickers']:,} tickers, {A['frame_delisted_share']:.1%} of them delisted by the last date.",
      f"- Drawn (first {A['drawn']} of the seeded random order): **{A['with_prices']} with prices**, {A['none']} with none "
      f"({', '.join(x['ticker'] + (' (delisted)' if not x['alive'] else ' (alive)') for x in A['none_tickers'])}).",
      f"- Delisted share: frame {A['frame_delisted_share']:.1%}, drawn {A['drawn_delisted_share']:.1%}, obtained {A['obtained_delisted_share']:.1%} "
      f"({A['obtained_delisted']} securities). Flag for missing data concentrated in delisted names (below 70% of the frame's share): "
      f"**{'YES' if A['delisted_share_flag'] else 'no'}**. Inconclusive (fewer than 300 with prices): **{'YES' if A['inconclusive'] else 'no'}**.",
      f"- Securities in A_T: {A['securities_A']} (B: {A['securities_B']}); ever eligible {A['ever_eligible']}; eligible per day: median "
      f"{A['eligible_per_day_median']:.0f}, range {A['eligible_per_day_min']} to {A['eligible_per_day_max']}.",
      f"- **How many distress-type delistings the free data holds:** {A['delisted_last_year_fell_over_50pct']} of the {A['delisted_flagged_in_panel']} "
      f"securities flagged as delisted fell by more than 50% in their last 252 bars ({A['delisted_last_year_fell_over_50pct'] / max(A['delisted_flagged_in_panel'], 1):.0%}).",
      f"- Listing windows cut for ticker reuse: {A['windows_cut_for_ticker_reuse']} tickers. Daily returns above +1000% anywhere in the panel: {A['suspect_returns_gt_10x']} "
      f"(data errors are not audited beyond that). Largest absolute daily returns among eligible securities: "
      + "; ".join(f"{x['security']} {x['date']} {x['ret']:+.0%}" for x in A["largest_abs_daily_returns_among_eligible"]) + ".",
      "- Securities with a bar, by year: " + ", ".join(f"{k}: {v}" for k, v in A["securities_with_a_bar_by_year"].items()) + ".", ""]

counts = {"excl0": 0, "total": 0, "not explained": 0, "largely exposure": 0}
for cname in C["costs"]:
    L += [f"## Parts 1 and 2: {cname} one-way cost", "",
          "| trial | A_T Sharpe | B Sharpe | B - A_T | 95% interval | p (boot) | neutral B - A_T | 95% interval | reading |", "|---|---|---|---|---|---|---|---|---|"]
    for tid, v in T.items():
        a, n = v[cname]["raw"], v[cname]["neutral"]
        rd = reading(a["diff"], n["diff"])
        counts["total"] += 1; counts["excl0"] += (a["diff_lo"] > 0) or (a["diff_hi"] < 0); counts[rd] += 1
        L.append(f"| {tid} | {a['A_sharpe']:.2f} | {a['B_sharpe']:.2f} | {a['diff']:+.2f} | [{a['diff_lo']:+.2f}, {a['diff_hi']:+.2f}] | {a['p_boot']:.3f} | "
                 f"{n['diff']:+.2f} | [{n['diff_lo']:+.2f}, {n['diff_hi']:+.2f}] | {rd} |")
    L.append("")
widths = [v["10bp"]["raw"]["diff_hi"] - v["10bp"]["raw"]["diff_lo"] for v in T.values()]
L += ["**Summary of the measured part.**", "",
      f"- {counts['excl0']} of {counts['total']} intervals for B - A_T exclude zero (about {0.05 * counts['total']:.1f} would by chance alone); they are not independent "
      "(two holds and two cost levels of four factors).",
      f"- The 95% intervals are about {np.mean(widths):.2f} Sharpe wide (average over the trials at 10 bp). A difference smaller than that cannot be told from zero in this "
      "sample, and that is **not** evidence that the bias is zero.",
      f"- Reading rule on the neutralised factors: {counts['not explained']} not explained by the other exposures, {counts['largely exposure']} largely exposure "
      "(a convention for reading; when the raw difference is itself indistinguishable from zero it carries little).", "",
      "### Market exposure (10 bp, raw factors)", "", "Regression of each net return series on the equal-weight return of A_T's eligible securities (Newey-West).", "",
      "| trial | beta A | beta B | alpha A (annual, t) | alpha B (annual, t) |", "|---|---|---|---|---|"]
for tid, v in T.items():
    a = v["10bp"]["raw"]
    L.append(f"| {tid} | {a['A_beta']:+.2f} | {a['B_beta']:+.2f} | {a['A_alpha_annual'] * 100:+.1f}% ({a['A_alpha_t']:+.1f}) | {a['B_alpha_annual'] * 100:+.1f}% ({a['B_alpha_t']:+.1f}) |")

L += ["", "## Part 3: a scenario, not a measurement", "",
      f"Delistings are put back into A_T at annual rates {C['rates']} of the names alive at the start of each year, with a delisting return of {C['delist_returns']}, "
      f"drawn {' or '.join(C['hazards'])}, {C['n_draws']} random draws per cell, at {C['scenario_cost']}. The grid was fixed before the result. "
      "The columns are `Sharpe(B) - Sharpe(A_scenario)` over the grid cells. **They depend on the assumed rates and returns, and some of Tiingo's delistings "
      "are already distress cases, so the scenario partly double counts.**", "",
      "| trial | measured B - A_T | scenario: lowest cell | mean over cells | highest cell | cells where B - A_scenario > 0 |", "|---|---|---|---|---|---|"]
for tid, v in S.items():
    d = [v["B_sharpe"] - c["mean_sharpe_A_scenario"] for c in v["cells"]]
    L.append(f"| {tid} | {v['B_sharpe'] - v['A_T_sharpe']:+.2f} | {min(d):+.2f} | {np.mean(d):+.2f} | {max(d):+.2f} | {sum(x > 0 for x in d)}/{len(d)} |")
L += ["", "By delisting return (mean over all cells and trials of `Sharpe(B) - Sharpe(A_scenario)`):", ""]
for dr in C["delist_returns"]:
    d = [v["B_sharpe"] - c["mean_sharpe_A_scenario"] for v in S.values() for c in v["cells"] if c["delist_return"] == dr]
    L.append(f"- delisting return {dr:+.0%}: {np.mean(d):+.3f}")
for rate in C["rates"]:
    d = [v["B_sharpe"] - c["mean_sharpe_A_scenario"] for v in S.values() for c in v["cells"] if c["annual_rate"] == rate]
    L.append(f"- annual rate {rate:.0%}: {np.mean(d):+.3f}")
L.append("")
(HERE / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
print("REPORT.md", len("\n".join(L)))
