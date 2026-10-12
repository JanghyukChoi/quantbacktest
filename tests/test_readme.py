"""The README's first example is run as written, and its printed numbers are checked against the ones the README shows.

RM1 the example runs from a fresh folder, writes review.html and prints the three lines shown, with the same numbers (to within rounding)
RM2 the page it writes carries the "Read this first" box, and the JSON of the review has a schema version and no NaN
"""
from __future__ import annotations
import contextlib, io, json, os, re, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _blocks():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    sec = text.split("## See it work", 1)[1].split("\n## ", 1)[0]
    code = re.search(r"```python\n(.*?)```", sec, re.S).group(1)
    shown = re.search(r"```text\n(.*?)```", sec, re.S).group(1)
    return code, shown


def _numbers(s):
    return [float(x) for x in re.findall(r"-?\d+\.\d+", s.replace("%", ""))]


def test_readme_example():
    code, shown = _blocks()
    with tempfile.TemporaryDirectory(prefix="readme-") as d:
        cwd = os.getcwd()
        os.chdir(d)
        out = io.StringIO()
        try:
            ns = {}
            import warnings
            with warnings.catch_warnings(), contextlib.redirect_stdout(out):
                warnings.simplefilter("ignore")
                exec(compile(code, "README.md", "exec"), ns)
            page = Path("review.html").read_text(encoding="utf-8")
            rv = ns["rv"]
            data = json.loads(rv.to_json(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"{c} in the JSON")))           # NaN and Infinity are not JSON
        finally:
            os.chdir(cwd)
    got, want = out.getvalue().strip().splitlines(), shown.strip().splitlines()
    assert len(got) == len(want) == 3, got
    for g, w in zip(got, want):
        a, b = _numbers(g), _numbers(w)
        assert len(a) == len(b) and all(abs(x - y) <= 0.03 for x, y in zip(a, b)), (g, w)           # rounding may move the last digit on another platform
        assert re.sub(r"-?\d+\.\d+", "#", g) == re.sub(r"-?\d+\.\d+", "#", w), (g, w)
    print("RM1 the README example runs as written and prints the numbers the README shows  PASS")
    assert "Read this first" in page and "Deflated Sharpe" in page
    assert data["schema"] == "pitbacktest/portfolio-review" and data["schema_version"] == 1 and isinstance(data["notes"], list)
    print("RM2 the page has the box that says what limits the numbers; the JSON has a schema version and no NaN  PASS")


def test_agents_workflow():
    """The workflow AGENTS.md tells an assistant to follow runs as written, on a vendor-style CSV."""
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) == 3
    with tempfile.TemporaryDirectory(prefix="agents-") as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            import numpy as np
            import pandas as pd
            rng = np.random.default_rng(1)
            days = pd.bdate_range("2019-01-01", periods=700)
            rows = []
            for i in range(40):
                px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(days))))
                f = pd.DataFrame({"Date": days.strftime("%m/%d/%Y"), "Symbol": f"S{i:02d}", "Adj Close": px.round(4), "Volume": rng.integers(1e5, 1e6, len(days))})
                if i == 3:
                    f.loc[50, "Adj Close"] = 0.0                      # a bad print
                if i == 5:
                    f = pd.concat([f, f.iloc[[7]]])                   # a duplicated row
                rows.append(f)
            pd.concat(rows).sample(frac=1, random_state=0).to_csv("prices.csv", index=False)
            ns = {}
            out = io.StringIO()
            import warnings
            with warnings.catch_warnings(), contextlib.redirect_stdout(out):
                warnings.simplefilter("ignore")
                for code in blocks:
                    exec(compile(code, "AGENTS.md", "exec"), ns)
        finally:
            os.chdir(cwd)
    s_ = ns["s"]
    assert "reversal" in out.getvalue() and isinstance(s_["notes"], list) and (s_["variants"] is None or s_["variants"]["trials"] == 1)
    assert set(s_) >= {"notes", "figures", "sharpe_interval_95", "luck", "variants", "market", "delisting_stress", "reading"} and s_["figures"]["sharpe"] is not None
    assert "market_worst_day" in ns["panel"].audit() and ns["rv"].to_dict()["summary"] == s_
    for need in ("assert_causal", "recipes.describe()", "deflated", "docs/verification_status.md", "extra_trials", "primary_horizon", "summary()"):
        assert need in text
    print("RM3 the workflow in AGENTS.md runs as written on a vendor-style CSV  PASS")


if __name__ == "__main__":
    test_readme_example()
    test_agents_workflow()
    print("readme tests: all passed")
