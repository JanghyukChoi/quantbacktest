"""The version in pyproject.toml, in `pitbacktest.__version__` and in CHANGELOG.md must agree; the declared Python floor must match
what the CI matrix and the classifiers claim."""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pitbacktest


def _meta() -> dict:
    """The few pyproject fields these tests need, read with regular expressions: `tomllib` only exists from Python 3.11 and the
    package supports 3.10."""
    t = (ROOT / "pyproject.toml").read_text()
    proj = t[t.index("[project]"):t.index("[project.optional-dependencies]")]
    arr = lambda key: re.findall(r'"([^"]+)"', re.search(rf"^{key}\s*=\s*\[(.*?)^\]|^{key}\s*=\s*\[(.*?)\]", proj, re.S | re.M).group(0))
    return {"name": re.search(r'^name = "([^"]+)"', proj, re.M).group(1), "version": re.search(r'^version = "([^"]+)"', proj, re.M).group(1),
            "requires-python": re.search(r'^requires-python = "([^"]+)"', proj, re.M).group(1),
            "classifiers": arr("classifiers"), "dependencies": arr("dependencies")}


def test_versions_agree():
    meta = _meta()
    assert meta["name"] == pitbacktest.__name__ == "pitbacktest"
    assert meta["version"] == pitbacktest.__version__, (meta["version"], pitbacktest.__version__)
    top = re.search(r"^## (\d+\.\d+\.\d+)", (ROOT / "CHANGELOG.md").read_text(), re.M).group(1)
    assert top == meta["version"], f"CHANGELOG top entry {top} != {meta['version']}"
    print(f"P1 pyproject, __version__ and CHANGELOG all say {meta['version']}  PASS")


def test_python_floor_is_consistent():
    meta = _meta()
    floor = re.search(r">=\s*3\.(\d+)", meta["requires-python"]).group(1)
    classifiers = [c for c in meta["classifiers"] if c.startswith("Programming Language :: Python :: 3.")]
    lowest = min(int(c.rsplit(".", 1)[1]) for c in classifiers)
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert lowest == int(floor), (lowest, floor)
    assert f'python: "3.{floor}"' in ci, "the CI matrix must include the declared floor"
    pins = re.findall(r"pandas==([\d.]+) numpy==([\d.]+)", ci)
    assert pins, "the CI matrix must test the declared lower bounds of pandas and numpy"
    deps = " ".join(meta["dependencies"])
    assert f"pandas>={pins[0][0].rsplit('.', 1)[0]}" in deps and f"numpy>={pins[0][1].rsplit('.', 1)[0]}" in deps, (pins, deps)
    print(f"P2 Python floor 3.{floor}, the classifiers and the CI matrix agree; the CI tests the oldest pandas and numpy  PASS")


if __name__ == "__main__":
    test_versions_agree()
    test_python_floor_is_consistent()
    print("packaging tests: all passed")
