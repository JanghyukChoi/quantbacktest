"""Automatic mutation testing: break the code in small ways, one at a time, and see whether the tests notice.

usage: automut.py MODULE [--n 60] [--seed 0] [--jobs 6] [--timeout 240] [--full] [--python PATH]
  MODULE   path inside the package, e.g. pitbacktest/robustness.py
  --n      how many mutants to try, drawn at random from all that the operators give (0 = all)
  --full   also run the slow files (test_robustness, test_metamorphic, test_intraday); off by default for speed

Operators: a comparison turned over (< <= > >= == !=), + <-> -, * <-> /, and <-> or, an integer constant +1, `not` removed, a True/False flipped, a return value replaced by None.
Each mutant runs in its own copy of package and tests, test files fastest first, stopping at the first failing one: a failure or timeout kills the mutant, a clean pass is a survivor.
Output: score = killed / (killed + survived), and the survivors with file, line and what was changed, for a person (or an AI) to triage: a real gap gets a test, an equivalent mutant gets a note.
Docstring-only and logging changes are not mutated. Mutants are formed with ast.unparse, so comments in the copy are lost; the copy is thrown away."""
from __future__ import annotations
import argparse, ast, concurrent.futures as cf, os, random, shutil, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLOW = {"test_robustness.py", "test_metamorphic.py", "test_intraday.py", "test_readme.py"}
CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
ARI = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}


def candidates(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP:
            out.append((node, "cmp", f"{type(node.ops[0]).__name__} -> {CMP[type(node.ops[0])].__name__}"))
        elif isinstance(node, ast.BinOp) and type(node.op) in ARI:
            out.append((node, "ari", f"{type(node.op).__name__} -> {ARI[type(node.op)].__name__}"))
        elif isinstance(node, ast.BoolOp):
            out.append((node, "bool", "and <-> or"))
        elif isinstance(node, ast.Constant) and type(node.value) is int and not isinstance(node.value, bool) and 0 <= node.value < 1000:
            out.append((node, "int", f"{node.value} -> {node.value + 1}"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            out.append((node, "flag", f"{node.value} -> {not node.value}"))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            out.append((node, "not", "not removed"))
        elif isinstance(node, ast.Return) and node.value is not None and not (isinstance(node.value, ast.Constant) and node.value.value is None):
            out.append((node, "ret", "return value -> None"))
    # skip anything inside a docstring-only expression or a logging/warn call
    skip = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) in {"warn", "note", "_note", "print", "warning"}:
            skip.update(id(c) for c in ast.walk(n))
    return [c for c in out if id(c[0]) not in skip]


def mutate(src, index):
    tree = ast.parse(src)
    node, kind, what = candidates(tree)[index]
    if kind == "cmp":
        node.ops = [CMP[type(node.ops[0])]()]
    elif kind == "ari":
        node.op = ARI[type(node.op)]()
    elif kind == "bool":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif kind == "int":
        node.value += 1
    elif kind == "flag":
        node.value = not node.value
    elif kind == "not":
        node.__class__, node.__dict__ = node.operand.__class__, dict(node.operand.__dict__)
    elif kind == "ret":
        node.value = ast.Constant(None)
    return ast.unparse(ast.fix_missing_locations(tree)), node.lineno, what


def run_one(args):
    module, index, python, timeout, full = args
    work = Path(tempfile.mkdtemp(prefix="automut-"))
    try:
        shutil.copytree(REPO / "pitbacktest", work / "pitbacktest", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "tests", work / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        for extra in ("pyproject.toml", "README.md", "AGENTS.md", "LICENSE", "CHANGELOG.md"):
            if (REPO / extra).exists():
                shutil.copy(REPO / extra, work / extra)
        for d in ("docs", ".github"):
            if (REPO / d).exists():
                shutil.copytree(REPO / d, work / d, ignore=shutil.ignore_patterns("__pycache__", ".market_validation_*"))
        target = work / module
        try:
            if index < 0:                                                # baseline: the code unparsed and unchanged must pass everything
                new, line, what = ast.unparse(ast.parse(target.read_text())), 0, "baseline"
            else:
                new, line, what = mutate(target.read_text(), index)
            compile(new, str(target), "exec")
        except Exception as e:                                           # a mutant that does not even compile is not a test of anything
            return index, "invalid", 0, f"{type(e).__name__}", ""
        target.write_text(new)
        env = {**os.environ, "PITBT_FAST": "1", "PITBT_EXAMPLES": "12", "PYTHONDONTWRITEBYTECODE": "1"}
        files = sorted((work / "tests").glob("test_*.py"), key=lambda p: (p.name in SLOW, p.stat().st_size))
        files = [f for f in files if full or f.name not in SLOW]
        t0 = time.time()
        for f in files:
            try:
                r = subprocess.run([python, "-W", "ignore", str(f)], cwd=work, env=env, capture_output=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return index, "killed", line, what, f"{f.name} (timeout)"
            if r.returncode != 0:
                return index, "killed", line, what, f.name
        return index, "survived", line, what, f"{time.time() - t0:.0f}s all {len(files)} files passed"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module"); ap.add_argument("--n", type=int, default=60); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--timeout", type=int, default=240); ap.add_argument("--full", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    src = (REPO / a.module).read_text()
    total = len(candidates(ast.parse(src)))
    idx = list(range(total))
    random.Random(a.seed).shuffle(idx)
    idx = idx[:a.n] if a.n else idx
    print(f"{a.module}: {total} possible mutants, trying {len(idx)} with {a.jobs} workers", flush=True)
    base = run_one((a.module, -1, a.python, a.timeout, a.full))
    if base[1] != "survived":
        sys.exit(f"the unchanged code already fails ({base[4]}): fix that before measuring anything")
    print(f"baseline passes ({base[4]})", flush=True)
    res = []
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(run_one, [(a.module, i, a.python, a.timeout, a.full) for i in idx]):
            res.append(r)
            if r[1] == "survived":
                print(f"  SURVIVED line {r[2]}: {r[3]}   ({r[4]})", flush=True)
    k = sum(r[1] == "killed" for r in res); s = sum(r[1] == "survived" for r in res); inv = sum(r[1] == "invalid" for r in res)
    print(f"\n{a.module}: killed {k}, survived {s}, invalid {inv}; score {k / max(1, k + s):.1%}")


if __name__ == "__main__":
    main()
