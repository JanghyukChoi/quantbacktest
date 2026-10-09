# Releasing

The order matters because a version published to PyPI or TestPyPI can never be uploaded again (not even after deleting it); a mistake costs a new version number.

## Before every push

Run what CI runs, in the oldest environment as well as the current one (a numpy-2-only function once reached the public CI that way):

```bash
ruff check --select E9,F63,F7,F82,F811,F821,F822,F823 pitbacktest tests studies docs examples
for t in tests/test_*.py; do python -W ignore "$t" || break; done         # in Python 3.10 with pandas==2.0.3 numpy==1.24.4, and in the newest Python
```

And the validation on real data, if you changed an engine or an adapter: `python docs/market_validation.py all` (needs the local caches; read `docs/market_validation.md`).

## One-time setup (TestPyPI, then PyPI)

No token is stored in GitHub: the Release workflow publishes by trusted publishing. For each index:

1. Make an account (with two-factor authentication) on test.pypi.org, later on pypi.org.
2. Account settings, Publishing, add a **pending publisher**: project `pitbacktest`, owner `JanghyukChoi`, repository `quantbacktest`, workflow `release.yml`, environment `testpypi` (on pypi.org: `pypi`).
3. On GitHub, repository Settings, Environments: create `testpypi` (and `pypi`).

## Rehearsal on TestPyPI

1. Push `main` and wait until every CI job is green.
2. Actions, Release, Run workflow, repository **testpypi**. This builds the wheel and publishes it to test.pypi.org. It does not touch PyPI.
3. In a clean virtual environment, outside the repository:

```bash
python -m venv /tmp/t && /tmp/t/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pitbacktest
cd /tmp && /tmp/t/bin/python -c "import pitbacktest as q; print(q.__version__)"
/tmp/t/bin/python <path to the repository>/tests/smoke_installed.py
```

## Release on PyPI

Only after the rehearsal works. **Pushing a tag that starts with `v` publishes to the real PyPI** (the workflow also checks that the tag equals the version in `pyproject.toml`):

```bash
git tag v0.2.0 && git push origin v0.2.0
```

A published version cannot be replaced. If it is broken, yank it on PyPI (it stays, but resolvers skip it) and release the next version.

## Versions

Pre-releases while the verification gaps in `docs/verification_status.md` are open (`0.2.0`, alpha). Raise the status to beta only after a person other than the author has reviewed the code.
