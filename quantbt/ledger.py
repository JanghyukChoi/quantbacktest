"""Trial ledger: remember every backtest you ran, and feed the count into the deflated Sharpe.

The deflated Sharpe needs the number of strategies you tried. People under-report it, usually without meaning to,
because the variants they threw away are not in front of them. The ledger records each run as it happens, so the
count comes from the record and not from memory.

What it does and does not do
  - It records runs that go through it (`backtest_portfolio(..., ledger=...)` or `Ledger.record`). A run you did in
    a notebook without the ledger is invisible to it. It is an honesty aid, not a lock.
  - A trial is a distinct configuration: the same settings on the same factor values and the same data count once,
    however often you rerun them. Change a parameter, the factor values or the data and it is a new trial.
  - The file is append-only and each line carries the hash of the previous one, so deleting or editing a line in the
    middle is detectable (`verify`). Deleting the last lines is not detectable by the chain alone; keep the file in git.
  - `family` names the research question. Trials in different families do not inflate each other.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import validation

_GENESIS = "0" * 64


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_fingerprint(a) -> str:
    """Stable hash of a numeric array or frame (values at float32 precision, shape included)."""
    v = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return _sha(repr(v.shape).encode() + v.tobytes())[:16]


class Ledger:
    def __init__(self, path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "returns").mkdir(exist_ok=True)
        self.file = self.dir / "trials.jsonl"

    # ------------------------------------------------------------------ read
    def _rows(self) -> list[dict]:
        if not self.file.exists():
            return []
        return [json.loads(x) for x in self.file.read_text(encoding="utf-8").splitlines() if x.strip()]

    def trials(self, family: str | None = None) -> list[dict]:
        """One row per distinct configuration (the first time it was run)."""
        seen, out = set(), []
        for r in self._rows():
            if family is not None and r["family"] != family:
                continue
            key = (r["family"], r["config_hash"])
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out

    def n_trials(self, family: str | None = None) -> int:
        return len(self.trials(family))

    # ----------------------------------------------------------------- write
    def record(self, family: str, name: str, returns: pd.Series, config: dict) -> dict:
        """Append one run. `config` must contain everything that defines the trial (parameters, factor and data
        fingerprints); it is hashed to decide whether this is a new trial."""
        cfg = json.dumps(config, sort_keys=True, default=str)
        config_hash = _sha(cfg.encode())[:16]
        rows = self._rows()
        prev = rows[-1]["row_hash"] if rows else _GENESIS
        rpath = self.dir / "returns" / f"{config_hash}.csv"
        if not rpath.exists():
            returns.rename("ret").to_csv(rpath, header=True)
        s = returns.dropna()
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "family": family, "name": name, "config_hash": config_hash,
               "config": json.loads(cfg), "n_obs": int(len(s)),
               "sharpe_per_period": float(s.mean() / s.std(ddof=1)) if len(s) > 2 and s.std(ddof=1) > 0 else None,
               "prev": prev}
        row["row_hash"] = _sha(json.dumps(row, sort_keys=True).encode())
        with open(self.file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    # ---------------------------------------------------------------- checks
    def verify(self) -> dict:
        """Check the hash chain. `ok` is False if a line in the middle was edited, deleted or reordered."""
        prev = _GENESIS
        for i, r in enumerate(self._rows()):
            body = {k: v for k, v in r.items() if k != "row_hash"}
            if r["prev"] != prev or _sha(json.dumps(body, sort_keys=True).encode()) != r["row_hash"]:
                return {"ok": False, "first_bad_line": i + 1}
            prev = r["row_hash"]
        return {"ok": True, "first_bad_line": None}

    def returns_matrix(self, family: str) -> tuple[np.ndarray, list[str]]:
        """(T, N) returns of every distinct trial in the family, on the dates they all share."""
        tr = self.trials(family)
        if not tr:
            raise ValueError(f"no trials recorded for family {family!r}")
        cols = {}
        for r in tr:
            s = pd.read_csv(self.dir / "returns" / f"{r['config_hash']}.csv", index_col=0, parse_dates=True)["ret"]
            cols[f"{r['name']}#{r['config_hash'][:6]}"] = s
        df = pd.DataFrame(cols).dropna()
        if len(df) < 30:
            raise ValueError("fewer than 30 dates shared by all trials; record trials on the same sample")
        return df.to_numpy(), list(df.columns)

    def deflated_sharpe(self, family: str, *, periods_per_year: int = 252) -> dict:
        """Deflated Sharpe of the best trial in the family, with the trial count taken from the ledger."""
        R, names = self.returns_matrix(family)
        out = validation.deflated_sharpe(R, trials=self.n_trials(family), periods_per_year=periods_per_year)
        out["best_name"] = names[out["best"]]
        out["verify"] = self.verify()
        return out
