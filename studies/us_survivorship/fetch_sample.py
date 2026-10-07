"""Download the preregistered random sample (see PREREGISTRATION.md): walk the seeded random order of the frame and stop
at the first 500 tickers for which Tiingo returns prices. Resumable; stops cleanly at the free tier's limit.

    python fetch_sample.py              # reads TIINGO_API_KEY from the environment or ~/projects/liqmap/.env
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from quantbt.adapters import tiingo                # noqa: E402
from quantbt.equity import load_us_master          # noqa: E402

STORE = Path.home() / ".cache" / "quantbt" / "tiingo"
TARGET, SEED, SINCE, CHUNK = 500, 0, "2013-01-01", 25


def main() -> int:
    frame = tiingo.study_frame(load_us_master(), SINCE)
    order = tiingo.draw_order(frame["ticker"], seed=SEED)
    key = tiingo.load_key("~/projects/liqmap/.env")
    print(f"frame {len(order)} tickers; target {TARGET} with prices", flush=True)
    for i in range(0, len(order), CHUNK):
        part = order[i:i + CHUNK]
        r = tiingo.fetch_symbols(part, STORE, key=key, sleep=0.4, progress=False)
        have = sum((STORE / f"{tiingo._safe(t)}.json").exists() for t in order[:i + CHUNK])
        none = sum((STORE / f"{tiingo._safe(t)}.none").exists() for t in order[:i + CHUNK])
        print(f"prefix {i + len(part)}: {have} with prices, {none} none  ({r['calls']} requests this chunk)", flush=True)
        if r["stopped"]:
            print("STOPPED:", r["stopped"], flush=True)
            return 3
        if have >= TARGET:
            (HERE / "sample_prefix.json").write_text(json.dumps({"prefix_length": i + len(part), "with_prices": have, "none": none}))
            print("target reached", flush=True)
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
