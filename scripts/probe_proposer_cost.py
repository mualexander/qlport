#!/usr/bin/env python3
"""How long does the proposer take on each concrete corpus pattern?

§6's apparatus twice ran for fourteen hours on a single pattern and wrote nothing, and
the reason was an avoidable product construction in `regular.equivalent`. With that
fixed, one question is left open: how many patterns are still expensive, and how
expensive. A refusal recorded because a decision procedure ran out of patience is not a
language result, so the paper needs the distribution before it can choose a budget.

Each pattern is decided in its own process with a hard wall-clock cap, so a pathological
one cannot take the measurement with it. The patterns are independent, so this fans out
across cores -- which is the one part of §6's work that genuinely parallelises, the
serial cross-check being bounded by its slowest single pattern rather than by its total.

  python scripts/probe_proposer_cost.py --cap 60 --workers 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "processed" / "proposer_cost.json"


def _one(rx: str) -> dict:
    """Decide one pattern, timed. Runs in a worker process."""
    from qlport.regular import regex_to_globs

    t0 = time.time()
    try:
        g = regex_to_globs(rx)
        return {"regex": rx, "seconds": round(time.time() - t0, 3),
                "kind": g.kind, "exact": bool(g.exact),
                "globs": len(g.globs), "reason": g.reason}
    except Exception as exc:                      # a refusal is a result; a crash is not
        return {"regex": rx, "seconds": round(time.time() - t0, 3),
                "kind": "error", "reason": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=float, default=60.0,
                    help="seconds any one pattern may take before it is abandoned")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from crosscheck_globs import concrete_regexes

    regexes = sorted(concrete_regexes(), key=len)
    print(f"{len(regexes)} concrete patterns, cap {args.cap:g}s, {args.workers} workers",
          flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, rx): rx for rx in regexes}
        for i, fut in enumerate(as_completed(futures), 1):
            rx = futures[fut]
            try:
                row = fut.result(timeout=args.cap)
            except Exception:
                row = {"regex": rx, "seconds": None, "kind": "over-cap",
                       "reason": f"exceeded {args.cap:g}s"}
            rows.append(row)
            if row["kind"] in ("over-cap", "error") or (row["seconds"] or 0) > 1.0:
                print(f"  [{i:>3}/{len(regexes)}] "
                      f"{str(row['seconds']):>8}s {row['kind']:<11} {rx[:58]}", flush=True)

    timed = [r for r in rows if r["seconds"] is not None]
    over = [r for r in rows if r["kind"] == "over-cap"]
    timed.sort(key=lambda r: -r["seconds"])
    OUT.write_text(json.dumps({"cap_seconds": args.cap, "rows": rows}, indent=1) + "\n")

    print(f"\ndecided within the cap : {len(timed)} of {len(rows)}")
    print(f"over the cap           : {len(over)}")
    for r in over:
        print(f"    {r['regex']}")
    print(f"total decided time     : {sum(r['seconds'] for r in timed):.1f}s")
    print("slowest that finished  :")
    for r in timed[:8]:
        print(f"    {r['seconds']:8.2f}s  {r['kind']:<11} {r['regex'][:58]}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
