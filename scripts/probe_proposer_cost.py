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


class _OverCap(BaseException):
    """Deriving from `BaseException`, deliberately, so a cap cannot become a result.

    `regular.regex_to_globs` records anything its equivalence decision raises as
    `equivalence-undecided`, which is honest for a decision it could not take and wrong
    for a clock this script started. While this derived from `Exception` that `except`
    caught the cap, and a three-second run reported fourteen patterns as refusals in
    exactly 3.00 seconds each -- the same fourteen an uncapped run decides as exact
    unions of globs. A refusal recorded because the measurement ran out of patience is
    the error this paper is about, so the cap is not catchable as one.
    """


def _one(rx: str, cap: float, complete: bool = False) -> dict:
    """Decide one pattern, timed and capped. Runs in a worker process.

    The cap is armed with `SIGALRM` inside the worker, which is the only place it can
    bind. An earlier version of this script passed the cap to `Future.result`, but
    `as_completed` yields a future only once it has finished, so `result` returned
    instantly and the cap never fired: the probe reported true times and abandoned
    nothing, and one pattern ran for 261 seconds under a nominal 60-second limit. A
    budget that cannot interrupt the work is not a budget.
    """
    import signal

    def _raise(signum, frame):                    # noqa: ARG001 - handler signature
        raise _OverCap

    from qlport.regular import regex_to_globs, single_glob_by_search

    t0 = time.time()
    signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, cap)
    try:
        if complete:
            found = single_glob_by_search(rx)
            out = {"kind": "search", "found": found}
        else:
            g = regex_to_globs(rx)
            out = {"kind": g.kind, "exact": bool(g.exact),
                   "globs": len(g.globs), "reason": g.reason}
        return {"regex": rx, "seconds": round(time.time() - t0, 3), **out}
    except _OverCap:
        return {"regex": rx, "seconds": None, "kind": "over-cap",
                "reason": f"exceeded {cap:g}s"}
    except Exception as exc:                      # a crash is not a result either
        return {"regex": rx, "seconds": round(time.time() - t0, 3),
                "kind": "error", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=float, default=60.0,
                    help="seconds any one pattern may take before it is abandoned")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--complete", action="store_true",
                    help="probe single_glob_by_search, the complete procedure, instead")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from crosscheck_globs import concrete_regexes

    regexes = sorted(concrete_regexes(), key=len)
    print(f"{len(regexes)} concrete patterns, cap {args.cap:g}s, {args.workers} workers",
          flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, rx, args.cap, args.complete): rx
                   for rx in regexes}
        for i, fut in enumerate(as_completed(futures), 1):
            rx = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:              # the worker died rather than returned
                row = {"regex": rx, "seconds": None, "kind": "worker-lost",
                       "reason": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            if row["kind"] in ("over-cap", "error", "worker-lost") \
                    or (row["seconds"] or 0) > 1.0:
                print(f"  [{i:>3}/{len(regexes)}] "
                      f"{str(row['seconds']):>8}s {row['kind']:<11} {rx[:58]}", flush=True)

    timed = [r for r in rows if r["seconds"] is not None]
    over = [r for r in rows if r["kind"] == "over-cap"]
    timed.sort(key=lambda r: -r["seconds"])
    OUT.write_text(json.dumps(
        {"cap_seconds": args.cap, "complete_procedure": args.complete, "rows": rows},
        indent=1) + "\n")

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
