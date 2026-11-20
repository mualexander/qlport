"""What the undocumented reading of a glob's star is worth, measured over the corpus.

`STAR_MATCHES_EMPTY` records an answer Datadog does not document: whether the `*` of a
tag glob matches the empty string. E5 put it to a real account and it does, so that is
the reading §6 takes. The question this script answers is what the answer bought --- how
much of the corpus would be classified differently under the other reading.

It is not a small number and it is not the one §6 used to quote. Under the reading
taken, `.*` is the single glob `*` and `.+` is the union of `c*` over the alphabet.
Under the other, a star requires a character: `.+` becomes the single glob `*` in
principle, but every pattern whose star has to admit the empty string --- which is most
of them, since `.*` appears inside the great majority of this corpus's patterns --- has
no glob at all.

Re-decides every concrete pattern of `data/processed/regex_glob.csv` with the flag
flipped and writes `data/raw/globs/star_reading.json`. Needs notebook 04 to have run.

Usage: .venv/bin/python scripts/star_reading_check.py
"""
from __future__ import annotations

import collections
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import qlport.regular as R

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "processed" / "regex_glob.csv"
OUT = ROOT / "data" / "raw" / "globs" / "star_reading.json"


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} missing; run `make notebooks` (notebook 04)")
    with SOURCE.open(newline="") as fh:
        source = list(csv.DictReader(fh))

    assert R.STAR_MATCHES_EMPTY, "the recorded reading is the one with the empty string"
    rows = []
    R.STAR_MATCHES_EMPTY = False
    try:
        for i, r in enumerate(source, 1):
            started = time.perf_counter()
            got = R.regex_to_globs(r["regex"])
            rows.append({"regex": r["regex"], "occurrences": int(r["occurrences"]),
                         "taken": r["kind"], "other": got.kind,
                         "other_reason": got.reason})
            print(f"{i:4d}/{len(source)} {time.perf_counter() - started:7.2f}s "
                  f"{r['kind']:6s} -> {got.kind:6s} {r['regex'][:46]!r}", flush=True)
    finally:
        R.STAR_MATCHES_EMPTY = True

    def tally(key: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = collections.defaultdict(
            lambda: {"distinct": 0, "occurrences": 0})
        for row in rows:
            out[row[key]]["distinct"] += 1
            out[row[key]]["occurrences"] += row["occurrences"]
        return dict(out)

    moved = [r for r in rows if r["taken"] != r["other"]]
    payload = {
        "provenance": {
            "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reading": "a glob's star requires at least one character",
            "recorded_reading": "a glob's star matches the empty string",
            "patterns": len(rows),
            "occurrences": sum(r["occurrences"] for r in rows),
        },
        "taken": tally("taken"),
        "other": tally("other"),
        "moved": {
            "distinct": len(moved),
            "occurrences": sum(r["occurrences"] for r in moved),
            "by_transition": {
                f"{a}->{b}": {
                    "distinct": sum(1 for r in moved
                                    if (r["taken"], r["other"]) == (a, b)),
                    "occurrences": sum(r["occurrences"] for r in moved
                                       if (r["taken"], r["other"]) == (a, b)),
                }
                for a, b in sorted({(r["taken"], r["other"]) for r in moved})
            },
        },
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"  taken : {payload['taken']}")
    print(f"  other : {payload['other']}")
    print(f"  moved : {payload['moved']['distinct']} patterns, "
          f"{payload['moved']['occurrences']} occurrences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
