#!/usr/bin/env python3
r"""Decide the out-of-$\Sigma$ patterns over an alphabet that contains their characters.

§6 decides every equivalence over `TAG_ALPHABET`, the $41$ characters a Datadog tag
value may hold, and the comment above that constant argues the case: a value that is
not a legal tag value cannot appear on either side. For a pattern whose own literals
are legal tag values that is right and the comparison is sharp. For one whose literals
are not --- `LIST|GET`, `ContainerCreating`, `01-Inlet Ambient` --- both sides restrict
to the empty language, equivalence holds for that reason alone, and the gate cannot
fail. The verdict is still reported as exact.

$36$ of the $184$ distinct concrete patterns are in that position, carrying $135$ of the
$1{,}041$ matcher occurrences: $13.0\%$ by occurrence, decided by a comparison that
could not have come out the other way. PromQL label values are not lower-cased, so
these are ordinary Kubernetes and HTTP values rather than corpus defects.

This re-decides exactly those patterns over $\Sigma$ widened by the characters each one
uses, which is a comparison that can fail, and records whether the classification moves.
It changes no figure the paper reports: it says whether those figures were luck.

  python scripts/widen_alphabet_check.py            # re-decide and cache
  python scripts/widen_alphabet_check.py --dry-run  # list the patterns, decide nothing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import qlport.regular as R  # noqa: E402

CROSSCHECK = ROOT / "data" / "raw" / "globs" / "crosscheck.json"
OUT = ROOT / "data" / "raw" / "globs" / "widened.json"

# The characters a regex spends on structure rather than on content. A literal outside
# `TAG_ALPHABET` is what puts a pattern in this experiment; a metacharacter does not.
META = set(".^$*+?()[]{}|\\-")


def _sigma_class(chars: set[str]) -> str:
    r"""`chars` as a bracket expression greenery will read.

    Only `] \ - ^` are special inside a class, and greenery rejects the others escaped,
    which is the same rule `regular.drop_redundant_escapes` follows.
    """
    return "[" + "".join("\\" + c if c in "]\\-^" else c for c in sorted(chars)) + "]"


def out_of_sigma(per_regex: dict[str, dict]) -> list[tuple[str, dict, set[str]]]:
    """The patterns whose own literals $\\Sigma$ does not contain, with those extras."""
    base, found = set(R.TAG_ALPHABET), []
    for regex, record in per_regex.items():
        extra = {c for c in regex if c not in META and c not in base}
        if extra:
            found.append((regex, record, extra))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="list the patterns and the characters that select them")
    args = ap.parse_args()

    if not CROSSCHECK.exists():  # pragma: no cover - run scripts/crosscheck_globs.py
        print(f"{CROSSCHECK.relative_to(ROOT)} missing; run crosscheck_globs.py first")
        return 1
    cross = json.loads(CROSSCHECK.read_text())
    cases = out_of_sigma(cross["per_regex"])

    if args.dry_run:
        for regex, record, extra in cases:
            print(f"  {record['outcome']:<7} {''.join(sorted(extra))!r:28} {regex[:60]!r}")
        print(f"\n{len(cases)} of {len(cross['per_regex'])} patterns")
        return 0

    base, rows, t0 = set(R.TAG_ALPHABET), [], time.time()
    narrow_alphabet, narrow_class = R.TAG_ALPHABET, R._SIGMA_CLASS
    try:
        for regex, record, extra in cases:
            wide = base | extra
            # The decision reads these as module globals, so widening the alphabet is
            # rebinding them. Restored in the `finally` below, because leaving them
            # widened would silently change every later decision in this process.
            R.TAG_ALPHABET = "".join(sorted(wide))
            R._SIGMA_CLASS = _sigma_class(wide)
            got = R.regex_to_globs(regex)
            rows.append({
                "regex": regex, "narrow_kind": record["outcome"],
                "wide_kind": got.kind, "wide_reason": got.reason,
                "agrees": got.kind == record["outcome"],
                "extra_characters": sorted(extra),
                "alphabet_size": len(wide), "globs": len(got.globs),
            })
            mark = "OK " if rows[-1]["agrees"] else "!! "
            print(f"  {mark}{record['outcome']:<7}-> {got.kind:<7}"
                  f"{got.reason:<24} {regex[:44]!r}", flush=True)
    finally:
        R.TAG_ALPHABET, R._SIGMA_CLASS = narrow_alphabet, narrow_class

    agreed = sum(1 for r in rows if r["agrees"])
    print(f"\n{len(rows)} patterns re-decided over a widened alphabet; {agreed} agree")
    for r in rows:
        if not r["agrees"]:
            print(f"  DISAGREES: {r['regex']!r} {r['narrow_kind']} -> {r['wide_kind']}"
                  f" ({r['wide_reason']})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "narrow_alphabet_size": len(base),
        "patterns_total": len(cross["per_regex"]),
        "patterns_out_of_sigma": len(rows),
        "agreed": agreed,
        "seconds": round(time.time() - t0, 1),
        "rows": rows,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if agreed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
