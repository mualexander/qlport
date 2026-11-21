#!/usr/bin/env python
r"""The Dash0 converter's own coverage claim, parsed from a cached README.

Section 2 quotes two of another tool's figures in order to say why they are not
comparable with kappa. They are its numbers and not this paper's, so they cannot be
derived from anything here -- but quoting them from a live page would leave them resting
on nothing, which is what the unchecked-figure ratchet in `tests/test_paper_figures.py`
exists to prevent. So the README is cached under `data/raw/dash0/` with its commit and
sha256, exactly as the OpenObserve bundle is, and the figures are read out of the cache.

The repository is a feature branch with open pull requests. A force-push would change
what the live page says and change nothing here, which is the point: the paper attributes
to that tool what it claimed on the date recorded in the manifest.

Usage: .venv/bin/python scripts/dash0_claims.py [--write-registry]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "dash0"
MANIFEST = RAW / "MANIFEST.json"
REGISTRY = ROOT / "paper" / "quoted_figures.json"

# The one sentence the paper quotes. Anchored on its own wording so a reworded README
# fails here rather than silently matching some other percentage on the page.
CLAIM = re.compile(
    r"\*\*([0-9]+(?:\.[0-9]+)?)% of ([0-9,]+) distinct queries and monitors translate")


def figures() -> dict[str, float]:
    man = json.loads(MANIFEST.read_text())
    path = RAW / man["file"]
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != man["sha256"]:
        raise SystemExit(f"{path.name} does not match the manifest:\n"
                         f"  manifest {man['sha256']}\n  file     {got}")
    if len(raw) != man["bytes"]:
        raise SystemExit(f"{path.name} is {len(raw)} bytes, manifest says {man['bytes']}")

    found = CLAIM.findall(raw.decode())
    if not found:
        raise SystemExit("the quoted coverage sentence is not in the cached README")
    if len({f for f in found}) != 1:
        raise SystemExit(f"the README states the claim more than one way: {set(found)}")
    pct, total = found[0]
    return {"dd0.translate_pct": float(pct),
            "dd0.queries_and_monitors": int(total.replace(",", ""))}


def main() -> int:
    fig = figures()
    man = json.loads(MANIFEST.read_text())
    print(f"{man['repository']}\n  branch {man['branch']} at {man['commit'][:12]}, "
          f"retrieved {man['retrieved']}")
    for k, v in fig.items():
        print(f"  {k:<28} {v:,}")
    if "--write-registry" in sys.argv:
        reg = json.loads(REGISTRY.read_text())
        reg.update(fig)
        REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
        print(f"\nmerged {len(fig)} keys into {REGISTRY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
