#!/usr/bin/env python3
"""Record, by content, which paper sources a build read.

The suite's staleness check was an mtime comparison: `touch paper/main.log` satisfied
it, and everything downstream of it -- duplicate labels, undefined references, overfull
boxes, heading levels, the two page counts, build convergence -- went on reporting on a
build that no longer existed. The one test that would have noticed rebuilds both PDFs
inside the suite and is skipped unless `QLPORT_BUILD_PAPER=1` is set, which is to say
the whole log layer rested on the suite's single skip.

A digest cannot be touched into agreement. `make paper` runs this after pdflatex, and
`tests/test_paper_builds._log` recomputes it and refuses a log whose stamp disagrees.

The file is derived, so it is gitignored: a fresh checkout has no stamp and the first
`make paper` writes one, which is the same contract the `.log` files already have.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"
STAMP = PAPER / ".sources.sha256"


def sources() -> list[Path]:
    """Every file a build reads, in the order `_sources()` in the suite uses.

    The generated figures under `paper/figures/` count: they are build inputs that the
    notebooks and `scripts/heldout_split.py` rewrite, and leaving them out meant a
    regenerated figure left this stamp agreeing with a log that predated it.
    """
    return sorted([*PAPER.glob("*.tex"), *(PAPER / "sections").rglob("*.tex"),
                   *(PAPER / "short").rglob("*.tex"),
                   *(PAPER / "switching").rglob("*.tex"),
                   *(PAPER / "tables").glob("*.tex"),
                   *(PAPER / "figures").glob("*.pdf")])


# Matplotlib writes a wall-clock `/CreationDate` into every PDF it saves, so two runs
# of the same notebook over the same data produce byte-different figures. Hashing them
# raw made this stamp track when a figure was regenerated rather than what it contains:
# `make notebooks` invalidated the build log even when every plot was identical. The
# date is the only such field in these files -- verified by stripping it and comparing
# three regenerated figures byte for byte against the committed ones.
_PDF_VOLATILE = re.compile(rb"/CreationDate\s*\([^)]*\)")


def _stable(f: Path) -> bytes:
    """A file's bytes with any timestamp a writer embedded in them removed."""
    data = f.read_bytes()
    return _PDF_VOLATILE.sub(b"/CreationDate ()", data) if f.suffix == ".pdf" else data


def digest() -> tuple[str, int]:
    """One hash over every source's content, and how many there were.

    The count is stored beside the hash because a hash alone cannot distinguish a
    source being deleted from its contents changing, and a deleted section file is
    exactly the kind of change this is here to notice.
    """
    files = sources()
    h = hashlib.sha256()
    for f in files:
        h.update(_stable(f))
    return h.hexdigest(), len(files)


def main() -> int:
    d, n = digest()
    STAMP.write_text(f"{d} {n}\n")
    print(f"    paper/.sources.sha256 over {n} sources: {d[:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
