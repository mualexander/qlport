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
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"
STAMP = PAPER / ".sources.sha256"


def sources() -> list[Path]:
    """Every `.tex` a build reads, in the order `_sources()` in the suite uses."""
    return sorted([*PAPER.glob("*.tex"), *(PAPER / "sections").rglob("*.tex"),
                   *(PAPER / "tables").glob("*.tex")])


def digest() -> tuple[str, int]:
    """One hash over every source's bytes, and how many there were.

    The count is stored beside the hash because a hash alone cannot distinguish a
    source being deleted from its contents changing, and a deleted section file is
    exactly the kind of change this is here to notice.
    """
    files = sources()
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest(), len(files)


def main() -> int:
    d, n = digest()
    STAMP.write_text(f"{d} {n}\n")
    print(f"    paper/.sources.sha256 over {n} sources: {d[:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
