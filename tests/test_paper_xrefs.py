"""The paper's internal pointers must land somewhere, and its artifacts must be used.

`test_paper_figures.py` checks that every number is accounted for. This file checks the
things around the numbers: that a cross-reference resolves, that a labelled table is
actually pointed at, that every file the notebooks generate is in the document, and that
every script the paper names still exists.

Three findings in one read-through were of this shape. A measurement cited in two
sections was explained in none, and the cross-reference meant to supply it pointed at a
remark about something else. Two generated artifacts -- a table superseded by its own
figure, and a bar chart of a three-row table -- were still being written by notebooks
and read by nobody. A table of worked examples floated in a section that never mentioned
it, so nothing fixed where it landed.

What this does not catch is a reference that resolves to the wrong thing, which is the
interesting half: `\\ref{rem:gauge}` and `\\ref{rem:as-rate-placement}` are both valid
labels, and only a reader knows which one the sentence meant. Nothing mechanical will.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
# rglob: sections/moved/ holds the blocks main.tex defers to appendices.
TEX = sorted((PAPER / "sections").rglob("*.tex")) + [PAPER / "body.tex"]

# `\apphead{title}{label}` is how an apparatus file declares its heading and label
# at once, so it carries a label the plain `\label` pattern cannot see.
_LABEL = re.compile(r"\\label\{([^}]*)\}|\\apphead\{[^}]*\}\{([^}]*)\}")
_REF = re.compile(r"\\(?:c|C|auto)?ref\{([^}]*)\}")
_INPUT = re.compile(r"\\input\{([^}]*)\}")
_GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
# `scripts\slash foo.py`, `tests/test_foo.py`, `semantics/promql\_eval.py`
_SOURCE = re.compile(r"((?:[A-Za-z0-9.]|\\_|/|\\slash\s*)+\.py)")


def _strip(text: str) -> str:
    """Drop LaTeX comments. `\\%` is an escaped percent sign, not a comment."""
    return re.sub(r"(?<!\\)%.*", "", text)


def _scan(pattern: re.Pattern[str]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in TEX:
        for match in pattern.finditer(_strip(path.read_text())):
            captured = next((g for g in match.groups() if g is not None), "")
            for item in captured.split(","):
                found.setdefault(item.strip(), set()).add(path.name)
    return found


@pytest.fixture(scope="module")
def labels() -> dict[str, set[str]]:
    return _scan(_LABEL)


@pytest.fixture(scope="module")
def refs() -> dict[str, set[str]]:
    return _scan(_REF)


def _count(pattern: re.Pattern[str]) -> collections.Counter[str]:
    """Occurrences rather than files, which is the distinction `_scan` cannot make."""
    found: collections.Counter[str] = collections.Counter()
    for path in TEX:
        for match in pattern.finditer(_strip(path.read_text())):
            captured = next((g for g in match.groups() if g is not None), "")
            for item in captured.split(","):
                found[item.strip()] += 1
    return found


@pytest.fixture(scope="module")
def label_counts() -> collections.Counter[str]:
    return _count(_LABEL)


def test_every_label_is_defined_once(labels, label_counts):
    r"""Twice in one file is the case this did not see.

    `_scan` keys by file, so two `\label{fig:x}` in the same section collapsed into one
    entry and passed. LaTeX takes the last definition and warns, in one line of a build
    that prints hundreds, and every `\ref` to it then points wherever that last one
    landed -- which is the failure this file exists to catch, arriving by the one route
    it was blind to.
    """
    repeated = {name: sorted(files) for name, files in labels.items() if len(files) > 1}
    assert not repeated, f"defined in more than one file: {repeated}"
    twice = {name: n for name, n in sorted(label_counts.items()) if n > 1}
    assert not twice, f"defined more than once in one file: {twice}"


def test_every_reference_resolves(labels, refs):
    dangling = {name: sorted(files) for name, files in refs.items() if name not in labels}
    assert not dangling, f"\\ref to a label that does not exist: {dangling}"


def test_every_float_is_referenced(labels, refs):
    """A table or figure nobody points at floats wherever LaTeX likes.

    Section labels are exempt: a section earns its place by being in the document.
    """
    orphans = sorted(
        name for name in labels
        if name.split(":")[0] in {"tab", "fig"} and name not in refs
    )
    assert not orphans, f"labelled but never referenced in the prose: {orphans}"


def test_every_generated_artifact_is_in_the_paper():
    """A notebook that writes what the paper does not read is computing for nobody.

    The paper is the only consumer of `paper/tables` and `paper/figures`, so an unused
    file there means either a missing `\\input` or a notebook cell to delete. Both have
    happened; the second is harder to notice, because nothing fails.
    """
    inputs = set(_scan(_INPUT)) | set(_scan(_GRAPHIC))
    stems = {Path(name).name for name in inputs}
    unused = sorted(
        str(path.relative_to(PAPER))
        for directory, suffix in (("tables", ".tex"), ("figures", ".pdf"))
        for path in sorted((PAPER / directory).glob(f"*{suffix}"))
        if path.stem not in stems
    )
    assert not unused, f"generated but never used by the paper: {unused}"


def test_every_input_has_a_file():
    missing = sorted(
        name for name in set(_scan(_INPUT)) | set(_scan(_GRAPHIC))
        if not (PAPER / name).exists()
        and not (PAPER / f"{name}.tex").exists()
        and not (PAPER / f"{name}.pdf").exists()
    )
    assert not missing, f"\\input or \\includegraphics with no file: {missing}"


def test_every_source_file_the_paper_names_exists():
    """The paper cites the script behind each measurement; a rename must not silence that.

    Searched for anywhere under the repository root, because the paper writes
    `regular.py` and `semantics/promql_eval.py` without their `src/qlport` prefix.
    """
    missing: dict[str, list[str]] = {}
    for path in TEX:
        for match in _SOURCE.finditer(_strip(path.read_text())):
            name = (match.group(1).replace(r"\_", "_")
                                  .replace(r"\slash", "/")
                                  .replace(" ", ""))
            if not any(ROOT.glob(f"**/{name}")):
                missing.setdefault(name, []).append(path.name)
    assert not missing, f"named in the paper, not on disk: {missing}"


_CITE = re.compile(r"\\cite[tp]\{([^}]*)\}")
_BIBITEM = re.compile(r"\\bibitem\[[^\]]*\]\{([^}]*)\}")


def _cited() -> set[str]:
    return {key.strip()
            for path in TEX
            for match in _CITE.finditer(_strip(path.read_text()))
            for key in match.group(1).split(",")}


def test_every_citation_has_an_entry():
    entries = set(_BIBITEM.findall((PAPER / "references.tex").read_text()))
    dangling = sorted(_cited() - entries)
    assert not dangling, f"cited with no bibliography entry: {dangling}"


def test_every_entry_is_cited():
    """An entry nothing cites is usually the residue of a claim that moved.

    Repointing four claims at the Datadog pages that actually state them left the
    original entry cited by nothing, which is the signal that the repoint was
    incomplete rather than that the source stopped mattering.
    """
    entries = set(_BIBITEM.findall((PAPER / "references.tex").read_text()))
    orphans = sorted(entries - _cited())
    assert not orphans, f"in the bibliography, cited nowhere: {orphans}"


_PLACEHOLDER = re.compile(r"(?<![\\\w])(?:NUM|TODO|FIXME|XXX|TBD)[A-Z_\\]*")


def test_the_paper_has_no_placeholders_left():
    """A staged number that never got filled in reads as prose, not as an error.

    `test_paper_figures.py` looks for digits, so a token like `NUM\\_SHAPES` standing in
    for a figure passes every other check in this repository and renders in the PDF. It
    happened while the seventh pass was re-running an experiment the prose quotes.
    """
    left: dict[str, list[str]] = {}
    for path in TEX:
        for match in _PLACEHOLDER.finditer(_strip(path.read_text())):
            left.setdefault(match.group(0), []).append(path.name)
    assert not left, f"placeholder left in the paper: {left}"
