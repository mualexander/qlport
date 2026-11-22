r"""One reading of the paper's source, shared by every guard that scans it.

A guard that reads `path.read_text()` is reading a document nobody publishes:
everything behind a `%` is in that string and in no PDF. A sentence parked behind a `%`
while it was redrafted went on satisfying the phrase meant to pin the sentence that
replaced it, and commenting a line out is the ordinary way to redraft.

That was found, diagnosed and fixed in `test_paper_figures.py`, in a commit whose
message called it the headline fix. The suite had three comment strippers at the time
and seven other files that read the paper; the other two strippers stayed where they
were and every one of those files kept the pattern. `test_paper_builds.py`'s stripper
was worse than absent: its lookbehind wanted two backslashes, so it cut every line at
its first escaped percent -- forty-eight lines of section 7 alone -- and read a
document *smaller* than the published one, which no amount of stripping elsewhere
would have found. Forty-nine call sites read through here now, and
`test_every_guard_reads_the_document_the_paper_publishes` next door fails if a new raw
read appears anywhere in the suite.

The asymmetry worth knowing: a guard asserting a sentence is *present* can be satisfied
by a comment, which is a false green and silent; one asserting a figure is *absent* can
only be made to fail by a comment, which is a false red and loud. Only the first is
dangerous. Both go through here anyway, because one rule is cheaper to keep than two,
and the seven reads excused in `RAW_READS` are excused for reading something that is
not LaTeX rather than for reading it raw.

`published` is still not a document. There are two: `main.tex` defers half the
apparatus to appendices and `main_full.tex` sets it in place, so a passage inside
`\inlineonly` or `\shortonly` reaches one PDF and not the other. `rendered` resolves
that, and the phrase check in `test_paper_figures.py` asks for each pinned sentence in
both builds.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

# `\%` is an escaped percent sign and `\\%` is a line break followed by a comment;
# the lookbehind has to see exactly one backslash to mean the former.
COMMENT = re.compile(r"(?<!\\)%.*")


def published(path: Path) -> str:
    """One source file as both PDFs will have it: everything behind a `%` is gone."""
    return COMMENT.sub("", path.read_text())


def strip_comments(text: str) -> str:
    """`published` for text already in hand -- a `$...$` span, a line, a whole file."""
    return COMMENT.sub("", text)


def flat(text: str) -> str:
    """Whitespace collapsed, so a phrase survives the line break LaTeX ignores."""
    return re.sub(r"\s+", " ", text)


# The seven places in the suite that read something which is not LaTeX, so for which
# `%` is not a comment. Keyed on the line, because a line is what changes when the call
# does, and the point is that adding an eighth should be a decision.
RAW_READS = {
    'return files("qlport.grammars").joinpath(name).read_text()':
        "a Lark grammar, where % is not a comment",
    'source = (ROOT / "scripts" / "as_rate_order.py").read_text()':
        "a Python script, checked for the call order it documents",
    'corpus_md = (ROOT / "data" / "corpus" / "SOURCES.md").read_text()':
        "the corpus provenance note, which is Markdown",
    "for node in ast.walk(ast.parse(path.read_text())):":
        "a Python module, read as a syntax tree",
    "src = Path(__file__).read_text()":
        "that file's own source, where the ledgers are declared",
    r'for m in re.finditer(r"Gap\.([A-Z_]+)", f.read_text())}':
        "the translator's own source, for the gap codes it raises",
    'return log.read_text(errors="replace")':
        "pdflatex's log, which is neither LaTeX nor ours to strip",
}


def unstripped_reads(directory: Path) -> list[str]:
    """Every `.read_text()` in `directory` that neither strips comments nor is excused.

    Read off each file's syntax tree rather than its text, so that this module's own
    ledger -- six string literals that each quote a `read_text()` call -- cannot
    satisfy the audit it exists to narrow. `json` on the line is the other excuse: a
    cache is parsed, not scanned, and a `%` in it is data.
    """
    out = []
    for path in sorted(directory.glob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text()
        lines = text.split("\n")
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "read_text"):
                continue
            line = lines[node.lineno - 1].strip()
            if "json" in line or line in RAW_READS:
                continue
            out.append(f"  {path.name}:{node.lineno} {line}")
    return out


def suite_source(directory: Path) -> str:
    """Every test module's source, joined -- for audits of what the suite contains.

    Here rather than in a test file because a test that reads the suite's text would
    otherwise be an offender under `unstripped_reads`, and excusing the audit from its
    own rule is how a ledger starts growing excuses instead of shrinking.
    """
    return "\n".join(p.read_text() for p in sorted(directory.glob("*.py")))


def _group(text: str, open_brace: int) -> int:
    """The index of the `}` that closes the `{` at `open_brace`, nesting counted."""
    depth = 0
    i = open_brace
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unclosed group at {open_brace}")


def rendered(path: Path, full: bool) -> str:
    """One source file as *one* of the two builds will have it.

    `published` is not enough on its own, because there are two PDFs. `main.tex` sets
    `\\fullfalse` and defers half the apparatus to appendices; `main_full.tex` sets
    `\\fulltrue` and sets it in place. `\\inlineonly{X}` reaches only the second and
    `\\shortonly{X}` only the first, so a guard reading the source sees both sides of
    a condition no reader ever sees together, and two builds can be made to assert
    opposite relations about the same figure with every test green. Resolving the
    condition is what makes a check a check about a document.
    """
    text = published(path)
    for macro, keep in (("inlineonly", full), ("shortonly", not full)):
        needle = "\\" + macro + "{"
        while (at := text.find(needle)) != -1:
            end = _group(text, at + len(needle) - 1)
            body = text[at + len(needle):end]
            text = text[:at] + (body if keep else "") + text[end + 1:]
    # `\iffull A \else B \fi`, innermost first so a nested hook resolves with its parent
    pattern = re.compile(r"\\iffull(?!\w)((?:(?!\\iffull(?!\w)|\\else|\\fi).)*)"
                         r"(?:\\else((?:(?!\\iffull(?!\w)|\\else|\\fi).)*))?\\fi", re.S)
    while True:
        text, n = pattern.subn(
            lambda m: (m.group(1) or "") if full else (m.group(2) or ""), text)
        if not n:
            return text
