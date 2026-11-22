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

# Whether a `%` opens a comment is a question of PARITY, not of whether a backslash
# precedes it: `%` is a comment iff an even number of backslashes stands before it.
# `\%` is an escaped percent sign, `\\%` is a line break followed by a comment, and
# `\\\%` is a line break followed by an escaped percent. A lookbehind cannot count,
# so the pairs are matched and put back.
#
# Both halves of this have now been got wrong in this repository, in opposite
# directions. A lookbehind for two backslashes cut every line at its first `\%` and
# read a document SMALLER than the published one. A lookbehind for one accepted `\\%`
# and read a document LARGER -- which let a pinned sentence be satisfied by text
# behind a `\\%`, with the visible sentence saying the opposite and the suite green.
COMMENT = re.compile(r"(?<!\\)((?:\\\\)*)%.*")
# Python's own comments, for the audit below, which reads source lines.
_PY_COMMENT = re.compile(r"#.*")


def published(path: Path) -> str:
    """One source file as both PDFs will have it: everything behind a `%` is gone."""
    return strip_comments(path.read_text())


def strip_comments(text: str) -> str:
    """`published` for text already in hand -- a `$...$` span, a line, a whole file."""
    return COMMENT.sub(r"\1", text)


def flat(text: str) -> str:
    """Whitespace collapsed, so a phrase survives the line break LaTeX ignores."""
    return re.sub(r"\s+", " ", text)


# The seven places in the suite that read something which is not LaTeX, so for which
# `%` is not a comment. Keyed on the line, because a line is what changes when the call
# does, and the point is that adding an eighth should be a decision.
RAW_READS = {
    'return files("qlport.grammars").joinpath(name).read_text()':
        "a Lark grammar, where % is not a comment",
    "for node in ast.walk(ast.parse(path.read_text())):":
        "a Python module, read as a syntax tree",
    "src = Path(__file__).read_text()":
        "that file's own source, where the ledgers are declared",
    r'for m in re.finditer(r"Gap\.([A-Z_]+)", f.read_text())}':
        "the translator's own source, for the gap codes it raises",
    'return log.read_text(errors="replace")':
        "pdflatex's log, which is neither LaTeX nor ours to strip",
    "before = path.read_bytes()":
        "a corpus archive member, compared byte for byte across two extractions",
    'assert path.read_bytes() == before, "re-extraction changed the corpus"':
        "and the comparison itself",
}


# Every way of getting a file's bytes that the suite could use. `read_text` alone was
# audited for one commit, and a probe guard that read the paper raw through the other
# five -- a bound method, `getattr`, `read_bytes().decode()`, `open()` and
# `Path.open()` -- passed. An audit that names one spelling of the thing it forbids
# measures the author's imagination, not the suite.
_READERS = frozenset({"read_text", "read_bytes", "open"})

# Formats with no comment syntax to strip, named on the reading line itself.
_NOT_LATEX = ("json", "csv", ".md", ".py", ".log", ".toml")


def unstripped_reads(directory: Path) -> list[str]:
    """Every read in `directory` that neither strips comments nor is excused.

    Read off each file's syntax tree rather than its text, so that this module's own
    ledger -- seven string literals that each quote a read -- cannot satisfy the audit
    it exists to narrow. Both the call `p.read_text()` and the bare reference `f =
    p.read_text` count, since the second is the first with a step in between.

    A read whose line names a format that is not LaTeX is excused by rule rather than
    by entry, which is what keeps the ledger from growing every time the audit is
    widened: a `.csv`, a `.json`, a `.md`, a `.py` or a `.log` has no comment syntax
    this module could strip, and a `%` in one is data. The line is taken with its
    Python comment removed, because as a whole-line substring the old `json` excuse
    was satisfied by writing `# not json` after a raw read -- the same defect as
    keying a ledger on a magnitude instead of on a magnitude and a kind.

    What is left in `RAW_READS` is therefore only the reads whose line says nothing
    about what is being read: a bare variable, or `__file__`.
    """
    out = []
    for path in sorted(directory.rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text()
        lines = text.split("\n")
        for node in ast.walk(ast.parse(text)):
            name = None
            if isinstance(node, ast.Attribute) and node.attr in _READERS:
                name = node.attr
            elif (isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "open"):
                name = "open"
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "getattr" and len(node.args) > 1
                  and isinstance(node.args[1], ast.Constant)
                  and node.args[1].value in _READERS):
                name = str(node.args[1].value)
            if name is None:
                continue
            line = lines[node.lineno - 1].strip()
            code = _PY_COMMENT.sub("", line).strip() or line
            if any(fmt in code for fmt in _NOT_LATEX):
                continue
            if line in RAW_READS or code in RAW_READS:
                continue
            out.append(f"  {path.name}:{node.lineno} {line}")
    return sorted(set(out))


def suite_source(directory: Path) -> str:
    """Every test module's source, joined -- for audits of what the suite contains.

    Here rather than in a test file because a test that reads the suite's text would
    otherwise be an offender under `unstripped_reads`, and excusing the audit from its
    own rule is how a ledger starts growing excuses instead of shrinking.

    This module is excluded, for the reason `unstripped_reads` excludes it and one
    more: every key of `RAW_READS` is a string literal in this file, so a staleness
    audit that read this file would find every excuse present by construction and
    could never fail. It could not, for one commit, and an excuse naming a line that
    exists nowhere in the suite passed.
    """
    return "\n".join(p.read_text() for p in sorted(directory.rglob("*.py"))
                      if p.resolve() != Path(__file__).resolve())


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
    # `\iffalse A \fi` is how a block is parked while it is redrafted, which is the
    # scenario this module exists for: the text is in the source and in no PDF. It is
    # resolved first, and the same way for both builds. `\iftrue` is its complement.
    for name, keep in (("iffalse", False), ("iftrue", True)):
        text = _resolve(text, name, keep)
    # `\iffull A \else B \fi`, innermost first so a nested hook resolves with its parent
    return _resolve(text, "iffull", full)


def _resolve(text: str, name: str, keep: bool) -> str:
    r"""`\name A \else B \fi` reduced to A or B, innermost conditional first."""
    pattern = re.compile(
        rf"\\{name}(?!\w)((?:(?!\\{name}(?!\w)|\\else|\\fi).)*)"
        r"(?:\\else((?:(?!\\else|\\fi).)*))?\\fi", re.S)
    while True:
        text, n = pattern.subn(
            lambda m: (m.group(1) or "") if keep else (m.group(2) or ""), text)
        if not n:
            return text
