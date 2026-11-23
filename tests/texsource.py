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
would have found.

The asymmetry worth knowing: a guard asserting a sentence is *present* can be satisfied
by text no reader sees, which is a false green and silent; one asserting a figure is
*absent* can only be made to fail by it, which is a false red and loud. Only the first
is dangerous, and only the first is why this module has three readers instead of one.

THREE READERS, BECAUSE THERE ARE TWO DOCUMENTS AND NEITHER IS THE SOURCE. `main.tex`
sets `\fullfalse` and defers half the apparatus to appendices; `main_full.tex` sets
`\fulltrue` and sets it in place. So a passage inside `\inlineonly`, `\shortonly` or
`\iffull` reaches one PDF and not the other, and a passage inside `\iffalse` reaches
neither. A guard that reads the source sees all sides of every condition at once, and
two builds can be made to assert opposite relations about the same figure with the
whole suite green.

  `rendered(path, full)`  one document. The right reader for a claim about what the
                          paper says: ask for it in each build.
  `anywhere(path)`        the two documents joined. The right reader for a claim that
                          something is absent, which has to be absent from both.
  `source(path)`          comments stripped, conditionals intact -- the file as LaTeX
                          reads it, which is not a document and never what a reader
                          sees. For structural questions only: what a file inputs,
                          what it defines, where its labels are. Every call is
                          enumerated in `SOURCE_READS` for that reason.

There was a fourth, called `published`, and the name was the lie. It returned the
comment-stripped source, which is neither published nor a document, and fifty-one
guards read it. Three ledgers were moved to `rendered` in the commit that diagnosed
this; the other forty-eight were not, including a guard added in that same commit whose
own docstring said "both builds must carry the note". Parking the note behind
`\iffalse` and printing its opposite left every content guard green.

`rendered` also refuses to return text with a conditional still in it. The five it
resolves are the five the paper uses, and resolving by name is how the stripper's
defect comes back one level up: `\ifnum0=1 ... \fi` needs no macro definition, is not
in the list, and hid a pinned sentence from both PDFs at 795 passed. A reader not
knowing a construct must fail, not pass it through.
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
#
# Reverting this line to either bug once left the whole suite passing, because the fix
# was hand-checked in a shell and the hand-check was never committed. It is checked now
# by `test_the_comment_stripper_reads_a_percent_the_way_latex_does`, which owns its own
# cases: a table of them here would be a ledger the audit reads, satisfied by
# construction, which is a defect this suite has already had once.
COMMENT = re.compile(r"(?<!\\)((?:\\\\)*)%.*")
# Python's own comments, for the audit below, which reads source lines.
_PY_COMMENT = re.compile(r"#.*")

# Any conditional still standing in `rendered`'s output. `\iffull` is the paper's only
# one today; the others are here because a reader that passes an unknown construct
# through is a reader of a document nobody has.
_CONDITIONAL = re.compile(r"\\(if[a-zA-Z]*|else|fi)(?![a-zA-Z])")

# `\newif\iffull` declares the switch and `\fulltrue`/`\fullfalse` set it. Those are
# the two build drivers' first lines and they are not conditionals: nothing is
# conditional on them in the file that holds them, and leaving them in would have
# `rendered` refuse `main.tex` for containing the definition of the thing it resolves.
_DECLARE = re.compile(r"\\newif\\if([a-zA-Z]+)")


def strip_comments(text: str) -> str:
    """Comments gone, for text already in hand -- a `$...$` span, a line, a file."""
    return COMMENT.sub(r"\1", text)


def flat(text: str) -> str:
    """Whitespace collapsed, so a phrase survives the line break LaTeX ignores."""
    return re.sub(r"\s+", " ", text)


def rendered(path: Path, full: bool) -> str:
    r"""One source file as *one* of the two builds will have it.

    `full` picks the build: `False` is `main.tex`'s 103 pages, which defer the
    apparatus to appendices, and `True` is `main_full.tex`'s 102, which set it inline.

    Everything behind a `%` goes, then `\inlineonly{X}` and `\shortonly{X}` keep X in
    one build each, then `\iffalse A \fi` and `\iftrue A \fi` resolve the same way for
    both -- `\iffalse` being how a block is parked while it is redrafted, which is the
    scenario this module exists for -- and `\iffull A \else B \fi` resolves per build,
    innermost first so a nested hook resolves with its parent.

    Then anything left that looks like a conditional is an error, because a construct
    this function does not know is a construct it cannot have resolved.
    """
    text = strip_comments(path.read_text())
    for switch in _DECLARE.findall(text):
        text = re.sub(rf"\\newif\\if{switch}(?![a-zA-Z])", "", text)
        text = re.sub(rf"\\{switch}(?:true|false)(?![a-zA-Z])", "", text)
    for macro, keep in (("inlineonly", full), ("shortonly", not full)):
        needle = "\\" + macro + "{"
        while (at := text.find(needle)) != -1:
            end = _group(text, at + len(needle) - 1)
            body = text[at + len(needle):end]
            text = text[:at] + (body if keep else "") + text[end + 1:]
    for name, keep in (("iffalse", False), ("iftrue", True)):
        text = _resolve(text, name, keep)
    text = _resolve(text, "iffull", full)
    left = _CONDITIONAL.search(text)
    assert left is None, (
        f"{path.name} still has {left.group(0)!r} in it after rendering, so this is "
        f"not what the build prints: a conditional nothing here resolves keeps its "
        f"body in what every guard reads and out of every PDF. Teach `rendered` the "
        f"construct, or take it out of the paper."
    )
    return text


def builds(path: Path) -> tuple[str, str]:
    """Both documents, in build order: the 103-page one, then the 102-page one.

    For a claim about what the paper says, which has to hold in each of them.
    """
    return rendered(path, False), rendered(path, True)


def table(path: Path) -> str:
    r"""A generated table's text, which is one reading because it has to be.

    The tables under `paper/tables/` are written by the notebooks and inputted by both
    builds, so no conditional should ever appear in one. That is a property rather
    than a guarantee -- nothing stops a hand-edit putting an `\iffull` in a generated
    file -- so it is asserted here instead of assumed, which is what distinguishes
    this from calling `source` on a table and hoping.
    """
    short, full = builds(path)
    assert short == full, (
        f"{path.name} is read as a generated table but renders differently in the two "
        f"builds, so one of them prints something the other does not"
    )
    return short


def anywhere(path: Path) -> str:
    """The two documents joined: text that reaches a reader of either build.

    For a claim that something is *absent*. Reading the union is the conservative
    choice there -- a figure in one build and not the other still has to be accounted
    for -- and it is still not the source, so a numeral behind a `%` or an `\\iffalse`
    is not something this function will offer up to be accounted for or forbidden.
    """
    return "\n".join(builds(path))


# The structural reads: every place the suite wants the file rather than a document,
# because the question is about the source's shape and resolving the conditionals
# would destroy the answer. `\iffull\input{sections/moved/x}\fi` inputs that file in
# one build, and a guard collecting what the paper inputs has to see it in both.
#
# Keyed on the line, as `RAW_READS` is, so that adding one is a decision. Nothing here
# may ask what the paper *says*: that is what `rendered` and `anywhere` are for, and
# the ledger is the list of questions for which no document is the right thing to read.
SOURCE_READS = {
    "test_no_section_file_is_read_by_neither_build":
        "which files each build inputs -- a conditional \\input still inputs",
    "test_the_sections_define_no_macros_of_their_own":
        "whether a section defines a macro, which is a fact about the file",
    "test_every_deferred_block_is_hooked_once_and_listed_once":
        "the hooks and the appendix listings, which are conditionals by construction",
    "test_no_deferred_block_inputs_an_appendix_wrapper":
        "the input graph, read for cycles rather than for prose",
    "test_a_deferred_block_does_not_guard_its_own_children":
        "whether a block wraps its children in a condition it is already inside",
    "test_an_indent_suppressed_after_a_hook_is_conditional":
        "what stands above a hook line by line, before either build reorders it",
    "test_apref_is_used_only_where_it_can_resolve":
        "where the appendix macros are written, which is build-independent",
    "test_every_apref_target_exists":
        "the appendix heads and the pointers to them, declarations both",
    "_scan": "labels and the references to them, which no build renames",
    "_count": "the same, counted",
    "test_every_source_file_the_paper_names_exists":
        "the source-file citations, which name files rather than claims",
    "_cited": "the citation keys, counted against the bibliography",
    "test_every_citation_has_an_entry": "the bibliography, which is not prose",
    "test_every_entry_is_cited": "the bibliography again, the other way round",
    "test_the_paper_has_no_placeholders_left":
        "drafting placeholders, which must be in no build and so in neither document",
}


def source(path: Path) -> str:
    r"""The file as LaTeX reads it: comments gone, conditionals intact.

    Not a document. `\iffull A \else B \fi` keeps both A and B, so a claim about what
    the paper *says* must not be checked against this -- that is what `rendered` and
    `anywhere` are for. Use it only where the conditionals are the answer: what a file
    inputs, what it defines, where its labels are.
    """
    return strip_comments(path.read_text())


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
    "tree = ast.parse(path.read_text())":
        "a test module, read as a syntax tree for the audit of dead helpers",
    # The two places a `.tex` file is read raw on purpose. The staleness check hashes
    # what the build read, and a digest over comment-stripped text would agree across
    # an edit to a comment -- which is a change to the source and so to the build.
    "digest.update(path.read_bytes())":
        "every paper source's bytes, for the digest `make paper` records",
    "assert stamp.read_text().strip() == want, (":
        "that digest, which is a hex string and not LaTeX",
}


# Every way of getting a file's bytes that the suite could use. `read_text` alone was
# audited for one commit, and a probe guard that read the paper raw through the other
# five -- a bound method, `getattr`, `read_bytes().decode()`, `open()` and
# `Path.open()` -- passed. An audit that names one spelling of the thing it forbids
# measures the author's imagination, not the suite.
_READERS = frozenset({"read_text", "read_bytes", "open"})

# Formats with no comment syntax to strip, named where the read itself can be seen to
# name them.
_NOT_LATEX = ("json", "csv", ".md", ".py", ".log", ".toml")


def unstripped_reads(directory: Path) -> list[str]:
    """Every read in `directory` that neither strips comments nor is excused.

    Read off each file's syntax tree rather than its text, so that this module's own
    ledger -- seven string literals that each quote a read -- cannot satisfy the audit
    it exists to narrow. Both the call `p.read_text()` and the bare reference `f =
    p.read_text` count, since the second is the first with a step in between.

    A read that can be seen to name a format which is not LaTeX is excused by rule
    rather than by entry, which is what keeps the ledger from growing every time the
    audit is widened: a `.csv`, a `.json`, a `.md`, a `.py` or a `.log` has no comment
    syntax this module could strip, and a `%` in one is data.

    WHAT COUNTS AS NAMING IT is the part that was wrong. The excuse was read off the
    whole line, so two reads sharing a line shared one excuse, and
    `json.loads(cache.read_text()); body = TEX.read_text()` was excused entirely by
    the first half. A line carrying one read is still judged as a line, because the
    format is usually named by what consumes the text rather than by the read itself
    (`json.loads(p.read_text())`); a line carrying more than one is judged read by
    read, on each one's own source segment. The Python comment comes off first either
    way, because as a whole-line substring the old excuse was satisfied by writing
    `# not json` after a raw read -- the same defect as keying a ledger on a magnitude
    instead of on a magnitude and a kind.

    What is left in `RAW_READS` is therefore only the reads whose line says nothing
    about what is being read: a bare variable, or `__file__`.
    """
    out = []
    for path in sorted(directory.rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text()
        lines = text.split("\n")
        found = [(n, name) for n in ast.walk(ast.parse(text))
                 if (name := _reader_name(n)) is not None]
        crowded = {n.lineno for n, _ in found}
        crowded = {ln for ln in crowded
                   if sum(1 for n, _ in found if n.lineno == ln) > 1}
        for node, _ in found:
            line = lines[node.lineno - 1].strip()
            own = ast.get_source_segment(text, node) or line
            context = own if node.lineno in crowded else line
            code = _PY_COMMENT.sub("", context).strip() or context
            if any(fmt in code for fmt in _NOT_LATEX):
                continue
            if line in RAW_READS or _PY_COMMENT.sub("", line).strip() in RAW_READS:
                continue
            out.append(f"  {path.name}:{node.lineno} {line}")
    return sorted(set(out))


def _reader_name(node: ast.AST) -> str | None:
    """The spelling of the read `node` is, or None if it is not one."""
    if isinstance(node, ast.Attribute) and node.attr in _READERS:
        return node.attr
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "open"):
        return "open"
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "getattr" and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in _READERS):
        return str(node.args[1].value)
    return None


def source_reads(directory: Path) -> list[str]:
    """Every `source(...)` call in `directory` that `SOURCE_READS` does not account for.

    The companion to `unstripped_reads`, one level up. That audit asks whether a guard
    strips comments; this one asks whether a guard that strips them is then reading a
    document or a file. Forty-eight guards read the file while believing they read the
    paper, and the fix for three of them was shipped as the fix.

    Keyed on the function that holds the call, not on the line. `RAW_READS` keys on the
    line because the line is where a read says what it is reading, but a structural
    read says nothing of the kind: three guards in one module each hold a bare `text =
    source(path)`, and one line-keyed entry would have excused all three and any
    fourth written later. The question this ledger exists to force -- is this about the
    file or about what the paper says? -- is a question about a guard, so a guard is
    what an entry names.
    """
    return sorted(f"  {where} in {name}"
                  for name, sites in source_readers(directory).items()
                  if name not in SOURCE_READS for where in sites)


def source_readers(directory: Path) -> dict[str, list[str]]:
    """Each guard in `directory` that calls `source`, and where it does.

    Separate from `source_reads` so the ledger can be held to this exactly, in both
    directions: an entry for a guard that has stopped reading the source is as much a
    false statement as a guard reading it with no entry, and a ledger checked only for
    dead entries drifts in the direction that matters -- it keeps excusing what has
    moved on while saying nothing about what has arrived.
    """
    out: dict[str, list[str]] = {}
    for path in sorted(directory.rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        tree = ast.parse(path.read_text())
        for owner in ast.walk(tree):
            if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(owner):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "source"):
                    out.setdefault(owner.name, []).append(
                        f"{path.name}:{node.lineno}")
    return out


def suite_source(directory: Path) -> str:
    """Every test module's source, joined -- for audits of what the suite contains.

    Here rather than in a test file because a test that reads the suite's text would
    otherwise be an offender under `unstripped_reads`, and excusing the audit from its
    own rule is how a ledger starts growing excuses instead of shrinking.

    This module is excluded, for the reason `unstripped_reads` excludes it and one
    more: every key of `RAW_READS` and of `SOURCE_READS` is a string literal in this
    file, so a staleness audit that read this file would find every excuse present by
    construction and could never fail. It could not, for one commit, and an excuse
    naming a line that exists nowhere in the suite passed.
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
