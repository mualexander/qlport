"""The two builds, checked as builds rather than as sources.

Every other paper test reads `.tex` files. None of them had ever opened a `.log`, and
two defects shipped straight through that gap, both invisible because neither stops
`pdflatex` from exiting zero:

  * `\\apref` emitted `\\S\\ref{app:eval}` in the complete build, where the appendix
    wrappers -- and therefore their labels -- do not exist. Three undefined references.
  * A deferred block `\\input`-ed a whole appendix *wrapper*, so ten blocks were set
    twice in the complete build, five pages of duplicated prose with ten multiply-defined
    labels and every cross-reference to them pointing at the copy.

The second one is the instructive one. The per-file invariant held: each of those ten
blocks was hooked exactly once and listed exactly once. The duplication happened one
level up, at wrapper granularity, so a test over files could not see it. Hence both
halves below -- the static invariant over the hooks, and the log, which is the only
place a duplicate label is reported.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from texsource import (BUILD_VARIANT, COMMENT, RAW_READS, SOURCE_READS, anywhere,
                       raw_reads_used,
                       builds, flat,
                       rendered, source, source_reads, source_readers, strip_comments,
                       suite_source, table, unstripped_reads)

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SECTIONS = PAPER / "sections"
MOVED = SECTIONS / "moved"
WRAPPERS = {"app_proofs", "app_experiments", "app_eval"}
# The two twenty-page versions: `main_short` on the inexpressibility spine and
# `main_switching` on the empirical switching-cost one. Each is a different document
# rather than a further rendering of this one, so each is held to the same build
# hygiene as the other two and excluded from the page-length comparison below.
BUILDS = ("main", "main_full", "main_short", "main_switching")

_HOOK = re.compile(r"\\iffull\\input\{sections/moved/([A-Za-z0-9_]+)\}\\fi")
_INPUT = re.compile(r"\\input\{sections/([A-Za-z0-9_/]+)\}")


# Anything under paper/ that a build WRITES rather than reads. Everything else is an
# input and has to be inside the digest; the figures were outside it for a commit, so
# regenerating a plot left the stamp agreeing with a log that predated it.
_BUILD_OUTPUT = {".log", ".aux", ".out", ".toc", ".sha256", ".bbl", ".blg", ".fls",
                 ".fdb_latexmk", ".synctex.gz"}


# What each plot has to be about, by the axis text matplotlib writes into it. The
# corpus sizes are checked against the registry rather than spelled here.
_PLOT_MARKS = {
    "cumulative_coverage": ("queries fully covered", "features implemented"),
    "feature_frequency": ("SCOPE_VAR", "GROUP_BY"),
    "heldout_vocabulary": ("vocabulary refitted on a sample",
                           "the staircase, scored both ways", "dashboards read"),
    "rate_error_vs_window": ("rollup interval", "poisson", "bursty"),
}


def test_each_plot_is_the_plot_its_caption_says_it_is():
    """The figures are build inputs and nothing read them.

    `cp paper/figures/cumulative_coverage.pdf paper/figures/heldout_vocabulary.pdf`
    left the suite green with the held-out vocabulary figure printing the coverage
    staircase. The digest notices that a figure CHANGED, which is what it was added
    for, but says nothing about which plot is which -- re-stamp and the swap is clean.

    Two checks. Pairwise distinctness needs no tools and catches the swap outright. The
    axis text catches the subtler version, a figure regenerated from the wrong data,
    and ties the two corpus sizes the plots print to the registry that states them.
    """
    import hashlib
    import shutil
    import subprocess

    figures = sorted((PAPER / "figures").glob("*.pdf"))
    assert {f.stem for f in figures} == set(_PLOT_MARKS), (
        f"the figures are {sorted(f.stem for f in figures)} and this guard knows "
        f"{sorted(_PLOT_MARKS)}; teach it the new one or stop shipping it")

    seen: dict[str, str] = {}
    for f in figures:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        assert h not in seen, f"{f.name} is byte-identical to {seen[h]}"
        seen[h] = f.name

    if not shutil.which("pdftotext"):  # pragma: no cover - tool-dependent
        return
    reg = json.loads((PAPER / "quoted_figures.json").read_text())
    dd_size = f"n = {reg['nb03.corpus_occurrences']:,}"
    promql_sizes = set()
    for f in figures:
        got = subprocess.run(["pdftotext", str(f), "-"], capture_output=True,
                             text=True, check=True).stdout
        flat_text = " ".join(got.split())
        for mark in _PLOT_MARKS[f.stem]:
            assert mark in flat_text, (
                f"{f.name} does not carry {mark!r}, so it is not the plot the caption "
                f"describes")
        if f.stem in ("cumulative_coverage", "feature_frequency"):
            assert dd_size in flat_text, (
                f"{f.name} does not print the Datadog corpus size the registry "
                f"states: wanted {dd_size!r}")
            # The PromQL size is not a registry key -- it reaches the paper through a
            # table and sits in the coincidence list -- so the two plots are held to
            # each other instead of to a key that does not exist.
            promql_sizes |= {m.group(1) for m in
                             re.finditer(r"n = ([\d,]+)", flat_text)} - {dd_size[4:]}
    assert len(promql_sizes) == 1, (
        f"the two plots disagree on the PromQL corpus size: {sorted(promql_sizes)}")


def test_every_perturbation_in_the_battery_still_applies():
    r"""The harness's anchors, which go stale silently and then lie about the suite.

    `scripts/perturb.py` is how "verified by N perturbations" is checked. A case whose
    anchor no longer matches the source applies nothing, the suite stays green, and the
    case reports MISSED -- which reads as a hole in the guard it names rather than a
    hole in the harness. That has happened twice. Once a crashed run left a planted
    sentence in `eval_staircase.tex` and it was nearly committed, because `restore()`
    after a loop does not run if the loop raises; once a case reported MISSED for a
    defect that was closed, because it ran only the module where every failure is the
    build digest the harness filters out.

    So the dry run is part of the suite: it applies every case, restores at once, and
    reports only the anchors that no longer resolve. It does not run the battery --
    that is twenty minutes and a tree full of perturbations -- and it is not a
    substitute for running it. It is the check that the battery is still about this
    paper.
    """
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "perturb.py"),
                           "--dry-run"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, (
        "perturbations whose anchors no longer match the source:\n" + done.stdout
        + "\n  (a stale case tests nothing and reports MISSED, which reads as a hole "
          "in the suite)\n" + done.stderr[-800:])
    assert "0 stale" in done.stdout, done.stdout
    cases = int(done.stdout.split()[0])
    assert cases >= 43, f"the battery has shrunk to {cases} cases"


def test_no_macro_body_carries_a_figure():
    r"""A numeral in a macro body prints and no audit in this suite can see it.

    `test_every_prose_figure_is_accounted_for` scans `paper/sections/**` and
    `body.tex`. `macros.tex` is in neither, and `rendered` strips `\newcommand` bodies
    before it does anything else -- a change made so that rendering `macros.tex` would
    stop asserting on its own definitions. The two together leave a hole a reader can
    see straight through: `\newcommand{\hedge}{$123.4\%$}` used once in a section
    prints "123.4%" in both PDFs, and the unaccounted-figure census, the placeholder
    scan, the column rule and the spelled-claims guard are all blind to it.

    The rule is the narrow one that closes it: a macro body may carry no digit except
    as a parameter reference, and no math span. Figures belong in the prose where the
    census reads them, or in a generated table. Nothing in `macros.tex` needs either --
    its digits today are all package options and `#1`/`#2`, which is why this can be a
    prohibition rather than a ledger.
    """
    src = (PAPER / "macros.tex").read_text()
    bodies = []
    for m in re.finditer(r"\\(?:re)?newcommand\{\\(\w+)\}(?:\[\d+\])?\{", src):
        depth, i = 1, m.end()
        while i < len(src) and depth:
            if src[i] == "{" and src[i - 1] != "\\":
                depth += 1
            elif src[i] == "}" and src[i - 1] != "\\":
                depth -= 1
            i += 1
        bodies.append((m.group(1), src[m.end():i - 1]))
    assert bodies, "no macro definitions found, so this guard is reading the wrong file"

    bad = []
    for name, body in bodies:
        if re.search(r"(?<!#)\d", body):
            bad.append(f"  \\{name} carries a digit outside a parameter: {body[:70]!r}")
        if "$" in body:
            bad.append(f"  \\{name} carries a math span: {body[:70]!r}")
    assert not bad, (
        "a figure in a macro body prints and no figure audit here reads it:\n"
        + "\n".join(bad))


def test_the_digest_covers_every_file_a_build_could_read():
    """No input under paper/ may sit outside the stamp.

    `_sources` used to live here, duplicating `scripts/stamp_sources.py`'s list, and its
    docstring said "both sides recompute it, so a divergence between them fails loudly
    rather than silently". Nothing compared the two lists; the claim was false for as
    long as it was written. There is one list now, in the script, and this is the check
    the docstring was describing: enumerate paper/ from the filesystem and require that
    everything the build reads is in it.

    The figures are the reason. They were outside the digest for a commit, which let a
    regenerated plot coexist with a log that predated it. A `.csv` or `.bib` added as an
    input later would be the same defect again, and would fail here first.
    """
    listed = set(_stamper().sources())
    # main.pdf/main_full.pdf are what the build writes. quoted_figures.json is neither:
    # pdflatex never opens it, because the prose carries its figures as literal
    # numerals and the registry is what the SUITE compares them against. A change to it
    # must not invalidate a build log, and a change to the paper must not be excused by
    # it -- which is why it is named here rather than quietly matched by a suffix rule.
    outputs = {PAPER / "main.pdf", PAPER / "main_full.pdf",
               PAPER / "main_short.pdf", PAPER / "main_switching.pdf",
               PAPER / "quoted_figures.json"}
    missing = sorted(
        p.relative_to(ROOT) for p in PAPER.rglob("*")
        if p.is_file() and p not in listed and p not in outputs
        and p.suffix not in _BUILD_OUTPUT and not p.name.startswith("."))
    assert not missing, (
        "files under paper/ that are neither build outputs nor inside the digest:\n  "
        + "\n  ".join(str(m) for m in missing))
    assert any(p.suffix == ".pdf" for p in listed), "the figures left the digest again"
    assert len(listed) > 100, f"the digest covers only {len(listed)} files"


def _stamper():
    """`scripts/stamp_sources.py`, imported so the suite and `make paper` share one
    definition of what "the sources, by content" means."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_stamp_sources", ROOT / "scripts" / "stamp_sources.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(name: str) -> str:
    """The build log, insisting it was built from the sources that are here now.

    A stale log is worse than no log: it reports a clean build of a paper that no
    longer exists. So this fails rather than skips, and says what to run.

    BY CONTENT, NOT ONLY BY MTIME. This compared timestamps, so `touch paper/main.log`
    satisfied it and everything downstream -- duplicate labels, undefined references,
    overfull boxes, heading levels, the two page counts, build convergence -- went on
    reporting on a build that no longer existed. The one test that would have noticed
    rebuilds both PDFs inside the suite and is skipped by default, which is to say the
    whole log layer rested on the suite's single skip. `make paper` now records a
    digest of what it read, and a digest cannot be touched into agreement.
    """
    log = PAPER / f"{name}.log"
    assert log.exists(), f"{log.relative_to(ROOT)} is missing -- run `make paper`"
    # NO MTIME COMPARISON. There was one, and it is the check `touch paper/main.log`
    # defeated -- the reason this function reads a digest at all. Keeping it beside the
    # digest bought no detection the digest does not already give, and cost false
    # failures: re-running a notebook rewrites `paper/tables/*.tex` with identical
    # content and a new mtime, so a reproduction run that proves the tables are
    # unchanged reported a stale build. The digest answers the question the mtime was
    # standing in for, and answers it about content.
    stamp = PAPER / ".sources.sha256"
    assert stamp.exists(), (
        f"{stamp.relative_to(ROOT)} is missing -- run `make paper`, which records by "
        f"content what the build read"
    )
    # Through the script, not a second copy of its arithmetic. This recomputed the
    # hash itself over `path.read_bytes()`, which agreed with the script only as long
    # as the script hashed raw bytes too; the moment the script started stripping
    # matplotlib's embedded `/CreationDate` the two would have disagreed on every
    # figure, and the failure would have read as a stale build. A check that
    # reimplements the thing it checks has two places to keep right.
    want = "{} {}".format(*_stamper().digest())
    assert stamp.read_text().strip() == want, (
        f"{stamp.relative_to(ROOT)} does not match the sources here, so this log is "
        f"from a different paper -- run `make paper`"
    )
    return log.read_text(errors="replace")


# --- the logs -------------------------------------------------------------------

@pytest.mark.parametrize("build", BUILDS)
def test_no_label_is_defined_twice(build):
    """A duplicate label means some block was set twice, and no `.tex` test sees it."""
    dupes = sorted(set(re.findall(r"Label `([^']+)' multiply defined", _log(build))))
    assert not dupes, f"{build}.pdf defines these labels twice: {', '.join(dupes)}"


@pytest.mark.parametrize("build", BUILDS)
def test_no_reference_is_undefined(build):
    text = _log(build)
    missing = sorted(set(re.findall(r"Reference `([^']+)' on page", text)))
    assert not missing, f"{build}.pdf has undefined references: {', '.join(missing)}"
    assert "There were undefined references" not in text
    assert "Citation" not in text or not re.search(r"Citation `[^']+' on page", text)


@pytest.mark.parametrize("build", BUILDS)
def test_no_overfull_boxes(build):
    """Both builds are kept at zero, so any new one is a regression and not noise."""
    boxes = re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", _log(build))
    assert not boxes, f"{build}.pdf has {len(boxes)} overfull hbox(es): {boxes[:5]}"


@pytest.mark.parametrize("build", BUILDS)
def test_the_heading_levels_do_not_skip(build):
    """`hyperref` warns on a jump, which is how a \\subsubsection under an appendix with
    no \\subsection above it showed up -- it numbered as A.0.1 and read as a broken ref."""
    assert "bookmark levels is greater than one" not in _log(build)


def _reaches_a_deferred_block(top: Path) -> bool:
    """Whether a build can reach `sections/moved/` by following its own `\\input`s.

    `\\input{x}` is relative to paper/, so a reference resolves against PAPER with `.tex`
    appended; a reference naming no existing file is skipped rather than raising, since
    this walker is only asked a yes-or-no question about reachability. It returns on the
    first hit, so a hook inside `\\iffull` counts -- which is correct here, because the
    question is whether the log should have opened any block at all and `main.tex` and
    `main_full.tex` differ only in where those blocks land.
    """
    seen, queue = set(), [top]
    while queue:
        path = queue.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        text = source(path)
        if "sections/moved/" in text:
            return True
        for ref in re.findall(r"\\input\{([^}]*)\}", text):
            queue.append(PAPER / (ref if ref.endswith(".tex") else ref + ".tex"))
    return False


def test_no_block_is_typeset_twice():
    """What the page-count test below is a proxy for, asserted directly.

    The gap it was written to catch was five pages, caused by a wrapper being pulled in
    twice. A page count detects that only indirectly, and the indirection cost a real
    diagnosis: as the appendices grew, the deferred build's section headings widened the
    gap on their own and the proxy started failing on content rather than on duplication.
    This reads the LaTeX logs instead and asserts each moved block is opened exactly
    once per build, which is the property that actually matters.
    """
    import collections

    # Whether each build's SOURCE reaches a deferred block at all, so that "none in the
    # log" is a defect for the builds that input some and the expected reading for the one
    # that inputs none. `main_switching` states the reverse direction's results and proves
    # none of them, so it pulls in no block from sections/moved/; asserting a non-empty
    # list for every build made adding that build fail here, on a build with nothing to
    # duplicate.
    for build in BUILDS:
        opened = re.findall(r"sections/moved/([a-z0-9_]+)\.tex", _log(build))
        reachable = _reaches_a_deferred_block(PAPER / f"{build}.tex")
        assert opened or not reachable, (
            f"{build}.tex reaches a deferred block and {build}.log opened none")
        twice = {k: v for k, v in collections.Counter(opened).items() if v > 1}
        assert not twice, f"{build}.pdf typesets these more than once: {twice}"


def test_the_two_builds_are_the_same_length():
    """A backstop on the direct check above, not the primary signal.

    They hold the same prose, so the lengths should track. The deferred build promotes
    each apparatus block to its own section, and that costs pages as the appendices
    grow, so the bound is loose and the duplication check above is the tight one.
    """
    pages = {}
    for build in BUILDS:
        found = re.findall(r"\((\d+) pages", _log(build))
        assert found, f"no page count in {build}.log"
        pages[build] = int(found[-1])
    # Only these two. Each twenty-page build holds a subset of this prose by design, so a
    # length comparison against either would be asserting the opposite of what it is for.
    a, b = pages["main"], pages["main_full"]
    assert abs(a - b) <= 3, (
        f"main.pdf is {a} pages and main_full.pdf is {b}; the same prose in both means a "
        "gap this wide is something missing, or an apparatus block that moved build"
    )


# --- the guards' own reading ----------------------------------------------------

def test_every_guard_reads_the_document_the_paper_publishes():
    """Three rounds found this defect three times because the fix never swept.

    A guard that calls `path.read_text()` on a `.tex` file is asserting things about a
    document nobody publishes: everything behind a `%` is in that string and in no PDF.
    A sentence commented out while it was redrafted went on satisfying the phrase that
    pinned its replacement. One file was fixed, in the commit whose message called that
    the headline fix; three others kept the pattern, and this file had a stripper of its
    own whose lookbehind wanted two backslashes, so it truncated every line at its first
    escaped percent -- forty-eight lines of section 7 alone -- and read a document
    *smaller* than the published one. Fixing a defect where it was found rather than
    where it belongs
    is what this test exists to stop, so it scans the suite and not the paper.
    """
    offenders = unstripped_reads(ROOT / "tests")
    assert not offenders, (
        "read a document instead: `rendered`, `builds`, `anywhere` or `table` from "
        "texsource, or `source` if the question really is about the file\n"
        + "\n".join(offenders)
    )


def test_the_raw_read_ledger_names_exactly_the_reads_that_need_it():
    """Held to the suite in both directions, as `SOURCE_READS` already was.

    This asked whether each key occurs as a substring of the joined suite source,
    comments and docstrings included. So an excuse outlived the read it excused:
    commenting the read out and leaving its text in the comment passed this and the
    audit both. An equality cannot drift that way.
    """
    used = raw_reads_used(ROOT / "tests")
    assert set(RAW_READS) == used, (
        f"excuses no read needs: {sorted(set(RAW_READS) - used)}\n"
        f"reads excused by no entry: {sorted(used - set(RAW_READS))}"
    )


def test_the_comment_stripper_reads_a_percent_the_way_latex_does():
    r"""Whether a `%` opens a comment is parity, and this suite has had it wrong twice.

    A lookbehind cannot count, so both one-backslash and two-backslash lookbehinds are
    wrong, in opposite directions: one cuts every line at its first `\%` and reads a
    document smaller than the published one, the other accepts `\\%` -- a forced line
    break followed by a comment -- and reads one larger. The second is the dangerous
    direction, because a guard asserting a sentence is present is satisfied by text no
    reader sees.

    Reverting `COMMENT` to the one-backslash bug left the whole suite passing, through
    all fifty-one call sites, because the fix was checked by hand in a shell and the
    hand-check was never committed. Not to either bug: the two-backslash direction
    fails 28 tests, which is the asymmetry above and which a commit message
    here got wrong in the flattering direction. This is that hand-check. The cases are written out
    here rather than imported, because a table of them in `texsource` would be a
    ledger the audit reads and would be satisfied by construction.
    """
    cases = [
        (r"a % b", "a "),                      # no backslash: a comment
        (r"a \% b", r"a \% b"),              # one: an escaped percent sign
        (r"a \\% b", r"a \\"),             # two: a line break, then a comment
        (r"a \\\% b", r"a \\\% b"),      # three: a line break, then an escape
        (r"a \\\\% b", r"a \\\\"),     # four: two line breaks, then a comment
        (r"a \\\\\% b", r"a \\\\\% b"),
        # And the two shapes the paper actually contains.
        (r"v2 parses $99.9\%$ of them % and v1 parses less",
         r"v2 parses $99.9\%$ of them "),
        (r"\begin{corollary}[Quantiles]% \label{cor:quantile}",
         r"\begin{corollary}[Quantiles]"),
    ]
    wrong = [f"  {text!r} -> {strip_comments(text)!r}, want {want!r}"
             for text, want in cases if strip_comments(text) != want]
    assert not wrong, "the comment stripper is not reading parity:\n" + "\n".join(wrong)
    # And the regex is the one those cases were written for, not a rewrite of it.
    assert COMMENT.pattern == r"(?<!\\)((?:\\\\)*)%.*", COMMENT.pattern


def test_rendered_refuses_a_conditional_it_does_not_know(tmp_path):
    r"""Resolving conditionals by name is the stripper's defect one level up.

    `rendered` knows five: `\inlineonly`, `\shortonly`, `\iffalse`, `\iftrue` and
    `\iffull`. `\ifnum0=1 ... \fi` needs no macro definition, is in none of those,
    and passed straight through -- so a pinned sentence parked behind one satisfied
    its phrase check while both PDFs printed the opposite figure, at 795 passed and a
    real build. A reader that does not know a construct has to fail, not pass it on.
    """
    for hidden in (r"\ifnum0=1 HIDDEN \fi", r"\ifx\a\b HIDDEN \fi",
                   r"\ifcase0 HIDDEN\or SHOWN\fi", r"HIDDEN \fi"):
        probe = tmp_path / "probe.tex"
        probe.write_text("Before.\n" + hidden + "\nAfter.\n")
        for full in (False, True):
            with pytest.raises(AssertionError, match="after rendering"):
                rendered(probe, full)
    # The five it does know still resolve, and to the right side.
    probe = tmp_path / "known.tex"
    probe.write_text(r"\iffull F\else S\fi \inlineonly{I}\shortonly{H}"
                     r" \iffalse NEVER\fi \iftrue ALWAYS\fi")
    assert flat(rendered(probe, False)).strip() == "S H ALWAYS"
    assert flat(rendered(probe, True)).strip() == "F I ALWAYS"


def test_no_guard_reads_the_source_where_it_means_a_document():
    r"""`source` returns the file, which is neither PDF, so every call is a decision.

    Fifty-one guards read the comment-stripped source through a function called
    `published`, which was the name that hid this: it suggested a document and
    returned neither of the two. Three ledgers were moved to per-build reading in the
    commit that diagnosed the defect; the other forty-eight were not, one of them a
    guard added in that same commit whose docstring said "both builds must carry the
    note". Parking the note behind `\iffalse` left it in no PDF with the suite green.

    So `source` is for structural questions only -- what a file inputs, what it
    defines, where its labels are -- and `SOURCE_READS` is the list of them. A new
    call has to be added there, which is where the question "is this about the file or
    about the paper?" gets asked.
    """
    offenders = source_reads(ROOT / "tests")
    assert not offenders, (
        "these read the source where they may mean a document; if the question is "
        "what the paper says, use `builds` or `anywhere`:\n" + "\n".join(offenders)
    )


def test_the_source_ledger_names_exactly_the_guards_that_read_the_source():
    """Held to the suite in both directions, which is what `RAW_READS` is not.

    A ledger checked only for dead entries drifts the way that matters: it goes on
    excusing what has moved while saying nothing about what has arrived. So this is an
    equality, and a guard that stops reading the source has to give up its entry.
    """
    reading = source_readers(ROOT / "tests")
    assert set(SOURCE_READS) == set(reading), (
        f"no entry: {sorted(set(reading) - set(SOURCE_READS))}\n"
        f"no longer reads the source: {sorted(set(SOURCE_READS) - set(reading))}"
    )


def test_no_section_file_is_read_by_neither_build():
    """A `.tex` under sections/ that nothing inputs is in no PDF and in no guard.

    Which would be merely untidy, except that a binding and a pinned phrase name
    their file by basename. An orphan copy of a deferred block, one directory up,
    answers to that name ahead of the real one, so every check about the block reads
    the orphan while the paper prints the block. The two halves of that -- unambiguous
    resolution, and no orphan to resolve to -- are checked here and in
    `test_paper_figures.py::_section_path`.
    """
    # By path and not by basename. The first version of this compared stems, so a
    # copy of `moved/eval_heldout.tex` placed at `sections/eval_heldout.tex` --- which
    # is exactly the shadowing this test exists to prevent --- was itself excused by
    # the real file's `\\input`.
    inputs = set()
    for path in [*PAPER.glob("*.tex"), *SECTIONS.rglob("*.tex")]:
        for ref in re.findall(r"\\input\{([^}]*)\}", source(path)):
            rel = ref[len("sections/"):] if ref.startswith("sections/") else ref
            inputs.add(rel)
    orphans = sorted(
        p.relative_to(PAPER).as_posix() for p in SECTIONS.rglob("*.tex")
        if p.relative_to(SECTIONS).with_suffix("").as_posix() not in inputs)
    assert not orphans, (
        "under paper/sections and inputted by nothing:\n  " + "\n  ".join(orphans))


def test_the_sections_define_no_macros_of_their_own():
    """Every macro lives in `macros.tex`, where both builds expand it.

    A `\\newcommand` whose body nothing calls is prose that is in the source and in
    neither PDF, which satisfies a pinned phrase and a binding exactly as a comment
    used to. Barring the definitions is cheaper than resolving them, and the paper has
    never had one outside `macros.tex`.
    """
    bad = []
    for path in sorted(SECTIONS.rglob("*.tex")):
        for m in re.finditer(r"\\(?:re)?newcommand|\\def(?=\\)", source(path)):
            bad.append(f"  {path.relative_to(PAPER).as_posix()} defines a macro")
    assert not bad, "\n".join(sorted(set(bad))) + "\n  (define it in macros.tex)"


# --- the hooks ------------------------------------------------------------------

def test_every_deferred_block_is_hooked_once_and_listed_once():
    """Hooked from one body section so it reads in place; listed in one appendix so it
    reads in the other build. Missing from a wrapper and it vanishes from main.pdf."""
    hooks: dict[str, list[str]] = {}
    listed: dict[str, list[str]] = {}
    for path in [*SECTIONS.glob("*.tex"), *MOVED.glob("*.tex")]:
        text = source(path)
        for name in _HOOK.findall(text):
            hooks.setdefault(name, []).append(path.name)
        if path.stem in WRAPPERS:
            for ref in _INPUT.findall(text):
                if ref.startswith("moved/"):
                    listed.setdefault(ref.split("/", 1)[1], []).append(path.name)

    problems = []
    for path in sorted(MOVED.glob("*.tex")):
        name = path.stem
        h, l = hooks.get(name, []), listed.get(name, [])
        # A block nested inside another deferred block is included with its parent, so it
        # is hooked unconditionally there and listed nowhere.
        nested = any(
            f"\\input{{sections/moved/{name}}}" in source(p)
            for p in MOVED.glob("*.tex") if p != path
        )
        if nested:
            if h or l:
                problems.append(f"  {name}: nested in another block, but also {h+l}")
            continue
        if len(h) != 1:
            problems.append(f"  {name}: hooked from {len(h)} places {h}")
        if len(l) != 1:
            problems.append(f"  {name}: listed in {len(l)} appendices {l}")
    assert not problems, "\n".join(problems)


def test_no_deferred_block_inputs_an_appendix_wrapper():
    """This is the exact shape of the five-page duplication: `\\iffull\\input{app_*}` inside
    a file that is itself deferred pulls the whole wrapper, and its ten blocks, in again."""
    bad = []
    for path in sorted(MOVED.glob("*.tex")):
        for ref in _INPUT.findall(source(path)):
            if Path(ref).name in WRAPPERS:
                bad.append(f"  {path.name} inputs the wrapper sections/{ref}")
    assert not bad, "\n".join(bad)


def test_a_deferred_block_does_not_guard_its_own_children():
    """A deferred file is included exactly once either way, so an `\\iffull` inside it is
    false in the build where it lands in an appendix and its child disappears."""
    bad = []
    for path in sorted(MOVED.glob("*.tex")):
        for name in _HOOK.findall(source(path)):
            bad.append(f"  {path.name} hooks {name} with \\iffull; use a plain \\input")
    assert not bad, "\n".join(bad)


_DISPLAY_END = re.compile(
    r"\\end\{(theorem|lemma|proposition|corollary|definition|remark|table|figure|"
    r"center|align\*?|enumerate|itemize|proof)\}"
)


def test_an_indent_suppressed_after_a_hook_is_conditional():
    """A bare `\\noindent` after a hook is right only if a display precedes the hook too.

    The common shape is fine: `\\end{lemma}`, then the hooks, then `\\noindent` -- a display
    sits above the paragraph in both builds, the statement in one and the deferred remark
    in the other. What is wrong is a hook whose preceding prose is an ordinary paragraph,
    where the complete build gets its display from the hook and the deferred build gets
    nothing, so the paragraph merely loses its indent. Both instances of that were
    introduced by moving a block out from under the paragraph that led into it.
    """
    bad = []
    for path in sorted(SECTIONS.glob("*.tex")):
        lines = source(path).split("\n")
        for i, line in enumerate(lines):
            if not _HOOK.fullmatch(line.strip()):
                continue
            # What stands above this run of hooks in the deferred build?
            above = ""
            for j in range(i - 1, -1, -1):
                s = lines[j].strip()
                if s and not _HOOK.fullmatch(s):
                    above = s
                    break
            if _DISPLAY_END.search(above):
                continue
            for j in range(i + 1, min(i + 4, len(lines))):
                s = lines[j].strip()
                if s == "\\noindent":
                    bad.append(
                        f"  {path.name}:{j+1} bare \\noindent after a hook whose "
                        f"deferred-build predecessor is prose"
                    )
                    break
                if s and not _HOOK.fullmatch(s):
                    break
    assert not bad, "\n".join(bad) + "\n  (wrap as \\inlineonly{\\noindent}%)"


def test_apref_is_used_only_where_it_can_resolve():
    """`\\apref` names the appendix a block was deferred to, which only makes sense from the
    body. Inside a deferred block it tells the reader to consult the page they are on."""
    bad = []
    for path in sorted(MOVED.glob("*.tex")):
        text = source(path)
        for macro in ("\\apref{", "\\apin{"):
            if macro in text:
                bad.append(f"  {path.name} uses {macro[:-1]}; cite the block by its own label")
    assert not bad, "\n".join(bad)


def test_every_apref_target_exists():
    targets = set()
    for name in WRAPPERS:
        text = source(SECTIONS / f"{name}.tex")
        targets.update(re.findall(r"\\apphead\{[^}]*\}\{([^}]*)\}", text))
    used = set()
    for path in [*PAPER.glob("*.tex"), *SECTIONS.rglob("*.tex")]:
        text = source(path)
        used.update(re.findall(r"\\apref\{([^}]*)\}", text))
        used.update(re.findall(r"\\apin\{([^}]*)\}", text))
    assert used <= targets, f"\\apref to labels no wrapper defines: {sorted(used - targets)}"


def test_an_appendix_pointer_carries_its_own_preposition():
    """`\\apref` renders as "below", so a sentence needing "in" must use `\\apin`.

    Two call sites read "is set out Appendix~B" in the deferred build for three passes,
    because the sentence frame that works for "below" silently drops the preposition
    the appendix form needs. The macro now carries it; this keeps the frames apart.
    """
    bad = []
    for path in [*PAPER.glob("*.tex"), *SECTIONS.rglob("*.tex")]:
        # `source`, not a document: `rendered` expands `\apref` to what each build
        # prints, and this is a question about the call site rather than about either
        # rendering -- which preposition the author wrote in front of the macro.
        text = flat(source(path))
        # A PREPOSITION BEFORE `\apref` IS ALWAYS WRONG, because `\apin` is the macro
        # that carries one. The earlier version of this guard tested a whitelist of
        # leading words -- "out", "collected", "given", "listed", "recorded",
        # "stated" -- which omitted "in", and "in \apref" was three of the four sites
        # that printed "in below" in the complete build.
        for m in re.finditer(
                r"\b(in|at|of|from|within|under|into|through)\s+\\apref\{", text):
            bad.append(f'  {path.name}: "{m.group(1)} \\apref" wants \\apin')
        for m in re.finditer(r"(\w+(?:\s+\w+){0,2})\s+\\apref\{", text):
            lead = m.group(1).lower().split()[-1]
            if lead in {"out", "collected", "given", "listed", "recorded", "stated"}:
                bad.append(f'  {path.name}: "{m.group(1)} \\apref" wants \\apin')

    # AND WHAT THE COMPLETE BUILD PRINTS, because a call-site rule cannot see the
    # fourth shape: `\apref` used as a subject gives "below collects what the vendor
    # was asked", which is not a sentence. "below" is an adverb in that build and has
    # to sit where one fits.
    # "below" AS A SUBJECT, which needs a clause boundary in front of it. "Every claim
    # below names the rho it holds" is fine -- there "below" post-modifies "claim" --
    # so the lead has to be a conjunction, a relative pronoun or punctuation for this
    # to be the broken shape.
    _VERB_AFTER = re.compile(
        r"(?:\band\b|\bbut\b|\bor\b|\bwhich\b|\bthat\b|[,;:.])\s+below\s+"
        r"(collects?|gives?|reports?|treats?|lists?|shows?|states?|sets? out|"
        r"records?|holds?|carries|names?)\b")
    for path in [*PAPER.glob("*.tex"), *SECTIONS.rglob("*.tex")]:
        said = flat(rendered(path, True))
        for pattern, why in ((re.compile(r"\bin below\b"), "'in below'"),
                             (_VERB_AFTER, "'below' as a subject")):
            for m in pattern.finditer(said):
                bad.append(f"  {path.name} prints {why} in main_full.pdf: "
                           f"{said[max(0, m.start() - 40):m.end() + 20].strip()!r}")
    assert not bad, "\n".join(sorted(set(bad))) + (
        "\n  (\\apref renders as 'below'; \\apin as 'in Appendix N')")


def test_make_paper_is_reproducible_from_a_clean_tree():
    """Guards the thing the log tests assume: that `make paper` is what produced them."""
    if not os.environ.get("QLPORT_BUILD_PAPER"):
        pytest.skip("set QLPORT_BUILD_PAPER=1 to rebuild both PDFs inside the suite")
    done = subprocess.run(["make", "paper"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


# --- figures a generated table states and the prose restates ---------------------

def _registry() -> dict:
    import json
    return json.loads((PAPER / "quoted_figures.json").read_text())


def test_the_compared_shape_count_matches_the_differential_table():
    """§7 said 323 against the table's 329, for a run whose own fixes had moved it.

    The count has no registry key, so `BOUND` cannot reach it and the allowlist carried
    the stale value as "unregistered". The table is generated, so it is the authority.
    """
    printed = table(PAPER / "tables" / "differential.tex")
    m = re.search(r"Compared against both systems\s*&\s*([\d,]+)\s*&", printed)
    assert m, "the differential printed no longer has a 'Compared against both systems' row"
    compared = m.group(1)
    spelled = compared.replace(",", "{,}")
    total = _registry()["nb05.shapes"]
    uncompared = total - int(compared.replace(",", ""))
    assert uncompared >= 0
    for prose in builds(SECTIONS / "07_evaluation.tex"):
        assert f"${spelled}$" in prose, (
            f"tables/differential.tex compares {compared} shapes; §7 does not say so"
        )
        assert f"${total}$" in prose, (
            f"§7 should state the shape total {total} beside the {compared} compared"
        )


def test_the_feature_bound_slack_is_the_registry_value():
    """The feature-bound slack, which §7 once quoted as a different number entirely.

    `BOUND` will not take it -- 87 is too plain a value for a whole-file match to mean
    anything, which that file enforces deliberately. So it is pinned here to the clause
    that states it, which is narrow enough that the figure cannot be read off some other
    sentence. Deriving it from nb03.feature_ceiling_pct does not work: 97.3% is rounded,
    and the five counts it could stand for span a range wider than the error being
    caught.
    """
    slack = _registry()["nb03.feature_slack_occurrences"]
    for build in builds(SECTIONS / "07_evaluation.tex"):
        prose = flat(build)
        clause = re.search(
            r"the gap is \$([\d,]+)\$ occurrences that use nothing outside", prose)
        assert clause, "§7 no longer has the feature-bound slack clause this pins"
        assert clause.group(1).replace("{,}", "").replace(",", "") == str(slack), (
            f"§7 states the slack as {clause.group(1)}; the registry computes {slack}"
        )


def test_the_promql_population_matches_the_parse_table():
    """§4 and two appendix blocks quoted 8,096 where the table says 8,102.

    The population has no registry key, so `BOUND` could not reach it and the allowlist
    carried the stale value with the reason "unregistered (nb 00): PromQL expressions"
    -- an entry asserting that the dead figure was the live one. Two of the three stale
    sites sat within two lines of a generated figure whose own legend printed 8,102.
    The table is generated, so it is the authority, and the pre-correction value is
    barred outright rather than left to an allowlist to decline to question.
    """
    printed = table(PAPER / "tables" / "parse_coverage.tex")
    m = re.search(r"promql\s*&\s*([\d,]+)\s*&", printed)
    assert m, "tables/parse_coverage.tex no longer has a promql row"
    spelled = m.group(1).replace(",", "{,}")
    quoting = {
        "04_grammars.tex", "07_evaluation.tex",
        "moved/eval_language_use.tex", "moved/eval_staircase.tex",
    }
    for rel in sorted(quoting):
        for text in builds(SECTIONS / rel):
            assert f"${spelled}$" in text, (
                f"{rel} should state the PromQL population as {m.group(1)}"
            )
    for path in sorted(SECTIONS.rglob("*.tex")):
        assert "8{,}096" not in anywhere(path), (
            f"{path.name} still quotes the pre-correction PromQL population"
        )


@pytest.mark.parametrize("build", BUILDS)
def test_the_build_converged(build):
    """LaTeX's own rerun warning, which a two-pass build earns and nobody reads.

    The contents pages have a length, and that length shifts every page number after
    them. A build that typesets the contents from the pass before it finishes renders
    numbers that predate the contents themselves --- every entry low by the length of
    the contents, with a correct `.toc` on disk and a wrong one inside the PDF. It
    happens only on a clean tree, which is where `make paper` runs and not where a
    person rebuilding one section would ever see it. When this paper first gained a
    table of contents it shipped exactly that: the contents said the introduction began
    on page 2 and the introduction began on page 4.

    LaTeX reports it in one line. This is that line, read rather than scrolled past.
    """
    log = _log(build)
    stale = [m for m in ("Rerun to get", "may have changed") if m in log]
    assert not stale, (
        f"paper/{build}.log asks for another pass ({', '.join(stale)}): the PDF's "
        "cross-references and table of contents are a pass behind its sources"
    )


def test_no_module_defines_a_private_helper_nothing_calls():
    r"""A helper that reads like a check and is called by nothing is worse than none.

    `_vendor_dates` walked every vendor cache, pulled each `collected_at`, and carried
    its own `assert out, "no vendor cache carries a timestamp"`. Nothing called it. It
    read exactly like the guard behind `tables/vendor_provenance.tex`, which is why
    moving a date in that table by thirteen months passed -- the check it looked like
    did not exist, and its existence as a function is what stopped anyone writing one.
    Two others, `_math_text` and `_live_span_files`, were dead the same way.

    So the suite may not carry one. A helper written ahead of its caller fails here
    until the caller arrives, which is the right order for a check and its use.
    """
    dead = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        tree = ast.parse(path.read_text())
        defined = {n.name: n.lineno for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name.startswith("_")
                   and not n.name.startswith("__")}
        # A helper defined in this module is called by bare name. The first version of
        # this also counted every attribute name and every string constant as a use,
        # which is two escapes: a module-level `_cluster_bootstrap` passed because the
        # module calls `mod._cluster_bootstrap(...)` on an imported script, and any
        # dead helper passed if its name appeared anywhere as a string literal. The
        # two deliberate dynamic references -- `globals()["name"] = ...` for the
        # monkeypatch in `test_paper_figures`, and `getattr(x, "name")` -- are counted
        # for what they are rather than by letting every string count.
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for n in ast.walk(tree):
            if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name)
                    and n.value.func.id == "globals"
                    and isinstance(n.slice, ast.Constant)
                    and isinstance(n.slice.value, str)):
                used.add(n.slice.value)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "getattr" and len(n.args) > 1
                    and isinstance(n.args[1], ast.Constant)
                    and isinstance(n.args[1].value, str)):
                used.add(n.args[1].value)
        for name, line in sorted(defined.items(), key=lambda kv: kv[1]):
            if name not in used:
                dead.append(f"  {path.name}:{line} {name} is defined and never called")
    assert not dead, "\n".join(dead) + "\n  (call it, or delete it)"


def test_the_derived_inputs_the_suite_reads_are_present():
    """Named here so a fresh clone gets one legible failure instead of thirty-seven.

    `data/processed/` and the corpus sources are gitignored because they are derived,
    and `make data` is deliberately not a dependency of `make test` -- a machine
    without docker should still be able to run the suite against data it already has.
    The cost is that a clean checkout fails thirty-seven tests on missing input, each
    with a `FileNotFoundError` or a `KeyError` that looks exactly like a real failure.
    A reviewer cloning this repository cannot tell them apart.

    This does not fix that. It gives it a name and the command, so the one failure
    anybody reads first says what to run.
    """
    # `features.csv` is NOT here. It was, under a docstring saying these are the
    # inputs "the suite reads", and nothing read it: truncating it to five lines left
    # the suite green. Notebook 00 writes it and §7's feature census is derived inside
    # that notebook, so it is a published artefact rather than a suite input, and
    # listing it here claimed a dependency that does not exist.
    needed = [
        ROOT / "data" / "processed" / "translations.csv",
        ROOT / "data" / "processed" / "heldout_eval.json",
        ROOT / "data" / "processed" / "heldout_cv.json",
        ROOT / "data" / "processed" / "rollup_spans.json",
        ROOT / "data" / "corpus" / "queries.jsonl",
    ]
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
    assert not missing, (
        "the suite reads derived inputs this checkout does not have:\n  "
        + "\n  ".join(missing)
        + "\n  run `make notebooks && make data` (both are gitignored because they are "
          "derived, and neither is a dependency of `make test`). Every other failure "
          "about a missing file or a missing key is this one."
    )


def test_the_build_variant_macros_are_the_ones_rendered_expands():
    r"""`macros.tex` decides which macros are build-conditional; `rendered` must know
    all of them, and only them.

    `BUILD_VARIANT` held two for a commit --- `\inlineonly` and `\shortonly` --- while
    `macros.tex` defined five. `\apref`, `\apin` and `\apphead` are conditional in
    exactly the same way ("below" against "Appendix~N", a subsection against a
    section), so a sentence containing one was checked against the macro call rather
    than against either rendering, and `\apphead`'s heading level against neither.

    The set is read off `macros.tex` rather than listed here, so defining a sixth is
    what fails, which is the only arrangement that does not need the author to
    remember. `\todo` is excluded: it is conditional on nothing and its body has no
    `\iffull`.
    """
    text = source(PAPER / "macros.tex")
    defined = {m.group(1) for m in re.finditer(
        r"\\newcommand\{\\(\w+)\}(?:\[\d+\])?\{((?:[^{}]|\{[^{}]*\})*)\}", text)
        if r"\iffull" in m.group(2)}
    assert defined, "no build-conditional macro found in macros.tex; the scan broke"
    assert defined == set(BUILD_VARIANT), (
        f"macros.tex defines build-conditional macros rendered does not expand: "
        f"{sorted(defined - set(BUILD_VARIANT))}\n"
        f"BUILD_VARIANT names macros macros.tex does not define: "
        f"{sorted(set(BUILD_VARIANT) - defined)}"
    )
