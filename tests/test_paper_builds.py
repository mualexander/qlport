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
import os
import re
import subprocess
from pathlib import Path

import pytest

from texsource import (COMMENT, RAW_READS, SOURCE_READS, anywhere, builds, flat,
                       rendered, source, source_reads, source_readers, strip_comments,
                       suite_source, table, unstripped_reads)

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SECTIONS = PAPER / "sections"
MOVED = SECTIONS / "moved"
WRAPPERS = {"app_proofs", "app_experiments", "app_eval"}
BUILDS = ("main", "main_full")

_HOOK = re.compile(r"\\iffull\\input\{sections/moved/([A-Za-z0-9_]+)\}\\fi")
_INPUT = re.compile(r"\\input\{sections/([A-Za-z0-9_/]+)\}")


def _sources() -> list[Path]:
    return sorted(
        [*PAPER.glob("*.tex"), *SECTIONS.rglob("*.tex"), *(PAPER / "tables").glob("*.tex")]
    )


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
    newest = max(p.stat().st_mtime for p in _sources())
    assert log.stat().st_mtime >= newest, (
        f"{log.relative_to(ROOT)} is older than a paper source -- run `make paper`"
    )
    stamp = PAPER / ".sources.sha256"
    assert stamp.exists(), (
        f"{stamp.relative_to(ROOT)} is missing -- run `make paper`, which records by "
        f"content what the build read"
    )
    digest = hashlib.sha256()
    for path in _sources():
        digest.update(path.read_bytes())
    want = f"{digest.hexdigest()} {len(_sources())}"
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

    for build in BUILDS:
        opened = re.findall(r"sections/moved/([a-z0-9_]+)\.tex", _log(build))
        assert opened, f"no moved blocks in {build}.log"
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


def test_no_raw_read_is_excused_for_a_line_no_test_has():
    """A stale excuse is an assertion that a line exists which does not."""
    suite = suite_source(ROOT / "tests")
    dead = sorted(line for line in RAW_READS if line not in suite)
    assert not dead, f"RAW_READS excuses lines no test has: {dead}"


def test_the_comment_stripper_reads_a_percent_the_way_latex_does():
    r"""Whether a `%` opens a comment is parity, and this suite has had it wrong twice.

    A lookbehind cannot count, so both one-backslash and two-backslash lookbehinds are
    wrong, in opposite directions: one cuts every line at its first `\%` and reads a
    document smaller than the published one, the other accepts `\\%` -- a forced line
    break followed by a comment -- and reads one larger. The second is the dangerous
    direction, because a guard asserting a sentence is present is satisfied by text no
    reader sees.

    Reverting `COMMENT` to either bug left the whole suite passing, through all
    fifty-one call sites, because the fix was checked by hand in a shell and the
    hand-check was never committed. This is that hand-check. The cases are written out
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
        text = flat(anywhere(path))
        for m in re.finditer(r"(\w+(?:\s+\w+){0,2})\s+\\apref\{", text):
            lead = m.group(1).lower().split()[-1]
            if lead in {"out", "collected", "given", "listed", "recorded", "stated"}:
                bad.append(f"  {path.name}: \"{m.group(1)} \\apref\" wants \\apin")
    assert not bad, "\n".join(bad) + "\n  (\\apref renders as 'below'; \\apin as 'in Appendix N')"


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
    """§7 quoted this gap as 82 where the notebook computes 87.

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
        used = ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
                | {n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)})
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
    needed = [
        ROOT / "data" / "processed" / "translations.csv",
        ROOT / "data" / "processed" / "features.csv",
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
