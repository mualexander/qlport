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
    assert cases >= 70, f"the battery has shrunk to {cases} cases"


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


# The limit the two twenty-page versions are cut to. Not an exact pin: shortening either
# is always allowed, and only the spill is a defect.
TWENTY = 20


@pytest.mark.parametrize("build", ("main_short", "main_switching"))
def test_neither_twenty_page_version_runs_past_twenty_pages(build):
    """The length is the whole point of these two, and nothing here measured it.

    `main_switching` sat at exactly twenty. Six words added to its introduction --- one
    clause naming a term --- pushed the last two bibliography entries onto a
    twenty-first page, and the only test that reads a page count compares `main` with
    `main_full` and excludes these two by design. So the suite reported a clean build of
    a version that no longer fits what it is cut to fit.

    Like every check that reads a log, this one is about the last `make paper` and not
    about the sources: `_log` refuses a log that does not match them. That is why the
    perturbation battery has no case for it --- a case that edits a `.tex` would trip
    the digest instead, which reads as a stale build rather than as a spill.
    """
    found = re.findall(r"\((\d+) pages", _log(build))
    assert found, f"no page count in {build}.log"
    pages = int(found[-1])
    assert pages <= TWENTY, (
        f"{build}.pdf is {pages} pages and is cut to {TWENTY}; the overflow is the last "
        f"{pages - TWENTY} page(s), which is where the bibliography goes first"
    )


# --- the claims, by name --------------------------------------------------------

# The two twenty-page papers against the long one: `paper/sections/` is what `main` and
# `main_full` read, and `paper/short/` and `paper/switching/` are the other two
# documents. A claim about what *a paper* says has to be asked of each of them
# separately, because they are three arguments and not three renderings.
PAPERS = {
    "the long paper": (SECTIONS, SECTIONS / "01_introduction.tex"),
    "the switching paper": (PAPER / "switching", PAPER / "switching" / "01_intro.tex"),
    "the short paper": (PAPER / "short", None),
}

# `\emph{language-design conjecture}` is the naming, and it is the only place either
# paper may introduce the term.
_NAMING = r"\emph{language-design conjecture}"


def _said(directory: Path) -> dict[str, str]:
    """Every file of one paper, as both builds print it, flattened for phrase matching."""
    out = {}
    for path in sorted(directory.rglob("*.tex")):
        rel = path.relative_to(PAPER).as_posix()
        for i, text in enumerate(builds(path)):
            out[f"{rel}[{i}]"] = flat(text)
    return out


@pytest.mark.parametrize("paper", sorted(PAPERS))
def test_no_claim_in_any_paper_is_called_a_thesis(paper):
    r"""One word, two propositions, and they are each other's negation.

    "The thesis" named the language-design conjecture --- that what makes a migration
    expensive is a consequence of how the two languages are designed --- in eight
    places, among them every sentence confessing that a correction had run "towards the
    thesis" or "would have made the thesis look better", where the conjecture is what
    the error flattered. It named the opposite in two: `\subsection{The thesis survives
    in the other direction}` and an introduction that opened "The thesis survives, but
    not in the form it was posed". And in \S7.4 it named a third thing, the finding the
    paper now leads on and reports *instead* of the conjecture --- two baseline figures
    were "this paper's own thesis arriving from outside it", which is the dashboard
    layer and not the conjecture at all. So a reader met a claim that had been refuted,
    a claim that had been upheld and the claim that replaced both, under one word, with
    the refutation and the survival two subsections apart.

    Nothing recomputes prose, so the word is forbidden instead. The conjecture is named
    once per paper and its answer is directional, which the two guards below hold it to;
    the finding is called the finding. `thesis` here is the word, not the suffix:
    `hypothesis`, `parenthesis` and `synthesises` all end in it and all stay.
    """
    directory, _ = PAPERS[paper]
    bad = sorted({where for where, said in _said(directory).items()
                  if re.search(r"\bthes[ei]s\b", said)})
    assert not bad, (
        f"{paper} calls a claim a thesis in {', '.join(bad)} -- the word named the "
        f"conjecture, its negation and the finding that replaced both; say which")


@pytest.mark.parametrize("paper", sorted(PAPERS))
def test_the_conjecture_is_named_once_and_in_the_introduction(paper):
    """A term defined twice is two terms, and one defined late is used before it exists.

    The short paper is on the inexpressibility spine and argues no conjecture, so it
    names none and must not use the word either -- which is the case that makes this a
    three-way check rather than two assertions about the two papers that do.
    """
    directory, intro = PAPERS[paper]
    said = _said(directory)
    namings = {where: s.count(_NAMING) for where, s in said.items() if _NAMING in s}
    users = sorted({where for where, s in said.items() if "conjecture" in s})

    if intro is None:
        assert not users, (
            f"{paper} argues no conjecture and should not use the word: {users}")
        return

    assert sum(namings.values()) == 2, (  # once per build, and both builds print it
        f"{paper} names the conjecture {sum(namings.values())} times across its two "
        f"builds, not once in each: {namings}")
    where = {w.split("[")[0] for w in namings}
    assert where == {intro.relative_to(PAPER).as_posix()}, (
        f"{paper} names the conjecture in {where}, not in its introduction, so every "
        f"earlier use of the term is of a term the reader has not been given")
    assert users, f"{paper} names the conjecture and then never uses it"


def test_the_conjecture_gets_both_halves_of_its_answer():
    r"""Directional, stated as such, and in the two places that disagree.

    The equivocation was possible because the two answers live four subsections apart:
    \S8.1 is headed "The question as posed has a negative answer" and \S8.4 upholds the
    same conjecture for the reverse direction. Either heading alone reads as the whole
    verdict, which is what "The thesis survives" did. So the introduction must carry
    both halves in one sentence, and both headings must stand.

    And the conjecture may not "survive": that verb is what let a claim refuted in the
    direction teams travel read as one that came through. It holds in one direction and
    fails in the other, which is two verbs because it is two answers.
    """
    intro = flat(builds(SECTIONS / "01_introduction.tex")[0])
    assert ("its answer is directional --- it fails in the direction teams travel and "
            "holds, as theorems rather than as measurements, in the direction they do "
            "not") in intro, (
        "the introduction no longer gives both halves of the conjecture's answer in one "
        "sentence, which is where a reader meets it before either section argues it")

    discussion = flat(builds(SECTIONS / "08_discussion.tex")[0])
    for heading in ("The question as posed has a negative answer",
                    "The conjecture holds in the other direction"):
        assert heading in discussion, f"\\S8 no longer carries the heading {heading!r}"

    # By the sentence, not by a window of characters. A window of 120 failed on the
    # paragraph above that heading, which ends "What does not survive it is any claim
    # about a third tool" -- six characters away, a different verb about a different
    # thing. A heading is its own sentence here because it is its own claim: the
    # equivocation this guards was a heading reading as the whole verdict.
    for name, said in _said(SECTIONS).items():
        flattened = re.sub(r"\\(?:sub)*section\{([^}]*)\}", r" \1. ", said)
        for sentence in re.split(r"(?<=[.!?]) ", flattened):
            if "conjecture" in sentence:
                assert not re.search(r"\bsurviv", sentence), (
                    f"{name} has the conjecture surviving: {sentence!r} -- it holds in "
                    f"one direction and fails in the other, and one verb for both is "
                    f"the ambiguity this pair of guards exists to keep out")


def test_the_baseline_figures_arrive_at_the_finding_and_not_at_the_conjecture():
    r"""\S7.4's independent arrival is the dashboard layer, which is the whole point.

    The other translator invents a `[5m]` window and drops `.rollup` modifiers, so what
    it confirms from outside this paper is that a query does not say what it means --- the
    finding \S8.2 argues and the conjecture's refutation makes room for. Calling that
    arrival the paper's "thesis" pointed it at the claim \S8.1 gives a negative answer
    to, so the one piece of outside corroboration the paper has read as corroborating
    the thing the paper withdrew.
    """
    for i, said in enumerate(flat(t) for t in builds(SECTIONS / "07_evaluation.tex")):
        assert ("Two of the figures are \\S\\ref{sec:dashboard-layer}'s finding "
                "arriving from outside this paper.") in said, (
            f"build {i} of \\S7 no longer names what the baseline's two figures arrive "
            f"at; it has to be the finding and not the conjecture")
        assert "conjecture" not in said, (
            f"build {i} of \\S7 mentions the conjecture: the section reports "
            f"measurements, and \\S8 is where either half of its answer is argued")


# --- how deep the qualifications nest -------------------------------------------

# The most qualifications any one sentence may hold open at once, and how many sentences
# are allowed to sit at that number. The ratchet is the same device as
# `UNCHECKED_CEILING` in `test_paper_figures.py`: the bound may fall and may not climb.
NESTING_CEILING = 3
AT_CEILING = 29

# A clause that qualifies what came before it. Opened by a comma and a connective, so
# `, which` counts and the `which` of "the span which the widget pins" does not.
_CLAUSE = re.compile(
    r", (?:which|where|whose|when|while|since|because|although|though|unless|except|"
    r"rather than|and not|so that)\b")

# Environments whose contents are not prose. A table's `(` and a `cases` block's commas
# are not qualifications, and counting them put four tables and two proofs at the top of
# the first ranking this was built from.
_NOT_PROSE = (r"align\*?|equation\*?|cases|tabular|center|enumerate|itemize|array"
              r"|gather\*?")


def _braced(text: str, macro: str, repl: str) -> str:
    r"""`\macro{...}` replaced wholesale, counting braces so a nested one survives.

    A regex cannot do this: `\texttt{\{*-location\}}` has braces inside it, and
    `[^}]*` stops at the first one, leaving a stray `\}}` that reads as prose.
    """
    out, i, tag = [], 0, "\\" + macro + "{"
    while i < len(text):
        j = text.find(tag, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        out.append(repl)
        k, depth = j + len(tag), 1
        while k < len(text) and depth:
            depth += (text[k] == "{") - (text[k] == "}")
            k += 1
        i = k
    return "".join(out)


def _prose(text: str) -> str:
    """One build's file reduced to its prose, with every non-prose span one token."""
    t = flat(text)
    t = re.sub(r"\\\[.*?\\\]", " EQN ", t)
    t = re.sub(r"\\begin\{(" + _NOT_PROSE + r")\}.*?\\end\{\1\}", " BLOCK ", t, flags=re.S)
    # Code spans first: a bracket inside one is syntax. `\texttt{top(...)}` is a name.
    for macro in ("texttt", "verb", "mathit", "mathrm", "mathsf"):
        t = _braced(t, macro, "CODE")
    t = re.sub(r"\$[^$]*\$", "N", t)
    for macro in ("emph", "textbf", "textit"):
        t = _braced(t, macro, "X")
    t = re.sub(r"\\(?:sub)*section\{[^}]*\}|\\paragraph\{[^}]*\}", " ", t)
    t = re.sub(r"\\(?:S|ref|label|apref|apin|cite[a-z]*)\{?[^ }]*\}?", "REF", t)
    # An environment boundary ends a sentence. Without this `\begin{remark}[caption]`
    # glues the caption to the block's first sentence and the pair scores as one.
    t = re.sub(r"\\(?:begin|end)\{[A-Za-z*]+\}(?:\[[^\]]*\])?", ". ", t)
    # So does an `\input`, argument and all. The generic strip below takes the macro
    # name and leaves the path, which glued the sentence before each deferred block's
    # hook to the sentence after it: three sentences scored at the ceiling in the
    # complete build and nowhere else, and all three were this.
    t = re.sub(r"\\input\{[^}]*\}", ". ", t)
    t = re.sub(r"\\[A-Za-z]+\*?", " ", t)
    t = t.replace("\\{", "").replace("\\}", "")
    return re.sub(r"\s+", " ", t)


def _sentences(text: str) -> list[str]:
    """Split on a full stop that a sentence follows. A colon is not a full stop."""
    return [s for s in re.split(r"(?<=[.!?]) (?=[A-Z\"`(])", text)
            if len(s.split()) >= 10 and "BLOCK" not in s and "EQN" not in s]


def _depth(s: str) -> tuple[int, int]:
    r"""How many qualifications stand open at the deepest point of one sentence.

    An aside opens on `(` or the first `---` of a pair and closes on its partner; an
    aside with no partner is open to the full stop, which is the case this was written
    for -- \S5's caveat list had one, and the colon and all seven codes after it read
    as sitting inside it. A subordinate clause opens on `, which` and its kin and is
    counted open until its aside closes or a semicolon ends the item, because an
    unclosed qualification is one the reader is still carrying.
    """
    opens: list[str] = []
    clauses = [0]
    best = at = 0
    i = 0
    while i < len(s):
        if _CLAUSE.match(s, i):
            clauses[-1] += 1
        if s[i] == ";":
            # The item ends here, and with it every clause opened inside it. Without
            # this a seven-item list reads as seven qualifications held at once, which
            # is a long sentence and not a nested one.
            clauses[-1] = 0
        if s[i] == "(":
            opens.append("(")
            clauses.append(0)
        elif s[i] == ")" and opens and opens[-1] == "(":
            opens.pop()
            clauses.pop()
        elif s.startswith("---", i):
            if opens and opens[-1] == "-":
                opens.pop()
                clauses.pop()
            else:
                opens.append("-")
                clauses.append(0)
            i += 2
        if len(opens) + sum(clauses) > best:
            best, at = len(opens) + sum(clauses), i
        i += 1
    return best, at


def _scored(directory: Path) -> list[tuple[int, str, str, int]]:
    """Every prose sentence of one paper, in both builds, with its depth.

    The build index comes back with each row because the two builds hold the same
    prose: counting over the union of their sentences double-counts every sentence
    carrying an `\apref`, which prints "below" in one and "Appendix~N" in the other.
    """
    out = []
    for path in sorted(directory.rglob("*.tex")):
        rel = path.relative_to(PAPER).as_posix()
        for i, text in enumerate(builds(path)):
            for s in _sentences(_prose(text)):
                out.append((_depth(s)[0], rel, s, i))
    return out


def test_the_nesting_scorer_counts_what_it_says_it_counts():
    r"""The instrument, before the two guards that read it.

    A prose measurement nobody checks is worse than none: it reports a number that
    reads as a property of the paper and is a property of the regex. Each case below
    is one decision the scorer makes, and four of them were wrong in the first version
    -- it counted a table's brackets, `\texttt{top(...)}`'s, a caption glued to the
    sentence after it, and every item of a semicolon list at once.
    """
    cases = [
        ("A query does not say what it means at all.", 0),
        ("The interval is absent, which the dashboard supplies.", 1),
        ("The interval is absent, which the dashboard supplies, because it is a view.", 2),
        ("It is absent --- the dashboard supplies it --- and nothing else does.", 1),
        ("It is absent --- the dashboard supplies it, which is a view --- and so on.", 2),
        ("It is absent (the dashboard supplies it, which is a view) and so on.", 2),
        # An aside with no partner stays open: the list after the colon is inside it.
        ("It is absent --- the dashboard supplies it, which is a view: one, two.", 2),
        # A semicolon ends the item, so the clauses do not accumulate down a list.
        ("One, which is a; two, which is b; three, which is c.", 1),
    ]
    for text, want in cases:
        got, _ = _depth(text)
        assert got == want, f"{text!r} scored {got}, not {want}"

    # Brackets that are not qualifications: a code span's, and a table's.
    assert _depth(_prose(r"It emits \texttt{top(5, avg:m\{*\}, mean, desc)} here."))[0] == 0
    assert _prose(r"\begin{tabular}{ll}a & b\\\end{tabular}").strip() == "BLOCK"
    # An environment boundary ends a sentence. The first version of this asserted that
    # a caption plus the sentence after it is one sentence, which is true whether or
    # not the boundary is there, so the battery reported MISSED against a guard that
    # was asserting nothing. What matters is the text BEFORE the block: without the
    # break it runs into the block's first sentence and the two score as one.
    glued = _prose(r"The span the widget carries is absent here, which the "
                   r"dashboard's own time selector supplies "
                   r"\begin{remark}[Why it is]\label{r:x} "
                   r"It is a viewer's choice, which is recorded nowhere, "
                   r"because nothing writes it down.")
    assert len(_sentences(glued)) == 2, glued
    assert max(_depth(s)[0] for s in _sentences(glued)) == 2, glued


@pytest.mark.parametrize("paper", sorted(PAPERS))
def test_no_sentence_holds_more_than_three_qualifications_open(paper):
    r"""The finding this file's ceiling exists for, which nothing had ever measured.

    Four commit messages carried "some sentences carry three or four nested
    qualifications" as an open defect and none of them said which sentences, so the
    claim could not be acted on or closed. Measured, the distribution was 1310 / 812 /
    214 / 37 / 4 / 1 by depth: the tail the finding names is the five sentences at four
    and above, not the 214 at two, and an aside containing a relative clause is
    ordinary prose rather than a defect.

    All five are split now, each keeping every qualification it carried. The worst was
    \S5's: `Seven codes arise, over six conditions ---` opened an aside that never
    closed, so the colon and all seven codes read as inside it, and a reader held it
    for a hundred and twenty words.
    """
    directory, _ = PAPERS[paper]
    over = sorted(((d, where, s) for d, where, s, _ in _scored(directory)
                   if d > NESTING_CEILING), reverse=True)
    assert not over, "\n".join(
        f"{where} holds {d} qualifications open at once:\n  {s[:300]}"
        for d, where, s in over[:4])


def test_the_sentences_at_the_nesting_ceiling_do_not_multiply():
    """A ratchet, because the ceiling alone licenses every sentence up to it.

    Three open qualifications is the most the paper now asks of a reader, and 29
    sentences ask it. Nothing stops that becoming 229 one sentence at a time, which is
    how the tail this closed was built in the first place: no single edit added a
    sentence anybody would have called unreadable.
    """
    rows = [r for paper in PAPERS for r in _scored(PAPERS[paper][0])]
    # Per build, not over the union of the two: the same sentence appears in both, and
    # with one word different wherever it carries an `\apref`, so a union counts those
    # twice and the ledger would drift up by itself.
    per_build = [sum(1 for d, _, _, i in rows if d == NESTING_CEILING and i == b)
                 for b in (0, 1)]
    worst = max(per_build)
    assert worst <= AT_CEILING, (
        f"{worst} sentences hold {NESTING_CEILING} qualifications open, up from "
        f"{AT_CEILING}; split one before adding another")
    assert worst >= AT_CEILING - 4, (
        f"{worst} sentences are at the ceiling, well under the {AT_CEILING} this "
        f"ledger claims -- lower it, so the ratchet keeps biting")

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


MAKEFILE = ROOT / "Makefile"


def _makefile() -> str:
    """The Makefile, raw. It is not LaTeX, so `%` in it is a variable and not a comment.

    One read site rather than three, because `RAW_READS` is keyed on the line that does
    the reading and three copies of the same line are one entry with three places to
    drift from it.
    """
    return MAKEFILE.read_text()


def _data_scripts() -> list[Path]:
    """`DATA_SCRIPTS` as the Makefile defines it, continuations and all."""
    text = _makefile()
    start = text.index("DATA_SCRIPTS :=")
    body, i = "", start
    while i < len(text):
        line = text[i:text.index("\n", i)]
        body += line.rstrip("\\")
        i = text.index("\n", i) + 1
        if not line.rstrip().endswith("\\"):
            break
    return [ROOT / tok for tok in body.split()[2:]]


def _registry_write_is_gated(path: Path) -> bool | None:
    r"""Whether this script's registry write sits behind `--write-registry`.

    `None` when it does not write the registry at all. Read off the syntax tree and
    not by grep: a script that merely mentions the flag in its usage line would pass a
    substring test while writing unconditionally, which is the exact shape of the
    defect this is about.
    """
    tree = ast.parse(path.read_text())
    writes = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "REGISTRY"
    ]
    if not writes:
        return None
    gated: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "write_registry" not in test and "--write-registry" not in test:
            continue
        for stmt in node.body:
            for inner in ast.walk(stmt):
                gated.add(getattr(inner, "lineno", -1))
    return all(w in gated for w in writes)


def _data_script_comment() -> str:
    """The comment block above `DATA_SCRIPTS`, flattened, which is what makes the claims.

    Flattened because a claim is a sentence and a Makefile comment wraps at 88
    columns: every phrase this file looks for spans a line break somewhere, and the
    first version of the key-count check failed on one.
    """
    text = _makefile()
    head = text[:text.index("DATA_SCRIPTS :=")].rstrip("\n").split("\n")
    out = []
    for line in reversed(head):
        if not line.startswith("#"):
            break
        out.append(line.lstrip("#").strip())
    return re.sub(r"\s+", " ", " ".join(reversed(out)))


def test_the_data_scripts_write_the_registry_the_way_the_makefile_says():
    r"""Four claims about code, in a comment nothing had ever read.

    The comment above `DATA_SCRIPTS` said "None is passed --write-registry", which is
    the reproduction story the whole registry rests on: the committed figures are the
    record, and a rerun should disagree with them rather than overwrite them.
    `rollup_span_join.py` takes no arguments at all and merges its 13 `rs.` keys into
    `paper/quoted_figures.json` on every run, so `make data` has always rewritten the
    registry and a drifted span figure would have been written over the committed one.
    The same comment said `dash0_claims.py` "writes nothing at all", which is its
    registry merge; said "these three" of five scripts; and counted the tests that fail
    without the derived inputs at five and four, where removing the two files fails
    fourteen.

    This does not fix the script -- gating that write changes what `make data` does to
    a committed file, which is its own decision. It makes the comment true and keeps it
    true, which is the part that was free.
    """
    scripts = _data_scripts()
    said = _data_script_comment()

    assert len(scripts) == 5, f"DATA_SCRIPTS now names {len(scripts)}, not five"
    for path in scripts:
        assert path.exists(), f"DATA_SCRIPTS names {path.name}, which is not there"

    gated = {p.name: _registry_write_is_gated(p) for p in scripts}
    ungated = sorted(n for n, g in gated.items() if g is False)
    behind = sorted(n for n, g in gated.items() if g is True)

    assert ungated == ["rollup_span_join.py"], (
        f"the comment says rollup_span_join.py is the one script that writes the "
        f"registry ungated; the scripts say {ungated or 'none does'}")
    assert len(behind) == 4, (
        f"{len(behind)} scripts gate the write behind the flag, not the four the "
        f"comment claims: {behind}")

    # The comment's own words, against those counts.
    assert "these five produce the rest" in said
    assert "Four of the five gate their registry write behind --write-registry" in said
    assert "rollup_span_join.py is the exception" in said
    assert "no argument parsing at all" in said

    # `make data` passes the flag to nobody, which is the other half of the claim.
    recipe = _makefile()
    recipe = recipe[recipe.index("\ndata: install"):]
    recipe = recipe[:recipe.index("\n\n")]
    assert "--write-registry" not in recipe, (
        "the `data` recipe passes --write-registry, so a reproduction run rewrites "
        "every figure rather than disagreeing with it")

    # The key count the comment gives for the ungated script.
    keys = {k for k in _registry() if k.startswith("rs.")}
    assert f"merges its {len(keys)} `rs.` keys" in said, (
        f"the registry holds {len(keys)} `rs.` keys and the comment says otherwise")
    join = (ROOT / "scripts" / "rollup_span_join.py").read_text()
    assert len(re.findall(r'"rs\.[a-z_0-9]+":', join)) == len(keys), (
        "the join writes a different number of `rs.` keys than the registry holds")

    # And the one claim about a script that writes no derived file.
    dash0 = ast.parse((ROOT / "scripts" / "dash0_claims.py").read_text())
    processed = [n for n in ast.walk(dash0) if isinstance(n, ast.Constant)
                 and isinstance(n.value, str) and "processed" in n.value]
    assert not processed, (
        f"dash0_claims.py names data/processed, which the comment says it does not "
        f"write: {[n.value for n in processed]}")


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
