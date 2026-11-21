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

import os
import re
import subprocess
from pathlib import Path

import pytest

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


_COMMENT = re.compile(r"(?<!\\\\)%.*")


def _body(path: Path) -> str:
    """The file with LaTeX comments stripped.

    Both pointer checks below first fired on `macros.tex`, whose comment quotes the
    very mis-reading they exist to catch. A scan that cannot tell prose from a comment
    reports the documentation of a bug as the bug.
    """
    return "\n".join(_COMMENT.sub("", line) for line in path.read_text().split("\n"))


def _log(name: str) -> str:
    """The build log, insisting it is newer than every source that went into it.

    A stale log is worse than no log: it reports a clean build of a paper that no longer
    exists. So this fails rather than skips, and says what to run.
    """
    log = PAPER / f"{name}.log"
    assert log.exists(), f"{log.relative_to(ROOT)} is missing -- run `make paper`"
    newest = max(p.stat().st_mtime for p in _sources())
    assert log.stat().st_mtime >= newest, (
        f"{log.relative_to(ROOT)} is older than a paper source -- run `make paper`"
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


# --- the hooks ------------------------------------------------------------------

def test_every_deferred_block_is_hooked_once_and_listed_once():
    """Hooked from one body section so it reads in place; listed in one appendix so it
    reads in the other build. Missing from a wrapper and it vanishes from main.pdf."""
    hooks: dict[str, list[str]] = {}
    listed: dict[str, list[str]] = {}
    for path in [*SECTIONS.glob("*.tex"), *MOVED.glob("*.tex")]:
        text = path.read_text()
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
            f"\\input{{sections/moved/{name}}}" in p.read_text()
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
        for ref in _INPUT.findall(path.read_text()):
            if Path(ref).name in WRAPPERS:
                bad.append(f"  {path.name} inputs the wrapper sections/{ref}")
    assert not bad, "\n".join(bad)


def test_a_deferred_block_does_not_guard_its_own_children():
    """A deferred file is included exactly once either way, so an `\\iffull` inside it is
    false in the build where it lands in an appendix and its child disappears."""
    bad = []
    for path in sorted(MOVED.glob("*.tex")):
        for name in _HOOK.findall(path.read_text()):
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
        lines = path.read_text().split("\n")
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
        text = _body(path)
        for macro in ("\\apref{", "\\apin{"):
            if macro in text:
                bad.append(f"  {path.name} uses {macro[:-1]}; cite the block by its own label")
    assert not bad, "\n".join(bad)


def test_every_apref_target_exists():
    targets = set()
    for name in WRAPPERS:
        text = (SECTIONS / f"{name}.tex").read_text()
        targets.update(re.findall(r"\\apphead\{[^}]*\}\{([^}]*)\}", text))
    used = set()
    for path in [*PAPER.glob("*.tex"), *SECTIONS.rglob("*.tex")]:
        text = _body(path)
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
        flat = " ".join(_body(path).split())
        for m in re.finditer(r"(\w+(?:\s+\w+){0,2})\s+\\apref\{", flat):
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
    table = (PAPER / "tables" / "differential.tex").read_text()
    m = re.search(r"Compared against both systems\s*&\s*([\d,]+)\s*&", table)
    assert m, "the differential table no longer has a 'Compared against both systems' row"
    compared = m.group(1)
    prose = (SECTIONS / "07_evaluation.tex").read_text()
    spelled = compared.replace(",", "{,}")
    assert f"${spelled}$" in prose, (
        f"tables/differential.tex compares {compared} shapes; §7 does not say so"
    )
    total = _registry()["nb05.shapes"]
    uncompared = total - int(compared.replace(",", ""))
    assert uncompared >= 0 and f"${total}$" in prose, (
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
    prose = " ".join((SECTIONS / "07_evaluation.tex").read_text().split())
    clause = re.search(r"the gap is \$([\d,]+)\$ occurrences that use nothing outside", prose)
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
    table = (PAPER / "tables" / "parse_coverage.tex").read_text()
    m = re.search(r"promql\s*&\s*([\d,]+)\s*&", table)
    assert m, "tables/parse_coverage.tex no longer has a promql row"
    spelled = m.group(1).replace(",", "{,}")
    quoting = {
        "04_grammars.tex", "07_evaluation.tex",
        "moved/eval_language_use.tex", "moved/eval_staircase.tex",
    }
    for rel in sorted(quoting):
        text = (SECTIONS / rel).read_text()
        assert f"${spelled}$" in text, (
            f"{rel} should state the PromQL population as {m.group(1)}"
        )
    for path in sorted(SECTIONS.rglob("*.tex")):
        assert "8{,}096" not in path.read_text(), (
            f"{path.name} still quotes the pre-correction PromQL population"
        )
