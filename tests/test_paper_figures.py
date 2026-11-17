"""Every figure the paper states in prose must come from somewhere checkable.

Three numbers went stale in one afternoon before this existed: the abstract kept
$\\kappa$ at 95.6% after it moved to 95.5%, and §7 kept a translated count and an
emitted-expression count that were each one too high. None was visible in any table,
and nothing recomputed them, so nothing could notice.

The rule here is not that every numeral must match something. It is that every numeral
must be *accounted for*: either it appears in a generated table, or in the figures
registry a notebook writes, or in the allowlist below with a reason. A new unexplained
number fails, which forces the decision rather than letting it pass.

What this catches is arithmetic drift, which was eight of the eleven problems the last
full read-through found. It does not catch a number that is right and means the wrong
thing -- §5 quoted 87% for the kernel's reach when 87% was the implementation-order
figure -- and nothing mechanical will.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = sorted((ROOT / "paper" / "sections").glob("*.tex"))
MAIN = ROOT / "paper" / "main.tex"
TABLES = sorted((ROOT / "paper" / "tables").glob("*.tex"))
REGISTRY = ROOT / "paper" / "quoted_figures.json"

_NUM = re.compile(r"(\d[\d{},]*(?:\.\d+)?)\s*(\\%)?")


def _norm(tok: str, pct: str) -> tuple[str, float] | None:
    try:
        return ("pct" if pct else "num", round(float(tok.replace("{,}", "")
                                                   .replace(",", "")), 4))
    except ValueError:
        return None


def _math(text: str) -> list[str]:
    """The `$...$` spans, with LaTeX comments and escaped dollars removed first.

    `\\%` is not a comment and `\\$65M` is currency rather than math; both bit an
    earlier version of this extractor and silently emptied its input.
    """
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = text.replace(r"\$", "\x00")
    return re.findall(r"\$([^$]*)\$", text)


def _prose_figures() -> dict[tuple[str, float], set[str]]:
    out: dict[tuple[str, float], set[str]] = {}
    for path in [*SECTIONS, MAIN]:
        for span in _math(path.read_text()):
            for tok, pct in _NUM.findall(span):
                key = _norm(tok, pct)
                if key:
                    out.setdefault(key, set()).add(path.name)
    return out


def _accounted() -> set[tuple[str, float]]:
    ok: set[tuple[str, float]] = set()
    for path in TABLES:
        for tok, pct in _NUM.findall(re.sub(r"(?<!\\)%.*", "", path.read_text())):
            key = _norm(tok, pct)
            if key:
                ok.add(key)
                ok.add(("pct" if key[0] == "num" else "num", key[1]))
    if REGISTRY.exists():
        for value in json.loads(REGISTRY.read_text()).values():
            if isinstance(value, (int, float)):
                ok.add(("num", round(float(value), 4)))
                ok.add(("pct", round(float(value), 4)))
    return ok


# Figures that are neither tabulated nor registry-backed, each with the reason it is
# not. "Historical" means the prose deliberately quotes a superseded value to describe
# how a figure changed; those must not be recomputed. Entries marked "unregistered"
# are live figures that no notebook records yet -- they are the next ones to migrate,
# and listing them here is a statement that they are unchecked, not that they are fine.
ALLOWED: dict[float, str] = {
    # --- constants of the apparatus -------------------------------------------------
    3.15: "Prometheus version the measurements were taken against",
    3.7: "Prometheus version that introduced the anchored modifier",
    15: "scrape interval, seconds", 60: "rollup interval, seconds",
    20: "automatic rollup interval over an hour, seconds",
    300: "automatic rollup interval over a day, seconds",
    1800: "the one rollup width the differential window cannot hold two of",
    0: "zero", 1: "one", 2: "two", 3: "three", 6: "six",
    10: "ten", 64: "the glob-union bound B", 100: "a percentage scale factor",
    # --- E9: what `.weighted()` does, and what it would have done if weighted --------
    66.7: "E9: the plain mean of the irregular series, with and without the modifier",
    16.7: "E9: what a mean weighted by each point's interval would have read",
    56: "E9: distinct corpus forms carrying .weighted()",
    54: "E9: of those, the ones attaching it to sum:",
    0.8: "points of kappa that implementing .weighted() closed",
    # --- E1, E7, E8: values from cached experiment records ---------------------------
    1000: "E1: the counter value just before the scripted reset",
    103: "E1: what Datadog stored over the reset bucket",
    897: "E1: what a raw signed difference would have stored",
    25: "series-start check: the bucket sum for a series older than the window",
    1760: "E7: avg_over_time of the running totals at the end of the run",
    0.994: "E7: the relative difference after half an hour",
    0.5: "E7: the relative difference at the start of the run",
    # --- corpus figures from notebooks 00, 01, 04, 05: unregistered -------------------
    28501: "unregistered (nb 00): query strings before the LogQL drop",
    28490: "unregistered (nb 00): corpus size",
    11628: "unregistered (nb 00): Datadog formula expressions",
    8955: "unregistered (nb 00): space-aggregator occurrences",
    8472: "unregistered (nb 04): PromQL regex matchers",
    8096: "unregistered (nb 00): PromQL expressions",
    8766: "unregistered (nb 00): Datadog metric queries",
    7863: "unregistered (nb 00): PromQL panel queries",
    5389: "unregistered (test_corpus): distinct Datadog metric names",
    5227: "unregistered (nb 00): PromQL function occurrences",
    1013: "unregistered (nb 04): concrete regex patterns",
    763: "unregistered (nb 00): source documents",
    723: "unregistered (test_corpus): distinct grouping keys",
    625: "unregistered (nb 04): patterns with a glob",
    481: "unregistered (nb 00): distinct formula strings",
    388: "unregistered (nb 04): refused patterns",
    344: "unregistered (nb 04): patterns needing a union",
    281: "unregistered (nb 04): patterns with a single glob",
    238: "unregistered (nb 04): refusals that are .+ or contain it",
    221: "unregistered (nb 01): PromQL queries the first six features complete",
    183: "unregistered (nb 04): distinct concrete regexes",
    168: "unregistered (nb 04): of those, re-decided within budget",
    139: "unregistered (nb 00): top() occurrences, and boolean-scope queries",
    137: "unregistered (nb 00): PromQL alerting rules",
    133: "unregistered (nb 04): occurrences of .+",
    96: "unregistered (nb 00): PromQL recording rules",
    93: "unregistered (nb 01): queries needing a bare metric name",
    70: "unregistered (nb 03): .weighted() occurrences",
    63: "unregistered (nb 04): unions the complete procedure reached",
    49: "unregistered (nb 04): occurrences of the most elaborate regex",
    45: "unregistered (nb 04): refusals the complete procedure reached",
    36: "unregistered (nb 03): FILL_INTERPOLATION occurrences",
    34: "unregistered (nb 03): percentile-aggregator occurrences",
    31: "unregistered (nb 03): LABEL_NAME_GLOB occurrences",
    22: "unregistered (nb 00): per_second() occurrences",
    9: "unregistered (nb 04): occurrences of .*",
    8: "unregistered (nb 03): unparsed occurrences",
    99.9: "unregistered (nb 01): Datadog instantiated parse rate, v2",
    99.1: "unregistered (nb 01): PromQL instantiated parse rate",
    98: "unregistered (nb 02): upper end of the cumulative-difference agreement",
    95: "unregistered (nb 00): share of Datadog space aggregators that are sum/avg",
    92.6: "unregistered (nb 01): Datadog instantiated parse rate, v1",
    88: "unregistered (nb 04): share of regex matchers that are template variables",
    82.6: "unregistered (nb 00): formulas that are a bare reference",
    82: "unregistered (nb 00): PromQL queries carrying a template variable",
    71.2: "unregistered (nb 01): PromQL raw parse rate",
    69: "unregistered (nb 00): share of PromQL functions that are rate/irate",
    61.7: "unregistered (nb 04): patterns with a glob, as a share",
    56: "unregistered (nb 00): Datadog queries using by {...}",
    38.3: "unregistered (nb 04): refusals as a share of concrete patterns",
    34.0: "unregistered (nb 04): union share",
    27.7: "unregistered (nb 04): single-glob share",
    27.4: "unregistered (replay): aligned-grid agreement on the bucket edge",
    23: "unregistered (nb 03): shapes covering 80% -- also an allowed small integer",
    21962: "unregistered (nb 05): bucket comparisons in the differential",
    19: "unregistered (replay): lower end of rate's bucket disagreement",
    11.8: "unregistered (nb 01): Datadog raw parse rate",
    11.5: "unregistered (nb 00): formulas applying one function",
    8337: "unregistered (nb 05): occurrences compared in the differential",
    7.5: "unregistered (nb 00): formulas containing arithmetic",
    6.6: "unregistered (nb 00): as_count to as_rate ratio",
    4.8: "unregistered (nb 00): formulas with a numeric constant",
    4.4: "unregistered (nb 03): the bottom share",
    2.7: "unregistered (nb 01): PromQL queries the first six features complete",
    1.68: "unregistered (nb 02): worst relative error under a reset",
    1.4: "unregistered (nb 03): points of kappa the disjunction rule closed",
    1.1: "unregistered (nb 00): PromQL subquery share",
    0.9: "unregistered (nb 03): the 90% line on the coverage figure",
    0.3: "unregistered (nb 00): PromQL offset share",
    323: "unregistered (nb 05): shapes compared in the differential",
    2: "two",
}


# Historical figures: a value the prose quotes deliberately to describe how something
# changed. These are scoped to the sections entitled to say them, because an unscoped
# entry would license the stale value *anywhere* -- and a stale kappa in the abstract is
# exactly the bug this file exists to catch. The value and the place are both the claim.
ALLOWED_IN: dict[float, tuple[str, set[str]]] = {
    348: ("refusals under the commutation argument",
          {"01_introduction.tex", "05_translation.tex", "08_discussion.tex"}),
    124: ("disjunctive scopes before prop:disjunction",
          {"01_introduction.tex", "06_regular.tex", "07_evaluation.tex",
           "08_discussion.tex"}),
    120: ("of those, the single-key ones",
          {"01_introduction.tex", "07_evaluation.tex", "08_discussion.tex"}),
    116: ("translations once thought to carry ROLLUP_ALIGNMENT alone",
          {"07_evaluation.tex"}),
    94.2: ("kappa before the disjunction rule", {"07_evaluation.tex"}),
    95.6: ("kappa after it and before the counter-average fix", {"07_evaluation.tex"}),
    76.2: ("METRIC_TYPE_UNKNOWN before E6", {"05_translation.tex"}),
    6381: ("the occurrences that share carried", {"05_translation.tex"}),
    87: ("the implementation-order figure, contrasted with the kernel's reach",
         {"04_grammars.tex", "05_translation.tex", "07_evaluation.tex"}),
}


@pytest.fixture(scope="module")
def unexplained():
    ok = _accounted()
    allowed = {round(float(v), 4) for v in ALLOWED}
    out = {}
    for key, files in _prose_figures().items():
        if key in ok or key[1] in allowed:
            continue
        scoped = ALLOWED_IN.get(key[1])
        if scoped and set(files) <= scoped[1]:
            continue
        out[key] = sorted(files)
    return out


def test_the_registry_was_written():
    assert REGISTRY.exists(), f"{REGISTRY.name} missing; run `make notebooks`"


def test_every_prose_figure_is_accounted_for(unexplained):
    """A numeral in the prose must be in a table, in the registry, or allowlisted.

    If this fails on a number you just added, the fix is usually to record it in a
    notebook's figures block rather than to add it below: an allowlist entry says the
    number is unchecked, which is the state that let the last three drift.
    """
    assert not unexplained, "\n".join(
        f"  {kind} {value:g} in {', '.join(files)}"
        for (kind, value), files in sorted(unexplained.items()))


def test_the_allowlist_has_no_dead_entries():
    """An allowlist that outlives its numbers stops being a record of anything."""
    prose = {k[1] for k in _prose_figures()}
    dead = sorted(v for v in {**ALLOWED, **ALLOWED_IN}
                  if round(float(v), 4) not in prose)
    assert not dead, f"allowlisted but no longer in the paper: {dead}"


def test_a_historical_figure_is_not_licensed_outside_its_section():
    """The scoping is the point, so it is checked rather than trusted.

    Putting the superseded kappa in the abstract -- the exact drift that prompted this
    file -- must fail even though §7 is entitled to quote the same number.
    """
    abstract = MAIN.read_text()
    assert "95.6" not in abstract, "stale kappa in the abstract"
    stale = {k: v for k, v in _prose_figures().items() if k[1] == 95.6}
    assert stale, "the historical kappa should still be quoted somewhere"
    assert all(set(f) <= ALLOWED_IN[95.6][1] for f in stale.values()), stale


def test_the_registry_agrees_with_the_tables_on_kappa():
    """The two independent paths to the same figure must not diverge.

    The tables are written by one notebook cell and the registry by another; if a
    future edit recomputed either differently, every prose check would still pass
    while the paper contradicted itself.
    """
    reg = json.loads(REGISTRY.read_text())
    coverage = (ROOT / "paper" / "tables" / "translation_coverage.tex").read_text()
    assert f"{reg['nb03.kappa_occurrences']:,}" in coverage
    assert f"{reg['nb03.kappa_distinct']:,}" in coverage
    assert f"{reg['nb03.kappa_pct']}\\%" in coverage
