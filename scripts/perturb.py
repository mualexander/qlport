#!/usr/bin/env python3
"""Perturbation battery: break one claim, run the suite, report which guard fired.

A green suite says nothing about whether a guard would notice the defect it is named
for. This applies each perturbation in turn, runs the tests that could see it, and
reports CAUGHT or MISSED against the test the case names -- so "verified by N
perturbations" in a commit message is a claim this file can be run to check. It lived
outside the repository for four rounds, during which that claim was unverifiable by
anyone but its author; a reviewer said so, which is why it is here.

    python scripts/perturb.py [backup-dir]            # the whole battery, ~20 min
    python scripts/perturb.py --dry-run               # only: do the anchors resolve
    python scripts/perturb.py --only=registry         # the cases whose name matches
    python scripts/perturb.py --only=echo,rollup      # or any of several

`--only` is for the round that touches a handful of guards: running those cases and
the controls beside them is the check, and half an hour for the other ninety is not.
It takes several substrings, comma-separated, because a round's cases are named for
the defects they plant and those names have no word in common.
It narrows the run and not the dry run, which has to resolve every anchor or the
staleness it reports would depend on which cases were asked for.

The dry run is what `tests/test_paper_builds.py` calls. A case whose anchor has gone
stale applies nothing and reports MISSED, which reads as a hole in the suite rather
than a hole in the harness -- that has happened twice, once leaving a planted sentence
in the paper because `restore()` after a loop does not run if the loop raises.

Perturbations are applied to the working tree and restored afterwards. Do not run it
with uncommitted work you care about.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
DRY_RUN = "--dry-run" in sys.argv
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:]
             if a.startswith("--only=")), None)
BAK = Path(_args[0]) if _args else Path(tempfile.mkdtemp(prefix="perturb-"))
BAK.mkdir(parents=True, exist_ok=True)

TOUCHED: set[Path] = set()


def backup(rel: str) -> Path:
    p = ROOT / rel
    dst = BAK / rel.replace("/", "__")
    if not dst.exists():
        shutil.copy2(p, dst)
    TOUCHED.add(p)
    return p


def restore() -> None:
    for p in list(CREATED):
        p.unlink(missing_ok=True)
    CREATED.clear()
    for p in list(TOUCHED):
        dst = BAK / str(p.relative_to(ROOT)).replace("/", "__")
        shutil.copy2(dst, p)
    TOUCHED.clear()


def sub(rel: str, old: str, new: str, count: int = 1) -> None:
    p = backup(rel)
    t = p.read_text()
    assert t.count(old) == count, f"{rel}: {t.count(old)} of {old[:50]!r}, want {count}"
    p.write_text(t.replace(old, new))


def append(rel: str, text: str) -> None:
    p = backup(rel)
    p.write_text(p.read_text() + text)


def jset(rel: str, path: list, value) -> None:
    p = backup(rel)
    d = json.loads(p.read_text())
    node = d
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    p.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")


def jedit(rel: str, change) -> None:
    """A cache edited by a function, for what `jset` cannot reach: dropping an element
    of a list, or a change computed from what is already in the file."""
    p = backup(rel)
    d = json.loads(p.read_text())
    change(d)
    p.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")


def copy_over(dst_rel: str, src_rel: str) -> None:
    p = backup(dst_rel)
    p.write_bytes((ROOT / src_rel).read_bytes())


CREATED: set[Path] = set()


def write_file(rel: str, text: str) -> None:
    """A file that did not exist before, so restoring means deleting it.

    Unless it does exist, in which case deleting it is how the harness destroys the
    defect before the guard looks. The battery plants `tests/test_zz_probe.py` and
    then runs the module whose first test spawns this file with `--dry-run`; that
    nested run created the same path, added it to its own `CREATED`, and deleted it
    on the way out -- so the audit two hundred lines further down the module scanned
    a tree with the plant already gone and the case reported MISSED against a guard
    that catches it when it is run by hand. An existing file is backed up and put
    back by content, like every other perturbation.
    """
    p = ROOT / rel
    if p.exists():
        backup(rel)
    else:
        CREATED.add(p)
    p.write_text(text)


def run(mods: list[str], extra: list[str] | None = None) -> set[str]:
    cmd = [PY, "-m", "pytest", "-q", "--no-header", "-p", "no:randomly",
           *[f"tests/{m}" for m in mods], *(extra or [])]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout
    fired = {ln.split("::")[1].split(" ")[0]
             for ln in out.splitlines() if ln.startswith("FAILED") and "::" in ln}
    # the build tripwire is expected whenever a .tex changes; it is not a finding
    tripwire = {"test_no_label_is_defined_twice", "test_no_reference_is_undefined",
                "test_no_overfull_boxes", "test_the_heading_levels_do_not_skip",
                "test_no_block_is_typeset_twice", "test_the_two_builds_are_the_same_length",
                "test_the_build_converged",
                # Self-reference, not a finding: this test spawns a nested `--dry-run`
                # that re-applies every case to a tree where one case is ALREADY
                # applied, so that case's own anchor is legitimately gone and it reports
                # STALE. Every `.tex` case has carried it as noise; a control running
                # this module reads it as a false red, which is how it was found.
                "test_every_perturbation_in_the_battery_still_applies",
                # Same class as the two above it, and it was missed when it was written:
                # `_log` refuses a log that does not match the sources, so every guard
                # that reads one fails on any `.tex` perturbation. The control for the
                # sentence-scoped survival check reported FALSE RED against this and
                # nothing else, which is the digest and not a spill.
                "test_neither_twenty_page_version_runs_past_twenty_pages"}
    return {f for f in fired if f.split("[")[0] not in tripwire}


FIG, TAB, BLD, EXP = ("test_paper_figures.py", "test_paper_tables.py",
                      "test_paper_builds.py", "test_experiment_controls.py")

CASES: list[tuple[str, callable, list[str], str]] = [
    ("the stripper reverted to round five's bug",
     lambda: sub("tests/texsource.py",
                 'COMMENT = re.compile(r"(?<!\\\\)((?:\\\\\\\\)*)%.*")',
                 'COMMENT = re.compile(r"(?<!\\\\)()%.*")'),
     [BLD], "test_the_comment_stripper_reads_a_percent_the_way_latex_does"),
    ("the stripper reverted to round four's bug",
     lambda: sub("tests/texsource.py",
                 'COMMENT = re.compile(r"(?<!\\\\)((?:\\\\\\\\)*)%.*")',
                 'COMMENT = re.compile(r"(?<!\\\\\\\\)()%.*")'),
     [BLD], "test_the_comment_stripper_reads_a_percent_the_way_latex_does"),
    ("a pinned sentence parked behind \\ifnum0=1",
     lambda: sub("paper/sections/moved/eval_heldout.tex",
                 "The frozen translator, called exactly as",
                 "\\ifnum0=1 The frozen translator, called exactly as"),
     [FIG], "test_a_pinned_sentence_still_says_what_the_registry_says"),
    ("the rounding note parked behind \\iffalse",
     lambda: sub("paper/sections/04_grammars.tex",
                 "taken from the counts and not from each other",
                 "\\iffalse taken from the counts and not from each other \\fi "
                 "each read off the other"),
     [TAB], "test_every_printed_trio_either_reconciles_or_is_accounted_for"),
    ("a guard reading source() with no ledger entry",
     lambda: append("tests/test_paper_xrefs.py",
                    '\n\ndef test_probe_structural():\n'
                    '    assert source(PAPER / "body.tex")\n'),
     [BLD], "test_no_guard_reads_the_source_where_it_means_a_document"),
    ("a SOURCE_READS entry for a guard that does not read it",
     lambda: sub("tests/texsource.py",
                 '    "_scan": "labels and the references to them, which no build renames",',
                 '    "_scan": "labels and the references to them, which no build renames",\n'
                 '    "_probe_gone": "a guard that does not exist",'),
     [BLD], "test_the_source_ledger_names_exactly_the_guards_that_read_the_source"),
    ("a new guard reading a .tex raw",
     lambda: append("tests/test_paper_xrefs.py",
                    '\n\ndef test_probe_raw():\n'
                    '    body = (PAPER / "body.tex").read_text()\n'
                    '    assert body\n'),
     [BLD], "test_every_guard_reads_the_document_the_paper_publishes"),
    ("the in-sample margin's standard error set to zero",
     lambda: jset("data/processed/heldout_eval.json",
                  ["gain_dd_in_sample", "gain_se"], 0.0),
     [TAB], "test_every_margin_spread_is_a_function_of_its_stored_replicates"),
    ("one held-out margin replicate moved",
     lambda: jset("data/processed/heldout_eval.json",
                  ["gain_ex_held", "replicates", 0], 4.9),
     [TAB], "test_every_margin_spread_is_a_function_of_its_stored_replicates"),
    ("the bootstrap skew edited",
     lambda: jset("data/processed/heldout_eval.json",
                  ["kappa_extras", "jackknife", "bootstrap_skew"], -0.4),
     [TAB], "test_every_cluster_statistic_is_a_function_of_the_stored_tally"),
    ("the LALR conflict count changed to four",
     lambda: (sub("paper/tables/grammar_stats.tex",
                  "2$\\times$ \\texttt{Shift/Reduce on LPAR}",
                  "4$\\times$ \\texttt{Shift/Reduce on LPAR}", count=2),
              sub("paper/sections/moved/gr_conflict_cause.tex",
                  "carries $2$ conflicts", "carries $4$ conflicts")),
     [TAB], "test_the_grammar_table_is_the_table_the_parsers_build"),
    ("the model agreement moved twelve orders of magnitude",
     lambda: sub("paper/tables/model_verification.tex", "8.60e-14", "8.60e-02"),
     [TAB], "test_the_model_verification_table_says_floating_point_and_means_it"),
    ("the .rollup frequency changed by a hundred",
     lambda: sub("paper/tables/operator_frequency.tex",
                 "\\texttt{.rollup} & 331", "\\texttt{.rollup} & 431"),
     [TAB], "test_the_operator_table_is_a_recount_of_the_corpus"),
    ("the no-glob distinct counts raised by ten",
     lambda: sub("paper/tables/regex_glob_reasons.tex",
                 "admits & 4 & 74", "admits & 14 & 74"),
     [TAB], "test_the_no_glob_table_columns_are_the_registry_totals"),
    # ---- round seven: the seeded paths, the order they produce, and the ledgers ----
    ("greedy's tie-break back on the interpreter's hash salt",
     lambda: sub("scripts/heldout_split.py",
                 "pick = min(sorted(source.items()), key=lambda kv: (-kv[1], kv[0]))[0]",
                 "pick = source.most_common(1)[0][0]"),
     [TAB], "test_every_seeded_path_is_deterministic_across_processes"),
    ("the differential's seed back on hash() of the key",
     lambda: sub("scripts/shape_differential.py",
                 'hashlib.sha256(repr(key).encode()).digest()[:4], "big") % 2**31',
                 'b"\\0\\0\\0\\0", "big") + hash(key) % 2**31'),
     [TAB], "test_every_seeded_path_is_deterministic_across_processes"),
    ("the registry's ceilings written by the other tie-break",
     lambda: jset("paper/quoted_figures.json", ["nb00.ceiling_6_features_pct"], 1.6),
     [TAB], "test_the_one_tie_in_the_whole_corpus_order_is_broken_the_defensible_way"),
    ("cv.reps back to the per-seed count the caption used to quote",
     lambda: jset("paper/quoted_figures.json", ["cv.reps"], 200),
     [TAB], "test_the_vocabulary_ceiling_saturates_well_inside_the_corpus"),
    ("the staircase gap at a resolution the run does not have",
     lambda: jset("paper/quoted_figures.json", ["cv.dd_stair_gap_16_points"], 1.1),
     [TAB], "test_the_staircase_survives_being_scored_on_dashboards_it_was_not_built_from"),
    ("the staircase's step moved under the sentence describing it",
     lambda: sub("paper/tables/implementation_order.tex",
                 "5 & \\texttt{AGG\\_AVG} & space aggregator avg: & 3,889 & 1 & 1.6\\% \\\\",
                 "5 & \\texttt{AGG\\_AVG} & space aggregator avg: & 3,889 & 901 & 1.6\\% \\\\"),
     [TAB], "test_the_staircase_figure_the_ledger_cannot_see"),
    ("a percentage printed outside math mode",
     lambda: sub("paper/sections/moved/eval_staircase.tex",
                 "The shape is the finding.",
                 "The shape is the finding, and the step is worth 16.5\\% of it."),
     [FIG], "test_every_percentage_and_thousand_in_the_prose_is_in_math_mode"),
    ("a sixth build-conditional macro rendered expands nothing for",
     lambda: append("paper/macros.tex",
                    "\n\\newcommand{\\apwhere}[1]{\\iffull here\\else "
                    "Appendix~\\ref{#1}\\fi}\n"),
     [BLD], "test_the_build_variant_macros_are_the_ones_rendered_expands"),
    ("a provenance date moved thirteen months",
     lambda: sub("paper/tables/vendor_provenance.tex",
                 "E1 & 16 Nov 2026", "E1 & 16 Oct 2025"),
     [TAB], "test_the_provenance_table_dates_are_the_dates_its_own_caches_carry"),
    ("the observed rollup interval at 1h set to sixty",
     lambda: sub("paper/sections/moved/ing_rollup_spans.tex",
                 "Observed (s) & $10$ & $10$ & $10$ & $10$ & $10$ & $20$",
                 "Observed (s) & $10$ & $10$ & $10$ & $10$ & $10$ & $60$"),
     [EXP], "test_the_span_table_written_by_hand_is_the_table_e18_recorded"),
    ("the extrapolation control reading falsified",
     lambda: (jset("paper/quoted_figures.json", ["na.no_anchor_plain"], 51.7),
              sub("paper/sections/moved/ing_anchored_pf.tex",
                  "returns $31.7$", "returns $51.7$")),
     [EXP], "test_every_anchored_figure_is_the_one_the_cache_produced"),
    ("the staircase thresholds changed to four and three",
     lambda: sub("paper/sections/moved/eval_staircase.tex",
                 "Half the corpus\nneeds eight features; $90\\%$ needs sixteen of the "
                 "twenty-eight",
                 "Half the corpus\nneeds three features; $90\\%$ needs four of the "
                 "twenty-eight"),
     [FIG], "test_the_staircase_thresholds_written_as_words_are_the_registry_curve"),
    ("the column asserted empty without its qualification",
     lambda: sub("paper/sections/08_discussion.tex",
                 "The column\nholds nothing demonstrated now, which is a stronger claim "
                 "than any of the figures it\nused to hold and a weaker one than "
                 "``empty''",
                 "The column\nis empty now, which is a stronger claim than any of the "
                 "figures it\nused to hold"),
     [FIG], "test_the_inexpressible_column_carries_its_qualification_everywhere"),
    ("a dead private helper added",
     lambda: append("tests/test_paper_xrefs.py",
                    '\n\ndef _probe_never_called():\n    return 1\n'),
     [BLD], "test_no_module_defines_a_private_helper_nothing_calls"),
    ("the renamed staircase key left in the registry at its old value",
     lambda: jset("paper/quoted_figures.json", ["cv.dd_stair_gap_16_pct"], 1.2),
     [TAB], "test_every_cross_validation_figure_is_the_one_the_run_produced"),
    ("a cross-validation figure hand-edited away from its run",
     lambda: jset("paper/quoted_figures.json", ["cv.dd_tenth_coverage_pct"], 99.1),
     [TAB], "test_every_cross_validation_figure_is_the_one_the_run_produced"),
    ("a generated table made to render differently per build",
     lambda: sub("paper/tables/grammar_stats.tex", "datadog & v1 & 19",
                 "datadog & v1 & \\iffull 19\\else 20\\fi"),
     [TAB], "test_the_grammar_table_is_the_table_the_parsers_build"),
    # ---- round eight ---------------------------------------------------------------
    ("a pinned sentence parked behind \\unless\\iftrue",
     lambda: sub("paper/body.tex",
                 "and run on a disjoint repository of dashboards by other authors it",
                 "\\unless\\iftrue by other authors \\fi and run on a disjoint repository "
                 "of dashboards it"),
     # [FIG], not [BLD]. `rendered` raises, so what fails is every guard that reads
     # body.tex through it -- and none of those is in test_paper_builds.py, whose own
     # failures here are the digest tripwire this harness filters out. The case
     # reported MISSED for a hole that was closed, which is the harness lying about
     # the suite rather than the suite lying about the paper.
     [FIG], "test_a_pinned_sentence_still_says_what_the_registry_says"),
    ("both halves of the seeded fix reverted",
     lambda: (sub("scripts/heldout_split.py",
                  "pick = min(sorted(source.items()), key=lambda kv: (-kv[1], kv[0]))[0]",
                  "pick = source.most_common(1)[0][0]"),
              sub("scripts/heldout_split.py",
                  "            train = [f for n in sorted(train_names) for f in by_dash[n]]\n"
                  "            test = [f for n in names if n not in train_names for f in by_dash[n]]\n"
                  "            if not test:",
                  "            train = [f for n in train_names for f in by_dash[n]]\n"
                  "            test = [f for n in names if n not in train_names for f in by_dash[n]]\n"
                  "            if not test:")),
     [TAB], "test_every_seeded_path_is_deterministic_across_processes"),
    ("the baseline's independent confirmation doubled",
     lambda: jset("paper/quoted_figures.json", ["oo.invented_window"], 3844),
     [TAB], "test_every_baseline_figure_is_the_one_the_run_produced"),
    ("the type-inference error rate cut by a fifth",
     lambda: (jset("paper/quoted_figures.json", ["nb03.type_wrong_occurrences"], 391),
              jset("paper/quoted_figures.json", ["nb03.type_wrong_pct"], 5.2),
              jset("paper/quoted_figures.json", ["nb03.type_wrong_gauge_occurrences"], 192),
              jset("paper/quoted_figures.json", ["nb03.type_wrong_counter_occurrences"], 199)),
     [TAB], "test_every_type_inference_figure_is_the_one_the_join_produced"),
    ("the abstract's settle rate moved ten points",
     lambda: jset("paper/quoted_figures.json",
                  ["rs.span_declared_pct_of_rollup_default"], 16.0),
     [TAB], "test_every_rollup_span_figure_is_the_one_the_join_produced"),
    ("the PromQL template share cut to 57.8%",
     lambda: (jset("paper/quoted_figures.json", ["nb00.promql_with_variable"], 4684),
              jset("paper/quoted_figures.json", ["nb00.promql_with_variable_pct"], 57.8)),
     [FIG], "test_the_template_variable_counts_are_the_ones_the_corpus_carries"),
    ("a grammar decision's corpus cost doubled",
     lambda: sub("paper/tables/grammar_decisions.tex",
                 "UnexpectedToken & 474", "UnexpectedToken & 974"),
     [TAB], "test_the_grammar_decision_table_separates_the_witness_from_the_corpus"),
    ("a worked glob example's occurrences raised by a hundred",
     lambda: sub("paper/tables/regex_glob_examples.tex", " 133 \\\\", " 233 \\\\"),
     [TAB], "test_the_glob_example_table_is_a_recount_of_the_matcher_cache"),
    ("one plot copied over another",
     lambda: copy_over("paper/figures/heldout_vocabulary.pdf",
                       "paper/figures/cumulative_coverage.pdf"),
     [BLD], "test_each_plot_is_the_plot_its_caption_says_it_is"),
    ("a proof dropped from the short build only",
     lambda: sub("paper/sections/moved/reg_dichotomy_pf.tex",
                 "\\begin{proof}[Proof of Lemma~\\ref{lem:dichotomy}]",
                 "\\inlineonly{\\begin{proof}[Proof of Lemma~\\ref{lem:dichotomy}]"),
     ["test_paper_xrefs.py"], "test_the_proof_convention_matches_the_statements"),
    ("the column asserted empty with the subject first",
     lambda: sub("paper/sections/08_discussion.tex",
                 "This is a switching cost of its own",
                 "Nothing remains in the inexpressible column. This is a switching cost "
                 "of its own"),
     [FIG], "test_the_inexpressible_column_carries_its_qualification_everywhere"),
    ("a guard reading the paper through a computed attribute name",
     lambda: write_file("tests/test_zz_probe.py",
                        "from pathlib import Path\n\n"
                        "ROOT = Path(__file__).resolve().parent.parent\n\n\n"
                        "def test_probe():\n"
                        "    read = getattr(ROOT / 'paper' / 'body.tex', 'read' + '_text')\n"
                        "    assert read()\n"),
     [BLD], "test_every_guard_reads_the_document_the_paper_publishes"),
    # The word that named three propositions. Each paper is checked separately, so a
    # case per paper: the long one, the switching one, and the short one, which argues
    # no conjecture and so must not use either word.
    ("the thesis back in the discussion",
     lambda: sub("paper/sections/08_discussion.tex",
                 "That is not the result the conjecture predicted",
                 "That is not the result the thesis predicted"),
     [BLD], "test_no_claim_in_any_paper_is_called_a_thesis"),
    ("the thesis back in the switching paper",
     lambda: sub("paper/switching/05_coverage.tex",
                 "would have made the conjecture look better",
                 "would have made the thesis look better"),
     [BLD], "test_no_claim_in_any_paper_is_called_a_thesis"),
    ("the short paper given a thesis",
     lambda: sub("paper/short/04_forward.tex",
                 "which is what makes\n\\S\\ref{s:sec:regular} the interesting half.",
                 "which is what makes\n\\S\\ref{s:sec:regular} the interesting half and "
                 "the thesis of this version."),
     [BLD], "test_no_claim_in_any_paper_is_called_a_thesis"),
    ("the conjecture named a second time",
     lambda: sub("paper/sections/08_discussion.tex",
                 "in observability is a language-design\nartefact.",
                 "in observability is the \\emph{language-design conjecture}."),
     [BLD], "test_the_conjecture_is_named_once_and_in_the_introduction"),
    ("the naming moved out of the introduction",
     lambda: (sub("paper/sections/01_introduction.tex",
                  "the \\emph{language-design conjecture}",
                  "the language-design conjecture"),
              sub("paper/sections/02_background.tex",
                  "Four traditions bear on the question of",
                  "The \\emph{language-design conjecture} is what this literature "
                  "leaves untested. Four traditions bear on the question of")),
     [BLD], "test_the_conjecture_is_named_once_and_in_the_introduction"),
    ("the conjecture used and never named",
     lambda: sub("paper/sections/01_introduction.tex",
                 "the \\emph{language-design conjecture}",
                 "the language-design conjecture"),
     [BLD], "test_the_conjecture_is_named_once_and_in_the_introduction"),
    ("the short paper arguing a conjecture",
     lambda: sub("paper/short/04_forward.tex",
                 "it found no obstruction.",
                 "it found no obstruction, which is the conjecture failing."),
     [BLD], "test_the_conjecture_is_named_once_and_in_the_introduction"),
    ("the introduction giving one half of the answer",
     lambda: sub("paper/sections/01_introduction.tex",
                 "its answer is directional --- it\nfails in the direction teams travel "
                 "and holds, as theorems rather than as measurements,\nin the direction "
                 "they do not.",
                 "its answer is negative."),
     [BLD], "test_the_conjecture_gets_both_halves_of_its_answer"),
    ("the conjecture surviving again",
     lambda: sub("paper/sections/08_discussion.tex",
                 "\\subsection{The conjecture holds in the other direction}",
                 "\\subsection{The conjecture survives in the other direction}"),
     [BLD], "test_the_conjecture_gets_both_halves_of_its_answer"),
    ("the negative answer softened out of its heading",
     lambda: sub("paper/sections/08_discussion.tex",
                 "\\subsection{The question as posed has a negative answer}",
                 "\\subsection{What the question as posed answers}"),
     [BLD], "test_the_conjecture_gets_both_halves_of_its_answer"),
    ("the baseline arriving at the paper's own claim again",
     lambda: sub("paper/sections/07_evaluation.tex",
                 "Two of the figures are \\S\\ref{sec:dashboard-layer}'s finding "
                 "arriving from outside\nthis paper.",
                 "Two of the figures are this paper's own claim arriving from outside "
                 "it."),
     [BLD], "test_the_baseline_figures_arrive_at_the_finding_and_not_at_the_conjecture"),
    ("the conjecture argued in the evaluation",
     lambda: sub("paper/sections/07_evaluation.tex",
                 "How the comparison was run, and what it is not\nevidence of,",
                 "So the conjecture fails here too. How the comparison was run, and "
                 "what it is not\nevidence of,"),
     [BLD], "test_the_baseline_figures_arrive_at_the_finding_and_not_at_the_conjecture"),
    # The nesting the ceiling closed, put back one sentence at a time. The first is the
    # worst the paper had: an aside that never closed, with a colon and seven codes
    # inside it.
    ("the caveat list back inside an unclosed aside",
     lambda: sub("paper/sections/05_translation.tex",
                 "Seven codes arise over six conditions: the fourth condition has a code"
                 " for\neach of the two forms it chooses between, which is why "
                 "Theorem~\\ref{thm:kernel} has six\nhypotheses and the emitter seven "
                 "codes. They are",
                 "Seven codes arise, over six conditions --- the fourth has a code for "
                 "each of\nthe two forms it chooses between, which is why "
                 "Theorem~\\ref{thm:kernel} has six\nhypotheses and the emitter seven "
                 "codes:"),
     [BLD], "test_no_sentence_holds_more_than_three_qualifications_open"),
    ("the label-name row's four-deep chain restored",
     lambda: sub("paper/sections/moved/eval_rows_fill_glob.tex",
                 "selector cannot do: label names are not queryable there.\n"
                 "Remark~\\ref{rem:label-name} puts",
                 "selector cannot do, because label names are not queryable there ---\n"
                 "Remark~\\ref{rem:label-name} puts"),
     [BLD], "test_no_sentence_holds_more_than_three_qualifications_open"),
    ("the conflict cause back behind an aside",
     lambda: sub("paper/sections/04_grammars.tex",
                 "from our layering: making\nthe label list mandatory removes it "
                 "entirely, which is how we identify the cause. That\nchange also "
                 "rejects",
                 "from our layering --- making\nthe label list mandatory removes it "
                 "entirely, which is how we identify the cause, and\nalso rejects"),
     [BLD], "test_no_sentence_holds_more_than_three_qualifications_open"),
    ("the unverified default back inside its aside",
     lambda: sub("paper/sections/07_evaluation.tex",
                 "appears nowhere in the JSON. For those the default is therefore\n"
                 "\\emph{unverified} rather than wrong, which is what this paragraph "
                 "claimed.",
                 "appears nowhere in the JSON --- so for those the default is\n"
                 "\\emph{unverified}, which is what this paragraph claimed, rather than "
                 "wrong."),
     [BLD], "test_no_sentence_holds_more_than_three_qualifications_open"),
    # Each paper is scored separately, so the switching one needs its own case.
    ("a four-deep sentence added to the switching paper",
     lambda: append("paper/switching/05_coverage.tex",
                    "\nThe ceiling is ours --- a figure about this translator, which we "
                    "wrote, and not\nabout the language, which nobody has measured, "
                    "because no reverse translator exists\n--- so the number carries "
                    "that much and no more.\n"),
     [BLD], "test_no_sentence_holds_more_than_three_qualifications_open"),
    ("one more sentence at the nesting ceiling",
     lambda: append("paper/sections/04_grammars.tex",
                    "\nThe table is ours --- a fact about this grammar, which we "
                    "fitted, because no\npublished grammar exists --- and the number "
                    "carries that much.\n"),
     [BLD], "test_the_sentences_at_the_nesting_ceiling_do_not_multiply"),
    # The instrument. A prose measurement nobody checks reports on its own regex.
    ("the scorer stops closing an item at a semicolon",
     lambda: sub("tests/test_paper_builds.py",
                 '        if s[i] == ";":', '        if s[i] == "\\0":'),
     [BLD], "test_the_nesting_scorer_counts_what_it_says_it_counts"),
    ("the scorer counts a code span's brackets",
     lambda: sub("tests/test_paper_builds.py",
                 '    for macro in ("texttt", "verb", "mathit", "mathrm", "mathsf"):\n'
                 '        t = _braced(t, macro, "CODE")',
                 '    for macro in ("verb", "mathit", "mathrm", "mathsf"):\n'
                 '        t = _braced(t, macro, "CODE")'),
     [BLD], "test_the_nesting_scorer_counts_what_it_says_it_counts"),
    ("the scorer glues a caption to the sentence after it",
     lambda: sub("tests/test_paper_builds.py",
                 '    t = re.sub(r"\\\\(?:begin|end)\\{[A-Za-z*]+\\}'
                 '(?:\\[[^\\]]*\\])?", ". ", t)',
                 '    t = re.sub(r"\\\\(?:begin|end)\\{[A-Za-z*]+\\}'
                 '(?:\\[[^\\]]*\\])?", " ", t)'),
     [BLD], "test_the_nesting_scorer_counts_what_it_says_it_counts"),
    # The three families of notebook figures that nothing read. The two source-side
    # cases here were green against the whole suite before the guards were written,
    # which is the measurement the commit rests on.
    ("a ranking case dropped from the cache",
     lambda: jedit("data/raw/topk/ranking.json", lambda d: d["cases"].pop()),
     [TAB], "test_every_top_k_figure_is_the_one_the_two_probes_recorded"),
    ("a ranking case probed at a size the grid does not have",
     lambda: jset("data/raw/topk/ranking.json", ["cases", 0, "n"], 4),
     [TAB], "test_every_top_k_figure_is_the_one_the_two_probes_recorded"),
    ("one vendor selection recorded as not matching the model",
     lambda: jset("data/raw/datadog/top.json", ["cases", 0, "matches_model"], False),
     [TAB], "test_every_top_k_figure_is_the_one_the_two_probes_recorded"),
    # The summary a script wrote about its own run, against a recount of the run.
    ("the vendor summary no longer a recount of its own cases",
     lambda: jset("data/raw/datadog/top.json", ["agreement", "matching_model"], 21),
     [TAB], "test_every_top_k_figure_is_the_one_the_two_probes_recorded"),
    ("a workload count the registry no longer agrees with",
     lambda: jset("paper/quoted_figures.json", ["nb03.top_workload_series"], 6),
     [TAB], "test_every_top_k_figure_is_the_one_the_two_probes_recorded"),
    ("a workload series shorter than the bucket count",
     lambda: jset("data/raw/datadog/top.json",
                  ["provenance", "series", "a"], [100.0, 100.0]),
     [TAB], "test_every_top_k_figure_is_the_one_the_two_probes_recorded"),
    # The modifier count, and the claim that every form carrying it is on `sum:`.
    ("one weighted form no longer on sum:",
     lambda: jset("paper/quoted_figures.json",
                  ["nb03.weighted_on_sum_distinct"], 51),
     [TAB], "test_the_weighted_figures_count_the_modifier_and_not_the_substring"),
    ("the weighted count back to the substring match",
     lambda: (jset("paper/quoted_figures.json", ["nb03.weighted_distinct"], 56),
              jset("paper/quoted_figures.json",
                   ["nb03.weighted_on_sum_distinct"], 56)),
     [TAB], "test_the_weighted_figures_count_the_modifier_and_not_the_substring"),
    # Occurrences moved between the biggest and the smallest shape. The total is
    # unchanged, so the two totals the guard pins still hold and three of the four
    # milestones move: this is the case the whole suite passed before.
    ("occurrences moved between shapes, leaving the total alone",
     lambda: (jset("data/raw/shapes/differential.json",
                   ["rows", 0, "occurrences"], 303),
              jset("data/raw/shapes/differential.json",
                   ["rows", 332, "occurrences"], 901)),
     [TAB], "test_the_coverage_milestones_are_the_shape_population_they_claim"),
    ("a coverage milestone moved by one shape",
     lambda: jset("paper/quoted_figures.json", ["nb03.shapes_for_90pct"], 49),
     [TAB], "test_the_coverage_milestones_are_the_shape_population_they_claim"),
    # The notebook computes a 99% milestone and writes four of the five. A fifth key
    # with no target behind it is the shape that invites.
    ("a fifth coverage milestone with no target behind it",
     lambda: jset("paper/quoted_figures.json", ["nb03.shapes_for_99pct"], 200),
     [TAB], "test_the_coverage_milestones_are_the_shape_population_they_claim"),
    # The Makefile's claims about the data scripts, which nothing read for rounds.
    ("the comment back to three data scripts",
     lambda: sub("Makefile", "these six write the other five",
                 "these three write the other five"),
     [BLD], "test_the_data_scripts_write_the_registry_the_way_the_makefile_says"),
    # The write this script went ungated for rounds, ungated again.
    ("the join's registry write back outside its flag",
     lambda: sub("scripts/rollup_span_join.py",
                 '    if "--write-registry" in sys.argv:\n'
                 '        reg = json.loads(REGISTRY.read_text())\n'
                 '        reg.update(fig)\n'
                 '        REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True)'
                 ' + "\\n")\n'
                 '        print(f"\\nmerged {len(fig)} keys into '
                 '{REGISTRY.relative_to(ROOT)}")',
                 '    reg = json.loads(REGISTRY.read_text())\n'
                 '    reg.update(fig)\n'
                 '    REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True)'
                 ' + "\\n")'),
     [BLD], "test_the_data_scripts_write_the_registry_the_way_the_makefile_says"),
    # The history the comment keeps: "all five gate it" is worth reading only beside
    # the fact that one of them did not and nothing noticed.
    ("the comment dropping which script was the exception",
     lambda: sub("Makefile",
                 "# Five of them always did. rollup_span_join.py did not -- it had no "
                 "flag and no argument\n# parsing at all, so it merged",
                 "# All of them always did -- no flag and no argument\n"
                 "# parsing at all, so it merged"),
     [BLD], "test_the_data_scripts_write_the_registry_the_way_the_makefile_says"),
    # An `if` that no longer tests the flag: the case a grep for the flag would miss,
    # since the usage line still names it.
    ("a second script writing the registry ungated",
     lambda: sub("scripts/dash0_claims.py",
                 '    if "--write-registry" in sys.argv:\n'
                 '        reg = json.loads(REGISTRY.read_text())',
                 '    if True:\n'
                 '        reg = json.loads(REGISTRY.read_text())'),
     [BLD], "test_the_data_scripts_write_the_registry_the_way_the_makefile_says"),
    ("the data recipe passing --write-registry",
     lambda: sub("Makefile",
                 '\t  $(PY) "$$s" >/dev/null || { echo "$$s FAILED"; exit 1; }; \\',
                 '\t  $(PY) "$$s" --write-registry >/dev/null || '
                 '{ echo "$$s FAILED"; exit 1; }; \\'),
     [BLD], "test_the_data_scripts_write_the_registry_the_way_the_makefile_says"),
    ("the key count the comment gives for the join",
     lambda: sub("Makefile", "merged its 13 `rs.` keys", "merged its 11 `rs.` keys"),
     [BLD], "test_the_data_scripts_write_the_registry_the_way_the_makefile_says"),
    ("a seventh script added to DATA_SCRIPTS",
     lambda: sub("Makefile",
                 "                scripts/heldout_split.py scripts/heldout_eval.py",
                 "                scripts/heldout_split.py scripts/heldout_eval.py \\\n"
                 "                scripts/perturb.py"),
     [BLD], "test_the_data_scripts_write_the_registry_the_way_the_makefile_says"),
    # The list of derived inputs, which named four of the eight paths a clean clone
    # lacks while nothing compared it to a read.
    ("a derived input the suite reads dropped from the list",
     lambda: sub("tests/test_paper_builds.py",
                 '    (ROOT / "data" / "processed" / "promtool.json", _MAKE),\n', ""),
     [BLD], "test_the_derived_input_list_is_what_the_suite_reads"),
    ("a committed file back in the list of derived inputs",
     lambda: sub("tests/test_paper_builds.py",
                 '    (ROOT / "data" / "corpus" / "sources", _FETCH),',
                 '    (ROOT / "data" / "corpus" / "sources", _FETCH),\n'
                 '    (ROOT / "data" / "corpus" / "queries.jsonl", _FETCH),'),
     [BLD], "test_the_derived_input_list_is_what_the_suite_reads"),
    # A scan that stops seeing the suite's reads passes the equality by finding
    # nothing on either side, which is the floor's whole job.
    ("the scan for data paths blinded",
     lambda: sub("tests/test_paper_builds.py",
                 '    parts: list[str] = []\n'
                 '    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):',
                 '    parts: list[str] = []\n'
                 '    if True:\n        return None\n'
                 '    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):'),
     [BLD], "test_the_derived_input_list_is_what_the_suite_reads"),
    # The state this round found: the file listed, read, and produced by nothing the
    # printed command runs.
    ("the script behind a listed input dropped from the recipe",
     lambda: sub("Makefile", "scripts/openobserve_baseline.py ", ""),
     [BLD], "test_the_derived_input_list_is_what_the_suite_reads"),
    # A derived path written as one string rather than joined a segment at a time.
    # The suite spells it both ways, so a scan that reads only the joins would miss
    # this one; nothing in the tree exercised that branch until this case did.
    ("a derived path read as a single string",
     lambda: append("tests/test_paper_tables.py",
                    '\n\ndef test_probe_reads_a_string_path():\n'
                    '    assert (ROOT / "data/processed/probe_string.csv").exists()\n'),
     [BLD], "test_the_derived_input_list_is_what_the_suite_reads"),
    # A filename reaching `_derived` through one more hop than the scan was written
    # for. If the fixpoint stops at the first hop this file is invisible and the
    # equality holds, which is the silent narrowing the scan exists to refuse.
    ("a cache read through a second-hand helper",
     lambda: append("tests/test_paper_tables.py",
                    '\n\ndef _probe_cache(name: str) -> dict:\n'
                    '    return _derived(name)["figures"]\n\n\n'
                    'def test_probe_reads_a_cache():\n'
                    '    assert _probe_cache("probe_cache.json")\n'),
     [BLD], "test_the_derived_input_list_is_what_the_suite_reads"),

    # --- the eleven caches whose numbers nothing read -------------------------------
    # Every case below changes ONE number, which is the drift these guards exist for.
    # The battery's usual uniform edit would not do here: it adds the same amount to
    # both sides of an equality and so passes the recounts, which is exactly how E8's
    # hole survived the measurement that found the other ten.
    ("a glob's selection no longer its own response's",
     lambda: jset("data/raw/datadog/e5.json",
                  ["selected", "probe:web-,run:r1794849840,probe:web-*"], 2.0),
     [EXP], "test_every_glob_probe_recounts_its_selection_from_its_own_response"),
    ("a stored point the control never returned",
     lambda: jset("data/raw/datadog/e12.json",
                  ["stored", "probe:web-a,run:r1795036080"], 2.0),
     [EXP], "test_every_glob_probe_recounts_its_selection_from_its_own_response"),
    ("a submitted value the vendor never stored",
     lambda: jset("data/raw/datadog/e13.json",
                  ["submitted", 0, "series", 0, "points", 0, "value"], 2.0),
     [EXP], "test_every_submitted_point_is_one_the_control_read_back"),
    # The scope the points came back under is the vendor's construction, so the chain
    # is held on the values and the instants. Moving the first submitted point off the
    # grid is the edit that would make a cache belong to a different run.
    ("a submitted point off the run's own instant",
     lambda: jset("data/raw/datadog/e16.json",
                  ["submitted", 0, "series", 0, "points", 0, "timestamp"], 1795038240),
     [EXP], "test_every_submitted_point_is_one_the_control_read_back"),
    ("a response window that excludes its own points",
     lambda: jset("data/raw/datadog/e5.json",
                  ["responses", "control", "raw", "from_date"], 1794849900000),
     [EXP], "test_every_cached_response_agrees_with_its_own_bookkeeping"),
    ("a series reporting more points than it carries",
     lambda: jset("data/raw/datadog/e16.json",
                  ["responses", "control", "raw", "series", 0, "length"], 2),
     [EXP], "test_every_cached_response_agrees_with_its_own_bookkeeping"),
    # E14's verdict is the one thing in these caches that points cannot establish, and
    # the cache's copy of it was a free-standing boolean.
    ("the term survival flipped against its own echo",
     lambda: jset("data/raw/datadog/e14.json",
                  ["term_survived_the_parse", "negated-alone"], True),
     [EXP], "test_the_negated_match_all_is_read_off_the_echo_and_not_off_the_points"),
    ("a rollup value the response does not hold",
     lambda: jset("data/raw/datadog/e15.json", ["values", "rate-sum", 1], 35.0),
     [EXP], "test_the_rate_and_count_rollups_are_the_submitted_points_reduced"),
    # The number a reader would take for a rescaling, printed in the paper's own table.
    # Four of the five points fall in the fuller rollup bucket; 35 is what the cache's
    # stale note still says to expect, so this is the drift with a plausible story.
    ("the rate table printing five times the submitted value",
     lambda: sub("paper/sections/moved/reg_trial_probes.tex",
                 "& $28.0$ & $28.0$", "& $35.0$ & $35.0$"),
     [EXP], "test_the_rate_and_count_rollups_are_the_submitted_points_reduced"),
    # The note as it stood for four rounds: the arithmetic for a window that aligns
    # with the rollup, over a run whose window does not.
    ("the rollup note back to five times the submitted value",
     lambda: jset("data/raw/datadog/e15.json", ["interpretation", "note"],
                  "compare rate-avg against the submitted value and rate-sum against "
                  "5 times it; the count series is the control"),
     [EXP], "test_the_rate_and_count_rollups_are_the_submitted_points_reduced"),
    ("a reinterpretation that does not say what was reinterpreted",
     lambda: jset("data/raw/datadog/e15.json", ["provenance", "reinterpreted"],
                  "note recomputed from the cached responses"),
     [EXP], "test_every_reinterpreted_cache_says_what_was_reinterpreted"),
    ("a reinterpreted cache dropped from the ledger",
     lambda: sub("tests/test_experiment_controls.py",
                 '    "e15.json": "the rollup boundary",\n', ''),
     [EXP], "test_every_reinterpreted_cache_says_what_was_reinterpreted"),
    ("one declared series of the binop workload changed",
     lambda: jset("data/raw/datadog/e8.json",
                  ["provenance", "series", "b:g=x,h=p"], 4.0),
     [EXP], "test_the_binop_pairings_are_the_arithmetic_of_the_four_declared_series"),
    # Both sides of one pairing moved together. The test that compares the two
    # languages passes -- they still agree -- and only the arithmetic notices.
    ("both sides of a pairing moved together",
     lambda: jedit("data/raw/datadog/e8.json", lambda d: (
         d["cases"]["both_grouped_div"]["datadog"].__setitem__("g=x", [6.0]),
         d["cases"]["both_grouped_div"]["promql"].__setitem__("g=x", 6.0))),
     [EXP], "test_the_binop_pairings_are_the_arithmetic_of_the_four_declared_series"),
    ("a probe's verdict no longer its own selection",
     lambda: jset("data/raw/datadog/scope_empty_tag.json",
                  ["probes", "k:*", "matches_prediction"], False),
     [EXP], "test_the_scope_probes_replay_the_prediction_table_they_were_run_against"),
    ("a point count the grid does not have",
     lambda: jset("data/raw/datadog/scope_empty_tag.json",
                  ["provenance", "n_points"], 11),
     [EXP], "test_the_scope_probes_replay_the_prediction_table_they_were_run_against"),
    ("a gap-row summary that outlives its own response",
     lambda: jset("data/raw/gap_rows/e10_fill.json",
                  ["candidates", "bare selector", "summary",
                   '{"__name__": "qlport_fill", "svc": "b"}',
                   "points_past_lookback"], 5),
     [EXP], "test_the_gap_row_summaries_replay_the_scripts_own_condensation"),
    # Where the gap is, is a fact about the exposition the server was loaded with, and
    # that was a string in the cache that nothing compared with the script.
    ("the exposition the gap run was loaded with changed",
     lambda: jedit("data/raw/gap_rows/e10_fill.json", lambda d: d["meta"].__setitem__(
         "exposition", d["meta"]["exposition"].replace(
             'qlport_fill{svc="b"} 130 1700001800.000',
             'qlport_fill{svc="b"} 130 1700001900.000'))),
     [EXP], "test_the_gap_row_summaries_replay_the_scripts_own_condensation"),
    ("a label-name attempt recorded as returning series",
     lambda: jset("data/raw/gap_rows/e11_label_name.json",
                  ["attempts", "bare pattern in name position", "series_returned"], 1),
     [EXP], "test_the_gap_row_summaries_replay_the_scripts_own_condensation"),
    ("a matcher selection with a series its response did not return",
     lambda: jedit("data/raw/matchers/label_matchers.json",
                   lambda d: d["matchers"]['k=~".+"']["selected"].append(
                       {"other": "none"})),
     [EXP], "test_the_matcher_readings_replay_the_scripts_own_selection"),
    # The matcher cache's only numbers are the points, and the reading is about which
    # series come back rather than what they hold -- so the points were free until the
    # exposition was replayed against them.
    ("a matcher response point the exposition never wrote",
     lambda: jset("data/raw/matchers/label_matchers.json",
                  ["matchers", 'k=~".*"', "response", "data", "result", 0,
                   "values", 0, 1], "9"),
     [EXP], "test_the_matcher_readings_replay_the_scripts_own_selection"),
    ("a corpus regex dropped from the ones RE2 was asked about",
     lambda: jedit("data/raw/matchers/label_matchers.json",
                   lambda d: d["corpus_regexes"].pop(".*_bytes")),
     [EXP], "test_the_matcher_readings_replay_the_scripts_own_selection"),
    # A cache dropped out of both lists, which is how a replay would quietly stop
    # covering one without the ceiling moving.
    ("a replayed cache dropped from both lists",
     lambda: sub("tests/test_experiment_controls.py",
                 '    "test_the_rate_and_count_rollups_are_the_submitted_points_'
                 'reduced": (\n        "datadog/e15.json",),\n',
                 ''),
     [EXP], "test_every_experiment_cache_says_which_test_replays_it"),
    ("a cache naming a replay that does not exist",
     lambda: sub("tests/test_experiment_controls.py",
                 '    "test_the_matcher_readings_replay_the_scripts_own_selection": (',
                 '    "test_the_matcher_readings_replay_them": ('),
     [EXP], "test_every_experiment_cache_says_which_test_replays_it"),
]

CONTROLS: list[tuple[str, callable, list[str]]] = [
    ("the derived inputs listed in another order",
     lambda: sub("tests/test_paper_builds.py",
                 '    (ROOT / "data" / "processed" / "translations.csv", _MAKE),\n'
                 '    (ROOT / "data" / "processed" / "regex_glob.csv", _MAKE),',
                 '    (ROOT / "data" / "processed" / "regex_glob.csv", _MAKE),\n'
                 '    (ROOT / "data" / "processed" / "translations.csv", _MAKE),'),
     [BLD]),
    # A diagnostic field of the differential cache that no figure is read off. The
    # milestone guard reads one field of each row and has to leave the rest alone.
    ("a per-shape deviation changed in the differential cache",
     lambda: jset("data/raw/shapes/differential.json",
                  ["rows", 0, "median_d"], 0.5),
     [TAB]),
    # Three numbers in the experiment caches that stay unheld on purpose, because
    # nothing derives them and a guard asserting them would only be asserting that the
    # file is the file. The rollup width is the vendor's own choice for the window it
    # was asked over, and `resp_version` is its wire format.
    ("a vendor-chosen rollup interval in a response",
     lambda: jset("data/raw/datadog/e5.json",
                  ["responses", "control", "raw", "series", 0, "interval"], 12),
     [EXP]),
    ("a response's protocol version field",
     lambda: jset("data/raw/datadog/e13.json",
                  ["responses", "control", "raw", "resp_version"], 3),
     [EXP]),
    # `echo_differs` is the script's flag for a reader and explicitly not a verdict --
    # the echo is not verbatim even when nothing was dropped. E14's guard has to read
    # the echo itself, so flipping the flag must fire nothing.
    ("the reader's flag on whether an echo differs",
     lambda: jset("data/raw/datadog/e14.json",
                  ["responses", "negated-alone", "echo_differs"], False),
     [EXP]),
    ("a sentence reworded where no phrase pins it",
     lambda: sub("paper/sections/04_grammars.tex",
                 "Inventing a production for an undocumented construct would be worse",
                 "Inventing a production for an undocumented construct would be far worse"),
     [FIG, TAB]),
    ("a licensed figure added to a moved block",
     lambda: sub("paper/sections/moved/eval_staircase.tex",
                 "The shape is the finding.",
                 "The shape is the finding, and $99.9\\%$ is where it ends."),
     [FIG, TAB]),
    # The false positive the sentence-scoped check was written for. A window of 120
    # characters failed on the paragraph that ends six characters before \S8.4's
    # heading, so a neighbouring sentence saying that something does not survive must
    # not read as the conjecture surviving.
    ("a neighbouring sentence saying something does not survive",
     lambda: sub("paper/sections/08_discussion.tex",
                 "about a third tool, which may place the same three things somewhere "
                 "else.",
                 "about a third tool, which may place the same three things somewhere "
                 "else. Nothing above survives that qualification unchanged."),
     [BLD]),
    # A qualification added where the sentence has room: one aside and one clause is
    # depth two, which is 405 of the paper's sentences and not a defect.
    ("a qualification added to a sentence with room for it",
     lambda: sub("paper/sections/04_grammars.tex",
                 "The resolution is to shift, reading the parenthesis as the label "
                 "list, and it is the\nresolution Prometheus~3.15.0 itself makes",
                 "The resolution is to shift --- reading the parenthesis as the label "
                 "list, which is\nwhat the generator does --- and it is the\n"
                 "resolution Prometheus~3.15.0 itself makes"),
     [BLD]),
    # A Makefile comment reworded where it makes no claim about code.
    ("an unrelated Makefile comment reworded",
     lambda: sub("Makefile",
                 "# Two builds from one source:",
                 "# Two builds out of one source:"),
     [BLD]),
]

if DRY_RUN:
    stale = []
    for name, apply, *_ in CASES + CONTROLS:
        restore()
        try:
            apply()
        except Exception as exc:                                  # noqa: BLE001
            stale.append((name, f"{type(exc).__name__}: {exc}"[:160]))
    restore()
    print(f"{len(CASES)} cases, {len(CONTROLS)} controls, {len(stale)} stale")
    for name, why in stale:
        print(f"  STALE  {name}\n         {why}")
    raise SystemExit(1 if stale else 0)

print(f"{'perturbation':58s} {'verdict':8s} fired")
print("-" * 100)
bad = 0
for name, apply, mods, expect in CASES:
    if ONLY and not any(o in name for o in ONLY):
        continue
    restore()
    apply()
    fired = run(mods)
    ok = expect in {f.split("[")[0] for f in fired}
    bad += not ok
    others = sorted(f for f in fired if f.split("[")[0] != expect)
    extra = f"  (+{len(others)} more)" if others else ""
    print(f"{name:58s} {'CAUGHT' if ok else 'MISSED':8s} "
          f"{expect if ok else sorted(fired) or 'nothing'}{extra}")
restore()
print()
for name, apply, mods in CONTROLS:
    if ONLY and not any(o in name for o in ONLY):
        continue
    restore()
    apply()
    fired = run(mods)
    bad += bool(fired)
    print(f"{name:58s} {'PASSES' if not fired else 'FALSE RED':8s} "
          f"{sorted(fired) if fired else ''}")
restore()
print()
print("MISSED/FALSE-RED:", bad)
