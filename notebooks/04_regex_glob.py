# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 — Regex <-> glob (paper §6)
#
# The reverse direction's sub-problem. Datadog's tag filter is a **glob**: literal text
# separated by `*`, with prefix, suffix and substring wildcards documented and nothing
# else — in particular no single-character wildcard. PromQL's label matcher is a **regular
# expression**, anchored over the whole value. So:
#
# * **Glob -> regex** is exact by construction, and is what the main direction uses.
# * **Regex -> glob** is a question about two classes of regular languages, and the answer
#   is usually no. `regular.regex_to_globs` proposes a glob (or a finite union of globs,
#   which a Datadog scope can express as `k IN (...)`) from the regex's structure, then
#   *decides* equivalence of the two languages before returning it. Nothing is returned
#   unverified.
#
# Writes `paper/tables/regex_glob.tex`, `regex_glob_reasons.tex`,
# `regex_glob_examples.tex` and `data/processed/regex_glob.csv`.

# %%
import collections
import re
from pathlib import Path

import pandas as pd

from qlport import corpus as c
from qlport.regular import regex_to_globs

ROOT = Path.cwd() if (Path.cwd() / "paper").exists() else Path.cwd().parent
CORPUS = ROOT / "data" / "corpus"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "paper" / "tables"
for p in (TABLES, PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

EXAMPLE_CHARS = 34

# %% [markdown]
# ## LaTeX helpers
# Same as the other notebooks: bare `tabular`, wrapped in a float by the section file.

# %%
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}"}


def tex(s: object) -> str:
    return "".join(_TEX.get(ch, ch) for ch in str(s))


def code(s: object, width: int | None = None) -> str:
    s = " ".join(str(s).split())
    if width and len(s) > width:
        s = s[: width - 1] + "…"
    return r"\texttt{" + tex(s).replace("…", r"\dots{}") + "}"


def pct(x: float) -> str:
    return f"{x:.1%}".replace("%", r"\%")


def tabular(spec: str, header: list[str], rows: list[list[str]], path: Path) -> None:
    lines = [rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


# %% [markdown]
# ## The regexes the corpus actually contains
#
# Every `=~` and `!~` matcher in the PromQL half of the corpus, counted by occurrence.
# The split that matters first is not about regular languages at all: most of these are
# Grafana **template variables**, whose value is chosen when someone looks at the
# dashboard. `corpus.normalise` deliberately leaves a variable inside a string literal
# alone, because `{pod=~"$pod"}` already parses — so these are not regexes we know.
# Nothing can be said about converting them, and saying so is the first result.

# %%
queries = c.read_queries(CORPUS / "queries.jsonl")
promql = [q for q in queries if q.lang == "promql"]
MATCHER = re.compile(r'(\w+)\s*(=~|!~)\s*"((?:[^"\\]|\\.)*)"')

occurrences = collections.Counter()
for q in promql:
    for _label, _op, rx in MATCHER.findall(q.text):
        occurrences[rx] += 1

template = {r: n for r, n in occurrences.items() if "$" in r}
concrete = {r: n for r, n in occurrences.items() if "$" not in r}
print(f"{len(promql):,} PromQL queries carry {sum(occurrences.values()):,} regex matchers "
      f"({len(occurrences):,} distinct)")
print(f"  template-valued {len(template):>4} distinct  {sum(template.values()):>5} occurrences")
print(f"  concrete        {len(concrete):>4} distinct  {sum(concrete.values()):>5} occurrences")

# %% [markdown]
# ## Deciding each concrete regex
#
# `regex_to_globs` returns `single`, `union` or `none`. A `none` carries the name of the
# construct that caused it, so §6 can argue the refusals rather than only count them.

# %%
rows = []
for rx, n in sorted(concrete.items(), key=lambda kv: (-kv[1], kv[0])):
    got = regex_to_globs(rx)
    rows.append({"regex": rx, "occurrences": n, "kind": got.kind,
                 "n_globs": len(got.globs), "globs": " ".join(got.globs),
                 "reason": got.reason})
df = pd.DataFrame(rows)
df.to_csv(PROCESSED / "regex_glob.csv", index=False)

by_kind = df.groupby("kind").agg(distinct=("regex", "size"),
                                 occurrences=("occurrences", "sum"))
print(by_kind)

# %%
n_distinct, n_occ = len(df), int(df.occurrences.sum())


def share(kind: str) -> tuple[int, int]:
    sub = df[df.kind == kind]
    return len(sub), int(sub.occurrences.sum())


table_rows = []
for kind, label in (("single", r"One glob"),
                    ("union", r"A finite union of globs"),
                    ("none", r"Neither ($\bot$)")):
    d, o = share(kind)
    table_rows.append([label, f"{d:,}", pct(d / n_distinct), f"{o:,}", pct(o / n_occ)])
d_t, o_t = len(template), sum(template.values())
table_rows.append([r"\emph{Template-valued, not a regex}", f"{d_t:,}", "---",
                   f"{o_t:,}", "---"])
tabular("lrrrr", ["Outcome", "Distinct", "Share", "Occurrences", "Share"],
        table_rows, TABLES / "regex_glob.tex")

# %% [markdown]
# ## Why the refusals happen
#
# Three constructs account for all of them, and each is a property of globs rather than
# of this converter — §6 gives the argument for each. `nonempty-wildcard` is the
# interesting one: it is `.+`, and it is refused only because a glob's `*` is taken to
# match the empty string. Under the other reading `.*` would be the refused one instead.
# Either way one of the two most common regexes in the corpus has no glob, and which one
# is undocumented.

# %%
REASON_LABEL = {
    "single-character-wildcard": r"No single-character wildcard (\texttt{.}, \texttt{5..})",
    "nonempty-wildcard": r"No non-empty wildcard (\texttt{.+})",
    "unbounded-character-class": r"No bounded alphabet under a star (\texttt{[a-z]+})",
    "negated-character-class": r"No negated class (\texttt{[\textasciicircum{}a]})",
    "repeated-group": r"No repeated group (\texttt{(ab)*})",
    "interior-anchor": r"Anchor inside the pattern",
    "unparsed": r"Not parsed as a regular expression",
}
refused = df[df.kind == "none"]
reason_rows = []
for reason, sub in sorted(refused.groupby("reason"),
                          key=lambda kv: -kv[1].occurrences.sum()):
    shortest = min(sub.regex, key=len)
    reason_rows.append([REASON_LABEL.get(reason, code(reason)),
                        f"{len(sub):,}", f"{int(sub.occurrences.sum()):,}",
                        code(shortest, EXAMPLE_CHARS)])
tabular("lrrl", ["Construct with no glob", "Distinct", "Occurrences", "Shortest example"],
        reason_rows, TABLES / "regex_glob_reasons.tex")

# %% [markdown]
# ## Worked examples
# One per outcome, taken by occurrence count so the table shows what the corpus is made of.

# %%
WANTED = [("single", 3), ("union", 3), ("none", 3)]
ex_rows = []
for kind, k in WANTED:
    sub = df[df.kind == kind].nlargest(k, "occurrences")
    for _, r in sub.iterrows():
        target = (code(r.globs.replace(" ", ", "), EXAMPLE_CHARS) if r.kind != "none"
                  else r"$\bot$ " + REASON_LABEL.get(r.reason, r.reason).split("(")[0].strip())
        ex_rows.append([code(r.regex, EXAMPLE_CHARS), target, f"{int(r.occurrences):,}"])
tabular("llr", ["PromQL regex", "Datadog scope value", "Occurrences"], ex_rows,
        TABLES / "regex_glob_examples.tex")

# %% [markdown]
# ## Assertions
# The paper quotes these; a change that moves them should fail here, not in the PDF.

# %%
assert len(promql) == 8_107, len(promql)
assert (len(template), sum(template.values())) == (175, 7_459), (len(template), sum(template.values()))
assert (len(concrete), sum(concrete.values())) == (183, 1_013), (len(concrete), sum(concrete.values()))
assert share("single") == (60, 281), share("single")
assert share("union") == (73, 344), share("union")
assert share("none") == (50, 388), share("none")
assert set(refused.reason) == {
    "single-character-wildcard", "nonempty-wildcard", "unbounded-character-class",
}, set(refused.reason)
# Every returned glob set was verified by deciding equivalence, so "exact" is not a claim
# about the proposer.
assert df[df.kind != "none"].reason.eq("").all()
print("all assertions hold")

# %% [markdown]
# ## What this says
#
# * **Most regex matchers are not regexes we know.** 7,459 of 8,472 occurrences (88.0%)
#   are template variables resolved when a dashboard is viewed. The reverse direction
#   cannot be decided statically for them at all — not because of either language, but
#   because the query text does not contain the pattern.
# * **Of the concrete ones, a glob exists for 61.7% of occurrences**, and in more than
#   half of those only as a *union* of globs. The union is not a workaround: a Datadog
#   scope disjoins over one tag key with `IN (...)`, which is exactly the construction
#   §5's single-key disjunction proposition reads in the other direction: one fact, used
#   twice.
# * **The refusals are three constructs**, and the largest of them turns on an
#   undocumented detail of `*` rather than on anything about regular languages. 238 of
#   388 refused occurrences are `.+` and its relatives.
# * The decision procedure matters more than the proposer here. `regex_to_globs` could
#   propose anything; it is the equivalence check that makes "exact" mean something, and
#   `single_glob_by_search` is the complete procedure the tests check it against.
