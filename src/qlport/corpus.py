"""Corpus support: pull query strings out of dashboards and rule files, then
classify them by syntactic feature.

Notebook 00 turns the counts into (a) the implementation order for lower/datadog.py
and emit/promql.py and (b) the grammar-coverage table of paper §4.

Nothing here touches the network. scripts/fetch_corpus.py downloads; this module
interprets what was downloaded, and every function is pure on already-parsed
JSON/YAML, so the tests run on small literals.

Three languages appear in the corpus:
  "datadog"          metrics query strings  (sum:m{scope} by {k}.as_count())
  "datadog_formula"  the formula layer over named queries  (query1 / query2)
  "promql"           PromQL expressions

Features are read off the *raw* query text, because template variables and
boolean scopes are part of what real queries look like. Normalisation exists
only so that the grammars get a chance to parse them.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

LANGS = ("datadog", "datadog_formula", "promql")


@dataclass(frozen=True)
class Query:
    """One query string with enough provenance to cite it in the paper."""
    lang: str
    text: str
    source: str        # corpus source id, e.g. "integrations-core"
    origin: str        # file or dashboard id inside that source
    unit: str = ""     # widget / panel / rule name
    kind: str = ""     # widget | formula | panel | alert | record


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def _dd_definitions(doc: Any) -> Iterator[dict]:
    """Every widget definition in a Datadog dashboard; group widgets nest."""
    if not isinstance(doc, dict):
        return
    for w in doc.get("widgets") or []:
        if not isinstance(w, dict):
            continue
        defn = w.get("definition")
        if isinstance(defn, dict):
            yield defn
            yield from _dd_definitions(defn)


def _dd_requests(defn: dict) -> list[dict]:
    """`requests` is a list for most widget types and a dict for a few."""
    req = defn.get("requests")
    if isinstance(req, dict):
        return [v for v in req.values() if isinstance(v, dict)]
    if isinstance(req, list):
        return [v for v in req if isinstance(v, dict)]
    return []


def datadog_dashboard_queries(doc: Any, *, source: str, origin: str) -> list[Query]:
    """Metric queries and formulas of one Datadog dashboard JSON.

    Two request shapes exist: the legacy `q` string and the formulas-and-functions
    `queries` list. Only `data_source == "metrics"` queries are metric queries;
    log, event and security-signal queries are a different language and are skipped.
    """
    out: list[Query] = []
    dash = str(doc.get("title") or "") if isinstance(doc, dict) else ""
    for defn in _dd_definitions(doc):
        unit = str(defn.get("title") or dash)
        for req in _dd_requests(defn):
            q = req.get("q")
            if isinstance(q, str) and q.strip():
                out.append(Query("datadog", q.strip(), source, origin, unit, "widget"))
            for sub in req.get("queries") or []:
                if not isinstance(sub, dict) or sub.get("data_source") != "metrics":
                    continue
                text = sub.get("query")
                if isinstance(text, str) and text.strip():
                    out.append(Query("datadog", text.strip(), source, origin, unit, "widget"))
            for f in req.get("formulas") or []:
                if not isinstance(f, dict):
                    continue
                text = f.get("formula")
                if isinstance(text, str) and text.strip():
                    out.append(Query("datadog_formula", text.strip(), source, origin, unit, "formula"))
    return out


def _grafana_panels(doc: Any) -> Iterator[dict]:
    """Every panel, through collapsed rows (`panels`) and schema-v5 `rows`."""
    if not isinstance(doc, dict):
        return
    for p in doc.get("panels") or []:
        if isinstance(p, dict):
            yield p
            yield from _grafana_panels(p)
    for r in doc.get("rows") or []:
        if isinstance(r, dict):
            yield from _grafana_panels(r)


def grafana_dashboard_queries(doc: Any, *, source: str, origin: str) -> list[Query]:
    """PromQL expressions of one Grafana dashboard JSON.

    Only panel targets are taken. Template-variable queries (`label_values(...)`,
    `query_result(...)`) belong to Grafana's own variable mini-language, not PromQL,
    so they are left out of the corpus.
    """
    out: list[Query] = []
    dash = str(doc.get("title") or "") if isinstance(doc, dict) else ""
    for p in _grafana_panels(doc):
        unit = str(p.get("title") or dash)
        for t in p.get("targets") or []:
            if not isinstance(t, dict):
                continue
            expr = t.get("expr")
            if isinstance(expr, str) and expr.strip():
                out.append(Query("promql", expr.strip(), source, origin, unit, "panel"))
    return out


def prometheus_rule_queries(doc: Any, *, source: str, origin: str) -> list[Query]:
    """PromQL expressions of one PrometheusRule object (alerting and recording)."""
    out: list[Query] = []
    if not isinstance(doc, dict) or doc.get("kind") != "PrometheusRule":
        return out
    for group in (doc.get("spec") or {}).get("groups") or []:
        if not isinstance(group, dict):
            continue
        for rule in group.get("rules") or []:
            if not isinstance(rule, dict):
                continue
            expr = rule.get("expr")
            if not isinstance(expr, str) or not expr.strip():
                continue
            kind = "alert" if rule.get("alert") else "record"
            unit = str(rule.get("alert") or rule.get("record") or "")
            out.append(Query("promql", expr.strip(), source, origin, unit, kind))
    return out


# ---------------------------------------------------------------------------
# normalisation: make template-driven queries parseable
# ---------------------------------------------------------------------------
#  Grafana:  $node  ${node}  $__rate_interval  [[node]]
#  Datadog:  $host  $scope
_TMPL = r"\$\{[^{}]*\}|\$[A-Za-z_][\w.]*|\[\[[^\[\]]*\]\]"
TEMPLATE_VAR = re.compile(_TMPL)
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"' r"|'(?:[^'\\]|\\.)*'")
# A range bracket may build its duration out of several pieces: `[$__range]`, but
# also `[${__range_s}s]`, where the variable supplies only the number.
_PROM_RANGE_BODY = rf"(?:{_TMPL}|[0-9smhdwy\s])+"
_PROM_RANGE_TMPL = re.compile(rf"\[({_PROM_RANGE_BODY})?(:({_PROM_RANGE_BODY})?)?\]")
_DD_BRACES = re.compile(r"(\bby\s*)?\{([^{}]*)\}")

TMPL_PLACEHOLDER = "tmplvar"
TMPL_DURATION = "5m"
TMPL_SUBQUERY_STEP = "1m"


def _outside_strings(text: str, repl) -> str:
    """Apply `repl` to the parts of `text` that are not string literals."""
    out, i = [], 0
    for m in _STRING.finditer(text):
        out.append(repl(text[i:m.start()]))
        out.append(m.group(0))
        i = m.end()
    out.append(repl(text[i:]))
    return "".join(out)


def _prom_range_sub(m: re.Match) -> str:
    """Rewrite only the template-variable parts of a range or subquery bracket, so
    that literal durations in the corpus survive untouched."""
    if not TEMPLATE_VAR.search(m.group(0)):
        return m.group(0)
    body = m.group(0)[1:-1]
    rng, _, step = body.partition(":") if ":" in body else (body, "", None)
    rng = rng.strip()
    if TEMPLATE_VAR.search(rng) or not rng:
        rng = TMPL_DURATION
    if step is None:
        return f"[{rng}]"
    step = TMPL_SUBQUERY_STEP if TEMPLATE_VAR.search(step) else step.strip()
    return f"[{rng}:{step}]"


def normalise(lang: str, text: str) -> str:
    """Replace dashboard template variables by a concrete, parseable instance.

    PromQL: a variable in range position becomes a duration (`[$__rate_interval]`
    -> `[5m]`); any other variable outside a string literal becomes an identifier.
    Datadog: a variable inside a scope becomes `*`, the instance a dashboard shows
    by default; inside a `by {...}` list or elsewhere it becomes an identifier.
    Variables inside string literals are left alone -- PromQL matchers such as
    `{pod=~"$pod"}` already parse.
    """
    if lang == "promql":
        text = _outside_strings(text, lambda s: _PROM_RANGE_TMPL.sub(_prom_range_sub, s))
        return _outside_strings(text, lambda s: TEMPLATE_VAR.sub(TMPL_PLACEHOLDER, s))
    if lang in ("datadog", "datadog_formula"):
        def brace(m: re.Match) -> str:
            by, body = m.group(1) or "", m.group(2)
            sub = TMPL_PLACEHOLDER if by else "*"
            return f"{by}{{{TEMPLATE_VAR.sub(sub, body)}}}"
        text = _DD_BRACES.sub(brace, text)
        return _outside_strings(text, lambda s: TEMPLATE_VAR.sub(TMPL_PLACEHOLDER, s))
    raise ValueError(f"unknown language {lang!r}")


# ---------------------------------------------------------------------------
# lexical feature detection
# ---------------------------------------------------------------------------
_DD_SCOPE = re.compile(r":[A-Za-z][\w.]*\s*\{([^{}]*)\}")
_DD_GROUPING = re.compile(r"\bby\s*\{([^{}]*)\}")
_DD_AGG = re.compile(r"(?:^|[\s,(+\-*/])(p\d{1,3}|[a-z_]+)\s*:[A-Za-z][\w.]*\s*\{")
_DD_METHOD = re.compile(r"\.([a-z_][a-z0-9_]*)\s*\(")
_DD_FUNC = re.compile(r"(?<![\w.$])([a-z_][a-z0-9_]*)\s*\(")
# the tag-filter operators are case-insensitive: the corpus has `not x and *` too
_DD_BOOL_WORD = re.compile(r"(?<![\w$])(?i:AND|OR|NOT)(?![\w])")
_DD_IN = re.compile(r"(?<![\w$])(?i:NOT\s+)?(?i:IN)\s*\(")

_PROM_CALL = re.compile(r"(?<![\w:.$])([a-z_][a-z0-9_]*)\s*\(")
_PROM_AGGOPS = frozenset(
    "sum avg min max count count_values stddev stdvar topk bottomk quantile group "
    "limitk limit_ratio".split())
# `sum(x) by (k)` and `sum by (k) (x)` are both legal, so the operator is not always
# the identifier directly before a "(".
_PROM_AGG = re.compile(
    r"(?<![\w:.$])(" + "|".join(sorted(_PROM_AGGOPS, key=len, reverse=True)) +
    r")\s*(?:(?:by|without)\s*\([^()]*\)\s*)?\(")
_PROM_KEYWORDS = frozenset("by without on ignoring group_left group_right bool and or unless offset".split())


def _blank(m: re.Match) -> str:
    """Keep the delimiters, fill the body with dots: same length, no operators."""
    s = m.group(0)
    return s[0] + "." * (len(s) - 2) + s[-1]


def _mask(text: str) -> str:
    """Blank out string literals and `{...}` bodies so that operator searches cannot
    trip over a glob inside a scope or a regex inside a matcher. Length-preserving,
    so an offset found in the mask is an offset into the original text."""
    return re.sub(r"\{[^{}]*\}", _blank, _STRING.sub(_blank, text))


def _depth0_commas(masked: str) -> int:
    depth = n = 0
    for ch in masked:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            n += 1
    return n


def _dd_scope_terms(text: str) -> list[str]:
    terms: list[str] = []
    for body in _DD_SCOPE.findall(text):
        for t in re.split(r"[,()]|(?<![\w$])(?:AND|OR|NOT)(?![\w])", body):
            t = t.strip()
            if t:
                terms.append(t)
    return terms


def datadog_features(text: str) -> frozenset[str]:
    """Structural features of a Datadog metrics query, from its raw text."""
    f: set[str] = set()
    masked = _mask(text)

    for agg in _DD_AGG.findall(text):
        if re.fullmatch(r"p\d{1,3}", agg):
            f.add("AGG_PCTL")
        elif agg in ("sum", "avg", "min", "max", "count"):
            f.add(f"AGG_{agg.upper()}")
        else:
            f.add("AGG_OTHER")

    scopes = _DD_SCOPE.findall(text)
    for body in scopes:
        if body.strip() == "*":
            f.add("SCOPE_ALL")
        if TEMPLATE_VAR.search(body):
            f.add("SCOPE_VAR")
        if _DD_BOOL_WORD.search(body):
            f.add("SCOPE_BOOL")
        if _DD_IN.search(body):
            f.add("SCOPE_IN")
        if "(" in body:
            f.add("SCOPE_PAREN")
    for term in _dd_scope_terms(text):
        if term == "*" or TEMPLATE_VAR.fullmatch(term):
            continue
        if term.startswith("!"):
            f.add("SCOPE_NEG")
            term = term[1:]
        if ":" in term:
            value = term.split(":", 1)[1]
            f.add("SCOPE_GLOB" if ("*" in value or "?" in value) else "SCOPE_EQ")
        else:
            f.add("SCOPE_BARE")

    if _DD_GROUPING.search(text):
        f.add("GROUP_BY")

    methods = set(_DD_METHOD.findall(text))
    for name, code in (("as_count", "M_AS_COUNT"), ("as_rate", "M_AS_RATE"),
                       ("fill", "M_FILL"), ("weighted", "M_WEIGHTED")):
        if name in methods:
            f.add(code)
    for m in re.finditer(r"\.rollup\(([^()]*)\)", text):
        f.add("M_ROLLUP")
        f.add("M_ROLLUP_INTERVAL" if "," in m.group(1) else "M_ROLLUP_AGG_ONLY")

    if _DD_FUNC.search(text):
        f.add("FUNC_APPLIED")
    if re.search(r"[+\-*/]", masked):
        f.add("ARITH")
    if _depth0_commas(masked):
        f.add("MULTI_QUERY")
    if re.search(r"(?<![\w.])\d+(\.\d+)?(?![\w.])", masked):
        f.add("CONST")
    if "'" in text:
        f.add("STR_SINGLE_QUOTED")
    return frozenset(f)


def datadog_formula_features(text: str) -> frozenset[str]:
    """Structural features of the formula layer over named Datadog queries."""
    f: set[str] = set()
    masked = _mask(text)
    if re.fullmatch(r"[A-Za-z_]\w*", text.strip()):
        f.add("FORMULA_BARE_REF")
    if re.search(r"[+\-*/]", masked):
        f.add("FORMULA_ARITH")
    if _DD_FUNC.search(text):
        f.add("FORMULA_FUNC")
    if re.search(r"(?<![\w.])\d+(\.\d+)?(?![\w.])", masked):
        f.add("FORMULA_CONST")
    if TEMPLATE_VAR.search(text):
        f.add("FORMULA_VAR")
    return frozenset(f)


def promql_features(text: str) -> frozenset[str]:
    """Structural features of a PromQL expression, from its raw text."""
    f: set[str] = set()
    masked = _mask(text)

    if re.search(r"\[[^\[\]]*\]", text):
        f.add("SEL_RANGE")
    if re.search(r"\[[^\[\]]*:[^\[\]]*\]", text):
        f.add("SUBQUERY")
    if re.search(r"(?<![\w])offset(?![\w])", text):
        f.add("OFFSET")
    if re.search(r"@\s*(?:start\s*\(|end\s*\(|[0-9])", text):
        f.add("AT_MODIFIER")

    for body in re.findall(r"\{([^{}]*)\}", text):
        for op, code in (("=~", "MATCH_RE"), ("!~", "MATCH_NRE"), ("!=", "MATCH_NEQ")):
            if op in body:
                f.add(code)
        if re.search(r'(?<![=!~<>])=(?![=~])', body):
            f.add("MATCH_EQ")
        if "__name__" in body:
            f.add("SEL_NAMELESS")
    if re.match(r"^\s*\{", text):
        f.add("SEL_NAMELESS")

    if re.search(r"(?<![\w])by\s*\(", text):
        f.add("AGG_BY")
    if re.search(r"(?<![\w])without\s*\(", text):
        f.add("AGG_WITHOUT")
    if _PROM_AGG.search(text):
        f.add("AGG_APPLIED")
    for name in _PROM_CALL.findall(text):
        if name not in _PROM_AGGOPS and name not in _PROM_KEYWORDS:
            f.add("FUNC_APPLIED")

    if re.search(r"[+\-*/%^]", masked):
        f.add("BINOP_ARITH")
    if re.search(r"==|!=|<=|>=|<|>", masked):
        f.add("BINOP_CMP")
    if re.search(r"(?<![\w])(and|or|unless)(?![\w])", masked):
        f.add("BINOP_SET")
    if re.search(r"(?<![\w])bool(?![\w])", text):
        f.add("BOOL_MOD")
    if re.search(r"(?<![\w])on\s*\(", text):
        f.add("MATCH_ON")
    if re.search(r"(?<![\w])ignoring\s*\(", text):
        f.add("MATCH_IGNORING")
    if re.search(r"group_(?:left|right)", text):
        f.add("GROUP_LR")

    if re.search(r"(?<![\w.])\d+(\.\d+)?(?![\w.])", masked):
        f.add("NUM_LIT")
    if _STRING.search(text):
        f.add("STR_LIT")
    if TEMPLATE_VAR.search(text):
        f.add("TMPL_VAR")
    if _PROM_RANGE_TMPL.search(text) and TEMPLATE_VAR.search(text):
        for m in _PROM_RANGE_TMPL.finditer(text):
            if TEMPLATE_VAR.search(m.group(0)):
                f.add("TMPL_RANGE")
    if re.search(r"[a-z_]+\s*\([^()]*[a-z_]+\s*\(", text):
        f.add("NESTED_CALL")
    return frozenset(f)


def features(lang: str, text: str) -> frozenset[str]:
    if lang == "datadog":
        return datadog_features(text)
    if lang == "datadog_formula":
        return datadog_formula_features(text)
    if lang == "promql":
        return promql_features(text)
    raise ValueError(f"unknown language {lang!r}")


FEATURE_DOC: dict[str, str] = {
    # Datadog
    "AGG_SUM": "space aggregator sum:",
    "AGG_AVG": "space aggregator avg:",
    "AGG_MAX": "space aggregator max:",
    "AGG_MIN": "space aggregator min:",
    "AGG_COUNT": "space aggregator count:",
    "AGG_PCTL": "percentile space aggregator pNN:",
    "AGG_OTHER": "space aggregator outside the documented set",
    "SCOPE_ALL": r"scope {*}",
    "SCOPE_EQ": "scope term key:value",
    "SCOPE_NEG": "negated scope term !key:value",
    "SCOPE_GLOB": "wildcard in a scope value",
    "SCOPE_VAR": "dashboard template variable in the scope",
    "SCOPE_BOOL": "boolean scope (AND / OR / NOT)",
    "SCOPE_IN": "scope membership test IN (...)",
    "SCOPE_PAREN": "parenthesised scope",
    "SCOPE_BARE": "scope term that is a bare tag key",
    "GROUP_BY": "group by {...}",
    "M_AS_COUNT": ".as_count()",
    "M_AS_RATE": ".as_rate()",
    "M_ROLLUP": ".rollup(...)",
    "M_ROLLUP_INTERVAL": ".rollup(agg, interval)",
    "M_ROLLUP_AGG_ONLY": ".rollup(agg)",
    "M_FILL": ".fill(...)",
    "M_WEIGHTED": ".weighted()",
    "FUNC_APPLIED": "function applied to the query",
    "ARITH": "arithmetic between operands",
    "MULTI_QUERY": "comma-separated list of queries",
    "CONST": "numeric literal",
    "STR_SINGLE_QUOTED": "single-quoted string literal",
    # Datadog formula layer
    "FORMULA_BARE_REF": "formula is one query reference",
    "FORMULA_ARITH": "arithmetic over query references",
    "FORMULA_FUNC": "function over query references",
    "FORMULA_CONST": "numeric literal",
    "FORMULA_VAR": "template variable",
    # PromQL
    "SEL_RANGE": "range selector [d]",
    "SUBQUERY": "subquery [d:s]",
    "OFFSET": "offset modifier",
    "AT_MODIFIER": "@ modifier",
    "MATCH_EQ": "matcher =",
    "MATCH_NEQ": "matcher !=",
    "MATCH_RE": "matcher =~",
    "MATCH_NRE": "matcher !~",
    "SEL_NAMELESS": "selector without a metric name",
    "AGG_APPLIED": "aggregation operator",
    "AGG_BY": "aggregation by (...)",
    "AGG_WITHOUT": "aggregation without (...)",
    "BINOP_ARITH": "arithmetic binary operator",
    "BINOP_CMP": "comparison binary operator",
    "BINOP_SET": "set binary operator (and / or / unless)",
    "BOOL_MOD": "bool modifier",
    "MATCH_ON": "vector matching on (...)",
    "MATCH_IGNORING": "vector matching ignoring (...)",
    "GROUP_LR": "group_left / group_right",
    "NUM_LIT": "numeric literal",
    "STR_LIT": "string literal",
    "TMPL_VAR": "dashboard template variable",
    "TMPL_RANGE": "template variable as a range duration",
    "NESTED_CALL": "call nested inside a call",
}


def function_names(lang: str, text: str) -> Counter:
    """Applied function names. Datadog methods are keyed with a leading dot
    (`.as_count`) so that they do not collide with wrapping functions (`top`)."""
    c: Counter = Counter()
    if lang in ("datadog", "datadog_formula"):
        for name in _DD_METHOD.findall(text):
            c[f".{name}"] += 1
        for name in _DD_FUNC.findall(text):
            c[name] += 1
    elif lang == "promql":
        for name in _PROM_CALL.findall(text):
            if name not in _PROM_KEYWORDS and name not in _PROM_AGGOPS:
                c[name] += 1
    else:
        raise ValueError(f"unknown language {lang!r}")
    return c


def aggregation_ops(lang: str, text: str) -> Counter:
    """Space-aggregation operators, by name."""
    c: Counter = Counter()
    if lang == "datadog":
        for agg in _DD_AGG.findall(text):
            c[agg] += 1
    elif lang == "promql":
        for name in _PROM_AGG.findall(text):
            c[name] += 1
    else:
        raise ValueError(f"unknown language {lang!r}")
    return c


# ---------------------------------------------------------------------------
# grammar coverage
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParseOutcome:
    ok: bool
    blocker: str = ""   # the terminal the parser could not accept
    excerpt: str = ""   # the offending text


def parse_outcome(lang: str, text: str, version: str = "v2") -> ParseOutcome:
    """Try the grammar of `lang` on `text` and name what stopped it.

    The blocker is read off Lark's exception rather than guessed from a regex, so
    the gap table in paper §4 is mechanical: every row is a terminal the LALR(1)
    table had no transition for. `version` selects the starter grammar ("v1") or
    the corpus-informed one ("v2").
    """
    from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken

    from .parse import parse

    try:
        parse(lang, text, version)
        return ParseOutcome(True)
    except UnexpectedToken as e:
        return ParseOutcome(False, str(e.token.type), str(e.token)[:40])
    except UnexpectedCharacters as e:
        return ParseOutcome(False, "<no-token>", text[e.pos_in_stream:e.pos_in_stream + 20])
    except UnexpectedEOF:
        return ParseOutcome(False, "<eof>", "")
    except Exception as e:                      # grammar/visit errors must not be hidden
        return ParseOutcome(False, type(e).__name__, str(e)[:40])


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def write_queries(path: Path | str, queries: Iterable[Query]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(asdict(q), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_queries(path: Path | str) -> list[Query]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Query(**json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# explaining the failures: each candidate repair is one grammar change
# ---------------------------------------------------------------------------
# A failing query is explained by the smallest set of repairs that makes it parse.
# Every repair names one concrete production that grammar v1 lacked and v2 adds, so
# the gap table of paper §4 is verified by construction rather than classified by
# hand: the repair measures the gap against v1, and `tests/test_parse.py` then
# asserts that v2 parses the same construct unrepaired.
_DD_NO_AGG = re.compile(r"(?<![\w:.$])(?!by\b)([a-z][\w.]*)\s*\{")
_DD_AGG_PREFIX = re.compile(r"(?<![\w:.$])(?:count|p\d{1,3})\s*:")
_PROM_KEYWORDS_UPPER = (
    "BY WITHOUT ON IGNORING GROUP_LEFT GROUP_RIGHT OFFSET BOOL AND OR UNLESS "
    "SUM AVG MIN MAX COUNT TOPK BOTTOMK QUANTILE STDDEV STDVAR GROUP").split()
_PROM_KW_RE = re.compile(r"(?<![\w:])(" + "|".join(_PROM_KEYWORDS_UPPER) + r")(?![\w])")
_LOGQL = re.compile(r"\|\s*(?:logfmt|json|pattern|line_format|unwrap|label_format|regexp)\b"
                    r"|\|=|\|~|!~\s*`"
                    # a bare line filter: `{...} != "x"`. Comparing a selector to a
                    # string is a type error in PromQL, so this can only be LogQL.
                    r"|\}\s*(?:!=|=~|!~|=)\s*[\"\'`]")


_PROM_LABEL_LIST = re.compile(
    r"(?i:by|without|on|ignoring|group_left|group_right)\s*\(([^()]*)\)")
_PROM_QUOTED_MATCHER = re.compile(r'"([^"]*)"\s*(=~|!~|!=|=)')


def _prom_legal_label(name: str) -> str:
    """A bare-LABEL stand-in for a quoted (UTF-8) label name."""
    out = re.sub(r"\W", "_", name)
    return out if re.match(r"[A-Za-z_]", out) else "l" + out


def _prom_unquote_labels(text: str) -> str:
    """Replace quoted label names by legal bare ones, in a matcher and in a
    `by`/`without`/`on`/`ignoring`/`group_left`/`group_right` label list."""
    text = _PROM_QUOTED_MATCHER.sub(
        lambda m: _prom_legal_label(m.group(1)) + m.group(2), text)
    return _PROM_LABEL_LIST.sub(
        lambda m: m.group(0).replace(m.group(1), _STRING.sub(
            lambda s: _prom_legal_label(s.group(0)[1:-1]), m.group(1))), text)


def is_promql(text: str) -> bool:
    """False for LogQL: some community dashboards listed under the Prometheus
    datasource have Loki panels, and those are not evidence about PromQL."""
    return not _LOGQL.search(text)


def _dd_scope_map(text: str, fn) -> str:
    """Rewrite the body of every scope `:metric{...}` with `fn`."""
    return _DD_SCOPE.sub(lambda m: m.group(0)[: m.start(1) - m.start()] + fn(m.group(1)) + "}", text)


def _dd_drop_boolean_scope(body: str) -> str:
    return "*" if (_DD_BOOL_WORD.search(body) or _DD_IN.search(body) or "(" in body) else body


def _dd_flatten_tag_colons(body: str) -> str:
    out = []
    for term in body.split(","):
        key, sep, value = term.partition(":")
        out.append(key + sep + value.replace(":", "_") if sep else term)
    return ",".join(out)


def _dd_first_query(text: str) -> str:
    masked, depth = _mask(text), 0
    for i, ch in enumerate(masked):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            return text[:i]
    return text


REPAIRS: dict[str, tuple[tuple[str, Any], ...]] = {
    "datadog": (
        ("IMPLICIT_AGG", lambda t: _DD_NO_AGG.sub(r"avg:\1{", t)),
        ("AGG_SET", lambda t: _DD_AGG_PREFIX.sub("avg:", t)),
        ("BOOL_SCOPE", lambda t: _dd_scope_map(t, _dd_drop_boolean_scope)),
        ("QUERY_LIST", _dd_first_query),
        ("SINGLE_QUOTES", lambda t: re.sub(r"'([^']*)'", r'"\1"', t)),
        ("TAG_COLON", lambda t: _dd_scope_map(t, _dd_flatten_tag_colons)),
        # Only the whitespace the gap is actually about: after a `!`, and around the
        # term separator. Collapsing *all* whitespace in a scope would glue
        # `{* AND k:v}` into one tag term and so explain BOOL_SCOPE by accident.
        ("SCOPE_SPACING", lambda t: _dd_scope_map(
            t, lambda b: re.sub(r"\s*,\s*", ",", re.sub(r"!\s+", "!", b)))),
        ("NEGATED_WILDCARD", lambda t: t.replace("!*", "*")),
        ("UNARY_SIGN", lambda t: re.sub(r"^\s*[-+]\s*", "", t)),
        ("NEGATIVE_ARG", lambda t: re.sub(r"([(,]\s*)-(?=\d)", r"\1", t)),
        # `{*-location}`: a wildcard in the *key* half of a tag term, not the value.
        ("WILDCARD_TAG_KEY", lambda t: _dd_scope_map(
            t, lambda b: re.sub(r"\*(?=[\w.\-/])", "", b))),
    ),
    "promql": (
        ("KEYWORD_CASE", lambda t: _outside_strings(t, lambda s: _PROM_KW_RE.sub(
            lambda m: m.group(1).lower(), s))),
        # a label name is quoted precisely when it uses characters the bare LABEL
        # terminal forbids, so the repair has to supply a legal name as well.
        ("QUOTED_LABEL", _prom_unquote_labels),
        ("SIGNED_OFFSET", lambda t: re.sub(r"(offset\s*)[-+]\s*(?=[\w$])", r"\1", t)),
        ("LEADING_DOT_NUMBER", lambda t: re.sub(r"(?<![\w.])\.(?=\d)", "0.", t)),
        ("BACKTICK_STRING", lambda t: re.sub(
            r"`([^`]*)`",
            lambda m: '"' + m.group(1).replace("\\", "\\\\").replace('"', '\\"') + '"', t)),
        ("TEMPLATED_SYNTAX", lambda t: re.sub(
            rf"(?<![\w]){TMPL_PLACEHOLDER}\s*\(", "sum(",
            re.sub(rf"(?<![\w]){TMPL_PLACEHOLDER}(?![\w])", "1", t))),
    ),
}

REPAIR_DOC = {
    "IMPLICIT_AGG": "metric with no space-aggregator prefix",
    "AGG_SET": "count: and percentile pNN: aggregators",
    "BOOL_SCOPE": "boolean scope: AND / OR / NOT, IN (...), parentheses",
    "QUERY_LIST": "comma-separated list of queries in one request",
    "SINGLE_QUOTES": "single-quoted string literal",
    "TAG_COLON": "colon inside a tag value",
    "SCOPE_SPACING": "whitespace after a negation or around a separator",
    "NEGATED_WILDCARD": "negated wildcard scope term",
    "UNARY_SIGN": "leading unary sign",
    "NEGATIVE_ARG": "negative numeric argument",
    "WILDCARD_TAG_KEY": "wildcard in the key half of a tag term",
    "BACKTICK_STRING": "backtick-quoted raw string",
    "KEYWORD_CASE": "keywords are case-insensitive",
    "QUOTED_LABEL": "quoted (UTF-8) label name in a matcher or label list",
    "SIGNED_OFFSET": "signed offset duration",
    "LEADING_DOT_NUMBER": "number with no digit before the point",
    "TEMPLATED_SYNTAX": "template variable standing for syntax, not a value",
}


def _apply(lang: str, text: str, names: Iterable[str]) -> str:
    chosen = set(names)
    for name, fn in REPAIRS[lang]:
        if name in chosen:
            text = fn(text)
    return text


def repairs_needed(lang: str, text: str, version: str = "v1") -> tuple[str, ...] | None:
    """The smallest set of repairs that makes `text` parse under grammar `version`.

    `()` means it already parses; `None` means no combination of the known repairs
    is enough, which is the residual the gap table has to report honestly.
    Minimisation is one removal pass over an all-repairs-applied baseline, which is
    exact whenever the repairs are independent and conservative otherwise.

    The default is "v1", because a repair exists to describe what the starter
    grammar was missing; against v2 every repair but the residual returns `()`.
    """
    if parse_outcome(lang, text, version).ok:
        return ()
    names = [n for n, _ in REPAIRS[lang]]
    if not parse_outcome(lang, _apply(lang, text, names), version).ok:
        return None
    keep = list(names)
    for name in names:
        trial = [n for n in keep if n != name]
        if parse_outcome(lang, _apply(lang, text, trial), version).ok:
            keep = trial
    return tuple(keep)
