"""The regular sub-problem (paper §6): regex matcher <-> Datadog glob.

glob_to_regex is exact by construction. regex_to_glob proposes a glob from the regex's
structure; `equivalent` checks L(r) = L(g) by DFA equivalence (Rich Thm 9.5) and the
caller returns Bottom(REGEX_NOT_GLOB) when it fails. Uses `greenery` (pip extra "regular").
"""
from __future__ import annotations

import re


def glob_to_regex(g: str) -> str:
    """Datadog tag glob: * = any string, ? = one character (verify ? support)."""
    return "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in g)


def equivalent(regex_a: str, regex_b: str) -> bool:
    from greenery import parse  # optional dependency
    return parse(regex_a).equivalent(parse(regex_b))


def regex_to_glob(r: str) -> str | None:
    raise NotImplementedError
