"""Named semantic gaps. Every ⊥ returned by an emitter carries exactly one Gap.

The codes double as the rows of the gap catalogue (paper §5) and the failure
breakdown for κ (paper §7), so add a code rather than reusing a vague one.
"""
from dataclasses import dataclass
from enum import Enum


class Gap(str, Enum):
    # information barrier (paper §3): the target system does not store what the query reads
    PHI_ABSOLUTE = "query reads absolute counter values erased by ingestion"
    PHI_SERIES_START = "result depends on where a series begins inside the window"
    # expressiveness gaps: data exists, target language cannot say it
    OFFSET = "time-shifted selector has no target equivalent"
    SUBQUERY = "subqueries have no target equivalent"
    MANY_TO_ONE = "group_left/group_right vector matching"
    RATE_EXTRAPOLATION = "rate extrapolation differs; only ε-sound"
    REGEX_NOT_GLOB = "regular expression has no equivalent glob"
    ROLLUP_ALIGNMENT = "rollup buckets cannot be aligned with the target's windows"
    SCOPE_DISJUNCTION = "disjunctive scope is not one label-matcher conjunction"
    LABEL_NAME_GLOB = "wildcard over tag keys, not tag values"
    AGG_ROLLUP_ORDER = "space and time aggregation do not commute for this aggregator"
    FILL_INTERPOLATION = "fill() interpolation has no target equivalent"
    ROLLUP_DEFAULT = "Datadog's automatic rollup interval is undocumented"
    COUNTER_RESET = "cumulative difference misreads a counter reset"
    ANCHORED_UNVERIFIED = "anchored range semantics not yet verified against Prometheus"
    UNSUPPORTED_FUNCTION = "function not modelled yet"
    # measurement status, not a translation failure
    UNTESTED_LOOKBACK = "lookback longer than the test run; translated but not testable"


@dataclass(frozen=True)
class Bottom:
    """The ⊥ result of a translation, with the reason it was refused."""
    gap: Gap
    detail: str = ""

    def __str__(self) -> str:
        return f"⊥[{self.gap.name}] {self.detail}".rstrip()
