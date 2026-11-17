# Settling Datadog's undocumented behaviour (open item 5)

Three claims in the paper are stated relative to facts about Datadog that its
documentation was assumed not to settle. Paper §7.9 ("What is not measured") now separates
what has been read from what is still unmeasured, and points here for the experiments:

- §3.1 Proposition 3 holds for ρ = zero, and the restart rows of
  `rollup_translation_live.tex` compare against φ_zero. **Settled from the agent source**
  (Step 0): ρ = zero on the OpenMetrics path. §3.1's remark "Which ρ, and how we know" states it, and
  the ingestion mapping now reads ρ ∈ {zero, clamp}, and `semantics/ingest.py` implements all three. What is left is
  confirming the backend stores what the agent sends.
- Σ_Δ presumes buckets are right-closed and epoch-aligned, reported as
  `ROLLUP_ALIGNMENT` on every rollup the translator emits (100% of translations).
  **Settled by E2: left-closed, named by the start.** Epoch alignment was already
  documented; the closure was not, and the assumption was wrong.
- The rollup interval is assumed when the query does not state one, reported as
  `ROLLUP_DEFAULT` (96.9% of translations). **Measured by E4:** 20 s over an hour, 60 s
  over four hours, 300 s over a day. The assumed 60 s is a four-hour view.
- §6's regex → glob classification assumes a glob's `*` matches the empty string, which
  decides whether `.+` or `.*` is the one with no glob. **Settled by E5: it does.**

Reading the documentation properly turns out to answer more of this than expected, and
to contradict one thing the code currently does. **Read the "already answered" section
before running any experiment** — it removes two of them and reduces a third to a
confirmation.

## Already answered by the documentation

| Question | Status | Source |
|---|---|---|
| Order of time vs. space aggregation | **Answered — code fixed, κ 90.2% → 94.2%** | [Nested queries](https://docs.datadoghq.com/metrics/nested_queries/), [how weighted() works](https://docs.datadoghq.com/dashboards/guide/how-weighted-works/) |
| Epoch alignment of buckets | **Answered: aligned to UNIX time** | [Rollup](https://docs.datadoghq.com/dashboards/functions/rollup/) (monitors section) |
| Automatic rollup interval | **Answered as a lookup table**, not a formula | [Rollup](https://docs.datadoghq.com/dashboards/functions/rollup/) |
| Default time aggregator | Partly: `avg` is the enforced default; `.as_count()` sets it to `sum`; COUNT/RATE get `.as_count()` appended **in the UI only** | [Rollup](https://docs.datadoghq.com/dashboards/functions/rollup/) |
| Right- vs left-closure of buckets | **Not documented** | — |
| Reset policy ρ | **Answered by the agent source: ρ = zero on the OpenMetrics path** (ρ = clamp on the classic path) | [`monotonic_count.go`](https://github.com/DataDog/datadog-agent/blob/dcc0fd8530af/pkg/metrics/monotonic_count.go), [`base_scraper.py`](https://github.com/DataDog/integrations-core/blob/c2ab66f9a708/datadog_checks_base/datadog_checks/base/checks/openmetrics/v2/scraper/base_scraper.py) |

### Aggregation order: a bug, not an unknown — now fixed

> "each metric query in Datadog is evaluated with two layers of aggregation: first by
> time, then by space"

The `weighted()` page works `sum:kubernetes.cpu.requests{*} by {kube_container_name}.rollup(avg, 10)`
through explicitly: step 1 averages **each timeseries** over each 10 s interval; step 2
sums those averaged values per `kube_container_name`.

`lower/datadog.py` builds `Rollup(time_agg, Δ, Agg(space_agg, keys, Select))` — rollup
*outside* the aggregation, i.e. space first. `emit/promql.py` then tries to push the
rollup inside and refuses when the two do not commute, as `AGG_ROLLUP_ORDER`.

Under the documented order there is nothing to commute: the rollup is *already* inside,
and that is the shape PromQL writes naturally.

```
max:x{*}.rollup(avg, 60)   →   max(avg_over_time(x[60s]))
min:nifi.can_connect{*}    →   min(avg_over_time(nifi_can_connect[60s]))
```

Both are currently refused. The commuting cases were emitted correctly by accident,
because with the rollup inside an aggregation that does commute the two orders coincide
— which is why the 6,931 translated queries are unaffected.

**Fixed.** The IR now builds `Agg(space, keys, Rollup(time, Δ, Select))`, the
commutation logic and the `AGG_ROLLUP_ORDER` code are gone, and `.as_rate()`'s division
sits on the inner node because it is part of the time aggregation — observable only for
`count:`, where dividing after the space aggregation would scale a count of series by
the interval. All 348 occurrences (308 distinct) now translate: **κ went from 90.2% to
94.2%** of occurrences and the gap catalogue from five rows to four. §5 gained a
subsection stating the order, because getting it backwards leaves every emitted
translation looking plausible and shows up only in the refusals.

### Automatic interval: read the table, confirm one row

The interval is set by a documented time-frame × widget-type table ("the past hour" =
20 s for the API column), driven by a 1,500-point-per-series cap, not by a formula. So
`ROLLUP_DEFAULT` can cite a documented table instead of an assumption. Two caveats
worth confirming empirically (E4 below): the table is published for widgets, and the
API column may not match what `/api/v1/query` actually returns; and a manual interval
finer than the enforced one appears to be ignored rather than honoured — the parameter
is documented as applying "if larger than the query-enforced rollup interval", which is
an implication, not a statement.

## What only you can do

**Status: the trial exists** (US1, so `DD_SITE=datadoghq.com`; 14 days from 15 November
2026). The org has no metrics yet — the catalogue is empty — so nothing below can be read
off the UI until data flows.

**The order has changed.** Step 0 settled ρ from the source, so E1 — the experiment the
agent install is for — is now a confirmation of a known answer. E2 is the highest-stakes
*unsettled* question, because §3.1's Σ_Δ assumes right-closed buckets and nothing
documents that, and E2 needs only direct submission. So the agent install is no longer the
critical path; an API key and an application key are.


I cannot create the account or handle the key. Before any of this runs:

1. ~~Create a Datadog trial.~~ Done. Still needed: an API key, and an *application*
   key for the query API (Organization Settings → API Keys / Application Keys). Both are
   required — the query API rejects an API key alone.
2. Put them in `.env` as `DD_API_KEY` / `DD_APP_KEY` and `DD_SITE` (the site matters:
   `datadoghq.com` vs `datadoghq.eu` have different API hosts). `.env` is gitignored;
   the scripts read it and never log it.
3. Decide the billing exposure. A trial on the free tier with one custom metric and a
   handful of series is well inside limits, but custom metrics are the billed axis, so
   keep the tag cardinality at one or two series.

Everything below then runs from `scripts/trial_checks.py`, caching raw API responses
under `data/raw/datadog/` so the notebooks stay offline (README rule 3):

```
python scripts/trial_checks.py e2    # right- or left-closed buckets
python scripts/trial_checks.py e3    # default time aggregator for a COUNT
python scripts/trial_checks.py e4    # the automatic rollup interval
```

Each exits non-zero if its result is inconclusive, and prints the interpretation. Add
`--dry-run` to see the payloads and queries without keys or network.

## Step 0 — read the agent source — **done**

ρ is settled for the modelled path, and the guess that motivated this section was wrong.
Pinned at `datadog-agent` [`dcc0fd8530af`](https://github.com/DataDog/datadog-agent/blob/dcc0fd8530af/pkg/metrics/monotonic_count.go)
and `integrations-core` [`c2ab66f9a708`](https://github.com/DataDog/integrations-core/blob/c2ab66f9a708/datadog_checks_base/datadog_checks/base/checks/openmetrics/v2/scraper/base_scraper.py).

`MonotonicCount.addSample` has two modes, selected per sample by a `FlushFirstValue` flag:

```go
diff := mc.currentSample - mc.previousSample
if (mc.hasPreviousSample || mc.flushFirstValue) && diff >= 0. {
        mc.value += diff
} else if mc.flushFirstValue {
        mc.value = mc.currentSample
}
```

- **Flag set.** A sample below its predecessor contributes its own value, "the assumption
  being that the underlying raw counter was reset from 0" — this is **ρ = zero**, exactly
  PromQL's assumption.
- **Flag clear.** The decrease contributes nothing and the point is still emitted, so the
  reset interval carries δ = 0. That is **ρ = clamp**, which is *not* ρ = drop: nothing is
  discarded. `monotonic_count_test.go` pins both ("flush the lower value as-is" against
  "flush 0"), as does the accumulating case, where resets at two points of one interval
  give `17 = (12-9) + (20-10) + (17-13)`.

The OpenMetrics V2 scraper sets `self.flush_first_value = True` after each successful
scrape, so **every scrape from the second onward runs with the flag set**: the agent path
this paper models implements ρ = zero and §3.1 Proposition 3 stands as written. Two
exceptions, both of which the flag logic makes explicit rather than hiding:

- The **first** scrape runs with the flag clear, so its sample is not flushed at all —
  which is the `drop` branch, but for the series' first interval, not for a reset. The
  exception is `use_process_start_time`, where `first_scrape_handler` sets the flag mid-scrape
  if `process_start_time_seconds` exceeds the agent's own start time; it buffers counter
  metrics so the flip lands before they are transformed.
- A **failed** scrape resets the flag to `False`, so the next successful scrape drops its
  interval too. A transient scrape failure therefore injects a φ discontinuity that no
  PromQL expression models and no query can see.

### What this changes

1. The ingestion mapping's index set is wrong, not incomplete in a harmless way: ρ = drop is not
   what either agent path does to a reset. The real alternatives are
   ρ ∈ {zero, clamp}, with `drop` describing the first interval and the post-failure
   interval. §3.1 should say so and keep Proposition 3 on ρ = zero.
2. Proposition 3 needs one hypothesis it does not currently state. Under the flag,
   `mc.value = mc.currentSample` **overwrites** the accumulation rather than adding to it,
   so if two scrapes land in one flush interval and the second resets, the increase
   accumulated before the reset is discarded — φ_zero as defined would keep it. Exactness
   therefore needs *at most one sample per flush interval*, which the default 15 s check
   and 15 s flush satisfy but a finer check interval would not.
3. `COUNTER_RESET` stays a real caveat, and only on the cumulative difference, as §7 has
   it. The feared outcome — neither candidate exact, a new gap code — does not arise.

A useful side-finding: the commit that last touched this file (#59132, 2026-11-10) adds
`isDecomposable` with the comment that "deltas flushed for the parts of a window sum up to
the delta of the window, as the previous sample is kept across flushes". That is Datadog
asserting the decomposability Σ_Δ relies on, and it is citable.

Reading the source establishes intent; it does not establish what the backend stores.
E1 still confirms it end to end — and is now a confirmation of a known answer rather than
a four-way discrimination, so the ρ = drop and raw-difference columns of its table are
there only to catch a surprise.

## The rigs

Two ingestion paths, because the questions live in different components. State in §7
which one the paper models — φ is a property of the conversion step, not of "Datadog".

**Rig A (agent path), for E1.** A controllable `/metrics` endpoint exposing one counter
whose value follows a script. Prometheus scrapes it every 15 s (the existing pinned
image). A Datadog Agent container runs an OpenMetrics V2 check on the same endpoint at
the same interval. Both systems then hold the same samples, which is the dual-write
model of §3.1. Must run in real time: the intake rejects points more than ~1 h old, and
the agent's conversion is what is under test.

**Rig B (direct submission), for E2–E4.** `POST /api/v2/series` with `type: 1` (count)
and explicit timestamps. This bypasses the agent, so it says nothing about ρ, but it
gives exact control of sample timestamps — which E2 needs and the agent cannot provide.

## E1 — the reset policy ρ (Rig A) — **done: ρ = zero**

**Answer: ρ = zero, measured end to end through a real agent.** Over the bucket holding
the reset Datadog stored **103 across four points**, which is δ_zero and is distinct from
every rival: clamp predicts 3 over four points, drop 3 over three, and a raw signed
difference −897. The reset-to-0 case read 3, consistent with zero and clamp and ruling
out a special case at zero; the never-resetting control read four increments of one in
every bucket. Cached in `data/raw/datadog/e1.json`, regression-tested in
`tests/test_harness_datadog.py`, with the prediction table itself pinned in
`tests/test_reset.py`. Run by `scripts/reset_policy.py`.

This confirms Remark `rem:rho`'s reading of the agent source rather than overturning it,
so no figure in the paper moved. Two things are worth recording anyway:

- **Prometheus agreed on every complete bucket**, the reset bucket included, where
  `increase(x[60s] anchored)` returned 103. §3.1's restart claim had until now been the
  model tracking φ_zero rather than Datadog agreeing with it; that gap is closed.
- **The agent's first configuration measured nothing.** Naming `type: monotonic_count`
  on the metric asked the OpenMetrics v2 check for a transformer, and there is no such
  transformer type; the check errored rather than silently measuring something else,
  which is the good case. A Prometheus counter is already sent as a monotonic count, so
  the mapping needs no type at all.

### The original design


Expose a counter incrementing by 1 per scrape, then at one scrape drop the exposed value
from 1000 to 100, then resume incrementing by 1. Place the reset at the second of the
four scrapes in a 60 s bucket, and read both `.rollup(sum, 60)` and `.rollup(count, 60)`
for that bucket. The second reading is what separates `drop` from `clamp`:

| ρ | δ at the reset | `.rollup(sum, 60)` | `.rollup(count, 60)` |
|---|---|---|---|
| zero (PromQL's assumption) | 100 | **103** | 4 |
| drop (interval discarded) | — | **3** | **3** |
| clamp to zero | 0 | **3** | **4** |
| raw difference | −900 | **−897** | 4 |

All four predictions are distinct in at least one column. Include ≥10 min of warm-up
before the reset, since the first sample after the agent starts is omitted.

Repeat once with a reset to exactly 0 rather than 100, to check that `zero` and `clamp`
are not conflated by a special case at 0.

## E2 — are buckets right- or left-closed? (Rig B) — **done: left-closed**

**Answer: buckets are left-closed `[b, b+Δ)` and reported under their start.** Two runs at
two rollup intervals agree; each of the other three conventions is ruled out by at least
one reading. Cached in `data/raw/datadog/e2.json`, regression-tested in
`tests/test_harness_datadog.py`, and now stated as paper §3.1's closure remark.

This was the opposite of the assumption, and three things followed:

- `semantics/datadog_eval.py` now sums `[b, b+Δ)` and exposes `eval_instant`, because a
  PromQL window is right-closed `(u-Δ, u]` and so matches a left-closed bucket only when
  evaluated **strictly inside** it. At `u = b+Δ` the window takes the boundary sample
  Datadog assigns to the next bucket and misses the one Datadog assigns to this one —
  wrong on almost every bucket when the scrape interval divides the rollup interval.
- `scripts/verify_anchored.py` queries the shifted grid and records the convention and
  epsilon; it was re-run against Prometheus 3.15.0, and `rollup_translation_live.tex`
  came back **byte-identical**. A model and a candidate paired under the same convention
  agree with each other, so the error was in what the numbers were a claim about, not in
  the numbers.
- Three flaws in the apparatus surfaced and are fixed: serial polling that starved the
  second probe, metric names shared across runs that let one run read another's points,
  and an inference that assumed the mid-bucket probe's position instead of taking it.

The original plan below is kept because its prediction table is what the run was scored
against.

### The original design

**Two probes, not one.** Writing this up exposed that a single point cannot answer it,
because *two* conventions are unknown at once: whether a bucket is right-closed `(t−Δ, t]`
or left-closed `[t, t+Δ)`, and whether the timestamp Datadog reports for a bucket is its
start or its end. A point at exactly t\* is reported at t\* under **both** "right-closed,
labelled by end" and "left-closed, labelled by start" — the two hypotheses are
indistinguishable from it.

A second point in **mid-bucket** separates them, because t\*+Δ/2 lies in the same bucket
under either closure, so where it is reported depends only on the labelling. With the
labelling fixed, the on-grid point then fixes the closure:

| | on-grid point at t\* | mid-bucket point at t\*+Δ/2 |
|---|---|---|
| right-closed, labelled by start | t\*−Δ | t\* |
| right-closed, labelled by end | t\* | t\*+Δ |
| left-closed, labelled by start | t\* | t\* |
| left-closed, labelled by end | t\*+Δ | t\*+Δ |

All four rows are distinct as pairs. So E2 submits two points, on two metrics so each gets
its own query: value 1000 at a multiple of 60, and value 1000 at that plus 30, read back
with `.rollup(sum, 60)` over a window spanning both neighbouring buckets.
`interpret_closure` in `harness/datadog.py` does the inference and returns *all* hypotheses
consistent with the pair, so an unmodelled answer comes back inconclusive rather than
rounded to the nearest guess.

## E3 — the default time aggregator for a COUNT over the API (Rig B) — **done: avg**

**Answer: `avg`.** Four deltas of 1, 2, 3, 10 come back as 4, their mean, not 16. So the
enforced `avg` default applies and the `.as_count()` the UI appends does *not* extend to
the API. This **confirms** `METRIC_TYPE_UNKNOWN` rather than narrowing it: Datadog
averages stored deltas, `avg_over_time` averages running totals, and the precondition on
76.3% of translations is load-bearing.

One fix was needed first. A 240 s query window made Datadog choose a **1 s** rollup,
which put each of the four samples in its own bucket; the window is now an hour, which is
both coarse enough and what a dashboard would use.

### The original design

The documentation says `avg` is the enforced default but that COUNT and RATE metrics get
`.as_count()` appended *in the UI*. Whether a bare API query does the same is not
documented, and `METRIC_TYPE_UNKNOWN` depends on it.

Submit four COUNT points with deltas 1, 2, 3, 10. Query the metric with no modifier at
all, then with `.as_count()`, then with `.as_rate()`:

| Behaviour | Bare query returns |
|---|---|
| `avg` of the deltas | 4 |
| `sum` (as if `.as_count()`) | 16 |
| `sum/Δ` (as if `.as_rate()`) | 16/Δ — 0.267 at Δ = 60 |

The third row is why this cannot assume Δ = 60: a bare query is exactly the query whose
interval Datadog chooses, so the denominator is E4's unknown. The four points therefore go
into four *consecutive seconds* well inside a bucket (t\*+31…34), which keeps them in one
bucket at every interval Datadog plausibly picks, and the prediction is read against the
interval the response reports. `interpret_default_aggregator` checks that all four samples
sit strictly inside one bucket of that interval and refuses to name an aggregator
otherwise — comparing a partial bucket against a whole-bucket prediction would name one
confidently and wrongly.

## E4 — confirm the automatic interval table (Rig B) — **done**

**Answer: the interval is a function of the query range**, and the documented widget
table's API column is right where they overlap.

| range | reported `interval` | observed spacing |
|---|---|---|
| 1 min | 1 s | 10 s (the data's own spacing) |
| 5 min | 2 s | 10 s |
| 1 h | **20 s** | 20 s |
| 4 h | 60 s | 60 s |
| 1 d | 300 s | 300 s |

Two consequences. `DEFAULT_ROLLUP = 60.0` corresponds to a **four-hour** dashboard view,
so it is right for that range and wrong by 3× for the one-hour view the documentation is
written around — `ROLLUP_DEFAULT` is a conditional assumption, not an unverified one.
And a finer interval than the range allows is **rejected**, not silently widened:
`.rollup(sum, 1)` over a day fails with `86400 epochs above limit of 45000`, so the cap
is on points per query. An error is better than quietly coarser data.

### The original design

Submit a point every 10 s across the **past** hour — 351 points, inside the intake's
backfill window, so this costs one request batch rather than two hours of sitting with it.
The plan originally called for two hours of real time; that is unnecessary, because the
interval Datadog reports is a function of the *query range*, not of the data, so a 4 h or
1 d range can be asked about on an hour of data and the far buckets simply come back null.

Query with no `.rollup()` over ranges of 1 min, 5 min, 1 h, 4 h and 1 d, reading both the
response's own `interval` field and the modal spacing of the returned `pointlist` — they
are separate claims and the API is not promised to agree with the widget table (e.g. 20 s
for "the past hour"). Also submit `.rollup(sum, 1)` over a 1 d range to see whether a
finer-than-enforced interval is rejected with an error or silently widened.

## E5 — does a glob's `*` match the empty string? (Rig B) — **done: yes**

**Answer: `*` matches the empty string.** `probe:web-*` selects the series whose tag value
is exactly `web-` as well as `web-1`. So `.*` has a glob, `.+` has none, and §6's figures
stand as computed; `STAR_MATCHES_EMPTY = True` in `regular.py` is the right reading.

Two fixes were needed first: the control query is `by {probe}` and so returns two series,
which the single-series poller refused; and Datadog returns a scope as a comma-separated
tag list (`probe:web-,run:r123`), so matching the scope string whole found nothing.

### The original design

Paper §6 needs this and the documentation does not give it. "Prefix, suffix, and
substring wildcard tag value filtering are supported", with `pod_name: web-*` as the
prefix example, says nothing about whether `web-*` also matches the value `web-`
itself.

It decides which of the two most common PromQL regexes has no glob. A glob has exactly
one wildcard, which either requires a character or does not, so `.*` and `.+` cannot both
be expressible: under one reading `.+` has no glob, under the other `.*` has none. The
corpus uses `.+` 133 times and `.*` 9 times, so the reading is worth 124 occurrences of
§6's coverage either way.

Submit two series of one COUNT metric, tagged `rig:b` and distinguished by a second tag:
one with `probe:web-` and one with `probe:web-1`. Then read
`sum:qlport.e5.star{probe:web-*} by {probe}`.

| Returned series | Reading |
|---|---|
| both `web-` and `web-1` | `*` matches the empty string — the reading `regular.py` takes |
| `web-1` only | `*` requires at least one character |

The second outcome flips `STAR_MATCHES_EMPTY` in `src/qlport/regular.py`, after which
notebook 04 re-runs offline and §6's table changes in a known direction: `.+` and its
relatives become expressible and `.*` stops being. The refusal count does not simply
fall — `.*` and `*pod*`-style patterns move the other way — which is why the notebook
recomputes it rather than the paper adjusting it by hand.

A caveat worth recording with the result: a tag value that is empty after the prefix is
an odd thing to have in a real tag set, so if the `web-` series does not appear it is
worth confirming that Datadog stored it at all before concluding anything about `*`.
Query `sum:qlport.e5.star{probe:web-}` to check.

## E10 — does `top()` do what the documentation says? (Rig B) — **done: yes**

**Answer: yes, on all 22 selections, and the ranking is range-relative.** §7 classified
`top()`'s 139 occurrences as unimplemented work on the strength of an expressibility
argument with one measured half: the candidate expression was read off a real Prometheus,
while Datadog's side of it was a reading of the documentation. That is the weaker kind of
evidence and the kind this project has twice found wrong — the aggregation order and the
bucket closure were both read off the documentation and both wrong — so the vendor was
asked.

Five series as a GAUGE, 20 points each at 60 s, chosen so that ranking over the window
disagrees with ranking at an instant: a flat high series, a riser, a flat runner-up, a
flat low series, and one with a single early spike. The workload lives in
`src/qlport/harness/ranking.py` and is imported by both `scripts/verify_topk.py` and
`scripts/datadog_top.py`, so the two halves of the comparison cannot drift apart; the
tests pin each cached run to that one definition.

| Claim | Reading |
|---|---|
| ranks by an aggregation over the whole queried range | **22/22** selections match, over five ranking functions × n ∈ {2,3} × both directions |
| returns whole trajectories, not winning instants | the spiking series comes back with **all 20 buckets** under `max` |
| the selection depends on the queried range | by `max`, the full window gives {riser, spike} and the last ten buckets give {riser, flat high} — while `mean` does not move, so the probe discriminates |

The third row is the one that changes an argument rather than confirming one. §8 counts
`top()` as a third instance of a translation needing context the query does not carry;
that the context is load-bearing was asserted, and is now measured.

Two corrections fell out of the corpus side rather than the account. The corpus uses
**four** ranking functions, not the five §7 claimed — `mean` in 111 calls, `last` 34,
`max` 15, `sum` 3, and no `min` at all. And 5 of the 139 occurrences rank **ascending**,
which `topk` does not cover; `verify_topk.py` now probes `bottomk` for them, so the
verified claim reaches all 139 rather than 134.

  python scripts/datadog_top.py [--dry-run]

Costs five custom metric series for one run, against the one or two this document asks
for elsewhere: five is what a ranking needs, and they are one metric name used once.

## The dual-write replay — **done**

Not one of the numbered checks, but the experiment they were all working towards.
`scripts/replay_dualwrite.py` replays the busiest half hour of the NASA-HTTP July 1995
trace into both systems — cumulative to Prometheus, φ-deltas to Datadog — and compares
every candidate bucket by bucket against **Datadog**, not against `datadog_eval`.

| scrape grid | `x - x offset Δ` | `increase(x[Δ] anchored)` | `rate(x[Δ])·Δ` |
|---|---|---|---|
| unaligned, evaluated inside | 100% | 100% | 20.2% |
| unaligned, on the bucket edge | 100% | 100% | 20.2% |
| aligned, evaluated inside | 100% | 100% | 19.0% |
| **aligned, on the bucket edge** | **27.4%** | **27.4%** | 16.7% |

Two results. Both exact forms are exact against the vendor on real traffic. And the
closure convention of E2 is **invisible on an unaligned scrape** — off a boundary the two
half-open intervals hold the same samples — while on an aligned one it decides 72.6% of
buckets. That is why the wrong convention survived every earlier measurement: real
scrapes are jittered, and the synthetic workloads are the aligned case.

Not covered: counter resets. A request counter does not restart in half an hour, so
`COUNTER_RESET` still rests on the synthetic restart workloads.

Obtaining the trace (20 MB, not committed):

```
curl -o NASA_access_log_Jul95.gz ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz
```

## What each outcome changes

| Result | Changes |
|---|---|
| ρ = zero (**expected** — the source says so) | §3.1 Prop 3 stands as written; `COUNTER_RESET` stays a real caveat on the cumulative difference only. Confirms the source reading end to end |
| ρ = clamp | The backend disagrees with the OpenMetrics scraper's own flag, i.e. the flag is not reaching the aggregator. δ = 0 across the reset means the cumulative difference overstates by exactly the post-reset value; `COUNTER_RESET` widens from a caveat to a bound |
| ρ = drop, or a raw difference | Contradicts the source at a pinned commit. Re-read before believing it, then record the agent version as the discriminating variable |
| ~~right-closed~~ | not what happens |
| **left-closed (observed)** | Σ_Δ's half-open interval flips, and the evaluation instant moves off the bucket boundary to just inside it — a *full* interval from where the paper had it, not the half-bucket guessed here. No measured figure changed |
| bare API query behaves as `.as_count()` | `METRIC_TYPE_UNKNOWN` narrows to genuinely ambiguous gauges |
| interval table confirmed | `ROLLUP_DEFAULT` cites a documented table; `DEFAULT_ROLLUP = 60.0` in notebook 03 becomes range-dependent and κ is recomputed per range |
| `*` matches the empty string | §6 stands as written: `.+` has no glob, 238 refused occurrences |
| `*` requires a character | `STAR_MATCHES_EMPTY = False` in `regular.py`; §6's table is recomputed by notebook 04, with `.*`-shaped patterns moving into the refusals and `.+`-shaped ones out |
| **`top()` matches the documented rule (observed)** | §7's expressibility claim has two measured halves instead of one, and §8's "needs the dashboard's time range" stops being an assertion: narrowing the window moves the selection |
| `top()` ranks by something else | 139 occurrences move from unimplemented work to the inexpressible column, which inverts §7's split and §8's comparison |

## Threats to validity

- **Agent version.** Pin the agent image by digest and record it in every cached
  response, as `verify_anchored.py` does for Prometheus. Reset handling is the kind of
  thing that changes between minor versions.
- **One path is not all paths.** OTLP ingest sends raw monotonic values by default,
  which is a different φ again. Rig A answers for the OpenMetrics check only; §7 should
  say so rather than generalising to "Datadog".
- **Trial tier differences.** Confirm that rollup behaviour and retention on a trial
  match a paid account; if they differ, the measured interval table may not generalise.
- **Clock skew.** The agent timestamps samples itself. E2 avoids this by submitting
  timestamps directly; E1 should record both the exposition schedule and the timestamps
  Datadog reports back, and compare.
- **Backfill rejection.** Points older than ~1 h are refused, so all of this is
  wall-clock-bound. Budget an afternoon, not a batch job.

## Code this needs

- `src/qlport/harness/expose.py` — the scripted counter endpoint. Pure and testable
  without an account; the value schedule is a function of time.
- `src/qlport/harness/datadog.py` — **written.** Submit points, run queries, cache raw
  JSON; mirrors `harness/prometheus.py`, with a `pointlist` that keys by timestamp and
  keeps null buckets, since in E2 the nulls are the measurement. Covered offline by
  `tests/test_harness_datadog.py`, which pins the metric-type enum and the
  seconds/milliseconds asymmetry.
- `scripts/trial_checks.py` — **written.** One subcommand each, writing
  `data/raw/datadog/<experiment>.json` with the site, UTC collection time, submitted
  payloads, every raw response and the interpretation. `--dry-run` prints what it would
  send and touches neither the network nor the keys, so the payload shapes are reviewable
  without an account. E2–E4 run; `e1` exits with a pointer, since Rig A is not built and
  ρ is settled.
- A section in notebook 02 (for ρ) and notebook 05 (for the rest) reading only the
  cache, plus regression tests over it, exactly as `tests/test_harness.py` does for the
  Prometheus verification.

The first two are testable offline and could be written before the account exists.
