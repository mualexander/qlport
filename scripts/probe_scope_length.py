#!/usr/bin/env python3
"""How many literals will Datadog accept in one scope's `IN (...)` list? (E17)

§6 raised the union bound to $41^{2}$, which makes `5..` an exact union of $1{,}681$
globs. Exactness is decided, not assumed --- but a translation Datadog will not accept
is not a translation, and the bound now claims $88$ corpus occurrences are expressible
through scopes far longer than anything the corpus itself contains. Datadog documents a
200-character limit on a single tag and says nothing about the length of a query or the
size of an `IN` list, so this is the usual situation: a limit the paper depends on, read
from nothing, which §7 is about.

Read-only. It submits no metrics and creates no custom-metric billing; it asks
`GET /api/v1/query` whether a scope parses, which it answers whether or not the metric
exists or has data. What is being measured is acceptance, not results.

  python scripts/probe_scope_length.py --dry-run     # print the sizes and URL lengths
  python scripts/probe_scope_length.py               # ladder, then bisect the threshold

The URL is the thing to watch. `harness.datadog.query` sends the scope as a urlencoded
GET parameter, so an `IN` list of $1{,}681$ three-character values is about 8.4 kB of
query string -- close enough to the 8 kB header buffer most front ends ship with that
the limit, if there is one, may belong to an HTTP server rather than to Datadog.

Two runs of this were worthless and the second failure is the instructive one. The first
asked only whether a rung raised an HTTP error; every rung came back `status: ok` with no
series, which is also what silent truncation returns, so controls were added. The second
ran those controls against a metric that exists and the ladder against one that does not,
over a different window --- so the controls held, every rung still returned no series, and
the vacuity had been moved rather than closed. The ladder now asks the same metric over
the same window as the controls, and asks twice at every rung: a list containing the
value the metric carries must return a series, and the same list without it must not.
A rung counts as accepted only when both hold, which is what "parsed and applied at this
length" means.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport.harness.datadog import (  # noqa: E402
    RETRYABLE_STATUS, Credentials, QueryHTTPError, credentials, echoed_scopes, query,
)
from qlport.regular import TAG_ALPHABET  # noqa: E402

OUT = ROOT / "data" / "raw" / "datadog" / "e17_scope_length.json"

# Rungs above 1,681 are the ones that say whether the bound is near a limit or nowhere
# near it. 16,384 literals is a query string of about 140 kB, well past anything a
# front end is likely to pass, so a refusal up there is as informative as an acceptance.
LADDER = [1, 41, 100, 256, 512, 1000, 1681, 2000, 4096, 8192, 16384]


def values(n: int, *, avoid: str) -> list[str]:
    """`n` distinct tag values of the shape `5..` expands to, none of them `avoid`.

    Three characters while that suffices, which covers the $41^{2}$ the bound admits;
    a fourth is added only to probe past it, where the question is the length of the
    query rather than which pattern produced it. `avoid` is the value the control
    metric actually carries, excluded here so that the caller controls which of the two
    lists at each rung contains it.
    """
    import itertools

    sigma = sorted(TAG_ALPHABET)
    for width in (2, 3, 4):
        if len(sigma) ** width >= n + 1:
            out = ["5" + "".join(t) for t in itertools.product(sigma, repeat=width)
                   if "5" + "".join(t) != avoid]
            return out[:n]
    raise ValueError(f"cannot make {n} values")


def scope_query(metric: str, key: str, vals: list[str]) -> str:
    """The translated form §6 now claims for a two-free-position pattern."""
    return f"avg:{metric}{{{key} IN ({', '.join(vals)})}}"


def url_length(creds: Credentials | None, q: str, start: float, end: float) -> int:
    params = urllib.parse.urlencode({"query": q, "from": int(start), "to": int(end)})
    base = creds.base_url if creds else "https://api.datadoghq.com"
    return len(f"{base}/api/v1/query?{params}")


ATTEMPTS = 4                   # a read timeout is not an answer; see `_ask`


def _ask(creds: Credentials, q: str, start: float, end: float) -> dict:
    """One query, reduced to what the ladder reads: did it parse, and what came back.

    Retried on a transport failure, which is the one outcome that is not a measurement.
    The first run of the corrected ladder timed out at $256$ literals and succeeded at
    $5{,}096$, so the timeouts are the account's and not the query's -- but an
    inconclusive rung still stops the bisection dead, and a limit bracketed between
    $5{,}096$ and $8{,}192$ because of one is a worse number than the one that is there.
    A parse error is an answer. An HTTP status is one only when it is about the request
    rather than the server: a 431 is the measurement this probe is looking for, and a
    408 "request header timeout" is not. Reading a 408 as a refusal is how the second
    corrected run bisected to a boundary of exactly $4{,}096$ literals -- a power of two,
    which is what made it look like a finding -- after an earlier run had accepted
    $5{,}096$.
    """
    t0, attempts = time.time(), 0
    out: dict = {"url_bytes": url_length(creds, q, start, end)}
    while True:
        attempts += 1
        retry = False
        try:
            raw = query(creds, q, start, end, timeout=120.0)
        except QueryHTTPError as exc:           # an HTTP status, from `query`
            # Only a status about the request itself is an answer. A 408 or a 503 is the
            # server giving up, which is the same non-answer as a socket timeout.
            retry = exc.status in RETRYABLE_STATUS
            out |= {"parsed": None if retry else False, "failure": "http",
                    "status": exc.status, "detail": str(exc)[:300]}
        except ValueError as exc:               # Datadog said status=error
            out |= {"parsed": False, "failure": "query-error", "detail": str(exc)[:300]}
        except Exception as exc:                # a transport failure is not an answer
            retry = True
            out |= {"parsed": None, "failure": type(exc).__name__,
                    "detail": str(exc)[:300]}
        else:
            echoes = echoed_scopes(raw)
            # A retry that succeeds must not leave the failed attempt's verdict behind:
            # `attempt` reads these keys, and a row that says both "accepted" and
            # "TimeoutError" is the kind of record nobody reads twice.
            failed = {k: out.pop(k) for k in ("failure", "status", "detail")
                      if k in out}
            out |= {"parsed": True, "series": len(raw.get("series") or []),
                    "dd_status": raw.get("status"),
                    "echo_bytes": len(echoes[0]) if echoes else 0,
                    "echo_head": echoes[0][:120] if echoes else None}
            if failed:
                out["succeeded_after"] = failed
        if retry and attempts < ATTEMPTS:
            time.sleep(2.0 * attempts)
            continue
        break
    out |= {"attempts": attempts, "seconds": round(time.time() - t0, 2)}
    return out


def attempt(creds: Credentials, metric: str, key: str, value: str, n: int,
            start: float, end: float) -> dict:
    """Ask for `n` literals twice: with the metric's own value in the list, and without.

    Acceptance is the measurement and it takes both halves. A list that parses but is
    truncated, or dropped, returns no series when the value is in it; a list that is
    being ignored returns a series when the value is not. Only a rung that returns a
    series for the first and none for the second was parsed and applied in full.
    """
    filler = values(n - 1 if n > 0 else 0, avoid=value)
    including = scope_query(metric, key, filler + [value])
    excluding = scope_query(metric, key, values(n, avoid=value))
    inc, exc = (_ask(creds, including, start, end), _ask(creds, excluding, start, end))
    if inc["parsed"] is True and exc["parsed"] is True:
        accepted = bool(inc["series"]) and not exc["series"]
        applied = accepted
    elif inc["parsed"] is False or exc["parsed"] is False:
        accepted, applied = False, False
    else:
        accepted, applied = None, None
    return {"n": n, "url_bytes": inc["url_bytes"], "accepted": accepted,
            "applied": applied, "including": inc, "excluding": exc,
            "failure": inc.get("failure") or exc.get("failure"),
            "detail": inc.get("detail") or exc.get("detail")}


def controls(creds: Credentials, metric: str, key: str, value: str,
             start: float, end: float) -> list[dict]:
    """The checks without which "accepted" means nothing.

    Four rows now, not six. Two bare scopes establish that the metric has the value and
    not another, which is what every rung's include/exclude pair is read against. Two
    malformed scopes establish that the endpoint refuses anything at all, without which
    every acceptance below is unfalsifiable. The two long-list rows this function used
    to carry are gone because they are no longer a control: the ladder runs that exact
    pair at every rung, against this same metric and window.
    """
    cases = [
        ("bare-scope-present", f"avg:{metric}{{{key}:{value}}}", "series"),
        ("bare-scope-absent", f"avg:{metric}{{{key}:zzz}}", "empty"),
        ("malformed-unclosed", f"avg:{metric}{{{key} IN ({value}, b}}", "refused"),
        ("malformed-operator", f"avg:{metric}{{{key} QUX ({value})}}", "refused"),
    ]
    rows = []
    for name, q, expect in cases:
        try:
            raw = query(creds, q, start, end, timeout=90.0)
            got = "series" if (raw.get("series") or []) else "empty"
            detail = ""
        except (RuntimeError, ValueError) as exc:
            got, detail = "refused", str(exc)[:200]
        rows.append({"control": name, "expected": expect, "got": got,
                     "holds": got == expect, "query_bytes": len(q), "detail": detail})
        print(f"  {'OK ' if got == expect else '!! '}{name:<22} {got:<8}"
              f"(expected {expect})  query={len(q):,}B", flush=True)
    return rows


def _describe(r: dict) -> str:
    inc, exc = r["including"], r["excluding"]
    if r["accepted"] is None:
        return f"inconclusive  {r.get('failure','')} {str(r.get('detail',''))[:80]}"
    if r["accepted"] is False and (inc["parsed"] is False or exc["parsed"] is False):
        return f"REFUSED       {r.get('failure','')} {str(r.get('detail',''))[:80]}"
    if r["accepted"] is False:
        return (f"NOT APPLIED   including->{inc.get('series')} series, "
                f"excluding->{exc.get('series')} series")
    return "accepted and applied"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sizes and URL lengths, send nothing")
    # A run-scoped rig metric that still carries a point, and a tag value it has. The
    # ladder and the controls both use it: the whole point of this run is that they
    # must. `--metric` is an argument because these names carry a run id and will age
    # out of the account's retention, and a stale one must fail loudly.
    ap.add_argument("--metric", default="qlport.ar.r1794863577.cnt")
    ap.add_argument("--key", default="probe")
    ap.add_argument("--value", default="a")
    ap.add_argument("--days", type=float, default=7.0,
                    help="how far back to look for the rig's point")
    args = ap.parse_args()

    end = time.time()
    start = end - args.days * 86400

    if args.dry_run:
        print(f"{'literals':>9}  {'url bytes':>9}")
        for n in LADDER:
            q = scope_query(args.metric, args.key,
                            values(n - 1, avoid=args.value) + [args.value])
            print(f"{n:>9,}  {url_length(None, q, start, end):>9,}")
        q = scope_query(args.metric, args.key,
                        values(1680, avoid=args.value) + [args.value])
        print(f"\nthe scope at n=1681 begins: {q[:88]}...")
        return 0

    creds = credentials()

    print("controls:")
    control_rows = controls(creds, args.metric, args.key, args.value, start, end)
    if not all(r["holds"] for r in control_rows):
        # Without these the ladder is unfalsifiable, so it is not run and nothing is
        # cached. A stale `--metric` is the likely cause: these names carry a run id
        # and age out of retention.
        print("\ncontrols do not hold; the ladder would be unfalsifiable, so it is "
              "not run.\nPass --metric for a rig metric that still has a point.")
        return 1

    print("\nladder (each rung: one list containing the value, one without):")
    rows = []
    for n in LADDER:
        r = attempt(creds, args.metric, args.key, args.value, n, start, end)
        rows.append(r)
        print(f"  n={r['n']:>6,}  url={r['url_bytes']:>7,}B  {_describe(r)}", flush=True)

    # Bisect between the largest accepted and the smallest refused, so the number the
    # paper quotes is a measured boundary rather than a rung on the ladder.
    ok = [r["n"] for r in rows if r["accepted"] is True]
    bad = [r["n"] for r in rows if r["accepted"] is False]
    boundary = None
    if ok and bad and any(n > max(ok) for n in bad):
        lo, hi = max(ok), min(n for n in bad if n > max(ok))
        print(f"\nbisecting between {lo:,} accepted and {hi:,} refused", flush=True)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            r = attempt(creds, args.metric, args.key, args.value, mid, start, end)
            rows.append(r)
            print(f"  n={mid:>6,}  url={r['url_bytes']:>7,}B  {_describe(r)}", flush=True)
            if r["accepted"] is True:
                lo = mid
            elif r["accepted"] is False:
                hi = mid
            else:
                break
        boundary = {"largest_accepted": lo, "smallest_refused": hi}
        print(f"\nboundary: {lo:,} accepted, {hi:,} refused")
        # A boundary claims every larger list is refused. If a larger one was accepted,
        # the refusal was not about length and the bisection found nothing.
        above = [r["n"] for r in rows if r["accepted"] is True and r["n"] > hi]
        if above:
            boundary = {"largest_accepted": lo, "smallest_refused": hi,
                        "contradicted_by_accepted_above": sorted(above)}
            print(f"!! {max(above):,} literals were accepted above it; the refusal at "
                  f"{hi:,} is not a length limit and this is not a boundary")
    elif ok:
        print(f"\nno refusal up to {max(ok):,} literals "
              f"({max(r['url_bytes'] for r in rows):,} bytes of query string)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric": args.metric, "tag_key": args.key, "tag_value": args.value,
        "window_seconds": args.days * 86400,
        "alphabet_size": len(TAG_ALPHABET), "max_union_claimed": len(TAG_ALPHABET) ** 2,
        "controls": control_rows, "boundary": boundary, "rows": rows,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
