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

from qlport.harness.datadog import Credentials, credentials, query  # noqa: E402
from qlport.regular import TAG_ALPHABET  # noqa: E402

OUT = ROOT / "data" / "raw" / "datadog" / "e17_scope_length.json"

METRIC = "qlport.e12.wild"     # an existing rig metric; acceptance is what is measured
KEY = "probe"
LADDER = [1, 41, 100, 256, 512, 1000, 1681, 2000, 4096]
WINDOW = 300.0                 # a five-minute range: the smallest useful query


def values(n: int) -> list[str]:
    """`n` distinct tag values of the shape `5..` expands to.

    Three characters while that suffices, which covers the $41^{2}$ the bound admits;
    a fourth is added only to probe past it, where the question is the length of the
    query rather than which pattern produced it.
    """
    import itertools

    sigma = sorted(TAG_ALPHABET)
    for width in (2, 3, 4):
        if len(sigma) ** width >= n:
            return ["5" + "".join(t) for t in
                    itertools.islice(itertools.product(sigma, repeat=width), n)]
    raise ValueError(f"cannot make {n} values")


def scope_query(n: int) -> str:
    """The translated form §6 now claims for a two-free-position pattern."""
    return f"avg:{METRIC}{{{KEY} IN ({', '.join(values(n))})}}"


def url_length(creds: Credentials | None, n: int, start: float, end: float) -> int:
    params = urllib.parse.urlencode(
        {"query": scope_query(n), "from": int(start), "to": int(end)})
    base = creds.base_url if creds else "https://api.datadoghq.com"
    return len(f"{base}/api/v1/query?{params}")


def attempt(creds: Credentials, n: int, start: float, end: float) -> dict:
    """Ask for `n` literals. An accepted scope is the measurement, not its series."""
    t0 = time.time()
    row: dict = {"n": n, "url_bytes": url_length(creds, n, start, end)}
    try:
        raw = query(creds, scope_query(n), start, end, timeout=90.0)
    except RuntimeError as exc:                 # an HTTP status, from `query`
        row |= {"accepted": False, "failure": "http", "detail": str(exc)[:300]}
    except ValueError as exc:                   # Datadog said status=error
        row |= {"accepted": False, "failure": "query-error", "detail": str(exc)[:300]}
    except Exception as exc:                    # a transport failure is not an answer
        row |= {"accepted": None, "failure": type(exc).__name__, "detail": str(exc)[:300]}
    else:
        row |= {"accepted": True, "series": len(raw.get("series") or []),
                "dd_status": raw.get("status")}
    row["seconds"] = round(time.time() - t0, 2)
    return row


def controls(creds: Credentials, metric: str, key: str, value: str,
             start: float, end: float) -> list[dict]:
    """The checks without which "accepted" means nothing.

    The ladder on its own cannot distinguish a long scope that works from a long scope
    the endpoint ignores: the rig metrics carry no points in a five-minute window, so
    every rung returned `status: ok` with no series, which is also what silent truncation
    would return. E5 added this discipline after E2's first run could not tell a glob
    that selects nothing from a point that never arrived, and it applies here.

    Six rows. Two bare scopes establish that the metric has the value and not another.
    Two long lists establish that a list of $41^{2}$ is parsed and applied --- one
    containing the present value must return a series, one excluding it must not, which
    together rule out both truncation and the list being dropped. Two malformed scopes
    establish that the endpoint refuses anything, without which every acceptance above
    is unfalsifiable.
    """
    import itertools

    sigma = sorted(TAG_ALPHABET)
    junk = ["5" + "".join(t) for t in
            itertools.islice(itertools.product(sigma, repeat=2), len(TAG_ALPHABET) ** 2)]
    if value in junk:                      # the exclusion control depends on this
        raise ValueError(f"control value {value!r} collides with the filler list")

    cases = [
        ("bare-scope-present", f"avg:{metric}{{{key}:{value}}}", "series"),
        ("bare-scope-absent", f"avg:{metric}{{{key}:zzz}}", "empty"),
        ("long-list-including",
         f"avg:{metric}{{{key} IN ({', '.join(junk + [value])})}}", "series"),
        ("long-list-excluding",
         f"avg:{metric}{{{key} IN ({', '.join(junk)})}}", "empty"),
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sizes and URL lengths, send nothing")
    # A run-scoped rig metric that still carries a point, and a tag value it has. The
    # default is the one E17 was first run against; `--control-metric` is there because
    # these names carry a run id and will age out of the account's retention.
    ap.add_argument("--control-metric", default="qlport.ar.r1794863577.cnt")
    ap.add_argument("--control-key", default="probe")
    ap.add_argument("--control-value", default="a")
    ap.add_argument("--control-days", type=float, default=7.0,
                    help="how far back the controls look for the rig's point")
    args = ap.parse_args()

    end = time.time()
    start = end - WINDOW

    if args.dry_run:
        print(f"{'literals':>9}  {'url bytes':>9}")
        for n in LADDER:
            print(f"{n:>9,}  {url_length(None, n, start, end):>9,}")
        print(f"\nthe scope at n=1681 begins: {scope_query(1681)[:88]}...")
        return 0

    creds = credentials()

    print("controls:")
    control_rows = controls(creds, args.control_metric, args.control_key,
                            args.control_value, end - args.control_days * 86400, end)
    if not all(r["holds"] for r in control_rows):
        # Without these the ladder is unfalsifiable, so it is not run and nothing is
        # cached. A stale `--control-metric` is the likely cause: these names carry a
        # run id and age out of retention.
        print("\ncontrols do not hold; the ladder would be unfalsifiable, so it is "
              "not run.\nPass --control-metric for a rig metric that still has a point.")
        return 1

    print("\nladder:")
    rows = [attempt(creds, n, start, end) for n in LADDER]
    for r in rows:
        verdict = {True: "accepted", False: "REFUSED", None: "inconclusive"}[r["accepted"]]
        print(f"  n={r['n']:>5,}  url={r['url_bytes']:>6,}B  {verdict:<12} "
              f"{r.get('failure','')} {str(r.get('detail',''))[:90]}", flush=True)

    # Bisect between the largest accepted and the smallest refused, so the number the
    # paper quotes is a measured boundary rather than a rung on the ladder.
    ok = [r["n"] for r in rows if r["accepted"] is True]
    bad = [r["n"] for r in rows if r["accepted"] is False]
    boundary = None
    if ok and bad:
        lo, hi = max(ok), min(n for n in bad if n > max(ok))
        print(f"\nbisecting between {lo:,} accepted and {hi:,} refused", flush=True)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            r = attempt(creds, mid, start, end)
            rows.append(r)
            print(f"  n={mid:>5,}  url={r['url_bytes']:>6,}B  "
                  f"{'accepted' if r['accepted'] else 'REFUSED'}", flush=True)
            if r["accepted"] is True:
                lo = mid
            elif r["accepted"] is False:
                hi = mid
            else:
                break
        boundary = {"largest_accepted": lo, "smallest_refused": hi}
        print(f"\nboundary: {lo:,} accepted, {hi:,} refused")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric": METRIC, "tag_key": KEY, "window_seconds": WINDOW,
        "alphabet_size": len(TAG_ALPHABET), "max_union_claimed": len(TAG_ALPHABET) ** 2,
        "control_metric": args.control_metric, "controls": control_rows,
        "boundary": boundary, "rows": rows,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
