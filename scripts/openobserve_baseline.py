"""Run OpenObserve's own Datadog->PromQL translator over this paper's corpus.

The comparison §2.3 could not make until now. OpenObserve ships a hosted dashboard
migrator with no API and no source, but its translator is client-side JavaScript, so the
measurement needs no uploads: the bundle is cached under `data/raw/openobserve/` with its
sha256 and retrieval date, turbopack module 27461 is lifted out of it, and `node` drives
the exported `translateDDQuery` over the same normalised queries `translations.csv`
records our own outcome for.

Three things make the comparison fair rather than flattering:

  * Occurrence weighting, the same as $\\kappa$, from `translations.csv`'s `n` column.
  * Multi-query sources are held out. Our emitter returns a list for those and theirs
    returns one string, so a parse check would fail ours for a reason that is not a
    defect. Dropping them costs 117 occurrences and removes the asymmetry.
  * Validity is judged by *this paper's* PromQL grammar, which is the same instrument
    §4 reports and is independent of either translator.

The headline is not that one number beats another. It is that "emits something" and
"emits something that parses" are different measures, and that the gap between them is
where a typed refusal would have gone.

Usage: .venv/bin/python scripts/openobserve_baseline.py [--write-registry]
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "openobserve"
CHUNK = RAW / "f8ce70f9cbeedb16.js"
TRANSLATIONS = ROOT / "data" / "processed" / "translations.csv"
OUT = ROOT / "data" / "processed" / "openobserve.json"
REGISTRY = ROOT / "paper" / "quoted_figures.json"
MODULE_ID = "27461"

# The one module 27461 imports: classname and id helpers that only migrateDashboard
# uses. Stubbed rather than lifted, so nothing outside the translator is executed.
SHIM = """'use strict';
const exported = {};
const e = {
  s(pairs) {
    for (let i = 0; i < pairs.length; i += 2) {
      Object.defineProperty(exported, pairs[i],
        { get: pairs[i + 1], enumerable: true, configurable: true });
    }
  },
  i(id) {
    if (String(id) === '75157') {
      let n = 0;
      return { cn: (...a) => a.filter(Boolean).join(' '),
               generateId: () => 'id' + (++n),
               generateDashboardId: () => 'dash' + (++n) };
    }
    throw new Error('unexpected import ' + id);
  },
};
(e => {
%s
})(e);
const fs = require('fs');
const queries = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const q of queries) {
  let r;
  try { r = exported.translateDDQuery(q, {}); }
  catch (err) { r = { translated: null, confidence: 'threw', notes: [String(err)] }; }
  out.push({ norm: q,
             oo: r && r.translated != null ? r.translated : null,
             conf: r ? r.confidence : 'threw',
             notes: (r && r.notes) || [] });
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
"""


def extract_module() -> str:
    """Slice turbopack module 27461 out of the cached chunk."""
    src = CHUNK.read_text()
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"(\d{4,6}),\s*e=>\{", src)]
    for i, (pos, mid) in enumerate(starts):
        if mid != MODULE_ID:
            continue
        a = src.index("{", pos + len(mid) + 3)
        b = starts[i + 1][0] if i + 1 < len(starts) else len(src)
        seg = src[a:b].rstrip()
        while seg and not seg.endswith("}"):
            seg = seg[:-1]
        return seg[1:-1]
    raise SystemExit(f"module {MODULE_ID} not found in {CHUNK}")


def toplevel_comma(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        elif ch == "," and depth == 0:
            return True
    return False


def main() -> int:
    if not CHUNK.exists():
        raise SystemExit(f"{CHUNK} is missing; see data/raw/openobserve/MANIFEST.json")
    rows = list(csv.DictReader(TRANSLATIONS.open()))
    single = [r for r in rows if not toplevel_comma(r["norm"])]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "run.cjs").write_text(SHIM % extract_module())
        (tmp / "q.json").write_text(json.dumps([r["norm"] for r in single]))
        done = subprocess.run(
            ["node", str(tmp / "run.cjs"), str(tmp / "q.json"), str(tmp / "r.json")],
            capture_output=True, text=True,
        )
        if done.returncode:
            raise SystemExit("node failed:\n" + done.stdout + done.stderr)
        oo = {r["norm"]: r for r in json.loads((tmp / "r.json").read_text())}

    from qlport.parse import parse

    def parses(q: str) -> bool:
        try:
            parse("promql", q)
            return True
        except Exception:
            return False

    def occ(pred) -> int:
        return sum(int(r["n"]) for r in single if pred(r, oo[r["norm"]]))

    def emits(o) -> bool:
        return o["oo"] is not None and not o["oo"].startswith("# TODO")

    n = sum(int(r["n"]) for r in single)
    fig = {
        "oo.corpus": n,
        "oo.held_out_multi": sum(int(r["n"]) for r in rows) - n,
        "oo.emits": occ(lambda r, o: emits(o)),
        "oo.parses": occ(lambda r, o: emits(o) and parses(o["oo"])),
        "oo.ours_emits": occ(lambda r, o: r["outcome"] == "translated"),
        "oo.ours_parses": occ(
            lambda r, o: r["outcome"] == "translated" and parses(r["promql"])),
        "oo.ours_refuses": occ(lambda r, o: r["outcome"] == "bottom"),
        "oo.emits_where_we_refuse": occ(
            lambda r, o: r["outcome"] == "bottom" and emits(o)),
        "oo.dropped_modifier": occ(
            lambda r, o: any("was dropped" in x for x in o["notes"])),
        "oo.invented_window": occ(
            lambda r, o: any("Adjust interval as needed" in x for x in o["notes"])),
        "oo.passthrough": occ(
            lambda r, o: any("passed through as-is" in x for x in o["notes"])),
        "oo.cosmetic_note_only": occ(
            lambda r, o: bool(o["notes"])
            and all("dots converted" in x for x in o["notes"])),
    }
    fig["oo.invalid"] = fig["oo.emits"] - fig["oo.parses"]
    fig["oo.invalid_matchall"] = occ(
        lambda r, o: emits(o) and not parses(o["oo"]) and "{*}" in o["oo"])

    drop = re.compile(r'modifier "([^"]+)" has no PromQL equivalent and was dropped')
    per_mod: dict[str, int] = {}
    for r in single:
        for note in oo[r["norm"]]["notes"]:
            m = drop.search(note)
            if m:
                key = re.sub(r"\(.*\)", "", m.group(1)).lstrip(".")
                per_mod[key] = per_mod.get(key, 0) + int(r["n"])
    fig["oo.drops_total"] = sum(per_mod.values())
    for k, v in per_mod.items():
        fig[f"oo.dropped_{k}"] = v

    for name, v in list(fig.items()):
        if name in ("oo.invented_window", "oo.cosmetic_note_only"):
            fig[name + "_pct"] = round(100 * v / n, 1)
    fig["oo.substantive_note_pct"] = round(
        100 * occ(lambda r, o: bool(o["notes"])
                  and any("dots converted" not in x for x in o["notes"])) / n, 1)

    # Which of our gap codes they emit for anyway.
    by_gap: dict[str, int] = {}
    for r in single:
        if r["outcome"] == "bottom" and emits(oo[r["norm"]]):
            by_gap[r["gap"]] = by_gap.get(r["gap"], 0) + int(r["n"])

    for code, v in by_gap.items():
        fig[f"oo.gap_{code.lower()}"] = v

    OUT.write_text(json.dumps({"figures": fig, "by_gap": by_gap}, indent=1) + "\n")

    def row(label, a, b):
        return (f"{label} & {a:,} & {100 * a / n:.1f}\\% & {b:,} & {100 * b / n:.1f}\\% \\\\")

    table = "\n".join([
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{this paper} & \multicolumn{2}{c}{OpenObserve} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Outcome & Occ. & Share & Occ. & Share \\",
        r"\midrule",
        row("emits an expression", fig["oo.ours_emits"], fig["oo.emits"]),
        row("\\quad of which parses as PromQL", fig["oo.ours_parses"], fig["oo.parses"]),
        row("\\quad of which does not", fig["oo.ours_emits"] - fig["oo.ours_parses"],
            fig["oo.invalid"]),
        row("refuses with a named code", fig["oo.ours_refuses"], 0),
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"
    (ROOT / "paper" / "tables" / "baseline.tex").write_text(table)
    print("wrote paper/tables/baseline.tex")
    for k, v in fig.items():
        print(f"{k:28} {v:6}  {100 * v / n:5.1f}%")
    print("\nemitted anyway, by our gap code:")
    for k, v in sorted(by_gap.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22} {v:4}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")

    if "--write-registry" in sys.argv:
        # Append rather than re-sort: the notebooks own this file's order, and a
        # wholesale reordering would bury the one-line diff this script makes.
        reg = json.loads(REGISTRY.read_text())
        reg.update(fig)
        REGISTRY.write_text(json.dumps(reg, indent=1) + "\n")
        print(f"merged {len(fig)} keys into {REGISTRY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
