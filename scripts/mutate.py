#!/usr/bin/env python3
"""Mutation battery: change one operator in a module, run the tests that reach it.

`perturb.py` asks whether a guard notices a planted defect in the paper or in a cache.
This asks the same question of the library: if one operator in `corpus.py` were wrong,
would anything fail? A mutant nothing kills is a line the suite executes and does not
constrain, and a count of those is the only honest summary of what the tests are worth.

    python scripts/mutate.py                       # the whole battery, ~15 min
    python scripts/mutate.py --list                # the mutants, applying none
    python scripts/mutate.py --only=383,384        # just the mutants on those lines
    python scripts/mutate.py --target=src/qlport/metrics.py

The reason this file exists rather than a number in a commit message: "66 of 225 mutants
survive in corpus.py" was carried in ten consecutive messages and nobody but its author
could check it, which is the same objection that moved `perturb.py` into the repository.

How a mutant is run. Running the whole suite for each one is eight hours, so each mutant
runs only the tests that execute its line, which is sound -- a test that never reaches
the line cannot see the change -- and needs the line-to-test map that `--map` builds with
a trace hook over one suite run. The map is cached beside the battery; delete it or pass
`--map` to rebuild. A line no test executes at all is reported as `UNREACHED` rather than
surviving, because the two call for different work: coverage, or an assertion.

Nothing in the working tree is modified. A mutant is written into a throwaway copy of
the package and the tests are run with that copy first on `PYTHONPATH`, which shadows
the editable install -- so an interrupted run leaves a temporary directory behind and
nothing else. The first version of this file did edit the module in place and put it
back in a `finally`, which is what `perturb.py` does for the paper; a power cut during
the control run left `corpus.py` reformatted by `ast.unparse` with every comment
stripped, as a 508-line deletion staged for whoever committed next. A battery that can
do that to a library module is not worth the few lines it saves.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
CACHE = ROOT / ".mutate-cache"

# --- the operators ---------------------------------------------------------------------
# One mutation each, chosen so that every one of them is a change a person could make by
# accident. Deliberately no string-literal mutation: `corpus.py` is mostly regexes and
# feature names, a changed literal there is a different program rather than a slip, and
# the count would be dominated by them.

# What a surviving mutant is not. Eight of `corpus.py`'s 225 survive and one sits on a
# line no test executes, and all nine are equivalent edits or dead code rather than
# gaps. They are recorded here because the alternative is writing assertions whose only
# purpose is to move a number:
#
#   `frozen=True` on `Query` and on `ParseOutcome`. Observable only by attempting an
#   assignment; neither type is a dict key or a set member anywhere in the library, so
#   a test would assert the decorator and not any behaviour.
#
#   `n += 1` in `_depth0_commas` was on this list and is not any more, which is worth
#   keeping as a note on how soft the category is. It looked unobservable -- one
#   caller, reading the result for its truth value only -- and the test written for the
#   clamp beside it asserts the count of an unbalanced term directly, because no
#   well-formed query reaches that branch. That made the magnitude observable and
#   killed the mutant. "Equivalent" means equivalent at the boundary the tests happen
#   to use, not equivalent in the program.
#
#   `term = term[1:]` after `startswith("!")` in `datadog_features`. Enumerate the term
#   shapes and the feature set is the same either way: `!k` is BARE, `!k:v` is EQ and
#   `!k:*` is GLOB whether one leading character is dropped or two, because nothing
#   downstream reads the key.
#
#   The conjunction guarding `TMPL_RANGE`. The loop inside re-tests each bracket for a
#   variable, so the outer `and` is a short-circuit and not a condition; `or` gives the
#   same feature set on every input.
#
#   The `UnexpectedEOF` branch of `parse_outcome`. Not a gap but dead code: Lark's LALR
#   parser reports the end of input as an `UnexpectedToken` whose type is `$END`, which
#   the branch above it already catches. Left in place rather than deleted, because
#   removing a defensive branch from a parser wrapper on the strength of one parser
#   version's behaviour is a larger claim than this round is making.
#
#   `reverse=True` in the aggregation-operator alternation. The ordering is there so
#   that `count` cannot shadow `count_values`, and it does not have to be: the pattern
#   requires a `(` after the name, so a shortest-first alternation matches `count`,
#   fails on the `_` that follows, backtracks and matches `count_values` anyway. The
#   comment claims the ordering is load-bearing and the anchor is what actually carries
#   it. A test pins `count_values` being read whole either way.
#
#   `root=False` on the three recursive calls in `_grafana_panels`. Passing `root=True`
#   there would re-walk `__elements` from a nested model, which duplicates queries only
#   for a dashboard whose nested panel carries its own `__elements` map -- a shape
#   Grafana does not export. The flag is right and the defect it prevents is
#   unreachable, so the test asserts each panel is yielded once and leaves it there.

_COMPARE = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}
_BINOP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv,
    ast.Div: ast.Mult, ast.FloorDiv: ast.Mult, ast.Mod: ast.Mult,
}
_BOOL = {ast.And: ast.Or, ast.Or: ast.And}


def _describe(node: ast.AST, what: str) -> str:
    return f"line {node.lineno}: {what}"


class _Mutation:
    """One edit, as the tree surgery that makes it and the tree surgery that undoes it."""

    def __init__(self, lineno: int, what: str, apply, undo):
        self.lineno, self.what, self._apply, self._undo = lineno, what, apply, undo

    def __str__(self) -> str:
        return f"line {self.lineno}: {self.what}"


def _parents(tree: ast.Module) -> dict[int, tuple[ast.AST, str, int | None]]:
    """Where each node sits in its parent, so a node can be replaced by another.

    Keyed on `id`, which is safe because the tree is alive for as long as the map is.
    Needed only for `not` removal: every other operator here is mutated in place, and
    taking a `not` off means putting the operand where the `UnaryOp` was.
    """
    out: dict[int, tuple[ast.AST, str, int | None]] = {}
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        out[id(item)] = (parent, field, i)
            elif isinstance(value, ast.AST):
                out[id(value)] = (parent, field, None)
    return out


def _replace(where: tuple[ast.AST, str, int | None], node: ast.AST) -> None:
    parent, field, index = where
    if index is None:
        setattr(parent, field, node)
    else:
        getattr(parent, field)[index] = node


def mutations(tree: ast.Module) -> list[_Mutation]:
    out: list[_Mutation] = []
    where = _parents(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                swap = _COMPARE.get(type(op))
                if swap is None:
                    continue
                ops, j, new = node.ops, i, swap()
                out.append(_Mutation(
                    node.lineno,
                    f"{type(op).__name__} -> {swap.__name__}",
                    lambda ops=ops, j=j, new=new: ops.__setitem__(j, new),
                    lambda ops=ops, j=j, old=op: ops.__setitem__(j, old)))
        elif isinstance(node, ast.BinOp):
            swap = _BINOP.get(type(node.op))
            if swap is not None:
                out.append(_Mutation(
                    node.lineno, f"{type(node.op).__name__} -> {swap.__name__}",
                    lambda n=node, new=swap(): setattr(n, "op", new),
                    lambda n=node, old=node.op: setattr(n, "op", old)))
        elif isinstance(node, ast.BoolOp):
            swap = _BOOL[type(node.op)]
            out.append(_Mutation(
                node.lineno, f"{type(node.op).__name__} -> {swap.__name__}",
                lambda n=node, new=swap(): setattr(n, "op", new),
                lambda n=node, old=node.op: setattr(n, "op", old)))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            seat = where.get(id(node))
            if seat is not None:
                out.append(_Mutation(
                    node.lineno, "drop `not`",
                    lambda s=seat, n=node: _replace(s, n.operand),
                    lambda s=seat, n=node: _replace(s, n)))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                out.append(_Mutation(
                    node.lineno, f"{node.value} -> {not node.value}",
                    lambda n=node: setattr(n, "value", not n.value),
                    lambda n=node, old=node.value: setattr(n, "value", old)))
            elif isinstance(node.value, int) and not isinstance(node.value, bool):
                out.append(_Mutation(
                    node.lineno, f"{node.value} -> {node.value + 1}",
                    lambda n=node: setattr(n, "value", n.value + 1),
                    lambda n=node, old=node.value: setattr(n, "value", old)))
    return out


def _unparse(tree: ast.Module) -> str:
    return ast.unparse(tree)


def statement_spans(tree: ast.Module) -> dict[int, tuple[int, int]]:
    """Each mutable line, mapped to the line span of the statement that holds it.

    A trace hook reports a multi-line statement at its first line, so a mutant whose
    own line is a continuation -- the second half of a wrapped condition, a default in
    a signature that spans lines -- looks like a line no test executes. Looking its
    tests up by the whole statement's span fixes that. The alternative, a verdict of
    UNREACHED on a line that is covered, reads as a coverage gap that is not there,
    and the first run of this battery reported four.
    """
    spans: dict[int, tuple[int, int]] = {}
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.stmt):
            continue
        lo, hi = stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno
        for node in ast.walk(stmt):
            line = getattr(node, "lineno", None)
            if line is None:
                continue
            # Innermost wins: a nested statement's span is tighter than its parent's.
            old = spans.get(line)
            if old is None or (hi - lo) <= (old[1] - old[0]):
                spans[line] = (lo, hi)
    return spans


# --- which tests reach which line ------------------------------------------------------

_PLUGIN = '''
import json, os, sys
import pytest

TARGET = os.environ["MUTATE_TARGET"]
CURRENT = [None]
HITS = {}

def _line(frame, event, arg):
    if event == "line" and CURRENT[0]:
        HITS.setdefault(frame.f_lineno, set()).add(CURRENT[0])
    return _line

def _trace(frame, event, arg):
    if frame.f_code.co_filename != TARGET:
        return None
    return _line

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    CURRENT[0] = item.nodeid
    sys.settrace(_trace)
    try:
        yield
    finally:
        sys.settrace(None)
        CURRENT[0] = None

def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["MUTATE_MAP"], "w") as fh:
        json.dump({str(k): sorted(v) for k, v in HITS.items()}, fh)
'''

# Lines that run when the module is imported -- the regex compilations, the frozensets,
# the module-level tables. No test executes them, because the import happens during
# collection, so the line map says nothing about them and reporting them as unreached
# would be false: a mutant there is live for every test in the suite. They get the
# fallback list below instead.
FALLBACK = ["tests/test_corpus.py", "tests/test_translate.py", "tests/test_regular.py",
            "tests/test_shapes.py", "tests/test_ir_shape.py",
            "tests/test_differential_cache.py", "tests/test_paper_tables.py",
            "tests/test_paper_figures.py", "tests/test_harness.py"]


def import_lines(target: Path, module: str) -> set[int]:
    """Which lines of `target` run on `import module`, traced in a fresh interpreter."""
    probe = (
        "import sys, json\n"
        f"TARGET = {str(target)!r}\n"
        "seen = set()\n"
        "def _line(f, e, a):\n"
        "    if e == 'line':\n"
        "        seen.add(f.f_lineno)\n"
        "    return _line\n"
        "def _trace(f, e, a):\n"
        "    return _line if f.f_code.co_filename == TARGET else None\n"
        "sys.settrace(_trace)\n"
        f"import {module}\n"
        "sys.settrace(None)\n"
        "print(json.dumps(sorted(seen)))\n")
    done = subprocess.run([PY, "-c", probe], cwd=ROOT, capture_output=True, text=True)
    done.check_returncode()
    return set(json.loads(done.stdout.strip().splitlines()[-1]))


def build_map(target: Path, out: Path) -> dict[int, list[str]]:
    """One suite run with a line trace over `target`, as {line: [nodeid, ...]}."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "mutmap.py").write_text(_PLUGIN)
        env = {**_env(), "MUTATE_TARGET": str(target), "MUTATE_MAP": str(out),
               "PYTHONPATH": tmp}
        print(f"tracing {target.relative_to(ROOT)} over one suite run "
              f"(slow: the trace hook costs a few minutes)", flush=True)
        done = subprocess.run([PY, "-m", "pytest", "tests/", "-q", "--no-header",
                               "-p", "mutmap", "-p", "no:randomly"],
                              cwd=ROOT, capture_output=True, text=True, env=env)
        tail = done.stdout.strip().splitlines()[-1:] or ["(no output)"]
        print(f"  {tail[0]}", flush=True)
    return {int(k): v for k, v in json.loads(out.read_text()).items()}


def _env() -> dict:
    import os
    return dict(os.environ)


# --- running ---------------------------------------------------------------------------

def shadow(target: Path, tmp: Path) -> Path:
    """A throwaway copy of the package, and the path the mutant is written to.

    A copy of the whole package under `tmp`, put ahead of `src` on the path, shadows
    the real module so the tracked tree is never written to. A test that reads the
    module's *source* off disk still reads the real file, which is harmless: such a
    test cannot see a mutation anyway, and the two controls below bound what this
    arrangement can get wrong in either direction.

    Not via the `PYTHONPATH` environment variable, which does not work here and looked
    as though it did. `pyproject.toml` sets pytest's own `pythonpath = ["src"]`, and
    that plugin inserts its entries at the front of `sys.path` during configuration,
    ahead of anything the environment contributed -- so every mutant ran against the
    real module and the battery reported 43 survivors and no kills before anyone
    noticed the number was impossible.
    """
    package = target.parent              # <ROOT>/src/qlport
    src = package.parent                 # <ROOT>/src
    # A symlink farm that looks like the repository root, with the one package copied.
    # `grammar_audit.py` takes the repository root to be its own grandparent, so a
    # shadow that sits anywhere else sends every `ROOT / "data" / ...` read into a
    # directory that does not exist -- which is what the control caught, reported as
    # the reformatting breaking a test when it was the layout.
    (tmp / src.name).mkdir(parents=True, exist_ok=True)
    for entry in ROOT.iterdir():
        if entry.name != src.name:
            (tmp / entry.name).symlink_to(entry)
    for entry in src.iterdir():
        if entry.name != package.name:
            (tmp / src.name / entry.name).symlink_to(entry)
    shutil.copytree(package, tmp / src.name / package.name, dirs_exist_ok=True)
    return tmp / src.name / package.name / target.name


# Long enough for the fallback list, which runs nine modules and takes about a hundred
# seconds, and short enough that a mutant which does not terminate costs five minutes
# rather than the afternoon it cost once. Such a mutant exists in `corpus.py`: the
# escape decoder advances by `2 + width`, and the `Sub` mutation of that makes the
# increment zero for a two-digit escape, so `unquote_promql` spins. A program that does
# not terminate is a detected defect -- the suite would never go green on it -- so a
# timeout counts as a kill and says so, rather than being reported as a survivor.
TIMEOUT = 300.0


def kills(tests: list[str], shadow_root: Path) -> tuple[bool, str]:
    """Do any of these tests fail against the shadowed module? The first failure, if so."""
    try:
        done = subprocess.run([PY, "-m", "pytest", "-q", "--no-header", "-x",
                               "-p", "no:randomly", "-o", f"pythonpath={shadow_root}",
                               *tests],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return True, f"(no result in {TIMEOUT:.0f}s)"
    if done.returncode == 0:
        return False, ""
    for line in done.stdout.splitlines():
        if line.startswith("FAILED") and "::" in line:
            return True, line.split("::", 1)[1].split(" ")[0]
    # A crash rather than an assertion is still a kill, and worth naming as one.
    return True, "(error)"


def _record(target: Path, total: int, killed: int, survived: list, unreached: list,
            *, done: int) -> None:
    """The verdicts so far, on disk, so an interrupted run is still a measurement."""
    (CACHE / (target.name + ".result.json")).write_text(json.dumps({
        "total": total, "judged": done, "killed": killed,
        "survived": [str(m) for m in survived],
        "unreached": [str(m) for m in unreached],
    }, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="src/qlport/corpus.py")
    ap.add_argument("--only", default=None,
                    help="comma-separated line numbers to mutate")
    ap.add_argument("--list", action="store_true", help="print the mutants and stop")
    ap.add_argument("--map", action="store_true", help="rebuild the line-to-test map")
    args = ap.parse_args(argv)

    target = (ROOT / args.target).resolve()
    source = target.read_text()
    tree = ast.parse(source)
    muts = mutations(tree)
    only = {int(n) for n in args.only.split(",")} if args.only else None
    if only:
        muts = [m for m in muts if m.lineno in only]

    if args.list:
        for m in muts:
            print(m)
        print(f"\n{len(muts)} mutants")
        return 0

    CACHE.mkdir(exist_ok=True)
    mapfile = CACHE / (target.name + ".map.json")
    if args.map or not mapfile.exists():
        line_tests = build_map(target, mapfile)
    else:
        line_tests = {int(k): v for k, v in json.loads(mapfile.read_text()).items()}
        print(f"line map: {mapfile.relative_to(ROOT)} "
              f"({len(line_tests)} lines, --map to rebuild)", flush=True)

    module = "qlport." + target.stem
    at_import = import_lines(target, module)
    spans = statement_spans(tree)
    print(f"{len(at_import)} lines of {target.name} run at import; mutants there get "
          f"the fallback list, not a reachability verdict", flush=True)

    def reaching(lineno: int) -> list[str]:
        lo, hi = spans.get(lineno, (lineno, lineno))
        found: set[str] = set()
        for line in range(lo, hi + 1):
            found.update(line_tests.get(line) or [])
        return sorted(found)

    def imported(lineno: int) -> bool:
        lo, hi = spans.get(lineno, (lineno, lineno))
        return any(line in at_import for line in range(lo, hi + 1))

    survived, unreached, killed = [], [], 0
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="mutate-") as tmpdir:
        tmp = Path(tmpdir)
        stand_in = shadow(target, tmp)
        shadow_path = stand_in.parent.parent

        # The control, and the one this battery cannot do without. Every mutant is
        # written out by `ast.unparse`, which reformats the whole module and drops its
        # comments, so the module under test is never the file in git even when nothing
        # is mutated. If the reformatting alone broke a test, every mutant below would
        # be "killed" by that and the score would mean nothing.
        stand_in.write_text(_unparse(tree) + "\n")
        dead, by = kills(FALLBACK, shadow_path)
        if dead:
            print(f"the unmutated module, reformatted by ast.unparse, already fails "
                  f"{by} -- every kill below would be that and not the mutation",
                  file=sys.stderr)
            return 2
        print("control: the reformatted module, unmutated, passes", flush=True)

        # And the control in the other direction, which is the one this battery turned
        # out to need. If the shadow is not on the path, every test runs against the
        # real module, nothing can fail, and the report is "no mutant was killed" --
        # which reads as a suite that constrains nothing rather than as a harness that
        # tests nothing. So poison the shadow with a mutation no suite could miss and
        # require it to die before judging anything.
        poisoned = _unparse(tree).replace(
            "def normalise(lang: str, text: str) -> str:",
            "def normalise(lang: str, text: str) -> str:\n"
            "    raise AssertionError('the mutation battery is live')", 1)
        stand_in.write_text(poisoned + "\n")
        dead, by = kills(["tests/test_corpus.py"], shadow_path)
        if not dead:
            print("the shadowed module is not the one the tests import: a module that "
                  "raises on every call to `normalise` passed tests/test_corpus.py. "
                  "Every verdict below would be SURVIVED and none of them would mean "
                  "anything.", file=sys.stderr)
            return 2
        print(f"control: a module poisoned to raise is caught by {by}", flush=True)

        for i, m in enumerate(muts, 1):
            tests = reaching(m.lineno)
            how = f"{len(tests)} tests reach it"
            if not tests and imported(m.lineno):
                tests, how = FALLBACK, "import-time line, fallback list"
            if not tests:
                unreached.append(m)
                print(f"[{i:3d}/{len(muts)}] UNREACHED {m}", flush=True)
                continue
            m._apply()
            try:
                stand_in.write_text(_unparse(tree) + "\n")
                dead, by = kills(tests, shadow_path)
            finally:
                m._undo()
            if dead:
                killed += 1
                print(f"[{i:3d}/{len(muts)}] killed    {m}  <- {by}", flush=True)
            else:
                survived.append(m)
                print(f"[{i:3d}/{len(muts)}] SURVIVED  {m}  ({how})", flush=True)
            # After each mutant, not at the end. A run is half an hour and the first
            # one was lost to a power cut with 29 verdicts on screen and none on disk.
            _record(target, len(muts), killed, survived, unreached, done=i)

    total = len(muts)
    print(f"\n{total} mutants in {target.name}: {killed} killed, "
          f"{len(survived)} survived, {len(unreached)} on lines no test reaches "
          f"({time.time() - started:.0f}s)")
    if survived:
        print("\nsurvived:")
        for m in survived:
            print(f"  {m}")
    if unreached:
        print("\non lines no test reaches:")
        for m in unreached:
            print(f"  {m}")
    _record(target, total, killed, survived, unreached, done=total)
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
