"""Gate for part B of the breakage check (docs/HANDOFF_NEXT_SESSION.md section 0.5).

    HOME=/tmp/cf-home unshare -rn uv run python scripts/call_fit_gate/run_gate.py SOURCE COPY_DIR --python PY [--per-kind 20] [--seed 1]

Copies SOURCE to COPY_DIR (a git repository of its own), then for sampled functions makes one change at a time (add a required parameter,
rename a parameter, remove a default, delete the function), asks `explore diff` which calls no longer fit, runs the whole test suite, and
compares the flagged call sites with the `TypeError` / `NameError` the tests really raise there. Never touches SOURCE.
EXECUTES the repository's code: use it only on repositories you wrote. PY is the interpreter that runs the tests (with the repository's
dependencies and `coverage` installed). The tests get a small environment (PATH, HOME, LANG: no keys), and `unshare -rn` cuts the network;
the disk is NOT isolated: a test can still read or write any file your user can. The copy follows symbolic links in SOURCE as the files they
point to, so keep links that lead outside SOURCE out of the repository.

Thresholds, written before the first run (handoff section 0.5; the code is in src/argus/record/call_fit_gate.py): each kind of change is
judged on its own, CONTINUE when precision >= 0.95 and recall on linked sites >= 0.8 with 0 FACT claims; STOP when precision < 0.9;
INCONCLUSIVE when more than 20% of its sampled runs had to be skipped. A hand check of 30 flagged sites is listed in the output file.
Seed 0 was used while building the check; use a different seed for the verdict.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from argus.analysis.code_type import CodeType, classify_file
from argus.explore.call_fit import Verdict
from argus.explore.claims import Tier
from argus.explore.compare import compare_refs
from argus.extract import scan
from argus.model.schema import EdgeKind, Node, NodeKind
from argus.record.break_check import failed_tests
from argus.record.call_fit_gate import (
    MAX_SKIP_RATE,
    PRECISION_CONTINUE,
    PRECISION_STOP,
    RECALL_CONTINUE,
    BindingError,
    KindResult,
    Scores,
    Site,
    add_required_parameter,
    binding_errors,
    delete_function,
    expand_statements,
    kind_verdict,
    line_map,
    overall_verdict,
    pool,
    remove_first_default,
    rename_first_parameter,
    score_sites,
)

HERE = Path(__file__).resolve().parent
ARGUS_ROOT = HERE.parents[1]
TRACE_ARGS = ("-B", "-m", "pytest", "-q", "-rfE", "--tb=short", "-p", "no:randomly", "-p", "no:cacheprovider")
TIMEOUT_FACTOR = 10
CLEAN_TIMEOUT = 1800
COVERAGE_JSON_TIMEOUT = 600
HAND_CHECK = 30
TEST_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
GIT_ENV = ("-c", "user.email=gate@example.invalid", "-c", "user.name=gate", "-c", "commit.gpgsign=false")

Mutation = Callable[[str, int], str]
MUTATIONS: dict[str, Mutation] = {
    "add_required": add_required_parameter,
    "rename": rename_first_parameter,
    "remove_default": remove_first_default,
    "delete": delete_function,
}
POOLED_KINDS = ("add_required", "rename", "remove_default")  # scored per call-site line; delete is scored per (file, function name) and stays apart


def _load_break_check():
    spec = importlib.util.spec_from_file_location("run_break_check", HERE.parent / "break_check" / "run_break_check.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_break_check"] = module
    spec.loader.exec_module(module)
    return module


bc = _load_break_check()


@dataclass(frozen=True, slots=True)
class Row:
    kind: str
    function: str
    file: str
    line: int
    flagged: list[Site]
    observed: list[Site]
    linked: list[Site]
    unreached: list[Site]  # flagged, and no test reached the site: nothing can confirm or refute it
    unknown_observed: int  # real failures at linked sites that Argus called UNKNOWN (not checked)
    scores: Scores
    wrong_fact: int
    run_seconds: float


def test_env() -> dict[str, str]:
    return {key: os.environ[key] for key in TEST_ENV_KEYS if key in os.environ}


def run_tests(python: str, root: Path, timeout: float, args: tuple[str, ...] = TRACE_ARGS):
    return bc.run_pytest(python, root, timeout, args, env=test_env())


def git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", *GIT_ENV, *args], cwd=root, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
    return done.stdout


def init_git(root: Path) -> None:
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")


def executed_lines(python: str, root: Path) -> tuple[dict[str, set[int]], set[Site], float]:
    """Lines the clean test run reached ({file: lines, whole statements}), the call-binding errors it already raises, and its duration."""
    data = root / ".gate-coverage"
    done, seconds = run_tests(python, root, CLEAN_TIMEOUT, ("-m", "coverage", "run", f"--data-file={data}", *TRACE_ARGS[1:]))
    if done is None:
        raise RuntimeError("the clean coverage run did not finish")
    if bc.parse_run(done.returncode, done.stdout) is None:
        raise RuntimeError(f"the clean run is unusable (exit code {done.returncode}):\n{done.stdout[-2000:]}")
    out = root / ".gate-coverage.json"
    subprocess.run(
        [python, "-I", "-m", "coverage", "json", f"--data-file={data}", "-o", str(out), "-q"], cwd=root, check=True, capture_output=True,
        stdin=subprocess.DEVNULL, timeout=COVERAGE_JSON_TIMEOUT, env=test_env(),
    )
    lines: dict[str, set[int]] = {}
    for name, info in json.loads(out.read_text())["files"].items():
        path = Path(name).resolve()
        if path.is_relative_to(root.resolve()):
            rel = path.relative_to(root.resolve()).as_posix()
            lines[rel] = expand_statements(path.read_text(encoding="utf-8-sig", errors="replace"), set(info["executed_lines"]))
    return lines, {(e.file, e.line) for e in binding_errors(done.stdout)}, seconds


def functions_ran(lines: dict[str, set[int]], root: Path) -> set[Site]:
    """(file, function name) for every function with at least one executed line in its body. Generous: it may not have reached the call."""
    ran: set[Site] = set()
    for file, hit in lines.items():
        try:
            tree = ast.parse((root / file).read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as err:
            print(f"  note: {file} is not parsed for the delete count ({type(err).__name__})", file=sys.stderr)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any(node.lineno < n <= (node.end_lineno or node.lineno) for n in hit):
                ran.add((file, node.name))
    return ran


def candidates(root: Path) -> list[Node]:
    """Production functions and methods with at least one known caller."""
    model = scan(root)
    called = {e.dst_id for e in model.edges if e.kind is EdgeKind.CALLS}
    return sorted(
        (n for n in model.nodes if n.kind in (NodeKind.FUNCTION, NodeKind.METHOD) and n.id in called and classify_file(n.provenance.file) is not CodeType.TEST),
        key=lambda n: n.id,
    )


def sample_for(kind: str, nodes: list[Node], root: Path, per_kind: int, seed: int) -> tuple[list[tuple[Node, str]], int]:
    """(up to `per_kind` (function, mutated source) pairs, how many functions the change did not apply to), chosen at random with a fixed seed."""
    shuffled = list(nodes)
    random.Random(f"{seed}:{kind}").shuffle(shuffled)
    found: list[tuple[Node, str]] = []
    refused = 0
    for node in shuffled:
        try:
            source = (root / node.provenance.file).read_text(encoding="utf-8-sig")
            found.append((node, MUTATIONS[kind](source, node.provenance.line)))
        except (ValueError, SyntaxError, UnicodeDecodeError, OSError):
            refused += 1
            continue
        if len(found) == per_kind:
            break
    return found, refused


def related(error: BindingError, node: Node) -> bool:
    """Whether a raised error is about calling `node`. A binding TypeError starts with the function's qualified name (Python 3.10+), a
    NameError / AttributeError quotes its name."""
    if error.exc == "TypeError":
        parts = node.qualname.split(".")
        return error.message.partition("() ")[0] in {".".join(parts[i:]) for i in range(len(parts))}
    return f"'{node.name}'" in error.message


def flagged_sites(view, node: Node, kind: str) -> tuple[set[Site], set[Site], int]:
    """(flagged, linked, FACT claims) for `node` in the diff of the mutated copy."""
    if kind == "delete":
        hits = [(g, e) for g in view.interface_callers for e in g.edges if e.callee == node.qualname]

        def site(group, edge) -> Site:
            return edge.claim.evidence[0].file, group.caller.rpartition(".")[2]

        flagged = {site(g, e) for g, e in hits if g.caller_state == "unchanged"}
        return flagged, {site(g, e) for g, e in hits}, sum(e.claim.tier is Tier.FACT for _, e in hits)
    callees = {node.qualname, node.qualname.rpartition(".")[0]} if node.name == "__init__" else {node.qualname}
    fits = [f for f in view.call_fits if f.callee in callees]
    flagged = {(f.file, f.line) for f in fits if f.verdict is Verdict.BREAKS}
    return flagged, {(f.file, f.line) for f in fits}, sum(f.claim.tier is Tier.FACT for f in fits)


def remap(sites: set[Site], file: str, mapping: dict[int, int]) -> set[Site]:
    """`sites` with the lines of `file` moved to where the mutation put them; lines the mutation changed or removed are dropped."""
    return {s for s in sites if s[0] != file} | {(file, mapping[line]) for f, line in sites if f == file and line in mapping}


def run_one(python: str, root: Path, node: Node, kind: str, mutated: str, executed: set[Site], baseline: set[Site], timeout: float) -> Row | None:
    path = root / node.provenance.file
    if path.is_symlink():
        return None
    original = path.read_bytes()
    if kind != "delete":
        mapping = line_map(original.decode("utf-8-sig", errors="replace"), mutated)
        executed, baseline = remap(executed, node.provenance.file, mapping), remap(baseline, node.provenance.file, mapping)
    try:
        path.write_text(mutated, encoding="utf-8")
        view = compare_refs(root, "HEAD", include_tests=True)
        done, seconds = run_tests(python, root, timeout)
    finally:
        path.write_bytes(original)
    if done is None or failed_tests(done.stdout).collection_error or done.returncode not in (0, 1):
        return None
    flagged, linked, wrong_fact = flagged_sites(view, node, kind)
    errors = [e for e in binding_errors(done.stdout) if (kind == "delete" or (e.file, e.line) not in baseline) and related(e, node)]
    by_function = kind == "delete"
    observed = {(e.file, e.func.rpartition(".")[2]) if by_function else (e.file, e.line) for e in errors}
    unknown = 0 if by_function else sum(1 for f in view.call_fits if f.verdict is Verdict.UNKNOWN and (f.file, f.line) in observed)
    scores = score_sites(flagged, observed, linked, executed)
    unreached = flagged - executed - observed
    return Row(
        kind, node.qualname, node.provenance.file, node.provenance.line, sorted(flagged), sorted(observed), sorted(linked), sorted(unreached), unknown,
        scores, wrong_fact, seconds,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def kind_results(rows: list[Row], attempted: dict[str, int]) -> list[KindResult]:
    out = []
    for kind in MUTATIONS:
        group = [r for r in rows if r.kind == kind]
        if attempted.get(kind):
            out.append(KindResult(kind, attempted[kind], attempted[kind] - len(group), pool(r.scores for r in group), sum(r.wrong_fact for r in group)))
    return out


def canary(rows: list[Row]) -> str | None:
    """A change that adds a required parameter must make reached calls fail. None failing at all means the tests may not run the copy."""
    group = [r for r in rows if r.kind == "add_required"]
    if group and sum(r.scores.flagged_executed for r in group) > 0 and sum(r.scores.observed for r in group) == 0:
        return "CANARY: add_required flagged reached calls but no test failed anywhere: the tests may not be running the copy"
    return None


def report(rows: list[Row], attempted: dict[str, int], refused: dict[str, int]) -> list[str]:
    lines = [f"{'kind':<16}{'tried':>6}{'skipped':>8}{'n/a':>5}{'flagged':>8}{'reached':>8}{'confirmed':>10}{'observed':>9}{'linked':>7}{'caught':>7}{'unknown':>8}  precision recall linked-share  verdict"]
    results = kind_results(rows, attempted)
    for result in results:
        group = [r for r in rows if r.kind == result.kind]
        s = [r.scores for r in group]
        lines.append(
            f"{result.kind:<16}{result.attempted:>6}{result.skipped:>8}{refused.get(result.kind, 0):>5}{sum(len(r.flagged) for r in group):>8}{sum(x.flagged_executed for x in s):>8}"
            f"{sum(x.confirmed for x in s):>10}{sum(x.observed for x in s):>9}{sum(x.observed_resolved for x in s):>7}{sum(x.caught for x in s):>7}{sum(r.unknown_observed for r in group):>8}"
            f"  {_fmt(result.pooled.precision):>9} {_fmt(result.pooled.recall):>6} {_fmt(result.pooled.resolved_share):>12}  {kind_verdict(result)}"
        )
    pooled_rows = [r for r in rows if r.kind in POOLED_KINDS]
    total = pool(r.scores for r in pooled_rows)
    wrong_fact = sum(r.wrong_fact for r in rows)
    lines.append(f"line-level kinds pooled: precision {_fmt(total.precision)}  recall {_fmt(total.recall)}  linked-share {_fmt(total.resolved_share)}  FACT claims {wrong_fact}")
    warning = canary(rows)
    if warning:
        lines.append(warning)
    verdict = "INCONCLUSIVE" if warning else overall_verdict(results)
    lines.append(f"VERDICT: {verdict}  (thresholds: precision >= {PRECISION_CONTINUE}, recall >= {RECALL_CONTINUE}, stop below {PRECISION_STOP}, inconclusive above {MAX_SKIP_RATE:.0%} skipped; each kind on its own)")
    return lines


def hand_check(rows: list[Row], root: Path, seed: int) -> list[dict[str, object]]:
    """Flagged sites no test reached (the tests can neither confirm nor refute them): a seeded sample to read by hand."""
    unreached = [(r, site) for r in rows if r.kind != "delete" for site in r.unreached]
    random.Random(seed).shuffle(unreached)
    out = []
    for row, (file, line) in unreached[:HAND_CHECK]:
        try:
            text = (root / file).read_text(encoding="utf-8-sig").splitlines()[line - 1].strip()
        except (OSError, IndexError, UnicodeDecodeError):
            text = "(unreadable)"
        out.append({"mutation": row.kind, "function": row.function, "site": f"{file}:{line}", "line": text, "note": "line numbers are those of the mutated copy"})
    return out


def _sha(root: Path) -> dict[str, object]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "uncommitted_changes": None}
    return {"sha": sha, "uncommitted_changes": dirty}


def run_config(args: argparse.Namespace, clean_seconds: float, baseline: int) -> dict[str, object]:
    version = subprocess.run([args.python, "--version"], capture_output=True, text=True, check=False)
    return {
        "started": datetime.now(UTC).isoformat(timespec="seconds"), "source": str(args.source), "source_git": _sha(args.source), "argus_git": _sha(ARGUS_ROOT),
        "python": args.python, "python_version": (version.stdout or version.stderr).strip(), "platform": platform.platform(), "seed": args.seed,
        "per_kind": args.per_kind, "kinds": args.kinds, "timeout_factor": args.timeout_factor, "clean_seconds": round(clean_seconds, 1), "clean_binding_errors": baseline,
        "thresholds": {"precision_continue": PRECISION_CONTINUE, "recall_continue": RECALL_CONTINUE, "precision_stop": PRECISION_STOP, "max_skip_rate": MAX_SKIP_RATE},
        "test_env_keys": list(TEST_ENV_KEYS), "pythonpath_in_gate_env": os.environ.get("PYTHONPATH"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("copy_dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--per-kind", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-factor", type=float, default=TIMEOUT_FACTOR, help="a mutated run longer than this many times the clean run is stopped and counted as skipped")
    parser.add_argument("--kinds", nargs="+", choices=list(MUTATIONS), default=list(MUTATIONS))
    parser.add_argument("--out", type=Path, default=None, help="write the run settings, every row and the hand-check sample here as JSON")
    args = parser.parse_args()
    try:
        bc.ensure_safe_copy_dir(args.source, args.copy_dir)
    except bc.UnsafeCopyError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    bc.copy_repo(args.source, args.copy_dir)
    init_git(args.copy_dir)
    ran, baseline, seconds = executed_lines(args.python, args.copy_dir)
    functions = functions_ran(ran, args.copy_dir)
    lines_hit = {(f, n) for f, hit in ran.items() for n in hit}
    config = run_config(args, seconds, len(baseline))
    print(f"Call-fit gate: {args.per_kind} functions per change, seed {args.seed}; clean run {seconds:.0f} s; {len(baseline)} binding errors already in the clean run", flush=True)
    nodes = candidates(args.copy_dir)
    rows: list[Row] = []
    attempted: dict[str, int] = {}
    refused: dict[str, int] = {}
    for kind in args.kinds:
        sample, refused[kind] = sample_for(kind, nodes, args.copy_dir, args.per_kind, args.seed)
        attempted[kind] = len(sample)
        for node, mutated in sample:
            row = run_one(args.python, args.copy_dir, node, kind, mutated, functions if kind == "delete" else lines_hit, baseline, args.timeout_factor * seconds)
            if row is not None:
                rows.append(row)
            if git(args.copy_dir, "status", "--porcelain", "--", "*.py").strip():
                raise RuntimeError(f"a source file was left changed after {kind} on {node.qualname}")
        print(f"  done: {kind} ({attempted[kind]} tried, {refused[kind]} functions the change did not apply to)", flush=True)
    lines = report(rows, attempted, refused)
    print(*lines, sep="\n")
    sample = hand_check(rows, args.copy_dir, args.seed)
    if args.out:
        args.out.write_text(json.dumps({"config": config, "report": lines, "rows": [asdict(r) for r in rows], "hand_check": sample}, indent=1))
        print(f"settings, rows and a hand-check sample of {len(sample)} written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
