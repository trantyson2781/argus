"""Gate for part A of the breakage check (`argus check`; docs/HANDOFF_NEXT_SESSION.md section 0.5).

    HOME=/tmp/cg-home unshare -rn uv run python scripts/check_gate/run_gate.py SOURCE COPY_DIR --python PY [--per-kind 20] [--seed 1]

Copies SOURCE to COPY_DIR (a git repository of its own), runs the suite twice for the tests that already fail or are flaky, records a run of
half of the test files (the PARTIAL recording) and then a full run (the FRESH recording, made on the same code and command the truth comes
from, so its recall is optimistic by construction), then for sampled functions makes one change at a time (a `raise` as the first line, or
every returned value replaced by `None`), runs the WHOLE suite for the truth, runs `argus check` on the same change, and compares: did the
picked tests contain the ones that really fail (recall), did it report them as newly failing (blame recall), did it report a test that does
not really fail (false alarm), how long did it take against the clean suite. Also scored: the picks without any recording (static) and with
the partial recording. A check that could not decide counts against the tool (it picked nothing); only a truth run that cannot be read is
skipped. Never touches SOURCE. EXECUTES the repository's code: use it only on repositories you wrote. The tests get a small environment (no
keys) and `unshare -rn` cuts the network; the disk is NOT isolated; the recording runs with this process's environment, so run the gate
with HOME pointed at an empty folder. Thresholds are in src/argus/record/check_gate.py (written before the first run). Seed 0 may be
used while building; use a different seed for the verdict. NOT measured: a recording that is STALE (made before other edits).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from argus.analysis.code_type import CodeType, classify_file
from argus.explore.compare import compare_refs
from argus.explore.sides import DiffError
from argus.extract import scan
from argus.model.schema import Node, NodeKind
from argus.record.break_check import insert_raise
from argus.record.call_fit_gate import nullify_returns
from argus.record.check import CheckError, check_change, select_tests
from argus.record.check_gate import MAX_SKIP_RATE, MIN_TRUTH, RATIO_CONTINUE, RECALL_CONTINUE, RECALL_STOP, Pooled, Row, pool_rows, verdict
from argus.record.run import record_command
from argus.record.runner import PYTEST_ARGS, UnsafeCopyError, copy_repo, ensure_safe_copy_dir, parse_run, run_pytest, scrubbed_env
from argus.record.store import Recording

CLEAN_TIMEOUT = 1800
GIT_ENV = ("-c", "user.email=gate@example.invalid", "-c", "user.name=gate", "-c", "commit.gpgsign=false")
ARGUS_ROOT = Path(__file__).resolve().parents[2]
Mutation = Callable[[str, int], str]
MUTATIONS: dict[str, Mutation] = {"raise": insert_raise, "return": nullify_returns}


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *GIT_ENV, *args], cwd=root, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout


def full_run(python: str, root: Path, timeout: float, env: dict[str, str]) -> tuple[frozenset[str] | None, float]:
    """(failing tests, seconds) of the whole suite, or (None, seconds) when the run timed out or could not be read."""
    done, seconds = run_pytest(python, root, timeout, PYTEST_ARGS, env)
    failed = parse_run(done.returncode, done.stdout) if done is not None else None
    return (None if failed is None else failed.tests), seconds


def sample_for(kind: str, nodes: list[Node], root: Path, per_kind: int, seed: int) -> tuple[list[tuple[Node, str]], int]:
    """(up to `per_kind` (function, mutated source) pairs, how many functions the change did not apply to), chosen with a fixed seed."""
    shuffled = list(nodes)
    random.Random(f"{seed}:{kind}").shuffle(shuffled)
    found: list[tuple[Node, str]] = []
    refused = 0
    for node in shuffled:
        try:
            found.append((node, MUTATIONS[kind]((root / node.provenance.file).read_text(encoding="utf-8-sig"), node.provenance.line)))
        except (ValueError, SyntaxError, UnicodeDecodeError, OSError):
            refused += 1
            continue
        if len(found) == per_kind:
            break
    return found, refused


def run_one(
    python: str, root: Path, node: Node, kind: str, mutated: str, bad: frozenset[str], clean_seconds: float, partial: Recording | None, env: dict[str, str],
) -> Row:
    path = root / node.provenance.file
    original = path.read_bytes()
    empty = Row(node.qualname, kind, frozenset(), frozenset(), frozenset(), frozenset(), 0.0)
    try:
        path.write_text(mutated, encoding="utf-8")
        failing, _ = full_run(python, root, 4 * clean_seconds, env)
        if failing is None:
            return _row(empty, skipped=True, reason="the full run on the changed code could not be read")
        truth = failing - bad
        begun = time.monotonic()
        try:
            result = check_change(root, "HEAD", python, max(120.0, 2 * clean_seconds))
        except (CheckError, OSError) as err:
            return _row(empty, truth=truth, ratio=(time.monotonic() - begun) / clean_seconds, reason=f"the check failed: {err}")
        seconds = time.monotonic() - begun
        view = compare_refs(root, "HEAD")
        model = scan(root)
        static = select_tests(view, model, None, root)
        halved = select_tests(view, model, partial, root) if partial is not None else static
    except DiffError as err:
        return _row(empty, skipped=True, reason=f"the comparison failed: {err}")
    finally:
        path.write_bytes(original)
    if result.unreadable:  # the tool did not decide: it picked nothing and reported nothing, which counts against it
        return _row(empty, truth=truth, static=frozenset(s.test for s in static.tests), partial=frozenset(s.test for s in halved.tests), ratio=seconds / clean_seconds, reason=result.unreadable)
    return Row(
        node.qualname, kind, truth, frozenset(s.test for s in result.selection.tests), frozenset(s.test for s in static.tests), frozenset(result.newly_failing),
        seconds / clean_seconds, False, frozenset(s.test for s in halved.tests),
    )


def _fields(row: Row) -> dict[str, object]:
    return {name: getattr(row, name) for name in Row.__slots__}


def _row(base: Row, **changes: object) -> Row:
    return Row(**{**_fields(base), **changes})


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def line_for(label: str, pooled: Pooled) -> str:
    return (
        f"{label:<8}{pooled.runs:>5}{pooled.skipped:>8}{pooled.truth_total:>7}{_fmt(pooled.recall):>8}{_fmt(pooled.blame_recall):>7}{_fmt(pooled.static_recall):>8}"
        f"{_fmt(pooled.partial_recall):>9}{pooled.false_newly_failing:>7}{pooled.partly_missed_changes:>8}{_fmt(pooled.median_ratio):>7}{pooled.untested_rows:>9}  {verdict(pooled)}"
    )


def report(rows: list[Row]) -> list[str]:
    lines = [f"{'change':<8}{'runs':>5}{'skipped':>8}{'truth':>7}{'recall':>8}{'blame':>7}{'static':>8}{'partial':>9}{'false':>7}{'missed':>8}{'time':>7}{'untested':>9}  verdict"]
    for kind in MUTATIONS:
        lines.append(line_for(kind, pool_rows(r for r in rows if r.kind == kind)))
    total = pool_rows(rows)
    lines.append(line_for("ALL", total))
    lines.append("recall: against a FRESH full recording of the same command (optimistic by construction); partial: half the test files recorded; static: no recording")
    lines.append("untested: runs where nothing was selected and nothing really failed (left out of the time); missed: changes with a really failing test that was not picked")
    reasons = sorted({r.reason for r in rows if r.reason})
    lines += [f"  reason: {r}" for r in reasons[:10]]
    if total.static_recall is not None and total.static_recall < RECALL_STOP:
        lines.append("note: without a recording the picks are well under half of the failing tests: `argus check` should then only warn")
    lines.append(
        f"VERDICT: {verdict(total)}  (recall and blame >= {RECALL_CONTINUE}, median time <= {RATIO_CONTINUE} of the clean suite, 0 false alarms, >= {MIN_TRUTH} failing tests; "
        f"stop below recall {RECALL_STOP}; inconclusive above {MAX_SKIP_RATE:.0%} skipped)"
    )
    return lines


def _sha(root: Path) -> dict[str, object]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "uncommitted_changes": None}
    return {"sha": sha, "uncommitted_changes": dirty}


def record_checked(root: Path, python: str, extra: tuple[str, ...] = ()) -> Recording:
    recording = record_command(root, [python, *PYTEST_ARGS, *extra]).recording
    if recording.processes.saved == 0 or recording.processes.errors:
        raise RuntimeError(f"the recording is unusable ({recording.processes.saved} processes saved, {recording.processes.errors} recorder errors)")
    return recording


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("copy_dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--per-kind", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    try:
        ensure_safe_copy_dir(args.source, args.copy_dir)
    except UnsafeCopyError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    copy_repo(args.source, args.copy_dir)
    git(args.copy_dir, "init", "-q", "-b", "main")
    git(args.copy_dir, "add", "-A")
    git(args.copy_dir, "commit", "-q", "-m", "base")
    with tempfile.TemporaryDirectory(prefix="argus-gate-home-") as home:
        env = scrubbed_env(Path(home))
        first, s1 = full_run(args.python, args.copy_dir, CLEAN_TIMEOUT, env)
        second, s2 = full_run(args.python, args.copy_dir, CLEAN_TIMEOUT, env)
        if first is None or second is None:
            print("STOPPED: a clean run of the suite could not be read", file=sys.stderr)
            return 2
        bad, seconds = first | second, max(s1, s2)
        model = scan(args.copy_dir)
        test_files = sorted({n.provenance.file for n in model.nodes if n.kind in (NodeKind.FUNCTION, NodeKind.METHOD) and classify_file(n.provenance.file) is CodeType.TEST})
        partial = record_checked(args.copy_dir, args.python, tuple(test_files[::2]))  # recorded first, so the full recording below is the newest one
        full = record_checked(args.copy_dir, args.python)
        print(f"Check gate: {args.per_kind} functions per change, seed {args.seed}; clean suite {seconds:.0f} s; {len(bad)} tests already failing or flaky; partial recording of {len(test_files[::2])} of {len(test_files)} test files", flush=True)
        nodes = sorted((n for n in model.nodes if n.kind in (NodeKind.FUNCTION, NodeKind.METHOD) and classify_file(n.provenance.file) is not CodeType.TEST), key=lambda n: n.id)
        rows: list[Row] = []
        refused: dict[str, int] = {}
        for kind in MUTATIONS:
            sample, refused[kind] = sample_for(kind, nodes, args.copy_dir, args.per_kind, args.seed)
            for node, mutated in sample:
                rows.append(run_one(args.python, args.copy_dir, node, kind, mutated, bad, seconds, partial, env))
                if git(args.copy_dir, "status", "--porcelain", "--", "*.py").strip():
                    raise RuntimeError(f"a source file was left changed after {kind} on {node.qualname}")
            print(f"  done: {kind} ({len(sample)} tried, {refused[kind]} functions the change did not apply to)", flush=True)
    lines = report(rows)
    print(*lines, sep="\n")
    if args.out:
        version = subprocess.run([args.python, "--version"], capture_output=True, text=True, check=False)
        config = {
            "started": datetime.now(UTC).isoformat(timespec="seconds"), "source": str(args.source), "source_git": _sha(args.source), "argus_git": _sha(ARGUS_ROOT),
            "python": args.python, "python_version": (version.stdout or version.stderr).strip(), "platform": platform.platform(), "seed": args.seed,
            "per_kind": args.per_kind, "clean_seconds": round(seconds, 1), "already_failing_or_flaky": sorted(bad), "pytest_args": list(PYTEST_ARGS),
            "refused_per_kind": refused, "test_files": len(test_files), "partial_recording_files": len(test_files[::2]),
            "recordings": {"full": {"exit_code": full.exit_code, "processes": asdict(full.processes)}, "partial": {"exit_code": partial.exit_code, "processes": asdict(partial.processes)}},
            "thresholds": {"recall": RECALL_CONTINUE, "stop": RECALL_STOP, "ratio": RATIO_CONTINUE, "max_skip_rate": MAX_SKIP_RATE, "min_truth": MIN_TRUTH},
        }
        sets = ("truth", "selected", "static", "newly_failing", "partial")
        args.out.write_text(json.dumps({"config": config, "report": lines, "rows": [{**_fields(r), **{k: sorted(getattr(r, k)) for k in sets}} for r in rows]}, indent=1))
        print(f"settings and rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
