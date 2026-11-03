"""Break check runner (handoff backlog 24).

    HOME=/tmp/bc-home unshare -rn uv run python scripts/break_check/run_break_check.py SOURCE COPY_DIR [--per-tier 20] [--seed 0] [--python PY]

Copies SOURCE to COPY_DIR, breaks sampled functions there one at a time, runs the tests and prints the report. Never touches SOURCE.
EXECUTES the repository's code: use it only on repositories you wrote. PY is the interpreter that runs the tests (Python >= 3.12, with the
repository's dependencies installed). The tests inherit this process's environment (keys included): `unshare -rn` cuts the network and the
HOME override keeps the tests away from your files; nothing in the code does either for you.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from argus.analysis.code_type import CodeType, classify_file
from argus.explore.blast import build_blast_tree
from argus.extract import scan
from argus.record.break_check import (
    TIERS,
    Candidate,
    FailedTests,
    PrecisionRecall,
    baseline_flaky,
    drop_flaky,
    insert_raise,
    kendall_tau_b,
    permutation_p,
    pooled_precision_recall,
    reaching_tests,
    sample_functions,
    tier_of,
)
from argus.record.run import record_command
from argus.record.runner import (  # noqa: F401  (re-exported: the gate scripts and tests reach these through this module)
    PYTEST_ARGS,
    SKIPPED_DIRS,
    UnsafeCopyError,
    copy_repo,
    ensure_safe_copy_dir,
    parse_run,
    run_pytest,
)

# Written down before the first run (plan of 2026-10-24); the verdict lines below only apply them.
TAU_CONTINUE, P_CONTINUE = 0.3, 0.05
RECALL_CONTINUE, PRECISION_CONTINUE, PRECISION_RENAME = 0.9, 0.8, 0.5
PERMUTATIONS = 9999
CLEAN_TIMEOUT = 1800  # seconds for a clean run
BREAK_TIMEOUT_FACTOR = 10  # a broken run may take this many times as long as the slowest clean run


class CanaryError(Exception):
    """The tests ran no code from the copy, so breaking the copy cannot change anything."""


@dataclass(frozen=True, slots=True)
class FunctionResult:
    candidate: Candidate
    reached: frozenset[str]  # tests the recording says reached it (flaky removed)
    failed: frozenset[str]  # tests that fail when it is broken (flaky removed)
    collection_error: bool  # the function could not be broken or the broken run could not be read: no claim is made from it

    @property
    def break_not_noticed(self) -> bool:
        """The recording says tests reached it, yet breaking it fails none."""
        return bool(self.reached) and not self.failed and not self.collection_error


@dataclass(frozen=True, slots=True)
class BreakReport:
    results: tuple[FunctionResult, ...]
    flaky: frozenset[str]
    tau: float | None  # tier (LOW 0, MED 1, HIGH 2) vs number of failing tests (H1)
    p_value: float | None
    pooled: PrecisionRecall  # H2
    radius_tau: float | None = None  # radius vs number of failing tests, for a finer look than three tiers


@contextmanager
def mutated_file(path: Path, line: int) -> Iterator[None]:
    """Insert the raise into the function at `line` of `path`; on exit, even after an error, restore the bytes exactly."""
    if path.is_symlink():
        raise ValueError(f"{path} is a link: writing to it would change the file it points to")
    original = path.read_bytes()
    broken = insert_raise(original.decode("utf-8-sig"), line).encode("utf-8")  # a ValueError here leaves the file untouched
    try:
        path.write_bytes(broken)
        yield
    finally:
        path.write_bytes(original)


def _clean_failures(python: str, root: Path) -> tuple[frozenset[str], float]:
    done, seconds = run_pytest(python, root, CLEAN_TIMEOUT)
    if done is None:
        raise RuntimeError(f"the clean test run did not finish in {CLEAN_TIMEOUT} s")
    failed = parse_run(done.returncode, done.stdout)
    if failed is None:
        raise RuntimeError(f"the clean test run is unusable (exit code {done.returncode}):\n{done.stdout[-2000:]}")
    return failed.tests, seconds


def _break_one(python: str, copy_dir: Path, cand: Candidate, timeout: float) -> FailedTests | None:
    """The failing tests with `cand` broken, or None when it could not be broken or the run could not be read."""
    try:
        with mutated_file(copy_dir / cand.path, cand.line):
            done, _ = run_pytest(python, copy_dir, timeout)
    except (ValueError, SyntaxError, UnicodeDecodeError):  # a body on the signature line, a file the parser or the decoder refuses
        return None
    return None if done is None else parse_run(done.returncode, done.stdout)


def run_break_check(source: Path, copy_dir: Path, *, per_tier: int, seed: int, python: str = sys.executable) -> BreakReport:
    """Copy `source` to `copy_dir` (refusing an unsafe one first), run the tests twice clean to find flaky ones, record one run,
    scan, sample, and for each sampled function break it, run `python -m pytest -rfE --tb=no -q`, parse and restore."""
    ensure_safe_copy_dir(source, copy_dir)
    copy_repo(source, copy_dir)

    first, seconds_1 = _clean_failures(python, copy_dir)
    second, seconds_2 = _clean_failures(python, copy_dir)
    flaky = baseline_flaky(first, second)
    timeout = BREAK_TIMEOUT_FACTOR * max(seconds_1, seconds_2)
    recording = record_command(copy_dir, [python, *PYTEST_ARGS]).recording
    if recording.exit_code not in (0, 1) or recording.processes.errors:
        raise RuntimeError(f"the recording is unusable (exit code {recording.exit_code}, {recording.processes.errors} recorder errors): 'reached' would be empty")
    if not any(f.node_id and f.calls > 0 and classify_file(f.file) is not CodeType.TEST for f in recording.functions):
        raise CanaryError("no production function of the copy ran: the tests may be importing the original code, not the copy")

    sources = {p.relative_to(copy_dir).as_posix(): p.read_text(encoding="utf-8-sig", errors="replace") for p in copy_dir.rglob("*.py")}
    candidates = sample_functions(tier_of(build_blast_tree(scan(copy_dir), copy_dir), sources), per_tier, seed)

    results = []
    for cand in candidates:
        failed = _break_one(python, copy_dir, cand, timeout)
        results.append(FunctionResult(
            cand, drop_flaky(reaching_tests(recording, cand.node_id), flaky),
            frozenset() if failed is None else drop_flaky(failed.tests, flaky), failed is None,
        ))
    return _report(tuple(results), flaky, seed)


def _report(results: tuple[FunctionResult, ...], flaky: frozenset[str], seed: int) -> BreakReport:
    usable = [r for r in results if not r.collection_error]  # a run nobody could read says nothing about the function
    rank = {tier: len(TIERS) - 1 - i for i, tier in enumerate(TIERS)}  # LOW 0, MED 1, HIGH 2
    failures = [len(r.failed) for r in usable]
    tiers = [rank[r.candidate.tier] for r in usable]
    radii = [r.candidate.radius for r in usable]
    return BreakReport(
        results, flaky, kendall_tau_b(tiers, failures), permutation_p(tiers, failures, seed=seed, n=PERMUTATIONS),
        pooled_precision_recall((r.reached, r.failed) for r in usable), kendall_tau_b(radii, failures),
    )


def exit_code_for(report: BreakReport) -> int:
    """2 when nothing could be read: a run that measured nothing is not a success."""
    return 0 if any(not r.collection_error for r in report.results) else 2


def describe_source(source: Path) -> list[str]:
    """What was copied: the commit and every uncommitted file, because the copy takes the working tree as it is."""
    def git(*args: str) -> str | None:
        done = subprocess.run(["git", *args], cwd=source, capture_output=True, text=True, check=False)
        return done.stdout.rstrip("\n") if done.returncode == 0 else None  # a status line starts with a space: strip only newlines

    commit = git("rev-parse", "--short", "HEAD")
    if commit is None:
        return [f"Source {source}: not a git repository, so the exact code tested is not recorded"]
    dirty = (git("status", "--porcelain") or "").splitlines()
    lines = [f"Source {source} at commit {commit}"]
    if dirty:
        lines.append(f"  {len(dirty)} uncommitted file(s) were copied too: " + ", ".join(line[3:] for line in dirty))
    return lines


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.2f}"


def verdicts(report: BreakReport) -> list[str]:
    """The gate lines. A result that cannot be computed is never a pass."""
    tau, p, pr = report.tau, report.p_value, report.pooled
    if tau is None or p is None:
        h1 = "H1 UNDECIDED: tier or failure counts are all equal, no rank order exists"
    elif tau <= 0:
        h1 = "H1 STOP: higher tiers do not break more tests; the tree must not imply test impact"
    elif tau >= TAU_CONTINUE and p < P_CONTINUE:
        h1 = "H1 CONTINUE: higher tiers break more tests"
    elif tau < TAU_CONTINUE:
        h1 = "H1 FIX: a weak positive link; retune the cut points and rerun"
    else:
        h1 = f"H1 UNCLEAR: tau {tau:.2f} is large enough but p {p:.3f} is not below {P_CONTINUE}; a bigger sample is needed"
    if pr.precision is None or pr.recall is None:
        h2 = "H2 UNDECIDED: precision or recall is undefined"
    elif pr.recall >= RECALL_CONTINUE and pr.precision >= PRECISION_CONTINUE:
        h2 = "H2 CONTINUE: 'run by N tests' matches the tests that fail"
    elif pr.precision < PRECISION_RENAME:
        h2 = "H2 RENAME: reached is not failing; call it 'reached by N tests'"
    else:
        h2 = "H2 FIX: name the cause of the misses (fixtures, threads, swallowed errors) and rerun"
    return [h1, h2]


def print_report(report: BreakReport) -> None:
    usable = [r for r in report.results if not r.collection_error]
    print(f"Functions broken: {len(report.results)}   readable runs: {len(usable)}   flaky or already failing tests left out: {len(report.flaky)}")
    for tier in TIERS:
        failures = [len(r.failed) for r in usable if r.candidate.tier == tier]
        if failures:
            print(f"  {tier:<4} n={len(failures):<3} median failing tests {statistics.median(failures):<5} none failing {sum(f == 0 for f in failures)}")
    print(f"H1 tier vs failing tests: tau-b {_fmt(report.tau)}  p {_fmt(report.p_value)}   (radius vs failing tests: tau-b {_fmt(report.radius_tau)})")
    print(f"H2 pooled over readable runs: precision {_fmt(report.pooled.precision)}  recall {_fmt(report.pooled.recall)}")
    for line in verdicts(report):
        print(line)
    unread = [r for r in report.results if r.collection_error]
    if unread:
        by_tier = ", ".join(f"{t} {sum(r.candidate.tier == t for r in unread)}" for t in TIERS if any(r.candidate.tier == t for r in unread))
        print(f"\nUnreadable runs by tier: {by_tier}. A break that stops the code loading is dropped, and such functions are often the most depended-on,"
              " so the tiers may be biased: " + ", ".join(r.candidate.qualname for r in unread))
    untested = [r for r in usable if r.reached and not r.failed]
    print(f"\nReached by a test yet breaking it fails none ({len(untested)}): " + ", ".join(r.candidate.qualname for r in untested))
    orphans = [r for r in usable if r.candidate.tier == "HIGH" and not r.reached]
    print(f"Important but untested (HIGH tier, no test reaches it) ({len(orphans)}): " + ", ".join(r.candidate.qualname for r in orphans))
    missed = [r for r in usable if r.failed - r.reached]
    print(f"Tests failed that the recording did not list ({len(missed)}): " + ", ".join(f"{r.candidate.qualname} (+{len(r.failed - r.reached)})" for r in missed))
    print("\nEvery function:")
    for r in report.results:
        print(f"  {r.candidate.tier:<4} radius {r.candidate.radius:<4} reached {len(r.reached):<3} failed {len(r.failed):<3} {'UNREADABLE ' if r.collection_error else ''}{r.candidate.qualname}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("copy_dir", type=Path)
    parser.add_argument("--per-tier", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    version = subprocess.run([args.python, "-m", "pytest", "--version"], capture_output=True, text=True, check=False)
    print(f"Break check: {args.per_tier} per tier, seed {args.seed}, python {args.python}, {(version.stdout or version.stderr).strip()}")
    print(*describe_source(args.source), sep="\n")
    try:
        report = run_break_check(args.source, args.copy_dir, per_tier=args.per_tier, seed=args.seed, python=args.python)
    except (UnsafeCopyError, CanaryError, RuntimeError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    print_report(report)
    return exit_code_for(report)


if __name__ == "__main__":
    raise SystemExit(main())
