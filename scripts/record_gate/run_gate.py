"""Record-mode gate (backlog item 20): overhead, and how many unresolved call sites gain an observed target.

    uv run python scripts/record_gate/run_gate.py ROOT [--repeat 3] [--sample 30] [--seed 0] -- COMMAND...

Runs COMMAND in ROOT plain and under `argus record` (REPEAT times each, output hidden), then scans ROOT and measures the last recording.
EXECUTES ROOT's code: use it only on repositories you wrote. The sample lines are for checking by hand: open the line and decide whether
the observed target is the function that line really called.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from argus.extract import scan
from argus.record.gate import gain_report, sample_gains
from argus.record.run import record_command

OVERHEAD_LIMIT = 2.0


@contextlib.contextmanager
def hidden_output() -> Iterator[None]:
    """Send this process's stdout and stderr, and its children's, to /dev/null."""
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, null):
            os.close(fd)


def plain_run(root: Path, argv: list[str]) -> tuple[float, int]:
    begun = time.monotonic()
    code = subprocess.run(argv, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode
    return time.monotonic() - begun, code if code >= 0 else 128 - code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    if "--" not in sys.argv:
        parser.error("give the command after --")
    cut = sys.argv.index("--")
    args = parser.parse_args(sys.argv[1:cut])
    argv = sys.argv[cut + 1 :]
    root = args.root.resolve()
    if not argv:
        parser.error("give the command after --")

    plain_run(root, argv)  # warm-up, discarded: the first run pays for cold caches and would flatter the recorded runs
    plain, plain_codes, recorded, recorded_codes = [], [], [], []
    with hidden_output():
        for _ in range(args.repeat):  # alternate, so drift in the machine's speed hits both sides alike
            seconds, code = plain_run(root, argv)
            plain.append(seconds)
            plain_codes.append(code)
            result = record_command(root, argv)
            recorded.append(result.recording.seconds)
            recorded_codes.append(result.recording.exit_code)
    rec = result.recording
    ratio = statistics.median(recorded) / statistics.median(plain)
    same_outcome = set(plain_codes) == set(recorded_codes) and len(set(plain_codes)) == 1

    report = gain_report(scan(root), root, rec)
    share = lambda n, d: f"{100 * n / d:.1f}%" if d else "n/a"  # noqa: E731
    matched = sum(1 for f in rec.functions if f.node_id)
    print(f"Command: {' '.join(argv)}   (exit code {rec.exit_code})")
    print(f"Plain:    median {statistics.median(plain):.2f} s of {[round(s, 2) for s in plain]}")
    print(f"Recorded: median {statistics.median(recorded):.2f} s of {[round(s, 2) for s in recorded]}   ratio {ratio:.2f}   gate < {OVERHEAD_LIMIT}: {'PASS' if ratio < OVERHEAD_LIMIT else 'FAIL'}")
    if not same_outcome:
        print(f"!! INVALID: exit codes differ or vary (plain {plain_codes}, recorded {recorded_codes}); the ratio compares different work.")
    print(f"Processes saved {rec.processes.saved}, lost {rec.processes.lost}, recorder errors {rec.processes.errors}; {len(rec.functions)} functions ran, {matched} matched to the scan")
    print(f"Unresolved call sites: {report.unresolved}; in functions that ran: {report.in_code_that_ran}")
    print(f"Gained an observed target (name-confirmed): {len(report.gains)}; unverified position candidates, not counted: {len(report.candidates)}")
    in_production = sum(1 for g in report.gains if not g.is_test)
    print(f"  = {share(len(report.gains), report.in_code_that_ran)} of sites in code that ran, {share(len(report.gains), report.unresolved)} of all unresolved sites")
    print(f"  production code only: {in_production} of {report.in_production_that_ran} sites in production code that ran = {share(in_production, report.in_production_that_ran)}")
    print(f"  of those, {report.internal_looking} call a name the repo defines (the only ones that can gain by name): {share(in_production, report.internal_looking)} gained")
    print(f"\nSample for the hand check (seed {args.seed}, up to {args.sample}); mark each RIGHT or WRONG:")
    for n, gain in enumerate(sample_gains(report.gains, args.sample, args.seed), 1):
        source = (root / gain.file).read_text(encoding="utf-8", errors="replace").splitlines()[gain.line - 1].strip()
        print(f"{n:>3}. [{'test code' if gain.is_test else 'production'}] {gain.file}:{gain.line}   {source}\n       call {gain.callee_text}(...)  ->  observed {gain.target}")
    print(f"\nPosition candidates (unverified, check by hand; up to {args.sample}):")
    for gain in report.candidates[: args.sample]:
        source = (root / gain.file).read_text(encoding="utf-8", errors="replace").splitlines()[gain.line - 1].strip()
        print(f"     {gain.file}:{gain.line}   {source}\n       call {gain.callee_text}(...)  ->  observed {gain.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
