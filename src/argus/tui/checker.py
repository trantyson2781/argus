"""The C key: run `argus check` (the tests that reach the uncommitted change, on throwaway copies) as a child process and say what came of it.

Like the recorder, the TUI never imports the checker: it reads the child's exit code and its `Result` line, and says "could not be read"
whenever the two disagree, so a broken child can never look like a green check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from argus.explore.render_diff import printable
from argus.tui.recorder import ChildJob

MAX_TESTS_SHOWN = 3
NEWLY = re.compile(r"(\d+) newly failing")


def check_argv(root: Path, base: str | None, python: str | None = None) -> list[str]:
    # -P: do not put the current folder first on sys.path, a module of the repository must not shadow one Argus needs
    return [sys.executable, "-P", "-m", "argus.cli", "check", str(root), *([f"--base={base}"] if base else []), *([f"--python={python}"] if python else [])]


def question(root: Path, base: str | None, python: str | None = None, source: str = "") -> str:
    using = f" with {python} ({source})" if python else ""
    return printable(
        f'Run "argus check" in {root}{using}? It compares the working tree with {base or "HEAD"} and runs the tests that reach your change on throwaway '
        "copies: it executes your code.  y = yes, any other key = cancel"
    )


class CheckJob(ChildJob):
    """One `argus check` child process."""

    def __init__(self, root: Path, base: str | None, python: str | None = None) -> None:
        super().__init__(check_argv(root, base, python))


def _newly_failing(lines: list[str], after: int) -> list[str]:
    """The test ids listed under the `newly failing` heading of the report."""
    found: list[str] = []
    listing = False
    for line in lines[after + 1:]:
        if line.startswith("    "):
            if listing:
                found.append(line.strip())
        elif line.startswith("  "):
            listing = "newly failing" in line
    return found


def finished_text(exit_code: int, seconds: float, output: list[str], stopped: bool, base: str = "HEAD") -> str:
    """One sentence for the status line from the child's exit code (0 none newly failing, 1 newly failing, 2 not decided, 3 no test ran) and its report.
    The report must agree with the exit code, and its warning lines (`!`) are counted: a green sentence never hides them."""
    if stopped and exit_code not in (0, 1, 2, 3):  # a child that had finished before the stop arrived keeps its real result
        return f"Check stopped after {seconds:.0f} s."
    lines = [printable(line) for line in output]
    at = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith("Result")), None)
    body = lines[at].partition(":")[2].strip() if at is not None else ""
    newly = int(m.group(1)) if (m := NEWLY.search(body)) else 0
    head = f"Check vs {base}:"
    if at is not None:
        warnings = sum(1 for line in lines[:at] if line.strip().startswith("!"))
        note = f" ({warnings} warning{'' if warnings == 1 else 's'} in the report: run argus check to read {'it' if warnings == 1 else 'them'})" if warnings else ""
        if exit_code == 0 and " ran," in body and not newly:
            return f"{head} {body}{note}"
        if exit_code == 1 and newly:
            ids = _newly_failing(lines, at)
            shown = ", ".join(ids[:MAX_TESTS_SHOWN]) + (f", +{len(ids) - MAX_TESTS_SHOWN} more" if len(ids) > MAX_TESTS_SHOWN else "")
            return f"{head} NEWLY FAILING — {body}" + (f" {shown}" if shown else "") + note
        if exit_code == 1:
            return f"{head} FAILING — {body} (run argus check for the list){note}"
        if exit_code == 2:
            return f"Check not decided: {body.removeprefix('not decided.').strip()}{note}"
        if exit_code == 3:
            return f"{head} no test ran. {body}{note}"
    tail = " | ".join(line.strip() for line in lines[-3:] if line.strip())
    return f"Check could not be read (exit {exit_code}, {seconds:.1f} s; run argus check)" + (f": {tail}" if tail else "")
