"""The R key: run `argus record` on the repository's tests as a separate process and say what came of it.

The TUI never imports the recorder (tests/record/test_guard.py): the recording runs in a child process, in its own process group so
that stopping it also stops pytest and anything pytest started. Output goes to a temporary file, so a chatty run cannot fill a pipe.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from argus.explore.render_diff import printable
from argus.record.store import DIRECTORY, Recording

STOP_WAIT_SECONDS = 5
TAIL_BYTES = 4096
OUTPUT_BYTES = 262144


def test_command(root: Path, python: str | None = None) -> list[str]:
    """What gets recorded: with `python` (the user's choice) or, without, the repository's own Python when it has a `.venv`, else the Python that runs Argus."""
    venv = root / ".venv" / "bin" / "python"
    return [python or (str(venv) if venv.is_file() else sys.executable), "-m", "pytest", "-q"]


def record_argv(root: Path, python: str | None = None) -> list[str]:
    # -P: do not put the current folder first on sys.path, a module of the repository must not shadow one Argus needs
    return [sys.executable, "-P", "-m", "argus.cli", "record", "-C", str(root), "--", *test_command(root, python)]


def question(root: Path, python: str | None = None, source: str = "") -> str:
    where = f" ({source})" if source else ""
    return printable(f'Record "{" ".join(test_command(root, python))}"{where} in {root}? It executes your code (the same as running it yourself).  y = yes, any other key = cancel')


def snapshot(root: Path) -> dict[str, int]:
    """Every recording file with its modification time; {} when there is no folder yet. Other read errors are raised."""
    try:
        return {p.name: p.stat().st_mtime_ns for p in (root / DIRECTORY).glob("*.json")}
    except FileNotFoundError:
        return {}


def fresh(before: dict[str, int], after: dict[str, int]) -> str | None:
    """The newest file that is new or was rewritten since `before` (a rewrite in the same second keeps its name), or None."""
    changed = [name for name, stamp in after.items() if before.get(name) != stamp]
    return max(changed, default=None)


def counts(rec: Recording) -> tuple[int, int]:
    """(functions that ran, of those matched to the scan), counted like `argus record` does."""
    ran = [f for f in rec.functions if f.calls > 0]
    return len(ran), sum(1 for f in ran if f.node_id)


def _ended(exit_code: int) -> str:
    return f"was ended by signal {-exit_code}" if exit_code < 0 else f"exited with {exit_code}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 'es'}"


def finished_text(
    *, saved: str | None, exit_code: int, seconds: float, ran: int, matched: int, tail: list[str], stopped: bool,
    lost: int = 0, errors: int = 0, unreadable: str | None = None,
) -> str:
    if saved and unreadable is not None:  # a file is there, its content is not known: never a green "Recorded"
        return printable(f"A recording file appeared but cannot be read ({unreadable}); the command {_ended(exit_code)} after {seconds:.1f} s.")
    if saved:
        lead = "Recorded: the command exited with 0" if exit_code == 0 else f"Recorded, but the command {_ended(exit_code)}"
        body = f"{lead} after {seconds:.1f} s"
        body += f" · {ran} functions ran, {matched} matched to the scan" if ran else " · no function ran"
        if lost:
            body += f" · ! {_plural(lost, 'process')} did not save (their calls are missing)"
        if errors:
            body += f" · ! the recorder failed {errors} time{'' if errors == 1 else 's'}"
        return body
    if stopped:
        return f"Stopped after {seconds:.0f} s; no new recording was saved."
    shown = printable(" | ".join(line.strip() for line in tail if line.strip()))
    return f"Not recorded (the command {_ended(exit_code)} after {seconds:.1f} s)" + (f": {shown}" if shown else "")


class ChildJob:
    """One child process of the TUI (`argus record`, `argus check`), in its own process group, its output in a temporary file."""

    def __init__(self, argv: list[str]) -> None:
        self.stopped, self._asked = False, 0.0
        self._log = tempfile.TemporaryFile()  # noqa: SIM115 - closed by close()
        self._begun = time.monotonic()
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - our own argv, no shell
                argv, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        except OSError:
            self._log.close()
            raise

    def poll(self) -> int | None:
        return self._proc.poll()

    def seconds(self) -> float:
        return time.monotonic() - self._begun

    def _signal(self, sig: signal.Signals) -> None:
        try:
            os.killpg(self._proc.pid, sig)  # the group, so pytest and anything it started
        except ProcessLookupError:
            pass  # nothing left in the group

    def stop(self, wait: bool = True) -> None:
        """Ask the group to end (SIGTERM). With `wait`, give it a few seconds and then kill whatever is left; without, the caller
        keeps calling `escalate()` and `sweep()` while it watches the child."""
        if self._proc.poll() is None:
            self.stopped, self._asked = True, time.monotonic()
            self._signal(signal.SIGTERM)
            if not wait:
                return
            try:
                self._proc.wait(STOP_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        self.sweep()

    def escalate(self) -> None:
        """A stop was asked for and the leader is still alive after the grace time: kill the group."""
        if self.stopped and self._proc.poll() is None and time.monotonic() - self._asked > STOP_WAIT_SECONDS:
            self._signal(signal.SIGKILL)

    def sweep(self) -> None:
        """After a stop: end every process the leader left behind (one that ignored SIGTERM, a worker). No-op when no stop was asked."""
        if self.stopped:
            self._signal(signal.SIGKILL)

    def _read(self, limit: int) -> list[str]:
        try:
            size = self._log.seek(0, os.SEEK_END)
            self._log.seek(max(0, size - limit))
            return self._log.read().decode("utf-8", "replace").splitlines()
        except (OSError, ValueError):
            return ["(the output could not be read)"]

    def tail(self, lines: int = 3) -> list[str]:
        return self._read(TAIL_BYTES)[-lines:]

    def output(self) -> list[str]:
        """The output, up to its last OUTPUT_BYTES: a long list of failing tests must not push the report's summary line out of reach."""
        return self._read(OUTPUT_BYTES)

    def close(self) -> None:
        self._log.close()


class RecordJob(ChildJob):
    """One `argus record` child process."""

    def __init__(self, root: Path, python: str | None = None) -> None:
        super().__init__(record_argv(root, python))
