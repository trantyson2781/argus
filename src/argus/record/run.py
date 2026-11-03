"""Run a command under the call watcher and save what ran as a recording (`argus record -- CMD...`).

This is the one place Argus executes the repository's code, so it only happens on this explicit command, and never from `watch` or the
TUI. The command runs in the repository with the terminal attached. A small `sitecustomize` on PYTHONPATH starts the watcher in every
Python process the command starts; the command's exit code is kept.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import argus
from argus.analysis.code_type import CodeType, classify_file
from argus.explore.claims import file_hash
from argus.extract import scan
from argus.extract.discovery import discover
from argus.record import boot
from argus.record.match import Match, match_functions
from argus.record.store import ObservedCaller, ObservedFunction, ObservedSite, ObservedTest, Processes, Recording, write_recording

SECOND_INTERRUPT = 2
THIRD_INTERRUPT = 3


class RecordError(Exception):
    """The command could not be run."""


@dataclass(frozen=True, slots=True)
class RecordResult:
    path: Path | None  # None: nothing was recorded and the caller asked for no empty file
    recording: Recording


def _hash_or_none(root: Path, rel: str) -> str | None:
    try:
        return file_hash(root, rel)
    except OSError:
        return None  # deleted or unreadable: treated like a file that changed


def _execute(argv: Sequence[str], root: Path, env: dict[str, str]) -> int:
    """Run the command to its end and return its exit code (128 + n when a signal ended it). The first Ctrl-C is left to the command,
    so it can finish and save; a second one asks it to stop, a third kills it.

    shortcut: only the command itself is signalled, so a worker it started can outlive a stop; upgrade with process groups if that bites.
    """
    proc: subprocess.Popen[bytes] | None = None
    interrupts = 0

    def on_interrupt(_signum: int, _frame: object) -> None:
        nonlocal interrupts
        interrupts += 1
        if proc is None:  # before the command exists there is nothing to stop; the first Ctrl-C only counts
            return
        if interrupts == SECOND_INTERRUPT:
            proc.terminate()
        elif interrupts >= THIRD_INTERRUPT:
            proc.kill()

    # The handler goes in first, so a Ctrl-C can never arrive in the gap between starting the command and watching it.
    previous = signal.signal(signal.SIGINT, on_interrupt) if threading.current_thread() is threading.main_thread() else None
    try:
        try:
            proc = subprocess.Popen(list(argv), cwd=root, env=env)
        except OSError as exc:
            raise RecordError(f"cannot run {argv[0]!r}: {exc.strerror or type(exc).__name__}") from exc
        code = proc.wait()
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    return code if code >= 0 else 128 - code


def _load(path: Path) -> dict[str, Any] | None:
    """One process's data in a fixed shape, or None when the file is damaged (a crash while writing, or something else wrote there)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "functions": [{k: fn[k] for k in ("key", "file", "first_line", "code_qualname", "calls", "flags")} for fn in data["functions"]],
            "callers": [(c["caller"], c["callee"], c["count"]) for c in data["callers"]],
            "sites": [(s["caller"], s["line"], s["callee"], s["count"]) for s in data["sites"]],
            "tests": [(t["test"], t["reached"]) for t in data["tests"]],
            "errors": data["errors"]["count"],
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _merge(out: Path) -> tuple[dict[str, Any], Processes]:
    """Add up what every process saved. A damaged file counts as a process that did not save, never as an error that loses the others."""
    good = {path.stem: loaded for path in sorted(out.glob("*.json")) if (loaded := _load(path)) is not None}
    started = {p.stem for p in out.glob("*.start")}
    functions: dict[str, dict[str, Any]] = {}
    calls: Counter[str] = Counter()
    callers: Counter[tuple[str | None, str]] = Counter()
    sites: Counter[tuple[str, int, str]] = Counter()
    tests: dict[str, set[str]] = {}
    errors = 0
    for data in good.values():
        for fn in data["functions"]:
            known = functions.setdefault(fn["key"], {**fn, "flags": []})
            known["flags"] = sorted(set(known["flags"]) | set(fn["flags"]))
            calls[fn["key"]] += fn["calls"]
        for caller, callee, count in data["callers"]:
            callers[(caller, callee)] += count
        for caller, line, callee, count in data["sites"]:
            sites[(caller, line, callee)] += count
        for test, reached in data["tests"]:
            tests.setdefault(test, set()).update(reached)
        errors += data["errors"]
    merged = {"functions": functions, "calls": calls, "callers": callers, "sites": sites, "tests": tests}
    return merged, Processes(saved=len(good), lost=len(started - set(good)), errors=errors)


def _node_of(file_changed: bool, match: Match) -> tuple[str | None, str | None]:
    return (None, "file_changed") if file_changed else (match.node_id, match.reason)


def _recording(root: Path, argv: Sequence[str], merged: dict[str, Any], processes: Processes, files: tuple[tuple[str, str | None], ...], run: dict[str, Any]) -> Recording:
    matches = match_functions(scan(root), merged["functions"].values())
    changed = {rel for rel, digest in files if digest is None}  # the scan describes the file now, not what ran
    functions = tuple(
        ObservedFunction(
            key, fn["file"], fn["first_line"], fn["code_qualname"], *_node_of(fn["file"] in changed, matches[key]), tuple(fn["flags"]), merged["calls"][key],
        )
        for key, fn in sorted(merged["functions"].items())
    )
    return Recording(
        argus_version=argus.__version__, python=sys.version.split()[0], command=tuple(argv), started_at=run["started_at"], seconds=run["seconds"],
        exit_code=run["exit_code"], files=files, functions=functions,
        sites=tuple(ObservedSite(c, ln, k, n) for (c, ln, k), n in sorted(merged["sites"].items())),
        callers=tuple(
            ObservedCaller(c, k, n) for (c, k), n in sorted(merged["callers"].items(), key=lambda it: (it[0][0] is not None, it[0][0] or "", it[0][1]))
        ),
        tests=tuple(ObservedTest(t, tuple(sorted(keys))) for t, keys in sorted(merged["tests"].items())),
        processes=processes,
    )


def record_command(root: Path, argv: Sequence[str], *, save_empty: bool = True) -> RecordResult:
    """Run `argv` in `root` with the call watcher on, save the recording under `root/.argus/recordings/` and return it.
    With `save_empty=False` a run in which no Python process saved anything writes no file (an empty file would become the newest recording and hide a good one)."""
    if not argv:
        raise ValueError("nothing to run: give a command after --")
    root = root.resolve()
    watched = {str(f.path): f.path.relative_to(root).as_posix() for f in discover(root)}
    tests = [path for path, rel in watched.items() if classify_file(rel) is CodeType.TEST]
    before = {rel: _hash_or_none(root, rel) for rel in watched.values()}
    with tempfile.TemporaryDirectory(prefix="argus-record-") as scratch:
        out = Path(scratch) / "processes"
        out.mkdir()
        config = Path(scratch) / "config.json"
        config.write_text(json.dumps({"files": watched, "test_files": tests}), encoding="utf-8")
        boot_dir = str(Path(boot.__file__).parent)
        env = {**os.environ, "ARGUS_RECORD_OUT": str(out), "ARGUS_RECORD_CONFIG": str(config)}
        env["PYTHONPATH"] = boot_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        started_at, begun = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), time.monotonic()
        exit_code = _execute(argv, root, env)
        seconds = round(time.monotonic() - begun, 2)
        merged, processes = _merge(out)
    after = {rel: _hash_or_none(root, rel) for rel in watched.values()}
    files = tuple(sorted((rel, after[rel] if after[rel] == before[rel] else None) for rel in watched.values()))
    try:
        recording = _recording(root, argv, merged, processes, files, {"started_at": started_at, "seconds": seconds, "exit_code": exit_code})
        return RecordResult(write_recording(root, recording) if save_empty or processes.saved else None, recording)
    except Exception as exc:  # noqa: BLE001 - the command already ran; say what was lost and how it ended
        raise RecordError(f"cannot save the recording ({type(exc).__name__}: {exc}); the command exited with {exit_code}") from exc
