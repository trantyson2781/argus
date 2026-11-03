"""The recording file: one versioned JSON document per run, under `.argus/recordings/` beside a `.gitignore` that hides it.

Only repository-relative paths are stored. A file in another version, or one that is damaged, is an error and never a partial load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT = "argus-recording"
VERSION = 1
DIRECTORY = Path(".argus") / "recordings"
IGNORE_FILE = Path(".argus") / ".gitignore"
MAX_BYTES = 256 * 1024 * 1024  # a full-suite recording is about 9 MB; anything far beyond this is not ours


class RecordingError(Exception):
    """The recording file cannot be used (damaged, or written by another version)."""


@dataclass(frozen=True, slots=True)
class ObservedFunction:
    key: str  # "<file>:<first line>:<co_qualname>", the name the other lists use
    file: str
    first_line: int
    code_qualname: str
    node_id: str | None  # the scanned node it matched, or None with `unmatched` saying why
    unmatched: str | None
    flags: tuple[str, ...]
    calls: int


@dataclass(frozen=True, slots=True)
class ObservedSite:
    caller: str  # function keys
    line: int
    callee: str
    count: int


@dataclass(frozen=True, slots=True)
class ObservedCaller:
    caller: str | None  # None: the nearest Python frame was outside the repository
    callee: str
    count: int


@dataclass(frozen=True, slots=True)
class ObservedTest:
    test: str
    reached: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Processes:
    saved: int  # processes that wrote their data
    lost: int  # processes that started but never saved (killed, or left through os._exit)
    errors: int = 0  # recorder failures inside the saved processes (the recorded program was not affected)


@dataclass(frozen=True, slots=True)
class Recording:
    argus_version: str
    python: str
    command: tuple[str, ...]
    started_at: str  # UTC, "YYYY-MM-DDTHH:MM:SSZ"
    seconds: float
    exit_code: int
    files: tuple[tuple[str, str | None], ...]  # repo file -> source hash at the end of the run; None when it changed during the run
    functions: tuple[ObservedFunction, ...]
    sites: tuple[ObservedSite, ...]
    callers: tuple[ObservedCaller, ...]
    tests: tuple[ObservedTest, ...]
    processes: Processes


def _document(rec: Recording) -> dict[str, Any]:
    return {
        "format": FORMAT, "version": VERSION, "argus_version": rec.argus_version, "python": rec.python, "command": list(rec.command),
        "started_at": rec.started_at, "seconds": rec.seconds, "exit_code": rec.exit_code, "files": dict(rec.files),
        "functions": [
            {"key": f.key, "file": f.file, "first_line": f.first_line, "code_qualname": f.code_qualname, "node_id": f.node_id,
             "unmatched": f.unmatched, "flags": list(f.flags), "calls": f.calls}
            for f in rec.functions
        ],
        "sites": [{"caller": s.caller, "line": s.line, "callee": s.callee, "count": s.count} for s in rec.sites],
        "callers": [{"caller": c.caller, "callee": c.callee, "count": c.count} for c in rec.callers],
        "tests": [{"test": t.test, "reached": list(t.reached)} for t in rec.tests],
        "processes": {"saved": rec.processes.saved, "lost": rec.processes.lost, "errors": rec.processes.errors},
    }


def _check_references(rec: Recording) -> None:
    """Every key in sites, callers and tests must name a listed function: the readers index by key and would otherwise fail on a bad file."""
    keys = {f.key for f in rec.functions}
    named = [s.caller for s in rec.sites] + [s.callee for s in rec.sites] + [c.callee for c in rec.callers] + [t.test for t in rec.tests]
    named += [c.caller for c in rec.callers if c.caller is not None] + [k for t in rec.tests for k in t.reached]
    missing = sorted({k for k in named if k not in keys})
    if missing:
        raise RecordingError(f"it refers to a function it does not list ({missing[0]!r})")


def _recording(doc: dict[str, Any]) -> Recording:
    if doc.get("format") != FORMAT or doc.get("version") != VERSION:
        raise RecordingError(f"not an {FORMAT} file of version {VERSION}")
    rec = Recording(
        argus_version=doc["argus_version"], python=doc["python"], command=tuple(doc["command"]), started_at=doc["started_at"],
        seconds=doc["seconds"], exit_code=doc["exit_code"], files=tuple(doc["files"].items()),
        functions=tuple(
            ObservedFunction(f["key"], f["file"], f["first_line"], f["code_qualname"], f["node_id"], f["unmatched"], tuple(f["flags"]), f["calls"])
            for f in doc["functions"]
        ),
        sites=tuple(ObservedSite(s["caller"], s["line"], s["callee"], s["count"]) for s in doc["sites"]),
        callers=tuple(ObservedCaller(c["caller"], c["callee"], c["count"]) for c in doc["callers"]),
        tests=tuple(ObservedTest(t["test"], tuple(t["reached"])) for t in doc["tests"]),
        processes=Processes(doc["processes"]["saved"], doc["processes"]["lost"], doc["processes"]["errors"]),
    )
    _check_references(rec)
    return rec


def write_recording(root: Path, rec: Recording) -> Path:
    """Write `rec` under `root/.argus/recordings/`. The project's own `.gitignore` is never edited; `.argus/.gitignore` hides the folder."""
    folder = root / DIRECTORY
    folder.mkdir(parents=True, exist_ok=True)
    ignore = root / IGNORE_FILE
    if not ignore.exists():
        ignore.write_text("*\n", encoding="utf-8")
    path = folder / (rec.started_at.replace("-", "").replace(":", "") + ".json")
    scratch = path.with_name(path.name + ".tmp")
    scratch.write_text(json.dumps(_document(rec), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    scratch.replace(path)  # a crash while writing leaves a .tmp file, never a half-written recording that would break the next read
    return path


def load_latest(root: Path) -> Recording | None:
    """The newest recording, or None when there is none. A damaged or foreign file raises `RecordingError`."""
    folder = root / DIRECTORY
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if not files:
        return None
    if files[-1].stat().st_size > MAX_BYTES:
        raise RecordingError(f"{files[-1].name} is larger than {MAX_BYTES // (1024 * 1024)} MB")
    try:
        return _recording(json.loads(files[-1].read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError, UnicodeDecodeError, RecursionError) as exc:
        raise RecordingError(f"{files[-1].name} is damaged: {type(exc).__name__}") from exc
