"""What the latest recording says about one symbol: OBSERVED facts, shown beside the facts read from the code and never mixed into them.

Only what ran is seen: a symbol with 0 calls did not run in that recording, which says nothing about other runs. The facts rest on the
files of the symbol, of its callers and of the tests; if any of them changed after the recording the facts are STALE.
"""

from __future__ import annotations

import shlex
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from argus.analysis.code_type import CodeType, classify_file
from argus.explore.claims import file_hash
from argus.record.store import ObservedFunction, Recording, RecordingError, load_latest


@dataclass(frozen=True, slots=True)
class SeenCaller:
    name: str  # qualname, or the code's own name and file when the scan could not match it
    count: int
    lines: tuple[int, ...]  # lines of the exact call sites seen for this caller
    also: bool  # static reading did not list it as a caller


@dataclass(frozen=True, slots=True)
class ObservedFacts:
    recorded_at: str
    command: str
    calls: int  # times the symbol started; 0 means it did not run
    callers: tuple[SeenCaller, ...]  # production callers, most calls first; test code is under `tests`
    outside_calls: int  # calls whose nearest Python frame was outside the repository (a framework, the standard library)
    tests: tuple[str, ...]  # test functions that reached the symbol while they ran, directly or not
    stale_files: tuple[str, ...]  # files these facts rest on that changed after the recording
    incomplete: tuple[str, ...]  # every reason the recording may be missing calls

    @property
    def stale(self) -> bool:
        return bool(self.stale_files)


def latest_recording(root: Path) -> tuple[Recording | None, str | None]:
    """(recording, None), (None, None) when there is none, or (None, message) when the newest file cannot be used."""
    try:
        return load_latest(root), None
    except RecordingError as exc:
        return None, f"the latest recording cannot be read ({exc})"


def _name(fn: ObservedFunction) -> str:
    return fn.node_id.partition(":")[2] if fn.node_id else f"{fn.code_qualname} ({fn.file}, not matched to the scan)"


def _is_test(fn: ObservedFunction) -> bool:
    return classify_file(fn.file) is CodeType.TEST


def _inside(root: Path, file: str) -> bool:
    """The recording is data from disk: a name that leaves the repository (`../x`, an absolute path, a link pointing out) is never opened."""
    try:
        return (root / file).resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def _changed(recorded: dict[str, str | None], root: Path, file: str) -> bool:
    if recorded.get(file) is None or not _inside(root, file):  # changed during the run, never part of it, or not a file of this repository
        return True
    try:
        return file_hash(root, file) != recorded[file]
    except OSError:
        return True


def changed_files(rec: Recording, root: Path, files: Collection[str]) -> frozenset[str]:
    """The files among `files` that changed after the recording, changed during it, or are not part of it."""
    recorded = dict(rec.files)
    return frozenset(f for f in files if _changed(recorded, root, f))


def _incomplete(rec: Recording) -> tuple[str, ...]:
    p = rec.processes
    notes = []
    if p.saved == 0:
        notes.append("nothing was recorded")
    if p.lost:
        notes.append(f"{p.lost} process{'' if p.lost == 1 else 'es'} did not save")
    if p.errors:
        notes.append(f"the recorder failed {p.errors} time{'' if p.errors == 1 else 's'}")
    return tuple(notes)


def observed_for(rec: Recording, node_id: str, file: str, root: Path, known_callers: Collection[str] | None = None) -> ObservedFacts:
    """The facts for the symbol `node_id` defined in `file`. `known_callers` are the qualnames static reading lists, so the rest are `also`;
    None means no list was computed, and then nothing is marked."""
    by_key = {f.key: f for f in rec.functions}
    mine = {f.key for f in rec.functions if f.node_id == node_id}
    counts: defaultdict[str, int] = defaultdict(int)
    lines: defaultdict[str, set[int]] = defaultdict(set)
    files = {file}
    outside = 0
    for c in rec.callers:
        if c.callee not in mine:
            continue
        if c.caller is None:
            outside += c.count
            continue
        caller = by_key[c.caller]
        if _is_test(caller):
            continue
        counts[_name(caller)] += c.count
        files.add(caller.file)
    for s in rec.sites:
        caller = by_key.get(s.caller)
        if s.callee in mine and caller is not None and not _is_test(caller):
            lines[_name(caller)].add(s.line)
    tests = sorted({_name(by_key[t.test]) for t in rec.tests if mine & set(t.reached)})
    files |= {by_key[t.test].file for t in rec.tests if mine & set(t.reached)}
    callers = tuple(
        SeenCaller(name, count, tuple(sorted(lines[name])), known_callers is not None and name not in known_callers)
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return ObservedFacts(
        recorded_at=rec.started_at.replace("T", " ")[:16] + " UTC", command=shlex.join((Path(rec.command[0]).name, *rec.command[1:])),  # the program by name, not by its path
        calls=sum(by_key[k].calls for k in mine), callers=callers, outside_calls=outside, tests=tuple(tests),
        stale_files=tuple(sorted(changed_files(rec, root, files))), incomplete=_incomplete(rec),
    )
