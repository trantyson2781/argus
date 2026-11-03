"""The call watcher behind `argus record`: counts which functions of one repository ran, who called them and which tests reached them.

Standard library only and no import from `argus`: it is loaded by `sitecustomize` inside the recorded command, whose interpreter may
live in another virtual environment. It uses `sys.monitoring` (Python 3.12+) and watches only the files it was given.

What it records, and what it deliberately does not claim:
  * `calls`: how often a function started.
  * `callers`: the nearest Python frame that was running when it started (None when that frame is outside the repo). A function that a
    C function such as `map` calls therefore has a caller (the line that wrote `map(...)`) but no call site.
  * `sites`: (caller, line, callee) taken only from CALL events, where the called object is a plain function or bound method of the repo.
    A site whose target is something else is switched off for good (`DISABLE`), so a site that sometimes calls repo code and sometimes
    not can miss later targets. That is a miss, never a wrong answer.
  * `tests`: for each function named `test*` in a listed test file, the functions that started while it was running, on its own thread.
Generators and coroutines start once, on the first resume, so their caller is whoever resumed them; they carry a flag for the matcher.
A failure inside a callback never reaches the recorded program: it is counted and the first message is kept (`errors`).
"""

from __future__ import annotations

import os
import sys
import threading
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping
from types import CodeType, FunctionType, MethodType
from typing import Any, NamedTuple

TOOL_IDS = (2, 3, 4)  # PROFILER_ID first, then the two ids that no tool is documented to own
GENERATOR_FLAGS = ((0x20, "generator"), (0x80, "coroutine"), (0x200, "async_generator"))
_MISSING = object()


class _Info(NamedTuple):
    key: str  # "<file>:<first line>:<co_qualname>"; the first line is the decorator's line for a decorated function
    file: str
    first_line: int
    qualname: str
    flags: tuple[str, ...]
    is_test: bool


class _Counts:
    """What one thread saw. Threads never share one, so the counters need no lock."""

    __slots__ = ("calls", "callers", "sites", "reached", "stack")

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.callers: Counter[tuple[str | None, str]] = Counter()
        self.sites: Counter[tuple[str, int, str]] = Counter()
        self.reached: defaultdict[str, set[str]] = defaultdict(set)
        self.stack: list[tuple[CodeType, str]] = []  # tests running on this thread, innermost last


class Recorder:
    """`files` maps absolute paths to repo-relative POSIX paths and lists exactly what is watched; `test_files` are absolute paths."""

    def __init__(self, files: Mapping[str, str], test_files: Collection[str] = ()) -> None:
        self._files = dict(files)
        self._test_files = frozenset(test_files)
        self._tool: int | None = None
        self._info: dict[CodeType, _Info] = {}
        self._paths: dict[str, str | None] = {}
        self._lines: dict[CodeType, tuple[list[int], list[int]]] = {}
        self._armed: set[CodeType] = set()
        self._local = threading.local()
        self._all: list[_Counts] = []
        self._lock = threading.Lock()
        self._errors = 0
        self._first_error: str | None = None

    # -- control ---------------------------------------------------------------------------------------------------------------------

    def start(self) -> None:
        if self._tool is not None:
            raise RuntimeError("this recorder is already running")
        mon = sys.monitoring
        for tool in TOOL_IDS:
            try:
                mon.use_tool_id(tool, "argus-record")
            except ValueError:
                continue
            self._tool = tool
            break
        else:
            raise RuntimeError(f"no free sys.monitoring tool id (tried {TOOL_IDS}): another tool, such as a debugger or profiler, holds them")
        events = mon.events
        mon.register_callback(self._tool, events.PY_START, self._on_start)
        mon.register_callback(self._tool, events.CALL, self._on_call)
        mon.register_callback(self._tool, events.PY_RETURN, self._on_leave)
        mon.register_callback(self._tool, events.PY_UNWIND, self._on_leave)
        mon.set_events(self._tool, events.PY_START | events.PY_UNWIND)  # PY_UNWIND cannot be a local event: it fires for every frame an exception leaves

    def stop(self) -> None:
        if self._tool is None:
            return
        mon, tool = sys.monitoring, self._tool
        self._tool = None
        mon.set_events(tool, 0)
        for code in self._armed:
            mon.set_local_events(tool, code, 0)  # `clear_tool_id` does not exist before Python 3.14
        for event in (mon.events.PY_START, mon.events.CALL, mon.events.PY_RETURN, mon.events.PY_UNWIND):
            mon.register_callback(tool, event, None)
        mon.free_tool_id(tool)

    def reset_after_fork(self) -> None:
        """In a forked child: forget what the parent counted (the child saves only its own calls) and replace locks a parent thread may hold."""
        self._local = threading.local()
        self._all = []
        self._lock = threading.Lock()
        self._errors = 0
        self._first_error = None

    # -- looking code up -------------------------------------------------------------------------------------------------------------

    def _watched_path(self, filename: str) -> str | None:
        known = self._paths.get(filename, _MISSING)
        if known is _MISSING:
            absolute = filename if filename in self._files else os.path.abspath(filename)  # a script run as `python app.py` has a relative name
            known = absolute if absolute in self._files else None
            self._paths[filename] = known
        return known  # type: ignore[return-value]

    def _lookup(self, code: CodeType) -> _Info | None:
        info = self._info.get(code)
        if info is not None:
            return info
        path = self._watched_path(code.co_filename)
        if path is None:
            return None
        flags = tuple(name for bit, name in GENERATOR_FLAGS if code.co_flags & bit)
        file = self._files[path]
        is_test = path in self._test_files and code.co_name.startswith("test") and not flags
        info = _Info(f"{file}:{code.co_firstlineno}:{code.co_qualname}", file, code.co_firstlineno, code.co_qualname, flags, is_test)
        self._info[code] = info
        return info

    def _line(self, code: CodeType, offset: int) -> int:
        table = self._lines.get(code)
        if table is None:
            pairs = [(start, line) for start, _end, line in code.co_lines() if line is not None]
            table = ([p[0] for p in pairs], [p[1] for p in pairs])
            self._lines[code] = table
        index = bisect_right(table[0], offset) - 1
        return table[1][index] if index >= 0 else code.co_firstlineno

    def _counts(self) -> _Counts:
        try:
            return self._local.counts  # type: ignore[no-any-return]
        except AttributeError:
            counts = self._local.counts = _Counts()
            with self._lock:
                self._all.append(counts)
            return counts

    def _fail(self, exc: Exception) -> None:
        self._errors += 1
        if self._first_error is None:
            self._first_error = f"{type(exc).__name__}: {exc}"

    # -- the three callbacks ---------------------------------------------------------------------------------------------------------

    def _on_start(self, code: CodeType, offset: int) -> Any:
        try:
            info = self._lookup(code)
            if info is None:
                return sys.monitoring.DISABLE
            counts = self._counts()
            counts.calls[info.key] += 1
            back = sys._getframe(1).f_back  # frame 1 is the function that just started; its parent is the caller
            caller = self._lookup(back.f_code) if back is not None else None
            counts.callers[(caller.key if caller else None, info.key)] += 1
            if counts.stack:
                counts.reached[counts.stack[-1][1]].add(info.key)
            if info.is_test:
                counts.stack.append((code, info.key))
            if code not in self._armed:
                self._armed.add(code)
                events = sys.monitoring.events
                sys.monitoring.set_local_events(self._tool, code, events.CALL | (events.PY_RETURN if info.is_test else 0))
        except Exception as exc:  # a recorder bug must not break the recorded program
            self._fail(exc)
        return None

    def _on_call(self, code: CodeType, offset: int, callable_: object, arg0: object) -> Any:
        try:
            if type(callable_) is MethodType:
                callable_ = callable_.__func__
            if type(callable_) is not FunctionType:
                return sys.monitoring.DISABLE
            target = self._lookup(callable_.__code__)
            caller = self._lookup(code)
            if target is None or caller is None:
                return sys.monitoring.DISABLE
            self._counts().sites[(caller.key, self._line(code, offset), target.key)] += 1
        except Exception as exc:
            self._fail(exc)
        return None

    def _on_leave(self, code: CodeType, offset: int, _value: object) -> None:
        try:
            counts = getattr(self._local, "counts", None)  # a thread that never started a repo function has nothing to pop
            if counts is not None and counts.stack and counts.stack[-1][0] is code:
                counts.stack.pop()
        except Exception as exc:
            self._fail(exc)

    # -- the result ------------------------------------------------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Everything seen so far as plain JSON data, every list sorted, every key referring to an entry in `functions`."""
        calls: Counter[str] = Counter()
        callers: Counter[tuple[str | None, str]] = Counter()
        sites: Counter[tuple[str, int, str]] = Counter()
        reached: defaultdict[str, set[str]] = defaultdict(set)
        with self._lock:
            threads = list(self._all)
        for counts in threads:
            calls.update(counts.calls)
            callers.update(counts.callers)
            sites.update(counts.sites)
            for test, keys in counts.reached.items():
                reached[test] |= keys
        by_key: dict[str, _Info] = {info.key: info for info in list(self._info.values())}
        return {
            "functions": [
                {"key": i.key, "file": i.file, "first_line": i.first_line, "code_qualname": i.qualname, "calls": calls[i.key], "flags": list(i.flags)}
                for i in sorted(by_key.values(), key=lambda i: i.key)
            ],
            "callers": [
                {"caller": c, "callee": k, "count": n} for (c, k), n in sorted(callers.items(), key=lambda it: (it[0][0] is not None, it[0][0] or "", it[0][1]))
            ],
            "sites": [{"caller": c, "line": ln, "callee": k, "count": n} for (c, ln, k), n in sorted(sites.items())],
            "tests": [{"test": t, "reached": sorted(keys)} for t, keys in sorted(reached.items()) if keys],
            "errors": {"count": self._errors, "first": self._first_error},
        }
