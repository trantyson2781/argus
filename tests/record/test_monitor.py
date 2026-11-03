"""Record mode, slice 1: the call watcher (`argus.record.boot._argus_monitor`), tested in this process.

Rules under test: only files the caller lists are watched; every function that started is counted once per call; the caller of a
function is the nearest Python frame (None when that frame is outside the repo); a call SITE (caller, line, target) comes only from a
CALL event, so a function that a C function such as `map` calls has a caller but no site; tests reach the functions started while they
run (on their own thread only); threads sum; stopping gives the monitoring tool back. Standard library only, no scan involved.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from argus.record.boot._argus_monitor import Recorder

SRC = '''\
def helper(x):
    return x + 1


def caller_a(x):
    return helper(x)  # L:a


def caller_b(x):
    y = helper(x)  # L:b1
    return helper(y)  # L:b2


class K:
    def m(self, x):
        return helper(x)  # L:m


def via_map(xs):
    return list(map(helper, xs))  # L:map


def gen():
    yield 1


def deco(f):
    return f


@deco
def decorated(x):
    return x


def test_one():
    helper(1)


def test_two():
    helper(2)
    caller_a(1)


def test_boom():
    helper(3)
    raise ValueError("boom")


def not_a_test():
    helper(4)


def test_threaded():
    import threading

    worker = threading.Thread(target=helper, args=(5,))
    worker.start()
    worker.join()
'''
FOREIGN = "def foreign(f, x):\n    return f(x)\n"


def line_of(tag: str) -> int:
    return next(n for n, text in enumerate(SRC.splitlines(), 1) if text.endswith(f"# L:{tag}"))


class Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "m.py"
        self.path.write_text(SRC, encoding="utf-8")
        self.other = tmp_path / "other.py"
        self.other.write_text(FOREIGN, encoding="utf-8")
        self.ns: dict[str, Any] = {"__name__": "m"}
        exec(compile(SRC, str(self.path), "exec"), self.ns)
        self.foreign: dict[str, Any] = {}
        exec(compile(FOREIGN, str(self.other), "exec"), self.foreign)
        self.recorder = Recorder({str(self.path): "m.py"}, test_files={str(self.path)})

    def snapshot(self) -> dict[str, Any]:
        self.recorder.stop()
        return self.recorder.snapshot()


@pytest.fixture
def h(tmp_path: Path) -> Iterator[Harness]:
    harness = Harness(tmp_path)
    harness.recorder.start()
    yield harness
    harness.recorder.stop()


def fn(snap: dict[str, Any], qualname: str) -> dict[str, Any]:
    (found,) = [f for f in snap["functions"] if f["code_qualname"] == qualname]
    return found


def callers_of(snap: dict[str, Any], qualname: str) -> dict[str | None, int]:
    key = fn(snap, qualname)["key"]
    names = {f["key"]: f["code_qualname"] for f in snap["functions"]}
    return {names.get(c["caller"]): c["count"] for c in snap["callers"] if c["callee"] == key}


def sites_of(snap: dict[str, Any], qualname: str) -> list[tuple[str, int, int]]:
    key = fn(snap, qualname)["key"]
    names = {f["key"]: f["code_qualname"] for f in snap["functions"]}
    return sorted((names[s["caller"]], s["line"], s["count"]) for s in snap["sites"] if s["callee"] == key)


# -- what is counted -------------------------------------------------------------------------------------------------------------------


def test_each_call_of_a_function_is_counted(h: Harness) -> None:
    for x in range(3):
        h.ns["helper"](x)

    assert fn(h.snapshot(), "helper")["calls"] == 3


def test_a_function_records_its_file_its_first_line_and_a_stable_key(h: Harness) -> None:
    h.ns["helper"](1)

    helper = fn(h.snapshot(), "helper")

    assert (helper["file"], helper["first_line"]) == ("m.py", 1)
    assert helper["key"] == "m.py:1:helper"


def test_a_decorated_function_starts_at_its_decorator_line(h: Harness) -> None:
    h.ns["decorated"](1)

    decorated = fn(h.snapshot(), "decorated")

    assert decorated["first_line"] == SRC.splitlines().index("@deco") + 1  # the matcher must accept def line >= first_line


def test_code_in_files_that_were_not_listed_is_not_recorded(h: Harness) -> None:
    h.foreign["foreign"](h.ns["helper"], 1)

    snap = h.snapshot()

    assert [f["code_qualname"] for f in snap["functions"]] == ["helper"]
    assert callers_of(snap, "helper") == {None: 1}  # the nearest Python frame is outside the repo


def test_methods_use_their_qualified_name(h: Harness) -> None:
    h.ns["K"]().m(1)

    assert fn(h.snapshot(), "K.m")["calls"] == 1


def test_generators_are_flagged_because_their_caller_is_whoever_resumes_them(h: Harness) -> None:
    list(h.ns["gen"]())

    assert fn(h.snapshot(), "gen")["flags"] == ["generator"]


# -- callers and call sites --------------------------------------------------------------------------------------------------------------


def test_the_caller_is_the_nearest_repo_function_and_counts_are_kept_apart(h: Harness) -> None:
    h.ns["caller_a"](1)
    h.ns["caller_b"](1)
    h.ns["helper"](1)

    assert callers_of(h.snapshot(), "helper") == {"caller_a": 1, "caller_b": 2, None: 1}


def test_a_call_site_is_the_exact_line_and_two_lines_stay_two_sites(h: Harness) -> None:
    h.ns["caller_b"](1)
    h.ns["caller_b"](1)
    h.ns["K"]().m(1)

    assert sites_of(h.snapshot(), "helper") == [("K.m", line_of("m"), 1), ("caller_b", line_of("b1"), 2), ("caller_b", line_of("b2"), 2)]


def test_a_function_called_by_a_c_function_has_a_caller_but_no_site(h: Harness) -> None:
    h.ns["via_map"]([1, 2])

    snap = h.snapshot()

    assert callers_of(snap, "helper") == {"via_map": 2}
    assert sites_of(snap, "helper") == []  # map(...) made the calls, not the line that wrote helper


def test_calls_that_leave_the_repo_make_no_site(h: Harness) -> None:
    h.ns["caller_a"](1)

    snap = h.snapshot()

    assert all(site["callee"] == fn(snap, "helper")["key"] for site in snap["sites"])  # len(), print() etc. never appear


# -- tests reach the functions that run inside them ---------------------------------------------------------------------------------------


def reached(snap: dict[str, Any]) -> dict[str, list[str]]:
    names = {f["key"]: f["code_qualname"] for f in snap["functions"]}
    return {names[t["test"]]: sorted(names[k] for k in t["reached"]) for t in snap["tests"]}


def test_a_test_function_reaches_everything_that_starts_inside_it(h: Harness) -> None:
    h.ns["test_one"]()
    h.ns["test_two"]()

    assert reached(h.snapshot()) == {"test_one": ["helper"], "test_two": ["caller_a", "helper"]}


def test_work_outside_any_test_is_not_credited_to_a_test(h: Harness) -> None:
    h.ns["test_one"]()
    h.ns["not_a_test"]()

    assert reached(h.snapshot()) == {"test_one": ["helper"]}


def test_a_test_that_raises_still_stops_being_current(h: Harness) -> None:
    with pytest.raises(ValueError):
        h.ns["test_boom"]()
    h.ns["not_a_test"]()

    assert reached(h.snapshot()) == {"test_boom": ["helper"]}


def test_a_thread_started_by_a_test_is_not_credited_to_it(h: Harness) -> None:
    h.ns["test_threaded"]()

    snap = h.snapshot()

    assert fn(snap, "helper")["calls"] == 1  # it ran, on the worker thread
    assert reached(snap) == {}


def test_only_functions_named_test_are_tests(h: Harness) -> None:
    h.ns["caller_a"](1)

    assert h.snapshot()["tests"] == []


# -- threads, stopping and the tool id -------------------------------------------------------------------------------------------------


def test_calls_from_several_threads_are_summed(h: Harness) -> None:
    def work() -> None:
        for _ in range(50):
            h.ns["helper"](1)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert fn(h.snapshot(), "helper")["calls"] == 200


def test_nothing_is_recorded_after_stop(h: Harness) -> None:
    h.ns["helper"](1)
    h.recorder.stop()
    h.ns["helper"](1)

    assert fn(h.recorder.snapshot(), "helper")["calls"] == 1


def test_stop_twice_is_harmless_and_frees_the_tool_id_for_the_next_recorder(h: Harness) -> None:
    h.recorder.stop()
    h.recorder.stop()

    second = Recorder({str(h.path): "m.py"})
    second.start()
    h.ns["helper"](1)
    second.stop()

    assert fn(second.snapshot(), "helper")["calls"] == 1


def test_the_snapshot_is_plain_json_and_does_not_depend_on_call_order(h: Harness) -> None:
    h.ns["caller_b"](1)
    h.ns["caller_a"](1)
    snap = h.snapshot()

    assert json.loads(json.dumps(snap)) == snap
    assert [f["key"] for f in snap["functions"]] == sorted(f["key"] for f in snap["functions"])


def test_a_busy_tool_id_falls_back_to_the_next_one_and_all_busy_is_an_error(tmp_path: Path) -> None:
    mon = sys.monitoring
    taken = [i for i in (2, 3, 4) if mon.get_tool(i) is None]
    for i in taken[:1]:
        mon.use_tool_id(i, "test-busy")
    try:
        recorder = Recorder({})
        recorder.start()
        recorder.stop()
        for i in taken[1:]:
            mon.use_tool_id(i, "test-busy")
        with pytest.raises(RuntimeError, match="tool id"):
            Recorder({}).start()
    finally:
        for i in taken:
            if mon.get_tool(i) == "test-busy":
                mon.free_tool_id(i)


def test_a_recorder_bug_never_reaches_the_recorded_program_and_is_reported(tmp_path: Path) -> None:
    class Broken(Recorder):
        def _line(self, code: Any, offset: int) -> int:
            raise RuntimeError("broken line table")

    harness = Harness(tmp_path)
    harness.recorder = Broken({str(harness.path): "m.py"})
    harness.recorder.start()
    try:
        assert harness.ns["caller_a"](1) == 2  # the program still gets its answer
    finally:
        harness.recorder.stop()

    assert harness.recorder.snapshot()["errors"] == {"count": 1, "first": "RuntimeError: broken line table"}


def test_a_clean_run_reports_no_errors(h: Harness) -> None:
    h.ns["caller_a"](1)

    assert h.snapshot()["errors"] == {"count": 0, "first": None}


def test_after_a_fork_reset_the_child_forgets_its_parents_counts_but_keeps_what_it_knows(h: Harness) -> None:
    h.ns["caller_a"](1)

    h.recorder.reset_after_fork()
    h.ns["helper"](1)
    snap = h.snapshot()

    assert fn(snap, "helper")["calls"] == 1 and fn(snap, "caller_a")["calls"] == 0
    assert callers_of(snap, "helper") == {None: 1}
