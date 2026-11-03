"""Record mode, slice 4: what a recording says about one symbol (`argus.record.observed`).

Rules under test: the facts are OBSERVED, from the last recording only. A symbol that did not run says so (0 calls, not "unknown"). Callers
that static reading did not list are marked `also`. Test code is told apart from callers. Calls whose nearest Python frame is outside the
repository are counted, not named. If a file the facts rest on changed after the recording they are STALE, and every reason the recording
may be incomplete is carried along.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from argus.explore.claims import file_hash
from argus.record.observed import latest_recording, observed_for
from argus.record.store import (
    ObservedCaller, ObservedFunction, ObservedSite, ObservedTest, Processes, Recording, RecordingError, write_recording,
)

FILES = {"pkg/mod.py": "def helper(x):\n    return x\n", "pkg/other.py": "def dispatch(k):\n    return None\n\n\ndef main():\n    return None\n",
         "tests/test_mod.py": "def test_x():\n    pass\n"}
HELPER = "function:pkg.mod.helper"


def fn(key: str, file: str, line: int, name: str, node_id: str | None, calls: int = 1) -> ObservedFunction:
    return ObservedFunction(key, file, line, name, node_id, None if node_id else "anonymous", (), calls)


def build(root: Path, **over) -> Recording:
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    h = fn("pkg/mod.py:1:helper", "pkg/mod.py", 1, "helper", HELPER, calls=11)
    dispatch = fn("pkg/other.py:1:dispatch", "pkg/other.py", 1, "dispatch", "function:pkg.other.dispatch")
    main = fn("pkg/other.py:5:main", "pkg/other.py", 5, "main", "function:pkg.other.main")
    test = fn("tests/test_mod.py:1:test_x", "tests/test_mod.py", 1, "test_x", "function:tests.test_mod.test_x")
    anon = fn("pkg/other.py:9:<lambda>", "pkg/other.py", 9, "<lambda>", None)
    fields = dict(
        argus_version="0.1.0", python="3.13", command=("pytest", "-q"), started_at="2026-10-24T14:02:11Z", seconds=1.0, exit_code=0,
        files=tuple((rel, file_hash(root, rel)) for rel in sorted(FILES)),
        functions=(h, dispatch, main, test, anon),
        sites=(ObservedSite(dispatch.key, 2, h.key, 5), ObservedSite(main.key, 6, h.key, 2)),
        callers=(
            ObservedCaller(None, h.key, 3), ObservedCaller(dispatch.key, h.key, 5), ObservedCaller(main.key, h.key, 2),
            ObservedCaller(test.key, h.key, 1), ObservedCaller(anon.key, h.key, 4),
        ),
        tests=(ObservedTest(test.key, (h.key,)),), processes=Processes(saved=1, lost=0),
    )
    return Recording(**{**fields, **over})


@pytest.fixture
def rec(tmp_path: Path) -> Recording:
    return build(tmp_path)


def facts(rec: Recording, root: Path, node_id: str = HELPER, file: str = "pkg/mod.py", known: tuple[str, ...] | None = ()):
    return observed_for(rec, node_id, file, root, known)


def test_a_symbol_that_ran_reports_calls_when_and_how(tmp_path: Path, rec: Recording) -> None:
    got = facts(rec, tmp_path)

    assert (got.calls, got.recorded_at, got.command) == (11, "2026-10-24 14:02 UTC", "pytest -q")
    assert not got.stale and got.incomplete == ()


def test_callers_are_named_with_their_call_lines_and_test_code_is_listed_apart(tmp_path: Path, rec: Recording) -> None:
    got = facts(rec, tmp_path)

    by_name = {c.name: c for c in got.callers}
    assert set(by_name) == {"pkg.other.dispatch", "pkg.other.main", "<lambda> (pkg/other.py, not matched to the scan)"}  # no test function here
    assert by_name["pkg.other.dispatch"].count == 5 and by_name["pkg.other.dispatch"].lines == (2,)
    assert got.tests == ("tests.test_mod.test_x",)


def test_calls_from_outside_the_repository_are_counted_not_named(tmp_path: Path, rec: Recording) -> None:
    assert facts(rec, tmp_path).outside_calls == 3


def test_callers_are_sorted_by_how_often_they_called(tmp_path: Path, rec: Recording) -> None:
    assert [c.count for c in facts(rec, tmp_path).callers] == [5, 4, 2]


def test_a_caller_static_reading_did_not_list_is_marked_also(tmp_path: Path, rec: Recording) -> None:
    got = facts(rec, tmp_path, known=("pkg.other.main",))

    assert {c.name: c.also for c in got.callers}["pkg.other.main"] is False
    assert {c.name: c.also for c in got.callers}["pkg.other.dispatch"] is True


def test_without_a_static_caller_list_nothing_is_marked_also(tmp_path: Path, rec: Recording) -> None:
    assert not any(c.also for c in observed_for(rec, HELPER, "pkg/mod.py", tmp_path, None).callers)


def test_a_symbol_the_recording_never_saw_start_did_not_run(tmp_path: Path, rec: Recording) -> None:
    got = facts(rec, tmp_path, node_id="function:pkg.mod.never", file="pkg/mod.py")

    assert got.calls == 0 and got.callers == () and got.tests == ()


def test_a_function_that_was_only_called_and_never_started_did_not_run(tmp_path: Path, rec: Recording) -> None:
    fns = tuple(replace(f, calls=0) if f.node_id == HELPER else f for f in rec.functions)

    assert facts(replace(rec, functions=fns), tmp_path).calls == 0


# -- stale ------------------------------------------------------------------------------------------------------------------------------


def test_nothing_changed_since_the_recording_is_not_stale(tmp_path: Path, rec: Recording) -> None:
    assert facts(rec, tmp_path).stale_files == ()


def test_the_symbols_own_file_changing_makes_the_facts_stale(tmp_path: Path, rec: Recording) -> None:
    (tmp_path / "pkg/mod.py").write_text("def helper(x):\n    return x + 1\n", encoding="utf-8")

    got = facts(rec, tmp_path)

    assert got.stale and got.stale_files == ("pkg/mod.py",)


def test_a_callers_or_tests_file_changing_makes_the_facts_stale(tmp_path: Path, rec: Recording) -> None:
    (tmp_path / "pkg/other.py").write_text("# edited\n", encoding="utf-8")
    (tmp_path / "tests/test_mod.py").write_text("# edited\n", encoding="utf-8")

    assert facts(rec, tmp_path).stale_files == ("pkg/other.py", "tests/test_mod.py")


def test_a_file_that_changed_during_the_run_or_vanished_since_is_stale(tmp_path: Path, rec: Recording) -> None:
    during = replace(rec, files=tuple((f, None if f == "pkg/mod.py" else h) for f, h in rec.files))
    (tmp_path / "pkg/other.py").unlink()

    assert facts(during, tmp_path).stale_files == ("pkg/mod.py", "pkg/other.py")


def test_a_file_the_recording_does_not_know_is_stale(tmp_path: Path, rec: Recording) -> None:
    (tmp_path / "pkg/new.py").write_text("def f():\n    pass\n", encoding="utf-8")

    assert facts(rec, tmp_path, node_id="function:pkg.new.f", file="pkg/new.py").stale_files == ("pkg/new.py",)


# -- incomplete -------------------------------------------------------------------------------------------------------------------------


def test_every_reason_the_recording_may_be_incomplete_is_carried(tmp_path: Path, rec: Recording) -> None:
    got = facts(replace(rec, processes=Processes(saved=1, lost=2, errors=1)), tmp_path)

    assert got.incomplete == ("2 processes did not save", "the recorder failed 1 time")


def test_a_recording_with_no_saved_process_says_nothing_was_recorded(tmp_path: Path, rec: Recording) -> None:
    assert "nothing was recorded" in facts(replace(rec, processes=Processes(saved=0, lost=0)), tmp_path).incomplete


# -- loading ----------------------------------------------------------------------------------------------------------------------------


def test_no_recording_is_none_without_an_error(tmp_path: Path) -> None:
    assert latest_recording(tmp_path) == (None, None)


def test_the_newest_recording_is_loaded(tmp_path: Path, rec: Recording) -> None:
    write_recording(tmp_path, rec)

    loaded, error = latest_recording(tmp_path)

    assert loaded == rec and error is None


def test_a_damaged_recording_is_a_message_not_an_exception(tmp_path: Path, rec: Recording) -> None:
    path = write_recording(tmp_path, rec)
    path.write_text("{broken", encoding="utf-8")

    loaded, error = latest_recording(tmp_path)

    assert loaded is None and error is not None and path.name in error


def test_a_file_name_that_leaves_the_repository_is_never_read(tmp_path: Path, rec: Recording, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.record import observed

    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    evil = ObservedFunction("evil", "../outside_secret.txt", 1, "g", "function:pkg.evil.g", None, (), 1)
    hostile = replace(
        rec, functions=(*rec.functions, evil), files=(*rec.files, ("../outside_secret.txt", "bb")),
        callers=(*rec.callers, ObservedCaller("evil", rec.functions[0].key, 1)),
    )
    read: list[str] = []
    real = observed.file_hash
    monkeypatch.setattr(observed, "file_hash", lambda root, file: (read.append(file), real(root, file))[1])

    got = facts(hostile, tmp_path)

    assert "../outside_secret.txt" not in read and "../outside_secret.txt" in got.stale_files


def test_a_symlink_that_points_outside_the_repository_is_not_followed(tmp_path: Path, rec: Recording) -> None:
    outside = tmp_path.parent / "outside_target.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "pkg" / "link.py").symlink_to(outside)
    linked = replace(rec, files=(*rec.files, ("pkg/link.py", file_hash(tmp_path, "pkg/link.py"))))  # the hash of what the link points at: following it would look unchanged

    assert "pkg/link.py" in facts(linked, tmp_path, node_id="function:pkg.link.f", file="pkg/link.py").stale_files
