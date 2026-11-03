"""Record mode, slice 2: the recording file (`argus.record.store`).

Rules under test: one versioned JSON file per run under `.argus/recordings/`, written beside a `.gitignore` that hides it; the newest file
is the one the views read; a file in another version or a damaged one is an error, never a partial load; nothing absolute is stored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.record.store import (
    ObservedCaller, ObservedFunction, ObservedSite, ObservedTest, Processes, Recording, RecordingError, load_latest, write_recording,
)


def make(started_at: str = "2026-10-23T10:00:00Z", **over) -> Recording:
    fields = dict(
        argus_version="0.1.0", python="3.13.1", command=("pytest", "-q"), started_at=started_at, seconds=1.5, exit_code=0,
        files=(("pkg/mod.py", "ab12"), ("pkg/other.py", None)),
        functions=(
            ObservedFunction("pkg/mod.py:4:top", "pkg/mod.py", 4, "top", "function:pkg.mod.top", None, (), 3),
            ObservedFunction("pkg/mod.py:8:<lambda>", "pkg/mod.py", 8, "<lambda>", None, "anonymous", (), 1),
            ObservedFunction("tests/t.py:1:test_a", "tests/t.py", 1, "test_a", "function:tests.t.test_a", None, (), 1),
        ),
        sites=(ObservedSite("pkg/mod.py:4:top", 5, "pkg/mod.py:4:top", 2),),
        callers=(ObservedCaller(None, "pkg/mod.py:4:top", 3),),
        tests=(ObservedTest("tests/t.py:1:test_a", ("pkg/mod.py:4:top",)),),
        processes=Processes(saved=2, lost=1),
    )
    return Recording(**{**fields, **over})


def test_a_recording_round_trips_through_the_file(tmp_path: Path) -> None:
    rec = make()

    write_recording(tmp_path, rec)

    assert load_latest(tmp_path) == rec


def test_the_file_lives_under_argus_recordings_beside_a_gitignore_that_hides_it(tmp_path: Path) -> None:
    path = write_recording(tmp_path, make())

    assert path.parent == tmp_path / ".argus" / "recordings" and path.suffix == ".json"
    assert (tmp_path / ".argus" / ".gitignore").read_text(encoding="utf-8").strip() == "*"
    assert not (tmp_path / ".gitignore").exists()  # the project's own ignore file is never touched


def test_the_newest_file_is_the_one_that_is_loaded(tmp_path: Path) -> None:
    write_recording(tmp_path, make(started_at="2026-10-23T09:00:00Z", exit_code=1))
    write_recording(tmp_path, make(started_at="2026-10-23T11:00:00Z", exit_code=2))
    write_recording(tmp_path, make(started_at="2026-10-23T10:00:00Z", exit_code=3))

    latest = load_latest(tmp_path)

    assert latest is not None and latest.exit_code == 2


def test_no_recording_is_none_not_an_error(tmp_path: Path) -> None:
    assert load_latest(tmp_path) is None


def test_the_file_is_deterministic_and_holds_no_absolute_path(tmp_path: Path) -> None:
    first = write_recording(tmp_path, make()).read_text(encoding="utf-8")
    second = write_recording(tmp_path, make()).read_text(encoding="utf-8")

    assert first == second
    assert str(tmp_path) not in first
    assert json.loads(first)["format"] == "argus-recording" and json.loads(first)["version"] == 1


@pytest.mark.parametrize("damage", [lambda d: d.update(version=2), lambda d: d.update(format="other"), lambda d: d.pop("functions")])
def test_a_file_in_another_version_or_missing_a_part_is_an_error(tmp_path: Path, damage) -> None:
    path = write_recording(tmp_path, make())
    data = json.loads(path.read_text(encoding="utf-8"))
    damage(data)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RecordingError):
        load_latest(tmp_path)


def test_text_that_is_not_json_is_an_error(tmp_path: Path) -> None:
    path = write_recording(tmp_path, make())
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RecordingError):
        load_latest(tmp_path)


def test_a_leftover_temporary_file_is_ignored_and_none_is_left_behind(tmp_path: Path) -> None:
    folder = tmp_path / ".argus" / "recordings"
    folder.mkdir(parents=True)
    (folder / "99990101T000000Z.json.tmp").write_text("{half", encoding="utf-8")  # a crash while writing

    write_recording(tmp_path, make())

    assert load_latest(tmp_path) == make()
    assert [p.name for p in folder.glob("*.tmp")] == ["99990101T000000Z.json.tmp"]  # only the stray one


def test_a_recording_that_names_a_function_it_does_not_list_is_an_error(tmp_path: Path) -> None:
    for broken in (
        make(sites=(ObservedSite("ghost", 5, "pkg/mod.py:4:top", 1),)),
        make(callers=(ObservedCaller("ghost", "pkg/mod.py:4:top", 1),)),
        make(callers=(ObservedCaller(None, "ghost", 1),)),
        make(tests=(ObservedTest("ghost", ()),)),
        make(tests=(ObservedTest("tests/t.py:1:test_a", ("ghost",)),)),
    ):
        write_recording(tmp_path, broken)
        with pytest.raises(RecordingError, match="does not list"):
            load_latest(tmp_path)


def test_a_recording_that_is_too_large_or_too_deeply_nested_is_an_error_not_a_hang_or_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.record import store

    path = write_recording(tmp_path, make())
    monkeypatch.setattr(store, "MAX_BYTES", 10)
    with pytest.raises(RecordingError, match="larger than"):
        load_latest(tmp_path)

    monkeypatch.setattr(store, "MAX_BYTES", 10**9)
    path.write_text("[" * 100000, encoding="utf-8")
    with pytest.raises(RecordingError):
        load_latest(tmp_path)
