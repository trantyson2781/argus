"""The R key of the TUI: it asks first, then runs `argus record -- <python> -m pytest -q` as a separate process and reports in the status line.

Rules under test: R alone runs nothing; only `y` starts it and any other key (even `q`) cancels; one run at a time and R again stops it;
quitting stops the child; the result says whether a recording was saved; the TUI never imports the recorder (the guard in
tests/record/test_guard.py) and the real command leaves a recording that the panels read.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.widgets import Static

from argus.record.store import DIRECTORY
from argus.tui import recorder
from argus.tui.app import ArgusApp
from tests.explore.diff_helpers import GitRepo

SIZE = (120, 40)


def _status(app: ArgusApp) -> str:
    return str(app.query_one("#status", Static).content)


async def _until(pilot, predicate: Callable[[], bool], what: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"never saw: {what}")


class FakeJob:
    """Stands in for the child process: nothing is executed."""

    started: list[Path] = []

    def __init__(self, root: Path, python=None) -> None:
        self.root, self.code, self.stopped, self.lines = root, None, False, ["boom"]
        FakeJob.started.append(root)

    def close(self) -> None:
        pass

    def poll(self) -> int | None:
        return self.code

    def seconds(self) -> float:
        return 1.0

    def stop(self, wait: bool = True) -> None:
        self.stopped, self.code = True, -15

    def escalate(self) -> None:
        pass

    def sweep(self) -> None:
        pass

    def tail(self, lines: int = 3) -> list[str]:
        return self.lines[-lines:]


@pytest.fixture
def fake_job(monkeypatch: pytest.MonkeyPatch) -> type[FakeJob]:
    FakeJob.started = []
    monkeypatch.setattr("argus.tui.app.RecordJob", FakeJob)
    return FakeJob


@pytest.fixture
def repo(git_repo: GitRepo) -> GitRepo:
    git_repo.write_files({
        "shop/__init__.py": "",
        "shop/billing.py": "def charge(amount):\n    return amount + 1\n",
        "tests/test_billing.py": "from shop.billing import charge\n\n\ndef test_charge():\n    assert charge(1) == 2\n",
    })
    git_repo.commit("baseline")
    return git_repo


# -- the command ---------------------------------------------------------------------------------------------------------


def test_the_command_uses_the_repository_python_when_it_has_one(tmp_path: Path) -> None:
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")

    assert recorder.test_command(tmp_path) == [str(python), "-m", "pytest", "-q"]


def test_the_command_falls_back_to_the_python_running_argus(tmp_path: Path) -> None:
    assert recorder.test_command(tmp_path) == [sys.executable, "-m", "pytest", "-q"]


def test_the_child_is_the_record_command_of_this_repository(tmp_path: Path) -> None:
    argv = recorder.record_argv(tmp_path)

    assert argv[:5] == [sys.executable, "-P", "-m", "argus.cli", "record"]
    assert argv[argv.index("-C") + 1] == str(tmp_path) and argv[argv.index("--") + 1:] == recorder.test_command(tmp_path)


def test_the_question_names_the_command_and_says_it_runs_code(tmp_path: Path) -> None:
    text = recorder.question(tmp_path)

    assert "-m pytest -q" in text and "executes your code" in text and "y = yes" in text


# -- the sentence after the run ------------------------------------------------------------------------------------------


def test_a_saved_recording_is_reported_with_its_numbers() -> None:
    text = recorder.finished_text(saved="20261027T090117Z.json", exit_code=0, seconds=14.1, ran=1143, matched=1051, tail=[], stopped=False)

    assert text.startswith("Recorded") and "14.1 s" in text and "1143 functions ran" in text and "1051 matched" in text


def test_a_recording_of_a_failing_command_says_so() -> None:
    text = recorder.finished_text(saved="x.json", exit_code=1, seconds=3.0, ran=10, matched=9, tail=[], stopped=False)

    assert text.startswith("Recorded, but the command exited with 1")


def test_a_recording_file_that_cannot_be_read_is_not_reported_as_recorded() -> None:
    text = recorder.finished_text(saved="x.json", exit_code=0, seconds=8.8, ran=0, matched=0, tail=[], stopped=False, unreadable="damaged")

    assert text.startswith("A recording file appeared but cannot be read") and "damaged" in text and "Recorded:" not in text


def test_processes_that_did_not_save_are_named() -> None:
    text = recorder.finished_text(saved="x.json", exit_code=0, seconds=1.0, ran=5, matched=5, tail=[], stopped=False, lost=3, errors=1)

    assert "3 processes did not save" in text and "recorder failed 1 time" in text


def test_a_recording_in_which_no_function_ran_says_so() -> None:
    text = recorder.finished_text(saved="x.json", exit_code=5, seconds=1.0, ran=0, matched=0, tail=[], stopped=False)

    assert "no function ran" in text


def test_a_command_ended_by_a_signal_is_not_shown_as_a_negative_exit_code() -> None:
    text = recorder.finished_text(saved=None, exit_code=-9, seconds=1.0, ran=0, matched=0, tail=[], stopped=False)

    assert "-9" not in text and "signal 9" in text


# -- which recording is this run's ---------------------------------------------------------------------------------------


def test_a_file_written_during_the_run_is_new(tmp_path: Path) -> None:
    folder = tmp_path / DIRECTORY
    folder.mkdir(parents=True)
    (folder / "20260101T000000Z.json").write_text("{}")
    before = recorder.snapshot(tmp_path)
    (folder / "20260101T000005Z.json").write_text("{}")

    assert recorder.fresh(before, recorder.snapshot(tmp_path)) == "20260101T000005Z.json"


def test_a_file_overwritten_in_the_same_second_is_new_too(tmp_path: Path) -> None:
    folder = tmp_path / DIRECTORY
    folder.mkdir(parents=True)
    file = folder / "20260101T000000Z.json"
    file.write_text("{}")
    before = recorder.snapshot(tmp_path)
    file.write_text('{"again": 1}')
    import os

    os.utime(file, ns=(1, 1))  # a clock that did not move must not hide the rewrite

    assert recorder.fresh(before, recorder.snapshot(tmp_path)) == file.name


def test_nothing_new_means_no_recording(tmp_path: Path) -> None:
    folder = tmp_path / DIRECTORY
    folder.mkdir(parents=True)
    (folder / "20260101T000000Z.json").write_text("{}")
    before = recorder.snapshot(tmp_path)

    assert recorder.fresh(before, recorder.snapshot(tmp_path)) is None and recorder.snapshot(tmp_path / "nowhere") == {}


# -- stopping --------------------------------------------------------------------------------------------------------------


def test_stopping_also_ends_a_process_that_ignores_the_polite_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import signal as sig

    child = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    leader = f"import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', {child!r}]); print(p.pid, flush=True); time.sleep(60)"
    monkeypatch.setattr(recorder, "record_argv", lambda root, python=None: [sys.executable, "-c", leader])
    job = recorder.RecordJob(tmp_path)
    try:
        deadline = time.monotonic() + 20
        while not job.tail(1) and time.monotonic() < deadline:
            time.sleep(0.05)
        pid = int(job.tail(1)[0])
        job.stop()

        for _ in range(100):  # the kernel needs a moment to reap the killed child
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(pid, sig.SIGKILL)
            raise AssertionError("the process that ignores SIGTERM survived stop()")
    finally:
        job.close()


def test_no_recording_shows_the_last_lines_of_the_output() -> None:
    text = recorder.finished_text(saved=None, exit_code=2, seconds=0.5, ran=0, matched=0, tail=["one", "Error: needs Python 3.12"], stopped=False)

    assert text.startswith("Not recorded") and "needs Python 3.12" in text and "exited with 2" in text


def test_a_stopped_run_says_it_was_stopped() -> None:
    text = recorder.finished_text(saved=None, exit_code=-15, seconds=2.0, ran=0, matched=0, tail=[], stopped=True)

    assert text.startswith("Stopped") and "no new recording" in text


def test_output_with_control_characters_cannot_reach_the_screen() -> None:
    text = recorder.finished_text(saved=None, exit_code=1, seconds=1.0, ran=0, matched=0, tail=["\x1b[2Jgone"], stopped=False)

    assert "\x1b" not in text


# -- the keys ------------------------------------------------------------------------------------------------------------


def test_r_alone_asks_and_runs_nothing(repo: GitRepo, fake_job) -> None:
    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R")
            await _until(pilot, lambda: "y = yes" in _status(app), "the question")

    import asyncio

    asyncio.run(scenario())

    assert fake_job.started == []


@pytest.mark.parametrize("key", ["n", "x", "escape", "q"])
def test_any_key_but_y_cancels_and_q_does_not_quit(repo: GitRepo, fake_job, key: str) -> None:
    alive = {}

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R")
            await _until(pilot, lambda: "y = yes" in _status(app), "the question")
            await pilot.press(key)
            await _until(pilot, lambda: "cancelled" in _status(app), "the cancel message")
            alive["running"] = app.is_running

    import asyncio

    asyncio.run(scenario())

    assert fake_job.started == [] and alive["running"]


def test_y_starts_one_run_and_the_status_line_says_so(repo: GitRepo, fake_job) -> None:
    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: "Recording" in _status(app), "the running message")
            assert "R again to stop" in _status(app)
            await pilot.press("R")  # stops it, it does not ask again or start a second run
            await _until(pilot, lambda: "Stopped" in _status(app), "the stopped message")

    import asyncio

    asyncio.run(scenario())

    assert fake_job.started == [repo.root.resolve()]


def test_a_run_that_saved_nothing_says_not_recorded(repo: GitRepo, fake_job, monkeypatch: pytest.MonkeyPatch) -> None:
    def start_failed(self, root, python=None):
        self.root, self.lines, self.stopped = root, ["Error: cannot run"], False
        self.code = 2

    monkeypatch.setattr(FakeJob, "__init__", start_failed)

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: "Not recorded" in _status(app), "the failure message")
            assert "cannot run" in _status(app)

    import asyncio

    asyncio.run(scenario())


def test_a_child_that_cannot_start_is_reported_not_raised(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self, root, python=None):
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr("argus.tui.app.RecordJob", type("Refuses", (), {"__init__": refuse}))

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: "Not recorded" in _status(app), "the failure message")
            assert "could not be started" in _status(app)

    import asyncio

    asyncio.run(scenario())


def test_quitting_stops_the_child(repo: GitRepo, fake_job) -> None:
    jobs: list[FakeJob] = []

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: app._job is not None, "the job")
            jobs.append(app._job)
            await pilot.press("q")

    import asyncio

    asyncio.run(scenario())

    assert jobs and jobs[0].stopped


# -- the real thing ------------------------------------------------------------------------------------------------------


def test_the_real_command_saves_a_recording_the_app_then_reads(repo: GitRepo) -> None:
    seen = {}

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: _status(app).count("Recorded") >= 1 or "Not recorded" in _status(app), "the result")
            seen["status"] = _status(app)
            seen["recording"] = app.project.recording()[0]

    import asyncio

    asyncio.run(scenario())

    assert "Recorded" in seen["status"] and "Not recorded" not in seen["status"], seen["status"]
    assert seen["recording"] is not None and list((repo.root / DIRECTORY).glob("*.json"))


# -- failures inside the follower must not wedge the key -----------------------------------------------------------------


def _finished_job(code: int = 0, **extra):
    class Done(FakeJob):
        def __init__(self, root: Path, python=None) -> None:
            super().__init__(root)
            self.code = code
            for name, value in extra.items():
                setattr(self, name, value)

    return Done


def test_a_failure_while_reading_the_result_is_shown_and_the_key_works_again(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.tui.app.RecordJob", _finished_job())
    monkeypatch.setattr("argus.tui.app.finished_text", lambda **kw: 1 / 0)
    state = {}

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: "could not be read" in _status(app), "the failure message")
            await _until(pilot, lambda: app._job is None, "the job to be cleared")
            await pilot.press("R")
            await _until(pilot, lambda: "y = yes" in _status(app), "the key to ask again")
            state["ok"] = True

    import asyncio

    asyncio.run(scenario())

    assert state["ok"]


def test_a_damaged_recording_file_is_reported_with_the_reason(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    class Writes(FakeJob):
        def __init__(self, root: Path, python=None) -> None:
            super().__init__(root)
            folder = root / DIRECTORY
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "29990101T000000Z.json").write_text("not json")
            self.code = 0

    monkeypatch.setattr("argus.tui.app.RecordJob", Writes)

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: "cannot be read" in _status(app), "the damaged-file message")
            assert "Recorded:" not in _status(app)

    import asyncio

    asyncio.run(scenario())
