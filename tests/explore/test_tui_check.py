"""The C key of the TUI: it asks, then runs `argus check` (the tests that reach the uncommitted change, on throwaway copies) as a child process.

Rules under test: C alone runs nothing and any key but `y` (even `q`) cancels; one child at a time, and C or R during a run stops it; the sentence
after the run follows the command's exit codes (0 ran and none newly failing, 1 newly failing, 2 not decided, 3 no test ran); an unreadable
output is said to be unreadable, never a green result; the TUI never imports the checker (tests/record/test_guard.py); the real command runs.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.widgets import Static

from argus.tui import checker
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


# -- the command and the sentence ------------------------------------------------------------------------------------------


def test_the_child_is_the_check_command_of_this_repository(tmp_path: Path) -> None:
    assert checker.check_argv(tmp_path, None) == [sys.executable, "-P", "-m", "argus.cli", "check", str(tmp_path)]
    assert checker.check_argv(tmp_path, "main")[-1] == "--base=main"
    assert checker.check_argv(tmp_path, "-x")[-1] == "--base=-x"  # one argument: a ref that starts with a dash can never be read as an option


def test_the_question_says_what_is_compared_and_that_tests_run(tmp_path: Path) -> None:
    text = checker.question(tmp_path, None)

    assert "HEAD" in text and "executes your code" in text and "y = yes" in text and "main" in checker.question(tmp_path, "main")


OK = ["Check  HEAD (abc1234) -> working tree", "Selected 9 tests: 5 call it directly [I]", "Result [O]: 9 ran, none failed. Passing tests show those tests did not notice the change, not that the change is right."]
BAD = [
    "Result [O]: 9 ran, 2 newly failing, 0 already failing, 0 flaky.", "  newly failing [O] (passed on the base, fail with your change):",
    "    tests/test_a.py::test_one", "    tests/test_a.py::test_two", "  already failing [O] (also fail on the base):", "    tests/test_b.py::test_old",
]


def test_exit_0_reports_the_result_line_with_its_caveat() -> None:
    text = checker.finished_text(0, 14.0, OK, stopped=False)

    assert text.startswith("Check vs HEAD:") and "9 ran, none failed" in text and "not that the change is right" in text


def test_the_base_is_part_of_the_sentence() -> None:
    assert checker.finished_text(0, 1.0, OK, stopped=False, base="main").startswith("Check vs main:")


def test_a_green_result_with_warnings_says_how_many_and_where_to_read_them() -> None:
    lines = [
        "Selected 10 tests: 2 call it directly [I]", "  ! 9 selected tests not collected by pytest and not run: a, b, c", "  ! the recording is older than a change in m.py",
        "Result [O]: 1 ran, none failed. Passing tests show those tests did not notice the change, not that the change is right.",
    ]

    text = checker.finished_text(0, 1.0, lines, stopped=False)

    assert "1 ran, none failed" in text and "2 warnings" in text and "argus check" in text


def test_exit_0_with_already_failing_or_flaky_tests_is_readable_and_not_green() -> None:
    lines = ["Result [O]: 9 ran, 0 newly failing, 2 already failing, 1 flaky.", "  already failing [O] (also fail on the base):", "    tests/t.py::a"]

    text = checker.finished_text(0, 1.0, lines, stopped=False)

    assert text.startswith("Check vs HEAD:") and "2 already failing" in text and "1 flaky" in text and "none failed" not in text and "could not be read" not in text


def test_a_report_without_its_result_line_after_exit_1_is_called_failing_not_unreadable_green() -> None:
    text = checker.finished_text(1, 1.0, ["    t.py::t1", "    t.py::t2"], stopped=False)

    assert text.startswith("Check could not be read") and "exit 1" in text and "argus check" in text


def test_a_child_that_finished_before_the_stop_arrived_keeps_its_real_result() -> None:
    assert "NEWLY FAILING" in checker.finished_text(1, 1.0, BAD, stopped=True)


def test_exit_1_names_the_newly_failing_tests_and_only_those() -> None:
    text = checker.finished_text(1, 14.0, BAD, stopped=False)

    assert "NEWLY FAILING" in text and "2 newly failing" in text and "test_one" in text and "test_two" in text and "test_old" not in text


def test_many_newly_failing_tests_are_cut_and_counted() -> None:
    lines = ["Result [O]: 9 ran, 5 newly failing, 0 already failing, 0 flaky.", "  newly failing [O] (x):", *(f"    t.py::t{i}" for i in range(5))]

    text = checker.finished_text(1, 1.0, lines, stopped=False)

    assert "t0" in text and "t2" in text and "t3" not in text and "+2 more" in text


def test_exit_2_is_not_decided_with_the_reason() -> None:
    text = checker.finished_text(2, 1.0, ["Result: not decided. The package tests is imported from outside the copy (an editable install?)."], stopped=False)

    assert text.startswith("Check not decided") and "editable install" in text and "ran" not in text.split(":")[0]


def test_exit_3_says_no_test_ran_and_is_not_green() -> None:
    text = checker.finished_text(3, 1.0, ["Result: none of the selected tests could be run. no test found that reaches this change (a lower bound)"], stopped=False)

    assert "no test ran" in text.lower() and "none failed" not in text


def test_output_without_a_result_line_is_unreadable_and_shows_the_tail() -> None:
    text = checker.finished_text(1, 1.0, ["Traceback (most recent call last):", "ModuleNotFoundError: typer"], stopped=False)

    assert text.startswith("Check could not be read") and "ModuleNotFoundError" in text and "newly" not in text.lower()


def test_an_exit_code_that_disagrees_with_the_result_line_is_unreadable() -> None:
    assert checker.finished_text(0, 1.0, BAD, stopped=False).startswith("Check could not be read")  # exit 0 but "newly failing" in the output


def test_a_stopped_check_says_so() -> None:
    assert checker.finished_text(-15, 3.0, [], stopped=True).startswith("Check stopped")


def test_control_characters_in_the_output_cannot_reach_the_screen() -> None:
    assert "\x1b" not in checker.finished_text(1, 1.0, ["\x1b[2Jboom"], stopped=False)


# -- the keys --------------------------------------------------------------------------------------------------------------


class FakeCheck:
    started: list[Path] = []

    def __init__(self, root: Path, base: str | None, python=None) -> None:
        self.root, self.base, self.code, self.stopped, self.lines = root, base, None, False, OK
        FakeCheck.started.append(root)

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

    def output(self) -> list[str]:
        return self.lines

    def close(self) -> None:
        pass


@pytest.fixture
def fake_check(monkeypatch: pytest.MonkeyPatch) -> type[FakeCheck]:
    FakeCheck.started = []
    monkeypatch.setattr("argus.tui.app.CheckJob", FakeCheck)
    return FakeCheck


@pytest.fixture
def repo(git_repo: GitRepo) -> GitRepo:
    git_repo.write_files({
        "shop/__init__.py": "",
        "shop/billing.py": "def charge(amount):\n    return amount + 1\n",
        "tests/test_billing.py": "from shop.billing import charge\n\n\ndef test_charge():\n    assert charge(1) == 2\n",
    })
    git_repo.commit("baseline")
    return git_repo


def _run(coro) -> None:
    asyncio.run(coro)


def test_c_alone_asks_and_runs_nothing(repo: GitRepo, fake_check) -> None:
    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("C")
            await _until(pilot, lambda: "y = yes" in _status(app) and "argus check" in _status(app), "the question")

    _run(scenario())

    assert fake_check.started == []


@pytest.mark.parametrize("key", ["n", "escape", "q"])
def test_any_key_but_y_cancels(repo: GitRepo, fake_check, key: str) -> None:
    state = {}

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("C")
            await _until(pilot, lambda: "y = yes" in _status(app), "the question")
            await pilot.press(key)
            await _until(pilot, lambda: "cancelled" in _status(app), "the cancel message")
            state["alive"] = app.is_running

    _run(scenario())

    assert fake_check.started == [] and state["alive"]


def test_y_runs_the_check_and_the_result_reaches_the_status_line(repo: GitRepo, fake_check, monkeypatch: pytest.MonkeyPatch) -> None:
    class Quick(FakeCheck):
        def __init__(self, root: Path, base: str | None, python=None) -> None:
            super().__init__(root, base)
            self.code, self.lines = 1, BAD

    monkeypatch.setattr("argus.tui.app.CheckJob", Quick)

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("C", "y")
            await _until(pilot, lambda: "NEWLY FAILING" in _status(app), "the result")
            assert "test_one" in _status(app) and app._job is None

    _run(scenario())


def test_c_during_a_run_stops_it(repo: GitRepo, fake_check) -> None:
    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("C", "y")
            await _until(pilot, lambda: "Checking" in _status(app) and "C again to stop" in _status(app), "the running message")
            await pilot.press("C")
            await _until(pilot, lambda: "Check stopped" in _status(app), "the stopped message")

    _run(scenario())

    assert len(fake_check.started) == 1


def test_r_during_a_check_stops_the_check_and_does_not_start_a_recording(repo: GitRepo, fake_check) -> None:
    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("C", "y")
            await _until(pilot, lambda: "Checking" in _status(app), "the running message")
            await pilot.press("R")
            await _until(pilot, lambda: "Check stopped" in _status(app), "the stopped message")
            assert "Record" not in _status(app)

    _run(scenario())


def test_quitting_stops_the_check(repo: GitRepo, fake_check) -> None:
    jobs = []

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("C", "y")
            await _until(pilot, lambda: app._job is not None, "the job")
            jobs.append(app._job)
            await pilot.press("q")

    _run(scenario())

    assert jobs and jobs[0].stopped


def test_a_check_that_cannot_start_is_reported_not_raised(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self, root, base, python=None):
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr("argus.tui.app.CheckJob", type("Refuses", (), {"__init__": refuse}))

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("C", "y")
            await _until(pilot, lambda: "could not be started" in _status(app), "the failure message")

    _run(scenario())


# -- the real thing --------------------------------------------------------------------------------------------------------


def test_the_real_check_names_a_test_the_change_breaks(repo: GitRepo) -> None:
    repo.write("shop/billing.py", "def charge(amount):\n    return amount + 5\n")
    state = {}

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("C", "y")
            await _until(pilot, lambda: app._job is None and ("NEWLY FAILING" in _status(app) or "not" in _status(app).split("Check", 1)[-1]), "the result", 120)
            state["status"] = _status(app)

    _run(scenario())

    assert "NEWLY FAILING" in state["status"] and "test_charge" in state["status"], state["status"]
