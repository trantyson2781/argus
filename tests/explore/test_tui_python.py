"""Which Python the R and C keys use, the check before recording, and the ? help screen.

Rules under test: the Python is chosen in a fixed order (ARGUS_PYTHON, then the one-line file .argus/python, then the repository's .venv, then the Python that
runs Argus) and the question names it with its source; a choice that points at nothing runnable is an error and never falls back to another Python; R
refuses a Python older than 3.12 before running anything (recording cannot work there) and says what to do; C passes the chosen Python to `argus check`;
? shows every key the app binds, the chosen Python and the tmux settings, and Esc closes it.
"""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.widgets import Static

from argus.tui import pythonchoice
from argus.tui.app import ArgusApp
from argus.tui.helptext import help_lines
from tests.explore.diff_helpers import GitRepo

SIZE = (130, 45)


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# -- the order -------------------------------------------------------------------------------------------------------------


def test_the_python_running_argus_is_the_last_resort(tmp_path: Path) -> None:
    choice = pythonchoice.chosen_python(tmp_path, {})

    assert choice.path == sys.executable and "runs Argus" in choice.source and choice.error is None


def test_the_repository_venv_comes_before_that(tmp_path: Path) -> None:
    venv = _exe(tmp_path / ".venv" / "bin" / "python")

    choice = pythonchoice.chosen_python(tmp_path, {})

    assert choice.path == str(venv) and ".venv" in choice.source


def test_the_file_comes_before_the_venv(tmp_path: Path) -> None:
    _exe(tmp_path / ".venv" / "bin" / "python")
    other = _exe(tmp_path / "envs" / "py312" / "bin" / "python")
    (tmp_path / ".argus").mkdir()
    (tmp_path / ".argus" / "python").write_text(f"{other}\n", encoding="utf-8")

    choice = pythonchoice.chosen_python(tmp_path, {})

    assert choice.path == str(other) and choice.source == ".argus/python"


def test_the_environment_variable_comes_before_the_file(tmp_path: Path) -> None:
    file_python, env_python = _exe(tmp_path / "a" / "python"), _exe(tmp_path / "b" / "python")
    (tmp_path / ".argus").mkdir()
    (tmp_path / ".argus" / "python").write_text(str(file_python), encoding="utf-8")

    choice = pythonchoice.chosen_python(tmp_path, {"ARGUS_PYTHON": str(env_python)})

    assert choice.path == str(env_python) and choice.source == "ARGUS_PYTHON"


def test_a_relative_path_in_the_file_is_relative_to_the_repository(tmp_path: Path) -> None:
    _exe(tmp_path / "tools" / "py" / "bin" / "python")
    (tmp_path / ".argus").mkdir()
    (tmp_path / ".argus" / "python").write_text("tools/py/bin/python\n", encoding="utf-8")

    assert pythonchoice.chosen_python(tmp_path, {}).path == str(tmp_path / "tools" / "py" / "bin" / "python")


def test_an_empty_file_or_variable_is_ignored(tmp_path: Path) -> None:
    (tmp_path / ".argus").mkdir()
    (tmp_path / ".argus" / "python").write_text("  \n", encoding="utf-8")

    choice = pythonchoice.chosen_python(tmp_path, {"ARGUS_PYTHON": ""})

    assert choice.path == sys.executable and choice.error is None


@pytest.mark.parametrize("where", ["env", "file"])
def test_a_choice_that_points_at_nothing_runnable_is_an_error_and_never_falls_back(tmp_path: Path, where: str) -> None:
    _exe(tmp_path / ".venv" / "bin" / "python")
    missing = str(tmp_path / "nowhere" / "python")
    env = {}
    if where == "env":
        env["ARGUS_PYTHON"] = missing
    else:
        (tmp_path / ".argus").mkdir()
        (tmp_path / ".argus" / "python").write_text(missing, encoding="utf-8")

    choice = pythonchoice.chosen_python(tmp_path, env)

    assert choice.error and missing in choice.error and ("ARGUS_PYTHON" if where == "env" else ".argus/python") in choice.error


def test_a_folder_or_a_file_that_cannot_run_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    plain = tmp_path / "plain"
    plain.write_text("x")
    plain.chmod(0o644)

    assert pythonchoice.chosen_python(tmp_path, {"ARGUS_PYTHON": str(tmp_path / "d")}).error
    assert pythonchoice.chosen_python(tmp_path, {"ARGUS_PYTHON": str(plain)}).error


def test_control_characters_in_the_file_cannot_reach_the_screen(tmp_path: Path) -> None:
    (tmp_path / ".argus").mkdir()
    (tmp_path / ".argus" / "python").write_text("/no/\x1b[2Jwhere", encoding="utf-8")

    assert "\x1b" not in (pythonchoice.chosen_python(tmp_path, {}).error or "")


# -- its version -----------------------------------------------------------------------------------------------------------


def test_the_version_of_a_real_python_is_read() -> None:
    assert pythonchoice.python_version(sys.executable) >= (3, 12)


def test_a_python_that_does_not_exist_has_no_version() -> None:
    assert pythonchoice.python_version("/nowhere/python") is None


def test_output_that_is_not_a_version_is_no_version(tmp_path: Path) -> None:
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\necho hello\n")
    fake.chmod(0o755)

    assert pythonchoice.python_version(str(fake)) is None


# -- the keys --------------------------------------------------------------------------------------------------------------


def _status(app: ArgusApp) -> str:
    return str(app.query_one("#status", Static).content)


async def _until(pilot, predicate: Callable[[], bool], what: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"never saw: {what}")


@pytest.fixture
def repo(git_repo: GitRepo) -> GitRepo:
    git_repo.write_files({"shop/__init__.py": "", "shop/billing.py": "def charge(a):\n    return a\n"})
    git_repo.commit("baseline")
    return git_repo


def _use(repo: GitRepo, python: Path | str) -> None:
    (repo.root / ".argus").mkdir(exist_ok=True)
    (repo.root / ".argus" / "python").write_text(str(python), encoding="utf-8")


class Spy:
    """Stands in for RecordJob / CheckJob and remembers what it was started with."""

    calls: list[tuple] = []

    def __init__(self, root, *args):
        Spy.calls.append((root, *args))
        self.stopped, self.code = False, None

    def poll(self):
        return self.code

    def seconds(self):
        return 1.0

    def stop(self, wait=True):
        self.stopped, self.code = True, -15

    def escalate(self):
        pass

    def sweep(self):
        pass

    def tail(self, lines=3):
        return []

    def output(self):
        return []

    def close(self):
        pass


def test_the_question_names_the_python_and_where_it_came_from(repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    py = _exe(tmp_path / "envs" / "py312" / "bin" / "python")
    _use(repo, py)
    monkeypatch.delenv("ARGUS_PYTHON", raising=False)
    shown = {}

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R")
            await _until(pilot, lambda: "y = yes" in _status(app), "the record question")
            shown["record"] = _status(app)
            await pilot.press("n")
            await pilot.pause(0.2)
            await pilot.press("C")
            await _until(pilot, lambda: "argus check" in _status(app) and "y = yes" in _status(app), "the check question")
            shown["check"] = _status(app)

    asyncio.run(scenario())

    for text in shown.values():
        assert str(py) in text and ".argus/python" in text


def test_r_refuses_a_python_older_than_3_12_before_running_anything(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    Spy.calls = []
    monkeypatch.setattr("argus.tui.app.RecordJob", Spy)
    monkeypatch.setattr("argus.tui.app.python_version", lambda path: (3, 10))

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: "3.10" in _status(app), "the refusal")
            assert "3.12" in _status(app) and "ARGUS_PYTHON" in _status(app) and ".argus/python" in _status(app)

    asyncio.run(scenario())

    assert Spy.calls == []


def test_r_refuses_a_python_whose_version_cannot_be_read(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    Spy.calls = []
    monkeypatch.setattr("argus.tui.app.RecordJob", Spy)
    monkeypatch.setattr("argus.tui.app.python_version", lambda path: None)

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: "version" in _status(app) and "Not recorded" in _status(app), "the refusal")

    asyncio.run(scenario())

    assert Spy.calls == []


def test_r_starts_with_the_chosen_python(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    Spy.calls = []
    monkeypatch.setattr("argus.tui.app.RecordJob", Spy)
    monkeypatch.setattr("argus.tui.app.python_version", lambda path: (3, 12))
    monkeypatch.delenv("ARGUS_PYTHON", raising=False)
    _use(repo, sys.executable)

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: bool(Spy.calls), "the job")

    asyncio.run(scenario())

    assert Spy.calls[0][0] == repo.root.resolve() and Spy.calls[0][1] == sys.executable


def test_c_passes_the_chosen_python_to_the_check(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    Spy.calls = []
    monkeypatch.setattr("argus.tui.app.CheckJob", Spy)
    monkeypatch.delenv("ARGUS_PYTHON", raising=False)
    _use(repo, sys.executable)

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("C", "y")
            await _until(pilot, lambda: bool(Spy.calls), "the job")

    asyncio.run(scenario())

    assert Spy.calls[0][2] == sys.executable


@pytest.mark.parametrize("key", ["R", "C"])
def test_a_choice_that_points_at_nothing_asks_nothing_and_says_why(repo: GitRepo, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    Spy.calls = []
    monkeypatch.setattr("argus.tui.app.RecordJob", Spy)
    monkeypatch.setattr("argus.tui.app.CheckJob", Spy)
    monkeypatch.setenv("ARGUS_PYTHON", "/nowhere/python")

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press(key)
            await _until(pilot, lambda: "ARGUS_PYTHON" in _status(app) and "/nowhere/python" in _status(app), "the error")
            assert "y = yes" not in _status(app)

    asyncio.run(scenario())

    assert Spy.calls == []


# -- the help screen -------------------------------------------------------------------------------------------------------


def test_the_help_text_mentions_every_key_the_app_binds(tmp_path: Path) -> None:
    text = "\n".join(help_lines(pythonchoice.chosen_python(tmp_path, {})))
    bound = {b.key for b in ArgusApp.BINDINGS}
    shown = {"slash": "/", "question_mark": "?", "escape": "Esc"}

    for key in bound:
        assert shown.get(key, key) in text, f"the help does not mention {key}"


def test_the_help_text_names_the_python_and_the_agent_settings(tmp_path: Path) -> None:
    text = "\n".join(help_lines(pythonchoice.chosen_python(tmp_path, {})))

    assert sys.executable in text and "ARGUS_PYTHON" in text and ".argus/python" in text and "ARGUS_AGENT_PANE" in text


def test_a_choice_with_an_error_is_shown_in_the_help(tmp_path: Path) -> None:
    text = "\n".join(help_lines(pythonchoice.chosen_python(tmp_path, {"ARGUS_PYTHON": "/nowhere/python"})))

    assert "/nowhere/python" in text and "not" in text


def test_question_mark_opens_the_help_and_escape_closes_it(repo: GitRepo) -> None:
    seen = {}

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("question_mark")
            await _until(pilot, lambda: len(app.screen_stack) > 1, "the help screen")
            seen["text"] = " ".join(str(w.content) for w in app.screen.query(Static))
            await pilot.press("escape")
            await _until(pilot, lambda: len(app.screen_stack) == 1, "the help to close")
            await pilot.press("q")

    asyncio.run(scenario())

    assert "R " in seen["text"] and "record" in seen["text"].lower() and "ARGUS_PYTHON" in seen["text"]


def test_a_recording_made_with_the_chosen_python_is_reported_as_this_runs(repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a launcher that is not the repository's .venv python and not sys.executable, but ends up running the same interpreter
    launcher = tmp_path / "bin" / "py"
    launcher.parent.mkdir()
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    launcher.chmod(0o755)
    monkeypatch.delenv("ARGUS_PYTHON", raising=False)
    _use(repo, launcher)
    repo.write_files({"tests/test_billing.py": "from shop.billing import charge\n\n\ndef test_charge():\n    assert charge(1) == 1\n"})
    state = {}

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            await pilot.press("R", "y")
            await _until(pilot, lambda: app._job is None and any(w in _status(app) for w in ("Recorded", "Not recorded", "not from it", "Nothing")), "the result", 120)
            state["status"] = _status(app)

    asyncio.run(scenario())

    assert "Recorded" in state["status"] and "not from it" not in state["status"], state["status"]


# -- found by the audit ----------------------------------------------------------------------------------------------------


def test_a_home_folder_of_nobody_is_an_error_not_a_crash(tmp_path: Path) -> None:
    choice = pythonchoice.chosen_python(tmp_path, {"ARGUS_PYTHON": "~no_such_user_xyz/bin/python"})

    assert choice.error and "ARGUS_PYTHON" in choice.error


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any folder")
def test_a_folder_that_cannot_be_read_is_an_error_not_a_crash(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "python").write_text("x")
    locked.chmod(0)
    try:
        choice = pythonchoice.chosen_python(tmp_path, {"ARGUS_PYTHON": str(locked / "python")})
    finally:
        locked.chmod(0o755)

    assert choice.error


def test_a_venv_python_that_is_a_broken_link_is_an_error_and_not_a_quiet_fallback(tmp_path: Path) -> None:
    link = tmp_path / ".venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(tmp_path / "gone" / "python")

    choice = pythonchoice.chosen_python(tmp_path, {})

    assert choice.error and ".venv" in choice.error


def test_a_python_that_hangs_gets_no_version_after_its_time_limit(tmp_path: Path) -> None:
    slow = tmp_path / "python"
    slow.write_text("#!/bin/sh\nsleep 5\n")
    slow.chmod(0o755)
    began = time.monotonic()

    assert pythonchoice.python_version(str(slow), timeout=0.3) is None and time.monotonic() - began < 3
    assert pythonchoice.VERSION_TIMEOUT <= 3  # the screen thread waits for it


def test_a_failed_choice_forgets_the_question_before_it(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_PYTHON", "/nowhere/python")
    state = {}

    async def scenario() -> None:
        app = ArgusApp(repo.root, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            app._confirming, app._choice = "record", pythonchoice.PythonChoice(sys.executable, "left over")
            state["result"] = app._choose_python()
            state["confirming"], state["choice"] = app._confirming, app._choice

    asyncio.run(scenario())

    assert state == {"result": None, "confirming": "", "choice": None}
