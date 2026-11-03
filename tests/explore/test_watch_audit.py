"""`explore watch`, audit round 1: it survives failing polls and paints, keeps the last report on a failed refresh, and says what its baseline hides."""

from __future__ import annotations

import io
import os
import re
import signal
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from argus.cli import app
from argus.explore import cli, watch
from argus.explore.sides import DiffError, python_file_stamps
from argus.explore.watch import MAX_SETTLE_POLLS, WatchSettings, open_watch, run_watch
from tests.explore.diff_helpers import GitRepo

BILLING = "def charge(amount):\n    return amount + 1\n\n\ndef fee(amount):\n    return amount\n"
RUNNER = CliRunner()


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=240, highlight=False, force_terminal=False), buffer


def _drive(repo: GitRepo, sleeps: list, stop_after: int, include_status: bool = False) -> str:
    """Run the loop; the i-th sleep runs sleeps[i] (when it exists). Stops once `stop_after` sleeps have happened."""
    console, buffer = _console()
    count = {"n": 0}

    def sleep(_seconds: float) -> None:
        if count["n"] < len(sleeps) and sleeps[count["n"]] is not None:
            sleeps[count["n"]]()
        count["n"] += 1

    with open_watch(repo.root, None, False) as settings:
        run_watch(settings, console, sleep=sleep, should_stop=lambda: count["n"] >= stop_after, details=True)
    return buffer.getvalue()


# -- a failing poll does not end the session -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("error", [DiffError("git ls-files timed out after 60s."), PermissionError("denied")])
def test_a_failing_poll_is_reported_once_and_the_loop_recovers(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    real = watch.python_file_stamps
    state = {"n": 0}

    def flaky(path: Path):
        state["n"] += 1
        if 2 <= state["n"] <= 4:  # three polls in a row fail
            raise error
        return real(path)

    monkeypatch.setattr(watch, "python_file_stamps", flaky)
    edit = lambda: shop_repo.write("shop/api.py", "def v():\n    return 1\n")  # noqa: E731

    out = _drive(shop_repo, [None, None, None, None, edit, None, None], stop_after=8)

    assert out.count("poll failed") == 1 and "retrying" in out  # one note, not one per poll
    assert "shop.api.v" in out.split("poll failed")[-1]  # after recovering, the edit is painted


def test_a_missing_watched_directory_says_so_instead_of_git_is_missing(tmp_path: Path) -> None:
    with pytest.raises(DiffError, match="no longer exists"):
        python_file_stamps(tmp_path / "gone")


def test_a_paint_that_fails_is_shown_and_the_loop_continues(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    real = watch.print_diff
    state = {"n": 0}

    def flaky(console: Console, view: object) -> None:
        state["n"] += 1
        if state["n"] == 2:
            raise ValueError("cannot format")
        real(console, view)

    monkeypatch.setattr(watch, "print_diff", flaky)
    edit = lambda n: (lambda: shop_repo.write("shop/api.py", f"def w{n}():\n    return {n}\n"))  # noqa: E731

    out = _drive(shop_repo, [edit(1), None, None, edit(2), None, None], stop_after=7)

    assert "refresh failed: ValueError: cannot format" in out and "shop.api.w2" in out.split("refresh failed")[-1]


def test_a_closed_pipe_ends_the_command_quietly(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_a: object, **_k: object) -> None:
        raise BrokenPipeError

    monkeypatch.setattr(cli, "run_watch", broken)

    result = RUNNER.invoke(app, ["explore", "watch", str(shop_repo.root)])

    assert result.exit_code == 0 and "Traceback" not in result.output and result.exception is None


# -- failures are readable and the last report stays ---------------------------------------------------------------------------------


def test_a_failed_refresh_names_where_it_failed(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_settings: WatchSettings):
        raise KeyError

    monkeypatch.setattr(watch, "refresh", boom)

    out = _drive(shop_repo, [None], stop_after=1)

    assert re.search(r"refresh failed: KeyError: \s*at test_watch_audit\.py:\d+ in boom", out) and "the next change tries again" in out


# -- edits during a paint and the settle bound -----------------------------------------------------------------------------------------


def test_an_edit_made_during_a_refresh_is_painted_by_the_next_one(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    real = watch.refresh
    state = {"n": 0}

    def refresh_then_edit(settings: WatchSettings):
        result = real(settings)
        state["n"] += 1
        if state["n"] == 1:  # the first (launch) paint is still running when the agent writes a file
            shop_repo.write("shop/api.py", "def late():\n    return 1\n")
        return result

    monkeypatch.setattr(watch, "refresh", refresh_then_edit)

    out = _drive(shop_repo, [None, None, None], stop_after=4)

    assert "shop.api.late" in out.split("watching |")[-1]


def test_a_file_that_keeps_changing_cannot_postpone_a_refresh_forever(shop_repo: GitRepo) -> None:
    console, buffer = _console()
    sleeps = {"n": 0}

    def sleep(_seconds: float) -> None:
        sleeps["n"] += 1
        shop_repo.write("shop/api.py", f"def v{sleeps['n']}():\n    return {sleeps['n']}\n")  # a different size or mtime every time

    with open_watch(shop_repo.root, None, False) as settings:
        run_watch(settings, console, sleep=sleep, should_stop=lambda: buffer.getvalue().count("watching |") >= 2, details=True)

    assert sleeps["n"] == 1 + MAX_SETTLE_POLLS  # one poll, then the settle wait gives up after the bound and refreshes anyway


# -- the cheap check ---------------------------------------------------------------------------------------------------------------


def test_stamps_see_a_same_size_edit_and_an_mtime_only_change(shop_repo: GitRepo) -> None:
    path = shop_repo.root / "shop/billing.py"
    before = python_file_stamps(shop_repo.root)

    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))  # same length, different bytes
    same_size = python_file_stamps(shop_repo.root)
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))  # identical bytes, different time
    touched = python_file_stamps(shop_repo.root)

    assert same_size["shop/billing.py"][1] == before["shop/billing.py"][1]
    assert same_size != before and touched != same_size


def test_stamps_leave_out_the_directories_the_scanner_skips(shop_repo: GitRepo) -> None:
    shop_repo.write(".venv/lib/site.py", "X = 1\n")
    shop_repo.write("node_modules/pkg/x.py", "X = 1\n")

    names = set(python_file_stamps(shop_repo.root))

    assert not any(n.startswith((".venv/", "node_modules/")) for n in names) and "shop/billing.py" in names


# -- what the baseline hides, said on screen --------------------------------------------------------------------------------------


def test_the_status_line_says_what_each_baseline_leaves_out(shop_repo: GitRepo) -> None:
    launch = _drive(shop_repo, [None], stop_after=1)
    console, buffer = _console()
    with open_watch(shop_repo.root, "HEAD", False) as settings:
        run_watch(settings, console, sleep=lambda _s: None, should_stop=lambda: True, details=True)

    assert "(earlier edits not shown)" in launch
    assert "(that exact commit, not the merge-base)" in buffer.getvalue()


def test_a_baseline_taken_while_a_file_does_not_parse_is_flagged_as_incomplete(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount:\n")

    out = _drive(shop_repo, [None], stop_after=1)

    assert "baseline incomplete: 1 file did not parse at launch (relaunch to fix)" in out


# -- signals ------------------------------------------------------------------------------------------------------------------------


def test_the_command_stops_on_ctrl_c_and_puts_the_signal_handlers_back(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(*_a: object, **_k: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_watch", interrupted)
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP))

    result = RUNNER.invoke(app, ["explore", "watch", str(shop_repo.root)])

    assert result.exit_code == 0 and "stopped." in result.stdout
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)) == before


def test_the_stop_handler_turns_sigterm_and_sighup_into_the_same_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt):
        cli._stop_on_signal(signal.SIGTERM, None)


def test_a_failed_first_stamp_read_does_not_make_every_file_look_edited(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    real = watch.python_file_stamps
    state = {"reads": 0, "sleeps": 0}

    def first_read_fails(path: Path):
        state["reads"] += 1
        if state["reads"] == 1:
            raise DiffError("git ls-files timed out after 60s.")
        return real(path)

    def sleep(_seconds: float) -> None:
        state["sleeps"] += 1

    monkeypatch.setattr(watch, "python_file_stamps", first_read_fails)

    with open_watch(shop_repo.root, None, False) as settings:
        events = list(watch.watch_events(settings, sleep=sleep, should_stop=lambda: state["sleeps"] >= 3))

    assert [event.kind for event in events] == ["refreshed"]  # nothing was edited: the first good reading is only the baseline
