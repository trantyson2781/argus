"""The TUI (`argus`): four views over the existing Argus code, driven headless with Textual's pilot.

Rules under test: nothing is analysed in the UI (it shows what `watch`, `map`, `symbol` and `diff` produce); a save in the repository
reaches the Changes view without a keypress and without blocking keys; Enter on a changed row opens that symbol; `r` starts fresh;
`e` opens the editor at the row's file and line; temporary files are removed on exit; the bare `argus` command falls back to help
when it is not attached to a terminal.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.widgets import Input, OptionList, Static
from typer.testing import CliRunner

from argus.cli import app as cli_app
from argus.tui.app import ArgusApp, editor_command
from tests.explore.diff_helpers import GIT_ENV, GitRepo
from tests.explore.test_watch import BILLING

RUNNER = CliRunner()
SIZE = (120, 40)


def _text(app: ArgusApp, widget_id: str) -> str:
    return str(app.query_one(f"#{widget_id}", Static).content)


def _rows(app: ArgusApp) -> list[str]:
    box = app.query_one("#rows", OptionList)
    return [str(box.get_option_at_index(i).prompt) for i in range(box.option_count)]


async def _until(pilot, predicate: Callable[[], bool], what: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"never saw: {what}")


def _app(repo: GitRepo) -> ArgusApp:
    return ArgusApp(repo.root, poll=0.05, debounce=0.02)


def _add_function(repo: GitRepo, name: str = "extra") -> None:
    repo.write("shop/billing.py", BILLING + f"\n\ndef {name}():\n    return 1\n")


def test_the_first_view_says_nothing_changed_yet(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")

    asyncio.run(scenario())


def test_a_save_reaches_the_changes_view_without_a_keypress(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            _add_function(shop_repo)
            await _until(pilot, lambda: any("billing.extra" in row for row in _rows(app)), "the new function row")
            assert "edited shop/billing.py" in _text(app, "status")
            assert any("NEW" in row for row in _rows(app))

    asyncio.run(scenario())


def test_keys_switch_between_the_four_views_and_the_map_is_the_level_map(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("2")
            await _until(pilot, lambda: "LEVEL MAP" in _text(app, "map-text"), "the level map")
            await pilot.press("4")
            await _until(pilot, lambda: _text(app, "diff-text").startswith("Diff"), "the diff report")
            assert app.query_one("#views").current == "diff"
            await pilot.press("1")
            assert app.query_one("#views").current == "changes"

    asyncio.run(scenario())


def test_enter_on_a_changed_row_opens_that_symbol(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            _add_function(shop_repo)
            await _until(pilot, lambda: any("billing.extra" in row for row in _rows(app)), "the new function row")
            app.query_one("#rows", OptionList).focus()
            await pilot.press("down", "enter")  # the first row is highlighted by the app; Enter selects it
            await _until(pilot, lambda: "shop.billing.extra" in _text(app, "symbol-text"), "the symbol view")
            assert app.query_one("#views").current == "symbol"

    asyncio.run(scenario())


def test_the_symbol_search_box_finds_a_name_and_says_when_there_is_none(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3")
            box = app.query_one("#query", Input)
            box.value = "charge"
            await box.action_submit()
            await _until(pilot, lambda: "shop.billing.charge" in _text(app, "symbol-text"), "the symbol summary")
            box.value = "no_such_symbol_anywhere"
            await box.action_submit()
            await _until(pilot, lambda: "No symbol matches" in _text(app, "symbol-text"), "the lookup error")

    asyncio.run(scenario())


def test_d_toggles_the_long_report_and_back(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            await pilot.press("d")
            await _until(pilot, lambda: "Not determinable" in _text(app, "details"), "the long report")
            await pilot.press("d")
            assert app.query_one("#details-scroll").display is False

    asyncio.run(scenario())


def test_r_starts_fresh_from_the_current_state(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            _add_function(shop_repo)
            await _until(pilot, lambda: any("billing.extra" in row for row in _rows(app)), "the new function row")
            await pilot.press("r")
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status") and not _rows(app), "a clean view")

    asyncio.run(scenario())


def test_e_opens_the_editor_at_the_rows_file_and_line(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[tuple[str, int]] = []

    async def scenario() -> None:
        app = _app(shop_repo)
        monkeypatch.setattr(app, "open_in_editor", lambda file, line: opened.append((file, line)))
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            _add_function(shop_repo)
            await _until(pilot, lambda: any("billing.extra" in row for row in _rows(app)), "the new function row")
            await pilot.press("e")
            await pilot.pause(0.1)

    asyncio.run(scenario())
    assert opened == [("shop/billing.py", len(BILLING.splitlines()) + 3)]  # two blank lines, then the new def


@pytest.mark.parametrize(
    ("editor", "expected"),
    [("vim", ["vim", "+12", "a/b.py"]), ("code", ["code", "-g", "a/b.py:12"]), ("nano -w", ["nano", "-w", "+12", "a/b.py"])],
)
def test_the_editor_command_is_built_for_the_editor_in_use(editor: str, expected: list[str]) -> None:
    assert editor_command(editor, "a/b.py", 12) == expected


def test_quitting_removes_every_temporary_directory(shop_repo: GitRepo, scratch_tmp: Path) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            await pilot.press("q")

    asyncio.run(scenario())
    deadline = time.monotonic() + 10
    while list(scratch_tmp.glob("argus-*")) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not list(scratch_tmp.glob("argus-*"))


def test_a_directory_outside_git_is_a_message_not_a_traceback(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = ArgusApp(tmp_path, poll=0.05, debounce=0.02)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "not inside a git repository" in _text(app, "status"), "the error message")

    asyncio.run(scenario())


def test_the_bare_command_prints_help_when_it_is_not_attached_to_a_terminal() -> None:
    result = RUNNER.invoke(cli_app, [])

    assert result.exit_code == 0 and "explore" in result.stdout and "bench" in result.stdout


@pytest.mark.parametrize("stop_signal", [signal.SIGHUP, signal.SIGTERM])
def test_end_to_end_a_closed_terminal_ends_the_app_and_leaves_no_temporary_files(shop_repo: GitRepo, tmp_path: Path, stop_signal: signal.Signals) -> None:
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    proc = subprocess.Popen(
        [sys.executable, "-c", "from argus.cli import app; app()", "-C", str(shop_repo.root)],
        stdin=slave, stdout=slave, stderr=slave, env={**os.environ, **GIT_ENV, "TMPDIR": str(scratch), "TERM": "xterm-256color"},
        start_new_session=True,
    )
    os.close(slave)
    seen = b""
    try:
        deadline = time.monotonic() + 60
        while b"nothing changed yet" not in seen and time.monotonic() < deadline:
            if select.select([master], [], [], 0.2)[0]:
                try:
                    seen += os.read(master, 65536)
                except OSError:
                    break
        assert b"nothing changed yet" in seen, seen[-500:]
        proc.send_signal(stop_signal)
        proc.wait(timeout=30)
        while select.select([master], [], [], 0.3)[0]:  # whatever the app printed while shutting down
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            seen += chunk
        assert b"Traceback" not in seen and b"SystemExit" not in seen, seen[-800:]
        assert proc.returncode == 128 + stop_signal
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)
    assert not list(scratch.glob("argus-*"))


def test_number_keys_switch_views_from_the_symbol_view_until_a_text_box_is_asked_for(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3")
            assert not isinstance(app.focused, Input) and app.focused.id == "symbol-scroll"  # the result pane, so arrows scroll it
            await pilot.press("4")
            assert app.query_one("#views").current == "diff"
            await pilot.press("slash")
            assert app.query_one("#views").current == "symbol" and isinstance(app.focused, Input)
            await pilot.press("4")  # typed into the box, not a view switch
            assert app.query_one("#query", Input).value == "4" and app.query_one("#views").current == "symbol"
            await pilot.press("escape", "1")
            assert app.query_one("#views").current == "changes"

    asyncio.run(scenario())


def test_b_and_h_edit_the_diff_refs_and_enter_compares_them(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING + "\n\ndef extra():\n    return 1\n")
    shop_repo.commit("add extra")

    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("b")
            assert app.query_one("#views").current == "diff" and app.focused is app.query_one("#base", Input)
            base = app.query_one("#base", Input)
            base.value = "HEAD~1"
            await base.action_submit()
            await _until(pilot, lambda: "shop.billing.extra" in _text(app, "diff-text"), "the added function in the diff")
            base.value = "no-such-ref"
            await base.action_submit()
            await _until(pilot, lambda: "no-such-ref" in _text(app, "diff-text") and "Symbols" not in _text(app, "diff-text"), "the ref error")

    asyncio.run(scenario())


# -- audit fixes, group 1: the screen must not lie, and Enter / e must act on the row the user chose -------------------------------


def test_a_watch_that_cannot_start_says_so_instead_of_staying_on_starting(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.tui import app as app_module

    def boom(*_args: object):
        raise OSError("disk full")

    monkeypatch.setattr(app_module, "open_watch", boom)

    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "WATCH STOPPED" in _text(app, "status") and "disk full" in _text(app, "status"), "the stopped message")
            assert "press r" in _text(app, "status")

    asyncio.run(scenario())


def test_a_failure_while_drawing_one_update_is_shown_and_the_watch_keeps_going(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.tui import app as app_module

    real, calls = app_module.compact_rows, {"n": 0}

    def flaky(view, previous, log=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("drawing bug")
        return real(view, previous, log)

    monkeypatch.setattr(app_module, "compact_rows", flaky)

    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            _add_function(shop_repo, "one")
            await _until(pilot, lambda: "drawing bug" in _text(app, "status"), "the failure line")
            _add_function(shop_repo, "two")
            await _until(pilot, lambda: any("billing.two" in row for row in _rows(app)), "a later update still arrives")

    asyncio.run(scenario())


def test_an_unexpected_error_in_a_lookup_is_shown_in_the_pane_not_left_on_looking_up(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.tui import data

    def boom(*_args: object, **_kwargs: object):
        raise RuntimeError("lookup bug")

    monkeypatch.setattr(data, "build_symbol_view", boom)
    monkeypatch.setattr(data, "compare_refs", boom)
    monkeypatch.setattr(data, "build_levels", boom)

    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3")
            box = app.query_one("#query", Input)
            box.value = "charge"
            await box.action_submit()
            await _until(pilot, lambda: "Failed: RuntimeError: lookup bug" in _text(app, "symbol-text"), "the symbol failure")
            await pilot.press("4")
            await _until(pilot, lambda: "Failed: RuntimeError: lookup bug" in _text(app, "diff-text"), "the diff failure")
            await pilot.press("2")
            await _until(pilot, lambda: "Failed: RuntimeError: lookup bug" in _text(app, "map-text"), "the map failure")

    asyncio.run(scenario())


def test_a_slow_older_lookup_never_overwrites_a_newer_one(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.tui import app as app_module
    from rich.text import Text

    def fake(_project, query, _width):
        if query == "slow":
            time.sleep(1.0)
        return Text(f"result for {query}")

    monkeypatch.setattr(app_module, "symbol_text", fake)

    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3")
            box = app.query_one("#query", Input)
            box.value = "slow"
            await box.action_submit()
            box.value = "fast"
            await box.action_submit()
            await _until(pilot, lambda: "result for fast" in _text(app, "symbol-text"), "the newer result")
            await pilot.pause(1.5)  # the older lookup finishes now
            assert "result for fast" in _text(app, "symbol-text")

    asyncio.run(scenario())


def test_the_highlighted_row_stays_on_the_same_symbol_when_a_save_rebuilds_the_list(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            shop_repo.write("shop/billing.py", BILLING + "\n\ndef extra1():\n    return 1\n\n\ndef extra2():\n    return 2\n")
            await _until(pilot, lambda: len([r for r in _rows(app) if "billing.extra" in r and "new function" in r]) == 2, "two new rows")
            box = app.query_one("#rows", OptionList)
            target = next(i for i in range(box.option_count) if box.get_option_at_index(i).id == "shop.billing.extra2")
            box.highlighted = target
            shop_repo.write("shop/api.py", "def api_extra():\n    return 3\n")
            await _until(pilot, lambda: any("api_extra" in row for row in _rows(app)), "the third new row")
            assert box.highlighted_option.id == "shop.billing.extra2"

    asyncio.run(scenario())


def test_rows_carry_a_coloured_label_and_the_legend_explains_it(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            assert "not whether it is right" not in _text(app, "limits")  # nothing to explain yet
            _add_function(shop_repo, "labelled")
            await _until(pilot, lambda: any("billing.labelled" in row for row in _rows(app)), "the row")
            box = app.query_one("#rows", OptionList)
            row = box.get_option("shop.billing.labelled").prompt
            assert "LOW" in str(row) and any("green" in str(span.style) for span in row.spans)
            assert "not whether it is right" in _text(app, "limits") and "can't check" in _text(app, "limits")

    asyncio.run(scenario())


def test_the_panel_under_the_list_follows_the_highlighted_row(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            assert _text(app, "panel") == ""  # nothing selected yet
            shop_repo.write("shop/billing.py", BILLING.replace("(amount):\n    return amount + 1", "(amount, currency=None):\n    return amount + 1"))
            await _until(pilot, lambda: any("billing.charge" in row for row in _rows(app)), "the changed row")
            await _until(pilot, lambda: "Used by" in _text(app, "panel"), "the panel")
            panel = _text(app, "panel")
            assert panel.startswith("shop.billing.charge") and "Before" in panel and "currency = None" in panel
            assert "Saved" in panel and "git grep -n 'charge('" in panel

    asyncio.run(scenario())


def test_the_panel_is_hidden_when_there_is_nothing_to_show_and_with_the_list_in_the_long_report(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            assert app.query_one("#panel").display is False  # no change, no panel (and no stray border)
            _add_function(shop_repo, "shown")
            await _until(pilot, lambda: "shop.billing.shown" in _text(app, "panel"), "the panel")
            assert app.query_one("#panel").display is True
            await pilot.press("d")
            assert app.query_one("#panel").display is False
            await pilot.press("d")
            assert app.query_one("#panel").display is True

    asyncio.run(scenario())


def test_the_changed_row_of_the_newest_save_is_bold_and_shows_its_time(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            _add_function(shop_repo, "fresh")
            await _until(pilot, lambda: any("billing.fresh" in row for row in _rows(app)), "the row")
            prompt = app.query_one("#rows", OptionList).get_option("shop.billing.fresh").prompt
            assert re.search(r"saved \d\d:\d\d:\d\d$", str(prompt)) and any("bold" in str(span.style) for span in prompt.spans)

    asyncio.run(scenario())


def test_enter_opens_the_symbol_that_was_chosen_even_if_the_list_changes_meanwhile(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            _add_function(shop_repo, "chosen")
            await _until(pilot, lambda: any("billing.chosen" in row for row in _rows(app)), "the row")
            box = app.query_one("#rows", OptionList)
            option = box.get_option_at_index(next(i for i in range(box.option_count) if box.get_option_at_index(i).id == "shop.billing.chosen"))
            app.on_option_list_option_selected(OptionList.OptionSelected(box, option, 0))
            await _until(pilot, lambda: "shop.billing.chosen" in _text(app, "symbol-text"), "the chosen symbol")

    asyncio.run(scenario())


def test_a_missing_editor_is_a_notice_not_a_crash_and_the_path_is_absolute(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlib

    from argus.tui import app as app_module

    seen: list[list[str]] = []

    def fake_run(command, **_kwargs):
        seen.append(command)
        raise FileNotFoundError(command[0])

    import types

    monkeypatch.setattr(app_module, "subprocess", types.SimpleNamespace(run=fake_run))  # only the editor call, not the app's git calls
    monkeypatch.setenv("EDITOR", "no-such-editor")
    monkeypatch.delenv("VISUAL", raising=False)

    async def scenario() -> None:
        app = _app(shop_repo)
        monkeypatch.setattr(app, "suspend", contextlib.nullcontext)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            _add_function(shop_repo)
            await _until(pilot, lambda: any("billing.extra" in row for row in _rows(app)), "the new function row")
            await pilot.press("e")
            await pilot.pause(0.2)
            await pilot.press("2")  # the app is still alive and still takes keys
            assert app.query_one("#views").current == "map"

    asyncio.run(scenario())
    assert seen and seen[0][-1] == str(shop_repo.root.resolve() / "shop/billing.py") and seen[0][-2].startswith("+")


def test_a_broken_editor_setting_is_a_notice_not_a_crash(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlib

    monkeypatch.setenv("EDITOR", 'vim "unbalanced')
    monkeypatch.delenv("VISUAL", raising=False)

    async def scenario() -> None:
        app = _app(shop_repo)
        monkeypatch.setattr(app, "suspend", contextlib.nullcontext)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            _add_function(shop_repo)
            await _until(pilot, lambda: any("billing.extra" in row for row in _rows(app)), "the new function row")
            await pilot.press("e")
            await pilot.pause(0.2)
            await pilot.press("2")
            assert app.query_one("#views").current == "map"

    asyncio.run(scenario())


# -- audit fixes, group 2: clean shutdown, one warnings filter for the process, no stale panels after `r` ------------------------


def test_the_process_ignores_syntax_warnings_once_instead_of_per_thread() -> None:
    code = "import warnings, argus.cli; print(any(f[0] == 'ignore' and f[2] is SyntaxWarning for f in warnings.filters))"

    proc = subprocess.run([sys.executable, "-W", "default", "-c", code], capture_output=True, text=True, check=False, timeout=60)

    assert proc.stdout.strip() == "True", proc.stderr  # `warnings.catch_warnings()` is not thread-safe: the entry point sets one filter


def test_r_clears_the_long_report_and_the_limits_line_at_once(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            await pilot.press("d")
            await _until(pilot, lambda: "Not determinable" in _text(app, "details"), "the long report")
            assert "can't check" in _text(app, "limits")
            app.action_fresh()  # synchronous: nothing from the old view may survive until the new one arrives
            assert _text(app, "details") == "" and _text(app, "limits") == ""
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the fresh first view")

    asyncio.run(scenario())


def test_a_start_error_from_a_replaced_worker_cannot_overwrite_the_new_status(shop_repo: GitRepo) -> None:
    import threading

    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            old = threading.Event()
            app._set_status_if_current(old, "late message from an old worker")
            assert "late message" not in _text(app, "status")

    asyncio.run(scenario())


def test_finished_workers_are_not_kept_forever(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            for _ in range(3):
                await pilot.press("r")
                await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "a fresh first view")
            await _until(pilot, lambda: sum(1 for e in app._finished if not e.is_set()) == 1, "old workers finished")
            app._start_watch()  # pruning happens when a new worker starts
            assert len(app._finished) <= 2

    asyncio.run(scenario())


def test_the_report_panes_take_the_keyboard_so_arrows_and_page_keys_can_scroll_them(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            for key, pane in (("2", "map"), ("3", "symbol-scroll"), ("4", "diff-scroll"), ("1", "rows")):
                await pilot.press(key)
                await pilot.pause(0.05)
                assert app.focused is not None and app.focused.id == pane, (key, app.focused)

    asyncio.run(scenario())
