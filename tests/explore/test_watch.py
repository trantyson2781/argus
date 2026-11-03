"""`argus explore watch`: a baseline held for the whole session, a cheap change check, and a repaint of the diff report on every change."""

from __future__ import annotations

import io
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from argus.cli import app
from argus.explore.sides import DiffError, materialize_baseline, python_file_stamps
from argus.explore.watch import WatchSettings, open_watch, refresh, run_watch
from tests.explore.diff_helpers import GIT_ENV, GitRepo

BILLING = "def charge(amount):\n    return amount + 1\n\n\ndef fee(amount):\n    return amount\n"
RUNNER = CliRunner()


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=240, highlight=False, force_terminal=False), buffer


def _renders(text: str) -> int:
    return len(re.findall(r"^watching \|", text, flags=re.MULTILINE))


def _run(repo: GitRepo, script: list[Callable[[], None]], base: str | None = None, include_tests: bool = False) -> str:
    """Run the loop with a fake clock: the i-th sleep first runs script[i] (an edit), and the loop stops after the script is used up."""
    console, buffer = _console()
    calls = {"n": 0}

    def sleep(_seconds: float) -> None:
        if calls["n"] < len(script):
            script[calls["n"]]()
        calls["n"] += 1

    with open_watch(repo.root, base, include_tests) as settings:
        run_watch(settings, console, sleep=sleep, should_stop=lambda: calls["n"] > len(script), details=True)
    return buffer.getvalue()


# -- the baseline --------------------------------------------------------------------------------------------------------------------


def test_the_launch_baseline_is_the_working_tree_at_launch_including_uncommitted_work(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 9"))

    with materialize_baseline(shop_repo.root, None) as (scan_path, base):
        assert scan_path == shop_repo.root.resolve() and re.fullmatch(r"launch \d\d:\d\d:\d\d", base.label)
        assert (base.root / "shop/billing.py").read_text() == BILLING.replace("amount + 1", "amount + 9")


def test_a_ref_baseline_is_that_commit_and_ignores_uncommitted_work(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 9"))

    with materialize_baseline(shop_repo.root, "HEAD") as (_scan_path, base):
        assert re.fullmatch(r"HEAD \([0-9a-f]{7}\)", base.label)
        assert (base.root / "shop/billing.py").read_text() == BILLING


def test_a_bad_ref_or_a_directory_outside_git_is_a_diff_error(shop_repo: GitRepo, tmp_path: Path) -> None:
    with pytest.raises(DiffError, match="no-such-ref"), materialize_baseline(shop_repo.root, "no-such-ref"):
        pass
    outside = tmp_path / "plain"
    outside.mkdir()
    with pytest.raises(DiffError, match="not inside a git repository"), materialize_baseline(outside, None):
        pass


def test_the_baseline_temporary_files_are_removed_on_exit_and_on_error(shop_repo: GitRepo, scratch_tmp: Path) -> None:
    with materialize_baseline(shop_repo.root, None):
        assert any(scratch_tmp.glob("argus-*"))
    with pytest.raises(RuntimeError), materialize_baseline(shop_repo.root, None):
        raise RuntimeError("boom")

    assert not list(scratch_tmp.glob("argus-*"))


# -- the cheap change check ----------------------------------------------------------------------------------------------------------


def test_stamps_change_when_a_python_file_is_edited_added_or_deleted(shop_repo: GitRepo) -> None:
    before = python_file_stamps(shop_repo.root)

    shop_repo.write("shop/billing.py", BILLING + "\n\ndef extra():\n    return 1\n")
    edited = python_file_stamps(shop_repo.root)
    shop_repo.write("shop/new.py", "X = 1\n")
    added = python_file_stamps(shop_repo.root)
    shop_repo.delete("shop/api.py")
    deleted = python_file_stamps(shop_repo.root)

    assert len({tuple(sorted(s.items(), key=str)) for s in (before, edited, added, deleted)}) == 4
    assert "shop/new.py" in added and deleted["shop/api.py"] is None


def test_stamps_ignore_other_file_types_and_git_ignored_files(shop_repo: GitRepo) -> None:
    shop_repo.write(".gitignore", "ignored.py\n")
    shop_repo.commit("ignore")
    before = python_file_stamps(shop_repo.root)

    shop_repo.write("notes.md", "hello\n")
    shop_repo.write("ignored.py", "X = 1\n")

    assert python_file_stamps(shop_repo.root) == before


# -- one refresh ---------------------------------------------------------------------------------------------------------------------


def test_a_refresh_reports_only_what_changed_since_the_launch_snapshot(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 9"))  # work done before launch
    with open_watch(shop_repo.root, None, False) as settings:
        shop_repo.write("shop/cart.py", "from shop.billing import charge\n\n\ndef checkout(amount):\n    return charge(amount) + 1\n")

        view, seconds = refresh(settings)

    names = {c.qualname for c in view.symbols}
    assert "shop.cart.checkout" in names and "shop.billing.charge" not in names and seconds >= 0


def test_a_ref_baseline_reports_everything_since_that_commit(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 9"))
    with open_watch(shop_repo.root, "HEAD", False) as settings:
        view, _seconds = refresh(settings)

    assert {c.qualname for c in view.symbols} == {"shop.billing.charge"}


def test_a_refresh_does_not_touch_the_repository(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 9"))
    status = shop_repo.git("status", "--porcelain")
    content = (shop_repo.root / "shop/billing.py").read_bytes()

    with open_watch(shop_repo.root, None, False) as settings:
        refresh(settings)

    assert shop_repo.git("status", "--porcelain") == status and (shop_repo.root / "shop/billing.py").read_bytes() == content


# -- the loop ------------------------------------------------------------------------------------------------------------------------


def test_the_first_report_is_painted_at_once_and_nothing_repaints_while_nothing_changes(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [lambda: None, lambda: None, lambda: None])

    assert _renders(out) == 1 and "no structural change" in out
    assert re.search(
        r"^watching \| baseline: launch \d\d:\d\d:\d\d \(earlier edits not shown\) \| refreshed \d\d:\d\d:\d\d in \d+\.\d s \| Ctrl-C quits",
        out, flags=re.MULTILINE,
    )


def test_an_edit_is_painted_by_the_next_refresh(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [lambda: shop_repo.write("shop/billing.py", BILLING + "\n\ndef extra():\n    return 1\n"), lambda: None, lambda: None])

    assert _renders(out) == 2
    assert "shop.billing.extra" in out.split("watching |")[-1] and "shop.billing.extra" not in out.split("watching |")[1]


def test_edits_that_arrive_while_settling_are_coalesced_into_one_refresh(shop_repo: GitRepo) -> None:
    first = lambda: shop_repo.write("shop/billing.py", BILLING + "\n\ndef first():\n    return 1\n")  # noqa: E731
    second = lambda: shop_repo.write("shop/api.py", "def second():\n    return 2\n")  # noqa: E731 - lands during the settle wait

    out = _run(shop_repo, [first, second, lambda: None, lambda: None])

    last = out.split("watching |")[-1]
    assert _renders(out) == 2 and "shop.billing.first" in last and "shop.api.second" in last


def test_a_failed_refresh_is_shown_and_the_watcher_keeps_going(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.explore import watch

    real = watch.refresh
    state = {"n": 0}

    def flaky(settings: WatchSettings):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("scanner exploded")
        return real(settings)

    monkeypatch.setattr(watch, "refresh", flaky)
    edit = lambda n: (lambda: shop_repo.write("shop/api.py", f"def v{n}():\n    return {n}\n"))  # noqa: E731

    out = _run(shop_repo, [edit(1), lambda: None, lambda: None, edit(2), lambda: None, lambda: None])

    assert "refresh failed: RuntimeError: scanner exploded" in out and "shop.api.v2" in out.split("refresh failed")[-1]


def test_a_failure_of_the_very_first_refresh_is_shown_not_raised(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.explore import watch

    def boom(_settings: WatchSettings):
        raise RuntimeError("first one")

    monkeypatch.setattr(watch, "refresh", boom)

    assert "refresh failed: RuntimeError: first one" in _run(shop_repo, [lambda: None])


def test_a_file_that_does_not_parse_mid_edit_is_reported_and_not_fatal(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [lambda: shop_repo.write("shop/billing.py", "def charge(amount:\n"), lambda: None, lambda: None])

    assert _renders(out) == 2 and "does not parse" in out.split("watching |")[-1]


def test_every_temporary_directory_is_gone_after_the_loop(shop_repo: GitRepo, scratch_tmp: Path) -> None:
    _run(shop_repo, [lambda: shop_repo.write("shop/api.py", "def v():\n    return 1\n"), lambda: None, lambda: None])

    assert not list(scratch_tmp.glob("argus-*"))


def test_include_tests_is_passed_to_the_comparison(shop_repo: GitRepo) -> None:
    shop_repo.write("tests/__init__.py", "")
    shop_repo.write("tests/test_billing.py", "from shop.billing import charge\n\n\ndef test_charge():\n    assert charge(1)\n")
    shop_repo.commit("add test")
    edit = [lambda: shop_repo.write("shop/billing.py", BILLING.replace("(amount):\n    return amount + 1", "(amount, currency=None):\n    return amount + 1"))]

    hidden = _run(shop_repo, [*edit, lambda: None, lambda: None])
    shop_repo.git("checkout", "--", "shop/billing.py")  # the second launch must start from the committed file again
    shown = _run(shop_repo, [*edit, lambda: None, lambda: None], include_tests=True)

    assert "test caller group hidden" in hidden.split("watching |")[-1] and "tests.test_billing.test_charge" in shown


# -- the command ---------------------------------------------------------------------------------------------------------------------


def test_the_command_refuses_a_directory_outside_git_and_a_bad_ref_on_stderr(shop_repo: GitRepo, tmp_path: Path) -> None:
    outside = tmp_path / "plain"
    outside.mkdir()

    bad_dir = RUNNER.invoke(app, ["explore", "watch", str(outside)])
    bad_ref = RUNNER.invoke(app, ["explore", "watch", str(shop_repo.root), "--base", "no-such-ref"])

    assert bad_dir.exit_code == 2 and bad_dir.stdout == "" and "not inside a git repository" in bad_dir.stderr
    assert bad_ref.exit_code == 2 and bad_ref.stdout == "" and "no-such-ref" in bad_ref.stderr


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_end_to_end_a_real_process_repaints_after_an_edit_and_leaves_no_trace(shop_repo: GitRepo, tmp_path: Path, stop_signal: signal.Signals) -> None:
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", "from argus.cli import app; app()", "explore", "watch", str(shop_repo.root), "--details"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, **GIT_ENV, "TMPDIR": str(scratch)},
    )
    status_before = shop_repo.git("status", "--porcelain")
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    seen: list[str] = []

    def wait_for(marker: str, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=max(0.1, deadline - time.monotonic()))
            except queue.Empty:
                break
            if line is None:
                break
            seen.append(line)
            if marker in line:
                return
        pytest.fail(f"never saw {marker!r}:\n" + "".join(seen))

    try:
        wait_for("The caller list may be incomplete")  # the last line of the first report
        shop_repo.write("shop/billing.py", BILLING + "\n\ndef extra():\n    return 1\n")
        wait_for("shop.billing.extra")  # the repaint carries the edit
        wait_for("The caller list may be incomplete")  # ... and has finished
        proc.send_signal(stop_signal)
        proc.wait(timeout=30)
        err = proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert proc.returncode == 0 and "Traceback" not in err
    assert not list(scratch.glob("argus-*"))
    assert status_before == "" and shop_repo.git("status", "--porcelain").splitlines() == [" M shop/billing.py"]  # only the edit this test made
