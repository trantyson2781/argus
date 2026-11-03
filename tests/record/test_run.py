"""Record mode, slice 2: running a command under the recorder (`argus.record.run`) and the `argus record` command.

These tests start real child processes: the point is that the watcher reaches code the recorded command runs, including a worker
process it starts, without Argus being importable there.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus.cli import app
from argus.record.run import RecordError, record_command
from argus.record.store import Recording, load_latest

MOD = "def helper(x):\n    return x + 1\n\n\ndef main():\n    return helper(1)\n"
TEST = "from pkg.mod import helper\n\n\ndef test_helper():\n    assert helper(1) == 2\n"
RUN_MAIN = "import pkg.mod as m; m.main()"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text(MOD, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mod.py").write_text(TEST, encoding="utf-8")
    return tmp_path


def py(code: str, *flags: str) -> list[str]:
    return [sys.executable, *flags, "-c", code]


def node_calls(rec: Recording, node_id: str) -> int:
    return sum(f.calls for f in rec.functions if f.node_id == node_id)


def test_a_command_that_runs_repo_code_is_recorded_and_matched_to_nodes(repo: Path) -> None:
    result = record_command(repo, py(RUN_MAIN))
    rec = result.recording

    assert rec.exit_code == 0 and rec.processes.saved == 1 and rec.processes.lost == 0
    assert node_calls(rec, "function:pkg.mod.helper") == 1 and node_calls(rec, "function:pkg.mod.main") == 1
    names = {f.key: f.node_id for f in rec.functions}
    (site,) = [s for s in rec.sites if names[s.callee] == "function:pkg.mod.helper"]
    assert names[site.caller] == "function:pkg.mod.main" and site.line == 6
    assert result.path == load_path(repo)


def load_path(repo: Path) -> Path:
    (path,) = sorted((repo / ".argus" / "recordings").glob("*.json"))
    return path


def test_the_exit_code_of_the_command_is_kept_and_the_recording_is_still_written(repo: Path) -> None:
    rec = record_command(repo, py(RUN_MAIN + "; import sys; sys.exit(3)")).recording

    assert rec.exit_code == 3 and node_calls(rec, "function:pkg.mod.helper") == 1
    assert load_latest(repo) == rec


def test_pytest_tests_are_linked_to_the_functions_they_reach(repo: Path) -> None:
    rec = record_command(repo, [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]).recording

    by_key = {f.key: f for f in rec.functions}
    (test,) = [t for t in rec.tests if by_key[t.test].node_id == "function:tests.test_mod.test_helper"]
    assert [by_key[k].node_id for k in test.reached] == ["function:pkg.mod.helper"]


def test_the_command_runs_in_the_repository_and_files_are_listed_with_hashes(repo: Path) -> None:
    rec = record_command(repo, py("import os; assert os.getcwd() == " + repr(str(repo.resolve())))).recording

    assert rec.exit_code == 0
    files = dict(rec.files)
    assert set(files) == {"pkg/__init__.py", "pkg/mod.py", "tests/test_mod.py"} and all(files.values())


def test_a_file_changed_while_the_command_ran_has_no_hash(repo: Path) -> None:
    rec = record_command(repo, py(RUN_MAIN + "; open('pkg/mod.py', 'a').write('# edited\\n')")).recording

    files = dict(rec.files)
    assert files["pkg/mod.py"] is None and files["tests/test_mod.py"]
    changed = [f for f in rec.functions if f.file == "pkg/mod.py"]
    assert changed and all(f.node_id is None and f.unmatched == "file_changed" for f in changed)  # the scan no longer describes what ran


def test_isolated_python_cannot_be_recorded_and_the_summary_says_so(repo: Path) -> None:
    rec = record_command(repo, py("print(1)", "-I")).recording  # -I ignores PYTHONPATH and with it the recorder

    assert rec.exit_code == 0 and rec.processes.saved == 0 and rec.functions == ()


def test_a_worker_process_the_command_starts_is_recorded_too(repo: Path) -> None:
    child = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {RUN_MAIN!r}], check=True)"

    rec = record_command(repo, py(child)).recording

    assert rec.processes.saved == 2 and node_calls(rec, "function:pkg.mod.helper") == 1


def test_a_forked_child_that_exits_normally_counts_only_its_own_calls(repo: Path) -> None:
    code = f"import os, sys, pkg.mod as m; m.main(); pid = os.fork()\nif pid == 0:\n    m.main(); sys.exit(0)\nos.waitpid(pid, 0)"

    rec = record_command(repo, py(code)).recording

    assert rec.processes.saved == 2 and node_calls(rec, "function:pkg.mod.helper") == 2  # 3 would mean the child repeated its parent's count


def test_a_forked_child_that_leaves_through_os_exit_is_reported_as_lost(repo: Path) -> None:
    code = "import os, pkg.mod as m; pid = os.fork()\nif pid == 0:\n    m.main(); os._exit(0)\nos.waitpid(pid, 0)"

    rec = record_command(repo, py(code)).recording

    assert (rec.processes.saved, rec.processes.lost) == (1, 1) and node_calls(rec, "function:pkg.mod.helper") == 0


def test_the_users_own_sitecustomize_still_runs(repo: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    theirs = tmp_path_factory.mktemp("theirs")
    marker = theirs / "ran"
    (theirs / "sitecustomize.py").write_text(f"open({str(marker)!r}, 'w').write('yes')\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(theirs))

    rec = record_command(repo, py(RUN_MAIN)).recording

    assert marker.read_text(encoding="utf-8") == "yes" and node_calls(rec, "function:pkg.mod.helper") == 1


def test_a_second_interrupt_ends_the_command_and_the_recording_is_still_written(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    ready = tmp_path_factory.mktemp("ready") / "ready"

    def interrupt_twice() -> None:
        deadline = time.monotonic() + 60
        while not ready.exists() and time.monotonic() < deadline:  # wait for the command itself, never a fixed time
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.3)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt_twice, daemon=True).start()
    started = time.monotonic()
    result = record_command(repo, py(f"import pathlib, time; pathlib.Path({str(ready)!r}).write_text('x'); time.sleep(60)"))

    assert time.monotonic() - started < 30 and result.recording.exit_code == 143  # 128 + SIGTERM
    assert load_latest(repo) is not None


def test_nothing_to_run_and_a_missing_program_are_errors_and_write_nothing(repo: Path) -> None:
    with pytest.raises(ValueError):
        record_command(repo, [])
    with pytest.raises(RecordError, match="cannot run"):
        record_command(repo, ["argus-no-such-program-xyz"])

    assert not (repo / ".argus").exists()


def test_the_project_is_left_as_it_was_apart_from_the_argus_folder(repo: Path) -> None:
    before = {p.relative_to(repo).as_posix(): p.read_bytes() for p in repo.rglob("*.py")}

    record_command(repo, py(RUN_MAIN))

    after = {p.relative_to(repo).as_posix(): p.read_bytes() for p in repo.rglob("*.py")}
    assert before == after and not (repo / ".gitignore").exists()


# -- the command ---------------------------------------------------------------------------------------------------------------------


def test_the_record_command_warns_before_running_runs_and_summarises(repo: Path) -> None:
    result = CliRunner().invoke(app, ["record", "-C", str(repo), "--", *py(RUN_MAIN + "; print('hello from the child')")])

    assert result.exit_code == 0
    assert result.output.index("runs this command") < result.output.index("saved")  # the child writes to the real terminal, not to the captured output
    assert "functions ran" in result.output


def test_the_record_command_exits_with_the_exit_code_of_the_command(repo: Path) -> None:
    result = CliRunner().invoke(app, ["record", "-C", str(repo), "--", *py("import sys; sys.exit(7)")])

    assert result.exit_code == 7


def test_the_record_command_reports_an_unusable_command_with_exit_code_2(repo: Path) -> None:
    result = CliRunner().invoke(app, ["record", "-C", str(repo), "--", "argus-no-such-program-xyz"])

    assert result.exit_code == 2 and "cannot run" in result.output


# -- failures after the command has run must not lose the run -----------------------------------------------------------------------------


def test_a_recording_that_cannot_be_saved_is_reported_with_the_exit_code_of_the_command(repo: Path) -> None:
    (repo / ".argus").write_text("in the way", encoding="utf-8")  # a file where the folder should go

    with pytest.raises(RecordError, match=r"cannot save.*exited with 3"):
        record_command(repo, py("import sys; sys.exit(3)"))


def test_a_damaged_process_file_counts_as_a_lost_process_and_the_rest_is_kept(tmp_path: Path) -> None:
    from argus.record.run import _merge

    good = {"functions": [], "callers": [], "sites": [], "tests": [], "errors": {"count": 0, "first": None}}
    (tmp_path / "1.start").write_text("", encoding="utf-8")
    (tmp_path / "1.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "2.start").write_text("", encoding="utf-8")
    (tmp_path / "2.json").write_text("{half a file", encoding="utf-8")
    (tmp_path / "3.start").write_text("", encoding="utf-8")
    (tmp_path / "3.json").write_text(json.dumps({"unexpected": True}), encoding="utf-8")

    _, processes = _merge(tmp_path)

    assert (processes.saved, processes.lost) == (1, 2)


# -- an empty recording is not kept ------------------------------------------------------------------------------------


def test_an_empty_recording_is_not_written_when_the_caller_does_not_want_one(repo: Path) -> None:
    result = record_command(repo, py("print(1)", "-I"), save_empty=False)  # -I ignores PYTHONPATH and with it the recorder

    assert result.path is None and result.recording.processes.saved == 0
    assert not list((repo / ".argus" / "recordings").glob("*.json"))


def test_an_empty_recording_does_not_hide_a_good_one(repo: Path) -> None:
    good = record_command(repo, py(RUN_MAIN)).recording
    record_command(repo, py("print(1)", "-I"), save_empty=False)

    assert load_latest(repo).started_at == good.started_at


def test_the_command_line_says_nothing_was_recorded_and_keeps_the_commands_exit_code(repo: Path) -> None:
    result = CliRunner().invoke(app, ["record", "-C", str(repo), "--", *py("import sys; sys.exit(3)", "-I")])

    assert result.exit_code == 3 and "Nothing was recorded" in result.output and "no file was written" in result.output
    assert not list((repo / ".argus" / "recordings").glob("*.json"))
