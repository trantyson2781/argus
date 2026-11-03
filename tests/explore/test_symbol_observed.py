"""`argus explore symbol` shows what the latest recording saw next to what is read from the code (record mode, slice 4).

The fixture has a call through a dictionary, which static reading cannot follow, so the recording must add `dispatch` as a caller
that the code alone does not show. Observed facts carry their own tag, `[O]`, and a stale recording says so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus.cli import app
from argus.record.run import record_command

MOD = (
    "def helper(x):\n    return x + 1\n\n\ndef used(x):\n    return helper(x)\n\n\nHANDLERS = {'h': helper}\n\n\n"
    "def dispatch(key):\n    return HANDLERS[key](1)\n\n\ndef never(x):\n    return x\n"
)
TEST = "from pkg.mod import dispatch, used\n\n\ndef test_both():\n    assert used(1) == 2\n    assert dispatch('h') == 2\n"
PYTEST = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text(MOD, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mod.py").write_text(TEST, encoding="utf-8")
    return tmp_path


@pytest.fixture
def recorded(repo: Path) -> Path:
    record_command(repo, PYTEST)
    return repo


def symbol(repo: Path, query: str, *flags: str):
    return CliRunner().invoke(app, ["explore", "symbol", query, str(repo), *flags])


def seen_row(output: str) -> str:
    (row,) = [line for line in output.splitlines() if line.strip().startswith("Seen running")]
    return row


def test_the_summary_gets_a_seen_running_row_with_the_caller_static_reading_missed(recorded: Path) -> None:
    result = symbol(recorded, "pkg.mod.helper")

    assert result.exit_code == 0
    row = seen_row(result.output)
    assert "2 calls" in row and "also called by mod.dispatch" in row and "[O]" in row
    assert "mod.used" not in row  # static reading already lists it under Called by
    assert "run by 1 test" in row and "test_both" in row


def test_a_function_that_did_not_run_says_so(recorded: Path) -> None:
    assert "did not run in the last recording" in seen_row(symbol(recorded, "pkg.mod.never").output)


def test_nothing_is_added_without_a_recording(repo: Path) -> None:
    result = symbol(repo, "pkg.mod.helper")

    assert result.exit_code == 0 and "Seen running" not in result.output


def test_a_changed_file_marks_the_row_stale(recorded: Path) -> None:
    with (recorded / "pkg" / "mod.py").open("a", encoding="utf-8") as handle:
        handle.write("# edited after the recording\n")

    row = seen_row(symbol(recorded, "pkg.mod.helper").output)

    assert "STALE" in row and "pkg/mod.py" in row


def test_a_damaged_recording_is_reported_not_raised(recorded: Path) -> None:
    (path,) = (recorded / ".argus" / "recordings").glob("*.json")
    path.write_text("{broken", encoding="utf-8")

    result = symbol(recorded, "pkg.mod.helper")

    assert result.exit_code == 0 and "cannot be read" in result.output and "Seen running" not in result.output


def test_modules_and_classes_get_no_row(recorded: Path) -> None:
    assert "Seen running" not in symbol(recorded, "pkg.mod").output


def test_the_json_has_an_observed_object_labelled_observed(recorded: Path) -> None:
    data = json.loads(symbol(recorded, "pkg.mod.helper", "--json").output)["data"]

    observed = data["observed"]
    assert observed["label"] == "OBSERVED" and observed["calls"] == 2 and observed["stale_files"] == []
    assert {c["name"]: c["also"] for c in observed["callers"]} == {"pkg.mod.dispatch": True, "pkg.mod.used": False}
    assert observed["tests"] == ["tests.test_mod.test_both"] and data["observed_error"] is None
    assert data["definition"]["tier"] == "FACT"  # observed facts never become a tier of their own


def test_the_json_observed_is_null_without_a_recording_and_reports_a_damaged_one(repo: Path) -> None:
    data = json.loads(symbol(repo, "pkg.mod.helper", "--json").output)["data"]
    assert data["observed"] is None and data["observed_error"] is None

    record_command(repo, PYTEST)
    (path,) = (repo / ".argus" / "recordings").glob("*.json")
    path.write_text("{broken", encoding="utf-8")
    data = json.loads(symbol(repo, "pkg.mod.helper", "--json").output)["data"]
    assert data["observed"] is None and "cannot be read" in data["observed_error"]


def test_the_json_observed_is_null_for_a_module(recorded: Path) -> None:
    assert json.loads(symbol(recorded, "pkg.mod", "--json").output)["data"]["observed"] is None
