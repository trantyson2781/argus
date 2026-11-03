"""A malformed pyproject.toml must degrade to an UNKNOWN claim in every explore command, never a traceback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus.cli import app

RUNNER = CliRunner()


@pytest.fixture
def odd_repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text('project = "not a table"\n')
    (tmp_path / "p.py").write_text("def work():\n    return 1\n")
    return tmp_path


@pytest.mark.parametrize("command", [["map"], ["entrypoints"], ["symbol", "work"]])
def test_a_project_that_is_not_a_table_does_not_crash_the_command(odd_repo: Path, command: list[str]) -> None:
    result = RUNNER.invoke(app, ["explore", *command, str(odd_repo)])

    assert result.exit_code == 0 and "Traceback" not in result.output and not isinstance(result.exception, AttributeError)


def test_the_entrypoints_json_says_why_there_are_no_scripts(odd_repo: Path) -> None:
    doc = json.loads(RUNNER.invoke(app, ["explore", "entrypoints", str(odd_repo), "--json"]).stdout)

    assert doc["ok"] is True and any("not a table" in u["text"] for u in doc["data"]["unknowns"])
