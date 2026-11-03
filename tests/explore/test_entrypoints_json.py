"""`argus explore entrypoints --json`: scripts, commands, main guards, unknowns and (with --command) the call trace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.explore import cli
from argus.explore.entrypoints import find_entrypoints
from argus.explore.reach import call_tree
from argus.extract import scan
from tests.explore.json_helpers import claims_in, doc_of, json_keys, run_json, view_keys


def test_the_envelope_names_the_command_and_echoes_the_options(mini_repo: Path) -> None:
    doc = doc_of("entrypoints", str(mini_repo))

    assert doc["schema_version"] == 1 and doc["command"] == "entrypoints" and doc["ok"] is True
    assert doc["options"] == {"command": None, "depth": 2} and doc["resolver"] in ("calls-v1", "calls-v2")


def test_scripts_commands_and_guards_are_structured_rows(mini_repo: Path) -> None:
    data = doc_of("entrypoints", str(mini_repo))["data"]

    (script,) = data["scripts"]
    assert (script["name"], script["module"], script["attr"]) == ("mini", "mini.cli", "app") and script["claim"]["evidence"][0]["file"] == "pyproject.toml"
    commands = {c["display"]: c for c in data["commands"]}
    assert set(commands) == {"mini hello-world", "mini sub renamed"}
    hello = commands["mini hello-world"]
    assert (hello["path"], hello["function"], hello["module"], hello["file"]) == (["hello-world"], "mini.cli.hello_world", "mini.cli", "src/mini/cli.py")
    assert hello["claim"]["tier"] == "FACT" and commands["mini sub renamed"]["path"] == ["sub", "renamed"]
    (guard,) = data["main_guards"]
    assert guard["evidence"][0]["file"] == "src/mini/__main__.py" and guard["evidence"][0]["line"] == 3


def test_an_unresolvable_sub_app_stays_an_unknown_claim(mini_repo: Path) -> None:
    unknowns = doc_of("entrypoints", str(mini_repo))["data"]["unknowns"]

    assert any("ext_app" in u["text"] and u["tier"] == "UNKNOWN" for u in unknowns)


def test_command_adds_a_trace_of_resolved_calls_with_depth_and_call_sites(mini_repo: Path) -> None:
    doc = doc_of("entrypoints", str(mini_repo), "--command", "hello-world")

    assert doc["options"] == {"command": "hello-world", "depth": 2}
    (trace,) = doc["data"]["traces"]
    assert trace["command"]["display"] == "mini hello-world" and trace["command"]["claim"]["tier"] == "FACT"
    steps = trace["steps"]
    assert {s["callee"] for s in steps} >= {"mini.core.work", "mini.store.db.save"}
    first = steps[0]
    assert first["caller"] == "mini.cli.hello_world" and first["depth"] == 1 and first["module"] == "mini.core"
    assert {"group", "kind", "claim"} <= set(first) and first["claim"]["tier"] == "INFERENCE"


def test_the_trace_and_depth_option_match_the_text_views(mini_repo: Path) -> None:
    model = scan(mini_repo)
    eps = find_entrypoints(mini_repo, model)
    command = next(c for c in eps.commands if c.display == "mini hello-world")
    steps = call_tree(model, mini_repo, command.function_id, 1)

    doc = doc_of("entrypoints", str(mini_repo), "--command", "hello-world", "--depth", "1")

    assert [s["callee"] for s in doc["data"]["traces"][0]["steps"]] == [s.callee for s in steps]
    assert all(s["depth"] <= 1 for s in doc["data"]["traces"][0]["steps"])


def test_every_claim_of_the_views_is_in_the_document_unchanged(mini_repo: Path) -> None:
    model = scan(mini_repo)
    eps = find_entrypoints(mini_repo, model)
    command = next(c for c in eps.commands if c.display == "mini hello-world")
    expected = [*eps.all_claims(), command.claim, *(s.claim for s in call_tree(model, mini_repo, command.function_id, 2))]

    doc = doc_of("entrypoints", str(mini_repo), "--command", "hello-world")

    got = claims_in(doc)
    assert json_keys(got) == view_keys(expected)


def test_evidence_is_current_relative_and_fact_or_inference_has_evidence(mini_repo: Path) -> None:
    for claim in claims_in(doc_of("entrypoints", str(mini_repo), "--command", "sub renamed")):
        assert claim["tier"] == "UNKNOWN" or claim["evidence"]
        assert all(e["side"] == "current" and not e["file"].startswith("/") for e in claim["evidence"])


def test_the_output_is_deterministic_and_has_no_absolute_path(mini_repo: Path) -> None:
    first = run_json("entrypoints", str(mini_repo), "--command", "hello-world").stdout
    second = run_json("entrypoints", str(mini_repo), "--command", "hello-world").stdout

    assert first == second and str(mini_repo) not in first


def test_an_unknown_command_is_a_json_error_that_lists_the_known_ones(mini_repo: Path) -> None:
    result = run_json("entrypoints", str(mini_repo), "--command", "nope")

    assert result.exit_code == 2
    error = json.loads(result.stdout)["error"]
    assert error["kind"] == "command_not_found" and "hello-world" in error["message"]


def test_a_path_that_is_not_a_directory_is_a_json_error(tmp_path: Path) -> None:
    result = run_json("entrypoints", str(tmp_path / "missing"))

    assert result.exit_code == 2 and json.loads(result.stdout)["error"]["kind"] == "not_a_directory"


def test_an_unexpected_error_is_a_json_error_with_only_the_exception_type(mini_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("secret /home/x")

    monkeypatch.setattr(cli, "find_entrypoints", boom)

    result = run_json("entrypoints", str(mini_repo))

    assert result.exit_code == 1 and json.loads(result.stdout)["error"] == {"kind": "internal_error", "message": "RuntimeError"}
    assert "secret" not in result.stdout


def test_text_mode_still_reports_a_missing_command_on_stderr(mini_repo: Path) -> None:
    from tests.explore.json_helpers import runner
    from argus.cli import app

    result = runner.invoke(app, ["explore", "entrypoints", str(mini_repo), "--command", "nope"])

    assert result.exit_code == 2 and result.stdout == "" and "nope" in result.stderr
