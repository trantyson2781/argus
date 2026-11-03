"""`argus explore symbol --json`: the symbol view as one complete JSON document with structured callers, callees and unresolved calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from argus.cli import app
from argus.explore import cli
from argus.explore.symbols import build_symbol_view, resolve_symbol
from argus.extract import scan
from tests.explore.test_symbols import LIB_FILES

runner = CliRunner()


@pytest.fixture
def lib_repo(tmp_path: Path) -> Path:
    for rel, content in LIB_FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _default_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_RESOLVE_V2", raising=False)


def _run(repo: Path, query: str, *args: str, env: dict[str, str] | None = None) -> Result:
    return runner.invoke(app, ["explore", "symbol", query, str(repo), "--json", *args], env=env)


def _doc(repo: Path, query: str, *args: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = _run(repo, query, *args, env=env)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _claims(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "tier" in node and "evidence" in node:
            found.append(node)
        for value in node.values():
            found.extend(_claims(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_claims(value))
    return found


# -- envelope and the symbol itself -----------------------------------------------------------------------------------------------


def test_the_envelope_names_the_command_the_query_and_the_resolver(lib_repo: Path) -> None:
    doc = _doc(lib_repo, "twin")

    assert doc["schema_version"] == 1 and doc["command"] == "symbol" and doc["ok"] is True and doc["resolver"] == "calls-v2"
    assert doc["options"] == {"query": "twin", "include_tests": False}
    assert doc["data"]["node"] == {"qualname": "tinylib.shapes.twin", "name": "twin", "kind": "function", "file": "src/tinylib/shapes.py", "line": 28}


def test_definition_signature_and_an_unknown_purpose(lib_repo: Path) -> None:
    data = _doc(lib_repo, "twin")["data"]

    assert data["signature"] == "(r)" and data["decorators"] == []
    assert data["definition"]["tier"] == "FACT" and data["definition"]["evidence"][0] == {
        "file": "src/tinylib/shapes.py", "line": 28, "source_hash": data["definition"]["evidence"][0]["source_hash"], "side": "current",
    }
    assert data["purpose"]["tier"] == "UNKNOWN" and data["purpose"]["evidence"] == []


def test_a_class_lists_its_members_as_structured_rows(lib_repo: Path) -> None:
    members = _doc(lib_repo, "Circle")["data"]["members"]

    assert [(m["kind"], m["qualname"]) for m in members] == [
        ("method", "tinylib.shapes.Circle.__init__"), ("method", "tinylib.shapes.Circle.area"),
        ("method", "tinylib.shapes.Circle.scale"), ("method", "tinylib.shapes.Circle.describe"),
    ]
    assert all(m["claim"]["tier"] == "FACT" and m["claim"]["evidence"] for m in members)


# -- callers, callees, unresolved calls ---------------------------------------------------------------------------------------------


def test_callers_are_rows_with_a_name_a_kind_and_the_call_site(lib_repo: Path) -> None:
    data = _doc(lib_repo, "shapes.make")["data"]

    assert [(c["name"], c["kind"], c["module_level"], c["confidence"]) for c in data["callers"]] == [("tinylib.shapes.twin", "function", False, "INFERRED")] * 2
    assert {c["claim"]["evidence"][0]["line"] for c in data["callers"]} == {29}
    assert all(c["claim"]["tier"] == "INFERENCE" and c["override_possible"] is False for c in data["callers"])
    assert data["hidden_test_callers"] == 1


def test_include_tests_adds_the_test_callers_and_echoes_the_option(lib_repo: Path) -> None:
    doc = _doc(lib_repo, "shapes.make", "--include-tests")

    assert doc["options"]["include_tests"] is True and len(doc["data"]["callers"]) == 3 and doc["data"]["hidden_test_callers"] == 0
    assert any(c["name"].endswith("test_make") for c in doc["data"]["callers"])


def test_a_self_call_is_a_callee_row_with_override_possible(lib_repo: Path) -> None:
    (callee,) = _doc(lib_repo, "Circle.area")["data"]["callees"]

    assert callee["name"] == "tinylib.shapes.Circle.scale" and callee["kind"] == "method" and callee["override_possible"] is True
    assert callee["module_target"] is False and callee["claim"]["evidence"][0]["line"] == 14


def test_with_the_v1_resolver_a_module_alias_call_is_marked_as_a_module_target(lib_repo: Path) -> None:
    doc = _doc(lib_repo, "Circle.scale", env={"ARGUS_RESOLVE_V2": "0"})

    (callee,) = doc["data"]["callees"]
    assert doc["resolver"] == "calls-v1" and callee["name"] == "tinylib.units" and callee["module_target"] is True


def test_unresolved_calls_are_rows_with_the_call_as_written_a_reason_and_a_line(lib_repo: Path) -> None:
    data = _doc(lib_repo, "Circle.describe")["data"]

    (row,) = data["unresolved_calls"]
    assert row["callee"] == "other.render" and "variable or parameter" in row["reason"] and row["line"] == 21
    assert row["claim"]["tier"] == "UNKNOWN" and row["claim"]["evidence"][0]["line"] == 21  # an UNKNOWN claim can still point at its call site
    assert data["builtin_call_count"] == 2 and data["external_calls"] == []


def test_a_module_symbol_has_members_and_no_call_data(lib_repo: Path) -> None:
    data = _doc(lib_repo, "tinylib.units")["data"]

    assert [m["qualname"] for m in data["members"]] == ["tinylib.units.convert", "tinylib.units.make"]
    assert data["calls_scope"] == "none"
    assert data["callers"] is None and data["callees"] is None and data["unresolved_calls"] is None  # not computed for a module: null, not []


# -- guarantees ---------------------------------------------------------------------------------------------------------------------


def test_every_claim_of_the_view_is_in_the_document_unchanged(lib_repo: Path) -> None:
    model = scan(lib_repo)
    view = build_symbol_view(model, lib_repo, resolve_symbol(model, "Circle.describe"))
    view_claims = [view.definition, view.purpose, *view.members, *view.callers, *view.callees, *view.unknown_calls]

    expected = sorted((c.tier.value, c.text, tuple((e.file, e.line, e.source_hash) for e in c.evidence)) for c in view_claims)
    got = sorted((c["tier"], c["text"], tuple((e["file"], e["line"], e["source_hash"]) for e in c["evidence"])) for c in _claims(_doc(lib_repo, "Circle.describe")))

    assert got == expected


def test_each_structured_name_agrees_with_the_sentence_of_its_claim(lib_repo: Path) -> None:
    for query in ("shapes.make", "Circle.area", "Circle.scale"):
        data = _doc(lib_repo, query, "--include-tests")["data"]
        for row in (*data["callers"], *data["callees"]):
            assert row["name"] in row["claim"]["text"]


def test_fact_and_inference_claims_always_have_evidence_and_sides_are_current(lib_repo: Path) -> None:
    for query in ("shapes.make", "Circle", "Circle.describe"):
        for claim in _claims(_doc(lib_repo, query)):
            assert claim["tier"] == "UNKNOWN" or claim["evidence"]
            assert all(e["side"] == "current" and not e["file"].startswith("/") for e in claim["evidence"])


def test_the_output_is_deterministic_and_has_no_absolute_path(lib_repo: Path) -> None:
    first, second = _run(lib_repo, "shapes.make").stdout, _run(lib_repo, "shapes.make").stdout

    assert first == second and str(lib_repo) not in first


def test_the_output_does_not_depend_on_the_hash_seed(lib_repo: Path) -> None:
    outputs = set()
    for seed in ("1", "2", "3"):
        proc = subprocess.run(
            [sys.executable, "-c", "from argus.cli import app; app()", "explore", "symbol", "shapes.make", str(lib_repo), "--json", "--include-tests"],
            capture_output=True, text=True, check=True, timeout=120, env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert json.loads(proc.stdout)["ok"] is True
        outputs.add(proc.stdout)

    assert len(outputs) == 1


def test_nothing_is_cut_by_the_default_limit(tmp_path: Path) -> None:
    calls = "".join(f"def c{i:02d}():\n    return target()\n\n\n" for i in range(30))
    (tmp_path / "m.py").write_text("def target():\n    return 1\n\n\n" + calls, encoding="utf-8")

    assert len(_doc(tmp_path, "target")["data"]["callers"]) == 30  # the text report stops at 15


# -- errors -------------------------------------------------------------------------------------------------------------------------


def test_an_unknown_symbol_is_a_json_error_with_exit_2(lib_repo: Path) -> None:
    result = _run(lib_repo, "twinn")

    assert result.exit_code == 2
    doc = json.loads(result.stdout)
    assert doc["ok"] is False and doc["command"] == "symbol" and doc["error"]["kind"] == "symbol_lookup" and "Did you mean: twin" in doc["error"]["message"]


def test_an_ambiguous_symbol_is_a_json_error_that_lists_the_candidates(lib_repo: Path) -> None:
    doc = json.loads(_run(lib_repo, "make").stdout)

    assert doc["error"]["kind"] == "symbol_lookup" and "tinylib.shapes.make" in doc["error"]["message"] and "tinylib.units.make" in doc["error"]["message"]


def test_a_path_that_is_not_a_directory_is_a_json_error(tmp_path: Path) -> None:
    result = _run(tmp_path / "missing", "x")

    assert result.exit_code == 2 and json.loads(result.stdout)["error"]["kind"] == "not_a_directory"


def test_json_with_limit_is_refused_as_a_json_error(lib_repo: Path) -> None:
    result = _run(lib_repo, "twin", "--limit", "3")

    assert result.exit_code == 2 and json.loads(result.stdout)["error"]["kind"] == "bad_option"


def test_an_unexpected_error_is_a_json_error_with_only_the_exception_type(lib_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("secret /home/someone")

    monkeypatch.setattr(cli, "build_symbol_view", boom)

    result = _run(lib_repo, "twin")

    assert result.exit_code == 1 and json.loads(result.stdout)["error"] == {"kind": "internal_error", "message": "RuntimeError"}
    assert "secret" not in result.stdout


def test_text_mode_errors_still_go_to_stderr(lib_repo: Path) -> None:
    result = runner.invoke(app, ["explore", "symbol", "twinn", str(lib_repo)])

    assert result.exit_code == 2 and result.stdout == "" and "twin" in result.stderr
