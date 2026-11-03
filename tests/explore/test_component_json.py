"""`argus explore component NAME --json`: purpose, modules, dependencies both ways, public symbols and how the CLI reaches it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.explore import cli
from argus.explore.components import resolve_name
from argus.explore.entrypoints import find_entrypoints
from argus.explore.views import build_detail
from argus.extract import scan
from tests.explore.json_helpers import claims_in, doc_of, json_keys, run_json, view_keys


def test_the_envelope_and_the_component_identity(mini_repo: Path) -> None:
    doc = doc_of("component", "mini.core", str(mini_repo))

    assert doc["command"] == "component" and doc["ok"] is True and doc["options"] == {"name": "mini.core"}
    assert doc["data"]["label"] == "mini.core" and doc["data"]["modules"] == ["mini.core"]


def test_purpose_is_a_quoted_fact_and_a_missing_docstring_is_unknown(mini_repo: Path) -> None:
    core = doc_of("component", "mini.core", str(mini_repo))["data"]["purpose"]
    store = doc_of("component", "mini.store", str(mini_repo))["data"]["purpose"]

    assert core["tier"] == "FACT" and "Core logic" in core["text"] and core["evidence"][0]["file"] == "src/mini/core.py"
    assert store["tier"] == "UNKNOWN" and store["evidence"] == []


def test_dependencies_are_rows_with_a_count_in_both_directions(mini_repo: Path) -> None:
    data = doc_of("component", "mini.core", str(mini_repo))["data"]

    out = {d["other"]: d for d in data["imports_out"]}
    assert out["mini.store"]["count"] >= 1 and out["mini.store"]["claim"]["tier"] == "FACT"
    assert {d["other"] for d in data["imported_by"]} >= {"mini.cli", "mini.sub"}


def test_public_symbols_are_rows_with_module_name_and_kind(mini_repo: Path) -> None:
    symbols = doc_of("component", "mini.core", str(mini_repo))["data"]["symbols"]

    assert {(s["module"], s["name"], s["kind"]) for s in symbols} >= {("mini.core", "work", "function"), ("mini.core", "helper", "function")}
    assert all(s["claim"]["evidence"] for s in symbols)


def test_reached_from_lists_each_command_and_the_chain_into_the_component(mini_repo: Path) -> None:
    data = doc_of("component", "mini.core", str(mini_repo))["data"]

    reached = {r["command"]["display"]: r for r in data["reached_from"]}
    assert "mini hello-world" in reached
    (chain, *_rest) = reached["mini hello-world"]["chains"]
    assert chain[-1]["module"] == "mini.core" and chain[-1]["callee"] == "mini.core.work" and chain[-1]["claim"]["tier"] == "INFERENCE"
    cli_part = doc_of("component", "mini.cli", str(mini_repo))["data"]
    assert [c["display"] for c in cli_part["own_commands"]] == ["mini hello-world"]


def test_limitations_are_unknown_claims(mini_repo: Path) -> None:
    unknowns = doc_of("component", "mini.core", str(mini_repo))["data"]["unknowns"]

    assert unknowns and all(u["tier"] == "UNKNOWN" and u["evidence"] == [] for u in unknowns)


def test_every_claim_of_the_view_is_in_the_document_unchanged(mini_repo: Path) -> None:
    model = scan(mini_repo)
    label, group_of = resolve_name(model, "mini.core")
    view = build_detail(model, mini_repo, find_entrypoints(mini_repo, model), label, group_of)
    comp = view.component
    expected = [
        comp.purpose, *(s.claim for s in view.symbols), *(d.claim for d in comp.imports_out), *(d.claim for d in comp.imported_by),
        *(c.claim for c in view.own_commands), *(r.command.claim for r in view.reached_from),
        *(s.claim for r in view.reached_from for chain in r.chains for s in chain), *view.unknowns,
    ]

    got = claims_in(doc_of("component", "mini.core", str(mini_repo)))

    assert json_keys(got) == view_keys(expected)


def test_evidence_is_current_relative_and_the_output_is_deterministic(mini_repo: Path) -> None:
    first = run_json("component", "mini.core", str(mini_repo)).stdout
    second = run_json("component", "mini.core", str(mini_repo)).stdout

    assert first == second and str(mini_repo) not in first
    for claim in claims_in(json.loads(first)):
        assert claim["tier"] == "UNKNOWN" or claim["evidence"]
        assert all(e["side"] == "current" for e in claim["evidence"])


def test_an_unknown_component_is_a_json_error(mini_repo: Path) -> None:
    result = run_json("component", "nonexistent", str(mini_repo))

    assert result.exit_code == 2 and json.loads(result.stdout)["error"]["kind"] == "component_lookup"


def test_a_path_that_is_not_a_directory_is_a_json_error(tmp_path: Path) -> None:
    result = run_json("component", "x", str(tmp_path / "missing"))

    assert result.exit_code == 2 and json.loads(result.stdout)["error"]["kind"] == "not_a_directory"


def test_an_unexpected_error_is_a_json_error_with_only_the_exception_type(mini_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("secret /home/x")

    monkeypatch.setattr(cli, "build_detail", boom)

    result = run_json("component", "mini.core", str(mini_repo))

    assert result.exit_code == 1 and json.loads(result.stdout)["error"] == {"kind": "internal_error", "message": "RuntimeError"}
