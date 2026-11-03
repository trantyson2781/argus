"""`--json` for symbol / entrypoints / component, audit round 2: null for sections that were not computed, error scopes, determinism and text-mode guards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from argus.cli import app
from argus.explore import cli
from tests.explore.json_helpers import doc_of, run_json, runner
from tests.explore.test_symbols import LIB_FILES


@pytest.fixture(autouse=True)
def _default_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_RESOLVE_V2", raising=False)


@pytest.fixture
def lib_repo(tmp_path: Path) -> Path:
    for rel, content in LIB_FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


# -- sections that were not computed are null, not an empty list --------------------------------------------------------------------


def test_a_function_has_the_full_call_scope(lib_repo: Path) -> None:
    data = doc_of("symbol", "Circle.describe", str(lib_repo))["data"]

    assert data["calls_scope"] == "full" and isinstance(data["callees"], list) and isinstance(data["unresolved_calls"], list)


def test_a_class_has_callers_but_its_call_sections_are_null(lib_repo: Path) -> None:
    data = doc_of("symbol", "Circle", str(lib_repo))["data"]

    assert data["calls_scope"] == "callers_only" and [c["name"] for c in data["callers"]] == ["tinylib.shapes.make"]
    assert data["callees"] is None and data["unresolved_calls"] is None and data["external_calls"] is None and data["builtin_call_count"] is None


def test_a_module_has_no_call_data_at_all(lib_repo: Path) -> None:
    data = doc_of("symbol", "tinylib.units", str(lib_repo))["data"]

    assert data["calls_scope"] == "none" and data["hidden_test_callers"] is None and data["callers"] is None


# -- structured fields that were only checked in their empty case ----------------------------------------------------------------------


def test_a_call_made_at_module_level_is_a_module_level_caller(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("def target():\n    return 1\n\n\nVALUE = target()\n", encoding="utf-8")

    (caller,) = doc_of("symbol", "target", str(tmp_path))["data"]["callers"]

    assert caller["module_level"] is True and caller["kind"] == "module" and caller["name"] == "m" and "(module level)" in caller["claim"]["text"]


def test_external_calls_name_the_third_party_code_called(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("import requests\n\n\ndef fetch(url):\n    return requests.get(url)\n", encoding="utf-8")

    data = doc_of("symbol", "fetch", str(tmp_path))["data"]

    assert data["external_calls"] == ["requests.get"]


# -- error scope ---------------------------------------------------------------------------------------------------------------------


def test_a_bug_in_build_detail_is_an_internal_error_not_a_component_lookup(mini_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise KeyError("secret/path.py")

    monkeypatch.setattr(cli, "build_detail", boom)

    result = run_json("component", "mini.core", str(mini_repo))

    assert result.exit_code == 1 and json.loads(result.stdout)["error"] == {"kind": "internal_error", "message": "KeyError"}
    assert "secret" not in result.stdout


def test_a_component_with_no_production_modules_is_still_a_component_lookup_error(mini_repo: Path) -> None:
    result = run_json("component", "tests", str(mini_repo))

    assert result.exit_code == 2 and json.loads(result.stdout)["error"]["kind"] == "component_lookup"


@pytest.mark.parametrize(
    ("args", "target"),
    [(("symbol", "twin"), "build_symbol_view"), (("entrypoints",), "find_entrypoints"), (("component", "mini.core"), "build_detail")],
)
@pytest.mark.parametrize("error", [RecursionError, MemoryError])
def test_deep_nesting_is_a_clean_error_in_both_modes(
    lib_repo: Path, mini_repo: Path, monkeypatch: pytest.MonkeyPatch, args: tuple[str, ...], target: str, error: type[BaseException]
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise error()

    monkeypatch.setattr(cli, target, boom)
    repo = lib_repo if args[0] == "symbol" else mini_repo

    as_json = runner.invoke(app, ["explore", *args, str(repo), "--json"])
    as_text = runner.invoke(app, ["explore", *args, str(repo)])

    assert as_json.exit_code == 2 and json.loads(as_json.stdout)["error"]["kind"] == "too_deeply_nested"
    assert as_text.exit_code == 2 and as_text.stdout == "" and "nested too deeply" in as_text.stderr and "Traceback" not in as_text.stderr


# -- determinism for the other two commands, and text-mode guards ---------------------------------------------------------------------------


@pytest.mark.parametrize("args", [("entrypoints", "--command", "hello-world"), ("component", "mini.core")])
def test_the_output_does_not_depend_on_the_hash_seed(mini_repo: Path, args: tuple[str, ...]) -> None:
    outputs = set()
    for seed in ("1", "2", "3", "4"):
        proc = subprocess.run(
            [sys.executable, "-c", "from argus.cli import app; app()", "explore", args[0], *([str(mini_repo)] if args[0] == "entrypoints" else [args[1], str(mini_repo)]),
             *(args[1:] if args[0] == "entrypoints" else []), "--json"],
            capture_output=True, text=True, check=True, timeout=120, env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert json.loads(proc.stdout)["ok"] is True
        outputs.add(proc.stdout)

    assert len(outputs) == 1


def test_the_text_report_still_caps_callers_at_15_and_limit_changes_it(tmp_path: Path) -> None:
    calls = "".join(f"def c{i:02d}():\n    return target()\n\n\n" for i in range(30))
    (tmp_path / "m.py").write_text("def target():\n    return 1\n\n\n" + calls, encoding="utf-8")

    default = runner.invoke(app, ["explore", "symbol", "target", str(tmp_path)]).output
    three = runner.invoke(app, ["explore", "symbol", "target", str(tmp_path), "--limit", "3"]).output

    assert "+15 more" in default and default.count(" calls m.target") == 15
    assert "+27 more" in three and three.count(" calls m.target") == 3


def test_a_trace_names_its_command_with_the_same_object_as_the_command_list(mini_repo: Path) -> None:
    data: dict[str, Any] = doc_of("entrypoints", str(mini_repo), "--command", "hello-world")["data"]

    (trace,) = data["traces"]
    assert trace["command"] in data["commands"] and "claim" not in trace


def test_the_not_a_directory_text_message_still_shows_the_resolved_path(tmp_path: Path) -> None:
    result = runner.invoke(app, ["explore", "entrypoints", str(tmp_path / "missing")])

    assert result.exit_code == 2 and str(tmp_path / "missing") in result.stderr
