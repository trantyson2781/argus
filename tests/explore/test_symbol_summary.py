"""`argus explore symbol`: a short summary block on top (which CLI commands reach it, who calls it, which tests call it directly, how big it is).

Rules under test: the facts come from the existing call graph (no new analysis); every claim here is an INFERENCE and says so; a
symbol no command reaches says "not reached" instead of staying silent; tests are named only when they call the symbol directly
(a lower bound, so the text says "directly"); `--brief` prints the summary only; `--json` only gains fields.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from argus.cli import app
from argus.explore.symbols import build_symbol_view, resolve_symbol
from argus.extract import scan

RUNNER = CliRunner()


def _view(root: Path, query: str):
    model = scan(root)
    return build_symbol_view(model, root, resolve_symbol(model, query))


def test_a_function_knows_how_many_lines_it_spans(mini_repo: Path) -> None:
    assert _view(mini_repo, "mini.core.work").lines == 3


def test_every_command_that_reaches_the_function_is_listed_with_its_shortest_chain(mini_repo: Path) -> None:
    rows = {r.command: r.chain for r in _view(mini_repo, "mini.core.work").reached_from}

    assert rows == {
        "mini hello-world": ("mini.cli.hello_world", "mini.core.work"),
        "mini sub renamed": ("mini.sub.go", "mini.core.work"),
    }


def test_a_deeper_function_gets_the_whole_chain(mini_repo: Path) -> None:
    rows = {r.command: r.chain for r in _view(mini_repo, "mini.core.helper").reached_from}

    assert rows["mini hello-world"] == ("mini.cli.hello_world", "mini.core.work", "mini.core.helper")


def test_the_command_function_itself_counts_as_reached(mini_repo: Path) -> None:
    rows = _view(mini_repo, "mini.cli.hello_world").reached_from

    assert [(r.command, r.chain) for r in rows] == [("mini hello-world", ("mini.cli.hello_world",))]


def test_reach_rows_are_inferences_with_a_call_site(mini_repo: Path) -> None:
    row = _view(mini_repo, "mini.core.helper").reached_from[0]

    assert row.claim.tier.value == "INFERENCE" and row.claim.evidence


def test_tests_are_named_only_when_they_call_the_symbol_directly(mini_repo: Path) -> None:
    assert _view(mini_repo, "mini.core.work").tested_by == ("tests.test_core.test_work",)
    assert _view(mini_repo, "mini.core.helper").tested_by == ()  # only work() calls it, and the test calls work()


def test_a_symbol_no_command_reaches_is_not_silent_about_it(mini_repo: Path) -> None:
    (mini_repo / "src/mini/orphan.py").write_text("def orphan():\n    return 1\n")

    result = RUNNER.invoke(app, ["explore", "symbol", "mini.orphan.orphan", str(mini_repo)])

    assert result.exit_code == 0 and "not reached from any CLI command Argus can see" in result.stdout


def test_the_text_leads_with_a_summary_block_before_the_long_sections(mini_repo: Path) -> None:
    out = RUNNER.invoke(app, ["explore", "symbol", "mini.core.work", str(mini_repo)]).stdout

    assert out.index("Used by CLI") < out.index("Called by (")
    assert "mini hello-world" in out and "mini sub renamed" in out
    assert "Chain" in out and "cli.hello_world -> core.work" in out
    assert "Tested by" in out and "directly" in out and "test_work" in out
    assert "3 lines" in out


def test_brief_prints_the_summary_and_none_of_the_long_sections(mini_repo: Path) -> None:
    out = RUNNER.invoke(app, ["explore", "symbol", "mini.core.work", str(mini_repo), "--brief"]).stdout

    assert "Used by CLI" in out and "Called by (" not in out and "Calls (" not in out and "Unresolved calls" not in out


def test_json_only_gains_fields_and_a_module_has_no_reach_or_tests(mini_repo: Path) -> None:
    func = json.loads(RUNNER.invoke(app, ["explore", "symbol", "mini.core.work", str(mini_repo), "--json"]).stdout)["data"]
    module = json.loads(RUNNER.invoke(app, ["explore", "symbol", "mini.core", str(mini_repo), "--json"]).stdout)["data"]

    assert func["lines"] == 3 and func["tested_by"] == ["tests.test_core.test_work"]
    first = func["reached_from"][0]
    assert first["command"] == "mini hello-world" and first["path"] == ["mini.cli.hello_world", "mini.core.work"] and first["claim"]["tier"] == "INFERENCE"
    assert module["reached_from"] is None and module["tested_by"] is None


def test_the_help_lists_brief() -> None:
    assert "--brief" in RUNNER.invoke(app, ["explore", "symbol", "--help"]).stdout


def _tiny(tmp_path: Path, source: str, tests: str = "") -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\n')
    (tmp_path / "p.py").write_text(source)
    if tests:
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_p.py").write_text(tests)
    return tmp_path


def test_a_caller_that_calls_twice_is_one_caller(tmp_path: Path) -> None:
    root = _tiny(tmp_path, "def work():\n    return 1\n\n\ndef f():\n    work()\n    return work()\n")

    out = RUNNER.invoke(app, ["explore", "symbol", "work", str(root), "--brief"]).stdout

    assert "1 resolved: p.f  [I]" in out


def test_tests_in_different_classes_keep_their_class_and_helpers_are_not_tests(tmp_path: Path) -> None:
    tests = (
        "from p import work\n\n\nclass TestA:\n    def test_x(self):\n        work()\n\n\n"
        "class TestB:\n    def test_x(self):\n        work()\n\n\ndef make_thing():\n    return work()\n"
    )
    root = _tiny(tmp_path, "def work():\n    return 1\n", tests)

    view = _view(root, "p.work")
    out = RUNNER.invoke(app, ["explore", "symbol", "work", str(root), "--brief"]).stdout

    assert view.tested_by == ("tests.test_p.TestA.test_x", "tests.test_p.TestB.test_x")
    assert "TestA.test_x, TestB.test_x" in out and "make_thing" not in out


def test_a_project_without_a_cli_does_not_claim_the_symbol_is_unreached(tmp_path: Path) -> None:
    root = _tiny(tmp_path, "def work():\n    return 1\n")

    out = RUNNER.invoke(app, ["explore", "symbol", "work", str(root)]).stdout

    assert "no CLI commands found in this project" in out and "not reached from any CLI command" not in out


def test_brief_keeps_a_modules_members_and_says_what_it_hid(mini_repo: Path) -> None:
    module = RUNNER.invoke(app, ["explore", "symbol", "mini.core", str(mini_repo), "--brief"]).stdout
    hiding = RUNNER.invoke(app, ["explore", "symbol", "mini.core.work", str(mini_repo), "--brief"]).stdout

    assert "Contains" in module and "mini.core.work" in module
    assert "Unresolved calls" not in hiding  # helper() and save() both resolve here, so nothing is hidden


def test_brief_with_json_is_refused(mini_repo: Path) -> None:
    result = RUNNER.invoke(app, ["explore", "symbol", "mini.core.work", str(mini_repo), "--json", "--brief"])

    assert result.exit_code == 2 and "--brief" in json.loads(result.stdout)["error"]["message"]


def test_a_malformed_pyproject_is_reported_as_an_unresolved_entry_point_spot(tmp_path: Path) -> None:
    root = _tiny(tmp_path, "def work():\n    return 1\n")
    (root / "pyproject.toml").write_text('project = "not a table"\n')

    text = RUNNER.invoke(app, ["explore", "symbol", "work", str(root)])

    assert text.exit_code == 0 and "no CLI commands found in this project" in text.stdout and "could not resolve" in text.stdout


def test_a_failing_entry_point_scan_never_takes_the_symbol_view_down(tmp_path: Path, monkeypatch) -> None:
    from argus.explore import symbols

    def boom(*_args):
        raise RuntimeError("scan blew up")

    monkeypatch.setattr(symbols, "find_entrypoints", boom)
    root = _tiny(tmp_path, "def work():\n    return 1\n")

    text = RUNNER.invoke(app, ["explore", "symbol", "work", str(root)])
    doc = json.loads(RUNNER.invoke(app, ["explore", "symbol", "work", str(root), "--json"]).stdout)

    assert text.exit_code == 0 and "Used by CLI   unknown: the entry-point scan failed (RuntimeError)" in text.stdout
    assert doc["ok"] is True and doc["data"]["reached_from"] is None and "RuntimeError" in doc["data"]["reach_caveat"]


def test_unresolved_entry_point_spots_are_named_instead_of_blaming_obj_method(tmp_path: Path) -> None:
    cli = "import typer\n\napp = typer.Typer()\nname = 'x'\n\n\n@app.command(name=name)\ndef dyn():\n    return 1\n\n\n@app.command()\ndef real():\n    return 2\n"
    root = _tiny(tmp_path, "def work():\n    return 1\n")
    (root / "cli.py").write_text(cli)
    (root / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\n[project.scripts]\np = "cli:app"\n')

    out = RUNNER.invoke(app, ["explore", "symbol", "work", str(root)]).stdout

    assert "not reached from any CLI command Argus can see" in out and "could not resolve" in out and "obj.method()" not in out


def test_tests_are_a_lower_bound_and_hidden_test_code_callers_are_mentioned(tmp_path: Path) -> None:
    tests = "from p import work\n\nVALUE = work()\n"  # module level: not a test function
    root = _tiny(tmp_path, "def work():\n    return 1\n", tests)

    out = RUNNER.invoke(app, ["explore", "symbol", "work", str(root), "--brief"]).stdout

    assert "no test function calls it directly" in out and "1 caller(s) in test code" in out


def test_guard_no_cli_chain_hop_is_a_self_method_call_so_the_chain_row_needs_no_override_caveat(tmp_path: Path) -> None:
    """A chain starts at a command function and follows resolved calls. Today the only resolved calls that land on a method are
    `self.method()` calls, and those start inside another method, so no chain can contain one. If this test fails, the resolver
    started linking calls into methods: carry `override_possible` per hop into `ReachRow` and print the caveat on the Chain row."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "sc"\nversion = "0"\n[project.scripts]\nsc = "sc.cli:app"\n')
    (tmp_path / "src/sc").mkdir(parents=True)
    (tmp_path / "src/sc/__init__.py").write_text("")
    (tmp_path / "src/sc/cli.py").write_text("import typer\n\nfrom sc.svc import Svc\n\napp = typer.Typer()\n\n\n@app.command()\ndef go():\n    Svc.entry()\n    return helper()\n\n\ndef helper():\n    return 1\n")
    (tmp_path / "src/sc/svc.py").write_text(
        "class Svc:\n    @classmethod\n    def entry(cls):\n        return 1\n\n    def a(self):\n        return self.b()\n\n    def b(self):\n        return 2\n"
    )

    for query in ("sc.svc.Svc.a", "sc.svc.Svc.b", "sc.svc.Svc.entry"):
        assert _view(tmp_path, query).reached_from == ()  # not reached: nothing resolved leads into a method from a function
    assert [r.chain for r in _view(tmp_path, "sc.cli.helper").reached_from] == [("sc.cli.go", "sc.cli.helper")]
