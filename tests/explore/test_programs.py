"""Plain programs are entry points too: a `pyproject` script that points at a function, and an `if __name__ == "__main__":` block that calls repo functions.

Rules under test: every repo function called anywhere in a main guard is a root, shown as `python <file>` (so `main()`, `sys.exit(main())`,
`raise SystemExit(main())` and plain calls all work); a script whose target is a plain function is a root shown under the script's name; guards
that only run the test framework, guards in test code and calls Argus cannot resolve to a function of the repository (methods, library calls)
are no entry points and make no noise; programs are kept apart from CLI commands and routes, so every project read before reads the same now;
"reached from" counts them and the label changes only when there is something other than CLI commands.
"""

from __future__ import annotations

from pathlib import Path

from argus.explore.claims import Tier
from argus.explore.entrypoints import find_entrypoints
from argus.explore.symbols import build_symbol_view
from argus.extract import scan


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _programs(root: Path) -> dict[tuple[str, str], str]:
    """(display, function id) -> file"""
    return {(p.display, p.function_id): p.file for p in find_entrypoints(root, scan(root)).programs}


TOOL = "def main():\n    return 1\n\n\nif __name__ == '__main__':\n    main()\n"


def test_a_main_guard_that_calls_a_local_function_is_an_entry_point(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/tool.py": TOOL})

    assert _programs(tmp_path) == {("python p/tool.py", "function:p.tool.main"): "p/tool.py"}


def test_the_program_is_a_fact_with_the_guard_line_as_evidence(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/tool.py": TOOL})

    (program,) = find_entrypoints(tmp_path, scan(tmp_path)).programs

    assert program.claim.tier is Tier.FACT and program.claim.evidence[0].line == 5 and program.module == "p.tool"


def test_sys_exit_and_systemexit_wrappers_are_looked_through(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "",
        "p/a.py": "import sys\n\n\ndef main():\n    return 0\n\n\nif __name__ == '__main__':\n    sys.exit(main())\n",
        "p/b.py": "def main():\n    return 0\n\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n",
    })

    assert set(_programs(tmp_path)) == {("python p/a.py", "function:p.a.main"), ("python p/b.py", "function:p.b.main")}


def test_a_function_imported_into_the_guard_module_is_followed(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "",
        "p/core.py": "def run():\n    return 1\n",
        "p/go.py": "from p.core import run\n\nif __name__ == '__main__':\n    run()\n",
        "p/via.py": "from p import core\n\nif __name__ == '__main__':\n    core.run()\n",
    })

    assert set(_programs(tmp_path)) == {("python p/go.py", "function:p.core.run"), ("python p/via.py", "function:p.core.run")}


def test_every_repo_function_called_in_the_guard_is_a_root_once(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/job.py": "def a():\n    pass\n\n\ndef b():\n    pass\n\n\nif __name__ == '__main__':\n    a()\n    b()\n    a()\n"})

    assert set(_programs(tmp_path)) == {("python p/job.py", "function:p.job.a"), ("python p/job.py", "function:p.job.b")}


def test_test_framework_guards_are_not_entry_points(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/a.py": "import unittest\n\nif __name__ == '__main__':\n    unittest.main()\n",
                      "p/b.py": "import pytest\n\nif __name__ == '__main__':\n    pytest.main(['-q'])\n"})
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert eps.programs == () and eps.unknowns == ()


def test_methods_and_library_calls_in_a_guard_are_not_entry_points_and_not_noise(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/app.py": "import logging\n\n\nclass App:\n    def run(self):\n        pass\n\n\nif __name__ == '__main__':\n    logging.basicConfig()\n    App().run()\n"})
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert eps.programs == () and eps.unknowns == ()


def test_guards_in_test_code_are_not_entry_points(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "tests/test_x.py": "def helper():\n    pass\n\n\nif __name__ == '__main__':\n    helper()\n"})

    assert _programs(tmp_path) == {}


def test_a_typer_program_stays_a_command_and_its_guard_adds_nothing(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/cli.py": "import typer\n\napp = typer.Typer()\n\n\n@app.command()\ndef go():\n    pass\n\n\nif __name__ == '__main__':\n    app()\n"})
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert eps.programs == () and [c.display for c in eps.commands] == ["p.cli:app go"]


# -- pyproject scripts -----------------------------------------------------------------------------------------------------

PROJECT = '[project]\nname = "p"\nversion = "0"\n[project.scripts]\n'


def test_a_script_that_points_at_a_plain_function_is_an_entry_point_under_the_scripts_name(tmp_path: Path) -> None:
    _write(tmp_path, {"pyproject.toml": PROJECT + 'tool = "p.run:main"\n', "p/__init__.py": "", "p/run.py": "def main():\n    pass\n"})
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert [(p.display, p.function_id) for p in eps.programs] == [("tool", "function:p.run.main")]
    assert not any("not a Typer app" in c.text for c in eps.unknowns)
    assert eps.programs[0].claim.tier is Tier.FACT and eps.programs[0].claim.evidence[0].file == "pyproject.toml"


def test_a_script_below_a_source_folder_is_found_by_its_import_name(tmp_path: Path) -> None:
    _write(tmp_path, {"pyproject.toml": PROJECT + 'tool = "pkg.run:main"\n', "apps/api/src/pkg/__init__.py": "", "apps/api/src/pkg/run.py": "def main():\n    pass\n"})

    assert [p.display for p in find_entrypoints(tmp_path, scan(tmp_path)).programs] == ["tool"]


def test_a_script_whose_target_is_not_a_function_of_the_repository_stays_unknown(tmp_path: Path) -> None:
    _write(tmp_path, {"pyproject.toml": PROJECT + 'tool = "p.run:missing"\nother = "elsewhere:main"\n', "p/__init__.py": "", "p/run.py": "def main():\n    pass\n"})
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert eps.programs == () and sum("script `" in c.text for c in eps.unknowns) == 2


def test_a_script_and_a_guard_for_the_same_function_are_both_listed(tmp_path: Path) -> None:
    _write(tmp_path, {"pyproject.toml": PROJECT + 'tool = "p.run:main"\n', "p/__init__.py": "", "p/run.py": TOOL})

    assert sorted(p.display for p in find_entrypoints(tmp_path, scan(tmp_path)).programs) == ["python p/run.py", "tool"]


# -- reached from ----------------------------------------------------------------------------------------------------------


def _symbol(root: Path, node_id: str):
    model = scan(root)
    return build_symbol_view(model, root, next(n for n in model.nodes if n.id == node_id))


def test_a_function_called_by_a_program_is_reached_from_it_and_the_label_says_so(tmp_path: Path) -> None:
    from argus.explore.render_symbol import _reach_rows

    _write(tmp_path, {
        "p/__init__.py": "", "p/core.py": "def charge(x):\n    return x\n",
        "p/tool.py": "from p.core import charge\n\n\ndef main():\n    return charge(1)\n\n\nif __name__ == '__main__':\n    main()\n",
    })

    view = _symbol(tmp_path, "function:p.core.charge")

    assert [r.command for r in view.reached_from] == ["python p/tool.py"] and view.commands_found == 1
    assert _reach_rows(view)[0][0] == "Reached from"


def test_a_cli_only_project_keeps_its_old_label(tmp_path: Path) -> None:
    from argus.explore.render_symbol import _reach_rows

    _write(tmp_path, {"p/__init__.py": "", "p/cli.py": "import typer\nfrom p.core import charge\n\napp = typer.Typer()\n\n\n@app.command()\ndef go():\n    charge(1)\n",
                      "p/core.py": "def charge(x):\n    return x\n"})

    assert _reach_rows(_symbol(tmp_path, "function:p.core.charge"))[0][0] == "Used by CLI"


def test_explore_entrypoints_lists_the_programs(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from argus.cli import app

    _write(tmp_path, {"p/__init__.py": "", "p/tool.py": TOOL})

    result = CliRunner().invoke(app, ["explore", "entrypoints", str(tmp_path)])

    assert result.exit_code == 0 and "python p/tool.py" in result.output and "p.tool.main" in result.output


def test_a_script_that_points_at_a_function_the_package_re_exports_is_followed_to_its_definition(tmp_path: Path) -> None:
    _write(tmp_path, {
        "pyproject.toml": PROJECT + 'tool = "p:main"\n', "p/__init__.py": "from p.cli import main\n", "p/cli.py": "def main():\n    pass\n",
    })
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert [(x.display, x.function_id) for x in eps.programs] == [("tool", "function:p.cli.main")]
    assert not any("not a Typer app" in c.text for c in eps.unknowns)


def test_a_re_export_of_something_that_is_not_a_function_stays_unknown(tmp_path: Path) -> None:
    _write(tmp_path, {
        "pyproject.toml": PROJECT + 'tool = "p:main"\n', "p/__init__.py": "from p.cli import main\n", "p/cli.py": "class main:\n    pass\n",
    })
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert eps.programs == () and any("script `tool`" in c.text for c in eps.unknowns)


# -- found by the audit ----------------------------------------------------------------------------------------------------

CORE3 = "def a():\n    pass\n\n\ndef b():\n    pass\n\n\ndef c():\n    pass\n"


def test_calls_the_guard_does_not_execute_are_not_roots(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/core.py": CORE3, "p/m.py": (
        "from typing import TYPE_CHECKING\nfrom p.core import a, b, c\n\nif __name__ == '__main__':\n"
        "    def helper():\n        a()\n\n    f = lambda: b()\n\n    class K:\n        def run(self):\n            a()\n\n"
        "    if False:\n        c()\n    if TYPE_CHECKING:\n        c()\n"
    )})

    assert _programs(tmp_path) == {}


def test_the_else_branch_of_a_dead_if_still_runs(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/core.py": CORE3, "p/m.py": "from p.core import a, b\n\nif __name__ == '__main__':\n    if False:\n        a()\n    else:\n        b()\n"})

    assert set(_programs(tmp_path)) == {("python p/m.py", "function:p.core.b")}


def test_a_name_the_guard_rebinds_no_longer_means_the_import(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/core.py": CORE3, "p/m.py": (
        "from p import core\nfrom p.core import a, b\n\nif __name__ == '__main__':\n    a = print\n    a(1)\n    core = object()\n    core.b()\n"
    )})

    assert _programs(tmp_path) == {}


def test_a_name_rebound_at_module_level_no_longer_means_the_function(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/m.py": "def main():\n    pass\n\n\nmain = lambda: None\n\nif __name__ == '__main__':\n    main()\n"})

    assert _programs(tmp_path) == {}


def test_a_context_manager_function_is_not_a_root_because_calling_it_runs_nothing(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/m.py": (
        "import contextlib\nfrom contextlib import contextmanager\n\n\n@contextmanager\ndef cm():\n    yield\n\n\n@contextlib.contextmanager\ndef cm2():\n    yield\n\n\n"
        "def real():\n    pass\n\n\nif __name__ == '__main__':\n    cm()\n    cm2()\n    real()\n"
    )})

    assert set(_programs(tmp_path)) == {("python p/m.py", "function:p.m.real")}


def test_one_guard_that_reaches_a_function_through_two_roots_is_listed_once(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "", "p/core.py": "def charge():\n    pass\n",
        "p/m.py": "from p.core import charge\n\n\ndef a():\n    charge()\n\n\ndef b():\n    charge()\n\n\nif __name__ == '__main__':\n    a()\n    b()\n",
    })

    view = _symbol(tmp_path, "function:p.core.charge")

    assert [r.command for r in view.reached_from] == ["python p/m.py"]
