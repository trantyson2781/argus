"""Record mode runs the repository's code, so only the explicit `argus record` command may reach it.

The standing rule: never from `explore watch`, the TUI or any other explore view. This test reads their source, not their behaviour, so
a later import cannot slip in unnoticed.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "argus"


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return found


def runs_code(name: str) -> bool:
    """The recorder, the test runner, the commands built on them and the code the recorder injects execute the repository; the store and the readers do not."""
    return name in {"argus.record.run", "argus.record.cli", "argus.record.runner", "argus.record.check"} or name.startswith("argus.record.boot")


def test_the_watching_views_never_import_the_recorder() -> None:
    offenders = [
        f"{path.relative_to(SRC)} imports {name}"
        for folder in ("explore", "tui")
        for path in sorted((SRC / folder).rglob("*.py"))
        for name in sorted(imports_of(path))
        if runs_code(name)
    ]

    assert offenders == []


def test_the_guard_can_see_an_import(tmp_path: Path) -> None:
    sample = tmp_path / "x.py"
    sample.write_text("from argus.record.run import record_command\nfrom argus.record import run\nimport argus.record\n", encoding="utf-8")

    assert {"argus.record.run", "argus.record"} <= imports_of(sample)
    assert runs_code("argus.record.run") and runs_code("argus.record.boot.sitecustomize") and not runs_code("argus.record.observed")
    assert runs_code("argus.record.runner") and runs_code("argus.record.check")
    assert not runs_code("argus.record.store")
