from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from argus.benchmark import provenance
from argus.benchmark.provenance import code_fingerprint, git_revision

# The modules that decide what the agent is shown and how a run is scored.
_ENTRY_MODULES = (
    "argus.benchmark.runner",
    "argus.benchmark.evaluator",
    "argus.benchmark.workspace",
    "argus.executor.aider_executor",
)


def _origin(module: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:  # `from pkg.mod import Name`: the candidate is a name, not a module
        return None
    return Path(spec.origin) if spec and spec.origin else None


def _argus_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("argus"):
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("argus"))
    return found


def _import_closure(entries: tuple[str, ...]) -> set[Path]:
    seen: set[Path] = set()
    todo = list(entries)
    while todo:
        path = _origin(todo.pop())
        if path is None or path in seen:
            continue
        seen.add(path)
        todo.extend(_argus_imports(path))
    return seen


def test_code_fingerprint_is_stable_and_short() -> None:
    assert code_fingerprint() == code_fingerprint() and len(code_fingerprint()) == 12


def test_fingerprint_covers_everything_the_agent_path_imports() -> None:
    covered = {Path(p).resolve() for p in provenance._BEHAVIOUR_SOURCES}
    closure = {p.resolve() for p in _import_closure(_ENTRY_MODULES)}
    # Package __init__ files hold no behaviour here (one-line docstrings).
    missing = {p.name for p in closure - covered if p.name != "__init__.py"}
    assert not missing, f"changing these would not change the run config hash: {sorted(missing)}"


def test_fingerprint_covers_the_task_view_firewall() -> None:
    names = {Path(p).name for p in provenance._BEHAVIOUR_SOURCES}
    assert {"tasks.py", "base.py", "runner.py"} <= names


def test_fingerprint_changes_when_a_covered_file_changes(tmp_path: Path) -> None:
    source = tmp_path / "view.py"
    source.write_text("FIELDS = ('task_id',)\n", encoding="utf-8")
    before = code_fingerprint((str(source),))
    source.write_text("FIELDS = ('task_id', 'gold_patch')\n", encoding="utf-8")
    assert code_fingerprint((str(source),)) != before


def test_git_revision_outside_a_repo_is_explicit(tmp_path: Path) -> None:
    assert git_revision(tmp_path) == "no-git"
