"""Every third-party module the package imports must be declared in pyproject.toml.

A developer environment can hide a missing declaration (another package happens to install the module), but `uv tool install` resolves
fresh and does not: a missing `click` once made the installed `argus` fail at import.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "argus"
EXTERNAL_RUNTIMES = {"aider"}  # `executor/_aider_driver.py` runs inside Aider's own virtualenv, on purpose
IMPORT_TO_DISTRIBUTION = {"yaml": "pyyaml"}


def _declared() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = [*project["dependencies"], *(r for extra in project.get("optional-dependencies", {}).values() for r in extra)]
    return {re.split(r"[<>=!~\[ ;]", requirement, maxsplit=1)[0].lower().replace("_", "-") for requirement in requirements}


def _imported() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
            for name in names:
                top = name.partition(".")[0]
                if top != "argus" and top not in sys.stdlib_module_names and top not in EXTERNAL_RUNTIMES:
                    found.setdefault(top, path)
    return found


def test_every_third_party_import_is_a_declared_dependency() -> None:
    declared = _declared()
    missing = {
        top: path.relative_to(ROOT) for top, path in _imported().items()
        if IMPORT_TO_DISTRIBUTION.get(top, top).lower().replace("_", "-") not in declared
    }

    assert not missing, f"imported but not declared in pyproject.toml: {missing}"


def test_every_declared_dependency_is_imported_somewhere() -> None:
    imported = {IMPORT_TO_DISTRIBUTION.get(top, top).lower().replace("_", "-") for top in _imported()}

    unused = _declared() - imported

    assert not unused, f"declared in pyproject.toml but never imported (each one is installed for nothing): {sorted(unused)}"
