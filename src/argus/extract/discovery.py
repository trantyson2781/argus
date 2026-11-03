"""Find Python source files and map each one to its module qualname.

Getting the qualname right is a precondition for everything else: import
resolution, re-export canonicalization and node identity all key off it.
PEP 420 namespace packages (directories with no `__init__.py`) are handled by
resolving paths against source roots rather than by walking up for `__init__.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXCLUDED_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox",
    "node_modules", "site-packages", ".eggs",
    ".venv", "venv", ".env", "env", "virtualenv",
    "build", "dist", ".idea", ".vscode",
})

HIDDEN_DIR_PREFIX = "."
"""A directory named `.x` cannot be imported, so it holds tooling or installed dependencies (e.g. `pip install --target .runtime-deps`)."""

DEFAULT_SOURCE_ROOT_CANDIDATES = ("src", "lib")


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One Python file, resolved to its import-visible module name."""

    path: Path
    qualname: str
    source_root: Path
    is_package_init: bool

    @property
    def package_qualname(self) -> str:
        """The package this module lives in ('' for a top-level module)."""
        return self.qualname.rpartition(".")[0]


def find_source_roots(root: Path) -> tuple[Path, ...]:
    """Directories that behave as import roots, most specific first."""
    roots = [root / c for c in DEFAULT_SOURCE_ROOT_CANDIDATES if (root / c).is_dir()]
    roots.append(root)
    return tuple(roots)


def is_excluded_dir(name: str, include_hidden: bool = False) -> bool:
    """The single exclusion policy for automatic discovery; `include_hidden` is the override hook."""
    if name in EXCLUDED_DIRS or name.endswith(".egg-info"):
        return True
    return not include_hidden and name.startswith(HIDDEN_DIR_PREFIX)


def _is_excluded(path: Path, root: Path, include_hidden: bool = False) -> bool:
    # Only directories below `root` count: a repo checked out under a hidden path must still be scanned.
    return any(is_excluded_dir(part, include_hidden) for part in path.relative_to(root).parts[:-1])


def qualname_for(path: Path, source_roots: tuple[Path, ...]) -> tuple[str, Path] | None:
    """Map a file path to (qualname, source_root), or None if outside every root."""
    for source_root in source_roots:
        try:
            relative = path.relative_to(source_root)
        except ValueError:
            continue
        parts = list(relative.parts)
        stem = parts[-1].removesuffix(".py")
        parts = parts[:-1] if stem == "__init__" else [*parts[:-1], stem]
        if not parts:
            # `src/__init__.py` — degenerate; treat the root itself as the module.
            return source_root.name, source_root
        return ".".join(parts), source_root
    return None


def discover(root: Path, include_hidden: bool = False) -> tuple[SourceFile, ...]:
    """Walk `root` and return every importable Python file, sorted by qualname."""
    root = root.resolve()
    if root.is_file():
        source_roots = find_source_roots(root.parent)
        candidates = [root]
    else:
        source_roots = find_source_roots(root)
        candidates = [p for p in root.rglob("*.py") if not _is_excluded(p, root, include_hidden)]

    found: dict[str, SourceFile] = {}
    for path in candidates:
        resolved = qualname_for(path, source_roots)
        if resolved is None:
            continue
        qualname, source_root = resolved
        # A file reachable from several roots keeps its most specific name; the
        # first matching root already is the most specific one.
        found.setdefault(
            qualname,
            SourceFile(
                path=path,
                qualname=qualname,
                source_root=source_root,
                is_package_init=path.name == "__init__.py",
            ),
        )
    return tuple(sorted(found.values(), key=lambda f: f.qualname))


def package_qualnames(files: tuple[SourceFile, ...]) -> tuple[str, ...]:
    """Every package implied by the discovered modules, including namespace ones."""
    packages: set[str] = set()
    for source_file in files:
        parts = source_file.qualname.split(".")
        # A package __init__ names the package itself; a module names its parents.
        depth = len(parts) if source_file.is_package_init else len(parts) - 1
        for index in range(1, depth + 1):
            packages.add(".".join(parts[:index]))
    return tuple(sorted(packages))
