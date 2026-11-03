"""Discovery must name every module the way an importer would name it."""

from __future__ import annotations

from pathlib import Path

from argus.extract.discovery import discover, package_qualnames


def test_maps_files_under_src_to_import_names(sample_repo: Path) -> None:
    files = discover(sample_repo)

    names = {f.qualname for f in files}
    assert "shop.billing.invoice" in names
    assert "shop" in names  # the package __init__ names the package itself
    assert "src.shop.billing.invoice" not in names


def test_discovers_pep420_namespace_package_modules(sample_repo: Path) -> None:
    names = {f.qualname for f in discover(sample_repo)}

    assert "plugins.alpha.entry" in names


def test_package_qualnames_include_namespace_packages(sample_repo: Path) -> None:
    packages = package_qualnames(discover(sample_repo))

    assert "shop" in packages
    assert "shop.billing" in packages
    assert "plugins" in packages
    assert "plugins.alpha" in packages


def test_excludes_virtualenvs_and_caches(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "vendored.py").write_text("x = 1")
    (tmp_path / "real.py").write_text("x = 1")

    names = {f.qualname for f in discover(tmp_path)}

    assert names == {"real"}


def test_package_init_reports_its_own_package_as_parent(sample_repo: Path) -> None:
    init = next(f for f in discover(sample_repo) if f.qualname == "shop.billing")

    assert init.is_package_init
    assert init.package_qualname == "shop"


DEPS_FILES = {
    "main.py": "import requests\nimport json\n\nrequests.get('x')\n",
    ".deps/requests/__init__.py": "from .api import get\n",
    ".deps/requests/api.py": "def get(url):\n    return url\n",
}


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def test_hidden_directories_are_not_discovered(tmp_path: Path) -> None:
    repo = _write(tmp_path, DEPS_FILES)
    assert [f.qualname for f in discover(repo)] == ["main"]


def test_hidden_directories_can_be_included_explicitly(tmp_path: Path) -> None:
    repo = _write(tmp_path, DEPS_FILES)
    assert any(f.path.parts[-3] == ".deps" for f in discover(repo, include_hidden=True))


def test_application_imports_do_not_resolve_into_excluded_dependency_dirs(tmp_path: Path) -> None:
    from argus.extract import scan

    model = scan(_write(tmp_path, DEPS_FILES))
    assert not [n for n in model.nodes if n.qualname.startswith(".") or "deps" in n.qualname]
    assert not [e for e in model.edges if "deps" in e.dst_id or "deps" in (e.dst_ref or "")]
    assert model.stats.files_scanned == 1


def test_a_hidden_root_directory_does_not_hide_the_whole_repo(tmp_path: Path) -> None:
    repo = _write(tmp_path / ".checkout", {"app.py": "x = 1\n"})
    assert [f.qualname for f in discover(repo)] == ["app"]
