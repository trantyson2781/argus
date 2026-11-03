"""A tiny Typer project with known structure for explorer tests."""

from __future__ import annotations

from pathlib import Path

import pytest

MINI_FILES: dict[str, str] = {
    "pyproject.toml": '[project]\nname = "mini"\nversion = "0"\n\n[project.scripts]\nmini = "mini.cli:app"\n',
    "src/mini/__init__.py": '"""Mini tool."""\n',
    "src/mini/__main__.py": 'from mini.cli import app\n\nif __name__ == "__main__":\n    app()\n',
    "src/mini/cli.py": (
        "import typer\nfrom external_lib import ext_app\n\nfrom mini.core import work\nfrom mini.sub import sub_app\n\n"
        "app = typer.Typer()\napp.add_typer(sub_app, name='sub')\napp.add_typer(ext_app, name='ext')\n\n\n"
        "@app.command()\ndef hello_world():\n    work()\n"
    ),
    "src/mini/sub.py": (
        '"""Sub commands."""\nimport typer\n\nfrom .core import work\n\nsub_app = typer.Typer()\n\n\n'
        '@sub_app.command("renamed")\ndef go():\n    work()\n'
    ),
    "src/mini/core.py": (
        '"""Core logic."""\nfrom mini.store.db import save\n\n\ndef work():\n    helper()\n    save()\n\n\ndef helper():\n    pass\n'
    ),
    "src/mini/store/__init__.py": "",
    "src/mini/store/db.py": "def save():\n    pass\n",
    "src/mini/broken.py": "def (:\n",
    "tests/test_core.py": "from mini.core import work\n\n\ndef test_work():\n    work()\n",
}


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    for rel, content in MINI_FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


# -- Diff Explorer v0.1: real git repositories in tmp_path ------------------------------------------------------------

import shutil  # noqa: E402
import tempfile  # noqa: E402

from tests.explore.diff_helpers import GIT_ENV, SHOP_FILES, GitRepo  # noqa: E402


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GitRepo:
    """An empty git repo. Git sees no global or system config, for the test and for the code under test."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)
    return GitRepo(tmp_path / "repo")


@pytest.fixture
def shop_repo(git_repo: GitRepo) -> GitRepo:
    """A small committed shop package: billing.charge/fee, cart.checkout/checkout_twice, api.view."""
    git_repo.write_files(SHOP_FILES)
    git_repo.commit("baseline")
    return git_repo


@pytest.fixture
def scratch_tmp(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the temp directory so a test can see every directory the code under test creates (and that it cleans up)."""
    scratch = tmp_path_factory.mktemp("scratch")
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return scratch
