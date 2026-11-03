"""Diff Explorer v0.1, slice 1: materializing the two sides from git (contract: see diff_helpers.py)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tests.explore.diff_helpers import GitRepo


def _materialize(repo_or_path: GitRepo | Path, base: str = "HEAD", head: str | None = None):
    from argus.explore.sides import materialize_sides

    path = repo_or_path.root if isinstance(repo_or_path, GitRepo) else repo_or_path
    return materialize_sides(path, base, head)


def _files_on_disk(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_sides_are_frozen_values_with_sorted_files_labels_and_untracked(shop_repo: GitRepo) -> None:
    from argus.explore.sides import Side

    shop_repo.write("shop/audit.py", "def record(x):\n    return x\n")
    with _materialize(shop_repo) as (base, head):
        assert isinstance(base, Side) and isinstance(head, Side)
        assert base.files == tuple(sorted(["shop/__init__.py", "shop/api.py", "shop/billing.py", "shop/cart.py"]))
        assert head.files == tuple(sorted([*base.files, "shop/audit.py"]))
        assert base.untracked == frozenset() and head.untracked == frozenset({"shop/audit.py"})
        assert base.label.startswith("HEAD (") and shop_repo.short() in base.label
        assert head.label == "working tree"
        with pytest.raises(dataclasses.FrozenInstanceError):
            base.label = "changed"  # type: ignore[misc]


def test_each_side_directory_holds_only_its_python_files_outside_the_checkout(shop_repo: GitRepo) -> None:
    shop_repo.write("README.md", "docs\n")
    shop_repo.write("pyproject.toml", "[project]\nname = 'shop'\n")
    shop_repo.commit("non-python files")
    with _materialize(shop_repo) as (base, head):
        assert _files_on_disk(base.root) == set(base.files) and _files_on_disk(head.root) == set(head.files)
        assert base.root != head.root
        assert not str(base.root).startswith(str(shop_repo.root)) and not str(head.root).startswith(str(shop_repo.root))


def test_working_tree_side_reads_the_disk_not_the_index_and_applies_the_tracking_rules(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/billing.py": "def charge(a):\n    return a\n", "shop/api.py": "X = 1\n"})
    git_repo.write(".gitignore", "generated/\n")
    git_repo.commit("base")
    git_repo.write("shop/billing.py", "def charge(a):\n    return a + 1\n")
    git_repo.stage("shop/billing.py")
    git_repo.write("shop/billing.py", "def charge(a):\n    return a + 2\n")  # unstaged edit on top of the staged one
    git_repo.delete("shop/api.py")  # tracked, deleted on disk, not staged
    git_repo.write("generated/gen.py", "def gen():\n    pass\n")  # ignored
    git_repo.write("shop/new.py", "def new():\n    pass\n")  # untracked, not ignored
    with _materialize(git_repo) as (base, head):
        assert "shop/api.py" in base.files and "shop/api.py" not in head.files
        assert "generated/gen.py" not in base.files and "generated/gen.py" not in head.files
        assert "shop/new.py" in head.files and "shop/new.py" not in base.files
        assert (head.root / "shop/billing.py").read_text(encoding="utf-8").endswith("a + 2\n")
        assert (base.root / "shop/billing.py").read_text(encoding="utf-8").endswith("return a\n")


def test_head_ref_builds_the_head_side_from_a_commit_and_ignores_the_working_tree(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/a.py": "def a():\n    return 1\n"})
    first = git_repo.commit("first")
    git_repo.write("shop/a.py", "def a():\n    return 2\n")
    git_repo.write("shop/b.py", "def b():\n    pass\n")
    git_repo.commit("second")
    git_repo.write("shop/a.py", "def a():\n    return 3\n")  # dirty working tree must be invisible
    git_repo.write("shop/untracked.py", "def u():\n    pass\n")
    with _materialize(git_repo, base=first, head="HEAD") as (base, head):
        assert (base.root / "shop/a.py").read_text(encoding="utf-8").endswith("return 1\n")
        assert (head.root / "shop/a.py").read_text(encoding="utf-8").endswith("return 2\n")
        assert "shop/b.py" in head.files and "shop/untracked.py" not in head.files
        assert head.untracked == frozenset()
        assert head.label.startswith("HEAD (")


@pytest.mark.parametrize("which", ["base", "head"])
@pytest.mark.parametrize("ref", ["--output=PWNED", "-p", "--no-index"])
def test_refs_starting_with_a_dash_are_rejected_before_git_sees_them(shop_repo: GitRepo, tmp_path: Path, which: str, ref: str) -> None:
    from argus.explore.sides import DiffError

    ref = ref.replace("PWNED", str(tmp_path / "pwned"))
    kwargs = {"base": ref} if which == "base" else {"base": "HEAD", "head": ref}
    with pytest.raises(DiffError), _materialize(shop_repo, **kwargs):
        pytest.fail("materialize_sides must not yield for a ref that looks like an option")
    assert not (tmp_path / "pwned").exists()


def test_unknown_base_and_head_refs_raise_diff_error_naming_the_ref(shop_repo: GitRepo) -> None:
    from argus.explore.sides import DiffError

    with pytest.raises(DiffError, match="no-such-ref"), _materialize(shop_repo, base="no-such-ref"):
        pytest.fail("unreachable")
    with pytest.raises(DiffError, match="no-such-head"), _materialize(shop_repo, base="HEAD", head="no-such-head"):
        pytest.fail("unreachable")


def test_directory_outside_a_git_repository_raises_diff_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.explore.sides import DiffError

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))  # never find an enclosing repo above tmp
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "m.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(DiffError, match="(?i)git repository"), _materialize(plain):
        pytest.fail("unreachable")


def test_repository_without_commits_raises_diff_error(git_repo: GitRepo) -> None:
    from argus.explore.sides import DiffError

    git_repo.write("m.py", "x = 1\n")
    with pytest.raises(DiffError), _materialize(git_repo):
        pytest.fail("unreachable")


def test_missing_path_raises_diff_error(tmp_path: Path) -> None:
    from argus.explore.sides import DiffError

    with pytest.raises(DiffError), _materialize(tmp_path / "does-not-exist"):
        pytest.fail("unreachable")


def test_symlinks_and_submodule_entries_are_skipped_on_both_sides(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/real.py": "def real():\n    pass\n"})
    git_repo.symlink("shop/link.py", "real.py")
    git_repo.add_gitlink("vendor/sub.py")
    git_repo.commit("with a symlink and a submodule entry")
    with _materialize(git_repo) as (base, head):
        for side in (base, head):
            assert "shop/real.py" in side.files
            assert "shop/link.py" not in side.files and "vendor/sub.py" not in side.files
            assert not (side.root / "shop/link.py").exists() and not (side.root / "vendor").exists()


def test_files_are_copied_as_raw_bytes_so_a_non_utf8_byte_cannot_crash_materialization(git_repo: GitRepo) -> None:
    raw = b"# caf\xe9 (latin-1)\ndef f():\n    return 1\n"
    git_repo.write("shop/latin.py", raw)
    git_repo.commit("latin-1 source")
    with _materialize(git_repo) as (base, head):
        assert (base.root / "shop/latin.py").read_bytes() == raw
        assert (head.root / "shop/latin.py").read_bytes() == raw


def test_temp_directories_are_deleted_when_the_context_exits(shop_repo: GitRepo, scratch_tmp: Path) -> None:
    with _materialize(shop_repo) as (base, head):
        roots = (base.root, head.root)
        assert all(root.exists() for root in roots)
    assert not any(root.exists() for root in roots)
    assert list(scratch_tmp.iterdir()) == []


def test_temp_directories_are_deleted_when_an_exception_escapes_the_context(shop_repo: GitRepo, scratch_tmp: Path) -> None:
    roots: tuple[Path, Path] | None = None
    with pytest.raises(RuntimeError, match="boom"), _materialize(shop_repo) as (base, head):
        roots = (base.root, head.root)
        raise RuntimeError("boom")
    assert roots is not None and not any(root.exists() for root in roots)
    assert list(scratch_tmp.iterdir()) == []


def test_a_subdirectory_path_scans_only_that_subtree_on_both_sides(git_repo: GitRepo) -> None:
    git_repo.write_files({"pkg/sub/mod.py": "def m():\n    pass\n", "pkg/other.py": "X = 1\n", "top.py": "Y = 2\n"})
    git_repo.commit("layout")
    git_repo.write("pkg/sub/new.py", "def n():\n    pass\n")
    git_repo.write("pkg/untracked_elsewhere.py", "Z = 3\n")
    with _materialize(git_repo.root / "pkg" / "sub") as (base, head):
        assert base.files == ("mod.py",)
        assert head.files == ("mod.py", "new.py") and head.untracked == frozenset({"new.py"})
