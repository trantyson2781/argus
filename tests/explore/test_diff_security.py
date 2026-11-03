"""Diff Explorer: the command must be safe to point at a repository you did not write (git config, symlinks, hostile paths)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from tests.explore.diff_helpers import BILLING, GitRepo, before_notes, diff_out, not_determinable, run_diff, symbol_lines

BODY_EDIT = BILLING.replace("amount + 1", "amount + 2")


def _arm_repo_local_command(repo: GitRepo, tmp_path: Path, key: str) -> Path:
    """Set `key` in the repository's own .git/config to a script that leaves a marker file when it runs."""
    marker = tmp_path / "executed.marker"
    script = tmp_path / "payload.sh"
    script.write_text(f"#!/bin/sh\necho ran >> {marker}\nprintf '\\0'\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    repo.git("config", key, str(script))
    return marker


def test_a_fsmonitor_command_configured_in_the_repository_is_never_executed(shop_repo: GitRepo, tmp_path: Path) -> None:
    key = "core.fsmonitor"  # the one key that plain `git ls-files` executes (verified); the argv test in test_diff_audit_fixes covers the rest
    marker = _arm_repo_local_command(shop_repo, tmp_path, key)
    shop_repo.write("shop/billing.py", BODY_EDIT)
    shop_repo.write("shop/new.py", "def fresh():\n    return 1\n")  # untracked file: exercises `ls-files --others`

    assert run_diff(shop_repo).exit_code == 0
    assert run_diff(shop_repo, "--base", "HEAD", "--head", "HEAD").exit_code == 0

    assert not marker.exists(), f"git ran the command set in {key} inside the repository being analysed"


def test_a_command_configured_in_a_worktree_config_is_never_executed(shop_repo: GitRepo, tmp_path: Path) -> None:
    shop_repo.git("config", "extensions.worktreeConfig", "true")
    marker = tmp_path / "executed.marker"
    script = tmp_path / "payload.sh"
    script.write_text(f"#!/bin/sh\necho ran >> {marker}\nprintf '\\0'\n", encoding="utf-8")
    script.chmod(0o755)
    shop_repo.git("config", "--worktree", "core.fsmonitor", str(script))

    assert run_diff(shop_repo).exit_code == 0
    assert not marker.exists()


def test_a_tracked_directory_replaced_by_a_symlink_to_outside_the_repository_is_not_read(shop_repo: GitRepo, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "billing.py").write_text("def leaked_secret():\n    return 'TOP-SECRET-VALUE'\n", encoding="utf-8")
    for rel in ("shop/__init__.py", "shop/billing.py", "shop/cart.py", "shop/api.py"):
        shop_repo.delete(rel)
    (shop_repo.root / "shop").rmdir()
    (shop_repo.root / "shop").symlink_to(outside)

    out = diff_out(shop_repo)

    assert "leaked_secret" not in out and "TOP-SECRET-VALUE" not in out
    assert symbol_lines(out, "shop.billing.charge") == []  # not presented as a removal: the directory could not be read safely
    assert "shop/billing.py" in not_determinable(out) and "outside the scanned directory" in not_determinable(out)


def test_a_tracked_file_replaced_by_a_symlink_is_reported_and_is_not_a_removal(shop_repo: GitRepo, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.py"
    target.write_text("def charge(amount):\n    return 0\n", encoding="utf-8")
    shop_repo.delete("shop/billing.py")
    (shop_repo.root / "shop/billing.py").symlink_to(target)

    out = diff_out(shop_repo)

    assert symbol_lines(out, "shop.billing.charge") == [] and symbol_lines(out, "shop.billing.fee") == []
    assert "shop/billing.py is a symbolic link" in not_determinable(out)
    assert "elsewhere" not in out


def test_an_untracked_symlink_to_python_is_reported_not_silently_dropped(shop_repo: GitRepo, tmp_path: Path) -> None:
    target = tmp_path / "other.py"
    target.write_text("def other():\n    return 1\n", encoding="utf-8")
    (shop_repo.root / "shop/linked.py").symlink_to(target)

    out = diff_out(shop_repo)

    assert "shop/linked.py is a symbolic link" in not_determinable(out)
    assert "other" not in before_notes(out)


def test_a_symlink_committed_in_the_base_and_a_regular_file_in_the_head_is_not_a_wall_of_additions(git_repo: GitRepo, tmp_path: Path) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/real.py": "def real():\n    return 1\n"})
    git_repo.symlink("shop/alias.py", "real.py")
    git_repo.commit("base has a symlink")
    git_repo.delete("shop/alias.py")
    git_repo.write("shop/alias.py", "def alias_fn():\n    return 2\n")
    git_repo.commit("head has a regular file")

    out = diff_out(git_repo, "--base", "HEAD~1", "--head", "HEAD")

    assert symbol_lines(out, "shop.alias.alias_fn") == []  # the base side could not be read, so "added" would be a false claim
    assert "shop/alias.py is a symbolic link" in not_determinable(out)


def test_a_submodule_entry_named_like_python_is_reported_as_not_a_regular_file(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/real.py": "def real():\n    return 1\n"})
    git_repo.stage("shop/__init__.py", "shop/real.py")
    git_repo.add_gitlink("vendor/sub.py")
    git_repo.git("commit", "-q", "-m", "with a submodule entry")  # not `commit()`: `add -A` would drop the gitlink, which has no files on disk
    git_repo.write("shop/real.py", "def real():\n    return 2\n")

    assert "vendor/sub.py is not a regular file" in not_determinable(diff_out(git_repo))


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_an_unreadable_file_is_reported_instead_of_aborting_or_vanishing(shop_repo: GitRepo) -> None:
    if os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    shop_repo.write("shop/billing.py", BODY_EDIT)
    (shop_repo.root / "shop/billing.py").chmod(0)

    try:
        result = run_diff(shop_repo)
    finally:
        (shop_repo.root / "shop/billing.py").chmod(0o644)

    assert result.exit_code == 0, result.output
    assert symbol_lines(result.output, "shop.billing.charge") == []
    assert "shop/billing.py could not be read" in not_determinable(result.output)


def test_a_ref_that_looks_like_a_git_option_is_refused_with_exit_code_2(shop_repo: GitRepo) -> None:
    for args in (("--base", "--output=/tmp/x"), ("--head", "-p")):
        result = run_diff(shop_repo, *args)
        assert result.exit_code == 2 and "cannot start with '-'" in result.output
