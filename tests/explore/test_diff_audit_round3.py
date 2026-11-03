"""Diff Explorer: regressions found by the 2026-10-17 audit panel (security, silent failures, correctness, tests)."""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.explore import sides
from tests.explore.diff_helpers import GitRepo, diff_out, not_determinable, run_diff, section, symbol_lines

FILES_TITLE = "Files changed outside any function or class"
THREE = "def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\ndef c():\n    return 3\n"


# -- git environment -------------------------------------------------------------------------------------------------------------------


def test_every_git_transport_is_refused_by_environment_because_a_repository_config_can_re_allow_one() -> None:
    assert sides._git_env()["GIT_ALLOW_PROTOCOL"] == "none"  # beats protocol.file.allow=always set in a hostile .git/config


def test_a_path_that_climbs_out_of_the_temp_tree_is_refused(tmp_path: Path) -> None:
    for rel in ("../outside.py", "/etc/passwd.py", "a/../../b.py"):
        with pytest.raises(sides.DiffError):
            sides._safe_target(tmp_path, rel)


def test_too_many_python_files_refuse_the_comparison_instead_of_filling_the_disk(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    git_repo.write_files({f"pkg/m{i}.py": "x = 1\n" for i in range(5)})
    git_repo.commit("base")
    monkeypatch.setattr(sides, "MAX_FILES", 3)

    result = run_diff(git_repo, "--base", "HEAD", "--head", "HEAD")

    assert result.exit_code == 2 and "more than 3" in result.output


# -- a base that is ahead of the head --------------------------------------------------------------------------------------------------


def test_a_base_that_is_ahead_of_an_explicit_head_is_an_error_not_an_empty_report(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/core.py": THREE})
    first = git_repo.commit("one")
    git_repo.write("shop/core.py", THREE.replace("return 1", "return 9"))
    second = git_repo.commit("two")

    result = run_diff(git_repo, "--base", second, "--head", first)

    assert result.exit_code == 2 and "ahead of" in result.output and "Swap" in result.output


def test_a_branch_that_is_behind_the_base_still_shows_the_work_in_the_working_tree(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/core.py": THREE})
    git_repo.commit("base")
    git_repo.git("checkout", "-q", "-b", "feature")
    git_repo.git("checkout", "-q", "main")
    git_repo.write("shop/core.py", THREE + "\n\ndef newfn():\n    return 4\n")
    git_repo.commit("main moves ahead")
    git_repo.git("checkout", "-q", "feature")
    git_repo.write("shop/core.py", THREE.replace("return 2", "return 7"))  # uncommitted work on the branch that is behind main

    out = diff_out(git_repo, "--base", "main")

    assert any("BODY_CHANGED" in line for line in symbol_lines(out, "shop.core.b"))
    assert symbol_lines(out, "shop.core.newfn") == []


# -- callers that cannot be compared are reported ---------------------------------------------------------------------------------------


DUP_CALLER = (
    "import sys\nfrom shop.util import helper\n\n\nif sys.platform:\n    def run():\n        return helper()\nelse:\n    def run():\n        return helper() + 1\n"
)


def test_a_caller_defined_twice_in_an_unchanged_file_is_reported_when_its_callee_is_removed(git_repo: GitRepo) -> None:
    git_repo.write_files({
        "shop/__init__.py": "", "shop/util.py": "def helper():\n    return 1\n\n\ndef other():\n    return 2\n", "shop/plat.py": DUP_CALLER,
    })
    git_repo.commit("base")
    git_repo.write("shop/util.py", "def other():\n    return 2\n")

    text = not_determinable(diff_out(git_repo))

    assert "shop/plat.py" in text and "missing from the caller sections" in text


def test_a_caller_in_a_file_that_no_longer_parses_is_left_out_and_the_file_is_named(git_repo: GitRepo) -> None:
    git_repo.write_files({
        "shop/__init__.py": "", "shop/util.py": "def helper():\n    return 1\n\n\ndef other():\n    return 2\n",
        "shop/caller.py": "from shop.util import helper\n\n\ndef go():\n    return helper()\n",
    })
    git_repo.commit("base")
    git_repo.write("shop/util.py", "def other():\n    return 2\n")
    git_repo.write("shop/caller.py", "def go(:\n")

    out = diff_out(git_repo)

    assert "shop.caller.go" not in section(out, "Direct callers of changed interfaces")
    assert "shop/caller.py" in not_determinable(out) and "callers located in it are missing" in not_determinable(out)


# -- renames and moves ------------------------------------------------------------------------------------------------------------------


def _module_to_package(repo: GitRepo, body_after: str) -> None:
    repo.write_files({"pkg/__init__.py": "", "pkg/mod.py": THREE})
    repo.commit("base")
    repo.delete("pkg/mod.py")
    repo.write("pkg/mod/__init__.py", body_after)


def test_a_module_turned_into_a_package_counts_its_real_symbols_and_does_not_claim_the_names_change(git_repo: GitRepo) -> None:
    _module_to_package(git_repo, THREE)

    out = diff_out(git_repo)

    renamed = section(out, "Renamed files")
    assert "pkg/mod.py -> pkg/mod/__init__.py" in renamed and "3 symbols" in renamed and "no functions" not in renamed
    assert "qualified names" not in out.split("Symbols", 1)[0]  # the title promises nothing about names


def test_a_module_turned_into_a_package_with_edits_keeps_every_body_change_visible(git_repo: GitRepo) -> None:
    many = "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(8))
    git_repo.write_files({"pkg/__init__.py": "", "pkg/mod.py": many})
    git_repo.commit("base")
    git_repo.delete("pkg/mod.py")
    git_repo.write("pkg/mod/__init__.py", many.replace("return 1\n", "return 100\n").replace("return 2\n", "return 200\n"))

    out = diff_out(git_repo)

    assert any("BODY_CHANGED" in line for line in symbol_lines(out, "pkg.mod.f1"))
    assert any("BODY_CHANGED" in line for line in symbol_lines(out, "pkg.mod.f2"))
    assert "new file" not in section(out, "Symbols")  # modified symbols are not "added", so nothing collapses
    files = section(out, FILES_TITLE)
    assert "pkg/mod.py" in files and "no functions or classes" not in files


def test_two_unrelated_empty_files_are_not_a_rename(git_repo: GitRepo) -> None:
    git_repo.write_files({"pkg/__init__.py": "", "pkg/e1.py": ""})
    git_repo.commit("base")
    git_repo.delete("pkg/e1.py")
    git_repo.write("pkg/zzz.py", "")

    out = diff_out(git_repo)

    assert section(out, "Renamed files") == ""
    files = section(out, FILES_TITLE)
    assert "1 new empty file" in files and "pkg/zzz.py" in files and "1 removed empty file" in files and "pkg/e1.py" in files


def test_new_empty_files_are_one_row_that_names_them(git_repo: GitRepo) -> None:
    git_repo.write_files({"pkg/__init__.py": ""})
    git_repo.commit("base")
    for i in range(5):
        git_repo.write(f"pkg/sub{i}/__init__.py", "")

    rows = [line for line in section(diff_out(git_repo), FILES_TITLE).splitlines() if "[FACT]" in line]

    assert len(rows) == 1 and "5 new empty files" in rows[0] and "pkg/sub0/__init__.py" in rows[0] and "more" in rows[0]
