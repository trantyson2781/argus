"""Diff Explorer: a base that is not an ancestor of the head is replaced by the merge-base (the point where the histories split)."""

from __future__ import annotations

from tests.explore.diff_helpers import GitRepo, diff_out, run_diff, symbol_lines

CORE = "def price(x):\n    return x * 5\n\n\ndef other():\n    return 1\n"


def _diverged(repo: GitRepo) -> str:
    """main and feature split at the returned commit. feature edits price(); main afterwards adds newfn(). Leaves feature checked out."""
    repo.write_files({"shop/__init__.py": "", "shop/core.py": CORE})
    split = repo.commit("base")
    repo.git("checkout", "-q", "-b", "feature")
    repo.write("shop/core.py", CORE.replace("x * 5", "x * 5 + 1"))
    repo.commit("feature edits price")
    repo.git("checkout", "-q", "main")
    repo.write("shop/core.py", CORE + "\n\ndef newfn():\n    return 2\n")
    repo.commit("main moves ahead")
    repo.git("checkout", "-q", "feature")
    return split


def _header(out: str) -> str:
    return out.splitlines()[0]


def test_work_done_only_on_the_base_branch_is_not_reported_as_removed_by_the_head_branch(git_repo: GitRepo) -> None:
    split = _diverged(git_repo)

    out = diff_out(git_repo, "--base", "main", "--head", "feature")

    assert symbol_lines(out, "shop.core.newfn") == []
    assert any("BODY_CHANGED" in line for line in symbol_lines(out, "shop.core.price"))
    assert "removed" not in out.split("Symbols", 1)[0]  # the summary has no "removed" count
    assert f"merge-base of main and feature ({split[:7]}) -> feature ({git_repo.short('feature')})" in _header(out)


def test_the_working_tree_is_compared_against_the_merge_base_of_the_base_and_head(git_repo: GitRepo) -> None:
    split = _diverged(git_repo)
    git_repo.write("shop/core.py", CORE.replace("x * 5", "x * 5 + 1").replace("return 1", "return 3"))  # uncommitted edit of other()

    out = diff_out(git_repo, "--base", "main")

    assert symbol_lines(out, "shop.core.newfn") == []
    assert any("BODY_CHANGED" in line for line in symbol_lines(out, "shop.core.price"))  # committed on the branch
    assert any("BODY_CHANGED" in line for line in symbol_lines(out, "shop.core.other"))  # still uncommitted
    assert f"merge-base of main and HEAD ({split[:7]})" in _header(out)


def test_a_base_that_is_already_an_ancestor_is_used_as_it_is_and_the_header_does_not_change(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/core.py": CORE})
    first = git_repo.commit("one")
    git_repo.write("shop/core.py", CORE.replace("x * 5", "x * 6"))
    second = git_repo.commit("two")

    for args in (("--base", first, "--head", second), ("--base", first), ()):
        header = _header(diff_out(git_repo, *args))
        assert "merge-base" not in header
        assert header.startswith(f"Diff  {first if args else 'HEAD'} ({first[:7] if args else second[:7]})")


def test_histories_with_no_common_ancestor_exit_with_code_2_and_say_why(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/core.py": CORE})
    git_repo.commit("base")
    git_repo.git("checkout", "-q", "--orphan", "island")
    git_repo.write("island.py", "def alone():\n    return 1\n")
    git_repo.commit("unrelated root")

    result = run_diff(git_repo, "--base", "main", "--head", "island")

    assert result.exit_code == 2
    assert "share no history" in result.output and "main" in result.output and "island" in result.output
