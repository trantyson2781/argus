"""Diff Explorer: the one-line summary and the collapsing of big new files into a single row."""

from __future__ import annotations

from tests.explore.diff_helpers import SHOP_FILES, GitRepo, diff_out, section

BIG = 60


def _functions(count: int, prefix: str = "f") -> str:
    return "".join(f"def {prefix}{i:03d}():\n    return {i}\n\n\n" for i in range(count))


def _summary(out: str) -> str:
    return next((line for line in out.splitlines() if line.startswith("Summary")), "")


def _symbol_rows(out: str) -> list[str]:
    return [line for line in section(out, "Symbols").splitlines() if line.strip().startswith("[FACT]")]


# -- summary ---------------------------------------------------------------------------------------------------------------------


def test_the_summary_counts_each_kind_of_change_and_sits_right_under_the_legend(git_repo: GitRepo) -> None:
    git_repo.write_files(SHOP_FILES)
    git_repo.commit("base")
    git_repo.write("shop/billing.py", "def charge(amount, currency):\n    return amount + 2\n")  # fee removed, charge: interface + body
    git_repo.write("shop/cart.py", git_repo.root.joinpath("shop/cart.py").read_text() + "\n\ndef extra():\n    return 1\n")  # 1 added

    out = diff_out(git_repo)

    assert out.splitlines()[2] == _summary(out)
    summary = _summary(out)
    for part in ("1 removed", "1 interface changed", "1 body changed", "1 added"):
        assert part in summary
    assert "caller" in summary  # callers to review are counted too


def test_the_summary_leaves_out_kinds_that_did_not_happen(git_repo: GitRepo) -> None:
    git_repo.write_files(SHOP_FILES)
    git_repo.commit("base")
    git_repo.write("shop/billing.py", "def charge(amount):\n    return amount + 5\n\n\ndef fee(amount):\n    return amount\n")

    summary = _summary(diff_out(git_repo))

    assert "1 body changed" in summary
    assert "removed" not in summary and "interface" not in summary and "added" not in summary


def test_a_clean_tree_has_no_summary_line(git_repo: GitRepo) -> None:
    git_repo.write_files(SHOP_FILES)
    git_repo.commit("base")

    assert _summary(diff_out(git_repo)) == ""


def test_the_summary_says_how_many_added_symbols_sit_in_new_files(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": ""})
    git_repo.commit("base")
    git_repo.write("shop/wide.py", _functions(BIG))

    assert f"{BIG} added (all {BIG} in 1 new file)" in _summary(diff_out(git_repo))


# -- big new files collapse -------------------------------------------------------------------------------------------------------


def test_a_big_new_file_is_one_row_and_the_title_keeps_the_true_symbol_count(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": ""})
    git_repo.commit("base")
    git_repo.write("shop/wide.py", _functions(BIG))

    out = diff_out(git_repo)

    rows = _symbol_rows(out)
    assert len(rows) == 1
    assert "+ new file" in rows[0] and "shop/wide.py:1" in rows[0] and f"{BIG} functions" in rows[0] and "(untracked file)" in rows[0]
    assert f"Symbols ({BIG})" in out
    assert "more symbols not shown" not in out
    assert "--all" in section(out, "Symbols")  # the row says how to see them


def test_all_expands_a_new_file_into_its_symbols(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": ""})
    git_repo.commit("base")
    git_repo.write("shop/wide.py", _functions(BIG))

    rows = _symbol_rows(diff_out(git_repo, "--all"))

    assert len(rows) == BIG and not any("new file" in r for r in rows)


def test_a_small_new_file_is_still_listed_symbol_by_symbol(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": ""})
    git_repo.commit("base")
    git_repo.write("shop/tiny.py", _functions(3))

    rows = _symbol_rows(diff_out(git_repo))

    assert len(rows) == 3 and not any("new file" in r for r in rows)


def test_many_new_symbols_in_a_file_that_already_existed_are_not_collapsed(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/wide.py": ""})
    git_repo.commit("base")
    git_repo.write("shop/wide.py", _functions(BIG))

    out = diff_out(git_repo)

    assert not any("new file" in r for r in _symbol_rows(out))
    assert "all" not in _summary(out).split("added", 1)[1]  # no "(all N in K new files)" for an old file


def test_a_collapsed_row_counts_as_one_row_against_the_cap_and_the_omitted_count_stays_exact(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/old.py": ""})
    git_repo.commit("base")
    git_repo.write("shop/a_wide.py", _functions(BIG))  # sorts first; collapses to 1 row (60 symbols)
    git_repo.write("shop/old.py", _functions(55, "g"))  # 55 symbols in an existing file

    out = diff_out(git_repo)

    assert len(_symbol_rows(out)) == 50  # 1 collapsed row + 49 symbols
    assert f"Symbols ({BIG + 55})" in out
    assert "+6 more symbols not shown (6 added, 0 removed, 0 modified)" in out


def test_a_new_file_in_a_commit_range_collapses_too(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": ""})
    first = git_repo.commit("base")
    git_repo.write("shop/wide.py", _functions(BIG))
    second = git_repo.commit("wide")

    rows = _symbol_rows(diff_out(git_repo, "--base", first, "--head", second))

    assert len(rows) == 1 and "+ new file" in rows[0] and "(untracked file)" not in rows[0]


def test_the_new_file_row_uses_the_singular_for_a_count_of_one(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": ""})
    git_repo.commit("base")
    git_repo.write("shop/mixed.py", "class Only:\n    pass\n\n\n" + _functions(6))

    row = _symbol_rows(diff_out(git_repo))[0]

    assert "1 class," in row and "6 functions" in row and "1 classes" not in row
