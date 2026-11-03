"""Diff Explorer: an empty result must not hide module-level edits, and a pure file rename is one rename, not a removal plus an addition."""

from __future__ import annotations

from tests.explore.diff_helpers import GitRepo, diff_out, section, symbol_lines

FILES_TITLE = "Files changed outside any function or class"
RENAMED_TITLE = "Renamed files"
CORE = "RATE = 5\n\n\ndef price(x):\n    return x * RATE\n\n\ndef other():\n    return 1\n"
BASE = {"shop/__init__.py": "", "shop/core.py": CORE, "shop/api.py": "from shop.core import price\n\n\ndef view(x):\n    return price(x)\n"}


def _base(repo: GitRepo, files: dict[str, str] | None = None) -> None:
    repo.write_files(files or BASE)
    repo.commit("base")


def _summary(out: str) -> str:
    return next((line for line in out.splitlines() if line.startswith("Summary")), "")


# -- module-level edits are listed ----------------------------------------------------------------------------------------------------


def test_a_constant_changed_at_module_level_is_listed_and_the_report_does_not_claim_nothing_changed(git_repo: GitRepo) -> None:
    _base(git_repo)
    git_repo.write("shop/core.py", CORE.replace("RATE = 5", "RATE = 7"))

    out = diff_out(git_repo)

    assert "[FACT] ~ shop/core.py" in section(out, FILES_TITLE) and "module-level" in section(out, FILES_TITLE)
    assert "no structural change" not in out
    assert "1 file changed outside functions" in _summary(out)


def test_a_comment_only_edit_is_listed_as_formatting_and_the_report_still_says_no_structural_change(git_repo: GitRepo) -> None:
    _base(git_repo)
    git_repo.write("shop/core.py", "# a new comment\n" + CORE.replace("return 1", "return 1  # trailing"))

    out = diff_out(git_repo)

    assert "comments or formatting only" in section(out, FILES_TITLE)
    assert "no structural change" in out
    assert _summary(out) == ""  # nothing structural to count


def test_a_class_attribute_change_counts_as_code_outside_functions(git_repo: GitRepo) -> None:
    _base(git_repo, {"shop/__init__.py": "", "shop/models.py": "class Item:\n    limit = 5\n\n    def name(self):\n        return 'x'\n"})
    git_repo.write("shop/models.py", "class Item:\n    limit = 9\n\n    def name(self):\n        return 'x'\n")

    assert "module-level or class-level" in section(diff_out(git_repo), FILES_TITLE)


def test_a_new_or_removed_file_without_functions_or_classes_is_listed(git_repo: GitRepo) -> None:
    _base(git_repo, {"shop/__init__.py": "", "shop/old.py": "OLD = 1\n"})
    git_repo.delete("shop/old.py")
    git_repo.write("shop/settings.py", "DEBUG = True\n")

    body = section(diff_out(git_repo), FILES_TITLE)

    assert "[FACT] + shop/settings.py" in body and "new file with no functions or classes" in body
    assert "[FACT] - shop/old.py" in body and "removed" in body


def test_a_file_that_also_has_a_symbol_change_is_not_listed_because_it_is_already_reported(git_repo: GitRepo) -> None:
    _base(git_repo)
    git_repo.write("shop/core.py", CORE.replace("RATE = 5", "RATE = 7").replace("return 1", "return 2"))

    out = diff_out(git_repo)

    assert section(out, FILES_TITLE) == ""
    assert any("BODY_CHANGED" in line for line in symbol_lines(out, "shop.core.other"))
    assert "Module-level code and class-level code" in section(out, "Not determinable")  # the standing limit still covers this file


def test_files_that_do_not_parse_or_define_a_name_twice_are_not_called_module_level(git_repo: GitRepo) -> None:
    dup = "if True:\n    def f():\n        return 1\nelse:\n    def f():\n        return 2\n"
    _base(git_repo, {"shop/__init__.py": "", "shop/dup.py": dup, "shop/bad.py": "x = 1\n"})
    git_repo.write("shop/dup.py", dup.replace("return 2", "return 3"))
    git_repo.write("shop/bad.py", "def (\n")

    out = diff_out(git_repo)

    assert "shop/dup.py" not in section(out, FILES_TITLE) and "shop/bad.py" not in section(out, FILES_TITLE)
    assert "shop/bad.py" in section(out, "Not determinable")


def test_the_module_level_rows_are_sorted_by_file(git_repo: GitRepo) -> None:
    _base(git_repo, {"shop/__init__.py": "", "shop/b.py": "B = 1\n", "shop/a.py": "A = 1\n"})
    git_repo.write("shop/b.py", "B = 2\n")
    git_repo.write("shop/a.py", "A = 2\n")

    rows = [line.split()[2] for line in section(diff_out(git_repo), FILES_TITLE).splitlines() if "[FACT]" in line]

    assert rows == ["shop/a.py", "shop/b.py"]


# -- pure file renames -----------------------------------------------------------------------------------------------------------------


def _rename(repo: GitRepo, edit: str | None = None) -> None:
    _base(repo)
    repo.delete("shop/core.py")
    repo.write("shop/pricing.py", CORE if edit is None else edit)


def test_a_pure_file_rename_is_one_row_and_its_symbols_are_not_counted_as_removed_and_added(git_repo: GitRepo) -> None:
    _rename(git_repo)

    out = diff_out(git_repo)

    assert "[FACT] > shop/core.py -> shop/pricing.py" in section(out, RENAMED_TITLE)
    assert "2 symbols" in section(out, RENAMED_TITLE) or "2 functions" in section(out, RENAMED_TITLE)
    assert "1 file renamed" in _summary(out)
    assert "removed" not in _summary(out).split("|")[0] and "added" not in _summary(out).split("|")[0]
    assert symbol_lines(out, "shop.pricing.price") == [] and symbol_lines(out, "shop.core.price") == []
    assert "hidden" in section(out, "Symbols") or section(out, "Symbols") == ""


def test_all_lists_the_symbols_of_a_renamed_file_too(git_repo: GitRepo) -> None:
    _rename(git_repo)

    out = diff_out(git_repo, "--all")

    assert symbol_lines(out, "shop.core.price") and symbol_lines(out, "shop.pricing.price")
    assert "[FACT] > shop/core.py -> shop/pricing.py" in section(out, RENAMED_TITLE)


def test_callers_of_the_old_names_are_still_reported_after_a_rename(git_repo: GitRepo) -> None:
    _rename(git_repo)

    out = diff_out(git_repo)

    assert "shop.api.view" in section(out, "Direct callers of changed interfaces")  # it still imports the old module path


def test_a_rename_with_edits_is_not_a_rename(git_repo: GitRepo) -> None:
    _rename(git_repo, CORE.replace("RATE = 5", "RATE = 6"))

    out = diff_out(git_repo)

    assert section(out, RENAMED_TITLE) == ""
    assert symbol_lines(out, "shop.core.price") and symbol_lines(out, "shop.pricing.price")


def test_identical_files_that_cannot_be_paired_one_to_one_are_not_called_renames(git_repo: GitRepo) -> None:
    _base(git_repo, {"shop/__init__.py": "", "shop/a.py": CORE, "shop/b.py": CORE})
    git_repo.delete("shop/a.py")
    git_repo.delete("shop/b.py")
    git_repo.write("shop/c.py", CORE)

    assert section(diff_out(git_repo), RENAMED_TITLE) == ""


def test_a_rename_with_edits_keeps_the_standing_caveat_that_only_pure_renames_are_detected(git_repo: GitRepo) -> None:
    _rename(git_repo, CORE.replace("RATE = 5", "RATE = 6"))

    assert "shows as a removal plus an addition" in section(diff_out(git_repo), "Not determinable")


def test_new_or_removed_empty_files_are_listed_in_one_row_each_so_the_header_count_adds_up(git_repo: GitRepo) -> None:
    _base(git_repo, {"shop/__init__.py": "", "shop/gone/__init__.py": ""})
    git_repo.delete("shop/gone/__init__.py")
    git_repo.write("shop/fresh/__init__.py", "")

    body = section(diff_out(git_repo), FILES_TITLE)

    assert "[FACT] + 1 new empty file: shop/fresh/__init__.py" in body
    assert "[FACT] - 1 removed empty file: shop/gone/__init__.py" in body
