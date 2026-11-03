"""Diff Explorer: files that cannot be analysed are never presented as additions, removals or "unchanged".

Three different situations are kept apart: a file that failed (does not parse, not UTF-8, cannot be read safely) is listed under
Not determinable; a directory the scanner skips on purpose (build/, venvs, hidden directories) is listed as not scanned, which is
a policy and not a failure; everything else is compared.
"""

from __future__ import annotations

import pytest

from tests.explore.diff_helpers import GitRepo, diff_out, flags_in, not_determinable, section, symbol_lines

LATIN_OLD = b'# -*- coding: latin-1 -*-\n\n\ndef greet():\n    return "caf\xe9"\n'
LATIN_NEW = b'# -*- coding: latin-1 -*-\n\n\ndef greet():\n    return "caf\xe8"\n'
UTF8_OLD = "def other():\n    return 1\n"


def test_a_real_edit_inside_a_non_utf8_file_is_not_reported_as_unchanged(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/other.py": UTF8_OLD})
    git_repo.write("shop/greet.py", LATIN_OLD)
    git_repo.commit("base")
    git_repo.write("shop/greet.py", LATIN_NEW)  # decoded with replacement characters, both versions look identical
    git_repo.write("shop/other.py", "def other():\n    return 2\n")

    out = diff_out(git_repo)

    assert "shop/greet.py is not valid UTF-8" in not_determinable(out)
    assert symbol_lines(out, "shop.greet.greet") == []
    assert flags_in(symbol_lines(out, "shop.other.other")) == {"BODY_CHANGED"}  # healthy files are still compared


def test_a_non_utf8_file_added_or_removed_does_not_add_or_remove_symbols(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/gone.py": LATIN_OLD})
    git_repo.commit("base")
    git_repo.delete("shop/gone.py")
    git_repo.write("shop/new.py", LATIN_NEW)

    out = diff_out(git_repo)

    assert section(out, "Symbols") == ""
    assert "shop/gone.py is not valid UTF-8" in not_determinable(out) and "shop/new.py is not valid UTF-8" in not_determinable(out)


def test_a_file_with_a_null_byte_is_reported_as_not_parsing_and_produces_no_symbols(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/ok.py": UTF8_OLD})
    git_repo.write("shop/nul.py", b"def f():\n    return 1\x00\n")
    git_repo.commit("base")
    git_repo.write("shop/nul.py", b"def f():\n    return 2\x00\n")

    out = diff_out(git_repo)

    assert "shop/nul.py does not parse" in not_determinable(out) and symbol_lines(out, "shop.nul.f") == []


def test_a_file_that_gets_a_byte_order_mark_is_reported_not_treated_as_removed_and_added(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/mod.py": UTF8_OLD})
    git_repo.commit("base")
    git_repo.write("shop/mod.py", b"\xef\xbb\xbf" + UTF8_OLD.encode())

    out = diff_out(git_repo)

    assert "shop/mod.py does not parse" in not_determinable(out) and symbol_lines(out, "shop.mod.other") == []


@pytest.mark.parametrize("directory", ["build", ".hidden", "venv", "node_modules", "pkg.egg-info"])
def test_changes_in_directories_the_scanner_skips_on_purpose_are_listed_as_not_scanned_and_not_as_failures(git_repo: GitRepo, directory: str) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/core.py": UTF8_OLD, f"{directory}/gen.py": "def gen():\n    return 1\n"})
    git_repo.commit("base")
    git_repo.write("shop/core.py", "def other():\n    return 2\n")
    git_repo.write(f"{directory}/gen.py", "def gen():\n    return 2\n")
    git_repo.write(f"{directory}/extra.py", "def extra():\n    return 1\n")

    out = diff_out(git_repo)

    assert flags_in(symbol_lines(out, "shop.core.other")) == {"BODY_CHANGED"}
    not_scanned = section(out, "Not scanned")
    assert f"{directory}/" in not_scanned and "2 changed .py files" in not_scanned
    assert f"{directory}/" not in not_determinable(out)  # an intentional exclusion is not an analysis failure
    assert out.splitlines()[0].count("1 file changed") == 1  # the header counts analysed files only


def test_a_clean_tree_with_only_excluded_changes_says_no_structural_change_and_still_lists_what_was_not_scanned(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "build/gen.py": "def gen():\n    return 1\n"})
    git_repo.commit("base")
    git_repo.write("build/gen.py", "def gen():\n    return 2\n")

    out = diff_out(git_repo)

    assert "no structural change" in out and "build/" in section(out, "Not scanned")


def test_a_file_the_scanner_silently_drops_is_reported_as_an_analysis_failure(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/thing.py": "def thing():\n    return 1\n"})
    git_repo.commit("base")
    git_repo.write("shop/thing/__init__.py", "def thing_pkg():\n    return 1\n")  # same module name as shop/thing.py: one of the two is dropped by scan()

    out = diff_out(git_repo)

    assert "not analysed" in not_determinable(out)
    assert "shop/thing" in not_determinable(out)


def test_the_view_is_complete_and_holds_the_unanalysed_files_even_when_the_display_hides_them(git_repo: GitRepo) -> None:
    from tests.explore.test_diff_view import build_view

    git_repo.write_files({"shop/__init__.py": "", "shop/greet.py": LATIN_OLD})
    git_repo.commit("base")
    git_repo.write("shop/greet.py", LATIN_NEW)
    git_repo.write("build/gen.py", "def gen():\n    return 1\n")

    view = build_view(git_repo)

    assert any("shop/greet.py is not valid UTF-8" in c.text for c in view.not_determinable)
    assert [(e.prefix, e.files) for e in view.not_scanned] == [("build/", 1)]
