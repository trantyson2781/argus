"""Diff Explorer v0.1: what 'working tree' means, and that both sides are treated alike (contract: see diff_helpers.py)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.explore.diff_helpers import (
    BILLING, GitRepo, affected_groups, before_notes, diff_out, flags_in, not_determinable, section, symbol_lines,
)

CHARGE = "shop.billing.charge"
BODY_EDIT = BILLING.replace("amount + 1", "amount + 2")


def test_staged_only_changes_are_included(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BODY_EDIT)
    shop_repo.stage("shop/billing.py")

    assert flags_in(symbol_lines(diff_out(shop_repo), CHARGE)) == {"BODY_CHANGED"}


def test_unstaged_only_changes_are_included(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BODY_EDIT)

    assert flags_in(symbol_lines(diff_out(shop_repo), CHARGE)) == {"BODY_CHANGED"}


def test_the_disk_wins_over_the_index_so_a_staged_edit_reverted_on_disk_is_no_change(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BODY_EDIT)
    shop_repo.stage("shop/billing.py")
    shop_repo.write("shop/billing.py", BILLING)  # back to HEAD content, still staged as the edit

    out = diff_out(shop_repo)

    assert "no structural change" in out and section(out, "Symbols") == ""


def test_untracked_python_files_are_included_marked_and_counted_in_the_header(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BODY_EDIT)
    shop_repo.write("shop/audit.py", "def record(event):\n    return event\n")

    out = diff_out(shop_repo)

    (line,) = symbol_lines(out, "shop.audit.record")
    assert "[FACT]" in line and "+ function" in line and "shop/audit.py:1" in line and "(untracked file)" in line
    assert "(untracked file)" not in symbol_lines(out, CHARGE)[0]
    assert re.search(r"2 files changed \(1 untracked\)", out.splitlines()[0])


def test_git_ignored_python_files_are_not_scanned(shop_repo: GitRepo) -> None:
    shop_repo.write(".gitignore", "generated/\nscratch_*.py\n")
    shop_repo.commit("ignore rules")
    shop_repo.write("generated/gen.py", "def gen():\n    return 1\n")
    shop_repo.write("shop/scratch_notes.py", "def notes():\n    return 1\n")

    out = diff_out(shop_repo)

    assert "no structural change" in out
    assert "shop.scratch_notes" not in out and "generated" not in out and "gen.py" not in out


def test_a_tracked_file_deleted_on_disk_counts_as_deleted(shop_repo: GitRepo) -> None:
    shop_repo.delete("shop/api.py")

    (line,) = symbol_lines(diff_out(shop_repo), "shop.api.view")

    assert "[FACT]" in line and "- function" in line and "was shop/api.py:4" in line


@pytest.mark.parametrize("how", ["git-mv", "plain-move"])
def test_a_renamed_file_lists_every_symbol_removed_and_added_with_all_and_one_rename_row_by_default(shop_repo: GitRepo, how: str) -> None:
    if how == "git-mv":
        shop_repo.git("mv", "shop/billing.py", "shop/pricing.py")
    else:
        (shop_repo.root / "shop/billing.py").rename(shop_repo.root / "shop/pricing.py")

    default = diff_out(shop_repo)
    out = diff_out(shop_repo, "--all")

    assert "[FACT] > shop/billing.py -> shop/pricing.py" in default and symbol_lines(default, "shop.billing.charge") == []
    for old in ("charge", "fee"):
        (removed,) = symbol_lines(out, f"shop.billing.{old}")
        assert "- function" in removed and "was shop/billing.py" in removed
        (added,) = symbol_lines(out, f"shop.pricing.{old}")
        assert "+ function" in added and "shop/pricing.py" in added
        assert ("(untracked file)" in added) is (how == "plain-move")


# -- both sides are treated the same ----------------------------------------------------------------------------------


def test_a_clean_tree_prints_no_structural_change_and_still_lists_what_is_not_determinable(shop_repo: GitRepo) -> None:
    out = diff_out(shop_repo)

    assert "no structural change" in out
    assert section(out, "Symbols") == "" and section(out, "Calls") == ""
    assert not_determinable(out).strip() != ""


def test_header_names_both_sides_the_file_counts_and_the_resolver_mode(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BODY_EDIT)

    header = diff_out(shop_repo).splitlines()[0]

    assert header.startswith("Diff") and f"HEAD ({shop_repo.short()})" in header and "-> working tree" in header
    assert re.search(r"1 files? changed", header) and "calls-v2 on both sides" in header


SELF_CART = (
    "class Cart:\n    def total(self, amount):\n        return self.price(amount)\n\n"
    "    def price(self, amount):\n        return amount\n"
)


@pytest.mark.parametrize(("mode", "label", "has_self_edge"), [("1", "calls-v2", True), ("0", "calls-v1", False)])
def test_one_resolver_mode_reaches_both_scans_so_it_never_invents_call_changes(git_repo: GitRepo, mode: str, label: str, has_self_edge: bool) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/cart.py": SELF_CART})
    git_repo.commit("base")
    git_repo.write("shop/cart.py", SELF_CART.replace("self.price(amount)", "self.price(amount) + 0").replace("return amount\n", "return amount + 1\n"))

    out = diff_out(git_repo, env={"ARGUS_RESOLVE_V2": mode})

    assert f"{label} on both sides" in out.splitlines()[0]
    assert "CALLS_CHANGED" not in before_notes(out) and section(out, "Calls") == ""  # a mixed-mode pair would show a phantom call change
    assert flags_in(symbol_lines(out, "shop.cart.Cart.total")) == {"BODY_CHANGED"}
    groups = affected_groups(out)
    assert bool(groups) is has_self_edge
    if has_self_edge:
        assert "via self; a subclass may override it" in groups[0].edges[0].rest


def test_temp_directory_paths_never_appear_in_the_output_and_are_removed_afterwards(shop_repo: GitRepo, scratch_tmp: Path) -> None:
    shop_repo.write("shop/billing.py", BODY_EDIT)

    out = diff_out(shop_repo)

    assert str(scratch_tmp) not in out
    assert list(scratch_tmp.iterdir()) == []
