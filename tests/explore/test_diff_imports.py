"""Diff Explorer: a subsystem import is reported (as a FACT) only when an import statement was added or deleted.

An import edge also appears or disappears when nothing was edited in the importing file: the module it names was added, removed or
moved, or the import became type-only. Those are resolution changes. They are not reported as imports, and are counted instead.
"""

from __future__ import annotations

from tests.explore.diff_helpers import GitRepo
from tests.explore.test_diff_view import build_view, unknown_text

PKG = {"shop/__init__.py": "", "shop/cart/__init__.py": "", "shop/audit/__init__.py": ""}
AUDIT_LOG = "def record():\n    return 1\n"


def _rows(view) -> list[tuple[str, str, str]]:
    return [(i.change, i.source, i.target) for i in view.imports]


def test_a_statement_that_already_existed_unresolved_is_not_reported_as_added_when_its_target_appears(git_repo: GitRepo) -> None:
    git_repo.write_files({**PKG, "shop/cart/a.py": "from shop.later.log import record\n\nX = 1\n"})  # shop.later does not exist yet
    git_repo.commit("base: the import is already written, and points at nothing")
    git_repo.write_files({"shop/later/__init__.py": "", "shop/later/log.py": AUDIT_LOG})

    view = build_view(git_repo)

    assert view.imports == ()
    assert "no import statement was edited" in unknown_text(view)


def test_a_statement_kept_while_its_target_module_is_deleted_is_not_reported_as_removed(git_repo: GitRepo) -> None:
    git_repo.write_files({**PKG, "shop/cart/a.py": "from shop.audit.log import record\n", "shop/audit/log.py": AUDIT_LOG})
    git_repo.commit("base")
    git_repo.delete("shop/audit/log.py")

    view = build_view(git_repo)

    assert view.imports == ()
    assert "no import statement was edited" in unknown_text(view)


def test_a_runtime_import_that_becomes_type_only_is_not_reported_as_removed(git_repo: GitRepo) -> None:
    git_repo.write_files({**PKG, "shop/cart/a.py": "from shop.audit.log import record\n", "shop/audit/log.py": AUDIT_LOG})
    git_repo.commit("base")
    git_repo.write("shop/cart/a.py", "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from shop.audit.log import record\n")

    view = build_view(git_repo)

    assert view.imports == ()


def test_a_deleted_import_statement_is_a_removed_subsystem_import_with_the_base_location(git_repo: GitRepo) -> None:
    git_repo.write_files({**PKG, "shop/cart/a.py": "X = 1\nfrom shop.audit.log import record\n", "shop/audit/log.py": AUDIT_LOG})
    git_repo.commit("base")
    git_repo.write("shop/cart/a.py", "X = 1\n")

    view = build_view(git_repo)

    assert _rows(view) == [("removed", "shop.cart", "shop.audit")]
    claim = view.imports[0].claim
    assert claim.tier == "FACT" and claim.evidence[0].location == "shop/cart/a.py:2"


def test_an_import_statement_and_its_new_target_added_together_is_a_fact_with_head_evidence(git_repo: GitRepo) -> None:
    git_repo.write_files({**PKG, "shop/cart/a.py": "X = 1\n"})
    git_repo.commit("base")
    git_repo.write_files({"shop/audit/log.py": AUDIT_LOG, "shop/cart/a.py": "X = 1\nfrom shop.audit.log import record\n"})

    view = build_view(git_repo)

    assert _rows(view) == [("added", "shop.cart", "shop.audit")]
    assert view.imports[0].claim.tier == "FACT" and view.imports[0].claim.evidence[0].location == "shop/cart/a.py:2"


def test_an_import_moved_to_another_line_or_another_file_of_the_same_subsystem_is_no_change(git_repo: GitRepo) -> None:
    git_repo.write_files({**PKG, "shop/cart/a.py": "from shop.audit.log import record\n", "shop/cart/b.py": "Y = 1\n", "shop/audit/log.py": AUDIT_LOG})
    git_repo.commit("base")
    git_repo.write("shop/cart/a.py", "# moved\n\nY = 0\n")
    git_repo.write("shop/cart/b.py", "\n\nfrom shop.audit.log import record\n")

    assert build_view(git_repo).imports == ()


def test_an_import_inside_a_function_counts_like_any_other_import_statement(git_repo: GitRepo) -> None:
    git_repo.write_files({**PKG, "shop/cart/a.py": "def f():\n    return 1\n", "shop/audit/log.py": AUDIT_LOG})
    git_repo.commit("base")
    git_repo.write("shop/cart/a.py", "def f():\n    from shop.audit.log import record\n    return record()\n")

    assert _rows(build_view(git_repo)) == [("added", "shop.cart", "shop.audit")]


def test_an_import_whose_target_moves_to_another_subsystem_with_the_statement_updated_is_one_removal_and_one_addition(git_repo: GitRepo) -> None:
    git_repo.write_files({**PKG, "shop/util/__init__.py": "", "shop/cart/a.py": "from shop.audit.log import record\n", "shop/audit/log.py": AUDIT_LOG})
    git_repo.commit("base")
    git_repo.delete("shop/audit/log.py")
    git_repo.write("shop/util/log.py", AUDIT_LOG)
    git_repo.write("shop/cart/a.py", "from shop.util.log import record\n")

    assert sorted(_rows(build_view(git_repo))) == [("added", "shop.cart", "shop.util"), ("removed", "shop.cart", "shop.audit")]
