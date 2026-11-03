"""Diff Explorer: a removed call is claimed only when its call-site text is gone or now resolves elsewhere.

The graph has no edge for an unresolved call, so "the edge disappeared" is ambiguous: the code may have been edited, or the call may
simply have stopped resolving (a shadowing parameter, a deleted import, a deleted target). Only the first is a removal.
"""

from __future__ import annotations

from tests.explore.diff_helpers import GitRepo
from tests.explore.test_diff_view import build_view, flags

BILLING = "def charge(a):\n    return a\n\n\ndef fee(a):\n    return a\n"
BASE = {"shop/__init__.py": "", "shop/billing.py": BILLING}


def _rows(view) -> list[tuple[str, str, str]]:
    return sorted((c.change, c.source, c.target) for c in view.calls)


def _repo(repo: GitRepo, cart: str, extra: dict[str, str] | None = None) -> None:
    repo.write_files({**BASE, "shop/cart.py": cart, **(extra or {})})
    repo.commit("base")


# -- two calls on one line ---------------------------------------------------------------------------------------------------

SHARED = "from shop.billing import charge, fee\n\n\ndef checkout(a):\n    return fee(a) + charge(a)\n"


def test_shared_line_one_call_deleted_reports_only_that_call_as_removed(git_repo: GitRepo) -> None:
    _repo(git_repo, SHARED)
    git_repo.write("shop/cart.py", SHARED.replace("fee(a) + charge(a)", "charge(a)"))

    assert _rows(build_view(git_repo)) == [("removed", "shop.cart.checkout", "shop.billing.fee")]


def test_shared_line_one_call_shadowed_is_lost_resolution_for_that_call_and_the_other_stays_resolved(git_repo: GitRepo) -> None:
    _repo(git_repo, SHARED)
    git_repo.write("shop/cart.py", SHARED.replace("checkout(a)", "checkout(a, fee)"))  # `fee` is now the parameter; the text is unchanged

    view = build_view(git_repo)

    assert view.calls == ()
    assert flags(view, "shop.cart.checkout") == {"SIGNATURE_CHANGED"}


def test_shared_line_both_calls_kept_but_swapped_in_order_is_no_call_change(git_repo: GitRepo) -> None:
    _repo(git_repo, SHARED)
    git_repo.write("shop/cart.py", SHARED.replace("fee(a) + charge(a)", "charge(a) + fee(a)"))

    assert build_view(git_repo).calls == ()


def test_shared_line_one_call_replaced_by_another_function_is_one_removal_and_one_addition(git_repo: GitRepo) -> None:
    _repo(git_repo, SHARED, {"shop/pricing.py": "def tax(a):\n    return a\n"})
    git_repo.write("shop/cart.py", "from shop.billing import charge\nfrom shop.pricing import tax\n\n\ndef checkout(a):\n    return tax(a) + charge(a)\n")

    assert _rows(build_view(git_repo)) == [
        ("added", "shop.cart.checkout", "shop.pricing.tax"), ("removed", "shop.cart.checkout", "shop.billing.fee"),
    ]


# -- attribute calls -----------------------------------------------------------------------------------------------------------

REPO = (
    "class Repo:\n    def save(self, x):\n        return x\n\n    def flush(self):\n        return 1\n\n"
    "    def run(self, x):\n        self.flush()\n        return self.save(x)\n"
)


def test_removing_a_self_method_call_is_a_removed_call_while_the_method_still_exists(git_repo: GitRepo) -> None:
    _repo(git_repo, REPO)
    git_repo.write("shop/cart.py", REPO.replace("        self.flush()\n", ""))

    assert _rows(build_view(git_repo)) == [("removed", "shop.cart.Repo.run", "shop.cart.Repo.flush")]


def test_a_self_method_call_that_stops_resolving_because_the_method_is_defined_twice_is_not_a_removal(git_repo: GitRepo) -> None:
    _repo(git_repo, REPO)
    git_repo.write("shop/cart.py", REPO + "\n    def flush(self):\n        return 2\n")  # now defined twice: the resolver steps back

    assert not any(c.change == "removed" for c in build_view(git_repo).calls)


def test_an_attribute_call_on_a_variable_never_resolved_is_never_a_call_row_in_either_direction(git_repo: GitRepo) -> None:
    attr = "def checkout(repo, a):\n    repo.save(a)\n    return a\n"
    _repo(git_repo, attr)
    git_repo.write("shop/cart.py", "def checkout(repo, a):\n    return a\n")

    view = build_view(git_repo)

    assert view.calls == () and flags(view, "shop.cart.checkout") == {"BODY_CHANGED"}


# -- module-qualified calls --------------------------------------------------------------------------------------------------

MODCALL = "import shop.billing as billing\n\n\ndef checkout(a):\n    return billing.fee(a)\n"


def test_removing_a_module_qualified_call_is_a_removed_call(git_repo: GitRepo) -> None:
    _repo(git_repo, MODCALL)
    git_repo.write("shop/cart.py", "import shop.billing as billing\n\n\ndef checkout(a):\n    return a\n")

    assert _rows(build_view(git_repo)) == [("removed", "shop.cart.checkout", "shop.billing.fee")]


def test_a_module_qualified_call_whose_module_alias_is_shadowed_is_lost_resolution_not_a_removal(git_repo: GitRepo) -> None:
    _repo(git_repo, MODCALL)
    git_repo.write("shop/cart.py", MODCALL.replace("checkout(a)", "checkout(a, billing)"))

    assert build_view(git_repo).calls == ()


def test_a_module_qualified_call_to_a_function_that_was_deleted_is_lost_resolution_not_a_removal(git_repo: GitRepo) -> None:
    _repo(git_repo, MODCALL)
    git_repo.write("shop/billing.py", "def charge(a):\n    return a\n")

    view = build_view(git_repo)

    assert view.calls == ()
    assert [s.qualname for s in view.symbols] == ["shop.billing.fee"]


# -- deleted targets and deleted imports -----------------------------------------------------------------------------------------

IMPORTED = "from shop.billing import fee\n\n\ndef checkout(a):\n    return fee(a)\n"


def test_the_import_deleted_but_the_call_text_kept_is_lost_resolution_not_a_removal(git_repo: GitRepo) -> None:
    _repo(git_repo, IMPORTED)
    git_repo.write("shop/cart.py", "def checkout(a):\n    return fee(a)\n")

    assert build_view(git_repo).calls == ()


def test_a_deleted_class_whose_constructor_call_text_remains_is_not_a_removed_call(git_repo: GitRepo) -> None:
    _repo(git_repo, "class Box:\n    pass\n\n\ndef make():\n    return Box()\n")
    git_repo.write("shop/cart.py", "def make():\n    return Box()\n")

    assert not any(c.change == "removed" for c in build_view(git_repo).calls)


def test_a_constructor_call_removed_while_the_class_remains_is_a_removed_call(git_repo: GitRepo) -> None:
    _repo(git_repo, "class Box:\n    pass\n\n\ndef make():\n    return Box()\n")
    git_repo.write("shop/cart.py", "class Box:\n    pass\n\n\ndef make():\n    return None\n")

    assert _rows(build_view(git_repo)) == [("removed", "shop.cart.make", "shop.cart.Box")]


def test_a_function_deleted_with_its_callee_lists_no_call_rows_for_the_deleted_function(git_repo: GitRepo) -> None:
    _repo(git_repo, IMPORTED)
    git_repo.write("shop/cart.py", "")

    view = build_view(git_repo)

    assert view.calls == ()
    assert [(s.change, s.qualname) for s in view.symbols] == [("removed", "shop.cart.checkout")]


def test_head_mode_lists_unanalysable_files_of_the_head_commit_too(git_repo: GitRepo) -> None:
    from tests.explore.diff_helpers import diff_out, not_determinable

    git_repo.write_files({"shop/__init__.py": "", "build/gen.py": "def gen():\n    return 1\n"})
    git_repo.write("shop/odd.py", b"# caf\xe9\ndef odd():\n    return 1\n")
    git_repo.commit("base")
    git_repo.write("build/gen.py", "def gen():\n    return 2\n")
    git_repo.write("shop/odd.py", b"# caf\xe8\ndef odd():\n    return 2\n")
    git_repo.commit("edits")

    out = diff_out(git_repo, "--base", "HEAD~1", "--head", "HEAD")

    assert "shop/odd.py is not valid UTF-8" in not_determinable(out) and "build/" in out.split("Not scanned", 1)[1]
