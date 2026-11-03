"""Diff Explorer v0.1: calls Argus cannot resolve, call comparison and files that do not parse (contract: see diff_helpers.py)."""

from __future__ import annotations

import re

from tests.explore.diff_helpers import (
    BILLING, GitRepo, affected_groups, call_rows, diff_out, direct_groups, flags_in, group_names, not_determinable, remove_fee_scenario,
    section, symbol_lines,
)

UNRESOLVED_NOTE = re.compile(r"\b1 calls? made by the changed functions could not be traced")


def test_a_caller_reached_only_through_obj_method_is_absent_counted_as_unresolved_and_never_guessed(git_repo: GitRepo) -> None:
    git_repo.write_files({
        "shop/__init__.py": "", "shop/billing.py": BILLING,
        "shop/cart.py": "def checkout(client, amount):\n    return client.charge(amount)\n",
    })
    git_repo.commit("base")
    git_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))
    git_repo.write("shop/cart.py", "def checkout(client, amount):\n    return client.charge(amount) + 1\n")

    out = diff_out(git_repo)

    assert direct_groups(out) == [] and affected_groups(out) == []  # no edge was guessed from the method name
    assert call_rows(out) == []
    assert flags_in(symbol_lines(out, "shop.cart.checkout")) == {"BODY_CHANGED"}
    notes = not_determinable(out)
    line = next(row for row in notes.splitlines() if UNRESOLVED_NOTE.search(row))
    assert line.strip().startswith("[UNKNOWN]") and "that caller is missing from the lists above" in line


def test_resolved_calls_added_or_removed_set_calls_changed_but_a_changed_call_count_does_not(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/cart.py", (
        "from shop.billing import charge, fee\n\n\n"
        "def checkout(amount):\n    return charge(amount) + fee(amount)\n\n\n"
        "def checkout_twice(amount):\n    return charge(amount)\n"
    ))
    shop_repo.write("shop/api.py", "from shop.cart import checkout\n\n\ndef view(amount):\n    return amount\n")

    out = diff_out(shop_repo)

    assert sorted(call_rows(out)) == [
        ("+", "shop.cart.checkout", "shop.billing.fee", "shop/cart.py", 5),
        ("-", "shop.api.view", "shop.cart.checkout", "shop/api.py", 5),
    ]
    assert "Calls (2)" in out
    assert flags_in(symbol_lines(out, "shop.cart.checkout")) == {"BODY_CHANGED", "CALLS_CHANGED"}
    assert flags_in(symbol_lines(out, "shop.api.view")) == {"BODY_CHANGED", "CALLS_CHANGED"}
    assert flags_in(symbol_lines(out, "shop.cart.checkout_twice")) == {"BODY_CHANGED"}  # two calls became one: same (kind, source, target)


def test_added_calls_to_external_code_and_builtins_are_not_calls_changed_and_not_in_the_calls_list(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "import os\n\n\ndef charge(amount):\n    os.getcwd()\n    return len([amount]) + amount\n\n\ndef fee(amount):\n    return amount\n")

    out = diff_out(shop_repo)

    assert flags_in(symbol_lines(out, "shop.billing.charge")) == {"BODY_CHANGED"}
    assert call_rows(out) == [] and section(out, "Calls") == ""


def test_loss_of_resolution_is_not_reported_as_a_removed_call(git_repo: GitRepo) -> None:
    remove_fee_scenario(git_repo)  # cart.checkout and reports.monthly still say fee(a), but fee no longer exists

    out = diff_out(git_repo)

    assert call_rows(out) == []
    assert symbol_lines(out, "shop.cart.checkout") == []  # unchanged file: no CALLS_CHANGED, nothing reported about it
    assert flags_in(symbol_lines(out, "shop.reports.monthly")) == {"BODY_CHANGED"}
    assert "shop.cart.checkout" in group_names(direct_groups(out))  # it is still listed as a (base graph) caller of the removed symbol


# -- files that do not parse -----------------------------------------------------------------------------------------

A_OK = "def helper():\n    return 1\n\n\ndef other():\n    return 2\n"
B = "from shop.a import helper\nfrom shop.e import thing\n\n\ndef use():\n    thing()\n    return helper()\n"
C = "from shop.e import thing\n\n\ndef top():\n    return thing()\n"
E = "def thing():\n    return 0\n\n\ndef thing2():\n    return 1\n"
BROKEN = "def helper(:\n    return 1\n"


def _parse_failure_repo(repo: GitRepo) -> None:
    repo.write_files({"shop/__init__.py": "", "shop/a.py": A_OK, "shop/b.py": B, "shop/c.py": C, "shop/e.py": E})
    repo.commit("base")
    repo.write("shop/a.py", BROKEN)


def test_a_file_that_stops_parsing_yields_no_removed_symbols_and_a_not_determinable_line(git_repo: GitRepo) -> None:
    _parse_failure_repo(git_repo)

    out = diff_out(git_repo)

    assert "shop.a." not in section(out, "Symbols") and "- function" not in out
    line = next(row for row in not_determinable(out).splitlines() if "shop/a.py does not parse in the working tree" in row)
    assert line.strip().startswith("[UNKNOWN]") and "compared" in line


def test_unchanged_files_calling_into_a_failed_file_get_no_calls_changed_and_no_removed_call(git_repo: GitRepo) -> None:
    _parse_failure_repo(git_repo)

    out = diff_out(git_repo)

    assert call_rows(out) == []
    assert symbol_lines(out, "shop.b.use") == []
    assert direct_groups(out) == [] and affected_groups(out) == []


def test_call_comparison_is_suppressed_for_modules_importing_from_a_failed_module_and_says_so(git_repo: GitRepo) -> None:
    _parse_failure_repo(git_repo)
    git_repo.write("shop/b.py", B.replace("    thing()\n", "    thing()\n    thing2()\n").replace("import thing", "import thing, thing2"))
    git_repo.write("shop/c.py", C.replace("return thing()", "return thing() + thing2()").replace("import thing", "import thing, thing2"))

    out = diff_out(git_repo)

    assert [(sign, src, dst) for sign, src, dst, *_ in call_rows(out)] == [("+", "shop.c.top", "shop.e.thing2")]  # control: c still compared
    assert flags_in(symbol_lines(out, "shop.b.use")) == {"BODY_CHANGED"}  # still a body change, but its calls are not compared
    line = next(row for row in not_determinable(out).splitlines() if "call comparison" in row)
    assert "suppressed" in line and "shop/b.py" in line and "shop/c.py" not in line


def test_a_file_that_failed_to_parse_in_the_base_is_excluded_and_reported_against_the_base_ref(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/a.py": BROKEN, "shop/e.py": E})
    git_repo.commit("base with a broken file")
    git_repo.write("shop/a.py", A_OK)

    out = diff_out(git_repo)

    assert "shop.a." not in section(out, "Symbols") and "+ function" not in out  # nothing in a.py is "added"
    line = next(row for row in not_determinable(out).splitlines() if "shop/a.py does not parse in" in row)
    assert "HEAD" in line
