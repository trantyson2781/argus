"""Diff Explorer v0.1: which flags a changed function gets (contract: see diff_helpers.py)."""

from __future__ import annotations

import pytest

from tests.explore.diff_helpers import (
    BILLING, GitRepo, affected_groups, before_notes, call_rows, diff_out, direct_groups, flags_in, group_names, section, symbol_lines,
)

CHARGE = "shop.billing.charge"
RULES = (
    "import functools\n\n\n"
    "def plain(x):\n    return x\n\n\n"
    "@functools.lru_cache(maxsize=8)\ndef cached(x):\n    return x\n\n\n"
    "def with_default(x, rate=1):\n    return x * rate\n\n\n"
    "def annotated(x: int) -> int:\n    return x\n\n\n"
    "def make_async(x):\n    return x\n\n\n"
    "def renamed(x):\n    return 0\n"
)


def test_body_only_change_sets_only_body_changed_and_lists_direct_callers_as_potentially_affected(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))

    out = diff_out(shop_repo)

    lines = symbol_lines(out, CHARGE)
    assert len(lines) == 1 and "[FACT]" in lines[0] and "shop/billing.py:1" in lines[0]
    assert flags_in(lines) == {"BODY_CHANGED"}
    assert "docstring" not in section(out, "Symbols")
    assert direct_groups(out) == []
    assert group_names(affected_groups(out)) == ["shop.cart.checkout", "shop.cart.checkout_twice"]
    assert "shop.api.view" not in before_notes(out)  # no transitive impact
    assert call_rows(out) == []


def test_formatting_only_change_reports_no_flag_and_no_structural_change(git_repo: GitRepo) -> None:
    git_repo.write_files({
        "shop/__init__.py": "",
        "shop/billing.py": "def charge(amount, rate):\n    label = 'total'\n    return (amount + 1) * rate, label\n",
        "shop/cart.py": "from shop.billing import charge\n\n\ndef checkout(a):\n    return charge(a, 2)\n",
    })
    git_repo.commit("base")
    git_repo.write("shop/billing.py", (
        "# a new header comment\n\n\n\n"
        'def charge(\n    amount,\n    rate,\n):  # trailing comment\n    # explain\n    label = "total"\n\n'
        "    return (\n        (amount + 1)\n        * rate\n    ), label\n"
    ))
    git_repo.write("shop/cart.py", "\n\n# shifted down\nfrom shop.billing import charge\n\n\ndef checkout(a):\n    return  charge( a,  2 )\n")

    out = diff_out(git_repo)

    assert "no structural change" in out
    assert section(out, "Symbols") == "" and section(out, "Calls") == ""
    assert flags_in(before_notes(out).splitlines()) == set()
    assert direct_groups(out) == [] and affected_groups(out) == []


def test_docstring_only_change_sets_body_changed_and_says_it_is_a_syntactic_change(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", 'def charge(amount):\n    """Charge the amount."""\n    return amount + 1\n\n\ndef fee(amount):\n    return amount\n')

    out = diff_out(shop_repo)

    assert flags_in(symbol_lines(out, CHARGE)) == {"BODY_CHANGED"}
    assert "syntactic change" in section(out, "Symbols")
    assert group_names(affected_groups(out)) == ["shop.cart.checkout", "shop.cart.checkout_twice"]


@pytest.mark.parametrize(
    ("qualname", "old", "new"),
    [
        ("shop.rules.cached", "@functools.lru_cache(maxsize=8)", "@functools.lru_cache(maxsize=16)"),
        ("shop.rules.plain", "def plain(x):", "@functools.cache\ndef plain(x):"),
        ("shop.rules.with_default", "rate=1", "rate=2"),
        ("shop.rules.annotated", "def annotated(x: int)", "def annotated(x: str)"),
        ("shop.rules.annotated", "-> int:", "-> float:"),
        ("shop.rules.make_async", "def make_async(x):", "async def make_async(x):"),
        ("shop.rules.renamed", "def renamed(x):", "def renamed(y):"),
    ],
    ids=["decorator-argument", "decorator-added", "default-value", "param-annotation", "return-annotation", "async-toggle", "param-rename"],
)
def test_interface_only_edits_set_signature_changed_and_never_body_changed(git_repo: GitRepo, qualname: str, old: str, new: str) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/rules.py": RULES})
    git_repo.commit("base")
    assert old in RULES
    git_repo.write("shop/rules.py", RULES.replace(old, new, 1))

    out = diff_out(git_repo)

    lines = symbol_lines(out, qualname)
    assert lines and flags_in(lines) == {"SIGNATURE_CHANGED"}, out
    assert "[FACT]" in lines[0]
    rows = [line for line in section(out, "Symbols").splitlines() if line.strip().startswith("[")]
    assert rows and all(line in lines for line in rows)  # nothing else changed


def test_combined_signature_and_body_change_prints_both_flags_on_one_line_with_the_signature_diff(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", 'def charge(amount, currency):\n    return amount * 2\n\n\ndef fee(amount):\n    return amount\n')

    out = diff_out(shop_repo)

    lines = symbol_lines(out, CHARGE)
    first = next(line for line in lines if "SIGNATURE_CHANGED" in line)
    assert "BODY_CHANGED" in first and "[FACT]" in first and "shop/billing.py:1" in first
    assert "was (amount)" in section(out, "Symbols") and "now (amount, currency)" in section(out, "Symbols")


def test_added_and_removed_functions_are_reported_as_added_and_removed_facts(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount + 1\n\n\ndef refund(amount):\n    return -amount\n")

    out = diff_out(shop_repo)

    added, removed = symbol_lines(out, "shop.billing.refund"), symbol_lines(out, "shop.billing.fee")
    assert len(added) == 1 and "[FACT]" in added[0] and "+ function" in added[0] and "shop/billing.py:5" in added[0]
    assert len(removed) == 1 and "[FACT]" in removed[0] and "- function" in removed[0] and "was shop/billing.py:5" in removed[0]
    assert symbol_lines(out, CHARGE) == []


# -- symbols that cannot be compared ---------------------------------------------------------------------------------

DUPLICATES = {
    "property": (
        "class Box:\n    @property\n    def size(self):\n        return 1\n\n    @size.setter\n    def size(self, v):\n        pass\n",
        "Box.size",
    ),
    "if-else": ("import os\n\nif os.name:\n    def pick():\n        return 1\nelse:\n    def pick():\n        return 2\n", "pick"),
    "overload": (
        "from typing import overload\n\n\n@overload\ndef ov(x: int) -> int: ...\n@overload\ndef ov(x: str) -> str: ...\n"
        "def ov(x):\n    return x\n",
        "ov",
    ),
    "class-defined-twice": ("class A:\n    def m(self):\n        return 1\n\n\nclass A:\n    def m(self):\n        return 2\n", "A.m"),
}


@pytest.mark.parametrize("case", sorted(DUPLICATES))
def test_symbols_defined_more_than_once_get_no_flags_and_are_listed_as_unknown(git_repo: GitRepo, case: str) -> None:
    source, local = DUPLICATES[case]
    git_repo.write_files({"shop/__init__.py": "", "shop/dup.py": source})
    git_repo.commit("base")
    edited = source.replace("return 1", "return 11").replace("return 2", "return 22").replace("pass", "return 5").replace("return x", "return x + 1")
    assert edited != source
    git_repo.write("shop/dup.py", edited)

    out = diff_out(git_repo)

    assert f"shop.dup.{local}" not in section(out, "Symbols") and "#2" not in section(out, "Symbols")
    unknown = [line for line in section(out, "Not determinable").splitlines() if "defined more than once" in line]
    assert unknown and all("[UNKNOWN]" in line for line in unknown)
    assert local.split(".")[-1] in " ".join(unknown)


def test_editing_a_nested_function_flags_inner_and_enclosing_with_calls_changed_on_inner_only(git_repo: GitRepo) -> None:
    base = "def g():\n    return 1\n\n\ndef outer():\n    def inner():\n        return 1\n    return inner()\n"
    git_repo.write_files({"shop/__init__.py": "", "shop/nest.py": base})
    git_repo.commit("base")
    git_repo.write("shop/nest.py", base.replace("        return 1\n    return inner()", "        return g()\n    return inner()"))

    out = diff_out(git_repo)

    assert flags_in(symbol_lines(out, "shop.nest.outer.inner")) == {"BODY_CHANGED", "CALLS_CHANGED"}
    assert flags_in(symbol_lines(out, "shop.nest.outer")) == {"BODY_CHANGED"}
    assert [(sign, src, dst) for sign, src, dst, *_ in call_rows(out)] == [("+", "shop.nest.outer.inner", "shop.nest.g")]


def test_classes_are_reported_only_when_added_or_removed(git_repo: GitRepo) -> None:
    base = "class Legacy:\n    pass\n\n\nclass Base1:\n    pass\n\n\nclass Base2:\n    pass\n\n\nclass Model(Base1):\n    pass\n"
    git_repo.write_files({"shop/__init__.py": "", "shop/models.py": base})
    git_repo.commit("base")
    git_repo.write("shop/models.py", base.replace("class Legacy:\n    pass\n\n\n", "class Fresh:\n    pass\n\n\n").replace("Model(Base1)", "Model(Base2)"))

    out = diff_out(git_repo)

    assert [("+ class" in line, "- class" in line) for line in symbol_lines(out, "shop.models.Fresh")] == [(True, False)]
    assert [("+ class" in line, "- class" in line) for line in symbol_lines(out, "shop.models.Legacy")] == [(False, True)]
    assert symbol_lines(out, "shop.models.Model") == []


def test_changing_only_a_class_decorator_or_base_lists_the_file_as_code_outside_functions_and_says_class_level_code_is_not_compared(git_repo: GitRepo) -> None:
    base = "import dataclasses\n\n\nclass Base1:\n    pass\n\n\nclass Base2:\n    pass\n\n\n@dataclasses.dataclass\nclass Model(Base1):\n    x: int = 1\n"
    git_repo.write_files({"shop/__init__.py": "", "shop/models.py": base})
    git_repo.commit("base")
    git_repo.write("shop/models.py", base.replace("@dataclasses.dataclass", "@dataclasses.dataclass(frozen=True)").replace("Model(Base1)", "Model(Base2)"))

    out = diff_out(git_repo)

    assert section(out, "Symbols") == "" and "no structural change" not in out
    assert "shop/models.py" in section(out, "Files changed outside any function or class")
    assert "class-level code" in section(out, "Not determinable")
