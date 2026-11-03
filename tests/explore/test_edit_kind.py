"""Edits are ranked by what they touch: a changed condition first, then a returned or raised value, then other statements.

Rule under test (found when the owner missed the real DCF fix under a cosmetic edit that had more callers): inside one attention level the
kind of edit comes before the known-caller count. The kind is read from the syntax trees of the old and the new function, nothing is run,
and anything the reader cannot classify keeps the old place. The row names the kind; it never says the edit is right or wrong.
"""

from __future__ import annotations

import ast

import pytest

from argus.explore.changes import Flag, SymbolChange
from argus.explore.edit_kind import edit_kind
from argus.explore.render_compact import EditLog, compact_rows
from argus.explore.review_order import order_symbols
from tests.explore.diff_helpers import GitRepo
from tests.explore.test_diff_view import build_view


def _kind(old: str, new: str) -> str:
    return edit_kind(ast.parse(old).body[0], ast.parse(new).body[0])  # type: ignore[arg-type]


# -- reading the edit ----------------------------------------------------------------------------------------------------


def test_a_changed_if_condition_is_a_condition_edit() -> None:
    assert _kind("def f(a, b):\n    if a > 0:\n        return b\n    return 0\n", "def f(a, b):\n    if a > 0 and b == b:\n        return b\n    return 0\n") == "condition"


def test_a_changed_while_assert_or_ternary_test_is_a_condition_edit() -> None:
    assert _kind("def f(a):\n    while a:\n        a -= 1\n", "def f(a):\n    while a > 1:\n        a -= 1\n") == "condition"
    assert _kind("def f(a):\n    assert a\n", "def f(a):\n    assert a > 0\n") == "condition"
    assert _kind("def f(a):\n    return 1 if a else 2\n", "def f(a):\n    return 1 if a > 1 else 2\n") == "condition"


def test_a_comprehension_filter_is_a_condition_edit() -> None:
    assert _kind("def f(xs):\n    return [x for x in xs if x]\n", "def f(xs):\n    return [x for x in xs if x > 0]\n") == "condition"


def test_a_changed_return_value_is_a_result_edit() -> None:
    assert _kind("def f(rows):\n    return set(rows)\n", "def f(rows):\n    return set(sorted(rows))\n") == "result"


def test_a_changed_raise_is_a_result_edit_and_a_bare_return_equals_return_none() -> None:
    assert _kind("def f():\n    raise ValueError('a')\n", "def f():\n    raise KeyError('a')\n") == "result"
    assert _kind("def f():\n    return None\n", "def f():\n    return\n") == ""
    assert _kind("def f():\n    yield 1\n", "def f():\n    yield 2\n") == "statements"  # a yield is not a returned or raised value


@pytest.mark.parametrize("old,new", [
    ("try:\n        a()\n    except ValueError:\n        pass", "try:\n        a()\n    except KeyError:\n        pass"),
    ("with a() as x:\n        pass", "with b() as x:\n        pass"),
    ("for x in xs:\n        a(x)", "for x in xs[1:]:\n        a(x)"),
    ("for x in xs:\n        break", "for x in xs:\n        continue"),
    ("match v:\n        case 1:\n            a()", "match v:\n        case 2:\n            a()"),
    ("raise E() from a", "raise E() from b"),
    ("assert x, 'a'", "assert x, 'b'"),
    ("a()\n    b()", "b()\n    a()"),
    ("import a", "import b"),
    ("global a", "global b"),
])
def test_edits_that_change_behaviour_are_never_left_unclassified(old: str, new: str) -> None:
    assert _kind(f"def f(x, xs, v, a, b):\n    {old}\n", f"def f(x, xs, v, a, b):\n    {new}\n") in ("statements", "result")


def test_a_pathologically_deep_expression_is_unclassified_not_a_crash() -> None:
    deep = "def f():\n    return " + "(" * 90 + "1" + ")" * 90 + "\n"
    nested = ast.parse("def f():\n    return 1\n").body[0]
    ret = nested.body[0]
    node = ret.value
    for _ in range(5000):
        node = ast.UnaryOp(op=ast.USub(), operand=node)
    ret.value = node

    assert edit_kind(nested, ast.parse(deep).body[0]) in ("", "result")  # type: ignore[arg-type]


def test_the_strongest_kind_wins_when_several_changed() -> None:
    old = "def f(a, b):\n    if a:\n        b = 1\n    return b\n"
    new = "def f(a, b):\n    if not a:\n        b = 2\n    return b + 1\n"

    assert _kind(old, new) == "condition"


def test_another_statement_is_a_statement_edit() -> None:
    assert _kind("def f(a):\n    a = a + 1\n    return a\n", "def f(a):\n    a = a + 2\n    return a\n") == "statements"
    assert _kind("def f(a):\n    log(a)\n    return a\n", "def f(a):\n    log(a, 1)\n    return a\n") == "statements"


def test_a_docstring_edit_changes_no_kind() -> None:
    assert _kind('def f(a):\n    """old"""\n    return a\n', 'def f(a):\n    """new words"""\n    return a\n') == ""


def test_an_edit_the_reader_cannot_classify_is_unknown_not_guessed() -> None:
    # only the name of a loop variable and a decorator differ: none of the three kinds changed
    assert _kind("def f(xs):\n    for x in xs:\n        pass\n", "def f(xs):\n    for y in xs:\n        pass\n") == ""


def test_nested_functions_count_as_part_of_the_function() -> None:
    old = "def f():\n    def g(a):\n        return a\n    return g\n"
    new = "def f():\n    def g(a):\n        return a + 1\n    return g\n"

    assert _kind(old, new) == "result"


# -- the order -----------------------------------------------------------------------------------------------------------


def _sym(name: str, kind: str, callers: int) -> SymbolChange:
    return SymbolChange(name, "function", "modified", frozenset({Flag.BODY_CHANGED}), None, None, False, "m/x.py", 1, known_callers=callers, edit_kind=kind)


def test_the_kind_of_edit_comes_before_the_caller_count() -> None:
    cosmetic = _sym("cosmetic", "result", 3)
    real = _sym("real", "condition", 1)
    other = _sym("other", "statements", 9)
    unknown = _sym("unknown", "", 9)

    assert [s.qualname for s in order_symbols((other, cosmetic, unknown, real))] == ["real", "cosmetic", "unknown", "other"]  # unread never sinks below known


def test_equal_kinds_still_order_by_callers() -> None:
    a, b = _sym("a", "condition", 1), _sym("b", "condition", 4)

    assert [s.qualname for s in order_symbols((a, b))] == ["b", "a"]


def test_a_removed_or_interface_changed_symbol_still_comes_first() -> None:
    sig = SymbolChange("sig", "function", "modified", frozenset({Flag.SIGNATURE_CHANGED}), None, None, False, "m/x.py", 1, known_callers=0)

    assert order_symbols((_sym("real", "condition", 50), sig))[0].qualname == "sig"


# -- on a real repository --------------------------------------------------------------------------------------------------

CORE = (
    "def watchlist(rows):\n    return set(rows)\n\n\n"
    "def dcf(fcf, debt):\n    if not fcf > 0:\n        return None\n    equity = fcf - (debt if debt == debt else 0.0)\n    return equity\n"
)
USE = (
    "from m.core import dcf, watchlist\n\n\n"
    "def a():\n    return watchlist([1])\n\n\n"
    "def b():\n    return watchlist([2])\n\n\n"
    "def c():\n    return watchlist([3])\n\n\n"
    "def d():\n    return dcf(1, 2)\n"
)
NEW_CORE = (
    "def watchlist(rows):\n    return set(sorted(rows))\n\n\n"
    "def dcf(fcf, debt):\n    if not (fcf > 0 and debt == debt):\n        return None\n    return fcf - debt\n"
)


def test_the_real_fix_ranks_above_the_cosmetic_edit_with_more_callers(git_repo: GitRepo) -> None:
    git_repo.write_files({"m/__init__.py": "", "m/core.py": CORE, "m/use.py": USE})
    git_repo.commit("base")
    git_repo.write("m/core.py", NEW_CORE)

    rows = [r for r in compact_rows(build_view(git_repo), None, EditLog())[0] if r.change is not None]

    assert [r.change.qualname for r in rows][:2] == ["m.core.dcf", "m.core.watchlist"]
    assert "logic edited, including a condition" in rows[0].text and "logic edited, including a returned or raised value" in rows[1].text
    assert rows[0].change.known_callers < rows[1].change.known_callers  # it won although it has fewer callers


@pytest.mark.parametrize("old,new,kind", [("return rows", "return list(rows)", "result")])
def test_the_kind_is_computed_for_a_modified_function(git_repo: GitRepo, old: str, new: str, kind: str) -> None:
    git_repo.write_files({"m/__init__.py": "", "m/core.py": f"def f(rows):\n    {old}\n"})
    git_repo.commit("base")
    git_repo.write("m/core.py", f"def f(rows):\n    {new}\n")

    (change,) = build_view(git_repo).symbols

    assert change.edit_kind == kind
