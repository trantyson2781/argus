"""Diff Explorer: suggested review order of changed symbols (Phase 1.5, Explorer v0.4 follow-up).

Rule under test: kind of change first (removed or interface-changed, then body-changed, then added), then more known callers first,
then file, line and name. The caller count is a lower bound (the call graph has blind spots), so the text says "at least N" and
no importance label is ever printed. The `--json` list keeps its old order and only gains a `known_callers` field.
"""

from __future__ import annotations

import json

from argus.explore.changes import Flag, SymbolChange
from argus.explore.review_order import order_symbols, review_key
from tests.explore.diff_helpers import GitRepo, diff_out, section
from tests.explore.json_helpers import run_json
from tests.explore.test_diff_view import build_view, change

CORE = (
    "def sig(x):\n    return x\n\n\n"
    "def body(x):\n    return x\n\n\n"
    "def gone(x):\n    return x\n"
)
USE = (
    "from m.core import body, gone, sig\n\n\n"
    "def a(x):\n    sig(x)\n    body(x)\n    return gone(x)\n\n\n"
    "def b(x):\n    sig(x)\n    return body(x)\n\n\n"
    "def c(x):\n    body(x)\n    return body(x)\n\n\n"  # two calls, one caller
    "def d(x):\n    return body(x)\n"
)
TEST_USE = "from m.core import body\n\n\ndef test_body():\n    body(1)\n"
NEW_CORE = (
    "def sig(x, extra):\n    return x\n\n\n"  # interface changed
    "def body(x):\n    return x + 1\n\n\n"  # body changed
    "def fresh(x):\n    return x\n"  # added; gone is removed
)


def scenario(repo: GitRepo) -> None:
    repo.write_files({"m/__init__.py": "", "m/core.py": CORE, "m/use.py": USE, "tests/test_use.py": TEST_USE})
    repo.commit("base")
    repo.write("m/core.py", NEW_CORE)


def symbol_rows(out: str) -> list[str]:
    return [line for line in section(out, "Symbols").splitlines() if line.strip().startswith("[")]


def _sym(qualname: str, change_kind: str, flags: frozenset[Flag] = frozenset(), known: int | None = None, file: str = "m/x.py", line: int = 1) -> SymbolChange:
    return SymbolChange(qualname, "function", change_kind, flags, None, None, False, file, line, known_callers=known)  # type: ignore[arg-type]


# -- the pure ordering -----------------------------------------------------------------------------------------------------------


def test_kind_of_change_comes_before_caller_count() -> None:
    body = _sym("body", "modified", frozenset({Flag.BODY_CHANGED}), known=50)
    sig = _sym("sig", "modified", frozenset({Flag.SIGNATURE_CHANGED}), known=1)
    removed = _sym("gone", "removed", known=2)
    added = _sym("fresh", "added")

    ordered = [s.qualname for s in order_symbols((added, body, removed, sig))]

    assert ordered == ["gone", "sig", "body", "fresh"]  # body has 50 callers but is not an interface change


def test_more_known_callers_come_first_inside_a_kind_and_ties_are_deterministic() -> None:
    low = _sym("low", "modified", frozenset({Flag.SIGNATURE_CHANGED}), known=1, file="m/z.py")
    high = _sym("high", "modified", frozenset({Flag.SIGNATURE_CHANGED}), known=9, file="m/z.py", line=50)
    tie_a = _sym("a_tie", "modified", frozenset({Flag.SIGNATURE_CHANGED}), known=3, file="m/a.py", line=2)
    tie_b = _sym("b_tie", "modified", frozenset({Flag.SIGNATURE_CHANGED}), known=3, file="m/a.py", line=9)

    first = [s.qualname for s in order_symbols((low, tie_b, high, tie_a))]
    second = [s.qualname for s in order_symbols((tie_a, high, low, tie_b))]

    assert first == ["high", "a_tie", "b_tie", "low"]
    assert first == second


def test_test_code_comes_after_production_code_whatever_the_kind_of_change() -> None:
    removed_test = _sym("test_gone", "removed", file="tests/test_x.py")
    body = _sym("body", "modified", frozenset({Flag.BODY_CHANGED}), known=0, file="src/m/x.py")
    added = _sym("fresh", "added", file="src/m/x.py")

    assert [s.qualname for s in order_symbols((removed_test, added, body))] == ["body", "fresh", "test_gone"]


def test_a_symbol_with_an_interface_and_a_body_change_counts_as_interface_changed() -> None:
    both = _sym("both", "modified", frozenset({Flag.SIGNATURE_CHANGED, Flag.BODY_CHANGED}), known=0)
    body = _sym("body", "modified", frozenset({Flag.BODY_CHANGED}), known=9)

    assert [s.qualname for s in order_symbols((body, both))] == ["both", "body"]


def test_an_unknown_count_sorts_like_zero_inside_its_kind() -> None:
    calls_only = _sym("calls_only", "modified", frozenset({Flag.CALLS_CHANGED}), known=None)
    body = _sym("body", "modified", frozenset({Flag.BODY_CHANGED}), known=2)

    assert [s.qualname for s in order_symbols((calls_only, body))] == ["body", "calls_only"]
    assert review_key(calls_only)[:2] == review_key(body)[:2]


# -- the count, on real scanned repositories ---------------------------------------------------------------------------------------


def test_known_callers_counts_distinct_production_callers_not_edges_or_tests(git_repo: GitRepo) -> None:
    scenario(git_repo)

    view = build_view(git_repo)

    assert change(view, "m.core.sig").known_callers == 2  # a, b
    assert change(view, "m.core.body").known_callers == 4  # a, b, c (c calls twice, counted once), d; the test caller is not counted
    assert change(view, "m.core.gone").known_callers == 1  # a, from the base graph
    assert change(view, "m.core.fresh").known_callers is None  # nothing is counted for an added symbol


def test_known_callers_is_the_same_with_and_without_test_callers_shown(git_repo: GitRepo) -> None:
    scenario(git_repo)

    assert change(build_view(git_repo, include_tests=True), "m.core.body").known_callers == change(build_view(git_repo), "m.core.body").known_callers


# -- the text report ---------------------------------------------------------------------------------------------------------------


def test_the_text_lists_symbols_in_review_order(git_repo: GitRepo) -> None:
    scenario(git_repo)

    rows = symbol_rows(diff_out(git_repo))
    names = [next(name for name in ("m.core.sig", "m.core.body", "m.core.gone", "m.core.fresh") if name in row) for row in rows]

    assert names == ["m.core.sig", "m.core.gone", "m.core.body", "m.core.fresh"]  # sig (2 callers) before gone (1); body has the most callers but is not an interface change


def test_the_text_says_at_least_and_never_gives_a_grade(git_repo: GitRepo) -> None:
    scenario(git_repo)

    out = diff_out(git_repo)
    sig_row = next(row for row in symbol_rows(out) if "m.core.sig" in row)
    gone_row = next(row for row in symbol_rows(out) if "m.core.gone" in row)
    fresh_row = next(row for row in symbol_rows(out) if "m.core.fresh" in row)

    assert "at least 2 known callers" in sig_row
    assert "at least 1 known caller" in gone_row and "known callers" not in gone_row
    assert "known caller" not in fresh_row
    for grade in ("HIGH", "LOW", "importance", "risk"):
        assert grade not in section(out, "Symbols")


def test_a_counted_symbol_with_no_known_caller_says_so_without_claiming_it_is_unused(git_repo: GitRepo) -> None:
    git_repo.write_files({"m/__init__.py": "", "m/core.py": "def lonely(x):\n    return x\n"})
    git_repo.commit("base")
    git_repo.write("m/core.py", "def lonely(x):\n    return x + 1\n")

    row = next(row for row in symbol_rows(diff_out(git_repo)) if "m.core.lonely" in row)

    assert "no known callers" in row
    assert "cannot see" in row  # the count is a floor


def test_the_symbols_section_explains_its_order_once(git_repo: GitRepo) -> None:
    scenario(git_repo)

    body = section(diff_out(git_repo), "Symbols")

    assert body.count("suggested review order") == 1
    assert "most known callers first" in body


def test_one_symbol_needs_no_order_line(git_repo: GitRepo) -> None:
    git_repo.write_files({"m/__init__.py": "", "m/core.py": "def lonely(x):\n    return x\n"})
    git_repo.commit("base")
    git_repo.write("m/core.py", "def lonely(x):\n    return x + 1\n")

    assert "suggested review order" not in diff_out(git_repo)


def test_the_text_is_byte_identical_between_runs(git_repo: GitRepo) -> None:
    scenario(git_repo)

    assert diff_out(git_repo) == diff_out(git_repo)


# -- the JSON: old order, new field ------------------------------------------------------------------------------------------------


def test_json_keeps_its_order_and_gains_known_callers(git_repo: GitRepo) -> None:
    scenario(git_repo)

    result = run_json("diff", str(git_repo.root))
    assert result.exit_code == 0, result.output
    symbols = json.loads(result.stdout)["data"]["symbols"]

    assert [s["qualname"] for s in symbols] == ["m.core.sig", "m.core.body", "m.core.fresh", "m.core.gone"]  # the old (file, line, name) order
    counts = {s["qualname"]: s["known_callers"] for s in symbols}
    assert counts == {"m.core.sig": 2, "m.core.body": 4, "m.core.gone": 1, "m.core.fresh": None}
