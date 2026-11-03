"""Attention labels for the compact view (Phase 1.5, product track; plain-language verdict for readers who do not read code).

Rules under test: CHECK for a removed symbol or a changed interface, LOOK for edited logic, LOW for a new symbol or a description-only edit.
A label says where to look first, never that a change is right, so the legend says so. Rows use everyday words, no file paths and none
of the report's technical terms; counts are lower bounds ("at least", "found"), never "none".
"""

from __future__ import annotations

from rich.text import Text

from argus.explore.changes import DiffView, Flag, SymbolChange
from argus.explore.render_compact import compact_lines, compact_rows, stylize_label
from argus.explore.review_order import Attention, attention
from tests.explore.diff_helpers import GitRepo
from tests.explore.test_diff_view import build_view, change
from tests.explore.test_review_order import scenario


def _sym(
    qualname: str, change_kind: str, flags: frozenset[Flag] = frozenset(), known: int | None = None, tests: int | None = None,
    docstring_only: bool = False, kind: str = "function",
) -> SymbolChange:
    return SymbolChange(
        qualname, kind, change_kind, flags, None, None, False, "src/m/x.py", 7, docstring_only=docstring_only, known_callers=known, test_callers=tests,
    )  # type: ignore[arg-type]


def _view(*symbols: SymbolChange) -> DiffView:
    return DiffView("HEAD", "working tree", True, 1, 0, symbols, (), (), (), 0, (), ())


def _rows(*symbols: SymbolChange) -> list[str]:
    rows, _ = compact_rows(_view(*symbols), None)
    return [row.text for row in rows]


# -- the rule ------------------------------------------------------------------------------------------------------------------


def test_removed_or_interface_changed_is_check_even_when_the_body_changed_too() -> None:
    assert attention(_sym("gone", "removed")) is Attention.CHECK
    assert attention(_sym("sig", "modified", frozenset({Flag.SIGNATURE_CHANGED}))) is Attention.CHECK
    assert attention(_sym("both", "modified", frozenset({Flag.SIGNATURE_CHANGED, Flag.BODY_CHANGED}))) is Attention.CHECK


def test_edited_logic_or_changed_calls_is_look() -> None:
    assert attention(_sym("body", "modified", frozenset({Flag.BODY_CHANGED}))) is Attention.LOOK
    assert attention(_sym("calls", "modified", frozenset({Flag.CALLS_CHANGED}))) is Attention.LOOK


def test_a_new_symbol_or_a_description_only_edit_is_low() -> None:
    assert attention(_sym("fresh", "added")) is Attention.LOW
    assert attention(_sym("doc", "modified", frozenset({Flag.BODY_CHANGED}), docstring_only=True)) is Attention.LOW


# -- test callers, counted on a real repository ----------------------------------------------------------------------------------------


def test_test_callers_counts_test_functions_that_call_a_changed_symbol_directly(git_repo: GitRepo) -> None:
    scenario(git_repo)

    view = build_view(git_repo)

    assert change(view, "m.core.body").test_callers == 1  # tests/test_use.py::test_body
    assert change(view, "m.core.sig").test_callers == 0
    assert change(view, "m.core.fresh").test_callers is None  # nothing is counted for an added symbol


def test_a_helper_in_a_test_file_is_not_counted_as_a_test(git_repo: GitRepo) -> None:
    git_repo.write_files({"m/__init__.py": "", "m/core.py": "def f(x):\n    return x\n", "tests/helpers.py": "from m.core import f\n\n\ndef run():\n    return f(1)\n"})
    git_repo.commit("base")
    git_repo.write("m/core.py", "def f(x):\n    return x + 1\n")

    assert change(build_view(git_repo), "m.core.f").test_callers == 0


# -- the words -------------------------------------------------------------------------------------------------------------------


def test_a_removed_symbol_says_it_was_used_not_that_it_is() -> None:
    (row,) = _rows(_sym("m.core.gone", "removed", known=2, tests=0))

    assert "CHECK" in row and "deleted" in row and "was used in at least 2 places" in row and "no test found" in row


def test_an_interface_change_says_inputs_or_outputs_and_pluralises_counts() -> None:
    (one,) = _rows(_sym("m.core.sig", "modified", frozenset({Flag.SIGNATURE_CHANGED}), known=1, tests=1))
    (many,) = _rows(_sym("m.core.sig", "modified", frozenset({Flag.SIGNATURE_CHANGED}), known=3, tests=2))

    assert "inputs or outputs changed" in one and "used in at least 1 place" in one and "1 test found" in one
    assert "used in at least 3 places" in many and "2 tests found" in many


def test_an_edit_with_no_known_user_does_not_claim_there_is_none() -> None:
    (row,) = _rows(_sym("m.core.body", "modified", frozenset({Flag.BODY_CHANGED}), known=0, tests=0))

    assert "LOOK" in row and "logic edited" in row and "no known users" in row and "no test found" in row


def test_a_new_function_or_class_is_named_and_carries_no_counts() -> None:
    (fn,) = _rows(_sym("m.core.fresh", "added"))
    (cls,) = _rows(_sym("m.core.Fresh", "added", kind="class"))

    assert "LOW" in fn and "new function" in fn and "used in" not in fn and "test" not in fn
    assert "new class" in cls


def test_a_description_only_edit_says_so() -> None:
    (row,) = _rows(_sym("m.core.doc", "modified", frozenset({Flag.BODY_CHANGED}), known=2, tests=0, docstring_only=True))

    assert "LOW" in row and "description edited" in row


def test_rows_use_everyday_words_and_no_file_paths() -> None:
    rows = _rows(
        _sym("a.sig", "modified", frozenset({Flag.SIGNATURE_CHANGED, Flag.BODY_CHANGED}), known=2, tests=0),
        _sym("a.calls", "modified", frozenset({Flag.CALLS_CHANGED}), known=0, tests=0),
        _sym("a.gone", "removed", known=0, tests=0),
    )

    for row in rows:
        assert not any(word in row.lower() for word in ("interface", "caller", "signature", "body", "x.py", ":7"))
    assert "now calls different functions" in rows[-1] or any("now calls different functions" in r for r in rows)


def test_the_most_urgent_label_comes_first_and_the_review_first_line_is_gone() -> None:
    lines, _ = compact_lines(
        _view(_sym("a.fresh", "added"), _sym("a.body", "modified", frozenset({Flag.BODY_CHANGED}), known=1, tests=0), _sym("a.sig", "removed", known=1, tests=0)),
        since="comparing with HEAD", edit=None, previous=None, refreshed="12:00:00 in 1.0 s",
    )

    labels = [line.split()[0] for line in lines if line.split()[:1] in (["CHECK"], ["LOOK"], ["LOW"]) and "where to look first" not in line]  # not the legend
    assert labels == ["CHECK", "LOOK", "LOW"]
    assert not any("Review first" in line for line in lines)


def test_a_legend_says_the_label_is_where_to_look_not_whether_it_is_right() -> None:
    lines, _ = compact_lines(_view(_sym("a.sig", "removed", known=1, tests=0)), since="s", edit=None, previous=None, refreshed="r")
    none, _ = compact_lines(_view(), since="s", edit=None, previous=None, refreshed="r")

    assert any("not whether it is right" in line for line in lines)
    assert not any("not whether it is right" in line for line in none)  # nothing to explain when nothing changed


def test_the_label_is_coloured_by_attention() -> None:
    check = Text("  NEW CHECK a.sig  deleted")
    low = Text("      LOW   a.fresh  new function")
    plain = Text("  tests: 3 added")

    for text in (check, low, plain):
        stylize_label(text)

    assert any("red" in str(span.style) for span in check.spans)
    assert any("green" in str(span.style) for span in low.spans)
    assert not plain.spans
