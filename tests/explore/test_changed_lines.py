"""Changed lines and the count of changes (Phase 1.5, product track; found by the owner's 2-minute test on 2026-10-24).

Rules under test: a modified function or method whose body changed carries the lines that differ, before and now, each with its
line number and the function's own indentation; the panel prints them as "Before" / "Now" rows right under the title, capped, with
control characters escaped; the Changes status line says how many changes are worth looking at, so a reader knows to scroll.
"""

from __future__ import annotations

import asyncio

from argus.explore.changed_lines import MAX_COMPARE, MAX_LINES, ChangedLine, diff_lines, function_diff
from argus.explore.changes import Flag, SymbolChange
from argus.explore.render_compact import MAX_ROWS, changes_summary
from argus.explore.render_panel import LABEL, MAX_CHANGED_SHOWN, panel_lines
from tests.explore.diff_helpers import GitRepo
from tests.explore.test_diff_view import build_view, change
from tests.explore.test_tui import SIZE, _app, _text, _until
from tests.explore.test_watch import BILLING

# -- the pure line comparison ----------------------------------------------------------------------------------------------------


def test_a_replaced_line_is_one_line_before_and_one_now_with_their_own_line_numbers() -> None:
    before = ["def f(a):", "    x = a", "    return x"]
    after = ["def f(a):", "    # why", "    x = a + 1", "    return x"]

    found = diff_lines(before, 10, after, 20)

    assert found == (ChangedLine("-", "    x = a", 11), ChangedLine("+", "    # why", 21), ChangedLine("+", "    x = a + 1", 22))


def test_an_inserted_line_has_only_a_now_row_and_a_deleted_line_only_a_before_row() -> None:
    assert diff_lines(["a", "b"], 1, ["a", "new", "b"], 1) == (ChangedLine("+", "new", 2),)
    assert diff_lines(["a", "old", "b"], 1, ["a", "b"], 1) == (ChangedLine("-", "old", 2),)


def test_identical_text_has_no_changed_lines() -> None:
    assert diff_lines(["a", "b"], 1, ["a", "b"], 5) == ()


def test_trailing_spaces_are_not_a_change_but_a_different_indentation_is() -> None:
    assert diff_lines(["    x = 1   "], 1, ["    x = 1"], 1) == ()
    assert diff_lines(["    x = 1"], 1, ["        x = 1"], 1) != ()  # moving a line into a block changes what the code does


def test_the_changed_lines_are_capped() -> None:
    before = [f"old {i}" for i in range(MAX_LINES * 2)]
    after = [f"new {i}" for i in range(MAX_LINES * 2)]

    assert len(diff_lines(before, 1, after, 1)) == MAX_LINES


# -- on a real repository --------------------------------------------------------------------------------------------------------

CORE = "def calc(a, b):\n    total = a + b\n    return total\n\n\ndef steady(x):\n    return x\n"


def _repo(repo: GitRepo, new_core: str) -> None:
    repo.write_files({"m/__init__.py": "", "m/core.py": CORE})
    repo.commit("base")
    repo.write("m/core.py", new_core)


def test_a_body_edit_carries_the_lines_before_and_now(git_repo: GitRepo) -> None:
    _repo(git_repo, CORE.replace("    total = a + b\n", "    # keep it simple\n    total = a + b + 1\n"))

    calc = change(build_view(git_repo), "m.core.calc")

    assert [(c.kind, c.text, c.line) for c in calc.changed_lines] == [("-", "    total = a + b", 2), ("+", "    # keep it simple", 2), ("+", "    total = a + b + 1", 3)]


def test_a_method_keeps_its_nesting_but_not_the_indent_of_its_class(git_repo: GitRepo) -> None:
    git_repo.write_files({"m/__init__.py": "", "m/k.py": "class K:\n    def go(self, n):\n        if n:\n            return 1\n        return 0\n"})
    git_repo.commit("base")
    git_repo.write("m/k.py", "class K:\n    def go(self, n):\n        if n > 1:\n            return 1\n        return 0\n")

    go = change(build_view(git_repo), "m.k.K.go")

    assert [(c.kind, c.text) for c in go.changed_lines] == [("-", "    if n:"), ("+", "    if n > 1:")]


def test_a_docstring_only_edit_shows_the_docstring_lines(git_repo: GitRepo) -> None:
    git_repo.write_files({"m/__init__.py": "", "m/d.py": 'def f():\n    """Old words."""\n    return 1\n'})
    git_repo.commit("base")
    git_repo.write("m/d.py", 'def f():\n    """New words."""\n    return 1\n')

    f = change(build_view(git_repo), "m.d.f")

    assert [(c.kind, c.text.strip()) for c in f.changed_lines] == [("-", '"""Old words."""'), ("+", '"""New words."""')]


def test_a_signature_only_edit_has_no_changed_lines_because_before_and_now_already_show_it(git_repo: GitRepo) -> None:
    _repo(git_repo, CORE.replace("def calc(a, b):", "def calc(a, b=0):"))

    calc = change(build_view(git_repo), "m.core.calc")

    assert Flag.SIGNATURE_CHANGED in calc.flags and calc.changed_lines == ()


def test_added_and_removed_functions_have_no_changed_lines(git_repo: GitRepo) -> None:
    _repo(git_repo, CORE.replace("def steady(x):\n    return x\n", "def fresh(x):\n    return x\n"))
    view = build_view(git_repo)

    assert change(view, "m.core.fresh").changed_lines == () and change(view, "m.core.steady").changed_lines == ()


def test_the_changed_lines_do_not_reach_the_json_document(git_repo: GitRepo) -> None:
    from argus.explore.jsonout import diff_json

    _repo(git_repo, CORE.replace("total = a + b", "total = a - b"))

    document = diff_json(build_view(git_repo))

    assert "changed_lines" not in str(document)  # schema v1 is unchanged


# -- the panel -------------------------------------------------------------------------------------------------------------------


def _sym(lines: tuple[ChangedLine, ...], change_kind: str = "modified") -> SymbolChange:
    return SymbolChange(
        "billing.charge", "function", change_kind, frozenset({Flag.BODY_CHANGED}), None, None, False, "billing/charge.py", 12,
        known_callers=2, test_callers=1, changed_lines=lines,  # type: ignore[arg-type]
    )


def test_the_panel_prints_before_and_now_rows_with_line_numbers_right_under_the_title() -> None:
    lines = panel_lines(_sym((ChangedLine("-", "    return a", 13), ChangedLine("+", "    return a + 1", 13))), None)

    assert lines[1] == f"{'Before':<{LABEL}}L13  " + "    return a"
    assert lines[2] == f"{'Now':<{LABEL}}L13  " + "    return a + 1"
    assert lines[3].startswith("Who uses it") or lines[3].startswith("Used by")


def test_a_long_change_shows_a_few_rows_and_says_how_many_more() -> None:
    many = tuple(ChangedLine("-" if i % 2 else "+", f"line {i}", i + 1) for i in range(MAX_CHANGED_SHOWN + 4))

    text = "\n".join(panel_lines(_sym(many), None))

    assert f"line {MAX_CHANGED_SHOWN - 1}" in text and f"line {MAX_CHANGED_SHOWN}" not in text
    assert "+4 more changed lines" in text


def test_a_very_long_line_is_cut() -> None:
    text = "\n".join(panel_lines(_sym((ChangedLine("+", "x" * 500, 1),)), None))

    assert "x" * 500 not in text and "…" in text


def test_control_characters_in_the_changed_text_are_escaped() -> None:
    text = "\n".join(panel_lines(_sym((ChangedLine("+", "\x1b[2Jboom", 1),)), None))

    assert "\x1b" not in text and "\\x1b" in text


def test_no_changed_lines_adds_no_rows_and_an_added_symbol_never_shows_any() -> None:
    plain = panel_lines(_sym(()), None)
    added = panel_lines(_sym((ChangedLine("+", "def x():", 1),), change_kind="added"), None)

    assert not any(line.startswith(("Before", "Now")) for line in plain)
    assert not any(line.startswith(("Before", "Now")) for line in added)


# -- the count in the status line ------------------------------------------------------------------------------------------------


def _change(qualname: str, change_kind: str, flags: frozenset[Flag], file: str = "m/x.py", docstring_only: bool = False) -> SymbolChange:
    return SymbolChange(qualname, "function", change_kind, flags, None, None, False, file, 1, docstring_only=docstring_only)  # type: ignore[arg-type]


class _View:
    def __init__(self, *symbols: SymbolChange, file_changes: tuple = (), renames: tuple = ()) -> None:
        self.symbols = symbols
        self.file_changes = file_changes
        self.renames = renames


def test_the_summary_counts_the_changes_worth_looking_at_by_label_and_the_test_changes() -> None:
    view = _View(
        _change("a", "modified", frozenset({Flag.SIGNATURE_CHANGED})), _change("b", "modified", frozenset({Flag.BODY_CHANGED})),
        _change("c", "added", frozenset()), _change("test_a", "added", frozenset(), file="tests/test_x.py"), _change("test_b", "added", frozenset(), file="tests/test_x.py"),
    )

    assert changes_summary(view) == "2 changes to look at (1 CHECK, 1 LOOK) · 1 LOW · 2 test functions changed"  # type: ignore[arg-type]


def test_the_summary_adds_the_scroll_hint_only_when_there_is_more_than_one_row_to_find() -> None:
    two = _View(_change("a", "modified", frozenset({Flag.BODY_CHANGED})), _change("b", "modified", frozenset({Flag.BODY_CHANGED})))
    one = _View(_change("a", "modified", frozenset({Flag.BODY_CHANGED})))

    assert changes_summary(two, scroll_hint=True).endswith("↑↓ shows the rest")  # type: ignore[arg-type]
    assert "↑↓" not in changes_summary(one, scroll_hint=True)  # type: ignore[arg-type]
    assert "↑↓" not in changes_summary(two)  # type: ignore[arg-type]


def test_the_summary_is_singular_for_one_change_and_empty_when_nothing_changed() -> None:
    one = _View(_change("a", "modified", frozenset({Flag.BODY_CHANGED})))

    assert changes_summary(one) == "1 change to look at (1 LOOK)"  # type: ignore[arg-type]
    assert changes_summary(_View()) == ""  # type: ignore[arg-type]


def test_only_low_changes_say_there_is_nothing_to_look_at() -> None:
    view = _View(_change("c", "added", frozenset()), _change("d", "modified", frozenset({Flag.BODY_CHANGED}), docstring_only=True))

    assert changes_summary(view) == "no function changes to look at · 2 LOW"  # type: ignore[arg-type]


def test_the_tui_status_says_how_many_changes_to_look_at_and_the_panel_shows_before_and_now(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2") + "\n\ndef extra():\n    return 1\n")
            await _until(pilot, lambda: "to look at" in _text(app, "status"), "the count")
            status = _text(app, "status")
            assert "1 change to look at (1 LOOK)" in status and "1 LOW" in status
            await _until(pilot, lambda: "Now" in _text(app, "panel"), "the changed lines in the panel")
            panel = _text(app, "panel")
            assert "amount + 1" in panel and "amount + 2" in panel
            await pilot.press("q")

    asyncio.run(scenario())


def test_a_small_window_keeps_room_for_the_list_and_still_shows_the_count(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=(48, 28)) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2") + "\n\ndef extra():\n    return 1\n")
            await _until(pilot, lambda: "to look at" in _text(app, "status"), "the count")
            await _until(pilot, lambda: "Now" in _text(app, "panel"), "the panel")
            await pilot.pause(0.2)
            assert app.query_one("#rows").size.height >= 5  # a whole wrapped row and part of the next, not squeezed out by the panel
            await pilot.press("q")

    asyncio.run(scenario())


# -- after the audit (2026-10-24) ------------------------------------------------------------------------------------------------


def test_a_column_zero_line_inside_a_string_does_not_shift_the_whole_function(git_repo: GitRepo) -> None:
    git_repo.write_files({"m/__init__.py": "", "m/k.py": "class K:\n    def go(self):\n        a = 1\n        return a\n"})
    git_repo.commit("base")
    git_repo.write("m/k.py", 'class K:\n    def go(self):\n        a = """\nsql\n"""\n        return a\n')

    go = change(build_view(git_repo), "m.k.K.go")

    assert [(c.kind, c.text) for c in go.changed_lines] == [("-", "    a = 1"), ("+", '    a = """'), ("+", "sql"), ("+", '"""')]


def test_tabs_are_shown_as_spaces_and_a_very_long_line_is_cut() -> None:
    tabbed = diff_lines(["def f():", "\treturn 1"], 1, ["def f():", "\treturn 2"], 1)
    long_line = diff_lines(["x"], 1, ["y" * 1000], 1)

    assert [c.text for c in tabbed] == ["    return 1", "    return 2"]
    assert len(long_line[1].text) <= 301 and long_line[1].text.endswith("…")


def test_a_function_too_large_to_compare_gives_a_note_and_no_lines() -> None:
    big = (1, [f"line {i}" for i in range(MAX_COMPARE)])

    lines, note = function_diff(big, (1, ["other"] * MAX_COMPARE))

    assert lines == () and "too large" in note


def test_a_body_that_differs_only_inside_the_syntax_tree_says_so_instead_of_showing_nothing() -> None:
    lines, note = function_diff((1, ["def f():", "    return 1"]), (1, ["def f():", "    return 1"]))

    assert lines == () and "look the same" in note


def test_an_unreadable_side_gives_a_note() -> None:
    assert function_diff(None, (1, ["x"])) == ((), "the old or the new text could not be read")
    assert function_diff((1, ["x"]), None) == ((), "the old or the new text could not be read")


def test_a_normal_edit_gives_lines_and_no_note() -> None:
    lines, note = function_diff((1, ["a", "b"]), (1, ["a", "c"]))

    assert [c.text for c in lines] == ["b", "c"] and note == ""


def test_the_panel_prints_the_note_when_there_are_no_lines_to_show() -> None:
    change = SymbolChange(
        "billing.charge", "function", "modified", frozenset({Flag.BODY_CHANGED}), None, None, False, "billing/charge.py", 12,
        known_callers=0, test_callers=0, changed_note="this function is too large to compare here",  # type: ignore[arg-type]
    )

    assert f"{'Changed':<{LABEL}}this function is too large to compare here" in panel_lines(change, None)


def test_a_long_replacement_still_shows_both_sides() -> None:
    block = tuple(ChangedLine("-", f"old {i}", i + 1) for i in range(7)) + tuple(ChangedLine("+", f"new {i}", i + 1) for i in range(7))

    text = "\n".join(panel_lines(_sym(block), None))

    assert "Before" in text and "Now" in text  # not only the first six old lines
    assert "+8 more changed lines" in text


def test_a_pure_deletion_fills_the_rows_with_old_lines() -> None:
    block = tuple(ChangedLine("-", f"old {i}", i + 1) for i in range(10))

    text = "\n".join(panel_lines(_sym(block), None))

    assert text.count("Before") == MAX_CHANGED_SHOWN and "Now" not in text


def test_when_the_cap_was_hit_the_count_of_more_lines_says_at_least() -> None:
    block = tuple(ChangedLine("+", f"new {i}", i + 1) for i in range(MAX_LINES))

    text = "\n".join(panel_lines(_sym(block), None))

    assert "at least" in text


def test_one_more_changed_line_is_singular() -> None:
    block = tuple(ChangedLine("+", f"new {i}", i + 1) for i in range(MAX_CHANGED_SHOWN + 1))

    assert "+1 more changed line (" in "\n".join(panel_lines(_sym(block), None))


def test_the_summary_counts_test_code_apart_from_config_code_and_adds_file_level_changes() -> None:
    view = _View(
        _change("a", "modified", frozenset({Flag.BODY_CHANGED})), _change("test_a", "added", frozenset(), file="tests/test_x.py"),
        _change("load", "modified", frozenset({Flag.BODY_CHANGED}), file="settings.py"),
        file_changes=("x",), renames=("y",),
    )

    assert changes_summary(view) == "1 change to look at (1 LOOK) · 1 test function changed · 1 in config or generated code · 2 file-level changes"  # type: ignore[arg-type]


def test_the_scroll_hint_points_to_the_long_report_when_more_rows_exist_than_the_list_shows() -> None:
    many = _View(*(_change(f"f{i}", "modified", frozenset({Flag.BODY_CHANGED})) for i in range(MAX_ROWS + 3)))

    assert changes_summary(many, scroll_hint=True).endswith(f"↑↓ shows {MAX_ROWS}, press d for all")  # type: ignore[arg-type]
