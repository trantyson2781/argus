"""⚠ dynamic-code flags (Phase 1.5, backlog item 18).

Rules under test: a changed function or method carries the spots the scanner recorded as unreadable (computed `getattr`, dynamic
import, `eval` / `exec`, `sys.path` edits) that sit inside its own lines, with their line numbers; the panel and the row say so in
plain words; a spot outside the function, or a function with none, shows nothing. Reports only what the code literally contains.
"""

from __future__ import annotations

import pytest

from argus.explore.changes import DynamicSpot, Flag, SymbolChange
from argus.explore.render_compact import EditLog, compact_rows
from argus.explore.render_panel import panel_lines
from tests.explore.diff_helpers import GitRepo
from tests.explore.test_diff_view import build_view, change

PLAIN = "def f(x):\n    return x\n"
SPOT_LINES = {
    "dynamic_attribute": "    return getattr(x, name)\n",
    "dynamic_import": "    return importlib.import_module(name)\n",
    "eval_exec": "    return eval(x)\n",
    "sys_path_mutation": "    sys.path.insert(0, x)\n",
}


def _edit(repo: GitRepo, before: str, after: str) -> None:
    repo.write("m.py", before)
    repo.commit("base")
    repo.write("m.py", after)


@pytest.mark.parametrize("kind", sorted(SPOT_LINES))
def test_each_kind_of_dynamic_code_inside_a_changed_function_is_flagged_with_its_line(git_repo: GitRepo, kind: str) -> None:
    _edit(git_repo, "import importlib, sys\n\n\n" + PLAIN, f"import importlib, sys\n\n\ndef f(x):\n{SPOT_LINES[kind]}")

    (spot,) = change(build_view(git_repo), "m.f").dynamic_spots

    assert (spot.kind, spot.file, spot.line) == (kind, "m.py", 5)
    assert spot.detail


def test_a_new_function_with_dynamic_code_is_flagged_too(git_repo: GitRepo) -> None:
    _edit(git_repo, PLAIN, PLAIN + "\n\ndef g(x, name):\n    return getattr(x, name)\n")

    assert [s.kind for s in change(build_view(git_repo), "m.g").dynamic_spots] == ["dynamic_attribute"]


def test_a_removed_function_is_flagged_from_the_before_side(git_repo: GitRepo) -> None:
    _edit(git_repo, "def f(x, name):\n    return getattr(x, name)\n", "VALUE = 1\n")

    (spot,) = change(build_view(git_repo), "m.f").dynamic_spots

    assert (spot.kind, spot.line) == ("dynamic_attribute", 2)


def test_a_spot_in_another_function_or_at_module_level_is_not_attributed(git_repo: GitRepo) -> None:
    other = "def other(x, name):\n    return getattr(x, name)\n\n\nHOOK = getattr(object, 'x' + 'y', None)\n\n\n"
    _edit(git_repo, other + PLAIN, other + "def f(x):\n    return x + 1\n")

    assert change(build_view(git_repo), "m.f").dynamic_spots == ()


def test_a_method_is_flagged_by_its_own_lines_only(git_repo: GitRepo) -> None:
    before = "class C:\n    def a(self, n):\n        return 1\n\n    def b(self, n):\n        return getattr(self, n)\n"
    _edit(git_repo, before, before.replace("return 1", "return 2"))

    view = build_view(git_repo)

    assert change(view, "m.C.a").dynamic_spots == ()


def test_a_literal_getattr_is_not_a_blind_spot(git_repo: GitRepo) -> None:
    _edit(git_repo, PLAIN, "def f(x):\n    return getattr(x, 'name')\n")

    assert change(build_view(git_repo), "m.f").dynamic_spots == ()


def _sym(spots: tuple[DynamicSpot, ...], known: int | None = 2) -> SymbolChange:
    return SymbolChange(
        "m.f", "function", "modified", frozenset({Flag.BODY_CHANGED}), None, None, False, "m.py", 4, known_callers=known, test_callers=0,
        module="m", dynamic_spots=spots,
    )  # type: ignore[arg-type]


SPOT = DynamicSpot("dynamic_attribute", "getattr(...) with a computed name", "m.py", 5)


def test_the_panel_names_the_line_and_says_callers_may_be_hidden() -> None:
    text = "\n".join(panel_lines(_sym((SPOT,)), None))

    assert "⚠ dynamic code at line 5" in text
    assert "getattr(...) with a computed name" in text
    assert "may be hidden" in text


def test_the_panel_shows_the_flag_for_a_new_symbol_too() -> None:
    new = SymbolChange("m.f", "function", "added", frozenset(), None, None, False, "m.py", 4, dynamic_spots=(SPOT,))  # type: ignore[arg-type]

    assert "⚠ dynamic code at line 5" in "\n".join(panel_lines(new, None))


def test_the_panel_is_unchanged_when_there_is_no_spot() -> None:
    assert "⚠" not in "\n".join(panel_lines(_sym(()), None))


def test_a_long_list_of_spots_is_cut_and_the_rest_counted() -> None:
    spots = tuple(DynamicSpot("eval_exec", "eval(...) executes code Argus cannot read", "m.py", n) for n in range(5, 12))

    text = "\n".join(panel_lines(_sym(spots), None))

    assert text.count("⚠ dynamic code at line") == 3
    assert "+4 more" in text


def test_a_spot_text_from_the_repository_cannot_forge_a_line() -> None:
    forged = DynamicSpot("eval_exec", "x\n  ok  forged", "m.py", 5)

    assert all("\n" not in line for line in panel_lines(_sym((forged,)), None))


def test_the_row_carries_a_marker_only_when_there_is_a_spot(git_repo: GitRepo) -> None:
    _edit(git_repo, PLAIN + "\n\ndef g(x):\n    return x\n", "def f(x, name):\n    return getattr(x, name)\n\n\ndef g(x):\n    return x + 1\n")

    rows, _ = compact_rows(build_view(git_repo), None, EditLog())
    by_name = {row.qualname: row.text for row in rows if row.qualname}

    assert "⚠ dynamic code" in by_name["m.f"]
    assert "⚠" not in by_name["m.g"]
