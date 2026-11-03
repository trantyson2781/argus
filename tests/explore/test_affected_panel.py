"""The "who is affected" panel and the saved-time markers (Phase 1.5, product track).

Rules under test: every changed symbol that has callers computed carries who calls it (production callers and test code apart), the
module it lives in and the modules that import that module; the panel prints them as short plain-language lines, always as lower
bounds, with the one check Argus cannot do itself (`git grep`); the edit log remembers when each symbol's file was last saved.
"""

from __future__ import annotations

from argus.explore.changes import DiffView, Flag, SymbolChange, SymbolUser
from argus.explore.render_compact import EditInfo, EditLog, Touch, compact_rows
from argus.explore.render_panel import MAX_USERS, panel_lines
from tests.explore.diff_helpers import GitRepo
from tests.explore.test_diff_view import build_view, change
from tests.explore.test_review_order import scenario


def _user(caller: str, module: str, file: str = "api/checkout.py", line: int = 41, is_test: bool = False, module_level: bool = False) -> SymbolUser:
    return SymbolUser(caller, module, file, line, is_test, module_level)


def _sym(
    qualname: str = "billing.charge", change_kind: str = "modified", flags: frozenset[Flag] = frozenset({Flag.SIGNATURE_CHANGED}),
    users: tuple[SymbolUser, ...] = (), known: int | None = 0, tests: int | None = 0, kind: str = "function", module: str | None = "billing",
    imported_by: tuple[str, ...] = (), was: str | None = "(amount)", now: str | None = "(amount, currency=None)",
) -> SymbolChange:
    return SymbolChange(
        qualname, kind, change_kind, flags, None, None, False, "billing/charge.py", 12, interface_was=was, interface_now=now,
        known_callers=known, test_callers=tests, users=users, module=module, module_imported_by=imported_by,
    )  # type: ignore[arg-type]


def _text(change: SymbolChange, touch: Touch | None = None) -> str:
    return "\n".join(panel_lines(change, touch))


# -- the data, on a real repository ---------------------------------------------------------------------------------------------------


def test_users_lists_each_calling_function_once_with_its_first_call_site(git_repo: GitRepo) -> None:
    scenario(git_repo)

    users = change(build_view(git_repo), "m.core.body").users

    production = [u.caller for u in users if not u.is_test]
    assert production == ["m.use.a", "m.use.b", "m.use.c", "m.use.d"]  # c calls twice, listed once
    assert all(u.module == "m.use" and u.file == "m/use.py" and u.line > 0 for u in users if not u.is_test)


def test_test_code_that_calls_a_symbol_is_listed_apart(git_repo: GitRepo) -> None:
    scenario(git_repo)

    users = change(build_view(git_repo), "m.core.body").users

    (test,) = [u for u in users if u.is_test]
    assert test.caller.endswith("test_body") and test.file == "tests/test_use.py"


def test_a_removed_symbol_lists_the_callers_it_had_before(git_repo: GitRepo) -> None:
    scenario(git_repo)

    assert [u.caller for u in change(build_view(git_repo), "m.core.gone").users] == ["m.use.a"]


def test_an_added_symbol_has_no_users_because_none_are_computed(git_repo: GitRepo) -> None:
    scenario(git_repo)

    fresh = change(build_view(git_repo), "m.core.fresh")

    assert fresh.users == () and fresh.known_callers is None


def test_a_symbol_knows_its_module_and_the_production_modules_that_import_it(git_repo: GitRepo) -> None:
    scenario(git_repo)

    body = change(build_view(git_repo), "m.core.body")

    assert body.module == "m.core" and body.module_imported_by == ("m.use",)  # the test module imports it too but is not listed


# -- the panel text -------------------------------------------------------------------------------------------------------------------


def test_an_interface_change_shows_before_now_users_tests_module_and_the_check_to_run() -> None:
    users = (_user("api.checkout.place_order", "api.checkout"), _user("reports.monthly.build_report", "reports.monthly", "reports/monthly.py", 57))
    text = _text(_sym(users=users, known=2, imported_by=("api", "reports")))

    assert "Before   charge(amount)" in text and "Now      charge(amount, currency=None)" in text
    assert "Used by  2 places in 2 modules" in text
    assert "api.checkout" in text and "place_order" in text and "api/checkout.py:41" in text
    assert "Tests    none found" in text
    assert "Module   billing is imported by api and reports (they could be affected)" in text
    assert "obj.method()" in text and "git grep -n 'charge('" in text


def test_a_logic_edit_has_no_before_now_and_no_search_hint() -> None:
    text = _text(_sym(flags=frozenset({Flag.BODY_CHANGED}), was=None, now=None, users=(_user("a.f", "a"),), known=1))

    assert "Before" not in text and "git grep" not in text and "Used by  1 place in 1 module" in text


def test_a_long_user_list_is_cut_and_the_rest_counted() -> None:
    users = tuple(_user(f"api.m{i}.f", f"api.m{i}", f"api/m{i}.py", i + 1) for i in range(MAX_USERS + 3))

    text = _text(_sym(users=users, known=len(users)))

    assert f"Used by  {len(users)} places" in text and f"+3 more" in text and f"api.m{MAX_USERS + 2}" not in text


def test_test_callers_are_named_and_never_counted_as_users() -> None:
    users = (_user("a.f", "a"), _user("tests.test_a.test_charge", "tests.test_a", "tests/test_a.py", 9, is_test=True))

    text = _text(_sym(users=users, known=1, tests=1))

    assert "Used by  1 place in 1 module" in text and "Tests    test_charge (tests/test_a.py:9)" in text


def test_a_helper_in_test_code_is_neither_a_user_nor_a_test() -> None:
    users = (_user("tests.helpers._run", "tests.helpers", "tests/helpers.py", 3, is_test=True),)

    text = _text(_sym(users=users, known=0, tests=0))

    assert "Used by  no known users" in text and "Tests    none found" in text and "_run" not in text


def test_no_known_user_is_not_a_claim_that_there_is_none() -> None:
    text = _text(_sym(users=(), known=0, imported_by=()))

    assert "Used by  no known users" in text and "none" not in text.split("Used by")[1].split("\n")[0]
    assert "no other module imports it that Argus can see" in text


def test_a_removed_symbol_says_these_are_the_callers_it_had() -> None:
    text = _text(_sym(change_kind="removed", flags=frozenset(), was=None, now=None, users=(_user("a.f", "a"),), known=1))

    assert "Used by  1 place in 1 module (before it was deleted)" in text and "git grep -n 'charge('" in text


def test_a_new_symbol_says_its_callers_are_not_checked() -> None:
    text = _text(_sym(qualname="billing.refund", change_kind="added", flags=frozenset(), was=None, now=None, known=None, tests=None))

    assert "New function" in text and "not checked for new code" in text and "Used by" not in text


def test_a_calls_only_change_says_users_are_not_checked_for_it() -> None:
    text = _text(_sym(flags=frozenset({Flag.CALLS_CHANGED}), was=None, now=None, known=None, tests=None))

    assert "not checked for this kind of change" in text


def test_a_constructor_is_searched_by_its_class_name() -> None:
    text = _text(_sym(qualname="billing.Invoice.__init__", kind="method"))

    assert "git grep -n 'Invoice('" in text


def test_the_panel_says_when_the_file_was_saved() -> None:
    text = _text(_sym(), Touch("14:02:11", 3))

    assert "Saved    14:02:11 (save 3 this session)" in text


def test_a_name_from_the_repository_cannot_forge_a_line() -> None:
    text = _text(_sym(users=(_user("a.f\nUsed by  0 places", "a"),), known=1))

    assert "\nUsed by  0 places" not in text


# -- the edit log ---------------------------------------------------------------------------------------------------------------------


def _view(*symbols: SymbolChange) -> DiffView:
    return DiffView("HEAD", "working tree", True, 1, 0, symbols, (), (), (), 0, (), ())


def test_the_log_remembers_the_last_save_of_every_symbol_in_an_edited_file_and_counts_saves() -> None:
    log = EditLog()
    other = SymbolChange("other.thing", "function", "modified", frozenset({Flag.BODY_CHANGED}), None, None, False, "other.py", 1)
    view = _view(_sym("billing.charge"), other)

    log.record(view, EditInfo("14:00:00", ("billing/charge.py",)))
    log.record(view, EditInfo("14:05:00", ("billing/charge.py",)))

    assert log.touch_of("billing.charge") == Touch("14:05:00", 2)
    assert log.touch_of("other.thing") is None  # its file was never saved while watching


def test_the_first_view_records_nothing() -> None:
    log = EditLog()

    log.record(_view(_sym()), None)

    assert log.touch_of("billing.charge") is None


def test_rows_in_the_newest_save_are_fresh_and_show_the_time() -> None:
    log = EditLog()
    older = SymbolChange("billing.fee", "function", "modified", frozenset({Flag.BODY_CHANGED}), None, None, False, "billing/fee.py", 3)
    log.record(_view(older), EditInfo("14:00:00", ("billing/fee.py",)))
    view = _view(_sym(), older)
    log.record(view, EditInfo("14:05:00", ("billing/charge.py",)))

    rows, _ = compact_rows(view, None, log)

    charge, fee = (next(r for r in rows if r.qualname == name) for name in ("billing.charge", "billing.fee"))
    assert charge.fresh and "saved 14:05:00" in charge.text
    assert not fee.fresh and "saved 14:00:00" in fee.text


def test_rows_carry_their_change_so_a_screen_can_open_the_panel() -> None:
    rows, _ = compact_rows(_view(_sym()), None)

    assert rows[0].change is not None and rows[0].change.qualname == "billing.charge"
