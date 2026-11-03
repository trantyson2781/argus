"""`argus explore watch` default view: only what changed. The long report (`--details`) is the old output and stays covered by test_watch.py.

Rules under test: a header names the file just edited and the time; each changed production symbol is one short row tagged NEW when it
arrived in the latest repaint; test code is one summary row; rows are capped; a file that does not parse is always visible; every
other limit is folded into one count line; nothing in the compact view claims more than the long report does.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable

from rich.console import Console
from typer.testing import CliRunner

from argus.cli import app
from argus.explore.changes import Flag, SymbolChange
from argus.explore.render_compact import MAX_ROWS, change_key
from argus.explore.watch import open_watch, run_watch
from tests.explore.diff_helpers import GitRepo
from tests.explore.test_watch import BILLING

RUNNER = CliRunner()


def _run(repo: GitRepo, script: list[Callable[[], None]]) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, highlight=False, force_terminal=False)
    calls = {"n": 0}

    def sleep(_seconds: float) -> None:
        if calls["n"] < len(script):
            script[calls["n"]]()
        calls["n"] += 1

    with open_watch(repo.root, None, False) as settings:
        run_watch(settings, console, sleep=sleep, should_stop=lambda: calls["n"] > len(script))
    return buffer.getvalue()


def _paints(out: str) -> list[str]:
    """Each repaint, from its `●` header to the next."""
    return [chunk for chunk in re.split(r"(?m)^(?=● )", out) if chunk.startswith("● ")]


def _add(repo: GitRepo, name: str, path: str = "shop/billing.py") -> Callable[[], None]:
    def edit() -> None:
        repo.write(path, (repo.root / path).read_text() + f"\n\ndef {name}():\n    return 1\n")

    return edit


def test_nothing_changed_says_so_and_stays_short(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [lambda: None])

    (paint,) = _paints(out)
    assert re.search(r"^● nothing changed yet$", paint, flags=re.MULTILINE)
    assert re.search(r"^  watching since launch \d\d:\d\d:\d\d \(earlier edits not shown\) \|", paint, flags=re.MULTILINE)
    assert "[FACT]" not in out and "Symbols (" not in out and "Not determinable" not in out
    assert len(paint.splitlines()) <= 5


def test_a_new_function_is_one_row_with_the_file_just_edited_in_the_header(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [_add(shop_repo, "extra"), lambda: None, lambda: None])

    last = _paints(out)[-1]
    assert re.search(r"^● \d\d:\d\d:\d\d  edited shop/billing\.py$", last, flags=re.MULTILINE)
    assert re.search(r"^  NEW LOW   billing\.extra +new function · saved \d\d:\d\d:\d\d$", last, flags=re.MULTILINE)
    assert "[FACT]" not in last and "Calls (" not in last


def test_a_row_is_new_only_in_the_repaint_where_it_first_appears(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [_add(shop_repo, "first"), lambda: None, _add(shop_repo, "second"), lambda: None, lambda: None])

    last = _paints(out)[-1]
    first = next(line for line in last.splitlines() if "billing.first" in line)
    second = next(line for line in last.splitlines() if "billing.second" in line)
    assert "NEW" not in first and "NEW" in second


def test_the_first_paint_marks_nothing_new_because_there_is_nothing_to_compare_with(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING)  # no change; the baseline is the working tree as launched
    out = _run(shop_repo, [lambda: None])

    assert "NEW" not in out


def test_an_interface_change_is_check_in_plain_words_with_a_lower_bound_of_users(shop_repo: GitRepo) -> None:
    edit = lambda: shop_repo.write("shop/billing.py", BILLING.replace("(amount):\n    return amount + 1", "(amount, currency=None):\n    return amount + 1"))  # noqa: E731
    out = _run(shop_repo, [edit, lambda: None, lambda: None])

    row = next(line for line in _paints(out)[-1].splitlines() if "billing.charge" in line)
    assert row.split()[1] == "CHECK" and "inputs or outputs changed" in row and re.search(r"used in at least \d+ places?", row)
    assert "SIGNATURE_CHANGED" not in row and "interface" not in row


def test_a_logic_edit_with_no_known_users_does_not_claim_there_are_none(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [lambda: shop_repo.write("shop/billing.py", BILLING.replace("return amount\n", "return amount * 2\n")), lambda: None, lambda: None])

    row = next(line for line in _paints(out)[-1].splitlines() if "billing.fee" in line)
    assert "LOOK" in row and "logic edited" in row and "no known users" in row and "none" not in row.replace("no known users", "")


def test_a_row_shows_when_its_file_was_last_saved_and_the_first_paint_shows_no_time(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [_add(shop_repo, "extra"), lambda: None, lambda: None])

    first, last = _paints(out)[0], _paints(out)[-1]
    row = next(line for line in last.splitlines() if "billing.extra" in line)
    assert re.search(r"new function · saved \d\d:\d\d:\d\d$", row)
    assert "saved" not in first


def test_test_code_is_one_summary_row_not_one_row_per_test(shop_repo: GitRepo) -> None:
    body = "".join(f"def test_{i}():\n    assert True\n\n\n" for i in range(3))
    out = _run(shop_repo, [lambda: shop_repo.write("tests/test_new.py", body), lambda: None, lambda: None])

    last = _paints(out)[-1]
    assert re.search(r"^  tests: 3 added$", last, flags=re.MULTILINE) and "test_0" not in last


def test_rows_are_capped_and_the_rest_is_counted(shop_repo: GitRepo) -> None:
    extra = MAX_ROWS + 4
    body = BILLING + "".join(f"\n\ndef f{i}():\n    return {i}\n" for i in range(extra))
    out = _run(shop_repo, [lambda: shop_repo.write("shop/billing.py", body), lambda: None, lambda: None])

    last = _paints(out)[-1]
    assert len(re.findall(r"(?m)^ +NEW LOW +billing\.f\d+", last)) == MAX_ROWS
    assert f"+{extra - MAX_ROWS} more (use --details)" in last


def test_several_files_in_one_burst_are_all_named(shop_repo: GitRepo) -> None:
    first = _add(shop_repo, "one")
    second = _add(shop_repo, "two", "shop/api.py")
    out = _run(shop_repo, [first, second, lambda: None, lambda: None])

    assert "edited shop/api.py, shop/billing.py" in _paints(out)[-1]


def test_a_file_that_does_not_parse_is_always_visible(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [lambda: shop_repo.write("shop/billing.py", "def charge(amount:\n"), lambda: None, lambda: None])

    warning = next(line for line in _paints(out)[-1].splitlines() if line.startswith("! "))
    assert "shop/billing.py" in warning and "does not parse" in warning and "nothing in it was compared" in warning


def test_every_limit_is_folded_into_one_count_line_that_points_to_details(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [_add(shop_repo, "extra"), lambda: None, lambda: None])

    last = _paints(out)[-1]
    assert re.search(r"^  \d+ things? Argus can't check \(--details\)$", last, flags=re.MULTILINE)
    assert "Dynamic dependencies" not in last


def test_the_status_line_says_when_it_started_and_how_to_start_fresh(shop_repo: GitRepo) -> None:
    out = _run(shop_repo, [lambda: None])

    assert re.search(r"refreshed \d\d:\d\d:\d\d in \d+\.\d s \| Ctrl-C quits \| restart to start fresh", out)


def test_a_ref_baseline_is_named_in_the_header(shop_repo: GitRepo) -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, highlight=False, force_terminal=False)
    with open_watch(shop_repo.root, "HEAD", False) as settings:
        run_watch(settings, console, sleep=lambda _s: None, should_stop=lambda: True)

    assert re.search(r"^  comparing with HEAD \([0-9a-f]{7}\) \|", buffer.getvalue(), flags=re.MULTILINE)


def test_the_details_flag_keeps_the_long_report(shop_repo: GitRepo) -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, highlight=False, force_terminal=False)
    with open_watch(shop_repo.root, None, False) as settings:
        run_watch(settings, console, sleep=lambda _s: None, should_stop=lambda: True, details=True)

    assert "watching | baseline:" in buffer.getvalue() and "Not determinable" in buffer.getvalue()


def test_the_command_accepts_details(shop_repo: GitRepo) -> None:
    result = RUNNER.invoke(app, ["explore", "watch", "--help"])

    assert result.exit_code == 0 and "--details" in result.stdout


def test_change_key_distinguishes_a_symbol_whose_kind_of_change_differs() -> None:
    def symbol(flag: Flag) -> SymbolChange:
        return SymbolChange("m.f", "function", "modified", frozenset({flag}), None, None, False, "m.py", 1)

    assert change_key(symbol(Flag.BODY_CHANGED)) != change_key(symbol(Flag.SIGNATURE_CHANGED))
