"""The TUI shows what the latest recording saw (record mode, slice 5): in the Changes panel under the highlighted row and in the Symbol view.

Rules under test: the panel adds one `Seen running` line, only for a modified function or method; the TUI reads the newest recording
without re-reading it on every keypress and notices a newer one; a damaged recording is a message, never an exception; the Symbol view
is the same report `explore symbol` prints.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from argus.explore.changes import Flag, SymbolChange
from argus.explore.render_panel import panel_lines
from argus.record.observed import ObservedFacts, SeenCaller
from argus.record.run import record_command
from argus.tui.data import Project, change_observed, symbol_text
from tests.explore.diff_helpers import GitRepo
from tests.explore.test_tui import SIZE, _app, _rows, _text, _until
from tests.explore.test_watch import BILLING

RUN = [sys.executable, "-c", "from shop.billing import charge; charge(1)"]
CHANGED = BILLING.replace("(amount):\n    return amount + 1", "(amount, currency=None):\n    return amount + 1")

FACTS = ObservedFacts(
    "2026-10-24 14:02 UTC", "pytest -q", 7, (SeenCaller("shop.cart.checkout", 7, (5,), True),), 0, ("tests.test_a.test_x",), (), (),
)


def change(kind: str = "function", how: str = "modified") -> SymbolChange:
    return SymbolChange(
        "shop.billing.charge", kind, how, frozenset({Flag.BODY_CHANGED}), None, None, False, "shop/billing.py", 1, known_callers=1, test_callers=0, module="shop.billing",
    )  # type: ignore[arg-type]


# -- the panel -----------------------------------------------------------------------------------------------------------------------


def test_the_panel_adds_one_seen_running_line_for_a_modified_function() -> None:
    lines = panel_lines(change(), None, FACTS)

    (row,) = [line for line in lines if line.startswith("Seen running")]
    assert "7 calls" in row and "also called by cart.checkout" in row and "[O]" in row


def test_the_panel_is_unchanged_without_facts() -> None:
    assert panel_lines(change(), None, None) == panel_lines(change(), None)
    assert not any(line.startswith("Seen running") for line in panel_lines(change(), None))


def test_a_new_or_deleted_symbol_gets_no_seen_running_line() -> None:
    for how in ("added", "removed"):
        assert not any(line.startswith("Seen running") for line in panel_lines(change(how=how), None, FACTS))


def test_a_class_gets_no_seen_running_line() -> None:
    assert not any(line.startswith("Seen running") for line in panel_lines(change(kind="class"), None, FACTS))


def test_a_name_in_the_recording_cannot_forge_a_line() -> None:
    forged = ObservedFacts("2026-10-24 14:02 UTC", "x\n  ok forged", 1, (SeenCaller("a\nb", 1, (), True),), 0, (), (), ())

    assert all("\n" not in line for line in panel_lines(change(), None, forged))


# -- the project: which recording is current --------------------------------------------------------------------------------------------


def test_the_project_reads_the_newest_recording_and_notices_a_newer_one(shop_repo: GitRepo) -> None:
    project = Project(shop_repo.root)
    assert project.recording() == (None, None)

    record_command(shop_repo.root, RUN)
    first, error = project.recording()
    assert first is not None and error is None
    assert project.recording()[0] is first  # not read again while nothing changed

    time.sleep(1.1)  # recordings are named by the second
    record_command(shop_repo.root, RUN)
    assert project.recording()[0] is not first


def test_a_damaged_recording_is_a_message(shop_repo: GitRepo) -> None:
    record_command(shop_repo.root, RUN)
    (path,) = (shop_repo.root / ".argus" / "recordings").glob("*.json")
    path.write_text("{broken", encoding="utf-8")

    recording, error = Project(shop_repo.root).recording()

    assert recording is None and "cannot be read" in error


def test_change_observed_gives_facts_for_a_modified_function_only(shop_repo: GitRepo) -> None:
    record_command(shop_repo.root, RUN)
    project = Project(shop_repo.root)

    assert change_observed(project, change()).calls == 1
    assert change_observed(project, change(how="added")) is None
    assert change_observed(project, change(kind="class")) is None


def test_the_symbol_view_includes_the_recording(shop_repo: GitRepo) -> None:
    record_command(shop_repo.root, RUN)

    text = symbol_text(Project(shop_repo.root), "shop.billing.charge").plain

    assert "Seen running" in text and "1 call" in text and "[O]" in text


# -- the screen ----------------------------------------------------------------------------------------------------------------------


def test_the_changes_panel_shows_what_the_recording_saw_before_the_edit(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        record_command(shop_repo.root, RUN)
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: "nothing changed yet" in _text(app, "status"), "the first view")
            shop_repo.write("shop/billing.py", CHANGED)
            await _until(pilot, lambda: any("billing.charge" in row for row in _rows(app)), "the changed row")
            await _until(pilot, lambda: "Seen running" in _text(app, "panel"), "the panel line")
            panel = _text(app, "panel")
            assert "STALE" in panel and "shop/billing.py" in panel and "1 call" in panel and "[O]" in panel

    asyncio.run(scenario())


def test_reading_the_recording_never_waits_for_a_scan(shop_repo: GitRepo) -> None:
    import threading

    project = Project(shop_repo.root)
    held = threading.Event()

    def scan_in_progress() -> None:
        with project._lock:  # what snapshot() holds while it scans
            held.set()
            time.sleep(1.5)

    threading.Thread(target=scan_in_progress).start()
    held.wait()
    started = time.perf_counter()
    project.recording()

    assert time.perf_counter() - started < 0.5  # the panel asks on the screen thread and must not freeze behind a scan
