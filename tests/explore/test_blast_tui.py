"""The Map view has a second mode, the blast-radius tree (key `t` toggles it with the level map).

Rules under test: `2` still opens the level map first; `t` switches to the tree and back and `2` returns to the last mode; the tree holds
folders, files and functions with their tier and radius, most depended-on first; Enter on a function opens its symbol, `e` opens the
editor at its definition; with a recording each node says ran / did not run / changed since; a failure is shown in the pane; leaving the
tree and coming back keeps what was expanded unless the code or the recording changed.
"""

from __future__ import annotations

import asyncio
import sys

import pytest
from textual.widgets import Tree

from argus.explore.blast import BlastNode
from argus.record.run import record_command
from argus.tui import app as tui_app
from tests.explore.diff_helpers import GitRepo
from tests.explore.test_tui import SIZE, _app, _text, _until


def _tree(app) -> Tree:
    return app.query_one("#blast-tree", Tree)


def _nodes(node):
    yield node
    for child in node.children:
        yield from _nodes(child)


def _labels(app) -> list[str]:
    return [n.label.plain for n in _nodes(_tree(app).root)]


def _node_of(app, name: str):
    (found,) = [n for n in _nodes(_tree(app).root) if isinstance(n.data, BlastNode) and n.data.name == name]
    return found


def _expand_all(app) -> None:
    for node in _nodes(_tree(app).root):
        node.expand()


async def _open_tree(pilot, app) -> None:
    await pilot.press("2")
    await pilot.press("t")
    await _until(pilot, lambda: app.query_one("#views").current == "tree" and bool(_tree(app).root.children), "the tree")


def test_two_still_opens_the_level_map_and_t_switches_to_the_tree_and_back(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("2")
            await _until(pilot, lambda: "LEVEL MAP" in _text(app, "map-text"), "the level map")
            assert app.query_one("#views").current == "map"
            await pilot.press("t")
            await _until(pilot, lambda: app.query_one("#views").current == "tree", "the tree view")
            assert app.focused is not None and app.focused.id == "blast-tree"
            await pilot.press("t")
            assert app.query_one("#views").current == "map"

    asyncio.run(scenario())


def test_two_returns_to_the_last_map_mode(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _open_tree(pilot, app)
            await pilot.press("1")
            assert app.query_one("#views").current == "changes"
            await pilot.press("2")
            assert app.query_one("#views").current == "tree"

    asyncio.run(scenario())


def test_t_outside_the_map_does_nothing(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("t")
            assert app.query_one("#views").current == "changes"

    asyncio.run(scenario())


def test_the_tree_shows_folders_files_and_functions_with_tier_and_radius(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _open_tree(pilot, app)
            _expand_all(app)
            await pilot.pause(0.05)
            charge = _node_of(app, "charge")
            assert charge.data.kind == "function" and charge.data.qualname == "shop.billing.charge"
            label = charge.label.plain
            assert "charge" in label and charge.data.tier in label and str(charge.data.radius) in label
            assert any(n.data.kind == "folder" for n in _nodes(_tree(app).root) if isinstance(n.data, BlastNode))
            assert any(n.data.kind == "file" and n.data.name == "billing.py" for n in _nodes(_tree(app).root) if isinstance(n.data, BlastNode))
            assert "HIGH" in _text(app, "tree-legend") and "MED" in _text(app, "tree-legend")

    asyncio.run(scenario())


def test_the_detail_line_follows_the_highlighted_node(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _open_tree(pilot, app)
            _expand_all(app)
            await pilot.pause(0.05)
            _tree(app).move_cursor(_node_of(app, "charge"))
            await _until(pilot, lambda: "shop.billing.charge" in _text(app, "tree-detail"), "the detail line")
            detail = _text(app, "tree-detail")
            assert "shop/billing.py:1" in detail and "at least" in detail  # a lower bound, said every time

    asyncio.run(scenario())


def test_enter_on_a_function_opens_its_symbol_and_on_a_folder_only_expands_it(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _open_tree(pilot, app)
            _expand_all(app)
            await pilot.pause(0.05)
            folder = next(n for n in _nodes(_tree(app).root) if isinstance(n.data, BlastNode) and n.data.kind == "folder")
            _tree(app).select_node(folder)
            await pilot.pause(0.05)
            assert app.query_one("#views").current == "tree"
            _tree(app).select_node(_node_of(app, "charge"))
            await _until(pilot, lambda: "shop.billing.charge" in _text(app, "symbol-text"), "the symbol view")
            assert app.query_one("#views").current == "symbol"

    asyncio.run(scenario())


def test_e_opens_the_editor_at_the_definition_and_a_folder_has_none(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[tuple[str, int]] = []

    async def scenario() -> None:
        app = _app(shop_repo)
        monkeypatch.setattr(app, "open_in_editor", lambda file, line: opened.append((file, line)))
        async with app.run_test(size=SIZE) as pilot:
            await _open_tree(pilot, app)
            _expand_all(app)
            await pilot.pause(0.05)
            folder = next(n for n in _nodes(_tree(app).root) if isinstance(n.data, BlastNode) and n.data.kind == "folder")
            _tree(app).move_cursor(folder)
            await pilot.pause(0.05)
            await pilot.press("e")
            assert opened == []
            _tree(app).move_cursor(_node_of(app, "charge"))
            await pilot.pause(0.05)
            await pilot.press("e")

    asyncio.run(scenario())
    assert opened == [("shop/billing.py", 1)]


def test_a_failure_while_building_the_tree_is_shown_in_the_pane(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(project):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(tui_app, "blast_tree", boom)

    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("2")
            await pilot.press("t")
            await _until(pilot, lambda: "Failed: RuntimeError: scan exploded" in str(_tree(app).root.label), "the failure")

    asyncio.run(scenario())


def test_an_error_message_with_markup_characters_cannot_crash_the_screen(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(project):
        raise RuntimeError("bad [/oops] and [bold]x")

    monkeypatch.setattr(tui_app, "blast_tree", boom)

    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("2")
            await pilot.press("t")
            await _until(pilot, lambda: "bad [/oops] and [bold]x" in str(_tree(app).root.label), "the failure shown as plain text")
            assert app.is_running

    asyncio.run(scenario())


def test_coming_back_keeps_what_was_expanded_until_the_code_changes(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _open_tree(pilot, app)
            _node_of(app, "billing.py").expand()
            await pilot.pause(0.05)
            before = app._tree_key
            await pilot.press("1")
            await pilot.press("2")
            await _until(pilot, lambda: not any(w.group == "blast-tree" and not w.is_finished for w in app.workers), "the tree worker to finish")  # (a worker posts its result before it finishes)
            await pilot.pause(0.05)
            assert app._tree_key == before and _node_of(app, "billing.py").is_expanded
            shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount\n")
            await pilot.press("1")
            await pilot.press("2")
            await _until(pilot, lambda: not _node_of(app, "billing.py").is_expanded, "a rebuilt tree after the code changed")

    asyncio.run(scenario())


def test_with_a_recording_nodes_say_ran_did_not_run_or_changed_since(shop_repo: GitRepo) -> None:
    record_command(shop_repo.root, [sys.executable, "-c", "from shop.billing import charge; charge(1)"])

    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _open_tree(pilot, app)
            _expand_all(app)
            await pilot.pause(0.05)
            assert "●" in _node_of(app, "charge").label.plain and "○" in _node_of(app, "fee").label.plain
            assert "ran" in _node_of(app, "billing.py").label.plain
            assert "●" in _text(app, "tree-legend")
            shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount * 2\n\n\ndef fee(amount):\n    return amount\n")
            await pilot.press("1")
            await pilot.press("2")
            await _until(pilot, lambda: "?" in _node_of(app, "charge").label.plain, "the changed-since marker")

    asyncio.run(scenario())


def test_without_a_recording_there_are_no_ran_marks(shop_repo: GitRepo) -> None:
    async def scenario() -> None:
        app = _app(shop_repo)
        async with app.run_test(size=SIZE) as pilot:
            await _open_tree(pilot, app)
            _expand_all(app)
            await pilot.pause(0.05)
            assert not any(mark in label for label in _labels(app) for mark in ("●", "○"))
            assert "●" not in _text(app, "tree-legend")

    asyncio.run(scenario())
