"""The Changes panel names the commands and routes that reach a changed function: `Reached from  engine analyze · POST /pay  [I]`.

Rules under test: only a modified function or method gets the line; it lists at most a few names and counts the rest; "[I]" because every hop is
read from the code (an INFERENCE, a lower bound); when entry points exist but none reaches the function the line says so with a `[?]`; when the project
has none, or the function is not a function, there is no line; a failed entry-point scan is shown as unknown, never as "not reached".
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from textual.widgets import Static

from argus.explore.reach import reached_displays
from argus.explore.render_panel import MAX_REACH, ReachNote, panel_lines
from argus.explore.entrypoints import find_entrypoints
from argus.extract import scan
from argus.tui.app import ArgusApp
from argus.tui.data import Project, reach_notes
from tests.explore.diff_helpers import GitRepo
from tests.explore.test_diff_view import build_view

CORE = "def charge(x):\n    return x\n"
WEB = "from fastapi import FastAPI\nfrom p.core import charge\n\napp = FastAPI()\n\n\n@app.post('/pay')\ndef pay():\n    return charge(1)\n"
CLI = "import typer\nfrom p.core import charge\n\napp = typer.Typer()\n\n\n@app.command()\ndef run():\n    charge(2)\n"
PYPROJECT = '[project]\nname = "p"\nversion = "0"\n[project.scripts]\nshop = "p.cli:app"\n'


def _repo(repo: GitRepo) -> GitRepo:
    repo.write_files({"p/__init__.py": "", "p/core.py": CORE, "p/web.py": WEB, "p/cli.py": CLI, "pyproject.toml": PYPROJECT})
    repo.commit("base")
    repo.write("p/core.py", "def charge(x):\n    if x < 0:\n        return 0\n    return x\n")
    return repo


def _change(repo: GitRepo):
    (change,) = build_view(repo).symbols
    return change


# -- the pure part ---------------------------------------------------------------------------------------------------


def test_reached_displays_follows_calls_from_every_entry_point(git_repo: GitRepo) -> None:
    repo = _repo(git_repo)
    model = scan(repo.root)
    eps = find_entrypoints(repo.root, model)

    found = reached_displays(model, repo.root, (*eps.commands, *eps.routes), {"function:p.core.charge", "function:p.web.pay"})

    assert sorted(found["function:p.core.charge"]) == ["POST /pay", "shop run"] and found["function:p.web.pay"] == ("POST /pay",)


def test_a_function_nothing_reaches_is_missing_from_the_answer(git_repo: GitRepo) -> None:
    repo = _repo(git_repo)
    model = scan(repo.root)
    eps = find_entrypoints(repo.root, model)

    assert reached_displays(model, repo.root, (*eps.commands, *eps.routes), {"function:p.core.nothing"}) == {}


# -- the panel line ----------------------------------------------------------------------------------------------------


def _panel(git_repo: GitRepo, note: ReachNote | None) -> str:
    return "\n".join(panel_lines(_change(_repo(git_repo)), None, reach=note))


def test_the_line_lists_the_names_and_is_marked_inferred(git_repo: GitRepo) -> None:
    text = _panel(git_repo, ReachNote(("shop run", "POST /pay"), 2, None))

    assert "Reached from  shop run · POST /pay  [I]" in text


def test_a_long_list_is_cut_and_counted(git_repo: GitRepo) -> None:
    names = tuple(f"cmd{i}" for i in range(MAX_REACH + 4))

    text = _panel(git_repo, ReachNote(names, len(names), None))

    assert f"cmd{MAX_REACH - 1}" in text and f"cmd{MAX_REACH}" not in text and "+4 more" in text


def test_entry_points_exist_but_none_reaches_it(git_repo: GitRepo) -> None:
    text = _panel(git_repo, ReachNote((), 3, None))

    assert "Reached from  no command or route Argus can see reaches it  [?]" in text


def test_a_project_without_entry_points_gets_no_line(git_repo: GitRepo) -> None:
    assert "Reached from" not in _panel(git_repo, ReachNote((), 0, None))
    assert "Reached from" not in _panel(git_repo, None)


def test_a_failed_scan_is_unknown_not_not_reached(git_repo: GitRepo) -> None:
    text = _panel(git_repo, ReachNote((), 0, "the entry-point scan failed (OSError)"))

    assert "Reached from  unknown: the entry-point scan failed (OSError)  [?]" in text


def test_repository_names_in_the_line_are_escaped(git_repo: GitRepo) -> None:
    text = _panel(git_repo, ReachNote(("GET /\x1b[2J",), 1, None))

    assert "\x1b" not in text


# -- the notes for a whole view ----------------------------------------------------------------------------------------


def test_the_notes_are_computed_for_every_modified_function(git_repo: GitRepo) -> None:
    repo = _repo(git_repo)
    view = build_view(repo)

    notes = reach_notes(Project(repo.root), view)

    note = notes["function:p.core.charge"]
    assert sorted(note.commands) == ["POST /pay", "shop run"] and note.found == 2 and note.caveat is None


def test_a_failing_entry_point_scan_becomes_a_note_on_every_function(git_repo: GitRepo, monkeypatch) -> None:
    repo = _repo(git_repo)
    view = build_view(repo)
    monkeypatch.setattr("argus.tui.data.Project.snapshot", lambda self: (_ for _ in ()).throw(OSError("disk")))

    notes = reach_notes(Project(repo.root), view)

    assert "failed" in (notes["function:p.core.charge"].caveat or "") and notes["function:p.core.charge"].commands == ()


# -- on screen ---------------------------------------------------------------------------------------------------------


def test_the_panel_in_the_app_shows_the_reached_from_line(git_repo: GitRepo) -> None:
    repo = _repo(git_repo)
    seen = {}

    async def scenario() -> None:
        app = ArgusApp(repo.root, base="HEAD", poll=0.05, debounce=0.02)
        async with app.run_test(size=(140, 40)) as pilot:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and "Reached from" not in str(app.query_one("#panel", Static).content):
                await pilot.pause(0.1)
            seen["panel"] = str(app.query_one("#panel", Static).content)

    asyncio.run(scenario())

    assert "Reached from" in seen["panel"] and "POST /pay" in seen["panel"] and "shop run" in seen["panel"]


def test_reachability_does_not_read_source_files(git_repo: GitRepo, monkeypatch) -> None:
    repo = _repo(git_repo)
    model = scan(repo.root)
    eps = find_entrypoints(repo.root, model)
    monkeypatch.setattr("argus.explore.reach.evidence_at", lambda *a, **k: (_ for _ in ()).throw(AssertionError("evidence is not needed to know what is reachable")))

    found = reached_displays(model, repo.root, (*eps.commands, *eps.routes), {"function:p.core.charge"})

    assert sorted(found["function:p.core.charge"]) == ["POST /pay", "shop run"]


def test_two_displays_of_one_function_are_each_listed_once(git_repo: GitRepo) -> None:
    repo = _repo(git_repo)
    repo.write("p/web.py", WEB + "\n\n@app.get('/pay')\n@app.put('/pay')\ndef pay2():\n    return charge(3)\n")
    model = scan(repo.root)
    eps = find_entrypoints(repo.root, model)

    found = reached_displays(model, repo.root, (*eps.routes, *eps.routes), {"function:p.core.charge"})

    assert sorted(found["function:p.core.charge"]) == ["GET /pay", "POST /pay", "PUT /pay"]
