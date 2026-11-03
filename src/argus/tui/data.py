"""What the TUI shows, as text: the same reports the CLI prints, rendered once into `rich` text.

No analysis lives here. Each function calls the code behind one CLI command and returns the report; failures come back as a
message to show, never as an exception, so a bad name or a ref typo can never take the screen down.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

from rich.console import Console
from rich.text import Text

from argus.explore.ascii import render_deps
from argus.explore.blast import BlastTree, build_blast_tree
from argus.explore.changes import SymbolChange
from argus.explore.compare import compare_refs
from argus.explore.diagram import build_levels
from argus.explore.entrypoints import Entrypoints, find_entrypoints
from argus.explore.reach import reached_displays
from argus.explore.render_panel import ReachNote
from argus.explore.symbols import unresolved_note
from argus.explore.render_diff import diff_lines
from argus.explore.render_symbol import print_symbol
from argus.explore.sides import DiffError, python_file_stamps
from argus.explore.symbols import build_symbol_view, resolve_symbol
from argus.extract import scan
from argus.extract.resolve_calls import resolve_v2_enabled
from argus.model.schema import ArchModel
from argus.record.observed import ObservedFacts, latest_recording, observed_for
from argus.record.store import DIRECTORY, Recording

REPORT_WIDTH = 100


class Project:
    """The live working tree: scanned on demand, and rescanned only when a `.py` file changed since the last scan."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self._lock = threading.Lock()
        self._stamps: dict[str, tuple[int, int] | None] | None = None
        self._cached: tuple[ArchModel, Entrypoints] | None = None
        self._recording_lock = threading.Lock()  # not `_lock`: the screen thread asks for the recording while another thread holds `_lock` to scan
        self._recording_key: tuple[str, int] | None = None
        self._recording: tuple[Recording | None, str | None] = (None, None)

    def snapshot(self) -> tuple[ArchModel, Entrypoints]:
        with self._lock:
            stamps = python_file_stamps(self.path)
            if self._cached is None or stamps != self._stamps:
                model = scan(self.path, resolve_v2=resolve_v2_enabled())
                self._cached, self._stamps = (model, find_entrypoints(self.path, model)), stamps
            return self._cached


    def recording(self) -> tuple[Recording | None, str | None]:
        """The newest recording, or the message saying why it cannot be read. Read again only when a different file is the newest."""
        folder = self.path / DIRECTORY
        try:
            newest = max(folder.glob("*.json"), default=None)
            key = (newest.name, newest.stat().st_mtime_ns) if newest else None
        except OSError:
            key = None
        with self._recording_lock:
            if key != self._recording_key or (key is None and self._recording != (None, None)):
                self._recording, self._recording_key = latest_recording(self.path), key
            return self._recording


def change_observed(project: Project, change: SymbolChange) -> ObservedFacts | None:
    """What the latest recording saw of a modified function or method (an added one cannot have run, a removed one is gone)."""
    recording, _ = project.recording()
    if recording is None or change.change != "modified" or change.kind not in ("function", "method"):
        return None
    known = {u.caller for u in change.users if not u.is_test} if change.known_callers is not None else None
    return observed_for(recording, change.node_id, change.file, project.path, known)


def blast_tree(project: Project) -> tuple[BlastTree, tuple[int, int]]:
    """The blast-radius tree of the working tree with the latest recording, and a key that changes when either the scan or the recording does."""
    model, _eps = project.snapshot()
    recording, _error = project.recording()
    return build_blast_tree(model, project.path, recording), (id(model), id(recording))


def _capture(draw, width: int = REPORT_WIDTH) -> Text:
    buffer = io.StringIO()
    draw(Console(file=buffer, width=width, force_terminal=True, color_system="standard", highlight=False, emoji=False))
    return Text.from_ansi(buffer.getvalue().rstrip("\n"))


def plain(lines: list[str]) -> Text:
    return Text("\n".join(lines), no_wrap=True, overflow="ignore")


def map_text(project: Project, width: int = REPORT_WIDTH) -> Text:
    try:
        model, eps = project.snapshot()
        return plain(render_deps(build_levels(model, project.path, eps), False))
    except (DiffError, OSError, ValueError) as err:
        return Text(str(err))


def symbol_text(project: Project, query: str, width: int = REPORT_WIDTH) -> Text:
    if not query.strip():
        return Text("Type a function, class or method name and press Enter.")
    try:
        model, _eps = project.snapshot()
        recording, recording_error = project.recording()
        view = build_symbol_view(model, project.path, resolve_symbol(model, query.strip()), recording=recording, recording_error=recording_error)
    except (LookupError, DiffError, OSError) as err:
        return Text(str(err))
    return _capture(lambda console: print_symbol(console, view), width)


def diff_text(project: Project, base: str, head: str, width: int = REPORT_WIDTH) -> Text:
    try:
        view = compare_refs(project.path, base.strip() or "HEAD", head.strip() or None)
    except DiffError as err:
        return Text(str(err))
    return plain(diff_lines(view))


def reach_notes(project: Project, view: DiffView) -> dict[str, ReachNote]:
    """For every modified function or method of the view: the commands and routes that reach it. A failed scan is a note on each of them."""
    targets = {c.node_id for c in view.symbols if c.change == "modified" and c.kind in ("function", "method")}
    if not targets:
        return {}
    try:
        model, eps = project.snapshot()
        entries = (*eps.commands, *eps.routes, *eps.programs)
        found = reached_displays(model, project.path, entries, targets)
        caveat = unresolved_note(eps)
    except Exception as err:  # noqa: BLE001 - reach is an extra: it must never take the Changes view down, and must not read as "not reached"
        return {t: ReachNote((), 0, f"the entry-point scan failed ({type(err).__name__}), so reach is not known") for t in targets}
    return {t: ReachNote(found.get(t, ()), len(entries), caveat) for t in targets}
