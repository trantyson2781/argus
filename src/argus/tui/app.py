"""The Argus terminal UI: Changes, Map, Symbol and Diff views over the existing `watch`, `map`, `symbol` and `diff` code.

A background thread consumes `watch_events` (the same stream the terminal `watch` uses) and hands each finished view to the screen,
so keys stay responsive while Argus scans. Everything shown is produced by code that already has its own tests.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import threading
import sys
import time
from collections.abc import Callable
from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import ContentSwitcher, Footer, Input, OptionList, Static, Tree
from textual.widgets.option_list import Option

from argus.explore.blast import BlastNode, BlastTree
from argus.explore.changes import DiffView
from argus.explore.render_blast import SYMBOLS, blast_detail, blast_label, blast_legend
from argus.explore.render_diff import printable
from argus.explore.render_compact import EditLog, Row, changes_summary, compact_rows, header_line, legend_lines, limits_line, stylize_label
from argus.explore.render_panel import panel_lines
from argus.explore.render_diff import diff_lines
from argus.explore.sides import DiffError
from argus.explore.watch import (
    DEBOUNCE_SECONDS, POLL_SECONDS, WatchEvent, WatchSettings, failure_text, incomplete_note, open_watch, since_text, watch_events,
)
from argus.explore.render_panel import ReachNote
from argus.tui.data import Project, blast_tree, change_observed, diff_text, map_text, plain, reach_notes, symbol_text
from argus.tui.checker import CheckJob
from argus.tui.checker import finished_text as check_text
from argus.tui.checker import question as check_question
from argus.tui.recorder import ChildJob, RecordJob, counts, finished_text, fresh, question, snapshot, test_command
from argus.tui.helptext import help_lines
from argus.tui.pythonchoice import MIN_RECORDING, PythonChoice, chosen_python, python_version
from argus.tui.tmux import agent_pane, finding_note, send_text

VSCODE_FAMILY = {"code", "codium", "cursor"}
MIN_REPORT_WIDTH = 60
STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)
TITLE = "argus   1 Changes   2 Map   3 Symbol   4 Diff    t tree/levels  d details  r fresh  R record  C check  s send  e edit  ? help  / search  b/h refs  Esc keys  q quit"


def _styled(row: Row) -> Text:
    text = Text(row.text)
    stylize_label(text)
    if row.fresh:
        text.stylize("bold")  # saved in the newest save
    return text


def editor_command(editor: str, file: str, line: int) -> list[str]:
    """The command that opens `file` at `line` in `editor` (`$VISUAL` / `$EDITOR`, which may carry flags)."""
    parts = shlex.split(editor) or ["vi"]
    if os.path.basename(parts[0]) in VSCODE_FAMILY:
        return [*parts, "-g", f"{file}:{line}"]
    return [*parts, f"+{line}", file]


class HelpScreen(ModalScreen[None]):
    """The ? screen: plain text, closed with Esc or ?."""

    BINDINGS = [Binding("escape", "dismiss", "Close"), Binding("question_mark", "dismiss", "Close")]  # Screen.dismiss closes it
    CSS = "HelpScreen { align: center middle; } #help { width: 100; max-width: 95%; height: auto; max-height: 90%; border: solid $primary; padding: 1 2; background: $surface; }"

    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self._lines = lines

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help"):
            yield Static(Text("\n".join(self._lines)))


class ArgusApp(App[None]):
    CSS = """
    #views { height: 1fr; }
    #changes, #symbol, #diff { height: 1fr; }
    #title { height: 1; background: $primary; color: $text; padding: 0 1; }
    #status { height: auto; padding: 0 1; }
    #limits { height: auto; padding: 0 1; color: $text-muted; }
    #panel { height: auto; max-height: 40%; overflow-y: auto; padding: 0 1; border-top: solid $primary; }  /* a small window scrolls the panel instead of cutting it */
    #rows { height: 1fr; min-height: 5; }  /* a small window keeps room for a row, not only for the panel */
    #details-scroll { height: 1fr; display: none; }
    #tree { height: 1fr; }
    #blast-tree { height: 1fr; }
    #tree-legend { height: auto; padding: 0 1; color: $text-muted; }
    #tree-detail { height: auto; min-height: 3; padding: 0 1; border-top: solid $primary; }
    #refs { height: 3; }
    #refs Input { width: 1fr; }
    VerticalScroll { overflow-x: auto; }
    """
    BINDINGS = [
        Binding("1", "show('changes')", "Changes"), Binding("2", "show('map')", "Map"),
        Binding("3", "show('symbol')", "Symbol"), Binding("4", "show('diff')", "Diff"),
        Binding("t", "toggle_map", "Tree/Levels"), Binding("d", "details", "Details"), Binding("r", "fresh", "Fresh start"), Binding("R", "record", "Record"), Binding("C", "check", "Check"), Binding("s", "send", "Send to agent"), Binding("e", "edit", "Edit"), Binding("question_mark", "help", "Help"),
        Binding("slash", "search", "Search"), Binding("b", "edit_ref('base')", "Base ref"), Binding("h", "edit_ref('head')", "Head ref"),
        Binding("escape", "release", "Keys back"), Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self, path: Path, *, base: str | None = None, include_tests: bool = False,
        poll: float = POLL_SECONDS, debounce: float = DEBOUNCE_SECONDS,
    ) -> None:
        super().__init__()
        self.project = Project(path)
        self.base, self.include_tests, self.poll, self.debounce = base, include_tests, poll, debounce
        self._stop = threading.Event()
        self._finished: list[threading.Event] = []  # one per watch worker: set once its temporary files are removed
        self._shown: frozenset[str] | None = None
        self._rows_by_id: dict[str, Row] = {}
        self._log = EditLog()
        self._tickets: dict[str, int] = {}  # newest lookup per pane: an older, slower result is dropped
        self._view: DiffView | None = None
        self._status = ""
        self._details = False
        self._map_mode = "map"  # what key 2 opens: the level map or the blast-radius tree
        self._tree_key: tuple[int, int] | None = None  # the scan and recording the tree on screen was built from
        self._reach_for: DiffView | None = None  # the view the notes below were computed for
        self._reach: dict[str, ReachNote] = {}  # commands and routes that reach each modified function of the view on screen (computed after the rows appear)
        self._job: ChildJob | None = None  # the `argus record` or `argus check` child, while one runs
        self._job_kind = ""  # "recording" or "check"
        self._confirming = ""  # "record" or "check" while the question is on screen: the next key answers
        self._choice: PythonChoice | None = None  # the Python the question named: the one a `y` runs
        self._record_line = ""  # the last thing R had to say; shown under the watch status until the next R

    # -- layout --------------------------------------------------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(TITLE, id="title")
        with ContentSwitcher(initial="changes", id="views"):
            with Vertical(id="changes"):
                yield Static("starting ...", id="status")
                yield OptionList(id="rows")
                yield Static(id="panel")
                with VerticalScroll(id="details-scroll"):
                    yield Static(id="details")
                yield Static(id="limits")
            with VerticalScroll(id="map"):
                yield Static(id="map-text")
            with Vertical(id="tree"):
                yield Static(id="tree-legend")
                yield Tree("scanning ...", id="blast-tree")
                yield Static(id="tree-detail")
            with Vertical(id="symbol"):
                yield Input(placeholder="function, class or method name, then Enter", id="query")
                with VerticalScroll(id="symbol-scroll"):
                    yield Static(id="symbol-text")
            with Vertical(id="diff"):
                with Horizontal(id="refs"):
                    yield Input("HEAD", placeholder="base ref", id="base")
                    yield Input(placeholder="head ref (empty: working tree)", id="head")
                with VerticalScroll(id="diff-scroll"):
                    yield Static(id="diff-text")
        yield Footer()

    def on_mount(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in STOP_SIGNALS:  # a closed terminal or `kill` ends the app like `q`, inside the event loop (no exception in a random frame)
            try:
                loop.add_signal_handler(sig, self._on_signal, sig)
            except (NotImplementedError, RuntimeError):  # not available on this platform: the default signal behaviour stays
                pass
        self._start_watch()

    def _on_signal(self, signum: int) -> None:
        self._stop.set()
        self._stop_record()
        self.exit(return_code=128 + signum)

    def on_unmount(self) -> None:
        self._stop.set()
        self._stop_record()

    # -- the watch worker ----------------------------------------------------------------------------------------------------------

    def _start_watch(self) -> None:
        self._stop.set()  # the previous worker, if any, ends and removes its temporary files
        stop = self._stop = threading.Event()
        finished = threading.Event()
        self._finished = [*(e for e in self._finished if not e.is_set()), finished]
        self._shown, self._rows_by_id, self._view, self._log = None, {}, None, EditLog()
        self.run_worker(lambda: self._watch(stop, finished), thread=True, group="watch", exit_on_error=False)

    def stop_watching(self, timeout: float = 15.0) -> None:
        """Ask every watch worker to stop and wait until its temporary files are gone."""
        self._stop.set()
        self._stop_record()
        for finished in self._finished:
            finished.wait(timeout)

    def _watch(self, stop: threading.Event, finished: threading.Event) -> None:
        try:
            with open_watch(self.project.path, self.base, self.include_tests) as settings:
                for event in watch_events(settings, poll=self.poll, debounce=self.debounce, should_stop=stop.is_set):
                    if stop.is_set():
                        return
                    self._post(self._on_event, settings, event, stop)
        except DiffError as err:
            self._post(self._set_status_if_current, stop, str(err))
        except Exception as err:  # noqa: BLE001 - a dead watcher must never look like a quiet one
            self._post(self._set_status_if_current, stop, f"WATCH STOPPED: {failure_text(err)}  (press r to restart)")
        finally:
            finished.set()

    def _post(self, function, *args) -> None:
        try:
            self.call_from_thread(function, *args)
        except RuntimeError:  # the app is already shutting down
            pass

    def _set_status(self, text: str) -> None:
        self._status = text
        self._render_status()

    def _render_status(self, extra: str = "") -> None:
        """The watch status, then a note from an event (`extra`), then what R last said."""
        lines = [line for line in (self._status, extra, self._record_line) if line]
        self.query_one("#status", Static).update(Text("\n".join(lines)))

    def _set_status_if_current(self, stop: threading.Event, text: str) -> None:
        if stop is self._stop:  # not a message from a worker that `r` has already replaced
            self._set_status(text)

    def _on_event(self, settings: WatchSettings, event: WatchEvent, stop: threading.Event) -> None:
        if stop is not self._stop:  # an event from a worker that `r` has already replaced
            return
        try:
            self._draw(settings, event)
        except Exception as err:  # noqa: BLE001 - one bad update must not end the watch; say so on screen
            self._render_status(f"screen update failed: {failure_text(err)}")

    def _start_reach(self, view: DiffView) -> None:
        """Compute which commands and routes reach the modified functions of `view` in a worker of its own, so neither the screen nor the watcher waits."""
        stop = self._stop

        def work() -> None:
            self._post(self._set_reach, stop, view, reach_notes(self.project, view))

        self.run_worker(work, thread=True, group="reach", exclusive=True, exit_on_error=False)

    def _set_reach(self, stop: threading.Event, view: DiffView, notes: dict[str, ReachNote]) -> None:
        if stop is self._stop and view is self._view:  # not for a view that a newer refresh has replaced
            self._reach, self._reach_for = notes, view
            self._show_panel()

    def _draw(self, settings: WatchSettings, event: WatchEvent) -> None:
        if event.kind != "refreshed":
            self._render_status(event.message)
            return
        self._view = event.view
        self._start_reach(event.view)
        self._log.record(event.view, event.edit)
        rows, self._shown = compact_rows(event.view, self._shown, self._log)
        lines = [header_line(event.edit, bool(self._shown)), f"  {since_text(settings)} | refreshed {time.strftime('%H:%M:%S')} in {event.seconds:.1f} s"]
        summary = changes_summary(event.view, scroll_hint=True)
        lines += [f"  {summary}"] if summary else []
        lines += [f"! {file} does not parse: nothing in it was compared" for file in event.view.failed_files]
        note = incomplete_note(settings)
        lines += [f"! {note}"] if note else []
        self._set_status("\n".join(lines))
        box = self.query_one("#rows", OptionList)
        kept = box.highlighted_option.id if box.highlighted is not None else None
        ids = [row.qualname or f"note-{i}" for i, row in enumerate(rows)]
        self._rows_by_id = dict(zip(ids, rows, strict=True))
        box.clear_options()
        box.add_options([Option(_styled(row), id=ids[i], disabled=row.qualname is None) for i, row in enumerate(rows)])
        first = next((i for i, row in enumerate(rows) if row.qualname), None)
        box.highlighted = ids.index(kept) if kept in ids else first  # the cursor stays on the symbol the user was on
        self._show_panel()
        self.query_one("#limits", Static).update(Text("\n".join([*legend_lines(event.view), limits_line(event.view)])))
        if self._details:
            self._render_details()

    # -- actions -------------------------------------------------------------------------------------------------------------------

    def action_show(self, name: str) -> None:
        if name == "map" and self._map_mode == "tree":
            name = "tree"
        self.query_one("#views", ContentSwitcher).current = name
        if name == "map":
            self._lookup("map-text", lambda width: map_text(self.project, width), "scanning ...")
        elif name == "tree":
            self._load_tree()
        elif name == "diff":
            self._diff()
        self._focus_main(name)  # never a text box: an input only takes keys when asked (`/`, `b`, `h`), so number keys always switch views

    def action_toggle_map(self) -> None:
        """t: switch the Map between the level map and the blast-radius tree."""
        current = self.query_one("#views", ContentSwitcher).current
        if current not in ("map", "tree"):
            self.notify("Open the Map first (2), then t switches between the level map and the tree.")
            return
        self._map_mode = "tree" if current == "map" else "map"
        self.action_show("map")

    def action_search(self) -> None:
        self.query_one("#views", ContentSwitcher).current = "symbol"
        self.query_one("#query", Input).focus()

    def action_edit_ref(self, which: str) -> None:
        self.action_show("diff")
        self.query_one(f"#{which}", Input).focus()

    def action_release(self) -> None:
        """Esc: leave a text box so the single-key commands work again."""
        self._focus_main(self.query_one("#views", ContentSwitcher).current or "changes")

    def _focus_main(self, view: str) -> None:
        """Give the keyboard to what the view scrolls or selects (the rows, or the report pane), so arrows and PageUp / PageDown work."""
        target = {"changes": "#rows", "map": "#map", "tree": "#blast-tree", "symbol": "#symbol-scroll", "diff": "#diff-scroll"}[view]
        self.query_one(target).focus()

    def _show_panel(self) -> None:
        """Fill the panel for the highlighted row (empty when no changed symbol is highlighted)."""
        box = self.query_one("#rows", OptionList)
        row = self._rows_by_id.get(box.highlighted_option.id) if box.highlighted is not None and box.highlighted_option else None
        change = row.change if row else None
        lines = panel_lines(change, self._log.touch_of(change.qualname), change_observed(self.project, change), self._reach.get(change.node_id)) if change else []
        panel = self.query_one("#panel", Static)
        panel.update(Text("\n".join(lines)))
        panel.display = bool(lines) and not self._details  # an empty panel would leave a stray border line

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id == "rows":
            self._show_panel()

    def action_details(self) -> None:
        self._details = not self._details
        self.query_one("#rows").display = not self._details
        self._show_panel()
        self.query_one("#details-scroll").display = self._details
        if self._details:
            self._render_details()

    def _render_details(self) -> None:
        if self._view is not None:
            self.query_one("#details", Static).update(plain(diff_lines(self._view)))

    def action_fresh(self) -> None:
        self._set_status("starting fresh from the current state ...")
        self.query_one("#rows", OptionList).clear_options()
        self.query_one("#details", Static).update("")
        self.query_one("#limits", Static).update("")
        self._rows_by_id, self._reach, self._reach_for = {}, {}, None
        self._show_panel()
        self._start_watch()

    def action_edit(self) -> None:
        if self.query_one("#views", ContentSwitcher).current == "tree":
            node = self.query_one("#blast-tree", Tree).cursor_node
            data = node.data if node else None
            if not isinstance(data, BlastNode) or data.kind == "folder":
                self.notify("Select a file, class or function first.")
                return
            self.open_in_editor(data.path, data.line or 1)
            return
        box = self.query_one("#rows", OptionList)
        row = self._rows_by_id.get(box.highlighted_option.id) if box.highlighted is not None else None
        if row is None or row.file is None or row.line is None:
            self.notify("Select a changed symbol first.")
            return
        self.open_in_editor(row.file, row.line)

    def open_in_editor(self, file: str, line: int) -> None:
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
        # an absolute path can never start with `-` or `+`, so a file name from the repository cannot become an editor option
        path = str(self.project.path / file)
        try:
            command = editor_command(editor, path, line)
            with self.suspend():
                done = subprocess.run(command, check=False)  # noqa: S603 - the user's own editor, no shell
        except (OSError, ValueError) as err:
            self.notify(f"Could not run the editor ({editor}): {err}", severity="error")
            return
        if done.returncode:
            self.notify(f"The editor exited with status {done.returncode}.", severity="warning")

    async def action_quit(self) -> None:
        self._stop.set()
        self._stop_record()
        self.exit()

    # -- the R key -----------------------------------------------------------------------------------------------------------------

    def action_record(self) -> None:
        """R: ask, then (on y) record the repository's tests in a child process; R (or C) during a run stops it."""
        self._ask_or_stop(self._ask_to_record)

    def action_check(self) -> None:
        """C: ask, then (on y) run `argus check` in a child process; C (or R) during a run stops it."""
        self._ask_or_stop(self._ask_to_check)

    def _ask_or_stop(self, ask) -> None:
        if self._job is not None:
            self._say(f"Stopping the {self._job_kind} ...")
            self._job.stop(wait=False)  # the follower kills what is left and reports
            return
        self.call_later(ask)  # after this key's own dispatch, so the key that asked is not also the answer

    def _ask_to_record(self) -> None:
        choice = self._choose_python()
        if choice is not None:
            self._confirming = "record"
            self._say(question(self.project.path, choice.path, choice.source))

    def _ask_to_check(self) -> None:
        choice = self._choose_python()
        if choice is not None:
            self._confirming = "check"
            self._say(check_question(self.project.path, self.base, choice.path, choice.source))

    def _choose_python(self) -> PythonChoice | None:
        """The Python R or C would use, or None after saying why none can be used (nothing is asked then)."""
        self._confirming, self._choice = "", None  # an earlier question is dead whatever comes next
        choice = chosen_python(self.project.path, os.environ)
        if choice.error:
            self._say(f"{choice.error} Nothing was run.")
            return None
        self._choice = choice
        return choice

    def action_help(self) -> None:
        self.push_screen(HelpScreen(help_lines(chosen_python(self.project.path, os.environ))))

    def action_send(self) -> None:
        """s: type a note about the highlighted finding into the Claude Code pane of this tmux window (no Enter)."""
        box = self.query_one("#rows", OptionList)
        row = self._rows_by_id.get(box.highlighted_option.id) if box.highlighted is not None and box.highlighted_option else None
        change = row.change if row else None
        if self.query_one("#views", ContentSwitcher).current != "changes" or change is None:
            self.notify("Select a changed symbol in the Changes view first.")
            return
        pane, message = agent_pane(os.environ)
        if pane is None:
            self._say(message)
            return
        fresh_reach = self._reach.get(change.node_id) if self._reach_for is self._view else None  # a note from an older refresh may be wrong now
        error = send_text(pane, finding_note(change, fresh_reach))
        self._say(error or f"Sent to {message}. The note waits in its input box and contains text from your code: read it, then press Enter there to send it, or clear it.")

    def on_key(self, event: events.Key) -> None:
        if not self._confirming:
            return
        event.stop()
        event.prevent_default()  # even `q` only cancels
        kind, self._confirming = self._confirming, ""
        if event.key == "y" and kind == "record":
            self._start_record()
        elif event.key == "y":
            self._start_check()
        else:
            self._say("Recording cancelled." if kind == "record" else "Check cancelled.")

    def _say(self, text: str) -> None:
        self._record_line = text
        self._render_status()

    def _start_record(self) -> None:
        root = self.project.path
        choice = self._choice
        if choice is None:
            self._say("Not recorded: no Python was chosen. Nothing was run.")
            return
        version = python_version(choice.path)
        if version is None:
            self._say(f"Not recorded: the version of {choice.path} could not be read (is it a Python?).")
            return
        if version < MIN_RECORDING:
            self._say(
                f"Not recorded: {choice.path} is Python {version[0]}.{version[1]}; recording needs {MIN_RECORDING[0]}.{MIN_RECORDING[1]} or newer. "
                "Set ARGUS_PYTHON, or write the path of a newer Python that has this repository's packages in .argus/python."
            )
            return
        try:
            before = snapshot(root)  # before the child exists, so a file it writes can never be in it
            job = RecordJob(root, choice.path)
        except OSError as err:
            self._say(f"Not recorded: the command could not be started ({err.strerror or type(err).__name__})")
            return
        self._job, self._job_kind = job, "recording"
        self._say("Recording ...")
        self.run_worker(lambda: self._follow(job, before, choice.path), thread=True, group="record", exit_on_error=False)

    def _start_check(self) -> None:
        try:
            if self._choice is None:
                self._say("No Python was chosen. Nothing was run.")
                return
            job = CheckJob(self.project.path, self.base, self._choice.path)
        except OSError as err:
            self._say(f"Check could not be started ({err.strerror or type(err).__name__})")
            return
        self._job, self._job_kind = job, "check"
        self._say("Checking ...")
        self.run_worker(lambda: self._follow_check(job), thread=True, group="record", exit_on_error=False)

    def _follow_check(self, job: ChildJob) -> None:
        """Worker thread: report the time while the check runs, then its sentence. Whatever goes wrong, the screen hears about it and the key is free again."""
        text = "The check ended but its result is unknown."
        try:
            while job.poll() is None:
                job.escalate()
                self._post(self._say, f"Checking ... {job.seconds():.0f} s   (C again to stop)")
                time.sleep(0.5)
            job.sweep()
            code = job.poll()
            text = check_text(code if code is not None else -1, job.seconds(), job.output(), job.stopped, self.base or "HEAD")
        except Exception as err:  # noqa: BLE001 - a dead follower would leave "Checking ..." on screen for good
            text = f"The check ended but its result could not be read: {failure_text(err)}"
        finally:
            job.close()
            self._post(self._record_done, job, text)

    def _follow(self, job: ChildJob, before: dict[str, int], python: str) -> None:
        """Worker thread: report the time while the child runs, then what it left. Whatever goes wrong, the screen hears about it and
        the key is free again."""
        text = "The recording ended but its result is unknown."
        try:
            while job.poll() is None:
                job.escalate()
                self._post(self._say, f"Recording ... {job.seconds():.0f} s   (R again to stop)")
                time.sleep(0.5)
            job.sweep()
            text = self._outcome(job, before, python)
        except Exception as err:  # noqa: BLE001 - a dead follower would leave "Recording ..." on screen for good
            text = f"The recording ended but its result could not be read: {failure_text(err)}"
        finally:
            job.close()
            self._post(self._record_done, job, text)

    def _outcome(self, job: ChildJob, before: dict[str, int], python: str) -> str:
        root = self.project.path
        saved = fresh(before, snapshot(root))
        rec, error = self.project.recording() if saved else (None, None)
        exit_code, seconds = job.poll(), job.seconds()
        if rec is not None and rec.command != tuple(test_command(root, python)):  # another `argus record` finished during this run
            return "A recording appeared during the run that is not from it (another argus record?); nothing is reported for this run."
        if rec is not None:
            exit_code, seconds = rec.exit_code, rec.seconds
        ran, matched = counts(rec) if rec else (0, 0)
        return finished_text(
            saved=saved, exit_code=exit_code if exit_code is not None else 0, seconds=seconds, ran=ran, matched=matched, tail=job.tail(),
            stopped=job.stopped, lost=rec.processes.lost if rec else 0,
            errors=rec.processes.errors if rec else 0, unreadable=(error or "no reason given") if saved and rec is None else None,
        )

    def _record_done(self, job: ChildJob, text: str) -> None:
        if job is self._job:
            self._job = None
        self._say(text)
        self._show_panel()  # the panel and the tree read the newest recording
        if self.query_one("#views", ContentSwitcher).current == "tree":
            self._load_tree()

    def _stop_record(self) -> None:
        if self._job is not None:
            self._job.stop()

    # -- the other views -----------------------------------------------------------------------------------------------------------

    def _report_width(self) -> int:
        return max(MIN_REPORT_WIDTH, self.size.width - 3)

    def _set_text(self, widget_id: str, text: Text) -> None:
        self.query_one(f"#{widget_id}", Static).update(text)

    def _lookup(self, widget_id: str, produce: Callable[[int], Text], placeholder: str) -> None:
        """Show `placeholder`, compute in a thread, and show the result unless a newer lookup for this pane has started."""
        ticket = self._tickets[widget_id] = self._tickets.get(widget_id, 0) + 1
        self._set_text(widget_id, Text(placeholder))
        width = self._report_width()

        def work() -> None:
            try:
                text = produce(width)
            except Exception as err:  # noqa: BLE001 - whatever went wrong must reach the pane, not an invisible log
                text = Text(f"Failed: {type(err).__name__}: {str(err) or '(no message)'}")
            self._post(self._set_text_if_current, widget_id, ticket, text)

        self.run_worker(work, thread=True, group=widget_id, exclusive=True, exit_on_error=False)

    def _set_text_if_current(self, widget_id: str, ticket: int, text: Text) -> None:
        if ticket == self._tickets.get(widget_id):
            self._set_text(widget_id, text)

    # -- the blast-radius tree -----------------------------------------------------------------------------------------------------

    def _load_tree(self) -> None:
        """Build the tree in a thread and show it unless a newer request started or nothing it was built from has changed (so the user's
        expanded folders survive a trip to another view)."""
        ticket = self._tickets["blast-tree"] = self._tickets.get("blast-tree", 0) + 1
        if self._tree_key is None:
            self.query_one("#blast-tree", Tree).root.set_label("scanning ...")

        def work() -> None:
            try:
                result: tuple[BlastTree, tuple[int, int]] | str = blast_tree(self.project)
            except Exception as err:  # noqa: BLE001 - whatever went wrong must reach the pane, not an invisible log
                result = f"Failed: {type(err).__name__}: {str(err) or '(no message)'}"
            self._post(self._fill_tree, ticket, result)

        self.run_worker(work, thread=True, group="blast-tree", exclusive=True, exit_on_error=False)

    def _fill_tree(self, ticket: int, result: tuple[BlastTree, tuple[int, int]] | str) -> None:
        if ticket != self._tickets.get("blast-tree"):
            return
        tree = self.query_one("#blast-tree", Tree)
        if isinstance(result, str):
            tree.clear()
            tree.root.set_label(Text(printable(result)))  # a Text, not a str: a str is parsed as markup and "[/x]" in an error would raise
            self._tree_key = None
            return
        built, key = result
        if key == self._tree_key:
            return
        self._tree_key = key
        tree.clear()
        tree.root.set_label("production code, most depended-on first" if built.roots else "no production code found")
        self.query_one("#tree-legend", Static).update(Text(blast_legend(built)))

        def add(parent, node: BlastNode, depth: int) -> None:
            if node.children:
                branch = parent.add(blast_label(node), data=node, expand=depth < 1)
                for child in node.children:
                    add(branch, child, depth + 1)
            else:
                parent.add_leaf(blast_label(node), data=node)

        for top in built.roots:
            add(tree.root, top, 0)
        tree.root.expand()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        data = event.node.data
        self.query_one("#tree-detail", Static).update(Text(blast_detail(data)) if isinstance(data, BlastNode) else Text(""))

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if isinstance(data, BlastNode) and data.kind in SYMBOLS and data.qualname:  # a folder, file or class only opens or closes
            self.action_show("symbol")
            self.query_one("#query", Input).value = data.qualname
            self._search(data.qualname)

    def _search(self, query: str) -> None:
        self._lookup("symbol-text", lambda width: symbol_text(self.project, query, width), "looking up ...")

    def _diff(self) -> None:
        base, head = self.query_one("#base", Input).value, self.query_one("#head", Input).value
        self._lookup("diff-text", lambda width: diff_text(self.project, base, head, width), "comparing ...")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query":
            self._search(event.value)
        else:
            self._diff()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.disabled or event.option.id is None:
            return
        self.action_show("symbol")  # the option id is the symbol's qualname, so a list rebuilt meanwhile cannot change what opens
        self.query_one("#query", Input).value = event.option.id
        self._search(event.option.id)


def run_tui(path: Path, base: str | None = None, include_tests: bool = False) -> None:
    """Run the app. A closed terminal (SIGHUP) or `kill` (SIGTERM) ends it like `q` and removes its temporary folders."""
    app = ArgusApp(path, base=base, include_tests=include_tests)
    try:
        app.run()
    finally:
        app.stop_watching()
    if app.return_code:
        sys.exit(app.return_code)
