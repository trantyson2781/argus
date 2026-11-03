"""`argus explore watch`: repaint the diff report whenever a Python file changes.

The baseline (the working tree as it was at launch, or a commit) is read and scanned once and kept for the whole session. A refresh
builds the current working tree, scans it, compares it with the cached baseline and repaints. Change detection is a cheap poll of file
sizes and modification times (no extra dependency); a burst of edits is waited out so it costs one refresh, and an edit that lands
during a refresh triggers the next one. Nothing is written to the repository.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.text import Text

from argus.explore.changes import PARSE_ERROR_KIND, DiffView, build_diff_view
from argus.explore.render_compact import EditInfo, EditLog, compact_lines, print_compact
from argus.explore.render_diff import print_diff
from argus.explore.sides import LAUNCH_LABEL_PREFIX, DiffError, Side, materialize_baseline, materialize_current, python_file_stamps
from argus.extract import scan
from argus.extract.resolve_calls import resolve_v2_enabled
from argus.model.schema import ArchModel

POLL_SECONDS = 0.5
DEBOUNCE_SECONDS = 0.3
MAX_SETTLE_POLLS = 10  # a file written continuously cannot postpone a refresh forever
MAX_ERROR_CHARS = 300
LAUNCH_NOTE = "earlier edits not shown"
REF_NOTE = "that exact commit, not the merge-base"


@dataclass(frozen=True, slots=True)
class WatchSettings:
    scan_path: Path
    base: Side
    base_model: ArchModel
    resolve_v2: bool
    include_tests: bool
    unparsed_at_launch: int = 0  # baseline files that did not parse: they stay out of every comparison for the whole session


@contextmanager
def open_watch(path: Path, base_ref: str | None, include_tests: bool) -> Iterator[WatchSettings]:
    """Take and scan the baseline. Everything temporary is removed on exit."""
    resolve_v2 = resolve_v2_enabled()
    with materialize_baseline(path, base_ref) as (scan_path, base):
        base_model = scan(base.root, resolve_v2=resolve_v2)
        unparsed = sum(1 for b in base_model.blind_spots if b.kind == PARSE_ERROR_KIND)
        yield WatchSettings(scan_path, base, base_model, resolve_v2, include_tests, unparsed)


def refresh(settings: WatchSettings) -> tuple[DiffView, float]:
    """Compare the working tree now with the baseline: (view, seconds taken)."""
    started = time.perf_counter()
    with materialize_current(settings.scan_path, settings.base) as head:
        head_model = scan(head.root, resolve_v2=settings.resolve_v2)
        view = build_diff_view(settings.base, head, settings.base_model, head_model, settings.resolve_v2, settings.include_tests)
    return view, time.perf_counter() - started


def incomplete_note(settings: WatchSettings) -> str | None:
    parts = []
    if settings.unparsed_at_launch:
        n = settings.unparsed_at_launch
        parts.append(f"{n} file{'' if n == 1 else 's'} did not parse at launch")
    if settings.base.skipped:
        n = len(settings.base.skipped)
        parts.append(f"{n} file{'' if n == 1 else 's'} skipped at launch")
    return f"baseline incomplete: {', '.join(parts)} (relaunch to fix)" if parts else None


def _one_line(text: str) -> str:
    """`text` with non-ASCII and control characters escaped, cut to MAX_ERROR_CHARS (the message may quote repository content)."""
    return ascii(text)[1:-1][:MAX_ERROR_CHARS]


def failure_text(err: Exception) -> str:
    message = _one_line(str(err))
    frames = traceback.extract_tb(err.__traceback__)
    where = f"  at {Path(frames[-1].filename).name}:{frames[-1].lineno} in {frames[-1].name}" if frames else ""
    return f"{type(err).__name__}: {message}{where}"


def _full_report(settings: WatchSettings, console: Console, view: DiffView, seconds: float) -> None:
    note = LAUNCH_NOTE if settings.base.label.startswith(LAUNCH_LABEL_PREFIX) else REF_NOTE
    status = f"watching | baseline: {settings.base.label} ({note}) | refreshed {time.strftime('%H:%M:%S')} in {seconds:.1f} s | Ctrl-C quits"
    console.clear()
    console.print(Text(status), soft_wrap=True)
    incomplete = incomplete_note(settings)
    if incomplete:
        console.print(Text(incomplete), soft_wrap=True)
    print_diff(console, view)


def since_text(settings: WatchSettings) -> str:
    """What the baseline is, in words: "watching since launch HH:MM:SS (earlier edits not shown)" or "comparing with HEAD (abc1234)"."""
    label = settings.base.label
    return f"watching since {label} ({LAUNCH_NOTE})" if label.startswith(LAUNCH_LABEL_PREFIX) else f"comparing with {label}"


def _compact_report(
    settings: WatchSettings, console: Console, view: DiffView, seconds: float, edit: EditInfo | None, previous: frozenset[str] | None, log: EditLog,
) -> frozenset[str]:
    lines, keys = compact_lines(view, since=since_text(settings), edit=edit, previous=previous, refreshed=f"{time.strftime('%H:%M:%S')} in {seconds:.1f} s", log=log)
    incomplete = incomplete_note(settings)
    console.clear()
    print_compact(console, [*lines[:2], *([f"! {incomplete}"] if incomplete else []), *lines[2:]])
    return keys


def _paint(
    settings: WatchSettings, console: Console, event: "WatchEvent", details: bool, previous: frozenset[str] | None, log: EditLog,
) -> frozenset[str] | None:
    """Paint one event; returns the keys of the compact rows now on screen (None: nothing new was painted).

    A failure while painting is printed under the previous report (which stays on screen) and never ends the session."""
    try:
        if event.kind != "refreshed":
            console.print(Text(event.message), soft_wrap=True)
            return previous
        log.record(event.view, event.edit)
        if details:
            _full_report(settings, console, event.view, event.seconds)
            return None
        return _compact_report(settings, console, event.view, event.seconds, event.edit, previous, log)
    except BrokenPipeError:
        raise  # the reader went away: stop quietly
    except Exception as err:  # noqa: BLE001 - one failed paint must not end the session
        console.print(Text(refresh_failed_text(err)), soft_wrap=True)
        return previous


def refresh_failed_text(err: Exception) -> str:
    return f"refresh failed: {failure_text(err)}  (still watching; the next change tries again; Ctrl-C quits)"


def _stamps(settings: WatchSettings) -> tuple[dict[str, tuple[int, int] | None] | None, str | None]:
    """(stamps, None), or (None, why) when git or the directory could not be read this time."""
    try:
        return python_file_stamps(settings.scan_path), None
    except (DiffError, OSError) as err:
        return None, _one_line(str(err))


def _changed_files(before: dict[str, tuple[int, int] | None], after: dict[str, tuple[int, int] | None]) -> tuple[str, ...]:
    """Files added, removed or rewritten between two stamp readings, sorted."""
    return tuple(sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name)))


@dataclass(frozen=True, slots=True)
class WatchEvent:
    """One thing the watch loop has to tell its display. `refreshed` carries a finished diff view; the others carry a message to show."""

    kind: Literal["refreshed", "refresh_failed", "poll_failed"]
    view: DiffView | None = None
    seconds: float = 0.0
    edit: EditInfo | None = None  # the newest burst of saved files; None for the first view
    message: str = ""


def _refreshed(settings: WatchSettings, edit: EditInfo | None) -> WatchEvent:
    try:
        view, seconds = refresh(settings)
    except Exception as err:  # noqa: BLE001 - one failed refresh must not end the session
        return WatchEvent("refresh_failed", message=refresh_failed_text(err))
    return WatchEvent("refreshed", view, seconds, edit)


def watch_events(
    settings: WatchSettings, *, poll: float = POLL_SECONDS, debounce: float = DEBOUNCE_SECONDS,
    sleep: Callable[[float], None] = time.sleep, should_stop: Callable[[], bool] = lambda: False,
) -> Iterator[WatchEvent]:
    """The first view at once, then a new one after every change until `should_stop()`; poll failures are reported once and retried.

    The stamps are taken before each refresh, so an edit made while it runs differs from them and is picked up next. Used by the
    terminal report (`run_watch`) and by the TUI, so both see exactly the same changes.
    """
    stamps, _ = _stamps(settings)
    yield _refreshed(settings, None)
    last_error: str | None = None
    while not should_stop():
        sleep(poll)
        current, error = _stamps(settings)
        if current is None:
            if error != last_error:
                yield WatchEvent("poll_failed", message=f"poll failed: {error}  (retrying)")
                last_error = error
            continue
        last_error = None
        if stamps is None:  # the first reading failed: this one is the baseline, not a change
            stamps = current
            continue
        if current == stamps:
            continue
        edited = _changed_files(stamps, current)
        for _ in range(MAX_SETTLE_POLLS):
            sleep(debounce)
            again, _ = _stamps(settings)
            if again is None or again == current:
                break
            edited = _changed_files(stamps, again)
            current = again
        stamps = current
        yield _refreshed(settings, EditInfo(time.strftime("%H:%M:%S"), edited))


def run_watch(
    settings: WatchSettings, console: Console, *, poll: float = POLL_SECONDS, debounce: float = DEBOUNCE_SECONDS,
    sleep: Callable[[float], None] = time.sleep, should_stop: Callable[[], bool] = lambda: False, details: bool = False,
) -> None:
    """Paint every event of `watch_events` until `should_stop()` (or Ctrl-C, which the caller handles).

    `details` selects the long report; the default is the compact one, which also tells which file was just edited and which rows are new.
    """
    shown: frozenset[str] | None = None
    log = EditLog()
    for event in watch_events(settings, poll=poll, debounce=debounce, sleep=sleep, should_stop=should_stop):
        shown = _paint(settings, console, event, details, shown, log)
