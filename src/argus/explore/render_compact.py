"""The default `argus explore watch` view: only what changed, one short row each.

Everything the long report (`--details`) says is still computed; this view hides it, it does not drop it. Each row is one sentence in
everyday words: what happened, how many places use it and whether a test was found (both lower bounds, so the text says "at least" and
"found"). The label (CHECK, LOOK, LOW) says where to look first and never that a change is right; the legend says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.text import Text

from argus.analysis.code_type import CodeType, classify_file, is_production
from argus.explore.changes import FILE_CHANGE_TEXT, DiffView, Flag, SymbolChange
from argus.explore.render import short_name
from argus.explore.render_diff import printable
from argus.explore.review_order import Attention, attention, order_symbols

MAX_ROWS = 8
MAX_FILES_NAMED = 3
NEW_TAG = "NEW "
NO_TAG = " " * len(NEW_TAG)
DETAILS_HINT = "use --details"


@dataclass(frozen=True, slots=True)
class EditInfo:
    """The newest burst of saved files, which is what the header names."""

    time: str
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Touch:
    """When a symbol's file was last saved (HH:MM:SS) and how many saves of it were seen."""

    time: str
    saves: int


class EditLog:
    """Remembers, per changed symbol, when its file was last saved while watching. One per watch session; the screen asks it for times.

    The clock time is stored, not an age: a screen that only repaints on a change would let "just now" go stale.
    """

    def __init__(self) -> None:
        self._touches: dict[str, Touch] = {}
        self.newest: str | None = None  # the time of the latest save, so rows from it can stand out

    def record(self, view: DiffView, edit: EditInfo | None) -> None:
        """Note the save that produced `view` (None: the first view, nothing was saved yet)."""
        if edit is None:
            return
        self.newest = edit.time
        saved = set(edit.files)
        for change in view.symbols:
            if change.file in saved:
                before = self._touches.get(change.qualname)
                self._touches[change.qualname] = Touch(edit.time, (before.saves if before else 0) + 1)

    def touch_of(self, qualname: str) -> Touch | None:
        return self._touches.get(qualname)


def change_key(change: SymbolChange) -> str:
    """Identity of one row across repaints: the symbol and what happened to it (a body edit that becomes an interface edit is news again)."""
    flags = ",".join(sorted(str(f) for f in change.flags))
    return f"{change.node_id}|{change.change}|{flags}"


LABEL_START = 2 + len(NEW_TAG)  # rows are "  " + NEW tag + label
LABEL_WIDTH = max(len(level.name) for level in Attention)
LABEL_STYLES = {Attention.CHECK.name: "bold red", Attention.LOOK.name: "bold yellow", Attention.LOW.name: "green"}
LEGEND = "  CHECK may break · LOOK logic edited · LOW new or description only  (where to look first, not whether it is right)"


EDIT_NOTES = {"condition": ", including a condition", "result": ", including a returned or raised value"}


def _what(change: SymbolChange) -> str:
    """What happened, in everyday words."""
    if change.change == "added":
        return f"new {change.kind}"
    if change.change == "removed":
        return "deleted"
    if Flag.SIGNATURE_CHANGED in change.flags:
        return "inputs, outputs and logic changed" if Flag.BODY_CHANGED in change.flags and not change.docstring_only else "inputs or outputs changed"
    if Flag.BODY_CHANGED in change.flags:
        return "description edited" if change.docstring_only else "logic edited" + EDIT_NOTES.get(change.edit_kind, "")
    return "now calls different functions"


def _users(change: SymbolChange) -> str:
    known = change.known_callers
    if known is None:
        return ""
    if known == 0:
        return "no known users"
    return f"{'was used' if change.change == 'removed' else 'used'} in at least {known} place{'' if known == 1 else 's'}"


def _tests(count: int | None) -> str:
    if count is None:
        return ""
    return "no test found" if count == 0 else f"{count} test{'' if count == 1 else 's'} found"


def _dynamic(change: SymbolChange) -> str:
    return "⚠ dynamic code" if change.dynamic_spots else ""


def _row(change: SymbolChange, is_new: bool, width: int, touch: Touch | None = None) -> str:
    saved = f"saved {touch.time}" if touch else ""
    detail = " · ".join(part for part in (_what(change), _users(change), _tests(change.test_callers), _dynamic(change), saved) if part)
    label = attention(change).name.ljust(LABEL_WIDTH)
    return f"  {NEW_TAG if is_new else NO_TAG}{label} {short_name(change.qualname).ljust(width)}  {detail}"


def stylize_label(text: Text) -> None:
    """Colour the attention label of a row, if the row has one."""
    style = LABEL_STYLES.get(text.plain[LABEL_START : LABEL_START + LABEL_WIDTH].strip())
    if style:
        text.stylize(style, LABEL_START, LABEL_START + LABEL_WIDTH)


def _tests_row(tests: list[SymbolChange]) -> str:
    counts = [(sum(1 for c in tests if c.change == kind), label) for kind, label in (("added", "added"), ("modified", "changed"), ("removed", "removed"))]
    return "  tests: " + ", ".join(f"{n} {label}" for n, label in counts if n)


def _files_text(files: tuple[str, ...]) -> str:
    named = ", ".join(files[:MAX_FILES_NAMED])
    return named + (f" and {len(files) - MAX_FILES_NAMED} more" if len(files) > MAX_FILES_NAMED else "")


def _other_rows(view: DiffView) -> list[tuple[str, str]]:
    """(key, text) for changes with no function row: edits outside functions, new or removed files without functions, renames."""
    rows = [(f"file|{f.file}|{f.kind}", f"~ {f.file}  {FILE_CHANGE_TEXT[f.kind]}") for f in view.file_changes if f.kind != "formatting_only"]
    rows += [(f"rename|{r.old}|{r.new}", f"~ {r.old} -> {r.new}  file renamed") for r in view.renames]
    return rows


@dataclass(frozen=True, slots=True)
class Row:
    """One line of the compact view. `qualname`, `file` and `line` are set on a changed symbol, so a UI can open it."""

    text: str
    qualname: str | None = None
    file: str | None = None
    line: int | None = None
    change: SymbolChange | None = None
    fresh: bool = False  # saved in the newest save, so a screen may make it stand out


def compact_rows(view: DiffView, previous: frozenset[str] | None, log: EditLog | None = None) -> tuple[list[Row], frozenset[str]]:
    """(rows, keys of every row now shown). `previous` is the keys of the last repaint: rows not in it are tagged NEW (None: tag nothing).
    `log` adds the time each symbol's file was last saved."""
    ordered = order_symbols(view.symbols)
    production = [c for c in ordered if is_production(classify_file(c.file))]
    tests = [c for c in ordered if c not in production]
    others = _other_rows(view)
    keys = frozenset([*(change_key(c) for c in ordered), *(key for key, _ in others)])

    def is_new(key: str) -> bool:
        return previous is not None and key not in previous

    shown = production[:MAX_ROWS]
    width = max((len(short_name(c.qualname)) for c in shown), default=0)
    touches = {c.qualname: log.touch_of(c.qualname) if log else None for c in shown}
    rows = [
        Row(printable(_row(c, is_new(change_key(c)), width, touches[c.qualname])), c.qualname, c.file, c.line, c, bool(touches[c.qualname] and log and touches[c.qualname].time == log.newest))
        for c in shown
    ]
    room = MAX_ROWS - len(shown)
    rows += [Row(printable(f"  {NEW_TAG if is_new(key) else NO_TAG}{text}")) for key, text in others[: max(room, 0)]]
    if tests:
        rows.append(Row(printable(_tests_row(tests))))
    hidden = len(production) - len(shown) + len(others) - max(room, 0)
    if hidden > 0:
        rows.append(Row(f"  +{hidden} more ({DETAILS_HINT})"))
    return rows, keys


def changes_summary(view: DiffView, *, scroll_hint: bool = False) -> str:
    """One line that says how many changes are worth a look, so a reader of a small window knows how many rows to find.
    CHECK and LOOK count as worth a look; LOW (new code, a description edit), test code, config or generated code and file-level
    changes are counted apart. Empty when nothing changed."""
    production = [c for c in view.symbols if is_production(classify_file(c.file))]
    tests = sum(classify_file(c.file) is CodeType.TEST for c in view.symbols)
    other = len(view.symbols) - len(production) - tests
    file_level = len(view.file_changes) + len(view.renames)
    if not view.symbols and not file_level:
        return ""
    counts = {label: sum(attention(c) is label for c in production) for label in (Attention.CHECK, Attention.LOOK, Attention.LOW)}
    important = counts[Attention.CHECK] + counts[Attention.LOOK]
    if important:
        detail = ", ".join(f"{counts[label]} {label.name}" for label in (Attention.CHECK, Attention.LOOK) if counts[label])
        parts = [f"{important} change{'' if important == 1 else 's'} to look at ({detail})"]
    else:
        parts = ["no function changes to look at"]
    for count, text in (
        (counts[Attention.LOW], "{} LOW"), (tests, "{} test function{} changed"), (other, "{} in config or generated code"),
        (file_level, "{} file-level change{}"),
    ):
        if count:
            parts.append(text.format(count, "" if count == 1 else "s"))
    hint = ""
    if scroll_hint and len(production) > 1:
        hint = f" — ↑↓ shows {MAX_ROWS}, press d for all" if len(production) > MAX_ROWS else " — ↑↓ shows the rest"
    return " · ".join(parts) + hint


def limits_line(view: DiffView) -> str:
    limits = len(view.not_determinable)
    return f"  {limits} thing{'' if limits == 1 else 's'} Argus can't check (--details)"


def legend_lines(view: DiffView) -> list[str]:
    """The legend, only when a labelled row is on screen."""
    return [LEGEND] if any(is_production(classify_file(c.file)) for c in view.symbols) else []


def header_line(edit: EditInfo | None, has_changes: bool) -> str:
    if edit:
        return printable(f"● {edit.time}  edited {_files_text(edit.files)}")
    return "● changes so far" if has_changes else "● nothing changed yet"


def compact_lines(
    view: DiffView, *, since: str, edit: EditInfo | None, previous: frozenset[str] | None, refreshed: str, log: EditLog | None = None,
) -> tuple[list[str], frozenset[str]]:
    """(lines, keys of every row now shown): the header, the rows and the folded limits."""
    rows, keys = compact_rows(view, previous, log)
    lines = [header_line(edit, bool(keys)), f"  {since} | refreshed {refreshed} | Ctrl-C quits | restart to start fresh"]
    lines += [f"! {file} does not parse: nothing in it was compared" for file in view.failed_files]
    lines += [""] if keys else []
    lines += [row.text for row in rows]
    lines += ["", *legend_lines(view), limits_line(view)]
    return [printable(line) for line in lines], keys


def print_compact(console: Console, lines: list[str]) -> None:
    for line in lines:
        text = Text(line)
        if line.startswith("●"):
            text.stylize("bold")
        elif line.startswith("!"):
            text.stylize("red")
        if line.startswith(f"  {NEW_TAG}"):
            text.stylize("bold green", 2, 2 + len(NEW_TAG.strip()))
        stylize_label(text)
        console.print(text, soft_wrap=True, highlight=False, emoji=False)
