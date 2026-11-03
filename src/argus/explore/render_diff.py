"""Terminal rendering for `argus explore diff`.

`diff_lines` turns a `DiffView` into plain text lines (deterministic, no colour, nothing read from disk); `print_diff` prints them.
Section titles start at column 0 and everything inside a section is indented, so the output is easy to scan and to test.
"""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.text import Text

from argus.analysis.code_type import classify_file, is_production
from argus.explore.call_fit import CallFit, Verdict
from argus.explore.changes import FILE_CHANGE_TEXT, INTERFACE_LABEL, CallerEdge, CallerGroup, DiffView, Flag, SymbolChange
from argus.explore.claims import Tier
from argus.explore.render import STYLE
from argus.explore.review_order import order_symbols

DEFAULT_CALLER_LIMIT = 20
DEFAULT_SYMBOL_LIMIT = 50
COLLAPSE_NEW_FILE_OVER = 5  # a new file with more symbols than this is one row; smaller ones stay listed in full
LEGEND = "[FACT] read from source   [INFERENCE] derived by resolution   [UNKNOWN] not determinable"
BODY_NOTE = "note: BODY_CHANGED means the code structure (syntax tree) differs. It does not show that behavior changed."
CALLS_NOTE = (
    "note: call changes are compared only for functions present on both sides. Calls made by added or removed functions "
    "are not listed here; an added function still appears as a caller below."
)
NO_CHANGE = "no structural change detected in the files that could be compared (see Not determinable for what is not compared)"
DOCSTRING_NOTE = "docstring-only edit: a syntactic change, not necessarily a behavioral one"
ORDER_NOTE = "suggested review order: production code before tests; removed or interface-changed, then body-changed, then added; most known callers first (a count is a minimum)"
DIRECT_TITLE = "Direct callers of changed interfaces (removed or SIGNATURE_CHANGED symbols)"
NOT_SCANNED_TITLE = "Not scanned (directories the scanner skips on purpose: build output, virtualenvs, hidden directories; not a failure)"
CALL_FIT_TITLE = "Calls checked against the new inputs (read from code, nothing was run)"
AFFECTED_TITLE = "Potentially affected (direct callers of BODY_CHANGED symbols only)"
FILES_TITLE = "Files changed outside any function or class"
RENAMED_TITLE = "Renamed files (content identical)"
EMPTY_NAMES_SHOWN = 3
FILE_SIGNS = {"outside_symbols": "~", "formatting_only": "~", "added": "+", "removed": "-", "relocated": "~"}
CALLER_MARKERS = {"changed": "(caller also changed)", "removed": "(caller removed)", "added": "(caller added)"}


def printable(line: str) -> str:
    """`line` with control characters, newlines and bidirectional overrides shown as escapes.

    File names and qualnames come from the repository being compared. Left as they are, an escape sequence could restyle or clear the
    terminal, and a newline could forge a report line.
    """
    return "".join(ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii") for ch in line)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _tag(tier: Tier) -> str:
    return f"[{tier.value}]"


def _header(view: DiffView) -> str:
    mode = "calls-v2" if view.resolve_v2 else "calls-v1"
    files = f"{view.files_changed} file{'' if view.files_changed == 1 else 's'} changed ({view.untracked_files} untracked)"
    return f"Diff  {view.base_label} -> {view.head_label}   {files}   {mode} on both sides"


def summary_counts(view: DiffView) -> dict[str, int]:
    """The counts behind the Summary line, shared with the JSON output so the two cannot drift apart."""
    renamed = {f for r in view.renames for f in (r.old, r.new)}
    symbols = tuple(c for c in view.symbols if c.file not in renamed)
    in_new = [c for c in symbols if c.change == "added" and c.file in view.new_files]
    modified = [c for c in symbols if c.change == "modified"]
    return {
        "removed": sum(1 for c in symbols if c.change == "removed"),
        "interface_changed": sum(1 for c in modified if Flag.SIGNATURE_CHANGED in c.flags),
        "body_changed": sum(1 for c in modified if Flag.BODY_CHANGED in c.flags),
        "calls_changed": sum(1 for c in modified if Flag.CALLS_CHANGED in c.flags),
        "added": sum(1 for c in symbols if c.change == "added"),
        "added_in_new_files": len(in_new),
        "new_files_with_added_symbols": len({c.file for c in in_new}),
        "renamed_files": len(view.renames),
        "outside_symbol_files": sum(1 for f in view.file_changes if f.kind != "formatting_only"),
        "interface_caller_groups": len(view.interface_callers),
        "affected_caller_groups": len(view.affected_callers),
    }


def _summary(view: DiffView) -> str | None:
    """One line of counts so the size of the change is clear before the detail. None when no symbol or caller changed."""
    n = summary_counts(view)
    counts = [(n["removed"], "removed"), (n["interface_changed"], "interface changed"), (n["body_changed"], "body changed"), (n["calls_changed"], "calls changed")]
    parts = [f"{count} {label}" for count, label in counts if count]
    if n["added"]:
        in_new, files = n["added_in_new_files"], n["new_files_with_added_symbols"]
        where = f" ({'all ' if in_new == n['added'] else ''}{in_new} in {files} new file{'' if files == 1 else 's'})" if in_new else ""
        parts.append(f"{n['added']} added{where}")
    if n["renamed_files"]:
        parts.append(_count(n["renamed_files"], "file") + " renamed")
    if n["outside_symbol_files"]:
        parts.append(_count(n["outside_symbol_files"], "file") + " changed outside functions or classes")
    callers = [(n["interface_caller_groups"], "of changed interfaces"), (n["affected_caller_groups"], "of changed bodies")]
    review = ", ".join(f"{count} {label}" for count, label in callers if count)
    if not parts and not review:
        return None
    return "Summary  " + ", ".join(parts) + (f"  |  caller groups to review: {review}" if review else "")


def _callers_note(known: int | None) -> str:
    if known is None:
        return ""
    if known == 0:
        return "   (no known callers; calls Argus cannot see are not counted)"
    return f"   (at least {_count(known, 'known caller')})"


def _symbol_lines(change: SymbolChange) -> list[str]:
    where = f"{change.file}:{change.line}"
    note = _callers_note(change.known_callers)
    if change.change == "added":
        return [f"  [FACT] + {change.kind}  {change.qualname}   {where}" + ("   (untracked file)" if change.untracked else "")]
    if change.change == "removed":
        return [f"  [FACT] - {change.kind}  {change.qualname}   was {where}{note}"]
    lines: list[str] = []
    fact_flags = [str(f) for f in Flag if f in change.flags and f is not Flag.CALLS_CHANGED]
    if fact_flags:
        lines.append(f"  [FACT] ~ {', '.join(fact_flags)}  {change.qualname}   {where}{note}")
        if change.interface_was is not None:
            lines.append(f"           was {change.interface_was}   now {change.interface_now}")
        if change.docstring_only:
            lines.append(f"           {DOCSTRING_NOTE}")
    if Flag.CALLS_CHANGED in change.flags:
        lines.append(f"  [INFERENCE] ~ {Flag.CALLS_CHANGED}  {change.qualname}   {where}")
    return lines


def _edge_line(edge: CallerEdge) -> str:
    claim = edge.claim
    site = claim.evidence[0]
    verb, was = ("called", "was ") if edge.was else ("calls", "")
    line = f"    {_tag(claim.tier)} {verb} {edge.callee} [{edge.callee_label}]   {was}{site.location}   {edge.confidence}   hash {site.source_hash}"
    if edge.was:
        line += " (before-snapshot)"
    if edge.override_possible:
        line += "  via self; a subclass may override it"
    return line


def _group_lines(group: CallerGroup) -> list[str]:
    header = f"  {group.caller}" + (" (module level)" if group.module_level else "") + (" (test code)" if group.is_test else "")
    marker = CALLER_MARKERS.get(group.caller_state)
    if group.caller_state == "unchanged" and any(e.callee_label == INTERFACE_LABEL for e in group.edges):
        marker = "(caller still exists, unchanged)"
    if marker:
        header += f"   {marker}"
    return [header, *(_edge_line(e) for e in group.edges)]


def _rows(symbols: tuple[SymbolChange, ...], new_files: frozenset[str]) -> list[tuple[SymbolChange, ...]]:
    """Display rows in view order. The symbols of a big new file form one row (at the place of its first symbol); every other symbol is its own row."""
    per_file = Counter(c.file for c in symbols if c.file in new_files and c.change == "added")  # a modified symbol is never folded away
    rows: list[tuple[SymbolChange, ...]] = []
    seen: set[str] = set()
    for change in symbols:
        if change.change != "added" or per_file[change.file] <= COLLAPSE_NEW_FILE_OVER:
            rows.append((change,))
        elif change.file not in seen:
            seen.add(change.file)
            rows.append(tuple(c for c in symbols if c.file == change.file and c.change == "added"))
    return rows


def _new_file_line(row: tuple[SymbolChange, ...]) -> str:
    first = row[0]
    kinds = ", ".join(f"{n} {kind if n == 1 else kind + ('es' if kind == 'class' else 's')}" for kind, n in sorted(Counter(c.kind for c in row).items()))
    untracked = "   (untracked file)" if first.untracked else ""
    return f"  [FACT] + new file  {first.file}:{first.line}   {len(row)} symbols ({kinds}); use --all to list them{untracked}"


def _picked(rows: list[tuple[SymbolChange, ...]], limit: int | None) -> tuple[list[tuple[SymbolChange, ...]], list[SymbolChange]]:
    """(shown rows, omitted symbols). When a cap applies, production code is kept before test code; shown rows stay in view order."""
    if limit is None or len(rows) <= limit:
        return rows, []
    ranked = sorted(range(len(rows)), key=lambda i: (not is_production(classify_file(rows[i][0].file)), i))
    keep = set(ranked[:limit])
    return [r for i, r in enumerate(rows) if i in keep], [c for i, r in enumerate(rows) if i not in keep for c in r]


def _symbols_section(symbols: tuple[SymbolChange, ...], limit: int | None, new_files: frozenset[str]) -> list[str]:
    symbols = order_symbols(symbols)
    # A cap means the compact view; with no cap (--all) a new file is listed symbol by symbol.
    rows = _rows(symbols, new_files) if limit is not None else [(c,) for c in symbols]
    shown, omitted = _picked(rows, limit)
    lines = ["", f"Symbols ({len(symbols)})"]
    if len(symbols) > 1:
        lines.append(f"  {ORDER_NOTE}")
    for row in shown:
        lines.extend([_new_file_line(row)] if len(row) > 1 else _symbol_lines(row[0]))
    if omitted:
        counts = {kind: sum(1 for c in omitted if c.change == kind) for kind in ("added", "removed", "modified")}
        breakdown = ", ".join(f"{n} {kind}" for kind, n in counts.items())
        in_tests = sum(1 for c in omitted if not is_production(classify_file(c.file)))
        if in_tests:
            breakdown += f"; {in_tests} in test code"
        lines.append(f"  +{len(omitted)} more symbols not shown ({breakdown}); raise --limit or use --all")
    if any(Flag.BODY_CHANGED in change.flags for change in symbols):
        lines.append(f"  {BODY_NOTE}")
    return lines


def _renamed_lines(view: DiffView) -> list[str]:
    if not view.renames:
        return []
    lines = ["", RENAMED_TITLE]
    for r in view.renames:
        what = "no functions or classes" if r.symbols == 0 else _count(r.symbols, "symbol")
        lines.append(f"  [FACT] > {r.old} -> {r.new}   {what}")
    return lines


def _file_change_lines(view: DiffView) -> list[str]:
    if not view.file_changes:
        return []
    lines = ["", FILES_TITLE]
    lines += [f"  [FACT] {FILE_SIGNS[f.kind]} {f.file}   {FILE_CHANGE_TEXT[f.kind]}" for f in view.file_changes if not f.kind.endswith("_empty")]
    for kind, sign, noun in (("added_empty", "+", "new empty file"), ("removed_empty", "-", "removed empty file")):
        names = [f.file for f in view.file_changes if f.kind == kind]
        if names:
            more = f" and {len(names) - EMPTY_NAMES_SHOWN} more" if len(names) > EMPTY_NAMES_SHOWN else ""
            lines.append(f"  [FACT] {sign} {_count(len(names), noun)}: {', '.join(names[:EMPTY_NAMES_SHOWN])}{more}")
    return lines


def _not_scanned_lines(view: DiffView) -> list[str]:
    if not view.not_scanned:
        return []
    lines = ["", NOT_SCANNED_TITLE]
    for item in view.not_scanned:
        lines.append(f"  {item.prefix}  {item.files} changed .py file{'' if item.files == 1 else 's'}")
    return lines


def _caller_section(title: str, groups: tuple[CallerGroup, ...], limit: int | None) -> list[str]:
    if not groups:
        return []
    shown = groups if limit is None else groups[:limit]
    lines = ["", title]
    for group in shown:
        lines.extend(_group_lines(group))
    if len(shown) < len(groups):
        lines.append(f"  +{len(groups) - len(shown)} more (raise --limit or use --all)")
    return lines


def _fit_row(fit: CallFit) -> list[str]:
    label = "DOES NOT FIT" if fit.verdict is Verdict.BREAKS else "NOT CHECKED"
    note = "   (this call already did not fit before)" if fit.already_broken else ""
    test = " (test code)" if fit.is_test else ""
    return [f"  {_tag(fit.claim.tier)} {label}  {fit.caller}{test} -> {fit.callee}   {fit.file}:{fit.line}", f"      {fit.reason}{note}"]


def _call_fit_lines(fits: tuple[CallFit, ...], limit: int | None) -> list[str]:
    """Calls to a function whose inputs changed. Only calls that do not fit, or could not be checked, get a row; the rest are counted."""
    if not fits:
        return []
    rows = [f for f in fits if f.verdict is not Verdict.FITS]
    shown = rows if limit is None else rows[:limit]
    lines = ["", CALL_FIT_TITLE]
    for fit in shown:
        lines += _fit_row(fit)
    if len(shown) < len(rows):
        lines.append(f"  +{len(rows) - len(shown)} more (raise --limit or use --all)")
    fitting = len(fits) - len(rows)
    if fitting:
        lines.append(f"  {_count(fitting, 'call')} checked {'fits' if fitting == 1 else 'fit'} the new inputs; calls made through other objects are not seen")
    return lines


def _hidden_tests_lines(view: DiffView) -> list[str]:
    if not view.hidden_test_callers:
        return []
    noun = "group" if view.hidden_test_callers == 1 else "groups"
    return [f"  {view.hidden_test_callers} test caller {noun} hidden (use --include-tests)"]


def diff_lines(view: DiffView, limit: int | None = DEFAULT_CALLER_LIMIT, symbol_limit: int | None = DEFAULT_SYMBOL_LIMIT) -> list[str]:
    """The full report as plain lines. `limit` caps caller groups per caller section and `symbol_limit` the Symbols section (None: no cap).

    The view is never trimmed: every cap only shortens what is printed, and says how much it left out.
    """
    lines = [_header(view), LEGEND]
    summary = _summary(view)
    if summary:
        lines.append(summary)
    structural_files = [f for f in view.file_changes if f.kind != "formatting_only"]
    if not (view.symbols or view.calls or view.imports or view.interface_callers or view.affected_callers or view.renames or structural_files):
        lines += ["", NO_CHANGE]
    lines += _renamed_lines(view)
    # A renamed file's symbols are removed under the old path and added under the new one: --all lists them, the default view does not.
    renamed = {f for r in view.renames for f in (r.old, r.new)} if symbol_limit is not None else set()
    shown = tuple(c for c in view.symbols if c.file not in renamed)
    if shown:
        lines += _symbols_section(shown, symbol_limit, view.new_files)
    if len(shown) < len(view.symbols):
        lines.append(f"  ({len(view.symbols) - len(shown)} symbols in renamed files are not listed: their names change with the path; use --all)")
    lines += _file_change_lines(view)
    if view.calls:
        lines += ["", f"Calls ({len(view.calls)})"]
        for call in view.calls:
            site = call.claim.evidence[0]
            sign, where = ("+", site.location) if call.change == "added" else ("-", f"was {site.location}")
            lines.append(f"  {_tag(call.claim.tier)} {sign} {call.source} -> {call.target}   {where}")
        lines.append(f"  {CALLS_NOTE}")
    lines += _caller_section(DIRECT_TITLE, view.interface_callers, limit)
    lines += _caller_section(AFFECTED_TITLE, view.affected_callers, limit)
    lines += _call_fit_lines(view.call_fits, limit)
    hidden = _hidden_tests_lines(view)
    if hidden:
        lines += ["", "Callers in test code", *hidden]
    if view.imports:
        lines += ["", "Subsystem imports"]
        for item in view.imports:
            site = item.claim.evidence[0]
            sign, where = ("+", site.location) if item.change == "added" else ("-", f"was {site.location}")
            lines.append(f"  {_tag(item.claim.tier)} {sign} {item.source} -> {item.target}   {where}")
    lines += _not_scanned_lines(view)
    lines += ["", "Not determinable"]
    lines += [f"  {_tag(claim.tier)} {claim.text}" for claim in view.not_determinable]
    return [printable(line) for line in lines]


def print_diff(console: Console, view: DiffView, limit: int | None = DEFAULT_CALLER_LIMIT, symbol_limit: int | None = DEFAULT_SYMBOL_LIMIT) -> None:
    for line in diff_lines(view, limit, symbol_limit):
        text = Text(line)
        for tier, style in STYLE.items():
            text.highlight_words([f"[{tier}]"], style)
        console.print(text, soft_wrap=True, highlight=False, emoji=False)
