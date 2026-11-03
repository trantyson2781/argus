"""The "who is affected" panel under the Changes list: plain lines about the one highlighted symbol.

Everything comes from facts the diff already holds (callers, tests, the module and who imports it) and every list is a lower bound,
so the panel says "known", "found" and "could be affected", and ends with the one check Argus cannot do itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.explore.call_fit import Verdict
from argus.explore.changed_lines import MAX_LINES, ChangedLine
from argus.explore.changes import Flag, SymbolChange, SymbolUser, search_name
from argus.explore.render_compact import Touch
from argus.explore.render_observed import observed_text
from argus.explore.render_diff import printable
from argus.record.observed import ObservedFacts

MAX_USERS = 6
MAX_TESTS = 3
MAX_MODULES = 4
MAX_SPOTS = 3
MAX_FITS = 3
MAX_CHANGED_SHOWN = 6  # changed lines in the panel; the window is small and the rest is in `git diff`
MAX_LINE_WIDTH = 100
LABEL = 9  # width of the left column ("Used by  ")
BLIND_SPOTS = "Argus can't see calls through obj.method(), subclasses or dictionaries."


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _short(user: SymbolUser) -> str:
    if user.module_level:
        return "(module level)"
    return user.caller[len(user.module) + 1 :] if user.caller.startswith(f"{user.module}.") else user.caller


def _users_lines(change: SymbolChange) -> list[str]:
    production = [u for u in change.users if not u.is_test]
    if not production:
        return [f"{'Used by':<{LABEL}}no known users"]
    modules = len({u.module for u in production})
    after = " (before it was deleted)" if change.change == "removed" else ""
    lines = [f"{'Used by':<{LABEL}}{_plural(len(production), 'place')} in {_plural(modules, 'module')}{after}"]
    width = max(len(u.module) for u in production[:MAX_USERS])
    name_width = max(len(_short(u)) for u in production[:MAX_USERS])
    for user in production[:MAX_USERS]:
        lines.append(f"  {user.module:<{width}}  {_short(user):<{name_width}}  {user.file}:{user.line}")
    if len(production) > MAX_USERS:
        lines.append(f"  +{len(production) - MAX_USERS} more (Enter opens the full symbol view)")
    return lines


def _tests_line(change: SymbolChange) -> str:
    tests = [u for u in change.users if u.is_test and u.caller.rpartition(".")[2].startswith("test")]
    if not tests:
        return f"{'Tests':<{LABEL}}none found"
    named = ", ".join(f"{u.caller.rpartition('.')[2]} ({u.file}:{u.line})" for u in tests[:MAX_TESTS])
    more = f" and {len(tests) - MAX_TESTS} more" if len(tests) > MAX_TESTS else ""
    return f"{'Tests':<{LABEL}}{named}{more}"


MAX_REACH = 6


@dataclass(frozen=True, slots=True)
class ReachNote:
    """Which entry points reach a changed function: the names found, how many entry points the project has, and why the list may be short."""

    commands: tuple[str, ...]
    found: int
    caveat: str | None


def _reach_lines(change: SymbolChange, reach: ReachNote | None) -> list[str]:
    if reach is None or change.change != "modified" or change.kind not in ("function", "method"):
        return []
    label = "Reached from  "  # longer than the label column, so it keeps its own gap (like "Seen running")
    if reach.commands:
        more = f"  +{len(reach.commands) - MAX_REACH} more" if len(reach.commands) > MAX_REACH else ""
        return [f"{label}{' · '.join(reach.commands[:MAX_REACH])}{more}  [I]"]
    if reach.found == 0:
        return [f"{label}unknown: {reach.caveat}  [?]"] if reach.caveat else []
    why = reach.caveat or "calls through obj.method() or data are not followed"
    return [f"{label}no command or route Argus can see reaches it  [?]  ({why})"]


def _module_line(change: SymbolChange) -> list[str]:
    if change.module is None:
        return []
    importers = change.module_imported_by
    if not importers:
        return [f"{'Module':<{LABEL}}{change.module}: no other module imports it that Argus can see"]
    shown = importers[:MAX_MODULES]
    names = ", ".join(shown[:-1]) + (" and " if len(shown) > 1 else "") + shown[-1]
    more = f" and {len(importers) - MAX_MODULES} more" if len(importers) > MAX_MODULES else ""
    return [f"{'Module':<{LABEL}}{change.module} is imported by {names}{more} (they could be affected)"]


def _balanced(lines: tuple[ChangedLine, ...]) -> list[ChangedLine]:
    """At most MAX_CHANGED_SHOWN lines, in order, keeping some old and some new ones: a long replacement must not show only its old half."""
    if len(lines) <= MAX_CHANGED_SHOWN:
        return list(lines)
    old, new = [c for c in lines if c.kind == "-"], [c for c in lines if c.kind == "+"]
    half = MAX_CHANGED_SHOWN // 2
    n_new = min(len(new), max(half, MAX_CHANGED_SHOWN - len(old)))
    keep = {id(c) for c in (*old[: min(len(old), MAX_CHANGED_SHOWN - n_new)], *new[:n_new])}
    return [c for c in lines if id(c) in keep]


def _changed_rows(change: SymbolChange) -> list[str]:
    """What the code said before and what it says now, for a modified function whose body changed (or why that cannot be shown)."""
    if change.change != "modified":
        return []
    if not change.changed_lines:
        return [f"{'Changed':<{LABEL}}{change.changed_note}"] if change.changed_note else []
    shown = _balanced(change.changed_lines)
    rows = []
    for line in shown:
        text = line.text if len(line.text) <= MAX_LINE_WIDTH else line.text[: MAX_LINE_WIDTH - 1] + "…"
        rows.append(f"{'Before' if line.kind == '-' else 'Now':<{LABEL}}L{line.line}  {text}")
    hidden = len(change.changed_lines) - len(shown)
    if hidden <= 0:
        return rows
    at_least = "at least " if len(change.changed_lines) >= MAX_LINES else ""
    return rows + [f"  +{at_least}{hidden} more changed line{'' if hidden == 1 and not at_least else 's'} (git diff shows them all)"]


def _fit_lines(change: SymbolChange) -> list[str]:
    """Calls checked against the new inputs; only for a function whose inputs changed. Counts are lower bounds, never a verdict."""
    if Flag.SIGNATURE_CHANGED not in change.flags or change.change != "modified" or change.known_callers is None:
        return []
    fits = change.call_fits
    if not fits:
        return ["Calls checked  no known call to check"]
    broken = [f for f in fits if f.verdict is Verdict.BREAKS]
    unknown = [f for f in fits if f.verdict is Verdict.UNKNOWN]
    parts = [f"{len(broken)} {'does' if len(broken) == 1 else 'do'} not fit", f"{len(fits) - len(broken) - len(unknown)} fit"]
    if unknown:
        parts.append(f"{len(unknown)} not checked")
    lines = [f"Calls checked  {' · '.join(parts)}"]
    for fit in broken[:MAX_FITS]:
        lines.append(f"  {fit.caller}  {fit.file}:{fit.line}  {fit.reason}")
    if len(broken) > MAX_FITS:
        lines.append(f"  +{len(broken) - MAX_FITS} more (see explore diff)")
    return lines


def _dynamic_lines(change: SymbolChange) -> list[str]:
    spots = change.dynamic_spots
    lines = [f"⚠ dynamic code at line {s.line}: {s.detail}; callers and effects here may be hidden" for s in spots[:MAX_SPOTS]]
    return lines + ([f"  +{len(spots) - MAX_SPOTS} more"] if len(spots) > MAX_SPOTS else [])


def panel_lines(change: SymbolChange, touch: Touch | None, observed: ObservedFacts | None = None, reach: ReachNote | None = None) -> list[str]:
    """The panel for one changed symbol, as plain lines (repository names are escaped). `observed` is what the latest recording saw of it;
    `reach` is which commands and routes reach it."""
    lines = [f"{change.qualname}   ({change.file}:{change.line})", *_changed_rows(change)]
    if change.change == "added":
        lines.append(f"New {change.kind}: callers are not checked for new code.")
    elif change.known_callers is None:
        lines.append("Who uses it: not checked for this kind of change.")
    else:
        if change.interface_was is not None and change.interface_now is not None:
            lines += [f"{'Before':<{LABEL}}{change.qualname.rpartition('.')[2]}{change.interface_was}", f"{'Now':<{LABEL}}{change.qualname.rpartition('.')[2]}{change.interface_now}"]
        lines += [*_fit_lines(change), *_users_lines(change), _tests_line(change), *_reach_lines(change, reach)]
    lines += _module_line(change)
    if observed is not None and change.change == "modified" and change.kind in ("function", "method"):
        lines.append(f"Seen running  {observed_text(observed)}")  # a longer label than the column, so it keeps its own gap
    if touch:
        lines.append(f"{'Saved':<{LABEL}}{touch.time} (save {touch.saves} this session)")
    lines += _dynamic_lines(change)
    if change.known_callers is not None:
        lines.append(BLIND_SPOTS)
        name = search_name(change)
        if name:
            lines.append(f"Double-check with:  git grep -n '{name}('")
    return [printable(line) for line in lines]
