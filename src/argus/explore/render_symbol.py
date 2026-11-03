"""Terminal rendering for `argus explore symbol`."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.text import Text

from argus.explore.claims import Claim
from argus.explore.render import LEGEND, claim_line, short_name
from argus.explore.render_observed import observed_text
from argus.explore.symbols import SymbolView

DEFAULT_LIMIT = 15
_SELF_NOTE_OFF = (
    "V2 call resolution is disabled (ARGUS_RESOLVE_V2=0), so calls written as self.method() are not resolved "
    "and callers inside the same class may be missing."
)
_SELF_NOTE_ON = (
    "self.method() calls are linked only when the method has a single definition on its class (and is not a property); "
    "inherited, subclass-overridden and shadowed cases are not, so some callers may be missing."
)


MAX_NAMES = 4
CHAIN_ARROW = " -> "


def _names(names: list[str]) -> str:
    return ", ".join(names[:MAX_NAMES]) + (f", +{len(names) - MAX_NAMES} more" if len(names) > MAX_NAMES else "")


def _reach_rows(view: SymbolView) -> list[tuple[str, str]]:
    label, noun = ("Reached from", "command, route or program") if view.extra_found else ("Used by CLI", "CLI command")
    if view.reached_from is None:
        return [(label, f"unknown: {view.reach_caveat}  [?]")] if view.reach_caveat else []
    if view.reached_from:
        first = view.reached_from[0]
        rows = [
            (label, " · ".join(r.command for r in view.reached_from) + "  [I]"),
            ("Chain", CHAIN_ARROW.join(short_name(q) for q in first.chain) + f"  [I]  (shortest, from {first.command})"),
        ]
    elif view.commands_found:
        why = view.reach_caveat or "calls through obj.method() or data are not followed"
        rows = [(label, f"not reached from any {noun} Argus can see  [?]  ({why})")]
    else:
        rows = [(label, "no CLI commands found in this project, so reach is not known  [?]" + (f"  ({view.reach_caveat})" if view.reach_caveat else ""))]
    return rows


def _summary_rows(view: SymbolView) -> list[tuple[str, str]]:
    """(label, text) rows. `[I]` marks an inference (a resolved call chain), `[?]` something Argus could not determine."""
    rows = _reach_rows(view)
    if view.reached_from is not None:
        callers = sorted({r.name for r in view.caller_rows})
        via_self = sum(1 for r in view.caller_rows if r.override_possible)
        note = f"  ({via_self} via self.method(); a subclass may override it)" if via_self else ""
        rows.append(("Called by", (f"{len(callers)} resolved: {_names([short_name(c) for c in callers])}  [I]{note}" if callers else "no resolved callers  [?]")))
    if view.tested_by is not None:
        tests = [short_name(t) for t in view.tested_by]
        if tests:
            text = f"at least {len(tests)} test{'' if len(tests) == 1 else 's'} call it directly: {_names(tests)}  [I]"
        else:
            hidden = f"; {view.hidden_test_callers} caller(s) in test code (helpers or module level) are hidden, use --include-tests" if view.hidden_test_callers else ""
            text = f"no test function calls it directly  [?]  (a test may reach it through other code{hidden})"
        rows.append(("Tested by", text))
    if view.observed is not None:
        rows.append(("Seen running", observed_text(view.observed)))
    if view.observed_error:
        rows.append(("Recording", view.observed_error))
    if view.lines is not None:
        rows.append(("Size", f"{view.lines} line{'' if view.lines == 1 else 's'}"))
    return rows


def _print_summary(console: Console, view: SymbolView) -> None:
    rows = _summary_rows(view)
    if not rows:
        return
    console.print()
    for label, text in rows:
        console.print(Text(f"  {label.ljust(14)}{text}"), soft_wrap=True)


def print_symbol(console: Console, view: SymbolView, limit: int = DEFAULT_LIMIT, brief: bool = False) -> None:
    node = view.node
    console.print(f"[bold]{escape(node.kind.value)}  {escape(node.qualname)}[/bold]\n[dim]{escape(LEGEND)}[/dim]")
    console.print(claim_line(view.definition))
    if view.signature is not None:
        console.print(f"  signature: {escape(node.name + view.signature)}")
    if view.decorators:
        console.print(f"  decorators: {escape(', '.join(view.decorators))}")
    console.print(claim_line(view.purpose))
    _print_summary(console, view)
    if view.members:
        _section(console, "Contains", view.members, limit)
    if brief:
        if view.unresolved_rows:
            console.print(f"\n  [dim]{len(view.unresolved_rows)} unresolved call(s) not shown (omit --brief to list them)[/dim]")
        return
    if view.node.kind.value in ("function", "method", "class"):
        _print_calls(console, view, limit)


def _print_calls(console: Console, view: SymbolView, limit: int) -> None:
    _section(console, "Called by", view.callers, limit, empty="no resolved callers")
    if view.hidden_test_callers:
        console.print(f"  [dim]{view.hidden_test_callers} caller(s) in tests hidden (use --include-tests)[/dim]")
    if view.node.kind.value == "method":
        console.print(f"  [dim]{_SELF_NOTE_ON if view.resolve_v2 else _SELF_NOTE_OFF}[/dim]")
    if view.node.kind.value == "class":
        console.print("  [dim]Calls made inside a class are shown per method; inspect one, e.g. `Class.method`.[/dim]")
        return
    _section(console, "Calls", view.callees, limit, empty="no resolved callees")
    if view.external_calls:
        console.print(f"  [dim]also calls external code: {escape(_compact(view.external_calls))}[/dim]")
    if view.builtin_call_count:
        console.print(f"  [dim]{view.builtin_call_count} call(s) to Python builtins not listed[/dim]")
    _section(console, "Unresolved calls", view.unknown_calls, limit, empty="none")
    console.print("[dim]Unresolved calls exist in the code but Argus cannot say what they call. A short list is not proof of few calls.[/dim]")


def _section(console: Console, title: str, claims: tuple[Claim, ...], limit: int, empty: str | None = None) -> None:
    console.print(f"\n[bold]{title}[/bold] ({len(claims)})")
    for claim in claims[:limit]:
        console.print(claim_line(claim))
    if len(claims) > limit:
        console.print(f"  [dim]+{len(claims) - limit} more (raise --limit)[/dim]")
    if not claims and empty:
        console.print(f"  [dim]{empty}[/dim]")


def _compact(names: tuple[str, ...], shown: int = 8) -> str:
    return ", ".join(names[:shown]) + (f", +{len(names) - shown} more" if len(names) > shown else "")
