"""Terminal rendering. Every line shows its tier and, where it has one, `file:line` evidence."""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.markup import escape

from argus.explore.claims import Claim, Tier
from argus.explore.components import Component, Dependency
from argus.explore.entrypoints import Command, Entrypoints
from argus.explore.reach import CallStep
from argus.explore.views import DetailView, MapView

STYLE = {Tier.FACT: "green", Tier.INFERENCE: "yellow", Tier.UNKNOWN: "red"}
LEGEND = (
    "[FACT] read directly from source   [INFERENCE] derived by resolution, can be wrong   "
    "[UNKNOWN] could not be determined"
)
MAX_DEPS_SHOWN = 8
CALL_TREE_LIMITATION = "Calls through self/obj methods are not followed, so this list can be incomplete."


def short_name(qualname: str) -> str:
    """The last two dotted parts: `Class.method`, or `module.function` for a plain function."""
    return ".".join(qualname.split(".")[-2:])


def claim_line(claim: Claim, indent: int = 2) -> str:
    style = STYLE[claim.tier]
    where = "  ".join(e.location for e in claim.evidence[:1])
    extra = f" (+{len(claim.evidence) - 1} more)" if len(claim.evidence) > 1 else ""
    pad = " " * indent
    return f"{pad}[{style}]\\[{claim.tier.value}][/{style}] {escape(claim.text)}" + (f"  [dim]{escape(where)}{extra}[/dim]" if where else "")


def print_entrypoints(console: Console, eps: Entrypoints, trace: list[tuple[Command, tuple[CallStep, ...]]] | None) -> None:
    console.print("[bold]Entry points[/bold]")
    console.print(f"[dim]{escape(LEGEND)}[/dim]")
    for script in eps.scripts:
        console.print(claim_line(script.claim))
    for command in eps.commands:
        console.print(f"  [green]\\[FACT][/green] [bold]{escape(command.display)}[/bold] -> {escape(command.function_id.split(':', 1)[1])}  "
                      f"[dim]{escape(command.file)}:{command.line}[/dim]")
    for route in eps.routes:
        console.print(f"  [green]\\[FACT][/green] [bold]{escape(route.display)}[/bold] -> {escape(route.function_id.split(':', 1)[1])}  "
                      f"[dim]{escape(route.file)}:{route.line}[/dim]")
    for program in eps.programs:
        console.print(f"  [green]\\[FACT][/green] [bold]{escape(program.display)}[/bold] -> {escape(program.function_id.split(':', 1)[1])}  "
                      f"[dim]{escape(program.file)}:{program.line}[/dim]")
    for guard in eps.main_guards:
        console.print(claim_line(guard))
    for command, steps in trace or []:
        _print_trace(console, command, steps)
    if eps.unknowns:
        console.print("\n[bold]Unknown[/bold]")
        for claim in eps.unknowns:
            console.print(claim_line(claim))


def _print_trace(console: Console, command: Command, steps: tuple[CallStep, ...]) -> None:
    console.print(f"\n[bold]What `{escape(command.display)}` calls[/bold] [dim](resolved call edges, depth-limited)[/dim]")
    console.print(claim_line(command.claim))
    for step in steps:
        pad = "  " * (step.depth + 1)
        site = step.claim.evidence[0].location
        style = STYLE[step.claim.tier]
        console.print(f"{pad}[{style}]\\[{step.claim.tier.value}][/{style}] {escape(step.callee)} [dim]({escape(step.group)}, {escape(step.kind)}) called at {escape(site)}[/dim]")
    console.print(claim_line(Claim(CALL_TREE_LIMITATION, Tier.UNKNOWN)))


def print_map(console: Console, view: MapView, root_label: str) -> None:
    console.print(f"[bold]Architecture map[/bold] {escape(root_label)}  [dim]({view.files_scanned} files, scanned fresh)[/dim]")
    console.print(f"[dim]{escape(LEGEND)}[/dim]\n[bold]Entry points[/bold]")
    for script in view.entrypoints.scripts:
        console.print(claim_line(script.claim))
    top = Counter(c.path[0] if len(c.path) > 1 else "(direct)" for c in view.entrypoints.commands)
    summary = ", ".join(f"{name} ({n})" if name != "(direct)" else f"{n} direct" for name, n in sorted(top.items()))
    incomplete = "  [red](list may be incomplete: see Unknown)[/red]" if view.entrypoints.unknowns else ""
    console.print(f"  [green]\\[FACT][/green] {len(view.entrypoints.commands)} commands found: {escape(summary)}{incomplete}  [dim]run `argus explore entrypoints`[/dim]")
    console.print("\n[bold]Components[/bold] [dim](production code; most-imported first)[/dim]")
    for row in view.rows:
        _print_component_header(console, row.component, row.reached_from_cli)
        console.print(claim_line(row.component.purpose, 4))
    console.print("\n[bold]Unknown[/bold]")
    for claim in view.unknowns:
        console.print(claim_line(claim))


def _print_component_header(console: Console, component: Component, reached: bool) -> None:
    flag = "  [cyan]reachable from CLI via resolved calls[/cyan]" if reached else ""
    console.print(
        f"\n  [bold]{escape(component.name)}[/bold]  {len(component.modules)} module(s)  "
        f"imports {_names(component.imports_out)}  |  imported by {_names(component.imported_by)}{flag}"
    )


def _names(deps: tuple[Dependency, ...]) -> str:
    if not deps:
        return "-"
    shown = ", ".join(f"{escape(d.other)}({d.count})" for d in deps[:MAX_DEPS_SHOWN])
    return shown + (f", +{len(deps) - MAX_DEPS_SHOWN} more" if len(deps) > MAX_DEPS_SHOWN else "")


def print_detail(console: Console, view: DetailView) -> None:
    comp = view.component
    console.print(f"[bold]{escape(view.label)}[/bold]  {len(comp.modules)} module(s)\n[dim]{escape(LEGEND)}[/dim]")
    console.print("\n[bold]Purpose[/bold]")
    console.print(claim_line(comp.purpose))
    console.print("\n[bold]Modules and public symbols[/bold]")
    by_module = {m: [s for s in view.symbols if s.module == m] for m in comp.modules}
    for module, symbols in by_module.items():
        console.print(f"  [bold]{escape(module)}[/bold]")
        for symbol in symbols:
            console.print(claim_line(symbol.claim, 4))
    for title, deps in (("Depends on (imports)", comp.imports_out), ("Used by (imported by)", comp.imported_by)):
        console.print(f"\n[bold]{title}[/bold]")
        for dep in deps or ():
            console.print(claim_line(dep.claim))
        if not deps:
            console.print("  -")
    console.print("\n[bold]Reached from CLI entry points[/bold] [dim](shortest resolved call chain into each module)[/dim]")
    for command in view.own_commands:
        console.print(f"  [green]\\[FACT][/green] defines `{escape(command.display)}`  [dim]{escape(command.file)}:{command.line}[/dim]")
    for reach in view.reached_from:
        console.print(f"  [bold]{escape(reach.command.display)}[/bold]  [dim]{reach.command.file}:{reach.command.line}[/dim]")
        for chain in reach.chains:
            console.print(f"    -> {escape(chain[-1].module)}")
            for step in chain:
                style = STYLE[step.claim.tier]
                console.print(f"       [{style}]\\[{step.claim.tier.value}][/{style}] {escape(step.callee)}  [dim]called at {escape(step.claim.evidence[0].location)}[/dim]")
    if not view.own_commands and not view.reached_from:
        console.print("  [red]\\[UNKNOWN][/red] no entry point reaches this through resolved calls (it may still be used via methods or dynamically)")
    console.print("\n[bold]Unknown[/bold]")
    for claim in view.unknowns:
        console.print(claim_line(claim))
