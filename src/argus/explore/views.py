"""Assemble the data behind `explore map` and `explore component` (no printing here)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from argus.explore.claims import Claim, Tier
from argus.explore.components import Component, Symbol, analyze, public_symbols
from argus.explore.entrypoints import Command, Entrypoints
from argus.explore.grouping import default_group
from argus.explore.reach import CallGraph, CallStep, reach_tree
from argus.model.schema import ArchModel, Confidence

METHOD_CALL_LIMITATION = (
    "Calls made through `self.x()` or `obj.method()` are not extracted, so call chains are incomplete "
    "and a missing chain does not mean no call exists."
)
DYNAMIC_LIMITATION = "Dynamic dispatch, decorators other than Typer's, and runtime registration are not followed."


class EmptyComponentError(LookupError):
    """The named component exists as a name but has no production modules."""


@dataclass(frozen=True, slots=True)
class ComponentRow:
    component: Component
    reached_from_cli: bool


@dataclass(frozen=True, slots=True)
class MapView:
    files_scanned: int
    entrypoints: Entrypoints
    rows: tuple[ComponentRow, ...]
    unknowns: tuple[Claim, ...]


@dataclass(frozen=True, slots=True)
class EntryReach:
    command: Command
    chains: tuple[tuple[CallStep, ...], ...]


@dataclass(frozen=True, slots=True)
class DetailView:
    label: str
    component: Component
    symbols: tuple[Symbol, ...]
    own_commands: tuple[Command, ...]
    reached_from: tuple[EntryReach, ...]
    unknowns: tuple[Claim, ...]


def build_map(model: ArchModel, root: Path, entrypoints: Entrypoints) -> MapView:
    graph = CallGraph(model)
    reached = _cli_reached_groups(model, root, entrypoints.commands, graph)
    rows = tuple(ComponentRow(c, c.name in reached) for c in analyze(model, root, default_group))
    return MapView(model.stats.files_scanned, entrypoints, rows, _map_unknowns(model, entrypoints))


def build_detail(
    model: ArchModel, root: Path, entrypoints: Entrypoints, label: str, group_of: Callable[[str], str]
) -> DetailView:
    component = next((c for c in analyze(model, root, group_of) if c.name == label), None)
    if component is None:
        raise EmptyComponentError(f"{label!r} has no production modules")
    graph = CallGraph(model)
    members = set(component.modules)
    own = tuple(c for c in entrypoints.commands if c.module in members)
    reached = []
    for command in entrypoints.commands:
        chains = _chains_into(model, root, command, members, graph)
        if chains:
            reached.append(EntryReach(command, chains))
    return DetailView(
        label, component, public_symbols(model, root, component.modules), own, tuple(reached),
        (Claim(METHOD_CALL_LIMITATION, Tier.UNKNOWN),),
    )


def _chains_into(
    model: ArchModel, root: Path, command: Command, members: set[str], graph: CallGraph
) -> tuple[tuple[CallStep, ...], ...]:
    """For each member module other than the command's own, the shortest call chain that first lands in it."""
    best: dict[str, tuple[CallStep, ...]] = {}
    for chain in reach_tree(model, root, command.function_id, graph).values():
        module = chain[-1].module
        if module in members and module != command.module and (module not in best or len(chain) < len(best[module])):
            best[module] = chain
    return tuple(best[m] for m in sorted(best))


def _cli_reached_groups(
    model: ArchModel, root: Path, commands: tuple[Command, ...], graph: CallGraph
) -> set[str]:
    groups = {default_group(c.module) for c in commands}
    for command in commands:
        groups |= {chain[-1].group for chain in reach_tree(model, root, command.function_id, graph).values()}
    return groups


def _map_unknowns(model: ArchModel, entrypoints: Entrypoints) -> tuple[Claim, ...]:
    unknowns = list(entrypoints.unknowns)
    if model.stats.files_failed:
        unknowns.append(Claim(f"{model.stats.files_failed} file(s) failed to parse and are missing from this map", Tier.UNKNOWN))
    ambiguous = sum(1 for e in model.edges if e.confidence is Confidence.AMBIGUOUS)
    if ambiguous:
        unknowns.append(Claim(f"{ambiguous} relationship(s) exist but their target could not be resolved", Tier.UNKNOWN))
    unknowns += [Claim(METHOD_CALL_LIMITATION, Tier.UNKNOWN), Claim(DYNAMIC_LIMITATION, Tier.UNKNOWN)]
    return tuple(unknowns)
