"""Data behind the ASCII maps: subsystems, components, import arrows and layers. No printing here.

Containment (package -> component -> module) and imports (A imports B) are kept as separate things.
Arrows are always import edges; nothing here knows or implies execution order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from argus.explore.claims import Claim, Evidence, Tier
from argus.explore.components import analyze, default_group
from argus.explore.entrypoints import Command, Entrypoints
from argus.explore.grouping import MODULE_KINDS, is_production_node
from argus.model.schema import ArchModel, Confidence

COLLAPSE_ABOVE_MODULES = 5
INIT_NAME = "__init__"
OPEN_ALL = "*"  # `--open` on the wrapper package of a one-package repository: expand every subsystem
SINGLE_MODULE = "(single module)"  # shown for a component that is one file, not a package
DOCSTRING_PREFIX = 'docstring: "'
IMPORT_DYNAMIC_LIMITATION = (
    "Imports done through importlib or exec are not seen; arrows include imports inside functions "
    "and TYPE_CHECKING blocks."
)


UNNAMED_GROUP = "(unnamed)"


def top_group(module: str) -> str:
    """First non-empty dotted segment, so a malformed name (leading dot) can never yield a blank subsystem."""
    return next((part for part in module.split(".") if part), UNNAMED_GROUP)


@dataclass(frozen=True, slots=True)
class Arrow:
    """`src` imports `dst`; `count` import edges, `claim` cites the first import site."""

    src: str
    dst: str
    count: int
    claim: Claim

    @property
    def evidence(self) -> Evidence:
        return self.claim.evidence[0]


@dataclass(frozen=True, slots=True)
class Entry:
    label: str
    module: str
    commands: int
    claim: Claim


@dataclass(frozen=True, slots=True)
class Part:
    """A component inside a subsystem."""

    name: str
    short: str
    modules: tuple[str, ...]
    purpose: Claim

    @property
    def collapsed(self) -> bool:
        return len(self.modules) > COLLAPSE_ABOVE_MODULES


@dataclass(frozen=True, slots=True)
class Subsystem:
    name: str
    purpose: Claim
    parts: tuple[Part, ...]
    entries: tuple[Entry, ...]


@dataclass(frozen=True, slots=True)
class SystemView:
    label: str
    files: int
    subsystems: tuple[Subsystem, ...]
    arrows: tuple[Arrow, ...]
    opened: tuple[str, str] | None
    unknowns: tuple[Claim, ...]


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    shown: str
    modules: int
    cycle: tuple[str, ...]
    out: tuple[Arrow, ...]
    purpose: str = ""  # quoted docstring line or "no docstring ...": only filled for the level map
    commands: int = 0  # CLI commands defined in this component: only filled for the level map


@dataclass(frozen=True, slots=True)
class DepsView:
    label: str
    scope: str
    scope_purpose: Claim | None
    layers: tuple[tuple[Node, ...], ...]
    outward: tuple[Arrow, ...]
    inward: tuple[Arrow, ...]
    opened: bool
    unknowns: tuple[Claim, ...]
    detailed: bool = False  # the level map: boxes carry a purpose line and an entry badge


def purpose_text(purpose: Claim) -> str:
    """The quoted docstring line, or an explicit statement that none exists."""
    if purpose.tier is Tier.UNKNOWN:
        return "no docstring: purpose not stated"
    text = purpose.text
    return text[len(DOCSTRING_PREFIX):-1] if text.startswith(DOCSTRING_PREFIX) else text


def component_names(model: ArchModel) -> set[str]:
    return {default_group(n.qualname) for n in model.nodes if n.kind in MODULE_KINDS and is_production_node(n)}


def resolve_open(model: ArchModel, name: str) -> tuple[str, str]:
    """('subsystem' | 'component', label) for user input; LookupError lists the options."""
    comps = component_names(model)
    subsystems = {top_group(c) for c in comps}
    if name in subsystems:
        return "subsystem", name
    if name in comps:
        return "component", name
    hits = sorted({x for x in subsystems | comps if x.endswith("." + name)})
    if len(hits) == 1:
        return resolve_open(model, hits[0])
    options = ", ".join(sorted(subsystems) + sorted(comps - subsystems))
    raise LookupError(f"{'ambiguous' if hits else 'no subsystem or component named'} {name!r}; options: {options}")


def layer_groups(names: set[str], pairs: set[tuple[str, str]]) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Import layering: importers above what they import; import cycles collapsed into one group.

    Layer = longest import path from the top, so every arrow points to a strictly higher layer number.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(names))
    graph.add_edges_from(sorted(pairs))
    condensed = nx.condensation(graph)
    members = {n: tuple(sorted(condensed.nodes[n]["members"])) for n in condensed}
    return tuple(
        tuple(sorted((members[n] for n in generation), key=lambda group: group[0]))
        for generation in nx.topological_generations(condensed)
    )


def build_system(model: ArchModel, root: Path, entrypoints: Entrypoints, open_name: str | None) -> SystemView:
    opened = resolve_open(model, open_name) if open_name else None
    components = analyze(model, root, default_group)
    tops = analyze(model, root, top_group)
    # One package holding everything would be one box with no arrows. When its parts import each other, they become the subsystems.
    flat = len(tops) == 1 and any(c.imports_out for c in components)
    if flat and opened:  # the wrapper package opens every part; a component name opens that one part
        opened = ("subsystem", OPEN_ALL if opened == ("subsystem", tops[0].name) else opened[1])
    group = default_group if flat else top_group
    subs = components if flat else tops
    units = analyze(model, root, lambda m: m) if flat else components
    inits = {n.qualname for n in model.nodes if n.kind in MODULE_KINDS and n.provenance.file.endswith("__init__.py")}
    arrows = tuple(Arrow(c.name, d.other, d.count, d.claim) for c in subs for d in c.imports_out)
    entries = _entries(entrypoints)
    subsystems = {
        t.name: Subsystem(
            t.name, t.purpose,
            tuple(sorted(
                (Part(c.name, _short(c.name, t.name, inits), c.modules, c.purpose) for c in units if group(c.name) == t.name),
                key=lambda p: p.name,
            )),
            tuple(e for e in entries if group(e.module) == t.name),
        )
        for t in subs
    }
    layers = layer_groups(set(subsystems), {(a.src, a.dst) for a in arrows})
    ordered = tuple(subsystems[name] for layer in layers for group in layer for name in group)
    return SystemView(root.name, model.stats.files_scanned, ordered, arrows, opened, _unknowns(model, entrypoints))


def build_deps(model: ArchModel, root: Path, open_name: str | None) -> DepsView:
    names = component_names(model)
    subsystems = sorted({top_group(n) for n in names})
    if not subsystems:
        raise ValueError("no production modules found to map")
    if open_name is None and len(subsystems) > 1:
        return _deps(model, root, None, None, top_group, lambda m: True, "all subsystems")
    kind, label = resolve_open(model, open_name) if open_name else ("subsystem", subsystems[0])
    if kind == "subsystem":
        return _deps(
            model, root, kind, label,
            lambda m: default_group(m) if top_group(m) == label else top_group(m),
            lambda m: top_group(m) == label, f"subsystem {label}",
        )
    modules = [m for m in _production_modules(model) if default_group(m) == label]
    if len(modules) < 2:
        raise ValueError(f"{label} has 1 module; there is nothing to expand")
    return _deps(
        model, root, kind, label,
        lambda m: m if default_group(m) == label else (default_group(m) if top_group(m) == top_group(label) else top_group(m)),
        lambda m: default_group(m) == label, f"component {label} (modules)",
    )


def build_levels(model: ArchModel, root: Path, entrypoints: Entrypoints) -> DepsView:
    """Every component of the repository in import levels, with each component's purpose and CLI commands."""
    if not component_names(model):
        raise ValueError("no production modules found to map")
    return _deps(model, root, None, None, default_group, lambda m: True, "all components", entrypoints)


def _deps(
    model: ArchModel, root: Path, kind: str | None, label: str | None,
    grouper: Callable[[str], str], inside: Callable[[str], bool], scope_text: str, entrypoints: Entrypoints | None = None,
) -> DepsView:
    comps = {c.name: c for c in analyze(model, root, grouper)}
    inner = {grouper(m) for m in _production_modules(model) if inside(m)}
    arrows = {
        name: tuple(Arrow(name, d.other, d.count, d.claim) for d in comps[name].imports_out) for name in sorted(inner)
    }
    inner_pairs = {(a.src, a.dst) for out in arrows.values() for a in out if a.dst in inner}
    outward = tuple(a for out in arrows.values() for a in out if a.dst not in inner)
    inward = tuple(
        Arrow(d.other, name, d.count, d.claim)
        for name in sorted(inner) for d in comps[name].imported_by if d.other not in inner
    )
    layers = tuple(
        tuple(
            Node(
                n, _relative(n, kind, label), len(comps[n].modules), group if len(group) > 1 else (),
                tuple(sorted((a for a in arrows[n] if a.dst in inner), key=lambda a: (-a.count, a.dst))),
                *_detail(comps[n], n, grouper, entrypoints),
            )
            for group in layer for n in group
        )
        for layer in layer_groups(inner, inner_pairs)
    )
    if kind == "subsystem":
        purpose = _top_purpose(model, root, label)
    else:
        purpose = comps[label].purpose if label in comps else None
    return DepsView(root.name, scope_text, purpose, layers, outward, inward, kind is not None, _unknowns(model, None), entrypoints is not None)


def _detail(component, name: str, grouper: Callable[[str], str], entrypoints: Entrypoints | None) -> tuple[str, int]:
    """(purpose, commands) for a level-map box; empty for the other dependency views."""
    if entrypoints is None:
        return "", 0
    return purpose_text(component.purpose), sum(1 for c in entrypoints.commands if grouper(c.module) == name)


def _top_purpose(model: ArchModel, root: Path, label: str) -> Claim | None:
    return next((c.purpose for c in analyze(model, root, top_group) if c.name == label), None)


def _relative(name: str, kind: str | None, label: str | None) -> str:
    if label is None:
        return name
    if name == label:
        return INIT_NAME
    return name[len(label) + 1:] if name.startswith(label + ".") else name


def _short(name: str, subsystem: str, inits: set[str]) -> str:
    if name != subsystem:
        return name[len(subsystem) + 1:]
    return INIT_NAME if name in inits else SINGLE_MODULE


def _production_modules(model: ArchModel) -> list[str]:
    return sorted({n.qualname for n in model.nodes if n.kind in MODULE_KINDS and is_production_node(n)})


def _entries(eps: Entrypoints) -> tuple[Entry, ...]:
    counts = Counter(c.display.split()[0] for c in eps.commands)
    if eps.scripts:
        return tuple(Entry(s.name, s.module, counts.get(s.name, 0), s.claim) for s in eps.scripts)
    first: dict[str, Command] = {}
    for command in eps.commands:
        first.setdefault(command.display.split()[0], command)
    return tuple(
        Entry(label, label.split(":")[0], counts[label], Claim(f"root app {label}", Tier.FACT, first[label].claim.evidence[-1:]))
        for label in sorted(counts)
    )


def _unknowns(model: ArchModel, eps: Entrypoints | None) -> tuple[Claim, ...]:
    out: list[Claim] = []
    if model.stats.files_failed:
        out.append(Claim(f"{model.stats.files_failed} file(s) failed to parse and are missing from this map", Tier.UNKNOWN))
    ambiguous = sum(1 for e in model.edges if e.confidence is Confidence.AMBIGUOUS)
    if ambiguous:
        out.append(Claim(f"{ambiguous} relationship(s) of any kind in the model could not be resolved; maps may omit them", Tier.UNKNOWN))
    if eps is not None and eps.unknowns:
        out.append(Claim(
            f"entry-point list may be incomplete ({len(eps.unknowns)} unresolved item(s)); run `argus explore entrypoints`",
            Tier.UNKNOWN,
        ))
    out.append(Claim(IMPORT_DYNAMIC_LIMITATION, Tier.UNKNOWN))
    return tuple(out)
