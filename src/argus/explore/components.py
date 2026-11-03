"""Components: modules grouped by package, with docstring purpose and import dependencies.

Purpose text is only ever quoted from a docstring (FACT) or reported as UNKNOWN; it is never written for the code.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from argus.explore.claims import Claim, Tier, evidence_at, tier_for
from argus.explore.grouping import IMPORT_KINDS, MODULE_KINDS, default_group, is_production_node, node_index, owner_module
from argus.model.ids import is_unresolved
from argus.model.schema import ArchModel, Confidence, Edge, Node, NodeKind

__all__ = ["Component", "Dependency", "Symbol", "analyze", "default_group", "public_symbols", "resolve_name"]

_WEAKEST_FIRST = (Confidence.AMBIGUOUS, Confidence.INFERRED, Confidence.EXTRACTED)


@dataclass(frozen=True, slots=True)
class Dependency:
    other: str
    count: int
    claim: Claim


@dataclass(frozen=True, slots=True)
class Component:
    name: str
    modules: tuple[str, ...]
    purpose: Claim
    imports_out: tuple[Dependency, ...]
    imported_by: tuple[Dependency, ...]


@dataclass(frozen=True, slots=True)
class Symbol:
    module: str
    name: str
    kind: str
    claim: Claim


def analyze(model: ArchModel, root: Path, group_of: Callable[[str], str]) -> tuple[Component, ...]:
    """All production components under `group_of`, ordered by how many others import them, then name."""
    nodes = node_index(model)
    modules = {n.qualname: n for n in model.nodes if n.kind in MODULE_KINDS and is_production_node(n)}
    members: dict[str, list[str]] = defaultdict(list)
    for qualname in sorted(modules):
        members[group_of(qualname)].append(qualname)
    out, into = _dependencies(model, nodes, modules, group_of)
    components = [
        Component(
            name, tuple(mods), _purpose(root, name, modules),
            _as_deps(root, out.get(name, {}), "imports"), _as_deps(root, into.get(name, {}), "is imported by"),
        )
        for name, mods in members.items()
    ]
    return tuple(sorted(components, key=lambda c: (-len(c.imported_by), c.name)))


def _dependencies(
    model: ArchModel, nodes: dict[str, Node], modules: dict[str, Node], group_of: Callable[[str], str]
) -> tuple[dict[str, dict[str, list[Edge]]], dict[str, dict[str, list[Edge]]]]:
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    into: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for edge in model.edges:
        if edge.kind not in IMPORT_KINDS or is_unresolved(edge.dst_id):
            continue
        src, dst = nodes.get(edge.src_id), nodes.get(edge.dst_id)
        if not (src and dst):
            continue
        src_mod, dst_mod = owner_module(src, nodes), owner_module(dst, nodes)
        if src_mod not in modules or dst_mod not in modules:
            continue
        a, b = group_of(src_mod), group_of(dst_mod)
        if a != b:
            out[a][b].append(edge)
            into[b][a].append(edge)
    return out, into


def _as_deps(root: Path, by_other: dict[str, list[Edge]], verb: str) -> tuple[Dependency, ...]:
    deps = []
    for other, edges in by_other.items():
        edges = sorted(edges, key=lambda e: (e.provenance.file, e.provenance.line))
        weakest = min((e.confidence for e in edges), key=_WEAKEST_FIRST.index)
        first = next(e for e in edges if e.confidence is weakest).provenance  # cite a site as weak as the claim
        plural = "" if len(edges) == 1 else "s"
        claim = Claim(
            f"{verb} {other} ({len(edges)} import edge{plural})", tier_for(weakest),
            (evidence_at(root, first.file, first.line),),
        )
        deps.append(Dependency(other, len(edges), claim))
    return tuple(sorted(deps, key=lambda d: (-d.count, d.other)))


def _first_paragraph(doc: str) -> str:
    """First docstring paragraph on one line, so a sentence wrapped over two lines is never cut in half."""
    lines: list[str] = []
    for line in doc.strip().splitlines():
        if not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines)


def _purpose(root: Path, name: str, modules: dict[str, Node]) -> Claim:
    node = modules.get(name)
    doc = (node.attrs.get("docstring") if node else None) or ""
    first = _first_paragraph(doc)
    if node is None or not first:
        return Claim(f"{name} has no docstring of its own, so its responsibility is not stated at this level", Tier.UNKNOWN)
    return Claim(f'docstring: "{first}"', Tier.FACT, (evidence_at(root, node.provenance.file, node.provenance.line),))


def public_symbols(model: ArchModel, root: Path, modules: tuple[str, ...]) -> tuple[Symbol, ...]:
    """Public top-level classes and functions of the given modules, each quoted with its first docstring line."""
    wanted = set(modules)
    nodes = node_index(model)
    out = []
    for node in model.nodes:
        if node.kind not in (NodeKind.CLASS, NodeKind.FUNCTION) or not node.is_public:
            continue
        parent = nodes.get(node.parent_id) if node.parent_id else None
        if parent is None or parent.kind not in MODULE_KINDS or parent.qualname not in wanted:
            continue
        doc = next((x.strip() for x in (node.attrs.get("docstring") or "").splitlines() if x.strip()), "")
        label = f"{node.kind.value} {node.name}" + (f" - {doc}" if doc else "")
        out.append(Symbol(
            parent.qualname, node.name, node.kind.value,
            Claim(label, Tier.FACT, (evidence_at(root, node.provenance.file, node.provenance.line),)),
        ))
    return tuple(sorted(out, key=lambda s: (s.module, s.claim.evidence[0].line)))


def resolve_name(model: ArchModel, name: str) -> tuple[str, Callable[[str], str]]:
    """Resolve user input to (label, group_of): an exact component, an exact module, or a unique suffix.

    Raises LookupError listing candidates when nothing or more than one thing matches.
    """
    modules = sorted(n.qualname for n in model.nodes if n.kind in MODULE_KINDS and is_production_node(n))
    groups = sorted({default_group(m) for m in modules})
    if name in groups:
        return name, default_group
    if name in modules:
        return name, lambda module: module
    suffix = "." + name
    hits = sorted({g for g in groups if g.endswith(suffix)} | {m for m in modules if m.endswith(suffix)})
    if len(hits) == 1:
        return resolve_name(model, hits[0])
    raise LookupError(
        f"{'ambiguous' if hits else 'no component or module named'} {name!r}; "
        f"{'candidates' if hits else 'components'}: {', '.join(hits or groups)}"
    )
