"""What a function calls, per the model's resolved call edges. Every hop is an INFERENCE with a call site."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from argus.explore.claims import Claim, evidence_at, tier_for
from argus.explore.grouping import default_group, is_production_node, node_index, owner_module
from argus.model.ids import is_unresolved
from argus.model.schema import ArchModel, Confidence, Edge, EdgeKind, NodeKind

CALLABLE_KINDS = (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS)
EXPANDABLE_KINDS = (NodeKind.FUNCTION, NodeKind.METHOD)


@dataclass(frozen=True, slots=True)
class CallStep:
    depth: int
    caller_id: str
    callee_id: str
    callee: str
    kind: str
    module: str
    group: str
    claim: Claim


class CallGraph:
    """Resolved production call edges of one model; AMBIGUOUS edges are excluded, not presented as calls."""

    def __init__(self, model: ArchModel) -> None:
        self.nodes = node_index(model)
        self.by_src: dict[str, list[Edge]] = defaultdict(list)
        for edge in model.edges:
            if edge.kind is not EdgeKind.CALLS or edge.confidence is Confidence.AMBIGUOUS or is_unresolved(edge.dst_id):
                continue
            src, dst = self.nodes.get(edge.src_id), self.nodes.get(edge.dst_id)
            if src and dst and dst.kind in CALLABLE_KINDS and is_production_node(src) and is_production_node(dst):
                self.by_src[edge.src_id].append(edge)
        for edges in self.by_src.values():
            edges.sort(key=lambda e: (e.provenance.line, e.dst_id))

    def step(self, root: Path, edge: Edge, depth: int) -> CallStep | None:
        callee = self.nodes[edge.dst_id]
        module = owner_module(callee, self.nodes)
        if module is None:
            return None
        caller = self.nodes[edge.src_id]
        site = evidence_at(root, edge.provenance.file, edge.provenance.line)
        return CallStep(
            depth, edge.src_id, edge.dst_id, callee.qualname, callee.kind.value, module, default_group(module),
            Claim(f"{caller.qualname} calls {callee.qualname}", tier_for(edge.confidence), (site,)),
        )


def call_tree(model: ArchModel, root: Path, function_id: str, max_depth: int = 2) -> tuple[CallStep, ...]:
    """Depth-first, call-site ordered tree of resolved callees; each callee is expanded once."""
    calls = CallGraph(model)
    out: list[CallStep] = []
    seen = {function_id}

    def visit(current: str, depth: int) -> None:
        if depth > max_depth:
            return
        for edge in calls.by_src.get(current, ()):
            if edge.dst_id in seen:
                continue
            seen.add(edge.dst_id)
            step = calls.step(root, edge, depth)
            if step is None:
                continue
            out.append(step)
            if calls.nodes[edge.dst_id].kind in EXPANDABLE_KINDS:
                visit(edge.dst_id, depth + 1)

    visit(function_id, 1)
    return tuple(out)


def reach_tree(
    model: ArchModel, root: Path, function_id: str, graph: CallGraph | None = None
) -> dict[str, tuple[CallStep, ...]]:
    """Shortest call chain from `function_id` to every function/class it can reach (breadth-first)."""
    calls = graph or CallGraph(model)
    chains: dict[str, tuple[CallStep, ...]] = {function_id: ()}
    queue = deque([function_id])
    while queue:
        current = queue.popleft()
        for edge in calls.by_src.get(current, ()):
            if edge.dst_id in chains:
                continue
            step = calls.step(root, edge, len(chains[current]) + 1)
            if step is None:
                continue
            chains[edge.dst_id] = (*chains[current], step)
            if calls.nodes[edge.dst_id].kind in EXPANDABLE_KINDS:
                queue.append(edge.dst_id)
    del chains[function_id]
    return chains


def _reachable(graph: CallGraph, start: str) -> set[str]:
    """Every function or class reachable from `start` through resolved calls. No evidence is built: only the answer is needed."""
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for edge in graph.by_src.get(current, ()):
            if edge.dst_id in seen:
                continue
            seen.add(edge.dst_id)
            if graph.nodes[edge.dst_id].kind in EXPANDABLE_KINDS:
                queue.append(edge.dst_id)
    return seen


def reached_displays(model: ArchModel, root: Path, entries, targets: set[str]) -> dict[str, tuple[str, ...]]:
    """For each target function id: the display names of the entry points (CLI commands, web routes) whose calls reach it, in the order of
    `entries`, each name once. A target nothing reaches is absent. Every hop is a resolved call, so the answer is a lower bound."""
    graph = CallGraph(model)
    reachable: dict[str, set[str]] = {}
    found: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        if entry.function_id not in reachable:
            reachable[entry.function_id] = _reachable(graph, entry.function_id) if entry.function_id in graph.nodes else set()
        for target in targets:
            if (target == entry.function_id or target in reachable[entry.function_id]) and entry.display not in found[target]:
                found[target].append(entry.display)
    return {target: tuple(names) for target, names in found.items() if names}
