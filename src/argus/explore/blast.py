"""The blast-radius tree: the production code as folders, files, classes and functions, each marked with how much depends on it.

The radius of a function is the number of distinct production functions, methods and modules that reach it through resolved calls,
directly or through other callers. It is a lower bound: calls Argus cannot link (`obj.method()`, dictionaries, anything dynamic) are not
followed, so a high number means "at least this much", and a low one proves nothing. The tier (HIGH, MED, LOW) is a label for that
number and says where to be careful, never that a change is wrong. Test code is not counted and not shown.

With a recording, each node also says whether it ran (a function that only started counts), and whether the file it rests on changed
since: then the recording no longer describes it and it is marked stale rather than ran or did not run.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from argus.explore.grouping import is_production_node, node_index
from argus.explore.symbols import call_edges
from argus.model.schema import ArchModel, Confidence, Node, NodeKind
from argus.record.observed import changed_files
from argus.record.store import Recording

HIGH_SHARE = 0.9  # HIGH: among the most depended-on tenth of the functions ...
HIGH_FLOOR = 5  # ... and at least this many dependents, so a small repository has no HIGH by accident
MED_FLOOR = 3
FUNCTIONS = (NodeKind.FUNCTION, NodeKind.METHOD)
Tier = Literal["HIGH", "MED", "LOW"]
TIER_RANK = {"HIGH": 2, "MED": 1, "LOW": 0}


@dataclass(frozen=True, slots=True)
class BlastNode:
    kind: str  # folder, file, class, function or method
    name: str
    path: str  # repo-relative folder or file; for a symbol, the file that holds it
    qualname: str | None  # symbols only
    line: int | None  # symbols only
    radius: int  # dependents (see module doc); for a container, the largest radius inside it
    direct: int  # functions that call it directly; 0 for a container
    tier: Tier
    ran: int | None  # functions that started in the recording (1 or 0 for a function); None without a recording
    total: int  # functions counted in `ran`
    stale: bool  # a file this node rests on changed after the recording
    children: tuple[BlastNode, ...]


@dataclass(frozen=True, slots=True)
class BlastTree:
    roots: tuple[BlastNode, ...]
    high_at: int  # the radius from which a function is HIGH in this repository
    has_recording: bool


def _reverse_calls(model: ArchModel, nodes: dict[str, Node]) -> dict[str, set[str]]:
    """Callee id -> the production nodes that call it, over resolved calls only (a call that resolves to a module is not a call to a function)."""
    callers: defaultdict[str, set[str]] = defaultdict(set)
    for edge in call_edges(model):
        src, dst = nodes.get(edge.src_id), nodes.get(edge.dst_id)
        if src and dst and is_production_node(src) and is_production_node(dst) and edge.confidence is not Confidence.AMBIGUOUS and src.id != dst.id:
            callers[dst.id].add(src.id)
    return callers


def _reach(callers: dict[str, set[str]], start: str) -> int:
    seen: set[str] = set()
    queue = deque([start])
    while queue:
        for caller in callers.get(queue.popleft(), ()):
            if caller not in seen:
                seen.add(caller)
                queue.append(caller)
    return len(seen)


def _tier(radius: int, high_at: int) -> Tier:
    return "HIGH" if radius >= high_at else "MED" if radius >= MED_FLOOR else "LOW"


def build_blast_tree(model: ArchModel, root: Path, recording: Recording | None = None) -> BlastTree:
    """The tree for the production code of `model`, marked from `recording` when there is one. An empty repository gives no roots."""
    nodes = node_index(model)
    callers = _reverse_calls(model, nodes)
    symbols = [n for n in model.nodes if n.kind in (*FUNCTIONS, NodeKind.CLASS) and is_production_node(n)]
    radius = {n.id: _reach(callers, n.id) for n in symbols}
    members_of: defaultdict[str, list[Node]] = defaultdict(list)
    for n in symbols:
        if n.parent_id:
            members_of[n.parent_id].append(n)
    function_radii = sorted(radius[n.id] for n in symbols if n.kind in FUNCTIONS)
    p90 = function_radii[int(HIGH_SHARE * len(function_radii))] if function_radii else 0
    high_at = max(HIGH_FLOOR, p90)

    files = {n.provenance.file for n in model.nodes if n.kind is NodeKind.MODULE and is_production_node(n)}
    ran_ids = {f.node_id for f in recording.functions if f.node_id and f.calls > 0} if recording else None
    stale = changed_files(recording, root, files) if recording else frozenset()

    def symbol_node(node: Node) -> BlastNode | None:
        parent = nodes.get(node.parent_id) if node.parent_id else None
        if parent is not None and parent.kind in FUNCTIONS:  # a function inside a function belongs to its parent
            return None
        own = radius[node.id]
        if node.kind is NodeKind.CLASS:
            members = sorted(
                (m for m in (symbol_node(c) for c in members_of[node.id]) if m), key=_order,
            )
            top = max([own, *(m.radius for m in members)])
            return _container("class", node.name, node.provenance.file, members, top, high_at, node, stale, recording is not None)
        ran = None if ran_ids is None else int(node.id in ran_ids)
        return BlastNode(
            "method" if node.kind is NodeKind.METHOD else "function", node.name, node.provenance.file, node.qualname, node.provenance.line, own,
            len(callers.get(node.id, ())), _tier(own, high_at), ran, 1, node.provenance.file in stale, (),
        )

    by_file: defaultdict[str, list[BlastNode]] = defaultdict(list)
    for node in sorted(symbols, key=lambda n: (n.provenance.line, n.qualname)):
        parent = nodes.get(node.parent_id) if node.parent_id else None
        if parent is not None and parent.kind is NodeKind.CLASS:
            continue  # reached through its class
        built = symbol_node(node)
        if built is not None:
            by_file[node.provenance.file].append(built)
    file_nodes = [
        _container("file", PurePosixPath(f).name, f, sorted(by_file.get(f, []), key=_order), None, high_at, None, stale, recording is not None)
        for f in sorted(files)
    ]
    return BlastTree(tuple(_folders(file_nodes, high_at, recording is not None)), high_at, recording is not None)


def _order(node: BlastNode) -> tuple[int, str]:
    return (-node.radius, node.name)


def _container(
    kind: str, name: str, path: str, children: list[BlastNode], radius: int | None, high_at: int, symbol: Node | None,
    stale: frozenset[str], has_recording: bool,
) -> BlastNode:
    top = max([radius or 0, *(c.radius for c in children)])
    total = sum(c.total for c in children)
    ran = sum(c.ran or 0 for c in children) if has_recording else None
    return BlastNode(
        kind, name, path, symbol.qualname if symbol else None, symbol.provenance.line if symbol else None, top, 0, _tier(top, high_at), ran, total,
        path in stale or any(c.stale for c in children), tuple(children),
    )


def _folders(files: list[BlastNode], high_at: int, has_recording: bool) -> list[BlastNode]:
    """Group file nodes by folder. A folder whose only child is a folder is joined with it (`src/pkg`), so the tree starts where the code does."""

    def group(items: list[BlastNode], depth: int, prefix: str) -> list[BlastNode]:
        direct = [f for f in items if len(PurePosixPath(f.path).parts) == depth + 1]
        folders: defaultdict[str, list[BlastNode]] = defaultdict(list)
        for f in items:
            parts = PurePosixPath(f.path).parts
            if len(parts) > depth + 1:
                folders[parts[depth]].append(f)
        built = list(direct)
        for name, inside in sorted(folders.items()):
            children = group(inside, depth + 1, f"{prefix}{name}/")
            while len(children) == 1 and children[0].kind == "folder":  # join a chain of single folders
                name, children = f"{name}/{children[0].name}", list(children[0].children)
            path = f"{prefix}{name}"
            built.append(_container("folder", name, path, sorted(children, key=_order), None, high_at, None, frozenset(), has_recording))
        return sorted(built, key=_order)

    return group(files, 0, "")
