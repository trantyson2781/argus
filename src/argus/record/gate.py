"""The record-mode gate measure: of the call sites static reading could not link, how many gained an observed target?

A site is "unresolved" when the call is in a function or method of the scan and leaves no call edge (`explore.symbols.unresolved_in`).
It gains a target when the recording saw that function make a call at that line to a function the scan knows, and the target's name is
the last word of the call as written (`x.go(...)` -> `A.go`). That is the only basis counted.
A `position` match (the line holds one unresolved call and one observed target, but the names differ) is only a CANDIDATE: hand checks of
37 of them on Argus and stock_analysis found about four in ten wrong (a neighbouring call on the same line gets the credit), and the rest
were real aliases. Candidates are listed for a hand check and never counted.
A target the scan already linked from that line is left out: it is the call that was seen, not the unresolved one. Nothing else is claimed. Calls that leave the repository (library calls) are in the unresolved total and can never gain a repo target.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from argus.explore.grouping import is_production_node, node_index
from argus.explore.symbols import call_edges, unresolved_in
from argus.extract.python_ast import parse_file
from argus.model.schema import ArchModel, Node, NodeKind
from argus.record.store import Recording

FUNCTION_KINDS = (NodeKind.FUNCTION, NodeKind.METHOD)
Basis = Literal["name", "position"]


@dataclass(frozen=True, slots=True)
class Gain:
    caller: str  # qualnames
    file: str
    line: int
    callee_text: str  # the call as written
    target: str
    basis: Basis
    is_test: bool


@dataclass(frozen=True, slots=True)
class GainReport:
    unresolved: int  # unresolved call sites in functions and methods
    in_code_that_ran: int  # ... whose enclosing function ran at least once
    in_production_that_ran: int  # ... of those, the ones in production code (the rest are in tests)
    internal_looking: int  # ... of those, the ones whose call name is the name of some function in the repo (the only ones that can gain by name)
    gains: tuple[Gain, ...]  # name-confirmed only
    candidates: tuple[Gain, ...]  # position matches: unverified, never counted


def _observed(rec: Recording, nodes: dict[str, Node]) -> tuple[dict[tuple[str, int], set[str]], set[str]]:
    """((caller node id, line) -> observed target node ids, ids of functions that ran). Endpoints the scan could not match are left out."""
    node_of = {f.key: f.node_id for f in rec.functions}
    seen: dict[tuple[str, int], set[str]] = defaultdict(set)
    for s in rec.sites:
        caller, callee = node_of.get(s.caller), node_of.get(s.callee)
        if caller in nodes and callee in nodes:
            seen[(caller, s.line)].add(callee)  # type: ignore[arg-type]
    ran = {f.node_id for f in rec.functions if f.node_id and f.calls > 0}
    return seen, ran


def gain_report(model: ArchModel, root: Path, rec: Recording) -> GainReport:
    nodes = node_index(model)
    seen, ran = _observed(rec, nodes)
    edges_from: defaultdict[str, list] = defaultdict(list)
    for edge in call_edges(model):
        edges_from[edge.src_id].append(edge)
    functions_in: defaultdict[str, list[Node]] = defaultdict(list)
    for node in model.nodes:
        if node.kind in FUNCTION_KINDS and "#" not in node.qualname:  # a redefinition's calls are credited to the first definition
            functions_in[node.provenance.file].append(node)
    modules = {n.provenance.file: n.qualname for n in model.nodes if n.kind is NodeKind.MODULE}

    total = in_ran = in_prod = internal = 0
    repo_names = {n.name for n in model.nodes if n.kind in FUNCTION_KINDS}
    found: list[Gain] = []
    for file, functions in sorted(functions_in.items()):
        module = modules.get(file)
        if module is None:
            continue
        by_local = defaultdict(list)
        for call in parse_file(root / file, module).calls:
            by_local[call.caller_local].append(call)
        for node in functions:
            raw = by_local.get(node.qualname[len(module) + 1 :], [])
            rows, _ = unresolved_in(raw, edges_from[node.id], root, node, True)
            total += len(rows)
            if node.id not in ran:
                continue
            in_ran += len(rows)
            if is_production_node(node):
                in_prod += len(rows)
                internal += sum(1 for row in rows if row.callee.rpartition(".")[2] in repo_names)
            per_line = defaultdict(list)
            for row in rows:
                per_line[row.line].append(row)
            for line, group in per_line.items():
                linked = {e.dst_id for e in edges_from[node.id] if e.provenance.line == line}  # a call the scan already linked explains these
                observed = sorted(seen.get((node.id, line), set()) - linked)
                found.extend(_gains(node, file, line, group, [nodes[t] for t in observed]))
    order = lambda g: (g.file, g.line, g.callee_text, g.target)  # noqa: E731
    gains = tuple(sorted((g for g in found if g.basis == "name"), key=order))
    candidates = tuple(sorted((g for g in found if g.basis == "position"), key=order))
    return GainReport(total, in_ran, in_prod, internal, gains, candidates)


def _gains(caller: Node, file: str, line: int, group: list, targets: list[Node]) -> list[Gain]:
    """Each observed target explains at most one call on its line (`a.go(b.go(1))` seen once is one gain, not two)."""
    test = not is_production_node(caller)
    found: list[Gain] = []
    unclaimed = list(targets)
    for row in group:
        word = row.callee.rpartition(".")[2]
        named = [t for t in unclaimed if t.name == word]
        if named:
            found += [Gain(caller.qualname, file, line, row.callee, t.qualname, "name", test) for t in named]
            unclaimed = [t for t in unclaimed if t not in named]
        elif len(group) == 1 and len(unclaimed) == 1:
            found.append(Gain(caller.qualname, file, line, row.callee, unclaimed[0].qualname, "position", test))
    return found


def sample_gains(gains: tuple[Gain, ...], count: int, seed: int) -> tuple[Gain, ...]:
    """A reproducible random sample to check by hand (all of them when there are fewer than `count`)."""
    if len(gains) <= count:
        return gains
    return tuple(sorted(random.Random(seed).sample(gains, count), key=lambda g: (g.file, g.line, g.callee_text)))
