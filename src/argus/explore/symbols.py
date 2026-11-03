"""Symbol lookup and inspection: definition, purpose, callers, callees and calls Argus could not resolve.

Built only from the existing ArchModel plus a re-read of the one source file that holds the symbol. Calls the
extractor could not resolve leave no edge, so they are recovered from the raw call sites and always reported UNKNOWN.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from argus.explore.claims import Claim, Tier, evidence_at, tier_for
from argus.explore.entrypoints import find_entrypoints
from argus.explore.grouping import MODULE_KINDS, is_production_node, is_test_function, node_index, owner_module
from argus.explore.reach import CallGraph, reach_tree
from argus.extract.facts import RawCall
from argus.extract.python_ast import parse_file
from argus.extract.resolve_calls import resolve_v2_enabled
from argus.extract.scanner import BUILTIN_ROOTS
from argus.model.ids import is_unresolved
from argus.model.schema import ArchModel, Confidence, Edge, EdgeKind, Node, NodeKind
from argus.record.observed import ObservedFacts, observed_for
from argus.record.store import Recording

SYMBOL_KINDS = (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS, NodeKind.MODULE, NodeKind.PACKAGE)
CALLER_KINDS = (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS)
MAX_CANDIDATES = 10
MAX_SUGGESTIONS = 3


@dataclass(frozen=True, slots=True)
class MemberRow:
    kind: str
    qualname: str
    claim: Claim


@dataclass(frozen=True, slots=True)
class CallRow:
    """One resolved call edge seen from the symbol: `name` is the other end (the caller for callers, the callee for callees)."""

    name: str
    kind: str
    module_level: bool  # the caller is a module's top-level code
    module_target: bool  # the callee resolved only to a module: which function is called is not identified
    override_possible: bool
    confidence: str
    claim: Claim


@dataclass(frozen=True, slots=True)
class UnresolvedCall:
    callee: str  # the call as written in the source
    reason: str
    line: int
    claim: Claim  # UNKNOWN, citing the call site


@dataclass(frozen=True, slots=True)
class ReachRow:
    """A CLI command whose shortest resolved call chain ends at the symbol. `chain` is qualnames, the command's function first."""

    command: str
    chain: tuple[str, ...]
    claim: Claim


@dataclass(frozen=True, slots=True)
class SymbolView:
    node: Node
    definition: Claim
    signature: str | None
    purpose: Claim
    decorators: tuple[str, ...]
    member_rows: tuple[MemberRow, ...]
    caller_rows: tuple[CallRow, ...]
    hidden_test_callers: int
    callee_rows: tuple[CallRow, ...]
    external_calls: tuple[str, ...]
    builtin_call_count: int
    unresolved_rows: tuple[UnresolvedCall, ...]
    resolve_v2: bool  # whether self.method() resolution was on for the scan this view was built from
    lines: int | None = None  # lines the definition spans; None for a module or package
    reached_from: tuple[ReachRow, ...] | None = None  # None where it is not computed (module, package)
    tested_by: tuple[str, ...] | None = None  # test functions that call it directly: a lower bound; None where not computed
    commands_found: int = 0  # CLI commands and web routes Argus found in the project; 0 means "not reached" says nothing about this symbol
    extra_found: int = 0  # of those, the web routes and plain programs (the label then reads "Reached from" instead of "Used by CLI")
    reach_caveat: str | None = None  # why "reached from" may be incomplete or missing: unresolved entry-point spots, or the scan failed
    observed: ObservedFacts | None = None  # OBSERVED while the code ran (latest recording); None: no recording, or not computed for this kind
    observed_error: str | None = None  # the latest recording exists but cannot be read

    @property
    def members(self) -> tuple[Claim, ...]:
        return tuple(r.claim for r in self.member_rows)

    @property
    def callers(self) -> tuple[Claim, ...]:
        return tuple(r.claim for r in self.caller_rows)

    @property
    def callees(self) -> tuple[Claim, ...]:
        return tuple(r.claim for r in self.callee_rows)

    @property
    def unknown_calls(self) -> tuple[Claim, ...]:
        return tuple(r.claim for r in self.unresolved_rows)


# -- lookup ---------------------------------------------------------------


def resolve_symbol(model: ArchModel, query: str) -> Node:
    """The one symbol `query` names; raises LookupError (with candidates or suggestions) otherwise."""
    pool = [n for n in model.nodes if n.kind in SYMBOL_KINDS]
    for matches in (
        [n for n in pool if n.qualname == query],
        [n for n in pool if n.qualname.endswith(f".{query}")],
        [n for n in pool if n.name == query],
    ):
        if matches:
            return _pick(query, _prefer_production(matches))
    names = sorted({n.name for n in pool})
    close = difflib.get_close_matches(query, names, n=MAX_SUGGESTIONS)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise LookupError(f"No symbol matches {query!r}.{hint}")


def _prefer_production(nodes: list[Node]) -> list[Node]:
    return [n for n in nodes if is_production_node(n)] or nodes


def _pick(query: str, matches: list[Node]) -> Node:
    if len(matches) == 1:
        return matches[0]
    shown = sorted(f"{n.qualname} ({n.kind.value})" for n in matches)[:MAX_CANDIDATES]
    more = f" (+{len(matches) - MAX_CANDIDATES} more)" if len(matches) > MAX_CANDIDATES else ""
    raise LookupError(f"{query!r} is ambiguous. Use a longer name:\n  " + "\n  ".join(shown) + more)


# -- view -----------------------------------------------------------------


def build_symbol_view(
    model: ArchModel, root: Path, node: Node, include_tests: bool = False, resolve_v2: bool | None = None,
    recording: Recording | None = None, recording_error: str | None = None,
) -> SymbolView:
    resolve_v2 = resolve_v2_enabled(resolve_v2)  # None: the same default scan() applies
    nodes = node_index(model)
    callers, hidden = _callers(model, root, nodes, node, include_tests)
    callees, external = _callees(model, root, nodes, node)
    has_calls = node.kind in (NodeKind.FUNCTION, NodeKind.METHOD)
    unknown, builtins = unresolved_calls(model, root, nodes, node, resolve_v2) if has_calls else ((), 0)
    reached, commands_found, caveat, extra_found = _reach(model, root, nodes, node)
    return SymbolView(
        node, Claim(f"{node.kind.value} {node.qualname} is defined here", Tier.FACT, (evidence_at(root, node.provenance.file, node.provenance.line),)),
        signature_text(node), _purpose(root, node), tuple(node.attrs.get("decorators") or ()), _members(model, root, nodes, node),
        callers, hidden, callees, external, builtins, unknown, resolve_v2, _lines(node),
        reached, _tested_by(model, nodes, node), commands_found, extra_found, caveat, _observed(recording, node, root, callers), recording_error,
    )


def _observed(recording: Recording | None, node: Node, root: Path, callers: tuple[CallRow, ...]) -> ObservedFacts | None:
    """What the recording saw of a function or method. `callers` are the ones read from the code, so the rest can be marked `also`."""
    if recording is None or node.kind not in (NodeKind.FUNCTION, NodeKind.METHOD):
        return None
    return observed_for(recording, node.id, node.provenance.file, root, {r.name for r in callers})


def _lines(node: Node) -> int | None:
    end = node.provenance.end_line
    return end - node.provenance.line + 1 if node.kind in CALLER_KINDS and end else None


def unresolved_note(eps) -> str | None:
    """Why a list of commands and routes may be missing one: entry-point spots Argus could not resolve."""
    spots = len(eps.unknowns)
    return f"{spots} entry-point spot{'' if spots == 1 else 's'} Argus could not resolve (dynamic names or unparsed modules), so a command may be missing" if spots else None


def _reach(model: ArchModel, root: Path, nodes: dict[str, Node], node: Node) -> tuple[tuple[ReachRow, ...] | None, int, str | None, int]:
    """(commands that reach the symbol, commands found, caveat). A failed entry-point scan must not break `symbol`: it becomes a caveat."""
    if node.kind not in CALLER_KINDS:
        return None, 0, None, 0
    try:
        eps = find_entrypoints(root, model)
        graph = CallGraph(model)
        rows: list[ReachRow] = []
        shown: set[str] = set()  # one row per name: a guard that calls three functions is one `python file.py`, not three
        for command in (*eps.commands, *eps.routes, *eps.programs):
            if command.display in shown:
                continue
            if command.function_id == node.id:
                rows.append(ReachRow(command.display, (node.qualname,), command.claim))
                shown.add(command.display)
                continue
            start = nodes.get(command.function_id)
            steps = reach_tree(model, root, command.function_id, graph).get(node.id) if start else None
            if start and steps:
                chain = (start.qualname, *(step.callee for step in steps))
                rows.append(ReachRow(command.display, chain, Claim(f"{command.display} reaches {node.qualname}", Tier.INFERENCE, steps[-1].claim.evidence)))
                shown.add(command.display)
    except Exception as err:  # noqa: BLE001 - reach is an extra: a malformed pyproject or an odd file must not take the whole symbol view down
        return None, 0, f"the entry-point scan failed ({type(err).__name__}), so reach is not known", 0
    return tuple(rows), len(eps.commands) + len(eps.routes) + len(eps.programs), unresolved_note(eps), len(eps.routes) + len(eps.programs)


def _tested_by(model: ArchModel, nodes: dict[str, Node], node: Node) -> tuple[str, ...] | None:
    if node.kind not in CALLER_KINDS:
        return None
    callers = (nodes.get(e.src_id) for e in call_edges(model) if e.dst_id == node.id and e.confidence is not Confidence.AMBIGUOUS)
    tests = {c.qualname for c in callers if c and is_test_function(c)}
    return tuple(sorted(tests))


def signature_text(node: Node) -> str | None:
    payload = node.attrs.get("signature")
    if not payload:
        return None
    parts = []
    for p in payload["parameters"]:
        prefix = {"vararg": "*", "kwarg": "**"}.get(p["kind"], "")
        text = prefix + p["name"] + (f": {p['annotation']}" if p["annotation"] else "")
        parts.append(text + (f" = {p['default']}" if p["has_default"] else ""))
    returns = f" -> {payload['returns']}" if payload["returns"] else ""
    return f"({', '.join(parts)}){returns}"


def _purpose(root: Path, node: Node) -> Claim:
    doc = (node.attrs.get("docstring") or "").strip()
    first = " ".join(line.strip() for line in doc.split("\n\n")[0].splitlines())
    if not first:
        return Claim(f"{node.name} has no docstring, so its purpose is not stated", Tier.UNKNOWN)
    return Claim(f'docstring: "{first}"', Tier.FACT, (evidence_at(root, node.provenance.file, node.provenance.line),))


def _members(model: ArchModel, root: Path, nodes: dict[str, Node], node: Node) -> tuple[MemberRow, ...]:
    if node.kind not in (NodeKind.CLASS, NodeKind.MODULE, NodeKind.PACKAGE):
        return ()
    children = sorted((n for n in model.nodes if n.parent_id == node.id and n.kind in SYMBOL_KINDS), key=lambda n: n.provenance.line)
    return tuple(
        MemberRow(c.kind.value, c.qualname, Claim(f"{c.kind.value} {c.qualname}", Tier.FACT, (evidence_at(root, c.provenance.file, c.provenance.line),)))
        for c in children
    )


def call_edges(model: ArchModel) -> list[Edge]:
    return [e for e in model.edges if e.kind is EdgeKind.CALLS]


def _callers(model: ArchModel, root: Path, nodes: dict[str, Node], node: Node, include_tests: bool) -> tuple[tuple[CallRow, ...], int]:
    if node.kind not in CALLER_KINDS:
        return (), 0
    rows: list[tuple[Edge, Node]] = []
    hidden = 0
    for edge in call_edges(model):
        src = nodes.get(edge.src_id)
        if edge.dst_id != node.id or src is None:
            continue
        if not include_tests and not is_production_node(src):
            hidden += 1
            continue
        rows.append((edge, src))
    rows.sort(key=lambda r: (r[1].qualname, r[0].provenance.line))
    claims = tuple(
        CallRow(
            src.qualname, src.kind.value, src.kind in MODULE_KINDS, False, bool(e.attrs.get("override_possible")), e.confidence.value,
            Claim(f"{caller_label(src, nodes)} calls {node.qualname}{override_note(e)}", tier_for(e.confidence), (evidence_at(root, e.provenance.file, e.provenance.line),)),
        )
        for e, src in rows
    )
    return claims, hidden


def override_note(edge: Edge) -> str:
    return " (via self; a subclass may override it)" if edge.attrs.get("override_possible") else ""


def caller_label(src: Node, nodes: dict[str, Node]) -> str:
    if src.kind in (NodeKind.MODULE, NodeKind.PACKAGE):
        return f"{src.qualname} (module level)"
    return src.qualname


def _callees(model: ArchModel, root: Path, nodes: dict[str, Node], node: Node) -> tuple[tuple[CallRow, ...], tuple[str, ...]]:
    claims: list[CallRow] = []
    external: set[str] = set()
    for edge in sorted((e for e in call_edges(model) if e.src_id == node.id), key=lambda e: (e.provenance.line, e.dst_id)):
        if is_unresolved(edge.dst_id):
            external.add(edge.dst_ref or edge.dst_id)
            continue
        dst = nodes.get(edge.dst_id)
        if dst is None:
            continue
        site = evidence_at(root, edge.provenance.file, edge.provenance.line)
        note = " (a module: which function is called is not identified yet)" if dst.kind in MODULE_KINDS else ""
        claims.append(CallRow(
            dst.qualname, dst.kind.value, False, dst.kind in MODULE_KINDS, bool(edge.attrs.get("override_possible")), edge.confidence.value,
            Claim(f"{node.qualname} calls {dst.qualname}{note}{override_note(edge)}", tier_for(edge.confidence), (site,)),
        ))
    return tuple(claims), tuple(sorted(external))


def unresolved_calls(model: ArchModel, root: Path, nodes: dict[str, Node], node: Node, resolve_v2: bool) -> tuple[tuple[UnresolvedCall, ...], int]:
    module = owner_module(node, nodes)
    if module is None:
        return (), 0
    local = node.qualname[len(module) + 1 :]
    facts = parse_file(root / node.provenance.file, module)
    return unresolved_in([c for c in facts.calls if c.caller_local == local], [e for e in call_edges(model) if e.src_id == node.id], root, node, resolve_v2)


def unresolved_in(raw: list[RawCall], edges: list[Edge], root: Path, node: Node, resolve_v2: bool) -> tuple[tuple[UnresolvedCall, ...], int]:
    """(unresolved-call claims, builtin-call count) for a function whose call sites (`raw`) and call edges (`edges`) are already in hand."""
    edges_by_line = edges_by_source_line(edges)
    builtins = sum(1 for c in raw if is_builtin_call(c))
    unresolved = without_resolved([c for c in raw if not is_builtin_call(c)], edges_by_line)
    rows = tuple(
        UnresolvedCall(
            c.callee, (reason := _reason(c, resolve_v2)), c.line,
            Claim(f"{c.callee}(...): {reason}", Tier.UNKNOWN, (evidence_at(root, node.provenance.file, c.line),)),
        )
        for c in unresolved
    )
    return rows, builtins


def edges_by_source_line(edges: list[Edge]) -> dict[int, list[Edge]]:
    by_line: dict[int, list[Edge]] = defaultdict(list)
    for edge in edges:
        by_line[edge.provenance.line].append(edge)
    return by_line


def is_builtin_call(call: RawCall) -> bool:
    return _root(call) in BUILTIN_ROOTS


def without_resolved(calls: list[RawCall], edges_by_line: dict[int, list[Edge]]) -> list[RawCall]:
    """Drop the calls that have a resolved edge.

    An unresolved call leaves no edge, so the two are matched per source line: equal or more edges than calls means
    all resolved, no edges means none. A line mixing both is matched by name (the callee's first or last segment
    against the edge target), and only the surplus of unmatched calls is reported.
    """
    out: list[RawCall] = []
    by_line: dict[int, list[RawCall]] = defaultdict(list)
    for call in calls:
        by_line[call.line].append(call)
    for line, group in sorted(by_line.items()):
        edges = edges_by_line.get(line, [])
        if len(edges) >= len(group):
            continue
        if not edges:
            out.extend(group)
            continue
        targets = {(e.dst_ref or "").rpartition(".")[2] for e in edges}
        unmatched = [c for c in group if _root(c) not in targets and c.callee.rpartition(".")[2] not in targets]
        out.extend(unmatched[: len(group) - len(edges)])
    return out


def _root(call: RawCall) -> str:
    return call.callee.partition(".")[0]


def _reason(call: RawCall, resolve_v2: bool) -> str:
    root = _root(call)
    if root == "cls":
        return "method call on cls; class-method calls are not resolved"
    if root == "self":
        if call.callee.count(".") > 1:
            return "method call on an attribute of self; the attribute's type is unknown, so the call is not resolved"
        if resolve_v2:
            return "method call on self; not resolved (inherited, defined more than once, a property, shadowed, or called from a nested function)"
        return "method call on self; not resolved (ARGUS_RESOLVE_V2=0 turns off self.method() resolution)"
    if not (root.isidentifier() and root.isascii()):
        return "called on a literal value (string, bytes, number or container)"
    if "(" in call.callee or "[" in call.callee:
        return "called on the result of another expression"
    if "." in call.callee:
        return f"{root!r} is a variable or parameter, or an ambiguous import, so its type is unknown"
    return f"{root!r} is not imported or defined at module level (local, parameter or nested function)"
