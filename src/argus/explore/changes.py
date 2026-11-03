"""Diff Explorer: compare two scanned sides and report observable structural changes as evidence-linked claims.

Never claims a change is correct, safe or behaviorally different. Everything is read from the two scanned models (call edges keep
their original evidence and confidence) or from a normalized syntax-tree hash of function bodies. Nothing here runs git: the two
sides arrive already built (see `sides.py`), and the returned view holds no path into their temporary directories.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from argus.explore.call_fit import CallFit, check_signature_changes
from argus.explore.changed_lines import ChangedLine, function_diff
from argus.explore.edit_kind import edit_kind
from argus.explore.claims import SOURCE_HASH_BYTES, Claim, Evidence, Tier, evidence_at, tier_for
from argus.explore.grouping import IMPORT_KINDS, MODULE_KINDS, default_group, is_production_node, is_test_function, node_index, owner_module
from argus.explore.sides import Side
from argus.explore.symbols import call_edges, caller_label, edges_by_source_line, is_builtin_call, override_note, signature_text, without_resolved
from argus.extract.discovery import find_source_roots, is_excluded_dir, qualname_for
from argus.extract.facts import RawCall
from argus.extract.python_ast import parse_file
from argus.model.schema import ArchModel, Confidence, Edge, EdgeKind, Node, NodeKind

FUNCTION_KINDS = (NodeKind.FUNCTION, NodeKind.METHOD)
SYMBOL_KINDS = (*FUNCTION_KINDS, NodeKind.CLASS)
PARSE_ERROR_KIND = "parse_error"
# Blind-spot kinds that sit inside a function body. `star_import` is left out: it only exists at module level, so no function holds one.
DYNAMIC_KINDS = frozenset({"dynamic_import", "dynamic_attribute", "eval_exec", "sys_path_mutation"})
WORKING_TREE = "working tree"
_DUPLICATE_SUFFIX = re.compile(r"#\d+")
DOES_NOT_PARSE = "does not parse"
NOT_UTF8 = "is not valid UTF-8"
NOT_ANALYSED = "was not analysed: the scanner extracted no module from it (for example, another file defines the same module name)"

ChangeKind = Literal["added", "removed", "modified"]
# Why a changed file has no function or class change in it: code outside them changed, the syntax tree is identical (comments,
# formatting), or the file was added / removed without defining any function or class.
FileKind = Literal["outside_symbols", "formatting_only", "added", "removed", "relocated", "added_empty", "removed_empty"]
# State of a caller in this diff. "added" is the fourth state: a new function that calls a changed or removed symbol.
CallerState = Literal["unchanged", "changed", "added", "removed"]


class Flag(StrEnum):
    """Independent change flags of a function or method. None implies another."""

    SIGNATURE_CHANGED = "SIGNATURE_CHANGED"
    CALLS_CHANGED = "CALLS_CHANGED"
    BODY_CHANGED = "BODY_CHANGED"


CALLER_LABEL_FLAGS = (Flag.SIGNATURE_CHANGED, Flag.BODY_CHANGED)
INTERFACE_LABEL = "REMOVED"


@dataclass(frozen=True, slots=True)
class DynamicSpot:
    """A line inside a changed function that static reading cannot see through (a computed `getattr`, a dynamic import, `eval`)."""

    kind: str
    detail: str
    file: str
    line: int


@dataclass(frozen=True, slots=True)
class SymbolUser:
    """One function that calls a changed symbol, at its first call site. Test code is listed apart (`is_test`)."""

    caller: str  # qualified name
    module: str
    file: str
    line: int
    is_test: bool
    module_level: bool  # the call is in a module's top-level code, not in a function


@dataclass(frozen=True, slots=True)
class SymbolChange:
    """One added, removed or modified symbol. `claim` is a FACT; `calls_claim` is an INFERENCE (present iff CALLS_CHANGED)."""

    qualname: str
    kind: str
    change: ChangeKind
    flags: frozenset[Flag]
    claim: Claim | None  # FACT for added / removed / SIGNATURE_CHANGED / BODY_CHANGED; None when the only flag is CALLS_CHANGED
    calls_claim: Claim | None
    untracked: bool
    file: str  # head location (base location for a removed symbol)
    line: int
    interface_was: str | None = None  # decorators, async marker and signature before / after; set iff SIGNATURE_CHANGED
    interface_now: str | None = None
    docstring_only: bool = False  # BODY_CHANGED caused only by a docstring edit: a syntactic change, not necessarily a behavioral one
    known_callers: int | None = None  # distinct production callers the call graph links to it (a lower bound); None where no callers are computed (added, CALLS_CHANGED only)
    test_callers: int | None = None  # distinct test-code functions the call graph links to it (a lower bound); None exactly where known_callers is None
    users: tuple[SymbolUser, ...] = ()  # who calls it, production first; empty exactly where known_callers is None
    module: str | None = None  # the module it lives in (its base location for a removed symbol)
    module_imported_by: tuple[str, ...] = ()  # production modules that import that module (a lower bound: dynamic imports are not seen)
    dynamic_spots: tuple[DynamicSpot, ...] = ()  # unreadable lines inside this function or method (its base lines for a removed one)
    changed_lines: tuple[ChangedLine, ...] = ()  # the lines that differ, before and now; only where BODY_CHANGED is set (not in --json)
    call_fits: tuple[CallFit, ...] = ()  # calls checked against the new inputs; only where SIGNATURE_CHANGED is set (not in --json: DiffView.call_fits has them)
    changed_note: str = ""  # why there are no changed lines to show although the body changed (too large, the text looks the same, unreadable)
    edit_kind: str = ""  # "condition", "result" or "statements": the strongest part of the body that differs; "" when unclassified (not in --json)

    @property
    def node_id(self) -> str:
        return f"{self.kind}:{self.qualname}"


@dataclass(frozen=True, slots=True)
class CallChange:
    """A resolved call that exists on one side only. Compared only between functions present on both sides: calls made by
    added or removed functions are not listed here (an added function still appears as a caller in the caller sections).
    A removal is claimed only when the callsite text is gone or now resolves elsewhere, never for lost resolution."""

    change: Literal["added", "removed"]
    source: str
    target: str
    claim: Claim  # INFERENCE; evidence is the lowest-line call site (head side for added, base side for removed)


@dataclass(frozen=True, slots=True)
class ImportChange:
    change: Literal["added", "removed"]
    source: str  # subsystem
    target: str
    claim: Claim


@dataclass(frozen=True, slots=True)
class CallerEdge:
    callee: str
    callee_label: str  # INTERFACE_LABEL, or the callee's SIGNATURE_CHANGED / BODY_CHANGED flags joined by ", "
    claim: Claim  # INFERENCE with one Evidence: the call site
    was: bool  # evidence read from the base side
    confidence: str  # the scanned edge's confidence ("INFERRED"), copied from the model and never upgraded
    override_possible: bool  # the edge carries the "via self; a subclass may override it" note


@dataclass(frozen=True, slots=True)
class CallerGroup:
    caller: str
    module_level: bool
    caller_state: CallerState
    is_test: bool
    edges: tuple[CallerEdge, ...]


@dataclass(frozen=True, slots=True)
class FileChange:
    """A changed file in which no function or class changed. `claim` is a FACT about the whole file (evidence: its first line)."""

    file: str
    kind: FileKind
    claim: Claim


@dataclass(frozen=True, slots=True)
class FileRename:
    """A removed file and an added file with byte-identical content, paired one to one. Every qualified name in it changes with the path."""

    old: str
    new: str
    symbols: int  # added symbols in the new file
    claim: Claim  # FACT


@dataclass(frozen=True, slots=True)
class NotScanned:
    """Changed `.py` files under a directory the scanner skips on purpose (build/, virtualenvs, hidden directories). A policy, not a failure."""

    prefix: str  # e.g. "build/"
    files: int


@dataclass(frozen=True, slots=True)
class DiffView:
    base_label: str
    head_label: str
    resolve_v2: bool
    files_changed: int
    untracked_files: int
    symbols: tuple[SymbolChange, ...]
    calls: tuple[CallChange, ...]
    interface_callers: tuple[CallerGroup, ...]
    affected_callers: tuple[CallerGroup, ...]
    hidden_test_callers: int  # caller groups hidden because they are tests
    imports: tuple[ImportChange, ...]
    not_determinable: tuple[Claim, ...]
    not_scanned: tuple[NotScanned, ...] = ()
    new_files: frozenset[str] = frozenset()  # scanned files that exist on the head side only; every symbol in one is "added"
    file_changes: tuple[FileChange, ...] = ()
    renames: tuple[FileRename, ...] = ()
    failed_files: tuple[str, ...] = ()  # files that do not parse on either side: nothing in them is compared
    call_fits: tuple[CallFit, ...] = ()  # every call to a function with changed inputs, and whether it still fits (reads code, runs nothing)

    def claims(self) -> tuple[Claim, ...]:
        """Every claim in the view, in display order."""
        out: list[Claim] = []
        for symbol in self.symbols:
            out.extend(c for c in (symbol.claim, symbol.calls_claim) if c is not None)
        out.extend(c.claim for c in self.calls)
        for group in (*self.interface_callers, *self.affected_callers):
            out.extend(e.claim for e in group.edges)
        out.extend(i.claim for i in self.imports)
        out.extend(f.claim for f in (*self.file_changes, *self.renames))
        out.extend(f.claim for f in self.call_fits)
        out.extend(self.not_determinable)
        return tuple(out)


# -- per-side data -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Scanned:
    side: Side
    model: ArchModel
    nodes: dict[str, Node]
    calls: tuple[Edge, ...]  # the model's CALLS edges, filtered once
    edges_from: dict[str, list[Edge]]  # CALLS edges by source node id
    hashes: dict[str, str]  # file -> content hash, for every scanned file (directories skipped on purpose are left out)
    failed: dict[str, str]  # file -> why nothing in it can be compared on this side ("does not parse", "is not valid UTF-8", ...)


def _is_excluded(rel: str) -> bool:
    """True for a file under a directory the scanner skips on purpose; the same policy `scan()` applies."""
    return any(is_excluded_dir(part) for part in PurePosixPath(rel).parts[:-1])


def _content_hash(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=SOURCE_HASH_BYTES).hexdigest()


def _scanned(side: Side, model: ArchModel) -> _Scanned:
    failed = {b.provenance.file: DOES_NOT_PARSE for b in model.blind_spots if b.kind == PARSE_ERROR_KIND}
    for rel, reason in side.skipped:
        if not _is_excluded(rel):
            failed.setdefault(rel, reason)
    hashes: dict[str, str] = {}
    for rel in side.files:
        if _is_excluded(rel):
            continue
        data = (side.root / rel).read_bytes()
        hashes[rel] = _content_hash(data)
        try:
            data.decode("utf-8")  # scan() decodes with replacement characters, so two different non-UTF-8 files can look identical
        except UnicodeDecodeError:
            failed.setdefault(rel, NOT_UTF8)
    extracted = {n.provenance.file for n in model.nodes if n.kind in MODULE_KINDS}
    for rel in hashes:
        if rel not in extracted:
            failed.setdefault(rel, NOT_ANALYSED)  # the scanner dropped it without a word (for example a module-name collision)
    calls = tuple(call_edges(model))
    edges_from: dict[str, list[Edge]] = defaultdict(list)
    for edge in calls:
        edges_from[edge.src_id].append(edge)
    return _Scanned(side, model, node_index(model), calls, edges_from, hashes, failed)


@dataclass(frozen=True, slots=True)
class _Pair:
    base: _Scanned
    head: _Scanned
    changed_files: frozenset[str]
    failed_files: frozenset[str]  # failed on either side: excluded from every result
    failed_modules: dict[str, str]  # module qualname -> file, for files that failed
    suppressed: dict[str, tuple[str, ...]]  # file importing from a failed module -> the failed files (calls not compared)
    duplicates: frozenset[str]  # canonical qualnames defined more than once on either side
    blocked: frozenset[str]  # symbol ids that cannot be compared (failed file or defined more than once)


def _changed_files(base: _Scanned, head: _Scanned) -> frozenset[str]:
    return frozenset(rel for rel in base.hashes.keys() | head.hashes.keys() if base.hashes.get(rel) != head.hashes.get(rel))


def _not_scanned(base: Side, head: Side) -> tuple[NotScanned, ...]:
    """Changed files in directories the scanner skips on purpose, grouped by the first skipped directory on their path."""
    counts: Counter[str] = Counter()
    for rel in sorted(set(base.files) | set(head.files)):
        if not _is_excluded(rel):
            continue
        before = (base.root / rel).read_bytes() if rel in base.files else None
        after = (head.root / rel).read_bytes() if rel in head.files else None
        if before != after:
            parts = PurePosixPath(rel).parts
            first = next(i for i, part in enumerate(parts[:-1]) if is_excluded_dir(part))
            counts["/".join(parts[: first + 1]) + "/"] += 1
    return tuple(NotScanned(prefix, files) for prefix, files in sorted(counts.items()))


def _canonical(qualname: str) -> str:
    return _DUPLICATE_SUFFIX.sub("", qualname)


def _duplicate_names(model: ArchModel) -> set[str]:
    counts = Counter((n.kind, _canonical(n.qualname)) for n in model.nodes if n.kind in SYMBOL_KINDS)
    return {name for (_, name), count in counts.items() if count > 1}


def _has_duplicate_in_chain(qualname: str, duplicates: frozenset[str]) -> bool:
    parts = _canonical(qualname).split(".")
    return any(".".join(parts[:end]) in duplicates for end in range(1, len(parts) + 1))


def _module_of(side: Side, file: str) -> str | None:
    root = side.root.resolve()
    found = qualname_for(root / file, find_source_roots(root))
    return found[0] if found else None


def _failed_modules(base: _Scanned, head: _Scanned) -> dict[str, str]:
    modules: dict[str, str] = {}
    for scanned in (base, head):
        for file in scanned.failed:
            module = _module_of(scanned.side, file)
            if module:
                modules[module] = file
    return modules


def _suppressed_files(base: _Scanned, head: _Scanned, failed_modules: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """Files with an import from a module that failed to parse: their calls may look unresolved for that reason alone."""
    suppressed: dict[str, set[str]] = defaultdict(set)
    for scanned in (base, head):
        for edge in scanned.model.edges:
            if edge.kind not in IMPORT_KINDS or not edge.dst_ref:
                continue
            for module, file in failed_modules.items():
                if (edge.dst_ref == module or edge.dst_ref.startswith(f"{module}.")) and edge.provenance.file != file:
                    suppressed[edge.provenance.file].add(file)
    return {importer: tuple(sorted(files)) for importer, files in suppressed.items()}


def _pair(base: Side, head: Side, base_model: ArchModel, head_model: ArchModel) -> _Pair:
    b, h = _scanned(base, base_model), _scanned(head, head_model)
    failed_files = frozenset(b.failed) | frozenset(h.failed)
    duplicates = frozenset(_duplicate_names(base_model) | _duplicate_names(head_model))
    blocked = frozenset(
        n.id for scanned in (b, h) for n in scanned.model.nodes
        if n.kind in SYMBOL_KINDS and (n.provenance.file in failed_files or _has_duplicate_in_chain(n.qualname, duplicates))
    )
    failed_modules = _failed_modules(b, h)
    return _Pair(
        b, h, _changed_files(b, h), failed_files, failed_modules, _suppressed_files(b, h, failed_modules), duplicates, blocked,
    )


# -- function bodies -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BodyHash:
    name: str
    full: str  # every body statement, docstring included
    code: str  # the same without a leading docstring


def _digest(statements: list[ast.stmt]) -> str:
    dumped = "\x00".join(ast.dump(statement) for statement in statements)  # no line numbers: formatting cannot change it
    return hashlib.blake2b(dumped.encode("utf-8"), digest_size=SOURCE_HASH_BYTES).hexdigest()


def _is_docstring(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)


@dataclass(slots=True)
class _ParsedFile:
    hashes: dict[int, _BodyHash]  # keyed by the line of the `def`
    functions: dict[int, ast.FunctionDef | ast.AsyncFunctionDef]
    lines: list[str] = field(default_factory=list)  # the file's lines, split where the parser counts them


def _parse_bodies(side: Side, file: str) -> _ParsedFile:
    """Hash of every function body in `file`, keyed by the line of its `def` (the line the scanner records).

    A file or a single body that cannot be parsed or hashed (pathological nesting) is simply absent: its flag is reported as unknown.
    """
    text = (side.root / file).read_bytes().decode("utf-8", errors="replace")  # the decoding scan() uses
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return _ParsedFile({}, {})
    parsed = _ParsedFile({}, {}, [line.rstrip("\r\n") for line in io.StringIO(text, newline="").readlines()])
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            code = node.body[1:] if node.body and _is_docstring(node.body[0]) else node.body
            try:
                parsed.hashes[node.lineno] = _BodyHash(node.name, _digest(node.body), _digest(code))
            except RecursionError:
                continue
            parsed.functions[node.lineno] = node
    return parsed


class _BodyIndex:
    """Lazily parses each file once per side."""

    def __init__(self, side: Side) -> None:
        self._side = side
        self._files: dict[str, _ParsedFile] = {}

    def _parsed(self, file: str) -> _ParsedFile:
        if file not in self._files:
            self._files[file] = _parse_bodies(self._side, file)
        return self._files[file]

    def of(self, node: Node) -> _BodyHash | None:
        found = self._parsed(node.provenance.file).hashes.get(node.provenance.line)
        return found if found and found.name == node.name else None

    def function_of(self, node: Node) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        function = self._parsed(node.provenance.file).functions.get(node.provenance.line)
        return function if function is not None and function.name == node.name else None

    def source_of(self, node: Node) -> tuple[int, list[str]] | None:
        """(file line of the `def`, the function's lines without their common indentation), or None when the function cannot be found."""
        parsed = self._parsed(node.provenance.file)
        function = parsed.functions.get(node.provenance.line)
        if function is None or function.name != node.name or function.end_lineno is None:
            return None
        margin = function.col_offset  # strip the indentation of the `def` only: a line cut at column 0 inside a string must stay as it is
        return function.lineno, [line[margin:] if not line[:margin].strip() else line for line in parsed.lines[function.lineno - 1 : function.end_lineno]]

    def called_names(self, node: Node) -> list[str]:
        """Last name segment of every call written anywhere inside the function: nested functions and `if`/`while` tests included."""
        function = self._parsed(node.provenance.file).functions.get(node.provenance.line)
        if function is None or function.name != node.name:
            return []
        names: list[str] = []
        for call in ast.walk(function):
            if isinstance(call, ast.Call):
                target = call.func
                if isinstance(target, ast.Name):
                    names.append(target.id)
                elif isinstance(target, ast.Attribute):
                    names.append(target.attr)
        return names


def _interface_text(node: Node) -> str:
    signature = signature_text(node) or "()"
    is_async = bool((node.attrs.get("signature") or {}).get("is_async"))
    decorators = " ".join(f"@{d}" for d in node.attrs.get("decorators") or ())
    return " ".join(part for part in (decorators, "async" if is_async else "", signature) if part)


def _interface(node: Node) -> tuple[object, object]:
    return node.attrs.get("decorators"), node.attrs.get("signature")


# -- symbols ---------------------------------------------------------------------------------------------------------


def _eligible(pair: _Pair, scanned: _Scanned) -> dict[str, Node]:
    return {n.id: n for n in scanned.model.nodes if n.kind in SYMBOL_KINDS and n.id not in pair.blocked}


def _fact(text: str, scanned: _Scanned, node: Node) -> Claim:
    return Claim(text, Tier.FACT, (evidence_at(scanned.side.root, node.provenance.file, node.provenance.line),))


def _added_removed(pair: _Pair, base_ids: dict[str, Node], head_ids: dict[str, Node]) -> list[SymbolChange]:
    changes: list[SymbolChange] = []
    for node_id in sorted(head_ids.keys() - base_ids.keys()):
        node = head_ids[node_id]
        changes.append(SymbolChange(
            node.qualname, node.kind.value, "added", frozenset(), _fact(f"{node.kind.value} {node.qualname} added", pair.head, node),
            None, node.provenance.file in pair.head.side.untracked, node.provenance.file, node.provenance.line,
        ))
    for node_id in sorted(base_ids.keys() - head_ids.keys()):
        node = base_ids[node_id]
        changes.append(SymbolChange(
            node.qualname, node.kind.value, "removed", frozenset(), _fact(f"{node.kind.value} {node.qualname} removed", pair.base, node),
            None, False, node.provenance.file, node.provenance.line,
        ))
    return changes


def _static_flags(pair: _Pair, base_bodies: _BodyIndex, head_bodies: _BodyIndex, before: Node, after: Node) -> tuple[set[Flag], bool, list[str]]:
    """(SIGNATURE_CHANGED / BODY_CHANGED flags, docstring_only, problems) for one function or method present on both sides."""
    flags: set[Flag] = set()
    if _interface(before) != _interface(after):
        flags.add(Flag.SIGNATURE_CHANGED)
    if before.provenance.file == after.provenance.file and after.provenance.file not in pair.changed_files:
        return flags, False, []
    old, new = base_bodies.of(before), head_bodies.of(after)
    if old is None or new is None:
        return flags, False, [after.qualname]
    docstring_only = old.full != new.full and old.code == new.code
    if old.full != new.full:
        flags.add(Flag.BODY_CHANGED)
    return flags, docstring_only, []


@dataclass(frozen=True, slots=True)
class _CallComparison:
    calls: tuple[CallChange, ...]
    by_source: dict[str, tuple[Evidence, ...]]  # source id -> evidence of its added / removed calls


def _call_index(pair: _Pair, scanned: _Scanned) -> dict[tuple[str, str], Edge]:
    """Resolved call edges between functions, methods and classes: {(source id, target id): lowest-line edge}.

    Module-level calls, unresolved or external targets, module targets and anything touching a file that did not parse (or
    imports from one) are left out, so a loss of resolution cannot be mistaken for a change in the code.
    """
    index: dict[tuple[str, str], Edge] = {}
    for edge in scanned.calls:
        src, dst = scanned.nodes.get(edge.src_id), scanned.nodes.get(edge.dst_id)
        if src is None or dst is None or src.kind not in FUNCTION_KINDS or dst.kind not in SYMBOL_KINDS:
            continue
        if edge.src_id in pair.blocked or edge.dst_id in pair.blocked or src.provenance.file in pair.suppressed:
            continue
        key = (edge.src_id, edge.dst_id)
        current = index.get(key)
        if current is None or (edge.provenance.file, edge.provenance.line) < (current.provenance.file, current.provenance.line):
            index[key] = edge
    return index


class _CallSites:
    """The call sites written in each function of one side (one parse per file), and which of them did not resolve."""

    def __init__(self, scanned: _Scanned) -> None:
        self._scanned = scanned
        self._files: dict[str, tuple[RawCall, ...]] = {}
        self._unresolved: dict[str, list[RawCall]] = {}

    def of(self, function: Node) -> list[RawCall]:
        file = function.provenance.file
        module = owner_module(function, self._scanned.nodes)
        if module is None:
            return []
        if file not in self._files:
            self._files[file] = parse_file(self._scanned.side.root / file, module).calls
        local = function.qualname[len(module) + 1 :]
        return [call for call in self._files[file] if call.caller_local == local]

    def unresolved(self, function: Node) -> list[RawCall]:
        """Calls in `function` (builtins excluded) with no resolved edge. An unresolved call leaves no edge, so this matches per line."""
        if function.id not in self._unresolved:
            edges = edges_by_source_line(self._scanned.edges_from.get(function.id, []))
            self._unresolved[function.id] = without_resolved([c for c in self.of(function) if not is_builtin_call(c)], edges)
        return self._unresolved[function.id]


def _callsite_remains_unresolved(pair: _Pair, base_edge: Edge, target: Node, head_bodies: _BodyIndex, base_sites: _CallSites) -> bool:
    """True when the call is still written in the head function but nothing explains it: lost resolution, not a removed call.

    A call whose text is gone is a real removal. A call whose text remains and now resolves to another function is a retarget
    (reported as one removal plus one addition). The head function is judged from its own syntax tree, nested functions and
    `if`/`while` tests included (the scanner records neither): the removal is withheld when it still names the callee more times
    than the head graph has edges from this function to something of that name. Counting per name, not per line, means a resolved
    call cannot vouch for a different call that merely shares its line or its name.
    """
    src_base, src_head = pair.base.nodes[base_edge.src_id], pair.head.nodes[base_edge.src_id]
    on_line = {c.callee for c in base_sites.of(src_base) if c.line == base_edge.provenance.line}
    names = {t.rpartition(".")[2] for t in on_line if t.rpartition(".")[2] == target.name} or {t.rpartition(".")[2] for t in on_line}
    written = sum(1 for name in head_bodies.called_names(src_head) if name in names)
    explained = sum(
        1 for edge in pair.head.edges_from.get(src_head.id, ())
        if (dst := pair.head.nodes.get(edge.dst_id)) is not None and dst.name in names
    )
    return written > explained


def _compare_calls(pair: _Pair, common: set[str], head_eligible: dict[str, Node], base_sites: _CallSites, head_bodies: _BodyIndex) -> _CallComparison:
    base_index, head_index = _call_index(pair, pair.base), _call_index(pair, pair.head)
    rows: list[CallChange] = []
    evidence: dict[str, list[Evidence]] = defaultdict(list)
    for key in sorted(head_index.keys() - base_index.keys()):
        if key[0] in common:
            rows.append(_call_row("added", pair.head, head_index[key]))
            evidence[key[0]].append(rows[-1].claim.evidence[0])
    for key in sorted(base_index.keys() - head_index.keys()):
        if key[0] not in common or key[1] not in head_eligible:  # the target must still exist, or the call row would be a guess
            continue
        if _callsite_remains_unresolved(pair, base_index[key], head_eligible[key[1]], head_bodies, base_sites):
            continue
        rows.append(_call_row("removed", pair.base, base_index[key]))
        evidence[key[0]].append(rows[-1].claim.evidence[0])
    rows.sort(key=lambda r: (r.source, r.target, r.claim.evidence[0].file, r.claim.evidence[0].line, r.change))
    return _CallComparison(tuple(rows), {src: tuple(items) for src, items in evidence.items()})


def _call_row(change: Literal["added", "removed"], scanned: _Scanned, edge: Edge) -> CallChange:
    src, dst = scanned.nodes[edge.src_id], scanned.nodes[edge.dst_id]
    claim = Claim(
        f"{src.qualname} calls {dst.qualname}" if change == "added" else f"{src.qualname} no longer calls {dst.qualname}",
        Tier.INFERENCE, (evidence_at(scanned.side.root, edge.provenance.file, edge.provenance.line),),
    )
    return CallChange(change, src.qualname, dst.qualname, claim)


def _modified(
    pair: _Pair, common: list[str], base_ids: dict[str, Node], head_ids: dict[str, Node], calls: _CallComparison, base_bodies: _BodyIndex, head_bodies: _BodyIndex,
) -> tuple[list[SymbolChange], list[Claim]]:
    changes: list[SymbolChange] = []
    unreadable: list[str] = []
    for node_id in common:
        before, after = base_ids[node_id], head_ids[node_id]
        if after.kind not in FUNCTION_KINDS:
            continue
        flags, docstring_only, problems = _static_flags(pair, base_bodies, head_bodies, before, after)
        unreadable.extend(problems)
        call_evidence = calls.by_source.get(node_id, ())
        if call_evidence:
            flags.add(Flag.CALLS_CHANGED)
        if not flags:
            continue
        changed = function_diff(base_bodies.source_of(before), head_bodies.source_of(after)) if Flag.BODY_CHANGED in flags else ((), "")
        old_fn, new_fn = base_bodies.function_of(before), head_bodies.function_of(after)
        kind = edit_kind(old_fn, new_fn) if Flag.BODY_CHANGED in flags and not docstring_only and old_fn and new_fn else ""
        changes.append(_modified_change(pair, before, after, flags, docstring_only, call_evidence, changed, kind))
    problems_claims = [Claim(f"The body of {name} could not be compared, so its BODY_CHANGED flag is unknown", Tier.UNKNOWN) for name in sorted(unreadable)]
    return changes, problems_claims


def _modified_change(
    pair: _Pair, before: Node, after: Node, flags: set[Flag], docstring_only: bool, call_evidence: tuple[Evidence, ...], changed: tuple[tuple[ChangedLine, ...], str],
    kind: str = "",
) -> SymbolChange:
    fact_flags = [f for f in Flag if f in flags and f is not Flag.CALLS_CHANGED]
    claim = _fact(f"{after.qualname}: {', '.join(fact_flags)}", pair.head, after) if fact_flags else None
    calls_claim = Claim(f"{after.qualname}: its resolved calls changed", Tier.INFERENCE, call_evidence) if call_evidence else None
    sig = Flag.SIGNATURE_CHANGED in flags
    return SymbolChange(
        after.qualname, after.kind.value, "modified", frozenset(flags), claim, calls_claim, False, after.provenance.file, after.provenance.line,
        _interface_text(before) if sig else None, _interface_text(after) if sig else None, docstring_only and Flag.BODY_CHANGED in flags,
        changed_lines=changed[0], changed_note=changed[1], edit_kind=kind,
    )


# -- callers ---------------------------------------------------------------------------------------------------------


def _flag_label(flags: frozenset[Flag]) -> str:
    return ", ".join(flag for flag in CALLER_LABEL_FLAGS if flag in flags)


def _caller_state(pair: _Pair, caller_id: str, caller: Node, changed_ids: set[str]) -> CallerState:
    if caller_id not in pair.head.nodes:
        return "removed"
    if caller_id not in pair.base.nodes:
        return "added"
    if caller.kind in FUNCTION_KINDS or caller.kind is NodeKind.CLASS:
        return "changed" if caller_id in changed_ids else "unchanged"
    return "changed" if pair.head.nodes[caller_id].provenance.file in pair.changed_files else "unchanged"


@dataclass(slots=True)
class _Group:
    node: Node
    edges: list[CallerEdge]
    interface: bool = False
    callees: set[str] = field(default_factory=set)  # ids of the changed symbols this caller points at
    module: str = ""


def _edge_claim(scanned: _Scanned, edge: Edge, src: Node, callee: Node, was: bool) -> Claim:
    verb = "called" if was else "calls"
    return Claim(
        f"{caller_label(src, scanned.nodes)} {verb} {callee.qualname}{override_note(edge)}", tier_for(edge.confidence),
        (evidence_at(scanned.side.root, edge.provenance.file, edge.provenance.line),),
    )


def _collect_groups(pair: _Pair, removed: set[str], flagged: dict[str, frozenset[Flag]]) -> dict[str, _Group]:
    """One group per caller function. Edges to removed symbols come from the base graph, the others from the head graph."""
    groups: dict[str, _Group] = {}
    seen: set[tuple[str, str, str, int, bool]] = set()
    for scanned, targets, was in ((pair.base, removed, True), (pair.head, set(flagged), False)):
        for edge in scanned.calls:
            src, dst = scanned.nodes.get(edge.src_id), scanned.nodes.get(edge.dst_id)
            if src is None or dst is None or edge.dst_id not in targets or edge.src_id in pair.blocked:
                continue
            if src.provenance.file in pair.failed_files:
                continue
            site = (edge.src_id, edge.dst_id, edge.provenance.file, edge.provenance.line, was)
            if site in seen:
                continue
            seen.add(site)
            label = INTERFACE_LABEL if was else _flag_label(flagged[edge.dst_id])
            caller_edge = CallerEdge(dst.qualname, label, _edge_claim(scanned, edge, src, dst, was), was, edge.confidence.value, bool(edge.attrs.get("override_possible")))
            group = groups.setdefault(edge.src_id, _Group(src, [], module=owner_module(src, scanned.nodes) or ""))
            group.edges.append(caller_edge)
            group.callees.add(edge.dst_id)
            group.interface = group.interface or was or Flag.SIGNATURE_CHANGED in flagged[edge.dst_id]
    return groups


def _caller_counts(groups: dict[str, _Group]) -> tuple[dict[str, int], dict[str, int]]:
    """(production, test) caller counts: changed symbol id -> number of distinct caller functions linked to it. A lower bound: unlinked calls are not counted."""
    production: Counter[str] = Counter()
    tests: Counter[str] = Counter()
    for group in groups.values():
        if is_production_node(group.node):
            production.update(group.callees)
        elif is_test_function(group.node):
            tests.update(group.callees)
    return dict(production), dict(tests)


def _users_by_symbol(groups: dict[str, _Group]) -> dict[str, tuple[SymbolUser, ...]]:
    """Changed symbol qualname -> its callers, each listed once at its first call site; production callers first. A lower bound."""
    found: dict[str, dict[str, SymbolUser]] = defaultdict(dict)
    for group in groups.values():
        for edge in group.edges:
            site = edge.claim.evidence[0]
            known = found[edge.callee].get(group.node.qualname)
            if known is None or (site.file, site.line) < (known.file, known.line):
                found[edge.callee][group.node.qualname] = SymbolUser(
                    group.node.qualname, group.module, site.file, site.line, not is_production_node(group.node), group.node.kind in MODULE_KINDS,
                )
    return {callee: tuple(sorted(users.values(), key=lambda u: (u.is_test, u.caller))) for callee, users in found.items()}


def _importers(pair: _Pair, scanned: _Scanned) -> dict[str, tuple[str, ...]]:
    """Module -> the production modules that import it (any import statement, resolved to a module in the model)."""
    found: dict[str, set[str]] = defaultdict(set)
    for edge in scanned.model.edges:
        if edge.kind not in IMPORT_KINDS:
            continue
        src, dst = scanned.nodes.get(edge.src_id), scanned.nodes.get(edge.dst_id)
        if src is None or dst is None or not is_production_node(src) or pair.failed_files & {src.provenance.file, dst.provenance.file}:
            continue
        importer, imported = owner_module(src, scanned.nodes), owner_module(dst, scanned.nodes)
        if importer and imported and importer != imported:
            found[imported].add(importer)
    return {module: tuple(sorted(importers)) for module, importers in found.items()}


def _dynamic_spots(node: Node | None, scanned: _Scanned) -> tuple[DynamicSpot, ...]:
    """The scanner's blind spots that fall inside the function's own lines. A node without a line range gets none (no guessing)."""
    if node is None or node.kind not in FUNCTION_KINDS or node.provenance.end_line is None:
        return ()
    first, last, file = node.provenance.line, node.provenance.end_line, node.provenance.file
    spots = (
        DynamicSpot(b.kind, b.detail, file, b.provenance.line)
        for b in scanned.model.blind_spots
        if b.kind in DYNAMIC_KINDS and b.provenance.file == file and first <= b.provenance.line <= last
    )
    return tuple(sorted(spots, key=lambda s: (s.line, s.kind)))


def _with_neighbours(
    changes: list[SymbolChange], counted: set[str], users: dict[str, tuple[SymbolUser, ...]], pair: _Pair,
    base_ids: dict[str, Node], head_ids: dict[str, Node],
) -> list[SymbolChange]:
    """Add who calls each counted symbol, and the module it lives in with the modules that import it."""
    base_importers, head_importers = _importers(pair, pair.base), _importers(pair, pair.head)
    out = []
    for c in changes:
        side, ids, importers = (pair.base, base_ids, base_importers) if c.change == "removed" else (pair.head, head_ids, head_importers)
        node = ids.get(c.node_id)
        module = owner_module(node, side.nodes) if node else None
        out.append(replace(
            c, users=users.get(c.qualname, ()) if c.node_id in counted else (), module=module,
            module_imported_by=importers.get(module, ()) if module else (), dynamic_spots=_dynamic_spots(node, side),
        ))
    return out


def _build_callers(groups: dict[str, _Group], pair: _Pair, changed_ids: set[str], include_tests: bool) -> tuple[tuple[CallerGroup, ...], tuple[CallerGroup, ...], int]:
    interface: list[CallerGroup] = []
    affected: list[CallerGroup] = []
    hidden = 0
    for caller_id, group in sorted(groups.items(), key=lambda item: item[1].node.qualname):
        is_test = not is_production_node(group.node)
        if is_test and not include_tests:
            hidden += 1
            continue
        edges = tuple(sorted(group.edges, key=lambda e: (e.claim.evidence[0].file, e.claim.evidence[0].line, e.callee)))
        built = CallerGroup(
            group.node.qualname, group.node.kind in (NodeKind.MODULE, NodeKind.PACKAGE), _caller_state(pair, caller_id, group.node, changed_ids),
            is_test, edges,
        )
        (interface if group.interface else affected).append(built)
    return tuple(interface), tuple(affected), hidden


# -- subsystem imports -----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ImportEdges:
    """Subsystem import edges of one side, with the import statements behind them (resolved or not)."""

    by_key: dict[tuple[str, str], list[Edge]]  # (importer subsystem, imported subsystem) -> edges, lowest line first
    statements: frozenset[tuple[str, str]]  # (importing file, imported name) for every import statement on the side


def _statement(edge: Edge) -> tuple[str, str]:
    return edge.provenance.file, edge.dst_ref or edge.dst_id


def _import_edges(pair: _Pair, scanned: _Scanned) -> _ImportEdges:
    by_key: dict[tuple[str, str], list[Edge]] = defaultdict(list)
    statements: set[tuple[str, str]] = set()
    for edge in scanned.model.edges:
        if edge.kind not in IMPORT_KINDS:
            continue
        statements.add(_statement(edge))
        src, dst = scanned.nodes.get(edge.src_id), scanned.nodes.get(edge.dst_id)
        if edge.kind is not EdgeKind.IMPORTS or edge.confidence is not Confidence.EXTRACTED or src is None or dst is None:
            continue  # only an import read straight off a statement and resolved to a module counts, never a retargeted or ambiguous one
        if pair.failed_files & {src.provenance.file, dst.provenance.file} or src.provenance.file in pair.suppressed:
            continue
        a, b = owner_module(src, scanned.nodes), owner_module(dst, scanned.nodes)
        if a is None or b is None or default_group(a) == default_group(b):
            continue
        by_key[(default_group(a), default_group(b))].append(edge)
    for edges in by_key.values():
        edges.sort(key=lambda e: (e.provenance.file, e.provenance.line))
    return _ImportEdges(by_key, frozenset(statements))


def _import_changes(pair: _Pair) -> tuple[tuple[ImportChange, ...], int]:
    """(subsystem import changes, how many were withheld).

    A subsystem import is reported only when an import statement was added (none of the head statements behind it existed on the
    base side) or deleted (none of the base statements behind it remain). Otherwise the edge changed only because something the
    statement names was added, removed or moved, or the import became type-only; that is a resolution change, so it is withheld.
    """
    base, head = _import_edges(pair, pair.base), _import_edges(pair, pair.head)
    rows: list[ImportChange] = []
    withheld = 0
    for change, keys, mine, other, scanned in (
        ("added", head.by_key.keys() - base.by_key.keys(), head, base, pair.head),
        ("removed", base.by_key.keys() - head.by_key.keys(), base, head, pair.base),
    ):
        for key in sorted(keys):
            fresh = [edge for edge in mine.by_key[key] if _statement(edge) not in other.statements]
            if not fresh:
                withheld += 1
                continue
            edge = fresh[0]
            claim = Claim(
                f"{key[0]} imports {key[1]}" if change == "added" else f"{key[0]} no longer imports {key[1]}", tier_for(edge.confidence),
                (evidence_at(scanned.side.root, edge.provenance.file, edge.provenance.line),),
            )
            rows.append(ImportChange(change, key[0], key[1], claim))
    return tuple(sorted(rows, key=lambda r: (r.source, r.target, r.change))), withheld


# -- not determinable ------------------------------------------------------------------------------------------------


def _unknown(text: str) -> Claim:
    return Claim(text, Tier.UNKNOWN)


def _where(label: str) -> str:
    return "the working tree" if label == WORKING_TREE else label


def _failed_file_lines(pair: _Pair) -> list[Claim]:
    lines: list[str] = []
    for file in sorted(pair.failed_files):
        reasons: dict[str, list[str]] = defaultdict(list)
        for scanned in (pair.base, pair.head):
            if file in scanned.failed:
                reasons[scanned.failed[file]].append(_where(scanned.side.label))
        what = "; ".join(f"{reason} in {' and in '.join(places)}" for reason, places in reasons.items())
        lines.append(f"{file} {what}: nothing in it can be compared, and callers located in it are missing")
    if pair.suppressed:
        files = ", ".join(sorted(pair.suppressed))
        lines.append(f"call comparison suppressed for {files}: they import from a file that does not parse, so their calls may look unresolved for that reason alone")
    return [_unknown(line) for line in lines]


def _duplicate_lines(pair: _Pair) -> list[Claim]:
    by_file: dict[str, set[str]] = defaultdict(set)
    for scanned in (pair.base, pair.head):
        for node in scanned.model.nodes:
            if node.kind in SYMBOL_KINDS and _canonical(node.qualname) in pair.duplicates and node.provenance.file in pair.changed_files:
                by_file[node.provenance.file].add(_canonical(node.qualname))
    return [
        _unknown(f"{file}: {', '.join(sorted(names))} defined more than once (property setter, if/else branch, @overload or redefinition): not compared")
        for file, names in sorted(by_file.items())
    ]


def _unresolved_line(pair: _Pair, head_ids: dict[str, Node], changed: list[SymbolChange], head_sites: _CallSites) -> list[Claim]:
    examples: list[str] = []
    for change in changed:
        node = head_ids.get(change.node_id)
        if node is None or change.change == "removed" or node.kind not in FUNCTION_KINDS:
            continue
        examples.extend(c.callee for c in head_sites.unresolved(node))
    if not examples:
        return []
    count = len(examples)
    noun, pronoun = ("call", "it reaches") if count == 1 else ("calls", "one of them reaches")
    return [_unknown(
        f"{count} {noun} made by the changed functions could not be traced (e.g. {examples[0]}): "
        f"if {pronoun} a changed function, that caller is missing from the lists above"
    )]


def _import_withheld_line(withheld: int) -> list[Claim]:
    if not withheld:
        return []
    noun = "import" if withheld == 1 else "imports"
    return [_unknown(
        f"{withheld} subsystem {noun} appeared or disappeared although no import statement was edited (the module it names was added, removed or moved, "
        "or the import became or stopped being type-only): not reported as an import change"
    )]


GREP_HINT_MAX_NAMES = 5


def search_name(change: SymbolChange) -> str | None:
    """The name to search the code for when this change can break callers the scanner misses (removed, or interface changed), else None.
    A constructor is searched by its class name, since constructor calls are written `Class(...)`."""
    if change.kind not in ("function", "method", "class") or not (change.change == "removed" or Flag.SIGNATURE_CHANGED in change.flags):
        return None
    parts = change.qualname.split(".")
    name = parts[-2] if parts[-1] == "__init__" and len(parts) > 1 else parts[-1]
    return name if name.isidentifier() else None  # quoted in the printed command, so only plain identifiers are allowed through


def _grep_hint(changes: list[SymbolChange]) -> list[str]:
    """A `git grep` for the names whose interface changed or that were removed: the one check that sees the callers the scanner misses."""
    names = {name for c in changes if (name := search_name(c))}
    if not names:
        return []
    shown = sorted(names)[:GREP_HINT_MAX_NAMES]
    more = f" (and {len(names) - len(shown)} more names)" if len(names) > len(shown) else ""
    patterns = " ".join(f"-e '{n}('" for n in shown)
    return [f"To look for callers Argus missed, search the code for the changed names (from the repository root): git grep -n {patterns}{more}"]


def _standing_lines(changes: list[SymbolChange], renames: tuple[FileRename, ...] = ()) -> list[Claim]:
    renamed = {f for r in renames for f in (r.old, r.new)}
    removed = [c for c in changes if c.change == "removed" and c.file not in renamed]
    added = [c for c in changes if c.change == "added" and c.file not in renamed]
    lines = [
        "Behavioral correctness and runtime effects. BODY_CHANGED is structural change detection (the syntax tree differs); it is not evidence that behavior changed",
        "Dynamic dependencies (getattr, reflection, monkeypatching, decorator side effects, dispatch tables)",
        "Module-level code and class-level code (class attributes, bases, class decorators) is not compared",
        "The caller list may be incomplete: Argus does not see calls made through a variable or alias, on obj.method(), to an inherited method, "
        "from a nested function, or inside an if/elif condition, so a missing caller does not mean there is none",
    ]
    lines.extend(_grep_hint(changes))
    if any(c.kind in ("function", "method") for c in changes if c.change != "modified"):
        lines.append("Call changes are compared only for functions present on both sides; calls made by added or removed functions are not listed as call changes (an added function still appears as a caller)")
    if removed:
        lines.append("For removed symbols: whether a surviving caller still calls the removed name or was updated (an unresolved call leaves no edge)")
    if removed and added:
        lines.append("Only a file renamed with identical content is detected; a rename with edits, or of a function inside a file, shows as a removal plus an addition")
    if any(c.qualname.endswith(".__init__") for c in changes if c.change != "added"):
        lines.append("Callers of a changed __init__ are not linked (constructor calls link to the class)")
    return [_unknown(line) for line in lines]


# -- files without a symbol change, and renames -----------------------------------------------------------------------


def _renames(pair: _Pair, changes: list[SymbolChange]) -> tuple[FileRename, ...]:
    """Removed and added files with the same content hash, paired only when exactly one of each has that hash (empty files tie often)."""
    gone: dict[str, list[str]] = defaultdict(list)
    new: dict[str, list[str]] = defaultdict(list)
    for rel in pair.changed_files - pair.failed_files:
        if rel not in pair.head.hashes:
            gone[pair.base.hashes[rel]].append(rel)
        elif rel not in pair.base.hashes:
            new[pair.head.hashes[rel]].append(rel)
    pairs = [
        (gone[d][0], new[d][0]) for d in gone.keys() & new.keys()
        if len(gone[d]) == 1 and len(new[d]) == 1 and (pair.head.side.root / new[d][0]).read_bytes().strip()  # empty files carry no evidence of a move
    ]
    return tuple(
        FileRename(old, now, sum(1 for n in pair.head.model.nodes if n.kind in SYMBOL_KINDS and n.provenance.file == now),
                   Claim(f"{old} renamed to {now} (content identical)", Tier.FACT, (evidence_at(pair.head.side.root, now, 1),)))
        for old, now in sorted(pairs)
    )


def _file_kind(pair: _Pair, rel: str) -> FileKind:
    if rel not in pair.base.hashes:
        return "added"
    if rel not in pair.head.hashes:
        return "removed"
    before = ast.dump(ast.parse((pair.base.side.root / rel).read_bytes()))  # no line numbers: comments and formatting cannot show
    return "formatting_only" if before == ast.dump(ast.parse((pair.head.side.root / rel).read_bytes())) else "outside_symbols"


FILE_CHANGE_TEXT: dict[str, str] = {
    "outside_symbols": "code outside functions changed (module-level or class-level, not compared)",
    "formatting_only": "comments or formatting only (syntax tree identical)",
    "added": "new file with no functions or classes",
    "removed": "file removed; it had no functions or classes",
    "relocated": "file added or removed; its functions and classes keep their qualified names and are compared by name",
    "added_empty": "new empty file",
    "removed_empty": "removed empty file",
}


def _file_changes(pair: _Pair, changes: list[SymbolChange], renames: tuple[FileRename, ...]) -> tuple[tuple[FileChange, ...], list[Claim]]:
    """Changed files that no SymbolChange accounts for. Files that failed to parse or define a name twice are skipped: they are already
    reported as not determinable, and calling them module-level would be a guess."""
    unsure = {n.provenance.file for scanned in (pair.base, pair.head) for n in scanned.model.nodes if n.id in pair.blocked}
    skip = {c.file for c in changes} | unsure | pair.failed_files | {f for r in renames for f in (r.old, r.new)}
    found: list[FileChange] = []
    problems: list[Claim] = []
    for rel in sorted(pair.changed_files - skip):
        try:
            kind = _file_kind(pair, rel)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            problems.append(_unknown(f"{rel} changed but could not be re-read to tell what kind of change it is"))
            continue
        side = pair.base if kind == "removed" else pair.head
        if kind in ("added", "removed"):
            if not (side.side.root / rel).read_bytes().strip():
                kind = "added_empty" if kind == "added" else "removed_empty"
            elif any(n.kind in SYMBOL_KINDS and n.provenance.file == rel for n in side.model.nodes):
                kind = "relocated"  # its symbols are not added or removed (same qualified names elsewhere), so "no functions" would be false
        found.append(FileChange(rel, kind, Claim(f"{rel}: {FILE_CHANGE_TEXT[kind]}", Tier.FACT, (evidence_at(side.side.root, rel, 1),))))
    return tuple(found), problems


def _blocked_caller_lines(pair: _Pair, removed: set[str], flagged: dict[str, frozenset[Flag]]) -> list[Claim]:
    """Callers of a removed or flagged symbol that sit in code defined more than once. They cannot be compared, and without this line they would vanish."""
    by_file: dict[str, set[str]] = defaultdict(set)
    for scanned, targets in ((pair.base, removed), (pair.head, set(flagged))):
        for edge in scanned.calls:
            src = scanned.nodes.get(edge.src_id)
            if src is None or edge.dst_id not in targets or edge.src_id not in pair.blocked or src.provenance.file in pair.failed_files:
                continue
            by_file[src.provenance.file].add(_canonical(src.qualname))
    return [
        _unknown(f"{file}: {', '.join(sorted(names))} calls a removed or changed symbol but is defined more than once, so it is not compared and is missing from the caller sections")
        for file, names in sorted(by_file.items())
    ]


# -- entry point -----------------------------------------------------------------------------------------------------


def _with_call_fits(changes: list[SymbolChange], fits: tuple[CallFit, ...]) -> list[SymbolChange]:
    """Each symbol with the calls checked against it. A call to a class is filed under the `__init__` that was checked."""
    by_target: dict[str, list[CallFit]] = defaultdict(list)
    for fit in fits:
        by_target[fit.target_id].append(fit)
    return [replace(c, call_fits=tuple(by_target[c.node_id])) if c.node_id in by_target else c for c in changes]


def _call_fits(pair: _Pair, by_id: dict[str, SymbolChange]) -> tuple[tuple[CallFit, ...], list[Claim]]:
    """Part B of the breakage check: head-side calls to a function whose inputs changed. The second part is a standing line when some
    calls could not even be looked at (their file does not parse, or a symbol name in them is duplicated)."""
    changed = frozenset(i for i, c in by_id.items() if c.change == "modified" and Flag.SIGNATURE_CHANGED in c.flags)
    kept = [
        e for e in pair.head.calls
        if e.src_id not in pair.blocked and e.dst_id not in pair.blocked and e.provenance.file not in pair.failed_files
    ]
    parents: dict[str, list[str]] = defaultdict(list)
    for edge in pair.head.model.edges:
        if edge.kind is EdgeKind.INHERITS:
            parents[edge.src_id].append(edge.dst_id)
    fits = check_signature_changes(
        kept, pair.base.nodes, pair.head.nodes, changed, pair.head.side.root, {k: tuple(v) for k, v in parents.items()},
    )
    targets = {i.rpartition(".")[0].replace("method:", "class:", 1) for i in changed if i.endswith(".__init__")} | changed
    kept_ids = {e.id for e in kept}
    skipped = sum(1 for e in pair.head.calls if e.dst_id in targets and e.id not in kept_ids)
    noun, verb = ("call", "was") if skipped == 1 else ("calls", "were")
    note = [_unknown(f"{skipped} {noun} to functions with changed inputs {verb} not checked: the file that makes the call does not parse, or a name in it is defined twice")] if skipped else []
    return fits, note


def build_diff_view(
    base: Side, head: Side, base_model: ArchModel, head_model: ArchModel, resolve_v2: bool, include_tests: bool = False,
) -> DiffView:
    """Compare two scanned sides. `resolve_v2` is the resolver mode both scans used (shown in the header, never mixed)."""
    pair = _pair(base, head, base_model, head_model)
    base_sites, head_sites = _CallSites(pair.base), _CallSites(pair.head)
    base_ids, head_ids = _eligible(pair, pair.base), _eligible(pair, pair.head)
    common = sorted(base_ids.keys() & head_ids.keys())
    base_bodies, head_bodies = _BodyIndex(pair.base.side), _BodyIndex(pair.head.side)
    calls = _compare_calls(pair, set(common), head_ids, base_sites, head_bodies)
    modified, unreadable = _modified(pair, common, base_ids, head_ids, calls, base_bodies, head_bodies)
    changes = sorted(
        [*_added_removed(pair, base_ids, head_ids), *modified], key=lambda c: (c.file, c.line, c.qualname, c.change),
    )
    by_id = {c.node_id: c for c in changes}
    removed = {node_id for node_id, c in by_id.items() if c.change == "removed"}
    flagged = {node_id: c.flags for node_id, c in by_id.items() if c.change == "modified" and c.flags & set(CALLER_LABEL_FLAGS)}
    changed_ids = {node_id for node_id, c in by_id.items() if c.change == "modified"}
    groups = _collect_groups(pair, removed, flagged)
    interface, affected, hidden = _build_callers(groups, pair, changed_ids, include_tests)
    production_callers, test_callers = _caller_counts(groups)
    counted = removed | set(flagged)
    changes = [
        replace(c, known_callers=production_callers.get(c.node_id, 0), test_callers=test_callers.get(c.node_id, 0)) if c.node_id in counted else c
        for c in changes
    ]
    changes = _with_neighbours(changes, counted, _users_by_symbol(groups), pair, base_ids, head_ids)
    call_fits, unchecked = _call_fits(pair, by_id)
    changes = _with_call_fits(changes, call_fits)
    imports, withheld = _import_changes(pair)
    renames = _renames(pair, changes)
    file_changes, unclassified = _file_changes(pair, changes, renames)
    not_determinable = [
        *_standing_lines(changes, renames), *unclassified, *_blocked_caller_lines(pair, removed, flagged), *_failed_file_lines(pair), *_duplicate_lines(pair), *unreadable, *unchecked, *_unresolved_line(pair, head_ids, changes, head_sites),
        *_import_withheld_line(withheld),
    ]
    return DiffView(
        base.label, head.label, resolve_v2, len(pair.changed_files), len(head.untracked & pair.changed_files), tuple(changes), calls.calls,
        interface, affected, hidden, imports, tuple(not_determinable), _not_scanned(base, head), frozenset(head.files) - frozenset(base.files),
        file_changes, renames, tuple(sorted(pair.failed_files)), call_fits,
    )
