"""Assemble an `ArchModel` from a directory of Python source.

Pipeline: discover files -> parse each to `FileFacts` -> build the symbol table ->
resolve every import/base/call into edges.

Nothing here calls an LLM and nothing here renders. Phase 0 is model only.
"""

from __future__ import annotations

import builtins
import time
from pathlib import Path

from argus.extract.discovery import SourceFile, discover, package_qualnames
from argus.extract.facts import FileFacts, RawSymbol
from argus.extract.python_ast import parse_file
from argus.extract.resolve import SymbolTable, Target
from argus.extract.resolve_calls import CallResolverV2, resolve_v2_enabled
from argus.model.ids import edge_id, node_id, unresolved_id
from argus.model.schema import (
    ArchModel,
    BlindSpot,
    Confidence,
    Edge,
    EdgeKind,
    ImportFlag,
    Node,
    NodeKind,
    Provenance,
    ScanStats,
)

EXTRACTOR = "scanner"
BUILTIN_ROOTS = frozenset(dir(builtins))
"""Names resolvable without any import; not model nodes, but not unknowns either."""
PROTOCOL_MARKERS = frozenset({"Protocol", "ABC"})


def _is_protocol_marker(base: str) -> bool:
    """`Protocol`, `typing.Protocol[T]`, `abc.ABC`: the name itself, never a name that merely contains one (`ABCMeta`, `ProtocolBase`)."""
    return base.partition("[")[0].rpartition(".")[2] in PROTOCOL_MARKERS


class _Builder:
    """Accumulates nodes, edges and contracts for one scan."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.blind_spots: list[BlindSpot] = []
        self.unresolved_calls = 0
        self.call_resolver: CallResolverV2 | None = None

    def rel(self, path: str) -> str:
        try:
            return str(Path(path).relative_to(self.root))
        except ValueError:
            return path

    def add_edge(
        self,
        kind: EdgeKind,
        src_id: str,
        dst_id: str,
        provenance: Provenance,
        confidence: Confidence,
        dst_ref: str | None = None,
        flags: frozenset[ImportFlag] = frozenset(),
        **attrs: object,
    ) -> None:
        self.edges.append(
            Edge(
                id=edge_id(kind, src_id, dst_id, provenance),
                kind=kind,
                src_id=src_id,
                dst_id=dst_id,
                provenance=provenance,
                confidence=confidence,
                dst_ref=dst_ref,
                flags=flags,
                attrs=dict(attrs),
            )
        )


def scan(root: Path, resolve_v2: bool | None = None) -> ArchModel:
    """Extract the architecture model for the codebase at `root`.

    `resolve_v2` selects the extra call-resolution rules (see resolve_calls.py); None defers to ARGUS_RESOLVE_V2, which is on unless set to 0.
    """
    started = time.perf_counter()
    root = root.resolve()
    files = discover(root)
    builder = _Builder(root if root.is_dir() else root.parent)

    facts: dict[str, FileFacts] = {}
    failed = 0
    for source_file in files:
        parsed = parse_file(source_file.path, source_file.qualname)
        if parsed.parse_error:
            failed += 1
            builder.blind_spots.append(
                BlindSpot(
                    kind="parse_error",
                    detail=parsed.parse_error,
                    provenance=Provenance(
                        file=builder.rel(str(source_file.path)), line=1, extractor=EXTRACTOR
                    ),
                )
            )
            continue
        facts[source_file.qualname] = parsed

    table = SymbolTable(files, facts)
    if resolve_v2_enabled(resolve_v2):
        builder.call_resolver = CallResolverV2(table, facts)
    by_qualname = {f.qualname: f for f in files}

    _build_packages(builder, files)
    for qualname, fact in facts.items():
        _build_module(builder, table, by_qualname[qualname], fact)

    duration_ms = int((time.perf_counter() - started) * 1000)
    return ArchModel(
        root=str(root),
        nodes=tuple(builder.nodes),
        edges=tuple(builder.edges),
        blind_spots=tuple(builder.blind_spots),
        stats=ScanStats(
            files_scanned=len(files), files_failed=failed, duration_ms=duration_ms
        ),
    )


def _build_packages(builder: _Builder, files: tuple[SourceFile, ...]) -> None:
    init_paths = {f.qualname: f.path for f in files if f.is_package_init}
    for package in package_qualnames(files):
        path = init_paths.get(package)
        provenance = Provenance(
            file=builder.rel(str(path)) if path else package.replace(".", "/"),
            line=1,
            extractor=EXTRACTOR,
        )
        parent = package.rpartition(".")[0]
        builder.nodes.append(
            Node(
                id=node_id(NodeKind.PACKAGE, package),
                kind=NodeKind.PACKAGE,
                name=package.rpartition(".")[2],
                qualname=package,
                provenance=provenance,
                parent_id=node_id(NodeKind.PACKAGE, parent) if parent else None,
                is_public=not package.rpartition(".")[2].startswith("_"),
                attrs={"is_namespace_package": path is None},
            )
        )
        if parent:
            builder.add_edge(
                EdgeKind.CONTAINS,
                node_id(NodeKind.PACKAGE, parent),
                node_id(NodeKind.PACKAGE, package),
                provenance,
                Confidence.EXTRACTED,
            )


def _build_module(
    builder: _Builder, table: SymbolTable, source_file: SourceFile, fact: FileFacts
) -> None:
    relative = builder.rel(fact.path)
    module_provenance = Provenance(file=relative, line=1, extractor=EXTRACTOR)
    module_node_id = node_id(NodeKind.MODULE, fact.qualname)
    package = source_file.package_qualname

    builder.nodes.append(
        Node(
            id=module_node_id,
            kind=NodeKind.MODULE,
            name=fact.qualname.rpartition(".")[2] or fact.qualname,
            qualname=fact.qualname,
            provenance=module_provenance,
            parent_id=node_id(NodeKind.PACKAGE, package) if package else None,
            is_public=not fact.qualname.rpartition(".")[2].startswith("_"),
            attrs={
                "docstring": fact.docstring,
                "is_package_init": source_file.is_package_init,
                "has_dunder_all": fact.dunder_all is not None,
            },
        )
    )
    if package:
        builder.add_edge(
            EdgeKind.CONTAINS,
            node_id(NodeKind.PACKAGE, package),
            module_node_id,
            module_provenance,
            Confidence.EXTRACTED,
        )

    index = _SymbolIndex(fact)
    _build_symbols(builder, fact, module_node_id, relative, index)
    _build_imports(builder, table, fact, module_node_id, relative)
    _build_inheritance(builder, table, fact, relative)
    _build_calls(builder, table, fact, module_node_id, relative, index)
    _build_protocol_edges(builder, table, fact, relative, index)

    for raw in fact.blind_spots:
        builder.blind_spots.append(
            BlindSpot(
                kind=raw.kind,
                detail=raw.detail,
                provenance=Provenance(file=relative, line=raw.line, extractor="python_ast"),
            )
        )


class _SymbolIndex:
    """Local-qualname lookups for one module, built once instead of re-scanned."""

    def __init__(self, fact: FileFacts) -> None:
        self._fact = fact
        self._by_local = {symbol.local_qualname: symbol for symbol in fact.symbols}

    def kind_of(self, local_qualname: str) -> NodeKind:
        symbol = self._by_local.get(local_qualname)
        return symbol.kind if symbol else NodeKind.CLASS

    def node_id_of(self, local_qualname: str) -> str:
        symbol = self._by_local.get(local_qualname)
        if symbol is None:
            return node_id(NodeKind.MODULE, self._fact.qualname)
        return _symbol_node_id(self._fact.qualname, symbol)


def _symbol_node_id(module: str, symbol: RawSymbol) -> str:
    return node_id(symbol.kind, f"{module}.{symbol.local_qualname}")


def _is_public(fact: FileFacts, symbol: RawSymbol) -> bool:
    if symbol.parent_local is not None:
        return symbol.is_conventionally_public
    if fact.dunder_all is not None:
        return symbol.name in fact.dunder_all
    return symbol.is_conventionally_public


def _build_symbols(
    builder: _Builder,
    fact: FileFacts,
    module_node_id: str,
    relative: str,
    index: _SymbolIndex,
) -> None:
    for symbol in fact.symbols:
        provenance = Provenance(
            file=relative, line=symbol.line, end_line=symbol.end_line, extractor="python_ast"
        )
        parent_id = (
            module_node_id
            if symbol.parent_local is None
            else node_id(
                index.kind_of(symbol.parent_local),
                f"{fact.qualname}.{symbol.parent_local}",
            )
        )
        builder.nodes.append(
            Node(
                id=_symbol_node_id(fact.qualname, symbol),
                kind=symbol.kind,
                name=symbol.name,
                qualname=f"{fact.qualname}.{symbol.local_qualname}",
                provenance=provenance,
                parent_id=parent_id,
                is_public=_is_public(fact, symbol),
                attrs={
                    "decorators": list(symbol.decorators),
                    "bases": list(symbol.bases),
                    "docstring": symbol.docstring,
                    "signature": symbol.signature.as_payload() if symbol.signature else None,
                    **symbol.attrs,
                },
            )
        )
        builder.add_edge(
            EdgeKind.CONTAINS,
            parent_id,
            _symbol_node_id(fact.qualname, symbol),
            provenance,
            Confidence.EXTRACTED,
        )


def _build_imports(
    builder: _Builder, table: SymbolTable, fact: FileFacts, module_node_id: str, relative: str
) -> None:
    for raw in fact.imports:
        target = table.resolve(fact.qualname, raw)
        provenance = Provenance(file=relative, line=raw.line, extractor="python_ast")
        kind = EdgeKind.IMPORTS_TYPE_ONLY if raw.type_only else EdgeKind.IMPORTS
        dst_id = _target_node_id(table, target)

        builder.add_edge(
            kind,
            module_node_id,
            dst_id,
            provenance,
            target.confidence,
            dst_ref=target.qualname,
            flags=raw.flags,
            is_internal=target.is_internal,
            canonicalized_from=target.canonicalized_from,
            resolved_via=target.resolved_via,
        )

        # A consumer importing through a package init still creates a real
        # re-export relationship worth recording separately.
        if target.canonicalized_from and target.is_internal:
            via = target.canonicalized_from.rpartition(".")[0]
            if table.is_module(via) and via != target.module:
                builder.add_edge(
                    EdgeKind.REEXPORTS,
                    node_id(NodeKind.MODULE, via),
                    dst_id,
                    provenance,
                    Confidence.EXTRACTED,
                    dst_ref=target.qualname,
                )


def _target_node_id(table: SymbolTable, target: Target) -> str:
    """The id of the node a resolved import/call target actually points at.

    A resolved `Target` only carries a bare top-level name (e.g. "ArchModel"),
    not the `NodeKind` needed to build the same id the defining module used
    (`class:pkg.mod.ArchModel`, not a kind-less `symbol:pkg.mod.ArchModel`).
    Without this lookup, every cross-module `from x import y` edge dangles to
    an id no node in the model actually has (#module-level-analysis-blind-spot).
    """
    if not target.is_internal or target.module is None:
        return unresolved_id(target.qualname)
    if target.symbol is None:
        return node_id(NodeKind.MODULE, target.module)
    kind = _symbol_kind_in_module(table, target.module, target.symbol)
    if kind is None:
        return unresolved_id(target.qualname)
    return node_id(kind, f"{target.module}.{target.symbol}")


def _symbol_kind_in_module(table: SymbolTable, module: str, symbol: str) -> NodeKind | None:
    """The `NodeKind` a top-level name was declared with, or `None` if unknown to Argus."""
    found = table.top_level_symbol(module, symbol)
    return found.kind if found else None


def _build_inheritance(
    builder: _Builder, table: SymbolTable, fact: FileFacts, relative: str
) -> None:
    for symbol in fact.symbols:
        if symbol.kind is not NodeKind.CLASS:
            continue
        src_id = _symbol_node_id(fact.qualname, symbol)
        provenance = Provenance(file=relative, line=symbol.line, extractor="python_ast")
        for base in symbol.bases:
            if "=" in base or "(" in base:
                continue  # metaclass= / generic call form: not a plain base
            resolved = _resolve_reference(table, fact, base)
            if resolved is None:
                is_builtin = base in BUILTIN_ROOTS
                kind = EdgeKind.IMPLEMENTS if _is_protocol_marker(base) else EdgeKind.INHERITS
                builder.add_edge(
                    kind,
                    src_id,
                    unresolved_id(f"builtins.{base}" if is_builtin else base),
                    provenance,
                    Confidence.EXTRACTED if is_builtin else Confidence.AMBIGUOUS,
                    dst_ref=f"builtins.{base}" if is_builtin else base,
                )
                continue

            target = table.top_level_symbol(resolved.module, resolved.symbol) if resolved.module and resolved.symbol else None
            extends_protocol = target is not None and target.kind is NodeKind.CLASS and any(_is_protocol_marker(b) for b in target.bases)

            kind = EdgeKind.IMPLEMENTS if (extends_protocol or _is_protocol_marker(base)) else EdgeKind.INHERITS
            builder.add_edge(
                kind,
                src_id,
                _target_node_id(table, resolved),
                provenance,
                resolved.confidence,
                dst_ref=resolved.qualname,
            )


def _build_calls(
    builder: _Builder,
    table: SymbolTable,
    fact: FileFacts,
    module_node_id: str,
    relative: str,
    index: _SymbolIndex,
) -> None:
    """Resolve call sites through the module's own import aliases.

    Deliberately conservative: a callee whose root name is neither an import
    alias nor a module-level definition is counted as unresolved rather than
    guessed at.
    """
    for call in fact.calls:
        resolved = _resolve_reference(table, fact, call.callee)
        dst_id, attrs = None, {}
        if builder.call_resolver is not None:
            refined = builder.call_resolver.refine(fact, call, resolved)
            resolved, dst_id, attrs = refined.target, refined.dst_id, refined.attrs
        if resolved is None:
            builder.unresolved_calls += 1
            continue
        src_id = (
            module_node_id if call.caller_local is None else index.node_id_of(call.caller_local)
        )
        builder.add_edge(
            EdgeKind.CALLS,
            src_id,
            dst_id or _target_node_id(table, resolved),
            Provenance(file=relative, line=call.line, extractor="python_ast"),
            Confidence.INFERRED,
            dst_ref=resolved.qualname,
            **attrs,
        )


def _resolve_reference(table: SymbolTable, fact: FileFacts, reference: str) -> Target | None:
    """Resolve dotted source text like `billing.charge` against a module's scope."""
    root, _, rest = reference.partition(".")
    if not root or root in BUILTIN_ROOTS:
        return None

    alias = table.alias(fact.qualname, root)
    if alias is not None:
        if alias.is_module_import:
            base_module, name = alias.module_ref, rest
        else:
            base_module, name = alias.module_ref, ".".join(filter(None, [alias.name, rest]))
        if not name:
            return table.canonicalize(base_module.rpartition(".")[0], base_module.rpartition(".")[2])
        head, _, _tail = name.partition(".")
        return table.canonicalize(base_module, head)

    if table.defines(fact.qualname, root):
        return Target(
            ref=f"{fact.qualname}.{root}",
            module=fact.qualname,
            symbol=root,
            is_internal=True,
            confidence=Confidence.EXTRACTED,
        )
    return None


def _build_protocol_edges(
    builder: _Builder,
    table: SymbolTable,
    fact: FileFacts,
    relative: str,
    index: _SymbolIndex,
) -> None:
    for raw in fact.protocol_impls:
        resolved = _resolve_reference(table, fact, raw.protocol_name)
        if (
            resolved
            and resolved.qualname
            not in (
                "typing.Protocol",
                "typing_extensions.Protocol",
                "abc.ABC",
                "Protocol",
                "ABC",
            )
        ):
            src_id = index.node_id_of(raw.class_local)
            provenance = Provenance(file=relative, line=1, extractor="python_ast")
            builder.add_edge(
                EdgeKind.IMPLEMENTS,
                src_id,
                _target_node_id(table, resolved),
                provenance,
                resolved.confidence,
                dst_ref=resolved.qualname,
            )
