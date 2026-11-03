"""Core architecture model: nodes, edges, contracts, provenance, snapshots.

Everything in Argus is a view or an action over this model. The schema is
frozen-dataclass based: extraction produces new values, never mutates.

Design invariants:
  * Every node and edge carries provenance (file:line + extractor + confidence).
  * Node identity is a deterministic string derived from kind + canonical qualname,
    so the same symbol has the same id across scans. This is what makes
    comparing two sides possible at all.
  * Edges may point at a target outside the model (a third-party library). Such
    edges keep the raw textual target in `dst_ref` and resolve to an
    unresolved node id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeKind(StrEnum):
    """What kind of thing a node is."""

    PACKAGE = "package"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    ATTRIBUTE = "attribute"
    CONSTANT = "constant"


class EdgeKind(StrEnum):
    """How one node relates to another."""

    CONTAINS = "contains"
    IMPORTS = "imports"
    IMPORTS_TYPE_ONLY = "imports-type-only"
    REEXPORTS = "reexports"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"


class Confidence(StrEnum):
    """How much the model trusts a fact.

    EXTRACTED  — read directly off the syntax tree; should be ~100% correct.
    INFERRED   — derived through resolution that can be wrong in edge cases.
    AMBIGUOUS  — a real relationship exists but Argus cannot say to what.
    """

    EXTRACTED = "EXTRACTED"
    INFERRED = "INFERRED"
    AMBIGUOUS = "AMBIGUOUS"


class ImportFlag(StrEnum):
    """Semantics that modify an import/dependency edge."""

    LAZY = "lazy"           # inside a function body — deferred, often cycle-avoidance
    OPTIONAL = "optional"   # under try/except ImportError — soft dependency
    CONDITIONAL = "conditional"  # under an `if` that is not TYPE_CHECKING
    STAR = "star"           # `from x import *` — contents not enumerable
    RELATIVE = "relative"   # written as a relative import


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a fact came from. Never optional — an unattributable fact is a bug."""

    file: str
    line: int
    extractor: str
    end_line: int | None = None
    column: int | None = None

    def as_ref(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True, slots=True)
class Node:
    """A thing in the architecture."""

    id: str
    kind: NodeKind
    name: str
    qualname: str
    provenance: Provenance
    confidence: Confidence = Confidence.EXTRACTED
    parent_id: str | None = None
    is_public: bool = True
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed relationship between two nodes."""

    id: str
    kind: EdgeKind
    src_id: str
    dst_id: str
    provenance: Provenance
    confidence: Confidence = Confidence.EXTRACTED
    dst_ref: str | None = None
    flags: frozenset[ImportFlag] = frozenset()
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BlindSpot:
    """A site where static reading provably cannot see through.

    Surfaced to the user rather than guessed at. These are the places a human
    reviewer should look hardest.
    """

    kind: str
    detail: str
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class ScanStats:
    files_scanned: int = 0
    files_failed: int = 0
    duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class ArchModel:
    """One immutable scan of a codebase's architecture."""

    root: str
    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    blind_spots: tuple[BlindSpot, ...] = ()
    git_ref: str | None = None
    tool_version: str = "0.1.0"
    stats: ScanStats = ScanStats()

    def nodes_of_kind(self, kind: NodeKind) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.kind is kind)

    def edges_of_kind(self, kind: EdgeKind) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.kind is kind)
