"""Intermediate per-file facts, produced before cross-file resolution.

A file is parsed once into `FileFacts`. Nothing here is resolved against other
files yet — import targets are still textual and relative levels are unresolved.
That separation is what lets Phase 3 cache per-file extraction by content hash
and only re-resolve the dirty subgraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from argus.model.schema import ImportFlag, NodeKind


@dataclass(frozen=True, slots=True)
class RawImport:
    """One imported name, exactly as written."""

    module: str | None       # `from <module> import ...`; None for plain `import x`
    name: str                # bound/imported name; "*" for star imports
    asname: str | None
    level: int               # relative-import dots (0 = absolute)
    line: int
    type_only: bool = False
    flags: frozenset[ImportFlag] = frozenset()

    @property
    def local_name(self) -> str:
        """The name actually bound in the importing module's namespace."""
        if self.asname:
            return self.asname
        if self.module is None:
            # `import a.b.c` binds `a`; `import a.b.c as d` binds `d`.
            return self.name.partition(".")[0]
        return self.name


@dataclass(frozen=True, slots=True)
class RawParameter:
    name: str
    kind: str  # positional-only | positional | vararg | keyword-only | kwarg
    annotation: str | None = None
    default: str | None = None
    has_default: bool = False


@dataclass(frozen=True, slots=True)
class RawSignature:
    parameters: tuple[RawParameter, ...] = ()
    returns: str | None = None
    is_async: bool = False

    def as_payload(self) -> dict[str, Any]:
        return {
            "parameters": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "annotation": p.annotation,
                    "default": p.default,
                    "has_default": p.has_default,
                }
                for p in self.parameters
            ],
            "returns": self.returns,
            "is_async": self.is_async,
        }


@dataclass(frozen=True, slots=True)
class RawSymbol:
    """A class, function, method, attribute or constant defined in a file."""

    kind: NodeKind
    name: str
    local_qualname: str          # dotted path within the module, e.g. "Invoice.total"
    line: int
    end_line: int | None = None
    parent_local: str | None = None
    decorators: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()
    signature: RawSignature | None = None
    docstring: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def is_conventionally_public(self) -> bool:
        return not any(part.startswith("_") for part in self.local_qualname.split("."))


@dataclass(frozen=True, slots=True)
class RawBlindSpot:
    kind: str
    detail: str
    line: int


@dataclass(frozen=True, slots=True)
class RawCall:
    """A call site, with the callee written as source text."""

    callee: str          # e.g. "billing.charge" or "charge"
    line: int
    caller_local: str | None  # local qualname of the enclosing function, if any


@dataclass(frozen=True, slots=True)
class RawProtocolImpl:
    class_local: str
    protocol_name: str


@dataclass(frozen=True, slots=True)
class FileFacts:
    qualname: str
    path: str
    docstring: str | None = None
    dunder_all: tuple[str, ...] | None = None
    dunder_all_line: int | None = None
    imports: tuple[RawImport, ...] = ()
    symbols: tuple[RawSymbol, ...] = ()
    calls: tuple[RawCall, ...] = ()
    blind_spots: tuple[RawBlindSpot, ...] = ()
    protocol_impls: tuple[RawProtocolImpl, ...] = ()
    parse_error: str | None = None

