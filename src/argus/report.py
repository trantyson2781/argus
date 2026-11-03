"""The two Phase 0 edge-quality gates, computed here rather than eyeballed.

* AMBIGUOUS rate (measured automatically): at most 15% of structural edges.
* Precision on EXTRACTED edges (measured by hand-verifying a reproducible random sample): at least 95%.

`argus explore sample` prints both.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from argus.model.schema import ArchModel, Confidence, Edge, EdgeKind

AMBIGUOUS_RATE_GATE = 0.15
"""Phase 0 coverage gate: at most 15% of structural edges may be AMBIGUOUS."""

EXTRACTED_PRECISION_GATE = 0.95
"""Phase 0 precision gate: at least 95% of the hand-checked EXTRACTED edges must be right."""

STRUCTURAL_EDGE_KINDS = frozenset(
    {EdgeKind.IMPORTS, EdgeKind.IMPORTS_TYPE_ONLY, EdgeKind.CALLS, EdgeKind.INHERITS}
)


def structural_edges(model: ArchModel) -> list[Edge]:
    return [e for e in model.edges if e.kind in STRUCTURAL_EDGE_KINDS]


def ambiguous_rate(model: ArchModel) -> float:
    """Share of structural edges whose target Argus could not resolve (0.0 when there are none)."""
    structural = structural_edges(model)
    if not structural:
        return 0.0
    return sum(1 for e in structural if e.confidence is Confidence.AMBIGUOUS) / len(structural)


@dataclass(frozen=True, slots=True)
class EdgeSample:
    """One edge presented for hand verification."""

    edge_id: str
    kind: str
    confidence: str
    source: str
    target: str
    site: str
    flags: tuple[str, ...]


def sample_edges(model: ArchModel, count: int = 50, seed: int = 0) -> tuple[EdgeSample, ...]:
    """Draw a reproducible random sample of structural edges to hand-verify.

    Precision cannot be computed automatically: a wrong edge looks exactly like
    a right one to the extractor that produced it. This makes the manual audit
    from the roadmap's week-1 checklist repeatable.
    """
    structural = structural_edges(model)
    rng = random.Random(seed)
    chosen = rng.sample(structural, min(count, len(structural)))
    index = {n.id: n.qualname for n in model.nodes}
    return tuple(
        EdgeSample(
            edge_id=edge.id,
            kind=edge.kind.value,
            confidence=edge.confidence.value,
            source=index.get(edge.src_id, edge.src_id),
            target=edge.dst_ref or index.get(edge.dst_id, edge.dst_id),
            site=edge.provenance.as_ref(),
            flags=tuple(sorted(f.value for f in edge.flags)),
        )
        for edge in sorted(chosen, key=lambda e: (e.provenance.file, e.provenance.line))
    )
