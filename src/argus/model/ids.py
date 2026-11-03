"""Deterministic identity for nodes and edges.

Node ids are *stable across scans*, so the same symbol on two sides of a comparison has the same id. That means the id may only depend on kind + canonical qualname —
never on file position, ordering, or content.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from argus.model.schema import EdgeKind, NodeKind

if TYPE_CHECKING:
    from argus.model.schema import Provenance

UNRESOLVED_PREFIX = "unresolved"


def node_id(kind: NodeKind, qualname: str) -> str:
    return f"{kind.value}:{qualname}"


def unresolved_id(ref: str) -> str:
    """Id for a target Argus knows exists but cannot place in the model.

    Third-party imports land here. They are real dependency endpoints and must
    be representable, but they are not nodes we extracted.
    """
    return f"{UNRESOLVED_PREFIX}:{ref}"


def is_unresolved(node_id_value: str) -> bool:
    return node_id_value.startswith(f"{UNRESOLVED_PREFIX}:")


def edge_id(kind: EdgeKind, src_id: str, dst_id: str, provenance: Provenance) -> str:
    """Edges are identified by endpoints plus site.

    The site is included because the same (src, dst, kind) triple can legitimately
    occur at several places, and each occurrence is separately reviewable.
    """
    raw = f"{kind.value}|{src_id}|{dst_id}|{provenance.file}:{provenance.line}"
    return hashlib.blake2b(raw.encode(), digest_size=12).hexdigest()
