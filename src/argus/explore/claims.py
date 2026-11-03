"""Evidence-linked claims. A FACT or INFERENCE without evidence cannot be constructed."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from argus.model.schema import Confidence

SOURCE_HASH_BYTES = 8


class Tier(StrEnum):
    """FACT: read off the source. INFERENCE: derived by resolution that can be wrong. UNKNOWN: not determinable."""

    FACT = "FACT"
    INFERENCE = "INFERENCE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class Evidence:
    file: str
    line: int
    source_hash: str

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    tier: Tier
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if self.tier is not Tier.UNKNOWN and not self.evidence:
            raise ValueError(f"{self.tier} claim needs evidence: {self.text!r}")


def file_hash(root: Path, file: str) -> str:
    return hashlib.blake2b((root / file).read_bytes(), digest_size=SOURCE_HASH_BYTES).hexdigest()


def evidence_at(root: Path, file: str, line: int) -> Evidence:
    """Evidence for `file:line`, hashed from the file as it is on disk now."""
    return Evidence(file, line, file_hash(root, file))


def tier_for(confidence: Confidence) -> Tier:
    """Map extractor confidence to a tier; never upgrades (INFERRED stays an inference)."""
    if confidence is Confidence.EXTRACTED:
        return Tier.FACT
    if confidence is Confidence.INFERRED:
        return Tier.INFERENCE
    return Tier.UNKNOWN
