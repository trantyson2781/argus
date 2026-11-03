from pathlib import Path

import pytest

from argus.explore.claims import Claim, Evidence, Tier, evidence_at, tier_for
from argus.model.schema import Confidence


def test_fact_without_evidence_is_rejected() -> None:
    with pytest.raises(ValueError):
        Claim("x", Tier.FACT)
    with pytest.raises(ValueError):
        Claim("x", Tier.INFERENCE)


def test_unknown_needs_no_evidence() -> None:
    assert Claim("not stated", Tier.UNKNOWN).evidence == ()


def test_evidence_carries_file_line_and_source_hash(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    ev = evidence_at(tmp_path, "a.py", 1)
    assert (ev.file, ev.line) == ("a.py", 1) and len(ev.source_hash) == 16
    other = tmp_path / "b.py"
    other.write_text("x = 2\n")
    assert evidence_at(tmp_path, "b.py", 1).source_hash != ev.source_hash


def test_evidence_for_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        evidence_at(tmp_path, "nope.py", 1)


def test_confidence_maps_to_tier_without_upgrading() -> None:
    assert tier_for(Confidence.EXTRACTED) is Tier.FACT
    assert tier_for(Confidence.INFERRED) is Tier.INFERENCE
    assert tier_for(Confidence.AMBIGUOUS) is Tier.UNKNOWN
    assert Evidence("a.py", 1, "h").file == "a.py"
