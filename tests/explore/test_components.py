from pathlib import Path

from argus.explore.claims import Tier
from argus.explore.components import analyze, default_group
from argus.extract import scan


def _by_name(root: Path):
    model = scan(root)
    return {c.name: c for c in analyze(model, root, default_group)}


def test_groups_are_first_two_segments_and_tests_are_excluded(mini_repo: Path) -> None:
    comps = _by_name(mini_repo)
    assert set(comps) == {"mini", "mini.__main__", "mini.cli", "mini.core", "mini.store", "mini.sub"}
    assert comps["mini.store"].modules == ("mini.store", "mini.store.db")


def test_docstring_is_a_fact_and_missing_docstring_is_unknown(mini_repo: Path) -> None:
    comps = _by_name(mini_repo)
    purpose = comps["mini.core"].purpose
    assert purpose.tier is Tier.FACT and "Core logic" in purpose.text
    assert (purpose.evidence[0].file, purpose.evidence[0].line) == ("src/mini/core.py", 1)
    assert comps["mini.store"].purpose.tier is Tier.UNKNOWN


def test_imports_between_components_carry_evidence(mini_repo: Path) -> None:
    comps = _by_name(mini_repo)
    dep = {d.other: d for d in comps["mini.core"].imports_out}["mini.store"]
    assert dep.claim.tier is Tier.FACT and dep.claim.evidence[0].file == "src/mini/core.py"
    assert {d.other for d in comps["mini.store"].imported_by} == {"mini.core"}
    assert "mini.core" not in {d.other for d in comps["mini.core"].imports_out}


def test_output_is_deterministic(mini_repo: Path) -> None:
    model = scan(mini_repo)
    assert analyze(model, mini_repo, default_group) == analyze(model, mini_repo, default_group)
