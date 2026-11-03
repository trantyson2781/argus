from pathlib import Path

from argus.explore.claims import Tier
from argus.explore.reach import call_tree
from argus.extract import scan


def test_call_tree_lists_resolved_callees_with_call_site_evidence(mini_repo: Path) -> None:
    steps = call_tree(scan(mini_repo), mini_repo, "function:mini.cli.hello_world", max_depth=2)
    assert [(s.depth, s.callee_id) for s in steps] == [
        (1, "function:mini.core.work"),
        (2, "function:mini.core.helper"),
        (2, "function:mini.store.db.save"),
    ]
    assert steps[0].claim.tier is Tier.INFERENCE and steps[0].claim.evidence[0].line == 14
    assert steps[2].group == "mini.store"


def test_call_tree_depth_limit_and_cycle_safety(mini_repo: Path) -> None:
    assert [s.depth for s in call_tree(scan(mini_repo), mini_repo, "function:mini.cli.hello_world", max_depth=1)] == [1]
