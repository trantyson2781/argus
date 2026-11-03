"""The blast-radius tree (`argus.explore.blast`): folders, files, classes and functions of the production code, each marked with how much
depends on it.

Rules under test: the radius of a function is the number of distinct production functions, methods and modules that reach it through
resolved calls (a lower bound: calls Argus cannot link are not followed); test code is never counted; a container shows the largest
radius inside it; the tier is a label for that number, never a verdict; children are listed most-depended-on first; with a recording each
node says whether it ran, and a node whose file changed after the recording says so instead.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from argus.explore import blast
from argus.explore.blast import BlastNode, build_blast_tree
from argus.explore.claims import file_hash
from argus.extract import scan
from argus.record.store import ObservedFunction, Processes, Recording

FILES = {
    "src/pkg/__init__.py": "",
    "src/pkg/core.py": "def base(x):\n    return x\n\n\ndef mid(x):\n    return base(x)\n\n\ndef top(x):\n    return mid(x)\n",
    "src/pkg/api/__init__.py": "",
    "src/pkg/api/views.py": (
        "from pkg.core import top\n\n\ndef view():\n    return top(1)\n\n\nclass Page:\n    def render(self):\n        def inner():\n            return 1\n"
        "        return view()\n"
    ),
    "src/pkg/util.py": "def lonely():\n    pass\n",
    "tests/test_core.py": "from pkg.core import base\n\n\ndef test_base():\n    base(1)\n    base(2)\n",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    return tmp_path


def walk(node: BlastNode):
    yield node
    for child in node.children:
        yield from walk(child)


def find(tree, name: str) -> BlastNode:
    (found,) = [n for root in tree.roots for n in walk(root) if n.name == name or n.qualname == name]
    return found


def test_radius_counts_every_function_that_reaches_it_and_direct_counts_the_nearest(repo: Path) -> None:
    tree = build_blast_tree(scan(repo), repo)

    assert [(find(tree, n).radius, find(tree, n).direct) for n in ("base", "mid", "top", "view", "lonely")] == [(4, 1), (3, 1), (2, 1), (1, 1), (0, 0)]
    # base is reached by mid, top, view and Page.render; the test function that also calls it is not counted


def test_a_class_shows_the_largest_radius_of_its_methods_and_nested_functions_are_left_out(repo: Path) -> None:
    tree = build_blast_tree(scan(repo), repo)

    page = find(tree, "Page")
    assert [c.name for c in page.children] == ["render"] and page.kind == "class"
    assert page.radius == max(c.radius for c in page.children)
    assert "inner" not in {n.name for root in tree.roots for n in walk(root)}


def test_test_code_is_not_in_the_tree_and_never_counts(repo: Path) -> None:
    tree = build_blast_tree(scan(repo), repo)

    assert "test_core.py" not in {n.name for root in tree.roots for n in walk(root)}
    assert find(tree, "base").direct == 1  # base is called twice from a test and once from mid


def test_a_container_shows_the_largest_radius_inside_it_and_a_single_folder_chain_is_joined(repo: Path) -> None:
    tree = build_blast_tree(scan(repo), repo)

    (root,) = tree.roots
    assert (root.kind, root.name) == ("folder", "src/pkg")  # src has only pkg inside
    assert root.radius == 4 and find(tree, "core.py").radius == 4 and find(tree, "views.py").radius == find(tree, "view").radius


def test_children_come_most_depended_on_first_then_by_name(repo: Path) -> None:
    tree = build_blast_tree(scan(repo), repo)

    assert [c.name for c in find(tree, "core.py").children] == ["base", "mid", "top"]
    assert [c.name for c in find(tree, "src/pkg").children][:2] == ["core.py", "api"]  # radius 4, then 1


def test_a_symbol_knows_where_to_open_it(repo: Path) -> None:
    tree = build_blast_tree(scan(repo), repo)

    mid = find(tree, "mid")
    assert (mid.qualname, mid.path, mid.line, mid.kind) == ("pkg.core.mid", "src/pkg/core.py", 5, "function")
    assert find(tree, "render").kind == "method"


def test_tiers_label_the_number_with_a_floor_so_a_small_repository_has_no_high(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = build_blast_tree(scan(repo), repo)
    assert {find(tree, n).tier for n in ("base", "mid", "lonely")} == {"MED", "LOW"} and tree.high_at == blast.HIGH_FLOOR

    monkeypatch.setattr(blast, "HIGH_FLOOR", 3)
    tree = build_blast_tree(scan(repo), repo)
    assert [find(tree, n).tier for n in ("base", "mid", "top", "view")] == ["HIGH", "MED", "LOW", "LOW"]


def test_a_container_takes_the_tier_of_its_largest_child(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(blast, "HIGH_FLOOR", 3)
    tree = build_blast_tree(scan(repo), repo)

    assert find(tree, "src/pkg").tier == "HIGH" and find(tree, "util.py").tier == "LOW"


def test_a_repository_with_no_production_code_has_an_empty_tree(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    pass\n", encoding="utf-8")

    assert build_blast_tree(scan(tmp_path), tmp_path).roots == ()


# -- the recording ---------------------------------------------------------------------------------------------------------------------


def recording(root: Path, ran: dict[str, int]) -> Recording:
    fns = tuple(
        ObservedFunction(f"k{i}", "x.py", 1, name, f"function:{name}", None, (), calls)
        for i, (name, calls) in enumerate(ran.items())
    )
    files = tuple((rel, file_hash(root, rel)) for rel in sorted(FILES))
    return Recording("0.1.0", "3.13", ("pytest",), "2026-10-24T14:02:11Z", 1.0, 0, files, fns, (), (), (), Processes(1, 0))


def test_without_a_recording_nothing_says_ran(repo: Path) -> None:
    tree = build_blast_tree(scan(repo), repo)

    assert not tree.has_recording and all(n.ran is None and not n.stale for root in tree.roots for n in walk(root))


def test_with_a_recording_a_function_says_whether_it_started_and_a_container_counts(repo: Path) -> None:
    rec = recording(repo, {"pkg.core.base": 3, "pkg.core.mid": 0, "pkg.core.top": 1})

    tree = build_blast_tree(scan(repo), repo, rec)

    assert tree.has_recording
    assert [(find(tree, n).ran, find(tree, n).total) for n in ("base", "mid", "top")] == [(1, 1), (0, 1), (1, 1)]  # mid was only called, never started
    assert (find(tree, "core.py").ran, find(tree, "core.py").total) == (2, 3)
    assert find(tree, "src/pkg").total == 6 and find(tree, "src/pkg").ran == 2  # base, top; views, render, lonely did not run


def test_a_file_changed_after_the_recording_is_stale_down_to_its_functions_and_up_to_its_folders(repo: Path) -> None:
    rec = recording(repo, {"pkg.core.base": 3})
    (repo / "src/pkg/core.py").write_text(FILES["src/pkg/core.py"] + "# edited\n", encoding="utf-8")

    tree = build_blast_tree(scan(repo), repo, rec)

    assert find(tree, "core.py").stale and find(tree, "base").stale and find(tree, "src/pkg").stale
    assert not find(tree, "util.py").stale and not find(tree, "lonely").stale


def test_a_file_the_recording_does_not_know_is_stale(repo: Path) -> None:
    rec = recording(repo, {})
    rec = replace(rec, files=tuple(f for f in rec.files if f[0] != "src/pkg/util.py"))

    assert find(build_blast_tree(scan(repo), repo, rec), "util.py").stale
