"""Diff Explorer: pathological inputs fail loudly and cleanly, and repeated work is not repeated per function."""

from __future__ import annotations

import pytest

from tests.explore.diff_helpers import GitRepo
from tests.explore.test_diff_view import unknown_text

CHANGED_FILES = 3
FUNCTIONS_PER_FILE = 12


def _wide_repo(repo: GitRepo, bodies: str) -> None:
    def source(tag: str) -> str:
        return "".join(f"def f{j}(x):\n    y = x + {tag}\n    obj.run(y)\n    return y\n\n\n" for j in range(FUNCTIONS_PER_FILE))

    repo.write_files({"pkg/__init__.py": "", **{f"pkg/m{i}.py": source("1") for i in range(CHANGED_FILES)}})
    repo.commit("base")
    repo.write_files({f"pkg/m{i}.py": source(bodies) for i in range(CHANGED_FILES)})


def test_each_changed_file_is_parsed_a_bounded_number_of_times_not_once_per_function(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    import argus.explore.changes as changes
    import argus.explore.symbols as symbols
    from argus.explore.sides import materialize_sides
    from argus.extract import scan

    _wide_repo(git_repo, "2")
    calls: list[str] = []

    def counting(original):
        def wrapper(path, *args, **kwargs):
            calls.append(str(path))
            return original(path, *args, **kwargs)
        return wrapper

    with materialize_sides(git_repo.root, "HEAD", None) as (base, head):
        base_model, head_model = scan(base.root), scan(head.root)
        for module in (changes, symbols):
            monkeypatch.setattr(module, "parse_file", counting(module.parse_file))
        view = changes.build_diff_view(base, head, base_model, head_model, True)

    assert len(view.symbols) == CHANGED_FILES * FUNCTIONS_PER_FILE
    assert 1 <= len(calls) <= 2 * CHANGED_FILES, f"{len(calls)} parses for {CHANGED_FILES * 2} (file, side) pairs"
    assert len(set(calls)) == len(calls), "a file was parsed twice by the same side"
    assert "could not be traced" in unknown_text(view)


def test_a_body_that_cannot_be_hashed_is_unknown_not_a_crash(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    import argus.explore.changes as changes
    from tests.explore.test_diff_view import build_view

    git_repo.write_files({"shop/__init__.py": "", "shop/a.py": "def one():\n    return 1\n"})
    git_repo.commit("base")
    git_repo.write("shop/a.py", "def one():\n    return 2\n")

    def too_deep(statements):
        raise RecursionError

    monkeypatch.setattr(changes, "_digest", too_deep)

    view = build_view(git_repo)

    assert "body of shop.a.one could not be compared" in unknown_text(view)
