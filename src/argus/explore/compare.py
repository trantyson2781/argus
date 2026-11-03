"""Compare two states of a git repository: the one place that turns refs into a `DiffView` (used by `explore diff` and the TUI)."""

from __future__ import annotations

from pathlib import Path

from argus.explore.changes import DiffView, build_diff_view
from argus.explore.sides import materialize_sides
from argus.extract import scan
from argus.extract.resolve_calls import resolve_v2_enabled


def compare_refs(path: Path, base: str, head: str | None = None, include_tests: bool = False, resolve_v2: bool | None = None) -> DiffView:
    """`base` against `head` (None: the working tree). Raises `DiffError` for a bad ref or a directory outside git."""
    resolve_v2 = resolve_v2_enabled(resolve_v2)
    with materialize_sides(path, base, head) as (base_side, head_side):
        return build_diff_view(
            base_side, head_side, scan(base_side.root, resolve_v2=resolve_v2), scan(head_side.root, resolve_v2=resolve_v2),
            resolve_v2, include_tests,
        )
