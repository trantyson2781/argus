"""`argus explore map --view levels`: every component of the repo in import levels, each box with its purpose and CLI entry badge.

Rules under test: one map for the whole repository (components of every subsystem together); a component sits strictly below the
ones that import it; entry points and purposes are the existing FACTs (docstring quote, command count), never invented; a
component without a docstring says so; an import cycle is shown as a cycle; output is ASCII, at most 78 columns, deterministic.
"""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from argus.cli import app
from argus.explore.ascii import MAX_WIDTH
from tests.explore.test_ascii_maps import demo_repo  # noqa: F401 - the fixture

RUNNER = CliRunner()


def _levels(root: Path, *extra: str) -> str:
    result = RUNNER.invoke(app, ["explore", "map", str(root), "--view", "levels", *extra])
    assert result.exit_code == 0, result.output
    return result.output


def _layers(out: str) -> dict[str, int]:
    layers: dict[str, int] = {}
    current = -1
    for line in out.splitlines():
        marker = re.match(r"L(\d+) ", line)
        current = int(marker.group(1)) if marker else current
        for name in re.findall(r"\+- (\S+)", line):
            layers[name] = current
    return layers


def test_components_of_every_subsystem_share_one_map_with_importers_above(demo_repo: Path) -> None:  # noqa: F811
    layers = _layers(_levels(demo_repo))

    assert {"alpha.cli", "alpha.report", "beta.core", "beta.store", "beta.util"} <= set(layers)
    assert layers["alpha.cli"] < layers["beta.core"] < layers["beta.store"]


def test_the_header_says_levels_are_import_depth_and_not_execution_order(demo_repo: Path) -> None:  # noqa: F811
    out = _levels(demo_repo)

    assert out.startswith("LEVEL MAP") and "NOT execution order" in out and "ENTRY" in out
    assert "..." not in "\n".join(out.splitlines()[:4])  # the legend is never cut off


def test_a_component_with_a_command_gets_an_entry_badge(demo_repo: Path) -> None:  # noqa: F811
    out = _levels(demo_repo)
    box = out[out.index("+- alpha.cli"):]
    box = box[: box.index("+---", 5)]

    assert re.search(r"ENTRY +1 command", box) and "beta.core" in box  # the arrow list is still there
    assert out.count("ENTRY 1") + out.count("ENTRY  1") <= 1  # only the component that defines the command


def test_a_purpose_is_a_docstring_quote_and_a_missing_one_is_said_not_invented(demo_repo: Path) -> None:  # noqa: F811
    out = _levels(demo_repo)
    store = out[out.index("+- beta.store"):][:400]
    util = out[out.index("+- beta.util"):][:400]

    assert "Storage." in store
    assert "no docstring" in util


def test_an_import_cycle_is_a_cycle_not_an_order(demo_repo: Path) -> None:  # noqa: F811
    out = _levels(demo_repo)

    assert out.count("CYCLE with") == 2


def test_levels_is_ascii_within_78_columns_and_deterministic(demo_repo: Path) -> None:  # noqa: F811
    first, second = _levels(demo_repo), _levels(demo_repo)

    assert first == second and first.isascii() and max(len(line) for line in first.splitlines()) <= MAX_WIDTH


def test_open_does_not_apply_to_levels_and_says_so(demo_repo: Path) -> None:  # noqa: F811
    result = RUNNER.invoke(app, ["explore", "map", str(demo_repo), "--view", "levels", "--open", "beta"])

    assert result.exit_code == 2 and "--open" in result.output
