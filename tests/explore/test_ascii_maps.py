"""ASCII system and dependency maps: golden output, width, ASCII-only, evidence, determinism, read-only."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus.cli import app
from argus.explore.ascii import MAX_WIDTH, tag
from argus.explore.claims import Tier

runner = CliRunner()
GOLDEN = Path(__file__).parent / "golden"
ROOT = Path(__file__).resolve().parents[2]

TWO_SUBSYSTEMS: dict[str, str] = {
    "pyproject.toml": '[project]\nname = "demo"\nversion = "0"\n\n[project.scripts]\nalpha = "alpha.cli:app"\n',
    "alpha/__init__.py": '"""Alpha: the user-facing application."""\n',
    "alpha/cli.py": "import typer\n\nfrom beta.core import run\n\napp = typer.Typer()\n\n\n@app.command()\ndef go():\n    run()\n",
    "alpha/report.py": "from beta.core import run\n\n\ndef show():\n    from beta.util import fmt\n    run()\n    fmt()\n",
    "beta/__init__.py": '"""Beta — shared services: core logic and storage ☃."""\n',
    "beta/core.py": '"""Core logic that spans\ntwo physical lines.\n\nSecond paragraph is not shown."""\nfrom beta.store.db import save\n\n\ndef run():\n    save()\n',
    "beta/store/__init__.py": '"""Storage."""\n',
    "beta/store/db.py": "from beta.store import a\n\n\ndef save():\n    a.x()\n",
    "beta/store/a.py": "def x():\n    pass\n",
    "beta/store/b.py": "def x():\n    pass\n",
    "beta/store/c.py": "def x():\n    pass\n",
    "beta/store/d.py": "def x():\n    pass\n",
    "beta/store/e.py": "def x():\n    pass\n",
    "beta/util.py": "def fmt():\n    pass\n",
    "beta/x.py": "from beta.y import g\n\n\ndef f():\n    g()\n",
    "beta/y.py": "from beta.x import f\n\n\ndef g():\n    f()\n",
}

CASES = {
    "system": ["--view", "system"],
    "system_open_beta": ["--view", "system", "--open", "beta"],
    "system_open_store": ["--view", "system", "--open", "beta.store"],
    "deps": ["--view", "deps"],
    "deps_open_beta": ["--view", "deps", "--open", "beta"],
    "deps_open_store": ["--view", "deps", "--open", "beta.store"],
    "deps_refs": ["--view", "deps", "--open", "beta", "--refs"],
    "system_refs": ["--view", "system", "--refs"],
}


@pytest.fixture
def demo_repo(tmp_path: Path) -> Path:
    tmp_path = tmp_path / "demo"  # fixed name: it appears in the map header
    for rel, content in TWO_SUBSYSTEMS.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def _run(root: Path, *args: str) -> str:
    result = runner.invoke(app, ["explore", "map", str(root), *args])
    assert result.exit_code == 0, result.output
    return result.output


def _normalise(text: str, root: Path) -> str:
    return text.replace(str(root), "<ROOT>")


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_output(name: str, demo_repo: Path) -> None:
    actual = _run(demo_repo, *CASES[name])
    golden = GOLDEN / f"{name}.txt"
    if os.environ.get("ARGUS_UPDATE_GOLDEN"):
        golden.write_text(actual, encoding="utf-8")
        pytest.skip("golden file regenerated; review the diff before committing")
    assert actual == golden.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(CASES))
def test_width_ascii_and_determinism(name: str, demo_repo: Path) -> None:
    first = _run(demo_repo, *CASES[name])
    assert first == _run(demo_repo, *CASES[name])
    assert first.isascii()
    assert max(len(line) for line in first.splitlines()) <= MAX_WIDTH


@pytest.mark.parametrize("view", ["system", "deps"])
def test_argus_itself_fits_78_columns_in_every_mode(view: str) -> None:
    for extra in ([], ["--refs"], ["--open", "argus"], ["--open", "argus.benchmark", "--refs"]):
        out = _run(ROOT, "--view", view, *extra)
        assert out.isascii() and max(len(line) for line in out.splitlines()) <= MAX_WIDTH, (view, extra)


def test_system_map_separates_containment_from_imports(demo_repo: Path) -> None:
    out = _run(demo_repo, "--view", "system")
    assert "CONTAINS" in out and "IMPORTS BETWEEN SUBSYSTEMS" in out
    assert "alpha ---> beta" in out and "beta ---> alpha" not in out
    assert "===>" not in out  # no call arrows in an import map
    assert "[+] store (7 modules)" in out  # collapsed above the module threshold


def test_purpose_is_quoted_docstring_or_unknown_never_invented(demo_repo: Path) -> None:
    out = _run(demo_repo, "--view", "system")
    assert "Alpha: the user-facing application." in out
    assert "Beta - shared services: core logic and storage" in out  # non-ASCII punctuation folded
    expanded = _run(demo_repo, "--view", "system", "--open", "beta")
    assert re.search(r"util .*no docstring.*\[\?\]", expanded)


def test_open_expands_a_collapsed_component_progressively(demo_repo: Path) -> None:
    assert "[-] store" in _run(demo_repo, "--view", "system", "--open", "beta.store")
    assert "[+] store" in _run(demo_repo, "--view", "system", "--open", "alpha")


def test_deps_layers_point_downward_and_state_they_are_not_execution_order(demo_repo: Path) -> None:
    out = _run(demo_repo, "--view", "deps", "--open", "beta")
    assert "NOT execution order" in out
    layers: dict[str, int] = {}
    current = -1
    for line in out.splitlines():
        marker = re.match(r"L(\d+) ", line)
        current = int(marker.group(1)) if marker else current
        for name in re.findall(r"\+- (\S+)", line):
            layers[name] = current
    assert layers["core"] < layers["store"]  # core imports store


def test_import_cycle_is_shown_as_a_cycle_not_a_layer_order(demo_repo: Path) -> None:
    out = _run(demo_repo, "--view", "deps", "--open", "beta")
    assert out.count("CYCLE with") == 2
    assert "CYCLE with: y" in out and "CYCLE with: x" in out


def test_default_deps_is_subsystem_level_and_hints_how_to_expand(demo_repo: Path) -> None:
    out = _run(demo_repo, "--view", "deps")
    assert "alpha" in out and "beta" in out and "--open" in out


def test_refs_point_at_lines_that_contain_the_import(demo_repo: Path) -> None:
    out = _run(demo_repo, "--view", "deps", "--open", "beta", "--refs")
    refs = re.findall(r"import at (\S+\.py):(\d+)", out)
    assert refs
    for file, line in refs:
        text = (demo_repo / file).read_text(encoding="utf-8").splitlines()[int(line) - 1]
        assert "import" in text, (file, line, text)


def test_arrow_counts_match_component_analysis(demo_repo: Path) -> None:
    from argus.explore.components import analyze, default_group
    from argus.extract import scan

    comps = analyze(scan(demo_repo), demo_repo, default_group)
    into_core = sum(d.count for c in comps if c.name.startswith("alpha") for d in c.imports_out if d.other == "beta.core")
    assert into_core == 2
    assert f"alpha ---> beta.core  {into_core} import edges" in _run(demo_repo, "--view", "deps", "--open", "beta")
    subsystem_total = sum(d.count for c in comps if c.name.startswith("alpha") for d in c.imports_out if d.other.startswith("beta"))
    assert f"alpha ---> beta  {subsystem_total} import edges" in _run(demo_repo, "--view", "system")


def test_unknown_open_name_lists_candidates(demo_repo: Path) -> None:
    result = runner.invoke(app, ["explore", "map", str(demo_repo), "--open", "nope"])
    assert result.exit_code == 2 and "beta.store" in result.output


def test_single_module_component_cannot_be_expanded_in_deps(demo_repo: Path) -> None:
    result = runner.invoke(app, ["explore", "map", str(demo_repo), "--view", "deps", "--open", "beta.core"])
    assert result.exit_code == 2 and "1 module" in result.output


def test_list_view_keeps_the_ranked_list(demo_repo: Path) -> None:
    assert "most-imported first" in _run(demo_repo, "--view", "list")


def test_maps_write_nothing(demo_repo: Path) -> None:
    def snapshot() -> dict[str, bytes]:
        return {str(p.relative_to(demo_repo)): p.read_bytes() for p in sorted(demo_repo.rglob("*")) if p.is_file()}

    before = snapshot()
    for name in CASES:
        _run(demo_repo, *CASES[name])
    assert snapshot() == before


def test_tier_tags_preserve_classification() -> None:
    assert tag(Tier.FACT) == "" and tag(Tier.INFERENCE) == "[I]" and tag(Tier.UNKNOWN) == "[?]"


def test_docstring_purpose_uses_the_whole_first_paragraph_not_one_physical_line(demo_repo: Path) -> None:
    out = _run(demo_repo, "--view", "system", "--open", "beta")
    assert "Core logic that spans two physical lines." in out.replace("\n      ", " ")
    assert "Second paragraph" not in out


def test_long_purpose_is_visibly_shortened_not_presented_as_complete() -> None:
    from argus.explore.ascii import MAX_PURPOSE_LINES, _purpose_lines
    from argus.explore.claims import Claim, Evidence

    long = Claim('docstring: "' + "word " * 80 + '"', Tier.FACT, (Evidence("a.py", 1, "h"),))
    lines = _purpose_lines(long, refs=False, width=60)
    assert len(lines) == MAX_PURPOSE_LINES and lines[-1].endswith('..."')


def test_repo_without_production_modules_is_a_clean_error(tmp_path: Path) -> None:
    (tmp_path / "test_a.py").write_text("def test_x():\n    pass\n")
    result = runner.invoke(app, ["explore", "map", str(tmp_path), "--view", "deps"])
    assert result.exit_code == 2 and "no production modules" in result.output


def test_package_init_is_named_init_not_root(demo_repo: Path) -> None:
    out = _run(demo_repo, "--view", "deps", "--open", "beta")
    assert "__init__" in out and "(root)" not in out


def test_unknown_tier_arrow_cites_a_site_as_weak_as_the_claim(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("import b\n" * 6)
    from argus.explore.components import _as_deps
    from argus.model.schema import Confidence, Edge, EdgeKind, Provenance

    def edge(line: int, conf: Confidence) -> Edge:
        return Edge(f"e{line}", EdgeKind.IMPORTS, "a", "b", Provenance("m.py", line, "t"), conf)

    deps = _as_deps(tmp_path, {"b": [edge(1, Confidence.EXTRACTED), edge(5, Confidence.AMBIGUOUS)]}, "imports")
    assert deps[0].claim.tier is Tier.UNKNOWN and deps[0].claim.evidence[0].line == 5


def test_invalid_module_names_never_produce_an_empty_subsystem() -> None:
    from argus.explore.diagram import top_group

    assert top_group(".runtime-deps.certifi.core") == "runtime-deps"
    assert top_group("app.core") == "app"
    assert top_group("") != "" and top_group("...") != ""
