import ast
"""End-to-end on the real Argus repo: the acceptance scenario."""

from pathlib import Path

from typer.testing import CliRunner

from argus.cli import app
from argus.explore.claims import Tier
from argus.explore.components import analyze, default_group
from argus.explore.entrypoints import find_entrypoints
from argus.explore.reach import call_tree
from argus.extract import scan

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


def _run_benchmark_call_site() -> str:
    """`file:line` of the call `bench run` makes, read from the source so edits above it don't break tests."""
    path = ROOT / "src/argus/benchmark/cli.py"
    line = next(n for n, text in enumerate(path.read_text().splitlines(), 1) if text.strip() == "run_benchmark(")
    return f"src/argus/benchmark/cli.py:{line}"


def test_entrypoints_on_argus_find_bench_run() -> None:
    result = runner.invoke(app, ["explore", "entrypoints", str(ROOT)])
    assert result.exit_code == 0, result.output
    assert "argus bench run" in result.output and "benchmark/cli.py" in result.output
    assert "argus.cli:app" in result.output


def test_entrypoints_command_trace_reaches_runner() -> None:
    result = runner.invoke(app, ["explore", "entrypoints", str(ROOT), "--command", "bench run"])
    assert result.exit_code == 0, result.output
    hop = [l for l in result.output.splitlines() if "argus.benchmark.runner.run_benchmark " in l]
    assert hop and "[INFERENCE]" in hop[0] and _run_benchmark_call_site() in hop[0]


def test_map_on_argus_lists_components_and_unknowns() -> None:
    result = runner.invoke(app, ["explore", "map", str(ROOT), "--view", "list"])
    assert result.exit_code == 0, result.output
    for name in ("argus.benchmark", "argus.executor", "argus.extract", "argus.model"):
        assert name in result.output
    assert "[UNKNOWN]" in result.output and "tests" not in result.output.split("Components")[1].split("Unknown")[0]


def test_component_accepts_group_or_module_and_shows_how_it_is_reached() -> None:
    result = runner.invoke(app, ["explore", "component", "argus.benchmark.runner", str(ROOT)])
    assert result.exit_code == 0, result.output
    reached = result.output.split("Reached from CLI entry points")[1]
    assert "argus bench run" in reached
    hop = [l for l in reached.splitlines() if "argus.benchmark.runner.run_benchmark" in l]
    assert hop and _run_benchmark_call_site() in hop[0]
    assert "src/argus/benchmark/runner.py" in result.output


def test_component_unknown_name_lists_candidates() -> None:
    result = runner.invoke(app, ["explore", "component", "nonexistent", str(ROOT)])
    assert result.exit_code == 2 and "argus.benchmark" in result.output


def test_component_ambiguous_suffix_is_refused() -> None:
    result = runner.invoke(app, ["explore", "component", "cli", str(ROOT)])
    assert result.exit_code == 2


def _line(ev) -> str:
    return (ROOT / ev.file).read_text(encoding="utf-8").splitlines()[ev.line - 1]


def test_argus_entrypoint_evidence_lines_contain_the_cited_construct() -> None:
    model = scan(ROOT)
    eps = find_entrypoints(ROOT, model)
    assert eps.scripts and eps.commands and eps.main_guards is not None
    for script in eps.scripts:
        assert script.name in _line(script.claim.evidence[0])
    for command in eps.commands:
        for ev in command.claim.evidence:
            assert ".command(" in _line(ev) or "add_typer(" in _line(ev), (command.display, ev)
    for guard in eps.main_guards:
        assert "__main__" in _line(guard.evidence[0])


def test_argus_claims_have_real_hashes_and_lines_in_range() -> None:
    model = scan(ROOT)
    claims = list(find_entrypoints(ROOT, model).all_claims())
    for comp in analyze(model, ROOT, default_group):
        claims.append(comp.purpose)
        claims.extend(d.claim for d in (*comp.imports_out, *comp.imported_by))
    assert claims
    for claim in claims:
        if claim.tier is Tier.UNKNOWN and not claim.evidence:
            continue
        for ev in claim.evidence:
            assert len(ev.source_hash) == 16
            assert 1 <= ev.line <= len((ROOT / ev.file).read_text(encoding="utf-8").splitlines()), (claim.text, ev)


def test_call_steps_cite_the_line_that_makes_the_call() -> None:
    model = scan(ROOT)
    command = next(c for c in find_entrypoints(ROOT, model).commands if c.display == "argus bench run")
    steps = call_tree(model, ROOT, command.function_id, max_depth=2)
    assert steps
    for step in steps:
        assert step.callee.rsplit(".", 1)[-1] in _line(step.claim.evidence[0]), step


def _hashes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_explore_writes_nothing(mini_repo: Path) -> None:
    before = _hashes(mini_repo)
    root = str(mini_repo)
    for argv in (
        ["map", root],
        ["entrypoints", root],
        ["entrypoints", root, "--command", "hello-world"],
        ["component", "mini.core", root],
    ):
        assert runner.invoke(app, ["explore", *argv]).exit_code == 0, argv
    assert _hashes(mini_repo) == before


def test_explorer_does_not_import_the_frozen_phase0_packages() -> None:
    for source in (ROOT / "src/argus/explore").glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.startswith(("argus.benchmark", "argus.executor")) for n in names), source
