"""`argus explore sample`: the two Phase 0 edge-quality gates, measured on any repository."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from argus.cli import app

runner = CliRunner()


def _sample(path: Path, *options: str):
    return runner.invoke(app, ["explore", "sample", str(path), *options])


def _rows(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.lstrip()[:1].isdigit()]


def test_a_healthy_repo_passes_the_ambiguous_gate_and_lists_edges_to_check_by_hand(sample_repo: Path) -> None:
    result = _sample(sample_repo, "--count", "5")

    assert result.exit_code == 0, result.output
    assert "AMBIGUOUS rate" in result.output and "PASS" in result.output
    assert "by hand" in result.output  # precision cannot be computed automatically
    assert 0 < len(_rows(result.output)) <= 5


def test_the_same_seed_draws_the_same_edges_and_another_seed_may_not(sample_repo: Path) -> None:
    first, second = _sample(sample_repo, "--count", "5", "--seed", "3"), _sample(sample_repo, "--count", "5", "--seed", "3")

    assert first.output == second.output


def test_a_repo_with_mostly_ambiguous_edges_fails_the_gate_with_exit_1(tmp_path: Path) -> None:
    (tmp_path / "shapes.py").write_text("\n\n".join(f"class C{i}(Missing{i}): ..." for i in range(6)) + "\n", encoding="utf-8")

    result = _sample(tmp_path)

    assert result.exit_code == 1 and "FAIL" in result.output


def test_a_path_that_is_not_a_directory_is_an_error_not_a_traceback(tmp_path: Path) -> None:
    result = _sample(tmp_path / "missing")

    assert result.exit_code == 2 and "Not a directory" in result.output and "Traceback" not in result.output


def test_a_repo_with_no_structural_edges_says_so(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("X = 1\n", encoding="utf-8")

    result = _sample(tmp_path)

    assert result.exit_code == 0 and "no structural edges" in result.output
