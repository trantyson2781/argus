from __future__ import annotations

import json
import subprocess
from pathlib import Path

from argus.benchmark.evaluator import EvalResult, SweBenchEvaluator
from argus.benchmark.tasks import Task

TASK = Task("o__r-1", "o/r", "abc1234", "p", "", "", ("t::a",), ())


def _runner(resolved: list[str] | None, seen: list, returncode: int = 0, stderr: str = "", extra: dict | None = None, raw: str | None = None):
    def run(cmd, **kwargs):
        seen.append(cmd)
        if resolved is not None:
            report_dir = Path(cmd[cmd.index("--report_dir") + 1])
            run_id = cmd[cmd.index("--run_id") + 1]
            body = raw if raw is not None else json.dumps({"resolved_ids": resolved, "unresolved_ids": [] if resolved else ["o__r-1"], **(extra or {})})
            (report_dir / f"argus.{run_id}.json").write_text(body)
        return subprocess.CompletedProcess(cmd, returncode, "", stderr)
    return run


def _evaluator(tmp_path: Path, runner) -> SweBenchEvaluator:
    return SweBenchEvaluator(tmp_path / "tasks.jsonl", tmp_path / "eval", command=("harness",), runner=runner)


def test_empty_patch_is_unresolved_without_running_the_harness(tmp_path: Path) -> None:
    seen: list = []
    assert _evaluator(tmp_path, _runner([], seen)).evaluate(TASK, "") == EvalResult(False, None)
    assert seen == []


def test_resolved_when_instance_in_report(tmp_path: Path) -> None:
    seen: list = []
    result = _evaluator(tmp_path, _runner(["o__r-1"], seen)).evaluate(TASK, "diff --git a/x b/x\n")
    assert result == EvalResult(True, None)
    cmd = seen[0]
    assert cmd[0] == "harness" and cmd[cmd.index("--instance_ids") + 1] == "o__r-1"
    assert cmd[cmd.index("--dataset_name") + 1] == str(tmp_path / "tasks.jsonl")


def test_unresolved_when_harness_lists_instance_as_unresolved(tmp_path: Path) -> None:
    assert _evaluator(tmp_path, _runner([], [])).evaluate(TASK, "diff\n") == EvalResult(False, None)


def test_harness_infra_failure_is_an_error_even_with_exit_zero(tmp_path: Path) -> None:
    runner = _runner([], [], extra={"unresolved_ids": [], "infra_failure_ids": ["o__r-1"], "failure_reasons": {"o__r-1": "image build failed"}})
    result = _evaluator(tmp_path, runner).evaluate(TASK, "diff\n")
    assert result.resolved is False and "image build failed" in (result.error or "")


def test_patch_that_does_not_apply_is_a_model_failure(tmp_path: Path) -> None:
    runner = _runner([], [], extra={"unresolved_ids": [], "error_ids": ["o__r-1"]})
    assert _evaluator(tmp_path, runner).evaluate(TASK, "diff\n") == EvalResult(False, None)


def test_instance_in_no_list_is_an_error_not_a_failure(tmp_path: Path) -> None:
    runner = _runner([], [], extra={"unresolved_ids": []})
    assert "no harness result list" in (_evaluator(tmp_path, runner).evaluate(TASK, "diff\n").error or "")


def test_corrupt_report_is_an_error_not_an_exception(tmp_path: Path) -> None:
    result = _evaluator(tmp_path, _runner([], [], raw="{truncated")).evaluate(TASK, "diff\n")
    assert result.error and not result.resolved


def test_missing_harness_binary_is_an_error_not_an_exception(tmp_path: Path) -> None:
    def run(cmd, **kwargs):
        raise FileNotFoundError("uvx")
    result = _evaluator(tmp_path, run).evaluate(TASK, "diff\n")
    assert result.error and "uvx" in result.error


def test_predictions_file_has_swebench_fields(tmp_path: Path) -> None:
    seen: list = []
    _evaluator(tmp_path, _runner([], seen)).evaluate(TASK, "PATCH")
    preds = Path(seen[0][seen[0].index("--predictions_path") + 1])
    row = json.loads(preds.read_text().splitlines()[0])
    assert row == {"instance_id": "o__r-1", "model_name_or_path": "argus", "model_patch": "PATCH"}


def test_harness_failure_is_an_error_not_a_silent_fail(tmp_path: Path) -> None:
    result = _evaluator(tmp_path, _runner(None, [], returncode=1, stderr="docker down")).evaluate(TASK, "diff\n")
    assert result.resolved is False and "docker down" in (result.error or "")


def test_missing_report_is_an_error(tmp_path: Path) -> None:
    result = _evaluator(tmp_path, _runner(None, [])).evaluate(TASK, "diff\n")
    assert result.error and "report" in result.error


def test_test_timeout_is_passed_to_the_harness_explicitly(tmp_path: Path) -> None:
    seen: list = []
    ev = SweBenchEvaluator(tmp_path / "t.jsonl", tmp_path / "eval", command=("h",), test_timeout_seconds=300,
                           runner=_runner([], seen))
    ev.evaluate(TASK, "diff\n")
    assert seen[0][seen[0].index("--timeout") + 1] == "300"
