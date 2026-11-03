"""Standardized evaluation via the SWE-bench Docker harness."""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from argus.benchmark.tasks import Task
from argus.proc import run_in_group

SWEBENCH_VERSION = "5.0.2"
DEFAULT_HARNESS_COMMAND = (
    "uvx", "--from", f"swebench=={SWEBENCH_VERSION}", "python", "-m", "swebench.harness.run_evaluation",
)
HARNESS_TIMEOUT_SECONDS = 3600
TEST_TIMEOUT_SECONDS = 1800  # per-instance test run limit inside the harness (passed explicitly)
_MODEL_NAME = "argus"
# Harness outcomes that reflect the patch, not the machine.
_PATCH_FAILURE_LISTS = ("unresolved_ids", "empty_patch_ids", "error_ids")
# Harness outcomes that say nothing about the patch; the evaluation must be retried.
_INFRA_LISTS = ("infra_failure_ids", "incomplete_ids", "ambiguous_failure_ids")


@dataclass(frozen=True, slots=True)
class EvalResult:
    resolved: bool
    error: str | None


class Evaluator(Protocol):
    def evaluate(self, task: Task, patch: str) -> EvalResult: ...


def interpret_report(report: object, task_id: str) -> EvalResult:
    """Map a harness summary to a verdict. Anything unrecognised is an error, never a failure."""
    if not isinstance(report, dict):
        return EvalResult(False, "harness report is not a JSON object")
    if task_id in report.get("resolved_ids", []):
        return EvalResult(True, None)
    for key in _INFRA_LISTS:
        if task_id in report.get(key, []):
            reason = report.get("failure_reasons", {}).get(task_id, "")
            return EvalResult(False, f"harness infrastructure problem ({key}) {reason}".strip())
    for key in _PATCH_FAILURE_LISTS:
        if task_id in report.get(key, []):
            return EvalResult(False, None)
    return EvalResult(False, f"instance {task_id} appears in no harness result list")


class SweBenchEvaluator:
    """Runs one prediction through `swebench.harness.run_evaluation`.

    `dataset_path` must be the JSONL the tasks were loaded from (SWE-bench row format),
    so the harness reads the same rows Argus saw. Infrastructure problems are returned
    as `error`, never as an unresolved task, so they cannot masquerade as model failures.
    """

    def __init__(
        self,
        dataset_path: Path,
        work_dir: Path,
        command: tuple[str, ...] = DEFAULT_HARNESS_COMMAND,
        max_workers: int = 1,
        timeout_seconds: int = HARNESS_TIMEOUT_SECONDS,
        test_timeout_seconds: int = TEST_TIMEOUT_SECONDS,
        runner: Callable[..., subprocess.CompletedProcess[str]] = run_in_group,
    ) -> None:
        self._dataset = dataset_path
        self._work_dir = work_dir
        self._command = command
        self._max_workers = max_workers
        self._timeout = timeout_seconds
        self._test_timeout = test_timeout_seconds
        self._runner = runner

    def evaluate(self, task: Task, patch: str) -> EvalResult:
        if not patch.strip():
            return EvalResult(False, None)
        try:
            return self._evaluate(task, patch)
        except (OSError, ValueError) as exc:
            return EvalResult(False, f"evaluator failed: {type(exc).__name__}: {exc}")

    def _evaluate(self, task: Task, patch: str) -> EvalResult:
        run_id = f"{task.task_id}-{uuid.uuid4().hex[:8]}"
        run_dir = self._work_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions = run_dir / "predictions.jsonl"
        predictions.write_text(
            json.dumps({"instance_id": task.task_id, "model_name_or_path": _MODEL_NAME, "model_patch": patch}) + "\n",
            encoding="utf-8",
        )
        cmd = [*self._command, "--dataset_name", str(self._dataset), "--predictions_path", str(predictions),
               "--instance_ids", task.task_id, "--max_workers", str(self._max_workers),
               "--timeout", str(self._test_timeout),
               "--run_id", run_id, "--report_dir", str(run_dir)]
        try:
            done = self._runner(cmd, cwd=run_dir, timeout=self._timeout)
        except subprocess.TimeoutExpired:
            return EvalResult(False, f"harness timed out after {self._timeout}s")
        if done.returncode != 0:
            return EvalResult(False, f"harness failed (exit {done.returncode}): {done.stderr.strip()[-1500:]}")
        report = run_dir / f"{_MODEL_NAME}.{run_id}.json"
        if not report.exists():
            return EvalResult(False, f"harness wrote no report at {report.name}")
        return interpret_report(json.loads(report.read_text(encoding="utf-8")), task.task_id)
