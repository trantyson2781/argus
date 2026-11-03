"""Run one configuration over a task list: execute, evaluate, trace."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from argus.benchmark.config import RunConfig
from argus.benchmark.evaluator import Evaluator
from argus.benchmark.tasks import Task
from argus.benchmark.traces import RunTrace, TraceWriter
from argus.benchmark.workspace import extract_patch
from argus.executor.base import Context, ExecutionResult, Executor


def count_patch_changes(patch: str) -> tuple[int, int]:
    """(files changed, lines added+removed) of a unified diff; only hunk bodies are counted."""
    files = lines = 0
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            files += 1
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line[:1] in ("+", "-"):
            lines += 1
    return files, lines


class _Attempt:
    """Everything needed to write one trace."""

    def __init__(self, task: Task, executor: Executor, config: RunConfig, attempt: int) -> None:
        self.task, self.executor, self.config, self.attempt = task, executor, config, attempt
        self.result = ExecutionResult()
        self.patch = ""
        self.seconds = 0.0
        self.reused = False

    def trace(self, success: bool, error: str | None, kind: str | None) -> RunTrace:
        files, lines = count_patch_changes(self.patch)
        r = self.result
        return RunTrace(
            task_id=self.task.task_id, agent=self.executor.name, model=self.executor.model,
            config_hash=self.config.config_hash, split=self.config.split,
            input_tokens=r.input_tokens, output_tokens=r.output_tokens, cost=r.cost, llm_calls=r.llm_calls,
            tool_calls=r.tool_calls, test_runs=None, attempts=self.attempt,
            runtime_seconds=round(self.seconds, 3), success=success, files_changed=files, lines_changed=lines,
            patch=self.patch, error=error, error_kind=kind if error else None, timed_out=r.timed_out,
            cost_unknown=r.cost_unknown, reused_patch=self.reused, timestamp=time.time(),
        )


def _evaluate(a: _Attempt, evaluator: Evaluator) -> RunTrace:
    try:
        verdict = evaluator.evaluate(a.task, a.patch)
    except Exception as exc:  # noqa: BLE001 - an evaluator crash is infrastructure, recorded and retried
        return a.trace(False, f"evaluator crashed: {type(exc).__name__}: {exc}", "evaluator")
    if verdict.error:
        return a.trace(False, verdict.error, "evaluator")
    return a.trace(verdict.resolved, None, None)


def _reuse(prior: RunTrace, a: _Attempt) -> None:
    """Carry over a previous attempt's agent output, so re-evaluating does not re-sample the model."""
    a.patch, a.seconds, a.reused = prior.patch, prior.runtime_seconds, True
    a.result = ExecutionResult(
        input_tokens=prior.input_tokens, output_tokens=prior.output_tokens, cost=prior.cost,
        llm_calls=prior.llm_calls, tool_calls=prior.tool_calls, timed_out=prior.timed_out,
        cost_unknown=prior.cost_unknown,
    )


def _run_one(task: Task, executor: Executor, evaluator: Evaluator, config: RunConfig,
             workspace_factory: Callable[[Task], Path], extract: Callable[[Path], str],
             prior: list[RunTrace]) -> RunTrace:
    a = _Attempt(task, executor, config, attempt=len(prior) + 1)
    if prior and prior[-1].error_kind == "evaluator" and prior[-1].patch:
        _reuse(prior[-1], a)
        return _evaluate(a, evaluator)
    try:
        workspace = workspace_factory(task)
    except Exception as exc:  # noqa: BLE001 - one bad checkout must not abort the run
        return a.trace(False, f"workspace: {type(exc).__name__}: {exc}", "workspace")
    started = time.monotonic()  # agent time only; checkout and network setup are excluded
    try:
        a.result = executor.execute(task.view(), Context(), workspace)
        a.patch = extract(workspace)
    except Exception as exc:  # noqa: BLE001 - recorded; Executor.execute should not raise, but must not kill the run
        a.seconds = time.monotonic() - started
        return a.trace(False, f"executor: {type(exc).__name__}: {exc}", "executor")
    a.seconds = time.monotonic() - started
    if a.result.error:
        return a.trace(False, a.result.error, "executor")
    return _evaluate(a, evaluator)


def run_benchmark(
    tasks: list[Task],
    executor: Executor,
    evaluator: Evaluator,
    writer: TraceWriter,
    config: RunConfig,
    workspace_factory: Callable[[Task], Path],
    extract: Callable[[Path], str] = extract_patch,
    on_trace: Callable[[RunTrace], None] | None = None,
) -> list[RunTrace]:
    """Run every task not already completed under this config; returns the new traces."""
    history = [t for t in writer.read() if t.config_hash == config.config_hash]
    done = {t.task_id for t in history if t.error is None}
    new: list[RunTrace] = []
    for task in tasks:
        if task.task_id in done:
            continue
        prior = [t for t in history if t.task_id == task.task_id]
        trace = _run_one(task, executor, evaluator, config, workspace_factory, extract, prior)
        writer.write(trace)
        history.append(trace)
        new.append(trace)
        if on_trace:
            on_trace(trace)
    return new
