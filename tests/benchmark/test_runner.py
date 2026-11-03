from __future__ import annotations

from pathlib import Path

from argus.benchmark.config import RunConfig
from argus.benchmark.evaluator import EvalResult
from argus.benchmark.runner import count_patch_changes, run_benchmark
from argus.benchmark.tasks import Task
from argus.benchmark.traces import TraceWriter, latest_by_task, read_traces
from argus.executor.base import Context, ExecutionResult
from tests.benchmark.test_traces import _trace

PATCH = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n-x\n+y\n+z\n"


def _task(i: int) -> Task:
    return Task(f"t{i}", "o/r", "c", f"p{i}", "", "", (), ())


class FakeExecutor:
    name = "fake"
    model = "m"

    def __init__(self, result: ExecutionResult | None = None, edit: bool = True) -> None:
        self.calls: list[str] = []
        self._result = result or ExecutionResult(input_tokens=100, output_tokens=10, cost=0.5, llm_calls=2)
        self._edit = edit

    def execute(self, task: Task, context: Context, workspace: Path) -> ExecutionResult:
        self.calls.append(task.task_id)
        if self._edit:
            (workspace / "a.py").write_text("y\nz\n")
        return self._result


class FakeEvaluator:
    def __init__(self, resolved: set[str], error: str | None = None) -> None:
        self.resolved, self.error, self.seen = resolved, error, []

    def evaluate(self, task: Task, patch: str) -> EvalResult:
        self.seen.append((task.task_id, patch))
        return EvalResult(task.task_id in self.resolved, self.error)


def _workspace_factory(root: Path, patch: str = PATCH):
    def make(task: Task) -> Path:
        ws = root / task.task_id
        ws.mkdir(parents=True, exist_ok=True)
        return ws
    return make


def _run(tmp_path, tasks, executor, evaluator, config=None, patch_of=lambda ws: PATCH):
    cfg = config or RunConfig(agent="fake", model="m", split="dev", n=len(tasks), seed=0)
    writer = TraceWriter(tmp_path / "runs.jsonl")
    return run_benchmark(tasks, executor, evaluator, writer, cfg, _workspace_factory(tmp_path / "ws"), extract=patch_of)


def test_count_patch_changes() -> None:
    assert count_patch_changes(PATCH) == (1, 3)
    assert count_patch_changes("") == (0, 0)


def test_run_records_one_trace_per_task_with_usage_and_success(tmp_path: Path) -> None:
    traces = _run(tmp_path, [_task(1), _task(2)], FakeExecutor(), FakeEvaluator({"t1"}))
    assert [(t.task_id, t.success) for t in traces] == [("t1", True), ("t2", False)]
    t = traces[0]
    assert (t.input_tokens, t.output_tokens, t.cost, t.llm_calls) == (100, 10, 0.5, 2)
    assert (t.files_changed, t.lines_changed) == (1, 3)
    assert t.agent == "fake" and t.model == "m" and t.split == "dev" and t.runtime_seconds >= 0
    assert read_traces(tmp_path / "runs.jsonl") == traces


def test_resume_skips_completed_tasks_for_same_config(tmp_path: Path) -> None:
    ex = FakeExecutor()
    _run(tmp_path, [_task(1)], ex, FakeEvaluator(set()))
    _run(tmp_path, [_task(1), _task(2)], ex, FakeEvaluator(set()))
    assert ex.calls == ["t1", "t2"]


def test_changed_config_reruns_everything(tmp_path: Path) -> None:
    ex = FakeExecutor()
    _run(tmp_path, [_task(1)], ex, FakeEvaluator(set()))
    _run(tmp_path, [_task(1)], ex, FakeEvaluator(set()), config=RunConfig("fake", "other-model", "dev", 1, 0))
    assert ex.calls == ["t1", "t1"]


def test_errored_runs_are_retried_and_latest_trace_wins(tmp_path: Path) -> None:
    ex = FakeExecutor(ExecutionResult(error="api down"))
    first = _run(tmp_path, [_task(1)], ex, FakeEvaluator(set()))
    assert first[0].error == "api down" and first[0].success is False
    _run(tmp_path, [_task(1)], FakeExecutor(), FakeEvaluator({"t1"}))
    latest = latest_by_task(read_traces(tmp_path / "runs.jsonl"))
    assert len(latest) == 1 and latest[0].success and latest[0].error is None


def test_executor_error_skips_evaluation(tmp_path: Path) -> None:
    ev = FakeEvaluator({"t1"})
    _run(tmp_path, [_task(1)], FakeExecutor(ExecutionResult(error="boom")), ev)
    assert ev.seen == []


def test_evaluator_error_is_recorded_and_not_a_success(tmp_path: Path) -> None:
    traces = _run(tmp_path, [_task(1)], FakeExecutor(), FakeEvaluator({"t1"}, error="docker down"))
    assert traces[0].success is False and traces[0].error == "docker down"


def test_workspace_failure_is_recorded_and_run_continues(tmp_path: Path) -> None:
    def failing(task: Task) -> Path:
        if task.task_id == "t1":
            raise RuntimeError("clone failed")
        (tmp_path / "ok").mkdir(exist_ok=True)
        return tmp_path / "ok"
    cfg = RunConfig("fake", "m", "dev", 2, 0)
    writer = TraceWriter(tmp_path / "runs.jsonl")
    traces = run_benchmark([_task(1), _task(2)], FakeExecutor(), FakeEvaluator({"t2"}), writer, cfg, failing,
                           extract=lambda ws: PATCH)
    assert traces[0].error and "clone failed" in traces[0].error
    assert traces[1].success is True


def test_evaluator_exception_is_recorded_and_the_run_continues(tmp_path: Path) -> None:
    class Boom:
        def evaluate(self, task, patch):
            if task.task_id == "t1":
                raise FileNotFoundError("uvx")
            return EvalResult(True, None)

    traces = _run(tmp_path, [_task(1), _task(2)], FakeExecutor(), Boom())
    assert traces[0].error_kind == "evaluator" and "uvx" in traces[0].error
    assert traces[1].success is True


def test_agent_timeout_is_evaluated_and_never_retried(tmp_path: Path) -> None:
    ex = FakeExecutor(ExecutionResult(input_tokens=500, cost=0.2, timed_out=True))
    ev = FakeEvaluator(set())
    traces = _run(tmp_path, [_task(1)], ex, ev)
    assert traces[0].timed_out and traces[0].error is None and traces[0].input_tokens == 500
    assert len(ev.seen) == 1
    _run(tmp_path, [_task(1)], ex, ev)
    assert ex.calls == ["t1"]


def test_failed_evaluation_is_retried_against_the_stored_patch_without_resampling_the_agent(tmp_path: Path) -> None:
    ex = FakeExecutor()
    first = _run(tmp_path, [_task(1)], ex, FakeEvaluator({"t1"}, error="docker down"))
    second = _run(tmp_path, [_task(1)], ex, FakeEvaluator({"t1"}))
    assert ex.calls == ["t1"]
    assert second[0].success and second[0].reused_patch and second[0].attempts == 2
    assert second[0].patch == first[0].patch and second[0].cost == first[0].cost


def test_executor_error_retry_runs_the_agent_again_and_counts_attempts(tmp_path: Path) -> None:
    _run(tmp_path, [_task(1)], FakeExecutor(ExecutionResult(error="api down")), FakeEvaluator(set()))
    second = _run(tmp_path, [_task(1)], FakeExecutor(), FakeEvaluator({"t1"}))
    assert second[0].attempts == 2 and not second[0].reused_patch and second[0].success


def test_executor_receives_a_view_without_gold_or_test_patches(tmp_path: Path) -> None:
    seen = []

    class Spy(FakeExecutor):
        def execute(self, task, context, workspace):
            seen.append(task)
            return super().execute(task, context, workspace)

    _run(tmp_path, [_task(1)], Spy(), FakeEvaluator(set()))
    assert not hasattr(seen[0], "gold_patch") and not hasattr(seen[0], "test_patch")


def test_unmeasured_telemetry_is_none_not_zero(tmp_path: Path) -> None:
    t = _run(tmp_path, [_task(1)], FakeExecutor(), FakeEvaluator(set()))[0]
    assert t.tool_calls is None and t.test_runs is None


def test_count_patch_changes_ignores_headers_but_counts_dashed_content() -> None:
    patch = "diff --git a/s.sql b/s.sql\n--- a/s.sql\n+++ b/s.sql\n@@ -1,2 +1,2 @@\n-- old comment\n+-- new comment\n ctx\n"
    assert count_patch_changes(patch) == (1, 2)


def test_run_config_hash_covers_behaviour_but_not_subset_choice() -> None:
    base = RunConfig("a", "m", "dev", 10, 0)
    assert base.config_hash == RunConfig("a", "m", "validation", 99, 7).config_hash
    for other in (RunConfig("b", "m", "dev", 10, 0), RunConfig("a", "x", "dev", 10, 0),
                  RunConfig("a", "m", "dev", 10, 0, extra=(("k", "v"),))):
        assert other.config_hash != base.config_hash


def test_completed_ids_ignores_errored_traces(tmp_path: Path) -> None:
    w = TraceWriter(tmp_path / "r.jsonl")
    w.write(_trace("ok"))
    w.write(_trace("bad", error="x"))
    assert w.completed_ids("cfg") == {"ok"}
