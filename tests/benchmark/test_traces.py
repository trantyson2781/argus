from __future__ import annotations

from pathlib import Path

import pytest

from argus.benchmark.traces import RunTrace, TraceError, TraceWriter, read_traces


def _trace(task_id: str = "t1", success: bool = True, **over) -> RunTrace:
    base = dict(
        task_id=task_id,
        agent="aider-vanilla",
        model="m",
        config_hash="cfg",
        split="dev",
        input_tokens=1000,
        output_tokens=200,
        cost=0.01,
        llm_calls=2,
        tool_calls=None,
        test_runs=None,
        attempts=1,
        runtime_seconds=12.5,
        success=success,
        files_changed=1,
        lines_changed=7,
        patch="diff",
        error=None,
    )
    base.update(over)
    return RunTrace(**base)


def test_writer_appends_one_json_line_per_run_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    writer = TraceWriter(path)
    writer.write(_trace("a"))
    writer.write(_trace("b", success=False))
    assert len(path.read_text().splitlines()) == 2
    assert read_traces(path) == [_trace("a"), _trace("b", success=False)]


def test_total_tokens_is_derived() -> None:
    assert _trace().total_tokens == 1200


def test_completed_ids_enable_resume(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path / "runs.jsonl")
    writer.write(_trace("a"))
    assert writer.completed_ids("cfg") == {"a"}
    assert writer.completed_ids("other-config") == set()


def test_read_traces_rejects_corrupt_line_in_the_middle(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    TraceWriter(path).write(_trace("a"))
    with path.open("a") as handle:
        handle.write("not json\n")
    TraceWriter(path).write(_trace("b"))
    with pytest.raises(TraceError, match=":2"):
        read_traces(path)


def test_read_traces_tolerates_a_truncated_final_line(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    TraceWriter(path).write(_trace("a"))
    with path.open("a") as handle:
        handle.write('{"task_id": "b", "agent"')
    assert [t.task_id for t in read_traces(path)] == ["a"]


def test_read_traces_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_traces(tmp_path / "none.jsonl") == []
