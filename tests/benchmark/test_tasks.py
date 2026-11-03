from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.benchmark.tasks import (
    Split,
    Task,
    TaskFileError,
    assign_split,
    load_tasks,
    select_subset,
)


def _task(i: int, repo: str = "o/r") -> dict:
    return {
        "instance_id": f"{repo.replace('/', '__')}-{i}",
        "repo": repo,
        "base_commit": "abc1234",
        "problem_statement": f"problem {i}",
        "patch": "diff --git a/x b/x\n",
        "test_patch": "diff --git a/t b/t\n",
        "FAIL_TO_PASS": json.dumps(["t::a"]),
        "PASS_TO_PASS": json.dumps([]),
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_load_tasks_parses_swebench_rows(tmp_path: Path) -> None:
    tasks = load_tasks(_write(tmp_path / "t.jsonl", [_task(1)]))
    assert tasks[0] == Task(
        task_id="o__r-1",
        repo="o/r",
        base_commit="abc1234",
        problem_statement="problem 1",
        gold_patch="diff --git a/x b/x\n",
        test_patch="diff --git a/t b/t\n",
        fail_to_pass=("t::a",),
        pass_to_pass=(),
    )


def test_load_tasks_rejects_row_missing_required_field(tmp_path: Path) -> None:
    row = _task(1)
    del row["base_commit"]
    with pytest.raises(TaskFileError, match="base_commit"):
        load_tasks(_write(tmp_path / "t.jsonl", [row]))


def test_load_tasks_rejects_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(TaskFileError, match="duplicate"):
        load_tasks(_write(tmp_path / "t.jsonl", [_task(1), _task(1)]))


def test_assign_split_is_deterministic_and_covers_all_splits() -> None:
    ids = [f"task-{i}" for i in range(600)]
    first = [assign_split(i) for i in ids]
    assert first == [assign_split(i) for i in ids]
    assert set(first) == set(Split)


def test_assign_split_proportions_are_roughly_20_20_60() -> None:
    splits = [assign_split(f"task-{i}") for i in range(3000)]
    share = {s: splits.count(s) / len(splits) for s in Split}
    assert 0.15 < share[Split.DEV] < 0.25
    assert 0.15 < share[Split.VALIDATION] < 0.25
    assert 0.55 < share[Split.HELDOUT] < 0.65


def test_select_subset_is_seeded_stable_and_split_filtered(tmp_path: Path) -> None:
    tasks = load_tasks(_write(tmp_path / "t.jsonl", [_task(i) for i in range(200)]))
    a = select_subset(tasks, Split.DEV, n=10, seed=0)
    b = select_subset(list(reversed(tasks)), Split.DEV, n=10, seed=0)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert len(a) == 10
    assert all(assign_split(t.task_id) is Split.DEV for t in a)
    assert [t.task_id for t in a] != [t.task_id for t in select_subset(tasks, Split.DEV, n=10, seed=1)]


def test_select_subset_returns_fewer_when_split_is_small(tmp_path: Path) -> None:
    tasks = load_tasks(_write(tmp_path / "t.jsonl", [_task(i) for i in range(5)]))
    assert len(select_subset(tasks, Split.DEV, n=100, seed=0)) <= 5


@pytest.mark.parametrize(
    ("field", "value"),
    [("instance_id", ".."), ("instance_id", "../x"), ("instance_id", "/abs"), ("repo", "../../x"),
     ("repo", "no-slash"), ("base_commit", "--upload-pack=x"), ("base_commit", "HEAD")],
)
def test_load_tasks_rejects_unsafe_identifiers(tmp_path: Path, field: str, value: str) -> None:
    row = _task(1)
    row[field] = value
    with pytest.raises(TaskFileError, match="unsafe|hex"):
        load_tasks(_write(tmp_path / "t.jsonl", [row]))


def test_task_view_hides_gold_and_test_patches(tmp_path: Path) -> None:
    task = load_tasks(_write(tmp_path / "t.jsonl", [_task(1)]))[0]
    view = task.view()
    assert view.problem_statement == "problem 1"
    assert not hasattr(view, "gold_patch") and not hasattr(view, "test_patch")
