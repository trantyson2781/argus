"""Benchmark tasks (SWE-bench row format) and dataset-split discipline.

Splits are derived from a hash of the task id, never from file order or a mutable
list, so a task can never silently migrate from development to held-out
(plan §18). Held-out tasks must only be used for major evaluations.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class TaskFileError(ValueError):
    """The task file is malformed."""


class Split(StrEnum):
    DEV = "dev"
    VALIDATION = "validation"
    HELDOUT = "heldout"


DEV_SHARE = 20
VALIDATION_SHARE = 20
_BUCKETS = 100


@dataclass(frozen=True, slots=True)
class TaskView:
    """What an executor may see. Deliberately excludes the gold patch and the tests."""

    task_id: str
    repo: str
    base_commit: str
    problem_statement: str


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    repo: str
    base_commit: str
    problem_statement: str
    gold_patch: str
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]

    def view(self) -> TaskView:
        return TaskView(self.task_id, self.repo, self.base_commit, self.problem_statement)


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_REQUIRED = ("instance_id", "repo", "base_commit", "problem_statement", "patch", "test_patch")


def _as_list(value: object, field: str, where: str) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TaskFileError(f"{where}: {field} is not valid JSON") from exc
    if not isinstance(value, list):
        raise TaskFileError(f"{where}: {field} must be a list")
    return tuple(str(v) for v in value)


def _parse_row(row: dict, where: str) -> Task:
    for key in _REQUIRED:
        if not isinstance(row.get(key), str):
            raise TaskFileError(f"{where}: missing or non-string field {key!r}")
    if not _TASK_ID_RE.match(row["instance_id"]) or ".." in row["instance_id"]:
        raise TaskFileError(f"{where}: unsafe instance_id {row['instance_id']!r}")
    if not _REPO_RE.match(row["repo"]) or ".." in row["repo"]:
        raise TaskFileError(f"{where}: unsafe repo {row['repo']!r}")
    if not _COMMIT_RE.match(row["base_commit"]):
        raise TaskFileError(f"{where}: base_commit is not a hex sha: {row['base_commit']!r}")
    return Task(
        task_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
        gold_patch=row["patch"],
        test_patch=row["test_patch"],
        fail_to_pass=_as_list(row.get("FAIL_TO_PASS", []), "FAIL_TO_PASS", where),
        pass_to_pass=_as_list(row.get("PASS_TO_PASS", []), "PASS_TO_PASS", where),
    )


def load_tasks(path: Path) -> list[Task]:
    """Load SWE-bench-style rows from a JSONL file."""
    tasks: list[Task] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskFileError(f"{where}: invalid JSON") from exc
        task = _parse_row(row, where)
        if task.task_id in seen:
            raise TaskFileError(f"{where}: duplicate task id {task.task_id!r}")
        seen.add(task.task_id)
        tasks.append(task)
    return tasks


def assign_split(task_id: str) -> Split:
    bucket = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest(), 16) % _BUCKETS
    if bucket < DEV_SHARE:
        return Split.DEV
    if bucket < DEV_SHARE + VALIDATION_SHARE:
        return Split.VALIDATION
    return Split.HELDOUT


def select_subset(tasks: list[Task], split: Split, n: int, seed: int) -> list[Task]:
    """A seeded, order-independent sample of up to `n` tasks from one split."""
    pool = sorted((t for t in tasks if assign_split(t.task_id) is split), key=lambda t: t.task_id)
    random.Random(seed).shuffle(pool)
    return sorted(pool[:n], key=lambda t: t.task_id)
