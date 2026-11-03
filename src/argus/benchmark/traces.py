"""Per-run machine-readable traces (plan §16). One JSON object per line."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


class TraceError(ValueError):
    """A trace file is corrupt."""


@dataclass(frozen=True, slots=True)
class RunTrace:
    task_id: str
    agent: str
    model: str
    config_hash: str
    split: str
    input_tokens: int
    output_tokens: int
    cost: float
    llm_calls: int
    tool_calls: int | None  # None = not measured by this executor
    test_runs: int | None
    attempts: int  # 1-based attempt number for this task under this config
    runtime_seconds: float
    success: bool
    files_changed: int
    lines_changed: int
    patch: str
    error: str | None = None
    # Where an error arose: "workspace", "executor" or "evaluator". Errors are infrastructure
    # problems and are retried; they never count as the model's failure.
    error_kind: str | None = None
    timed_out: bool = False  # the agent ran out of time: its own outcome, not retried
    cost_unknown: bool = False  # tokens were used but the model has no price: cost is a lower bound
    reused_patch: bool = False  # re-evaluation of a stored patch; usage is a copy, not new spend
    timestamp: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


_FIELDS = {f.name for f in fields(RunTrace)}


class TraceWriter:
    """Append-only JSONL trace log; also answers which tasks already ran."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, trace: RunTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(trace), sort_keys=True) + "\n")

    def read(self) -> list[RunTrace]:
        return read_traces(self.path)

    def completed_ids(self, config_hash: str) -> set[str]:
        """Tasks with a clean trace; errored runs are infrastructure failures and get retried."""
        return {t.task_id for t in read_traces(self.path) if t.config_hash == config_hash and t.error is None}


def read_traces(path: Path) -> list[RunTrace]:
    if not path.exists():
        return []
    traces: list[RunTrace] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            traces.append(RunTrace(**{k: v for k, v in row.items() if k in _FIELDS}))
        except (json.JSONDecodeError, TypeError) as exc:
            if number == len(lines) and isinstance(exc, json.JSONDecodeError):
                break  # a crash mid-write truncates only the last line; earlier runs stay readable
            raise TraceError(f"{path}:{number}: unreadable trace ({exc})") from exc
    return traces


def latest_by_task(traces: list[RunTrace]) -> list[RunTrace]:
    """The most recent trace per task (a retried task appears more than once in the log)."""
    return list({t.task_id: t for t in traces}.values())
