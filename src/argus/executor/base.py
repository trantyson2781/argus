"""Executor interface (plan Phase 3). Phase 0 only needs the vanilla case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from argus.benchmark.tasks import TaskView


@dataclass(frozen=True, slots=True)
class Context:
    """What Argus hands the executor beyond the task text. Empty means vanilla."""

    files: tuple[str, ...] = ()
    preamble: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    llm_calls: int = 0
    tool_calls: int | None = None  # None means "not measured", never a fabricated 0
    # Infrastructure problem: the model did not get a fair attempt. The task will be retried.
    error: str | None = None
    # The agent ran out of time. That is the agent's outcome and is NOT retried.
    timed_out: bool = False
    cost_unknown: bool = False


class Executor(Protocol):
    name: str
    model: str

    def execute(self, task: TaskView, context: Context, workspace: Path) -> ExecutionResult:
        """Edit `workspace` in place to resolve `task`. Never raises for model/tool failure."""
        ...
