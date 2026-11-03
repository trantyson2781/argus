"""Run configuration. Its hash keys resume and guards comparisons (same model, same budget)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunConfig:
    agent: str
    model: str
    split: str
    n: int
    seed: int
    # Anything else that changes behaviour: tool versions, token budgets, tasks-file hash.
    extra: tuple[tuple[str, str], ...] = ()

    @property
    def config_hash(self) -> str:
        """Hash of what changes how a task is run. Subset choice (split/n/seed) is excluded so
        a run can be extended to more tasks and still resume."""
        payload = json.dumps({"agent": self.agent, "model": self.model, "extra": self.extra}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
