"""Facts that make a run reproducible, folded into the run config."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import argus.benchmark.config as _config
import argus.benchmark.evaluator as _evaluator
import argus.benchmark.runner as _runner
import argus.benchmark.tasks as _tasks
import argus.benchmark.traces as _traces
import argus.benchmark.workspace as _workspace
import argus.executor._aider_driver as _driver
import argus.executor.aider_executor as _executor
import argus.executor.base as _base
import argus.proc as _proc

# Everything the agent path imports, `tasks` (the TaskView firewall) included. A test walks the
# import closure, so a new module on that path fails until it is listed here.
_BEHAVIOUR_SOURCES = tuple(
    m.__file__ for m in (
        _executor, _driver, _evaluator, _workspace, _runner, _tasks, _base, _config, _traces, _proc,
    )
)


def code_fingerprint(sources: tuple[str, ...] = _BEHAVIOUR_SOURCES) -> str:
    """Hash of the sources that decide what the agent sees and how a run is scored.

    Editing the prompt, what `TaskView` exposes, the driver flags, the workspace recipe, the run
    loop or the evaluator changes it, so traces produced by different behaviour can never be
    resumed into the same run.
    """
    digest = hashlib.sha256()
    for path in sources:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:12]


def git_revision(repo: Path) -> str:
    done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else "no-git"
