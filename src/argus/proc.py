"""Subprocess helper that kills the whole process group on timeout."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path


def run_in_group(
    cmd: Sequence[str], *, cwd: Path | None = None, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Like `subprocess.run(capture_output=True, text=True)`, but a timeout kills the children too.

    Raises `subprocess.TimeoutExpired` (with any partial output) after the group is dead, so no
    orphan keeps editing a workspace or holding a container after the caller gave up.
    """
    with subprocess.Popen(
        list(cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(list(cmd), timeout, output=stdout, stderr=stderr) from exc
        return subprocess.CompletedProcess(list(cmd), proc.returncode, stdout, stderr)
