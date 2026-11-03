"""Aider executor. Vanilla Aider is the Phase 0 baseline; Argus context plugs in later."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from argus.benchmark.tasks import TaskView
from argus.executor.base import Context, ExecutionResult
from argus.proc import run_in_group

DEFAULT_TIMEOUT_SECONDS = 1800
DRIVER_PATH = Path(__file__).with_name("_aider_driver.py")


def build_prompt(task: TaskView, context: Context) -> str:
    if context.preamble:
        return f"{context.preamble}\n\n{task.problem_statement}"
    return task.problem_statement


def find_aider_python() -> str:
    """The interpreter of Aider's own environment, read from the `aider` script shebang."""
    aider = shutil.which("aider")
    if not aider:
        raise FileNotFoundError("aider is not on PATH")
    lines = Path(aider).read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or not lines[0].startswith("#!"):
        raise FileNotFoundError(f"cannot find aider's interpreter from {aider}")
    parts = shlex.split(lines[0][2:])
    if len(parts) != 1:
        raise FileNotFoundError(f"unsupported aider shebang {lines[0]!r}; pass aider_python explicitly")
    return parts[0]


def aider_version() -> str:
    """Aider's installed version, recorded in the run config so a baseline is reproducible."""
    done = subprocess.run([find_aider_python(), "-c", "import aider; print(aider.__version__)"],
                          capture_output=True, text=True, check=False)
    version = done.stdout.strip()
    if done.returncode != 0 or not version:
        raise RuntimeError(f"cannot determine aider version: {done.stderr.strip()[-500:]}")
    return version


def _read_result(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class AiderExecutor:
    name = "aider-vanilla"

    def __init__(
        self,
        model: str,
        aider_python: str | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        map_tokens: int | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = run_in_group,
    ) -> None:
        self.model = model
        self._python = aider_python
        self.timeout_seconds = timeout_seconds
        self._map_tokens = map_tokens
        self._runner = runner

    def execute(self, task: TaskView, context: Context, workspace: Path) -> ExecutionResult:
        try:
            python = self._python or find_aider_python()
        except FileNotFoundError as exc:
            return ExecutionResult(error=str(exc))
        with tempfile.TemporaryDirectory(prefix="argus-aider-") as scratch:
            message = Path(scratch) / "message.txt"
            out = Path(scratch) / "result.json"
            message.write_text(build_prompt(task, context), encoding="utf-8")
            cmd = [python, str(DRIVER_PATH), "--model", self.model, "--repo", str(workspace),
                   "--message-file", str(message), "--out", str(out)]
            for name in context.files:
                cmd += ["--file", name]
            if self._map_tokens is not None:
                cmd += ["--map-tokens", str(self._map_tokens)]
            timed_out = False
            stderr = ""
            returncode = 0
            try:
                done = self._runner(cmd, cwd=workspace, timeout=self.timeout_seconds)
                returncode, stderr = done.returncode, done.stderr
            except subprocess.TimeoutExpired:
                timed_out = True
            data = _read_result(out)
        if data is None:
            if timed_out:
                return ExecutionResult(timed_out=True, error="aider timed out before reporting any usage")
            return ExecutionResult(error=f"aider driver failed (exit {returncode}): {stderr.strip()[-2000:]}")
        return self._from_data(data, timed_out, returncode, stderr)

    @staticmethod
    def _from_data(data: dict, timed_out: bool, returncode: int, stderr: str) -> ExecutionResult:
        try:
            usage = {
                "input_tokens": int(data["input_tokens"]), "output_tokens": int(data["output_tokens"]),
                "cost": float(data["cost"]), "llm_calls": int(data["llm_calls"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            return ExecutionResult(error=f"aider driver wrote an unreadable result: {exc!r}")
        error = data.get("error")
        api_errors = data.get("api_errors") or []
        if error is None and api_errors:
            error = "API failure during run: " + " | ".join(api_errors)[-1500:]
        if error is None and not timed_out and returncode != 0:
            error = f"aider driver exited {returncode}: {stderr.strip()[-1500:]}"
        return ExecutionResult(**usage, error=error, timed_out=timed_out and error is None,
                               cost_unknown=bool(data.get("cost_unknown")))
