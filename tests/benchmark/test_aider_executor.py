from __future__ import annotations

import json
import subprocess
from pathlib import Path

from argus.benchmark.tasks import TaskView
from argus.executor.aider_executor import AiderExecutor, build_prompt, find_aider_python
from argus.executor.base import Context

TASK = TaskView("t1", "o/r", "abc1234", "Fix the bug in foo().")
USAGE = {"input_tokens": 12345, "output_tokens": 678, "cost": 0.0123, "llm_calls": 3, "error": None, "api_errors": []}


def test_build_prompt_vanilla_is_just_the_problem_statement() -> None:
    assert build_prompt(TASK, Context()) == "Fix the bug in foo()."


def test_build_prompt_prepends_context_preamble() -> None:
    assert build_prompt(TASK, Context(preamble="NOTES")) == "NOTES\n\nFix the bug in foo()."


def _runner(payload: dict | None, seen: list, returncode: int = 0, stderr: str = "", timeout: bool = False):
    def run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        if payload is not None:
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(payload))
        if timeout:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, returncode, "", stderr)
    return run


def _ex(runner) -> AiderExecutor:
    return AiderExecutor("deepseek/deepseek-chat", aider_python="/py", runner=runner)


def test_execute_passes_model_files_and_reads_exact_usage(tmp_path: Path) -> None:
    seen: list = []
    result = _ex(_runner(USAGE, seen)).execute(TASK, Context(files=("a.py", "b.py")), tmp_path)
    cmd, kwargs = seen[0]
    assert cmd[0] == "/py" and "deepseek/deepseek-chat" in cmd and cmd.count("--file") == 2
    assert kwargs["cwd"] == tmp_path
    assert (result.input_tokens, result.output_tokens, result.llm_calls, result.cost) == (12345, 678, 3, 0.0123)
    assert result.error is None and not result.timed_out and result.tool_calls is None


def test_driver_exception_is_an_infrastructure_error(tmp_path: Path) -> None:
    assert _ex(_runner({**USAGE, "error": "boom"}, [])).execute(TASK, Context(), tmp_path).error == "boom"


def test_api_failure_mid_run_is_an_error_even_when_some_tokens_were_used(tmp_path: Path) -> None:
    payload = {**USAGE, "api_errors": ["RateLimitError 429"]}
    result = _ex(_runner(payload, [])).execute(TASK, Context(), tmp_path)
    assert result.error and "429" in result.error and result.input_tokens == 12345


def test_timeout_keeps_partial_spend_and_is_the_agents_outcome_not_an_error(tmp_path: Path) -> None:
    result = _ex(_runner(USAGE, [], timeout=True)).execute(TASK, Context(), tmp_path)
    assert result.timed_out and result.error is None
    assert result.input_tokens == 12345 and result.cost == 0.0123


def test_timeout_before_any_usage_is_reported_without_inventing_spend(tmp_path: Path) -> None:
    result = _ex(_runner(None, [], timeout=True)).execute(TASK, Context(), tmp_path)
    assert result.timed_out and result.error and result.input_tokens == 0


def test_nonzero_exit_without_result_reports_stderr(tmp_path: Path) -> None:
    result = _ex(_runner(None, [], returncode=1, stderr="ImportError: aider")).execute(TASK, Context(), tmp_path)
    assert result.error and "ImportError" in result.error


def test_malformed_result_file_is_an_error(tmp_path: Path) -> None:
    result = _ex(_runner({"input_tokens": "x"}, [])).execute(TASK, Context(), tmp_path)
    assert result.error and "unreadable" in result.error


def test_cost_unknown_flag_is_propagated(tmp_path: Path) -> None:
    result = _ex(_runner({**USAGE, "cost": 0.0, "cost_unknown": True}, [])).execute(TASK, Context(), tmp_path)
    assert result.cost_unknown


def test_find_aider_python_handles_env_shebang(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "aider"
    script.write_text("#!/usr/bin/env python3\nprint()\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    import pytest
    with pytest.raises(FileNotFoundError, match="shebang"):
        find_aider_python()
