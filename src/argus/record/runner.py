"""Run a repository's tests on a throwaway copy: make the copy safely, run pytest with a time limit in its own process group, read the result.

EXECUTES the repository's code, so only the explicit commands that use it (`argus check`, the gate scripts) may import this module;
`tests/record/test_guard.py` keeps `explore/` and `tui/` away from it. Shared by `argus check` and `scripts/break_check/run_break_check.py`.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from argus.record.break_check import FailedTests, failed_tests

SKIPPED_DIRS = (".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".argus")
# -B: never write .pyc files, so a run cannot load the compiled code of an earlier break (a .pyc is valid by whole-second mtime and size)
# No -q here: a repo whose addopts already has -q would become -qq, and pytest prints no summary line at -qq (nothing to count).
# shortcut: a repo that sets -qq itself still gets no count, when one reports "none could be run" for it, read the pass lines instead.
PYTEST_ARGS = ("-B", "-m", "pytest", "-rfE", "--tb=no", "-p", "no:randomly", "-p", "no:cacheprovider")
KILL_GRACE_SECONDS = 5
SCRUBBED_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TMPDIR")  # no keys, no tokens, and HOME is replaced by an empty folder


class UnsafeCopyError(Exception):
    """The copy directory is the source repository, inside it, contains it, or is not empty."""


def ensure_safe_copy_dir(source: Path, copy_dir: Path) -> None:
    """Raise UnsafeCopyError when `copy_dir` equals, lies inside, or contains `source` (after resolving links), or already holds files."""
    s, c = source.resolve(), copy_dir.resolve()
    if c == s or c.is_relative_to(s) or s.is_relative_to(c):
        raise UnsafeCopyError(f"{copy_dir} is the source repository, inside it, or contains it")
    if c.exists() and (not c.is_dir() or any(c.iterdir())):
        raise UnsafeCopyError(f"{copy_dir} is not an empty directory")


def copy_repo(source: Path, copy_dir: Path) -> None:
    """Copy `source` without the environment and cache folders. Links are copied as the files they point to, so a write into the copy
    can never reach a file outside it; a link that points nowhere is left out."""
    shutil.copytree(source, copy_dir, symlinks=False, ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*SKIPPED_DIRS), dirs_exist_ok=True)


def scrubbed_env(home: Path) -> dict[str, str]:
    """The environment a repository's tests get: a few harmless variables and an empty HOME, so no key or token of the user reaches them."""
    return {**{k: os.environ[k] for k in SCRUBBED_ENV_KEYS if k in os.environ}, "HOME": str(home)}


def run_pytest(
    python: str, root: Path, timeout: float, args: tuple[str, ...] = PYTEST_ARGS, env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str] | None, float]:
    """(finished process, seconds), or (None, seconds) after a timeout. The run gets its own process group, so a timeout takes the
    processes the tests started with it instead of leaving them to disturb the next run. `env` None means this process's environment."""
    begun = time.monotonic()
    proc = subprocess.Popen(
        [python, *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
        errors="replace", start_new_session=True, env=env,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except BaseException as exc:  # a timeout, or Ctrl-C: the tests run in their own session, so the group must be stopped here
        _stop_group(proc)
        if isinstance(exc, subprocess.TimeoutExpired):
            return None, time.monotonic() - begun
        raise
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, ""), time.monotonic() - begun


def _stop_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:  # everything already ended
        pass
    try:
        proc.communicate(timeout=KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:  # a child that left the group still holds the pipe: stop waiting for it
        if proc.stdout:
            proc.stdout.close()


def parse_run(returncode: int, stdout: str) -> FailedTests | None:
    """The failing tests of a run, or None when the run cannot be read: exit codes other than 0 and 1, a collection error, an exit code
    that disagrees with the summary (1 with no test named, 0 with one named)."""
    failed = failed_tests(stdout)
    if returncode not in (0, 1) or failed.collection_error or (returncode == 0) == bool(failed.tests):
        return None
    return failed
