"""Fresh per-run repository checkouts and patch extraction.

A workspace contains exactly one commit: `base_commit`, with no remotes, refs, or later history,
so nothing in it (not even `git log --all`) can reveal the upstream fix.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from argus.benchmark.tasks import Task

_GIT_TIMEOUT_SECONDS = 900
_EXCLUDES = (".aider*", ".argus/", "__pycache__/", "*.pyc")


class WorkspaceError(RuntimeError):
    """A workspace could not be prepared."""


def github_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _git(cwd: Path | None, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceError(f"git {' '.join(args)} timed out") from exc
    if done.returncode != 0:
        raise WorkspaceError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _has_commit(mirror: Path, sha: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=mirror, capture_output=True, check=False
    ).returncode == 0


def _ensure_mirror(task: Task, cache_dir: Path, repo_url: Callable[[str], str]) -> Path:
    mirror = cache_dir / (task.repo.replace("/", "__") + ".git")
    if mirror.exists():
        if not _has_commit(mirror, task.base_commit):
            _git(mirror, "fetch", "--quiet", "--all")
        return mirror
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cache_dir / (mirror.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    _git(None, "clone", "--quiet", "--mirror", "--", repo_url(task.repo), str(partial))
    partial.rename(mirror)  # a half-finished clone is never mistaken for a complete mirror
    return mirror


def prepare_workspace(
    task: Task,
    cache_dir: Path,
    work_dir: Path,
    repo_url: Callable[[str], str] = github_url,
) -> Path:
    """Materialise `task.base_commit` as a single-commit repository, deleting any previous copy."""
    mirror = _ensure_mirror(task, cache_dir, repo_url)
    work_root = work_dir.resolve()
    dest = (work_root / task.task_id).resolve()
    if dest.parent != work_root:
        raise WorkspaceError(f"task id escapes the work directory: {task.task_id!r}")
    if dest.exists():
        shutil.rmtree(dest)
    work_root.mkdir(parents=True, exist_ok=True)
    _git(None, "init", "--quiet", str(dest))
    _git(dest, "fetch", "--quiet", "--depth", "1", "--no-tags", str(mirror), task.base_commit)
    _git(dest, "checkout", "--quiet", "--detach", "FETCH_HEAD")
    (dest / ".git" / "FETCH_HEAD").unlink()  # it names the mirror, which holds the upstream fix
    for ref in _git(dest, "for-each-ref", "--format=%(refname)").split():
        _git(dest, "update-ref", "-d", ref)
    _git(dest, "reflog", "expire", "--expire=now", "--all")
    _git(dest, "gc", "--quiet", "--prune=now")
    exclude = dest / ".git" / "info" / "exclude"
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(_EXCLUDES) + "\n")
    return dest


def extract_patch(workspace: Path) -> str:
    """Unified diff of everything the run changed, including new files and binaries."""
    _git(workspace, "add", "-A")
    return _git(workspace, "diff", "--cached", "--no-color", "--binary", "HEAD")
