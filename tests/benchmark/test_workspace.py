from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from argus.benchmark.tasks import Task
from argus.benchmark.workspace import WorkspaceError, extract_patch, prepare_workspace


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "one")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text("x = 2\n")
    _git(repo, "commit", "-qam", "two")
    return repo, first, _git(repo, "rev-parse", "HEAD")


def _task(commit: str) -> Task:
    return Task("o__r-1", "o/r", commit, "p", "", "", (), ())


def test_prepare_workspace_checks_out_the_base_commit(tmp_path: Path, upstream) -> None:
    repo, first, _ = upstream
    ws = prepare_workspace(_task(first), tmp_path / "cache", tmp_path / "work", repo_url=lambda _r: str(repo))
    assert (ws / "a.py").read_text() == "x = 1\n"


def test_prepare_workspace_is_fresh_each_time(tmp_path: Path, upstream) -> None:
    repo, first, _ = upstream
    kwargs = dict(cache_dir=tmp_path / "cache", work_dir=tmp_path / "work", repo_url=lambda _r: str(repo))
    ws = prepare_workspace(_task(first), **kwargs)
    (ws / "a.py").write_text("dirty\n")
    ws2 = prepare_workspace(_task(first), **kwargs)
    assert (ws2 / "a.py").read_text() == "x = 1\n"


def test_prepare_workspace_unknown_commit_raises(tmp_path: Path, upstream) -> None:
    repo, _, _ = upstream
    with pytest.raises(WorkspaceError):
        prepare_workspace(_task("0" * 40), tmp_path / "cache", tmp_path / "work", repo_url=lambda _r: str(repo))


def test_extract_patch_includes_edits_and_new_files_but_not_aider_files(tmp_path: Path, upstream) -> None:
    repo, first, _ = upstream
    ws = prepare_workspace(_task(first), tmp_path / "cache", tmp_path / "work", repo_url=lambda _r: str(repo))
    (ws / "a.py").write_text("x = 3\n")
    (ws / "new.py").write_text("y = 1\n")
    (ws / ".aider.chat.history.md").write_text("noise")
    patch = extract_patch(ws)
    assert "a.py" in patch and "new.py" in patch and ".aider" not in patch


def test_extract_patch_empty_when_no_changes(tmp_path: Path, upstream) -> None:
    repo, first, _ = upstream
    ws = prepare_workspace(_task(first), tmp_path / "cache", tmp_path / "work", repo_url=lambda _r: str(repo))
    assert extract_patch(ws) == ""


def test_workspace_contains_no_future_history(tmp_path: Path, upstream) -> None:
    repo, first, second = upstream
    ws = prepare_workspace(_task(first), tmp_path / "cache", tmp_path / "work", repo_url=lambda _r: str(repo))
    assert _git(ws, "log", "--all", "--format=%H").split() == [first]
    assert _git(ws, "remote") == ""
    probe = subprocess.run(["git", "cat-file", "-e", second], cwd=ws, capture_output=True)
    assert probe.returncode != 0


def test_workspace_holds_no_path_to_the_mirror_or_upstream(tmp_path: Path, upstream) -> None:
    repo, first, _ = upstream
    cache = tmp_path / "cache"
    ws = prepare_workspace(_task(first), cache, tmp_path / "work", repo_url=lambda _r: str(repo))
    forbidden = (str(cache).encode(), str(repo).encode())  # the mirror holds the fix; the path would point at it
    leaks = [str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()
             and any(f in p.read_bytes() for f in forbidden)]
    assert not leaks, f"workspace files name the mirror or upstream: {leaks}"


def test_task_id_cannot_escape_work_dir(tmp_path: Path, upstream) -> None:
    repo, first, _ = upstream
    evil = Task("..", "o/r", first, "p", "", "", (), ())
    with pytest.raises(WorkspaceError, match="escapes"):
        prepare_workspace(evil, tmp_path / "cache", tmp_path / "work", repo_url=lambda _r: str(repo))


def test_partial_mirror_is_not_reused(tmp_path: Path, upstream) -> None:
    repo, first, _ = upstream
    cache = tmp_path / "cache"
    (cache / "o__r.git.partial").mkdir(parents=True)
    (cache / "o__r.git.partial" / "junk").write_text("x")
    ws = prepare_workspace(_task(first), cache, tmp_path / "work", repo_url=lambda _r: str(repo))
    assert (ws / "a.py").exists()


def test_extract_patch_handles_binary_files(tmp_path: Path, upstream) -> None:
    repo, first, _ = upstream
    ws = prepare_workspace(_task(first), tmp_path / "cache", tmp_path / "work", repo_url=lambda _r: str(repo))
    (ws / "blob.bin").write_bytes(bytes(range(256)))
    assert "GIT binary patch" in extract_patch(ws)
