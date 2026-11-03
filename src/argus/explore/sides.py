"""Diff Explorer: build the two sides of a comparison as temporary directories holding only their `.py` files.

This is the only Explorer module that runs git. Everything downstream scans the directories with the ordinary
`scan(root)` call, so a "before" and an "after" are produced by exactly the same code path. Nothing in the user's checkout
is touched, and the temporary directories are deleted when the context manager exits.
"""

from __future__ import annotations

import ast
import dataclasses
import errno
import io
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from argus.extract.discovery import is_excluded_dir

GIT_TIMEOUT_SECONDS = 60
# Options that stop git from running programs named in the repository's own config. `core.fsmonitor` is executed by plain
# `git ls-files`, so a repository you only want to read could otherwise run a command of its author's choice. `-c` outranks every
# config file (repository, worktree, global), so a hostile `.git/config` cannot switch these back on.
GIT_SAFETY_OPTIONS = ("-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "protocol.allow=never")
# A partial clone fetches a missing object from its promisor remote on demand, which runs `core.sshCommand` or `remote.<n>.uploadpack`
# from the repository's own config. Lazy fetching is switched off, prompts are refused, and variables that name programs are dropped.
# `GIT_ALLOW_PROTOCOL` outranks every config key, so a hostile `protocol.file.allow=always` cannot re-open a transport on a git that lacks GIT_NO_LAZY_FETCH.
GIT_ENV_OVERRIDES = {"GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": "none"}
GIT_ENV_DROPPED = ("GIT_SSH", "GIT_SSH_COMMAND", "GIT_EXTERNAL_DIFF", "GIT_PAGER", "GIT_ASKPASS", "GIT_PROXY_COMMAND")
MAX_FILE_BYTES = 2 * 1024 * 1024  # bigger `.py` files are skipped: reading and parsing them is unbounded work on untrusted input
MAX_TOTAL_BYTES = 256 * 1024 * 1024  # all `.py` content of one side; beyond this the comparison is refused rather than run out of memory
MAX_FILES = 50_000  # `.py` files on one side; empty files add no bytes, so the byte cap alone would let millions of them fill the disk
MAX_AST_DEPTH = 300  # deeper syntax trees overflow the scanner's recursion (a crash), so such files are skipped and reported
REGULAR_FILE_MODE_PREFIX = "100"  # 100644, 100755 and the legacy 100664
SYMLINK_MODE = "120000"
SYMLINK_REASON = "is a symbolic link (not followed)"
NOT_REGULAR_REASON = "is not a regular file (submodule entry or special file)"
OUTSIDE_REASON = "is reached through a symbolic link that leads outside the scanned directory (not read)"
SPARSE_REASON = "is outside the sparse checkout (not on disk)"
IGNORED_REASON = "is on disk but git ignores it or no longer tracks it (not scanned)"
DIRECTORY_REASON = "is inside a directory whose name ends in .py (not analysed)"
DEEP_REASON = "is nested too deeply to analyse (not analysed)"
BAD_NAME_REASON = "name is not valid UTF-8 (not analysed)"
SYMLINK_DIR_REASON = "is reached through a symbolic link directory (not followed)"
PYTHON_SUFFIX = ".py"
WORKING_TREE_LABEL = "working tree"
LAUNCH_LABEL_PREFIX = "launch "
SHORT_SHA_LENGTH = 7
SIDE_DIR_NAME = "repo"


class DiffError(Exception):
    """The comparison cannot be set up (not a repository, unknown ref, git failure). Carries a user-facing message."""


@dataclass(frozen=True, slots=True)
class Side:
    """One side of the comparison. `files` are POSIX paths relative to `root`, which mirrors the scan PATH."""

    label: str
    root: Path
    files: tuple[str, ...]
    untracked: frozenset[str]
    skipped: tuple[tuple[str, str], ...] = ()  # (path, reason) for `.py` files git lists but that were not read safely, sorted


def _git_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in GIT_ENV_DROPPED}
    env.update(GIT_ENV_OVERRIDES)
    return env


def _git(path: Path, *args: str, stdin: bytes | None = None, ok_codes: tuple[int, ...] = (0,)) -> bytes:
    """Run git in `path` and return raw stdout. Raises DiffError on a missing binary, a timeout or an exit code not in `ok_codes`."""
    command = ["git", "--no-optional-locks", *GIT_SAFETY_OPTIONS, *args]
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell; refs are validated and resolved to a SHA first
            command, cwd=path, input=stdin, capture_output=True, check=False, timeout=GIT_TIMEOUT_SECONDS, env=_git_env(),
        )
    except FileNotFoundError:
        if not path.is_dir():  # a missing working directory raises the same error as a missing git binary
            raise DiffError(f"{path} no longer exists.") from None
        raise DiffError("git is not installed or not on PATH.") from None
    except subprocess.TimeoutExpired:
        raise DiffError(f"git {args[0]} timed out after {GIT_TIMEOUT_SECONDS}s.") from None
    if proc.returncode not in ok_codes:
        detail = proc.stderr.decode("utf-8", errors="replace").strip() or f"exit code {proc.returncode}"
        if "promisor" in detail or "lazy fetching" in detail:
            detail += (
                "\nThis repository is missing objects (a partial clone). Argus never fetches on its own, because fetching can run "
                "commands named in the repository's config. Fetch the objects yourself (for example `git fetch --refetch`), then run it again."
            )
        raise DiffError(f"git {args[0]} failed: {detail}")
    return proc.stdout


def _require_work_tree(path: Path) -> None:
    if not path.is_dir():
        raise DiffError(f"Not a directory: {path}")
    try:
        inside = _git(path, "rev-parse", "--is-inside-work-tree").strip()
    except DiffError:
        raise DiffError(f"{path} is not inside a git repository.") from None
    if inside != b"true":
        raise DiffError(f"{path} is not inside a git repository work tree.")


def _resolve_commit(path: Path, ref: str) -> tuple[str, str]:
    """(full sha, label) for `ref`. A ref that looks like an option is refused before git sees it."""
    if ref.startswith("-"):
        raise DiffError(f"Invalid ref {ref!r}: a ref cannot start with '-'.")
    try:
        sha = _git(path, "rev-parse", "--verify", "--quiet", "--end-of-options", f"{ref}^{{commit}}").decode("ascii").strip()
    except DiffError:
        raise DiffError(f"Unknown ref {ref!r}: it does not name a commit in this repository.") from None
    return sha, f"{ref} ({sha[:SHORT_SHA_LENGTH]})"


def extract_commit(path: Path, ref: str, dest: Path) -> frozenset[str]:
    """Write the files git tracks at `ref` into `dest`, over what is there (the commit's version wins), and return their names.
    Raises DiffError for a bad ref.

    The tar is read with the `data` filter: no absolute paths, no `..`, no links that leave `dest`, no device files. The archive is held
    in memory (shortcut: fine for source trees, upgrade to a stream for repositories with large tracked data files). Files marked
    `export-ignore` in `.gitattributes` are not part of an archive.
    """
    _require_work_tree(path)
    sha, _ = _resolve_commit(path, ref)
    data = _git(path, "archive", "--format=tar", sha)
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            names = frozenset(m.name for m in tar.getmembers() if m.isfile())
            tar.extractall(dest, filter="data")
    except (tarfile.TarError, OSError) as err:
        raise DiffError(f"Cannot unpack {ref!r}: {err}") from None
    return names


def _merge_base(path: Path, base: str, base_sha: str, head: str, head_sha: str, head_is_ref: bool) -> tuple[str, str]:
    """(sha, label) of the commit where `base` and `head` split. Unchanged when `base` is already an ancestor of `head`.

    Comparing against the base branch's tip would report work done only on the base branch as removed by the head branch. When an
    explicit `head` is an ancestor of `base` there is nothing the head added since the split, so asking for it is refused rather than
    answered with an empty report. For the working tree, a `HEAD` behind the base is fine: the report is the work done since the split.
    """
    common = _git(path, "merge-base", base_sha, head_sha, ok_codes=(0, 1)).decode("ascii").strip()  # exit 1: no common ancestor
    if not common:
        raise DiffError(
            f"{base!r} and {head!r} share no history in this repository, so there is no common starting point to compare from "
            "(in a shallow clone, fetch more history)."
        )
    if common == base_sha:
        return base_sha, f"{base} ({base_sha[:SHORT_SHA_LENGTH]})"
    if head_is_ref and common == head_sha:
        raise DiffError(f"{base!r} is ahead of {head!r} ({head!r} is an ancestor of it), so {head!r} added nothing since the split. Swap --base and --head to see what {base!r} added.")
    return common, f"merge-base of {base} and {head} ({common[:SHORT_SHA_LENGTH]})"


def _split_records(raw: bytes) -> list[bytes]:
    return [record for record in raw.split(b"\0") if record]


def _is_python(rel: str) -> bool:
    return rel.endswith(PYTHON_SUFFIX)


def _check_count(count: int) -> None:
    if count > MAX_FILES:
        raise DiffError(f"One side has more than {MAX_FILES} Python files; refusing to compare a tree this large.")


def _check_total(total: int) -> None:
    if total > MAX_TOTAL_BYTES:
        raise DiffError(f"The Python files of one side add up to more than {MAX_TOTAL_BYTES} bytes; refusing to compare a tree this large.")


def _too_large_reason() -> str:
    return f"is larger than {MAX_FILE_BYTES} bytes (not read)"


def _skip_reason(mode: str) -> str:
    return SYMLINK_REASON if mode == SYMLINK_MODE else NOT_REGULAR_REASON


def _commit_blobs(path: Path, sha: str) -> tuple[dict[str, str], dict[str, str]]:
    """({relative path: blob sha}, {path: reason skipped}) for `.py` entries under `path` at commit `sha`. Sizes come from `ls-tree -l`."""
    blobs: dict[str, str] = {}
    skipped: dict[str, str] = {}
    total = 0
    for record in _split_records(_git(path, "ls-tree", "-r", "-z", "-l", sha)):
        meta, _, name = record.decode("utf-8", errors="surrogateescape").partition("\t")
        mode, kind, blob, size = meta.split()
        if not _is_python(name):
            continue
        if not (mode.startswith(REGULAR_FILE_MODE_PREFIX) and kind == "blob"):
            skipped[name] = _skip_reason(mode)
        elif int(size) > MAX_FILE_BYTES:
            skipped[name] = _too_large_reason()
        else:
            total += int(size)
            _check_total(total)
            blobs[name] = blob
            _check_count(len(blobs))
    return blobs, skipped


def _read_blobs(path: Path, blob_ids: list[str]) -> list[bytes]:
    """Raw contents of each blob, in order, using one `cat-file --batch` process."""
    if not blob_ids:
        return []
    out = _git(path, "cat-file", "--batch", stdin=("\n".join(blob_ids) + "\n").encode("ascii"))
    contents: list[bytes] = []
    pos = 0
    for blob in blob_ids:
        end = out.index(b"\n", pos)
        header = out[pos:end].split(b" ")
        if len(header) != 3 or header[1] != b"blob":
            raise DiffError(f"git cat-file could not read blob {blob}.")
        size = int(header[2])
        contents.append(out[end + 1 : end + 1 + size])
        pos = end + 1 + size + 1
    return contents


def _safe_target(root: Path, rel: str) -> Path:
    posix = PurePosixPath(rel)
    if posix.is_absolute() or ".." in posix.parts:
        raise DiffError(f"Refusing to write an unsafe path from git: {rel!r}")
    return root.joinpath(*posix.parts)


def _display_name(name: str) -> str:
    """`name` with undecodable bytes shown as \\xNN, so it can be printed and compared."""
    return name.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="backslashreplace")


def _too_deep(data: bytes) -> bool:
    """True when the file parses but its syntax tree is deeper than the scanner can walk. A plain syntax error is the scanner's to report."""
    try:
        tree = ast.parse(data.decode("utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        return False
    except (RecursionError, MemoryError):
        return True
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_AST_DEPTH:
            return True
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return False


def _unusable_reason(rel: str, data: bytes) -> str | None:
    """Why this file must not be written to the temporary tree or scanned, or None. Applies to commit and working-tree sides alike."""
    parts = PurePosixPath(rel).parts
    if any(part.endswith(PYTHON_SUFFIX) for part in parts[:-1]):
        return DIRECTORY_REASON  # `discover()` matches `*.py` directories too, and then fails reading one
    if len(data) > MAX_FILE_BYTES:
        return _too_large_reason()
    if any(is_excluded_dir(part) for part in parts[:-1]):
        return None  # never scanned, so never walked
    return DEEP_REASON if _too_deep(data) else None


def _has_bad_name(rel: str) -> bool:
    try:
        rel.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


def _admit(files: dict[str, bytes]) -> tuple[dict[str, bytes], dict[str, str]]:
    """Split `files` into those safe to write and scan, and {path: reason} for the rest."""
    admitted: dict[str, bytes] = {}
    refused: dict[str, str] = {}
    for rel, data in files.items():
        reason = BAD_NAME_REASON if _has_bad_name(rel) else _unusable_reason(rel, data)
        if reason is None:
            admitted[rel] = data
        else:
            refused[_display_name(rel)] = reason
    return admitted, refused


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    for rel, content in files.items():
        target = _safe_target(root, rel)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        except OSError as err:
            raise DiffError(f"Cannot copy {rel!r} to a temporary directory: {err.strerror or err.__class__.__name__}.") from None


def _build_commit_side(path: Path, root: Path, sha: str, label: str) -> Side:
    blobs, skipped = _commit_blobs(path, sha)
    names = sorted(blobs)
    admitted, refused = _admit(dict(zip(names, _read_blobs(path, [blobs[n] for n in names]), strict=True)))
    _write_files(root, admitted)
    skipped.update(refused)
    return Side(label=label, root=root, files=tuple(sorted(admitted)), untracked=frozenset(), skipped=tuple(sorted(skipped.items())))


def _working_tree_names(path: Path) -> tuple[set[str], set[str], dict[str, str]]:
    """(tracked, untracked, skipped) `.py` paths relative to `path`. Tracked means in the index as a regular file; ignored files are absent.

    `skipped` holds tracked entries that are not regular files (symlinks, submodules) or are not on disk because of a sparse checkout.
    """
    sparse = {
        record[2:].decode("utf-8", errors="surrogateescape") for record in _split_records(_git(path, "ls-files", "-t", "-z")) if record[:1] == b"S"
    }
    tracked: set[str] = set()
    skipped: dict[str, str] = {}
    for record in _split_records(_git(path, "ls-files", "-s", "-z")):
        meta, _, name = record.decode("utf-8", errors="surrogateescape").partition("\t")
        if not _is_python(name):
            continue
        mode = meta.split(" ")[0]
        if name in sparse:
            skipped[name] = SPARSE_REASON
        elif mode.startswith(REGULAR_FILE_MODE_PREFIX):
            tracked.add(name)
        else:
            skipped[name] = _skip_reason(mode)
    others = {
        name for name in (r.decode("utf-8", errors="surrogateescape") for r in _split_records(_git(path, "ls-files", "-z", "--others", "--exclude-standard")))
        if _is_python(name)
    }
    return tracked, others - tracked - skipped.keys(), skipped


def _diagnose(root: Path, rel: str) -> str | None:
    """Why `rel` could not be opened safely, or None when it is simply absent. Walks the path with lstat; used for the message only."""
    current = root
    parts = PurePosixPath(rel).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return None
        except OSError as err:
            return f"could not be read ({err.strerror or err.__class__.__name__})"
        if stat.S_ISLNK(mode):
            if index == len(parts) - 1:
                return SYMLINK_REASON
            try:
                outside = not current.resolve().is_relative_to(root)
            except RuntimeError:  # Python 3.12 raises on a symlink loop; later versions return a path
                return SYMLINK_DIR_REASON
            return OUTSIDE_REASON if outside else SYMLINK_DIR_REASON
        if index == len(parts) - 1 and not stat.S_ISREG(mode):
            return NOT_REGULAR_REASON
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            return None  # a file where a directory is expected: the tracked file is gone
    return "could not be read"


def _read_regular_file(root: Path, rel: str) -> tuple[bytes | None, str | None]:
    """(content, None) for a regular file reached without following any symlink; (None, None) when absent; (None, reason) otherwise.

    Each path component is opened relative to the previous one with O_NOFOLLOW, so a directory swapped for a link to somewhere
    else (the classic way to make a tool read files outside the repository) is refused rather than followed. `rel` is checked
    first: git never reports an absolute path or a `..` component, so one that appears means a hand-crafted index.
    """
    _safe_target(root, rel)
    parts = PurePosixPath(rel).parts
    opened: list[int] = []
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        opened.append(fd)
        for part in parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            opened.append(fd)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        opened.append(fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None, NOT_REGULAR_REASON
        if info.st_size > MAX_FILE_BYTES:
            return None, _too_large_reason()
        with open(os.dup(fd), "rb") as handle:
            return handle.read(MAX_FILE_BYTES + 1), None
    except FileNotFoundError:
        return None, None
    except OSError as err:
        reason = _diagnose(root, rel)
        if reason is None and err.errno not in (errno.ENOENT, errno.ENOTDIR):
            reason = f"could not be read ({err.strerror or err.__class__.__name__})"
        return None, reason
    finally:
        for fd in opened:
            os.close(fd)


def _build_working_tree_side(path: Path, root: Path) -> Side:
    tracked, untracked, skipped = _working_tree_names(path)
    contents: dict[str, bytes] = {}
    total = 0
    for rel in sorted(tracked | untracked):
        data, reason = _read_regular_file(path, rel)
        if data is not None:
            total += len(data)
            _check_total(total)
            contents[rel] = data
            _check_count(len(contents))
        elif reason is not None:
            skipped[_display_name(rel)] = reason
    admitted, refused = _admit(contents)
    _write_files(root, admitted)
    skipped.update(refused)
    return Side(
        label=WORKING_TREE_LABEL, root=root, files=tuple(sorted(admitted)), untracked=frozenset(untracked & admitted.keys()),
        skipped=tuple(sorted(skipped.items())),
    )


def _with_ignored_on_disk(scan_path: Path, base: Side, head: Side) -> Side:
    """Report files that are in the base but absent from the working-tree side only because git no longer lists them (ignored, untracked).

    Without this they would show up as removed although they are still on disk.
    """
    known = set(head.files) | {name for name, _ in head.skipped}
    extra = {rel: IGNORED_REASON for rel in base.files if rel not in known and os.path.lexists(scan_path / rel)}
    if not extra:
        return head
    return dataclasses.replace(head, skipped=tuple(sorted({*head.skipped, *extra.items()})))


@contextmanager
def materialize_sides(path: Path, base: str, head: str | None = None) -> Iterator[tuple[Side, Side]]:
    """Yield (base side, head side) built in temporary directories that are removed on exit.

    `head=None` means the working tree (what is on disk now, untracked non-ignored `.py` files included). A ref builds that
    side from the commit, exactly like the base side. `path` is the scan root and must be inside a git work tree.
    """
    scan_path = Path(path).resolve()
    _require_work_tree(scan_path)
    base_sha, base_label = _resolve_commit(scan_path, base)
    head_resolved = _resolve_commit(scan_path, head) if head is not None else None
    head_sha = head_resolved[0] if head_resolved else _resolve_commit(scan_path, "HEAD")[0]
    base_sha, base_label = _merge_base(scan_path, base, base_sha, head or "HEAD", head_sha, head is not None)
    workdir = Path(tempfile.mkdtemp(prefix="argus-diff-"))
    try:
        # Both roots have the same directory name: the scanner names a root-level `__init__.py` after its directory.
        base_root, head_root = workdir / "base" / SIDE_DIR_NAME, workdir / "head" / SIDE_DIR_NAME
        base_root.mkdir(parents=True)
        head_root.mkdir(parents=True)
        base_side = _build_commit_side(scan_path, base_root, base_sha, base_label)
        if head_resolved is None:
            head_side = _with_ignored_on_disk(scan_path, base_side, _build_working_tree_side(scan_path, head_root))
        else:
            head_side = _build_commit_side(scan_path, head_root, *head_resolved)
        yield base_side, head_side
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# -- watch mode: a baseline held for a whole session, and a fresh current side for every refresh ------------------------------------------


@contextmanager
def _temp_root(prefix: str) -> Iterator[Path]:
    """A fresh temporary directory holding one side's `repo` root (the scanner names a root-level `__init__.py` after it); removed on exit."""
    workdir = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        root = workdir / SIDE_DIR_NAME
        root.mkdir()
        yield root
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@contextmanager
def materialize_baseline(path: Path, base: str | None = None) -> Iterator[tuple[Path, Side]]:
    """Yield (scan path, baseline side). `base=None` is a snapshot of the working tree as it is now, labelled "launch HH:MM:SS";
    a ref is that commit, compared as it is (no merge-base). The temporary directory is removed on exit, also on an error."""
    scan_path = Path(path).resolve()
    _require_work_tree(scan_path)
    with _temp_root("argus-watch-") as root:
        if base is None:
            side = dataclasses.replace(_build_working_tree_side(scan_path, root), label=LAUNCH_LABEL_PREFIX + time.strftime("%H:%M:%S"))
        else:
            sha, label = _resolve_commit(scan_path, base)
            side = _build_commit_side(scan_path, root, sha, label)
        yield scan_path, side


@contextmanager
def materialize_current(scan_path: Path, base: Side) -> Iterator[Side]:
    """Yield the working tree as it is now, built exactly like the head side of `materialize_sides`; removed on exit."""
    with _temp_root("argus-watch-") as root:
        yield _with_ignored_on_disk(scan_path, base, _build_working_tree_side(scan_path, root))


def python_file_stamps(path: Path) -> dict[str, tuple[int, int] | None]:
    """{path: (mtime_ns, size)} for every `.py` file git lists (tracked or untracked, not ignored); None when listed but gone.

    One git call and one lstat per file, no file is read: this is the cheap check a watcher repeats. Two equal results mean no
    `.py` file was added, removed or rewritten in between.
    """
    stamps: dict[str, tuple[int, int] | None] = {}
    for record in _split_records(_git(path, "ls-files", "-z", "--cached", "--others", "--exclude-standard")):
        name = record.decode("utf-8", errors="surrogateescape")
        if not _is_python(name) or any(is_excluded_dir(part) for part in PurePosixPath(name).parts[:-1]):
            continue  # a directory the scanner skips (a virtualenv, node_modules) cannot change the report
        try:
            info = os.lstat(path / name)
        except OSError:
            stamps[name] = None
        else:
            stamps[name] = (info.st_mtime_ns, info.st_size)
    return stamps
