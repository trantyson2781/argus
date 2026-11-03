"""Which Python the R and C keys use to run the repository's tests, and what version it is.

Order: the ARGUS_PYTHON environment variable, the one-line file `.argus/python` in the repository, the repository's own `.venv`, the Python that
runs Argus. A choice that was made on purpose (the first two) but points at nothing runnable is an error: falling back to another Python would
run the tests somewhere the user did not choose.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from argus.explore.render_diff import printable

ENV = "ARGUS_PYTHON"
FILE = ".argus/python"
MIN_RECORDING = (3, 12)  # `sys.monitoring`, which the recorder is built on
VERSION_TIMEOUT = 3  # the screen thread waits for it


@dataclass(frozen=True, slots=True)
class PythonChoice:
    path: str
    source: str  # where the choice came from, in words
    error: str | None = None  # set when a deliberate choice cannot be used; `path` is then what was asked for


def _runnable(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:  # a folder that cannot be read: not runnable, and not a reason to crash
        return False


def _checked(value: str, root: Path, source: str) -> PythonChoice:
    try:
        path = Path(value).expanduser()
    except RuntimeError:  # `~nobody/...` for a user that does not exist
        return PythonChoice(value, source, printable(f"{source} points at {value}, whose home folder cannot be found."))
    path = path if path.is_absolute() else root / path  # not resolved: a venv's python is a link and must stay a path inside the venv
    if _runnable(path):
        return PythonChoice(str(path), source)
    return PythonChoice(value, source, printable(f"{source} points at {value}, which is not a runnable file."))


def chosen_python(root: Path, env: Mapping[str, str]) -> PythonChoice:
    explicit = env.get(ENV, "").strip()
    if explicit:
        return _checked(explicit, root, ENV)
    try:
        text = (root / FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    except (OSError, UnicodeDecodeError) as err:
        return PythonChoice("", FILE, f"{FILE} cannot be read ({type(err).__name__}).")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first:
        return _checked(first, root, FILE)
    venv = root / ".venv" / "bin" / "python"
    if _runnable(venv):
        return PythonChoice(str(venv), "the repository's .venv")
    if venv.is_symlink():  # a .venv whose python is gone: not a reason to run the tests somewhere else without saying so
        return PythonChoice(str(venv), "the repository's .venv", printable(f"The repository's .venv python ({venv}) is a broken link."))
    return PythonChoice(sys.executable, "the Python that runs Argus")


def python_version(path: str, run=subprocess.run, timeout: float = VERSION_TIMEOUT) -> tuple[int, int] | None:
    """(major, minor) of the Python at `path`, or None when it cannot be run or does not answer with a version. `-I`: no repository code is imported."""
    try:
        done = run([path, "-I", "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"], capture_output=True, text=True, timeout=timeout)
        major, minor = done.stdout.split()
        return (int(major), int(minor)) if done.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
