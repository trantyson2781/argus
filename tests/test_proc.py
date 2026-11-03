from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from argus.proc import run_in_group


def test_returns_output_and_exit_code(tmp_path: Path) -> None:
    done = run_in_group([sys.executable, "-c", "print('hi')"], cwd=tmp_path, timeout=20)
    assert done.returncode == 0 and done.stdout.strip() == "hi"


def test_timeout_kills_grandchildren(tmp_path: Path) -> None:
    marker = tmp_path / "alive"
    script = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',\"import time,pathlib;time.sleep(2);pathlib.Path(r'{marker}').write_text('x')\"]);"
        "time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_in_group([sys.executable, "-c", script], cwd=tmp_path, timeout=1)
    time.sleep(3)
    assert not marker.exists()
