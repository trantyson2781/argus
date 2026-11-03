"""Check gate runner (`scripts/check_gate/run_gate.py`) end to end on a tiny repository where the answer is known by construction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_gate" / "run_gate.py"
NEEDS_312 = pytest.mark.skipif(sys.version_info < (3, 12), reason="argus record needs Python >= 3.12 in the interpreter that runs the tests")

FILES = {
    "conftest.py": "",
    "pkg/__init__.py": "",
    "pkg/billing.py": "def charge(amount):\n    return amount * 2\n",
    "pkg/cart.py": "from pkg.billing import charge\n\n\ndef checkout(amount):\n    total = charge(amount)\n    return total\n",
    "tests/test_cart.py": "from pkg.cart import checkout\n\n\ndef test_checkout():\n    assert checkout(2) == 4\n",
}


@pytest.fixture(scope="module", name="gate")
def load_gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_run_gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_run_gate"] = module
    spec.loader.exec_module(module)
    return module


@NEEDS_312
def test_gate_end_to_end_picks_the_failing_test_and_never_touches_the_source(gate, tmp_path, monkeypatch):
    source = tmp_path / "source"
    for rel, text in FILES.items():
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / rel).write_text(text, encoding="utf-8")
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["run_gate", str(source), str(tmp_path / "copy"), "--python", sys.executable, "--per-kind", "2", "--seed", "1", "--out", str(out)])

    assert gate.main() == 0

    doc = json.loads(out.read_text())
    used = [r for r in doc["rows"] if not r["skipped"]]
    assert used and all(r["truth"] and set(r["truth"]) <= set(r["selected"]) for r in used)  # every really failing test was picked
    assert all(set(r["truth"]) <= set(r["newly_failing"]) for r in used)  # and blamed
    assert doc["config"]["recordings"]["full"]["processes"]["saved"] >= 1 and doc["config"]["partial_recording_files"] == 1
    assert all(set(r["newly_failing"]) <= set(r["truth"]) for r in used)  # and nothing else was blamed
    assert doc["config"]["seed"] == 1 and "VERDICT" in doc["report"][-1]
    assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
