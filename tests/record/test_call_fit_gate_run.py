"""Call-fit gate runner (`scripts/call_fit_gate/run_gate.py`) end to end on a tiny repository where the answer is known by construction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "call_fit_gate" / "run_gate.py"
NEEDS_COVERAGE = pytest.mark.skipif(importlib.util.find_spec("coverage") is None, reason="the gate measures which lines the tests reach with coverage")

FILES = {
    "conftest.py": "",
    "pkg/__init__.py": "",
    "pkg/billing.py": "def charge(amount, rate=1):\n    return amount * rate\n",
    "pkg/cart.py": "from pkg.billing import charge\n\n\ndef checkout(amount):\n    return charge(amount)\n",
    "tests/test_cart.py": "from pkg.cart import checkout\n\n\ndef test_checkout():\n    assert checkout(2) == 2\n",
}


@pytest.fixture(scope="module", name="gate")
def load_gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_gate"] = module
    spec.loader.exec_module(module)
    return module


def make_source(root: Path) -> Path:
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def test_related_matches_the_qualified_name_python_prints(gate):
    from argus.record.call_fit_gate import BindingError

    node = type("N", (), {"qualname": "pkg.cart.Cart.add", "name": "add"})()
    mine = BindingError("a.py", 1, "f", "TypeError", "Cart.add() missing 1 required positional argument: 'qty'")
    other = BindingError("a.py", 1, "f", "TypeError", "Other.add() missing 1 required positional argument: 'qty'")

    assert gate.related(mine, node) and not gate.related(other, node)


def test_remap_moves_lines_of_the_mutated_file_only(gate):
    sites = {("a.py", 1), ("a.py", 2), ("b.py", 2)}

    assert gate.remap(sites, "a.py", {1: 1, 2: 3}) == {("a.py", 1), ("a.py", 3), ("b.py", 2)}


@NEEDS_COVERAGE
def test_gate_end_to_end_flags_exactly_the_call_that_fails(gate, tmp_path, monkeypatch):
    source = make_source(tmp_path / "source")
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["run_gate", str(source), str(tmp_path / "copy"), "--python", sys.executable, "--per-kind", "2", "--seed", "1", "--kinds", "add_required", "--out", str(out)])
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}

    assert gate.main() == 0

    doc = json.loads(out.read_text())
    rows = {r["function"]: r for r in doc["rows"]}
    assert set(rows) == {"pkg.billing.charge", "pkg.cart.checkout"}  # the only two functions with a known caller
    for row in rows.values():  # every flagged call really fails, and every failing call was flagged
        assert row["flagged"] and row["flagged"] == row["observed"]
        assert row["wrong_fact"] == 0
    assert doc["config"]["seed"] == 1 and doc["config"]["python_version"].startswith("Python")
    assert "VERDICT: CONTINUE" in "\n".join(doc["report"])
    assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == before  # the source is never touched
