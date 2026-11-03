"""Shared fixtures: synthetic codebases written to tmp dirs.

The fixture package exercises the Python hazards from PYTHON-EXTRACTION §3 —
re-exports, TYPE_CHECKING imports, relative imports, lazy imports, star imports
and namespace packages — because those are what determine whether the model is
trustworthy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_FILES: dict[str, str] = {
    "src/shop/__init__.py": '''
"""Shop package. Re-exports the public surface."""
from .billing.invoice import Invoice, void_invoice
from .domain import Money

__all__ = ["Invoice", "Money", "void_invoice"]
''',
    "src/shop/domain.py": '''
"""Pure domain types."""
from dataclasses import dataclass

__all__ = ["Money"]

DEFAULT_CURRENCY = "USD"


@dataclass(frozen=True)
class Money:
    amount: int
    currency: str = DEFAULT_CURRENCY

    def add(self, other: "Money") -> "Money":
        return Money(self.amount + other.amount, self.currency)


def _internal_helper() -> None:
    return None
''',
    "src/shop/billing/__init__.py": "from .invoice import Invoice\n",
    "src/shop/billing/invoice.py": '''
"""Invoice lifecycle."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..domain import Money

if TYPE_CHECKING:
    from shop.infra.repo import InvoiceRepo

STRIPE_KEY = os.environ["STRIPE_KEY"]


class Invoice:
    """An invoice."""

    def __init__(self, total: Money) -> None:
        self.total = total

    async def charge(self, repo: "InvoiceRepo", *, retries: int = 3) -> bool:
        return True


def void_invoice(invoice: Invoice, reason: str | None = None) -> None:
    """Void an invoice."""
    return None
''',
    "src/shop/infra/__init__.py": "",
    "src/shop/infra/repo.py": '''
"""Persistence."""
import httpx

from shop import Invoice, Money
from shop.domain import Money as M2


class InvoiceRepo:
    def save(self, invoice: Invoice) -> None:
        return None

    def fetch(self) -> Money:
        from shop.billing.invoice import void_invoice  # lazy, breaks a cycle

        void_invoice(Invoice(M2(0)))
        return M2(1)

    def sync(self) -> None:
        httpx.get("https://example.test")
''',
    # PEP 420 namespace package: no __init__.py anywhere in the chain.
    "src/plugins/alpha/entry.py": '''
"""Namespace-package module."""
from shop.domain import Money


def make() -> Money:
    return Money(1)
''',
    "src/shop/dynamic.py": '''
"""Everything a static reader cannot see."""
import importlib
import sys

from shop.domain import *

name = "shop.billing"
mod = importlib.import_module(name)
value = getattr(mod, name.upper())
sys.path.append("/opt/extra")
eval("1 + 1")

try:
    import ujson
except ImportError:
    ujson = None
''',
}


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Write the fixture codebase to a temp dir and return its root."""
    for relative, content in FIXTURE_FILES.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.lstrip(), encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _isolate_resolver_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's exported ARGUS_RESOLVE_V2 must never change what a test means; tests opt in or out explicitly."""
    monkeypatch.delenv("ARGUS_RESOLVE_V2", raising=False)
