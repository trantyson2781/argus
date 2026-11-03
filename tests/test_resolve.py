"""Re-export canonicalization is the single highest-leverage correctness fix.

If `from shop import Invoice` resolves to the package root instead of
`shop.billing.invoice`, the whole dependency structure collapses into a lie.
"""

from __future__ import annotations

from pathlib import Path

from argus.extract.discovery import discover
from argus.extract.python_ast import parse_file
from argus.extract.resolve import SymbolTable
from argus.model.schema import Confidence


def _table(repo: Path) -> tuple[SymbolTable, dict]:
    files = discover(repo)
    facts = {f.qualname: parse_file(f.path, f.qualname) for f in files}
    return SymbolTable(files, facts), facts


def test_reexport_through_package_init_resolves_to_defining_module(sample_repo: Path) -> None:
    table, facts = _table(sample_repo)

    invoice_import = next(i for i in facts["shop.infra.repo"].imports if i.name == "Invoice")
    target = table.resolve("shop.infra.repo", invoice_import)

    assert target.module == "shop.billing.invoice"
    assert target.symbol == "Invoice"
    assert target.canonicalized_from == "shop.Invoice"
    assert target.confidence is Confidence.EXTRACTED


def test_reexport_chain_resolves_through_two_hops(sample_repo: Path) -> None:
    table, facts = _table(sample_repo)

    money_import = next(i for i in facts["shop.infra.repo"].imports if i.name == "Money")
    target = table.resolve("shop.infra.repo", money_import)

    assert target.qualname == "shop.domain.Money"


def test_relative_import_resolves_against_package(sample_repo: Path) -> None:
    table, facts = _table(sample_repo)

    money_import = next(i for i in facts["shop.billing.invoice"].imports if i.name == "Money")
    target = table.resolve("shop.billing.invoice", money_import)

    assert target.module == "shop.domain"
    assert target.symbol == "Money"


def test_relative_import_from_package_init_resolves_to_itself(sample_repo: Path) -> None:
    table, facts = _table(sample_repo)

    invoice_import = next(i for i in facts["shop"].imports if i.name == "Invoice")
    target = table.resolve("shop", invoice_import)

    assert target.module == "shop.billing.invoice"


def test_third_party_import_is_external_not_ambiguous(sample_repo: Path) -> None:
    table, facts = _table(sample_repo)

    httpx_import = next(i for i in facts["shop.infra.repo"].imports if i.name == "httpx")
    target = table.resolve("shop.infra.repo", httpx_import)

    assert target.is_internal is False
    assert target.qualname == "httpx"
    # We know exactly what was imported; it is simply outside the model.
    assert target.confidence is Confidence.EXTRACTED


def test_aliased_import_keeps_canonical_target(sample_repo: Path) -> None:
    table, facts = _table(sample_repo)

    aliased = next(i for i in facts["shop.infra.repo"].imports if i.asname == "M2")
    target = table.resolve("shop.infra.repo", aliased)

    assert target.qualname == "shop.domain.Money"


def test_name_not_findable_is_marked_ambiguous(sample_repo: Path) -> None:
    table, _ = _table(sample_repo)

    target = table.canonicalize("shop.domain", "NotDefinedAnywhere")

    assert target.confidence is Confidence.AMBIGUOUS


def test_sys_path_shifted_module_resolves_by_unique_suffix(tmp_path: Path) -> None:
    """`sys.path.insert(0, 'eval')` makes `eval/run_eval.py` importable as `run_eval`.

    Reporting it as a third-party dependency would be a confidently wrong edge, so
    a unique suffix match resolves it — marked INFERRED, since the path mutation
    itself is not statically verifiable.
    """
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "run_eval.py").write_text("def infer() -> int:\n    return 1\n")
    (tmp_path / "main.py").write_text(
        "import sys\nsys.path.insert(0, 'eval')\nimport run_eval\n"
        "from run_eval import infer\n"
    )
    table, facts = _table(tmp_path)

    module_import = next(i for i in facts["main"].imports if i.name == "run_eval")
    symbol_import = next(i for i in facts["main"].imports if i.name == "infer")

    assert table.resolve("main", module_import).module == "eval.run_eval"
    assert table.resolve("main", module_import).confidence is Confidence.INFERRED
    assert table.resolve("main", symbol_import).qualname == "eval.run_eval.infer"


def test_suffix_fallback_never_hijacks_a_stdlib_import(tmp_path: Path) -> None:
    """A local `json.py` must not capture `import json`."""
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "json.py").write_text("x = 1\n")
    (tmp_path / "main.py").write_text("import json\n")
    table, facts = _table(tmp_path)

    target = table.resolve("main", facts["main"].imports[0])

    assert target.is_internal is False
    assert target.qualname == "json"


def test_ambiguous_suffix_match_is_not_guessed(tmp_path: Path) -> None:
    """Two candidates means Argus cannot know; it must not pick one."""
    for package in ("a", "b"):
        (tmp_path / package).mkdir()
        (tmp_path / package / "shared.py").write_text("x = 1\n")
    (tmp_path / "main.py").write_text("import shared\n")
    table, facts = _table(tmp_path)

    assert table.resolve("main", facts["main"].imports[0]).is_internal is False
