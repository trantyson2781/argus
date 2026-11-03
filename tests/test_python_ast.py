"""Per-file parsing: signatures, import context, and blind spots."""

from __future__ import annotations

from pathlib import Path

from argus.extract.discovery import discover
from argus.extract.python_ast import parse_file
from argus.model.schema import ImportFlag, NodeKind


def _facts(repo: Path, qualname: str):
    source_file = next(f for f in discover(repo) if f.qualname == qualname)
    return parse_file(source_file.path, qualname)


def test_type_checking_imports_are_flagged_type_only(sample_repo: Path) -> None:
    facts = _facts(sample_repo, "shop.billing.invoice")

    repo_import = next(i for i in facts.imports if i.name == "InvoiceRepo")
    assert repo_import.type_only is True

    domain_import = next(i for i in facts.imports if i.name == "Money")
    assert domain_import.type_only is False


def test_relative_import_records_level(sample_repo: Path) -> None:
    facts = _facts(sample_repo, "shop.billing.invoice")

    domain_import = next(i for i in facts.imports if i.name == "Money")
    assert domain_import.level == 2
    assert ImportFlag.RELATIVE in domain_import.flags


def test_function_body_import_is_lazy(sample_repo: Path) -> None:
    facts = _facts(sample_repo, "shop.infra.repo")

    lazy = next(i for i in facts.imports if i.name == "void_invoice")
    assert ImportFlag.LAZY in lazy.flags


def test_import_guarded_by_import_error_is_optional(sample_repo: Path) -> None:
    facts = _facts(sample_repo, "shop.dynamic")

    optional = next(i for i in facts.imports if i.name == "ujson")
    assert ImportFlag.OPTIONAL in optional.flags


def test_extracts_full_annotated_signature(sample_repo: Path) -> None:
    facts = _facts(sample_repo, "shop.billing.invoice")

    charge = next(s for s in facts.symbols if s.name == "charge")
    assert charge.kind is NodeKind.METHOD
    assert charge.signature is not None
    assert charge.signature.is_async is True
    assert charge.signature.returns == "bool"

    retries = next(p for p in charge.signature.parameters if p.name == "retries")
    assert retries.kind == "keyword-only"
    assert retries.has_default is True
    assert retries.default == "3"
    assert retries.annotation == "int"


def test_dunder_all_is_captured(sample_repo: Path) -> None:
    facts = _facts(sample_repo, "shop.domain")

    assert facts.dunder_all == ("Money",)


def test_dynamic_behaviour_becomes_blind_spots(sample_repo: Path) -> None:
    facts = _facts(sample_repo, "shop.dynamic")

    kinds = {b.kind for b in facts.blind_spots}
    assert {"star_import", "dynamic_import", "dynamic_attribute", "eval_exec", "sys_path_mutation"} <= kinds


def test_syntax_error_is_reported_not_raised(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("def oops(:\n")

    facts = parse_file(broken, "broken")

    assert facts.parse_error is not None
    assert facts.symbols == ()


def test_tuple_unpacking_binds_every_name(tmp_path: Path) -> None:
    """`BUY_NOW, WATCH, AVOID = "a", "b", "c"` defines three module constants.

    Missing them leaves real symbols out of the public surface and makes every
    importer of them look unresolvable.
    """
    module = tmp_path / "labels.py"
    module.write_text('BUY_NOW, WATCH, AVOID = "BUY-NOW", "WATCH", "AVOID"\n')

    facts = parse_file(module, "labels")

    names = {s.name: s for s in facts.symbols}
    assert set(names) == {"BUY_NOW", "WATCH", "AVOID"}
    assert all(s.kind is NodeKind.CONSTANT for s in names.values())
    assert names["WATCH"].attrs["value"] == "'WATCH'"


def test_unpacking_from_an_unpairable_value_still_binds_names(tmp_path: Path) -> None:
    module = tmp_path / "pair.py"
    module.write_text("left, right = compute()\n")

    facts = parse_file(module, "pair")

    assert {s.name for s in facts.symbols} == {"left", "right"}
    assert all(s.attrs["value"] is None for s in facts.symbols)


def test_starred_unpacking_binds_the_starred_name(tmp_path: Path) -> None:
    module = tmp_path / "rest.py"
    module.write_text("head, *tail = [1, 2, 3]\n")

    facts = parse_file(module, "rest")

    assert {s.name for s in facts.symbols} == {"head", "tail"}


def test_version_guarded_function_redefinition_gets_distinct_ids(tmp_path: Path) -> None:
    """Regression: `certifi.core.where()` and similar version-guarded functions.

    Both branches of `if/else` are visited unconditionally since static
    extraction can't evaluate `sys.version_info`; the two `def where():`
    definitions are both real facts and must not collide onto one node id.
    """
    module = tmp_path / "core.py"
    module.write_text(
        "import sys\n"
        "if sys.version_info >= (3, 11):\n"
        "    def where():\n"
        "        return 'new'\n"
        "else:\n"
        "    def where():\n"
        "        return 'old'\n"
    )

    facts = parse_file(module, "core")

    where_symbols = [s for s in facts.symbols if s.name == "where"]
    assert len(where_symbols) == 2
    local_qualnames = {s.local_qualname for s in where_symbols}
    assert local_qualnames == {"where", "where#2"}
    # Display name stays the plain, unsuffixed name for both.
    assert {s.name for s in where_symbols} == {"where"}


def test_try_except_fallback_method_redefinition_gets_distinct_ids(tmp_path: Path) -> None:
    """Regression: click's `Context.lookup_default`-style try/except fallback methods."""
    module = tmp_path / "core.py"
    module.write_text(
        "class Context:\n"
        "    try:\n"
        "        def lookup_default(self):\n"
        "            return 1\n"
        "    except Exception:\n"
        "        def lookup_default(self):\n"
        "            return 2\n"
    )

    facts = parse_file(module, "core")

    methods = [s for s in facts.symbols if s.name == "lookup_default"]
    assert len(methods) == 2
    assert {s.local_qualname for s in methods} == {"Context.lookup_default", "Context.lookup_default#2"}
    assert all(s.parent_local == "Context" for s in methods)


def test_module_level_variable_reassigned_in_both_branches_gets_distinct_ids(tmp_path: Path) -> None:
    """Regression: `_CACERT_CTX = None` assigned in both branches of an if/else."""
    module = tmp_path / "core.py"
    module.write_text(
        "if True:\n"
        "    _CACERT_PATH = None\n"
        "else:\n"
        "    _CACERT_PATH = 'x'\n"
    )

    facts = parse_file(module, "core")

    path_symbols = [s for s in facts.symbols if s.name == "_CACERT_PATH"]
    assert len(path_symbols) == 2
    assert {s.local_qualname for s in path_symbols} == {"_CACERT_PATH", "_CACERT_PATH#2"}


def test_single_definition_is_unaffected_by_disambiguation(tmp_path: Path) -> None:
    """The overwhelmingly common case — one definition per name — keeps its plain id."""
    module = tmp_path / "plain.py"
    module.write_text("def solo():\n    pass\n")

    facts = parse_file(module, "plain")

    assert [s.local_qualname for s in facts.symbols] == ["solo"]
