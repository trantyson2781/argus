"""End-to-end model assembly."""

from __future__ import annotations

from pathlib import Path

from argus.extract import scan
from argus.model.ids import is_unresolved
from argus.model.schema import Confidence, EdgeKind, NodeKind
from argus.report import AMBIGUOUS_RATE_GATE, ambiguous_rate, sample_edges


def test_builds_packages_modules_and_symbols(sample_repo: Path) -> None:
    model = scan(sample_repo)

    qualnames = {n.qualname for n in model.nodes}
    assert "shop" in qualnames
    assert "shop.billing.invoice" in qualnames
    assert "shop.billing.invoice.Invoice" in qualnames
    assert "shop.billing.invoice.Invoice.charge" in qualnames


def test_namespace_package_is_marked(sample_repo: Path) -> None:
    model = scan(sample_repo)

    plugins = next(n for n in model.nodes if n.qualname == "plugins" and n.kind is NodeKind.PACKAGE)
    assert plugins.attrs["is_namespace_package"] is True


def test_import_edge_points_at_defining_module_not_package_root(sample_repo: Path) -> None:
    model = scan(sample_repo)
    repo_module = next(n for n in model.nodes if n.qualname == "shop.infra.repo")

    targets = {
        edge.dst_ref
        for edge in model.edges_of_kind(EdgeKind.IMPORTS)
        if edge.src_id == repo_module.id
    }

    assert "shop.billing.invoice.Invoice" in targets
    assert "shop.Invoice" not in targets


def test_import_edge_dst_id_resolves_to_a_real_node(sample_repo: Path) -> None:
    """A `from x import Y` edge must point at Y's actual node id, not a kind-less stub.

    Regression: the resolver only knows the bare name it resolved to, not the
    NodeKind it was declared with, so the id it builds must match what the
    defining module registered — otherwise every cross-module edge dangles and
    module-level analyses (health, communities, blast radius) see no coupling.
    """
    model = scan(sample_repo)
    node_ids = {n.id for n in model.nodes}
    invoice_class = next(n for n in model.nodes if n.qualname == "shop.billing.invoice.Invoice")

    import_edges = [
        edge
        for edge in model.edges_of_kind(EdgeKind.IMPORTS)
        if edge.dst_ref == "shop.billing.invoice.Invoice"
    ]
    assert import_edges
    for edge in import_edges:
        assert edge.dst_id == invoice_class.id
        assert edge.dst_id in node_ids


def test_type_only_edges_are_a_separate_kind(sample_repo: Path) -> None:
    model = scan(sample_repo)

    type_only = model.edges_of_kind(EdgeKind.IMPORTS_TYPE_ONLY)
    assert any(e.dst_ref == "shop.infra.repo.InvoiceRepo" for e in type_only)

    runtime_targets = {e.dst_ref for e in model.edges_of_kind(EdgeKind.IMPORTS)}
    assert "shop.infra.repo.InvoiceRepo" not in runtime_targets


def test_external_imports_are_unresolved_targets(sample_repo: Path) -> None:
    model = scan(sample_repo)

    httpx_edges = [e for e in model.edges if e.dst_ref == "httpx"]
    assert httpx_edges
    assert all(is_unresolved(e.dst_id) for e in httpx_edges)


def test_public_surface_respects_dunder_all(sample_repo: Path) -> None:
    model = scan(sample_repo)

    money = next(n for n in model.nodes if n.qualname == "shop.domain.Money")
    constant = next(n for n in model.nodes if n.qualname == "shop.domain.DEFAULT_CURRENCY")

    assert money.is_public is True
    # Declared in the module but absent from __all__: not part of the contract.
    assert constant.is_public is False


def test_inheritance_edges_resolve_through_imports(sample_repo: Path) -> None:
    model = scan(sample_repo)

    inherits = model.edges_of_kind(EdgeKind.INHERITS)
    assert all(e.confidence is not Confidence.INFERRED for e in inherits)


def test_blind_spots_are_collected_with_provenance(sample_repo: Path) -> None:
    model = scan(sample_repo)

    kinds = {b.kind for b in model.blind_spots}
    assert "dynamic_import" in kinds
    assert all(b.provenance.line > 0 for b in model.blind_spots)


def test_every_node_and_edge_carries_provenance(sample_repo: Path) -> None:
    model = scan(sample_repo)

    assert all(n.provenance.file and n.provenance.extractor for n in model.nodes)
    assert all(e.provenance.file and e.provenance.extractor for e in model.edges)


def test_ambiguous_rate_stays_within_the_phase_zero_gate(sample_repo: Path) -> None:
    model = scan(sample_repo)

    assert ambiguous_rate(model) <= AMBIGUOUS_RATE_GATE


def test_sample_is_reproducible_for_a_given_seed(sample_repo: Path) -> None:
    model = scan(sample_repo)

    first = sample_edges(model, count=10, seed=7)
    second = sample_edges(model, count=10, seed=7)

    assert [e.edge_id for e in first] == [e.edge_id for e in second]


def test_scan_is_deterministic(sample_repo: Path) -> None:
    first, second = scan(sample_repo), scan(sample_repo)

    assert (first.nodes, first.edges, first.blind_spots) == (second.nodes, second.edges, second.blind_spots)


def test_builtin_base_class_is_known_not_ambiguous(tmp_path: Path) -> None:
    """`class Boom(Exception)` is fully understood — it is simply outside the model."""
    (tmp_path / "errors.py").write_text("class Boom(Exception):\n    pass\n", encoding="utf-8")

    model = scan(tmp_path)

    edge = next(e for e in model.edges_of_kind(EdgeKind.INHERITS))
    assert edge.confidence is Confidence.EXTRACTED
    assert edge.dst_ref == "builtins.Exception"


def test_version_guarded_redefinitions_produce_no_duplicate_node_ids(tmp_path: Path) -> None:
    """The `certifi.core`/`click.core` shape: a function, a method, and a module-level constant each
    redefined in mutually exclusive branches within one file must not produce two nodes with one id."""
    (tmp_path / "core.py").write_text(
        "import sys\n"
        "\n"
        "if sys.version_info >= (3, 11):\n"
        "    _CACERT_PATH = None\n"
        "\n"
        "    def where():\n"
        "        return 'new'\n"
        "else:\n"
        "    _CACERT_PATH = 'x'\n"
        "\n"
        "    def where():\n"
        "        return 'old'\n"
        "\n"
        "\n"
        "class Context:\n"
        "    try:\n"
        "        def lookup_default(self):\n"
        "            return 1\n"
        "    except Exception:\n"
        "        def lookup_default(self):\n"
        "            return 2\n",
        encoding="utf-8",
    )

    model = scan(tmp_path)

    node_ids = [n.id for n in model.nodes]
    assert len(node_ids) == len(set(node_ids)), "duplicate node ids in extracted model"


def test_protocol_and_abc_bases_are_matched_by_name_not_by_substring(tmp_path: Path) -> None:
    (tmp_path / "shapes.py").write_text(
        "from abc import ABC, ABCMeta\nfrom typing import Protocol\n\n\n"
        "class ProtocolBase: ...\n\n\nclass Port(Protocol): ...\n\n\nclass Base(ABC): ...\n\n\n"
        "class Meta(ABCMeta): ...\n\n\nclass Child(ProtocolBase): ...\n\n\nclass Impl(Port): ...\n",
        encoding="utf-8",
    )

    model = scan(tmp_path)

    kinds = {e.src_id.partition(":")[2]: e.kind for e in model.edges if e.kind in (EdgeKind.INHERITS, EdgeKind.IMPLEMENTS)}
    assert kinds["shapes.Port"] is EdgeKind.IMPLEMENTS
    assert kinds["shapes.Base"] is EdgeKind.IMPLEMENTS
    assert kinds["shapes.Impl"] is EdgeKind.IMPLEMENTS  # a class that extends a Protocol class
    assert kinds["shapes.Meta"] is EdgeKind.INHERITS  # ABCMeta is a metaclass, not the ABC marker
    assert kinds["shapes.Child"] is EdgeKind.INHERITS  # a name that merely contains "Protocol"
