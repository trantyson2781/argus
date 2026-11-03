"""How modules are grouped into components, and which module owns a node."""

from __future__ import annotations

from argus.analysis.code_type import classify_file, is_production
from argus.model.schema import ArchModel, EdgeKind, Node, NodeKind

GROUP_DEPTH = 2
MODULE_KINDS = (NodeKind.MODULE, NodeKind.PACKAGE)
IMPORT_KINDS = (EdgeKind.IMPORTS, EdgeKind.IMPORTS_TYPE_ONLY)


def default_group(module: str) -> str:
    """`pkg.sub.mod` -> `pkg.sub`; a one- or two-segment name is its own component."""
    return ".".join(module.split(".")[:GROUP_DEPTH])


def node_index(model: ArchModel) -> dict[str, Node]:
    return {node.id: node for node in model.nodes}


def owner_module(node: Node, nodes: dict[str, Node]) -> str | None:
    """Qualname of the module or package containing `node` (itself, if it is one)."""
    current: Node | None = node
    while current is not None:
        if current.kind in MODULE_KINDS:
            return current.qualname
        current = nodes.get(current.parent_id) if current.parent_id else None
    return None


def is_test_function(node: Node) -> bool:
    """A test: a function in test code whose name starts with `test`. Helpers and fixtures in test files are not tests."""
    return not is_production_node(node) and node.kind not in MODULE_KINDS and node.name.startswith("test")


def is_production_node(node: Node) -> bool:
    file = node.provenance.file
    if not file.endswith(".py"):  # namespace package: the "file" is a directory
        file = f"{file}/__init__.py"
    return is_production(classify_file(file))
