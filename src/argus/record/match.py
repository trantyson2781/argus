"""Match a recorded function to a node of the scanned model.

A recorded function carries the file, the first line of its code object (the first decorator's line for a decorated function) and
`co_qualname`. It matches a node only when the file is scanned, the qualified name agrees (with `.<locals>` removed and the scanner's
`#2` suffix for a redefinition ignored) and the node is the first symbol the scanner found at or after that line. Anything else stays
unmatched with a reason, so a wrong link is never made to fill a gap.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from argus.model.schema import ArchModel, Node, NodeKind

REDEFINITION = re.compile(r"#\d+")
MODULE_CODE = "<module>"
SCOPE_KINDS = (NodeKind.MODULE, NodeKind.PACKAGE)


class Match(NamedTuple):
    node_id: str | None
    reason: str | None  # why there is no node: "file_not_scanned", "anonymous", "no_node" or "ambiguous"


def _anonymous(qualname: str) -> bool:
    return any(part.startswith("<") for part in qualname.split("."))


def match_functions(model: ArchModel, functions: Iterable[Mapping[str, Any]]) -> dict[str, Match]:
    """Key of each recorded function -> its match. `functions` need `key`, `file`, `first_line` and `code_qualname`."""
    modules = {n.provenance.file: n for n in model.nodes if n.kind is NodeKind.MODULE}
    lines: defaultdict[str, list[int]] = defaultdict(list)
    at: defaultdict[tuple[str, int, str], list[Node]] = defaultdict(list)
    for node in model.nodes:
        if node.kind in SCOPE_KINDS:
            continue
        lines[node.provenance.file].append(node.provenance.line)
        at[(node.provenance.file, node.provenance.line, REDEFINITION.sub("", node.qualname))].append(node)
    for found in lines.values():
        found.sort()

    return {fn["key"]: _match(fn, modules, lines, at) for fn in functions}


def _match(
    fn: Mapping[str, Any], modules: Mapping[str, Node], lines: Mapping[str, list[int]], at: Mapping[tuple[str, int, str], list[Node]]
) -> Match:
    file, qualname = fn["file"], fn["code_qualname"]
    module = modules.get(file)
    if module is None:
        return Match(None, "file_not_scanned")
    if qualname == MODULE_CODE:
        return Match(module.id, None)
    local = qualname.replace(".<locals>", "")
    if _anonymous(local):
        return Match(None, "anonymous")
    positions = lines.get(file, [])
    index = bisect_left(positions, fn["first_line"])
    if index == len(positions):
        return Match(None, "no_node")
    found = at.get((file, positions[index], f"{module.qualname}.{local}"), [])
    if not found:
        return Match(None, "no_node")
    return Match(found[0].id, None) if len(found) == 1 else Match(None, "ambiguous")
