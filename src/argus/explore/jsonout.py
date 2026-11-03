"""JSON output for `argus explore diff --json`: one complete document built from the same `DiffView` the text report prints.

The document is the data layer for watch mode, a terminal UI and agent hooks. `Claim.text` is prose for people; consumers should
read the structured fields (`qualname`, `flags`, `callee`, `evidence`). Nothing here is capped or shortened: the display caps of the
text report do not apply. Every path in the data is repository-relative, and `ensure_ascii` keeps control characters out of the raw output.

Failures: a known failure is one error object on stdout (exit 2, or 1 for an internal error). Option errors rejected by the CLI library
itself (for example `--limit 0`) are plain text on stderr with exit 2 and an empty stdout, so check the exit code. An error `message` can
echo the path or ref you typed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from argus import __version__
from argus.explore.call_fit import CallFit
from argus.explore.changes import INTERFACE_LABEL, CallerEdge, CallerGroup, DiffView, FileKind, Flag, SymbolChange
from argus.explore.claims import Claim, Evidence
from argus.explore.entrypoints import Command, Entrypoints
from argus.explore.reach import CallStep
from argus.explore.render_diff import summary_counts
from argus.explore.symbols import CallRow, SymbolView
from argus.explore.views import DetailView
from argus.record.observed import ObservedFacts

SCHEMA_VERSION = 1
Side = Literal["base", "head", "current"]  # "current": the one scan of a command that compares nothing
FLAG_ORDER = tuple(str(f) for f in Flag)
BASE_FILE_KINDS = frozenset({"removed", "removed_empty"})
HEAD_FILE_KINDS = frozenset({"added", "added_empty", "outside_symbols", "formatting_only"})


def claim_json(claim: Claim, side: Side | Mapping[Evidence, Side]) -> dict[str, Any]:
    """A claim with its evidence. `side` says which scanned state the evidence was read from: one side for the whole claim, or a
    map when the claim mixes both (a function that gained one call and lost another). A piece of evidence missing from the map is a bug."""
    def side_of(evidence: Evidence) -> Side:
        return side if isinstance(side, str) else side[evidence]

    return {
        "tier": claim.tier.value,
        "text": claim.text,
        "evidence": [{"file": e.file, "line": e.line, "source_hash": e.source_hash, "side": side_of(e)} for e in claim.evidence],
    }


def callee_flags(label: str) -> list[str]:
    """Split a caller edge's display label ("REMOVED" or "SIGNATURE_CHANGED, BODY_CHANGED") into names. Anything else is a bug upstream."""
    parts = label.split(", ")
    for part in parts:
        if part != INTERFACE_LABEL and part not in FLAG_ORDER:
            raise ValueError(f"unknown callee label {part!r} in {label!r}")
    return parts


def file_change_side(kind: FileKind, file: str, new_files: frozenset[str]) -> Side:
    """Which side a file-level claim reads from. A relocated file was added (its path is on the head side only) or removed."""
    if kind in BASE_FILE_KINDS:
        return "base"
    if kind in HEAD_FILE_KINDS:
        return "head"
    if kind == "relocated":
        return "head" if file in new_files else "base"
    raise ValueError(f"unknown file change kind {kind!r}")


def _symbol(change: SymbolChange, call_sides: Mapping[Evidence, Side]) -> dict[str, Any]:
    side: Side = "base" if change.change == "removed" else "head"
    return {
        "qualname": change.qualname, "kind": change.kind, "change": change.change,
        "flags": [f for f in FLAG_ORDER if Flag(f) in change.flags],
        "file": change.file, "line": change.line, "untracked": change.untracked,
        "interface_was": change.interface_was, "interface_now": change.interface_now, "docstring_only": change.docstring_only,
        "known_callers": change.known_callers,
        "claim": claim_json(change.claim, side) if change.claim else None,
        "calls_claim": claim_json(change.calls_claim, call_sides) if change.calls_claim else None,
    }


def _edge(edge: CallerEdge) -> dict[str, Any]:
    return {
        "callee": edge.callee, "callee_flags": callee_flags(edge.callee_label), "was": edge.was, "confidence": edge.confidence,
        "override_possible": edge.override_possible, "claim": claim_json(edge.claim, "base" if edge.was else "head"),
    }


def _call_fit(fit: CallFit) -> dict[str, Any]:
    return {
        "caller": fit.caller, "callee": fit.callee, "file": fit.file, "line": fit.line, "verdict": fit.verdict.value, "reason": fit.reason,
        "already_broken": fit.already_broken, "is_test": fit.is_test, "claim": claim_json(fit.claim, "head"),
    }


def _group(group: CallerGroup) -> dict[str, Any]:
    return {
        "caller": group.caller, "module_level": group.module_level, "caller_state": group.caller_state, "is_test": group.is_test,
        "edges": [_edge(e) for e in group.edges],
    }


def diff_json(view: DiffView) -> dict[str, Any]:
    """The `data` part of the document. Lists are complete and keep the view's deterministic order."""
    # A calls_claim cites the same evidence objects as the rows of `calls`, and every row knows its side.
    call_sides: dict[Evidence, Side] = {c.claim.evidence[0]: "base" if c.change == "removed" else "head" for c in view.calls}
    return {
        "base_label": view.base_label, "head_label": view.head_label,
        "summary": summary_counts(view),
        "files_changed": view.files_changed, "untracked_files": view.untracked_files,
        "symbols": [_symbol(c, call_sides) for c in view.symbols],
        "calls": [
            {"change": c.change, "source": c.source, "target": c.target, "claim": claim_json(c.claim, "base" if c.change == "removed" else "head")}
            for c in view.calls
        ],
        "interface_callers": [_group(g) for g in view.interface_callers],
        "affected_callers": [_group(g) for g in view.affected_callers],
        "hidden_test_callers": view.hidden_test_callers,
        "call_fits": [_call_fit(f) for f in view.call_fits],
        "imports": [
            {"change": i.change, "source": i.source, "target": i.target, "claim": claim_json(i.claim, "base" if i.change == "removed" else "head")}
            for i in view.imports
        ],
        "file_changes": [
            {"file": f.file, "kind": f.kind, "claim": claim_json(f.claim, file_change_side(f.kind, f.file, view.new_files))}
            for f in view.file_changes
        ],
        "renames": [{"old": r.old, "new": r.new, "symbols": r.symbols, "claim": claim_json(r.claim, "head")} for r in view.renames],
        "not_scanned": [{"prefix": n.prefix, "files": n.files} for n in view.not_scanned],
        "new_files": sorted(view.new_files),
        "not_determinable": [claim_json(c, "head") for c in view.not_determinable],
    }


def envelope(command: str, data: dict[str, Any], *, resolve_v2: bool, options: dict[str, Any]) -> dict[str, Any]:
    """The document around a command's data: schema version, tool version and the options the data was built with."""
    return {
        "schema_version": SCHEMA_VERSION, "command": command, "argus_version": __version__,
        "resolver": "calls-v2" if resolve_v2 else "calls-v1", "options": options, "ok": True, "data": data,
    }


def error_json(command: str, kind: str, message: str) -> dict[str, Any]:
    """The document for a failure. `kind` is a stable machine-readable name; `message` is for people."""
    return {"schema_version": SCHEMA_VERSION, "command": command, "ok": False, "error": {"kind": kind, "message": message}}


def dumps(doc: dict[str, Any]) -> str:
    """Pretty, ASCII-only JSON. DEL is escaped too, which `ensure_ascii` leaves raw."""
    return json.dumps(doc, indent=2, ensure_ascii=True).replace("\x7f", "\\u007f")


# -- explore symbol -----------------------------------------------------------------------------------------------------------------


def _call_row(row: CallRow) -> dict[str, Any]:
    return {
        "name": row.name, "kind": row.kind, "module_level": row.module_level, "module_target": row.module_target,
        "override_possible": row.override_possible, "confidence": row.confidence, "claim": claim_json(row.claim, "current"),
    }


CALLS_SCOPE_FULL = ("function", "method")  # callers, callees and unresolved calls are all computed
CALLS_SCOPE_CALLERS_ONLY = ("class",)  # callers (instantiations) only; the calls inside a class belong to its methods


def observed_json(facts: ObservedFacts | None) -> dict[str, Any] | None:
    """OBSERVED facts from the latest recording. They are not a claim tier: `label` says what they are, `stale_files` and `incomplete` say how far to trust them."""
    if facts is None:
        return None
    return {
        "label": "OBSERVED", "recorded_at": facts.recorded_at, "command": facts.command, "calls": facts.calls,
        "callers": [{"name": c.name, "count": c.count, "lines": list(c.lines), "also": c.also} for c in facts.callers],
        "outside_calls": facts.outside_calls, "tests": list(facts.tests), "stale_files": list(facts.stale_files), "incomplete": list(facts.incomplete),
    }


def symbol_json(view: SymbolView) -> dict[str, Any]:
    """The `data` part of `explore symbol --json`. Nothing is capped; the text report's `--limit` does not apply.

    A section that was not computed for this kind of symbol is null, never an empty list: `calls_scope` says which are real
    ("full" for a function or method, "callers_only" for a class, "none" for a module or package).
    """
    node = view.node
    kind = node.kind.value
    scope = "full" if kind in CALLS_SCOPE_FULL else "callers_only" if kind in CALLS_SCOPE_CALLERS_ONLY else "none"
    has_callers, has_calls = scope != "none", scope == "full"
    return {
        "node": {"qualname": node.qualname, "name": node.name, "kind": kind, "file": node.provenance.file, "line": node.provenance.line},
        "definition": claim_json(view.definition, "current"), "signature": view.signature, "decorators": list(view.decorators),
        "purpose": claim_json(view.purpose, "current"),
        "members": [{"kind": m.kind, "qualname": m.qualname, "claim": claim_json(m.claim, "current")} for m in view.member_rows],
        "calls_scope": scope,
        "lines": view.lines,
        "reached_from": (
            [{"command": r.command, "path": list(r.chain), "claim": claim_json(r.claim, "current")} for r in view.reached_from]
            if view.reached_from is not None else None
        ),
        "reach_caveat": view.reach_caveat,
        "tested_by": list(view.tested_by) if view.tested_by is not None else None,
        "observed": observed_json(view.observed), "observed_error": view.observed_error,
        "callers": [_call_row(r) for r in view.caller_rows] if has_callers else None,
        "hidden_test_callers": view.hidden_test_callers if has_callers else None,
        "callees": [_call_row(r) for r in view.callee_rows] if has_calls else None,
        "external_calls": list(view.external_calls) if has_calls else None,
        "builtin_call_count": view.builtin_call_count if has_calls else None,
        "unresolved_calls": (
            [{"callee": u.callee, "reason": u.reason, "line": u.line, "claim": claim_json(u.claim, "current")} for u in view.unresolved_rows]
            if has_calls else None
        ),
    }


# -- explore entrypoints and explore component ---------------------------------------------------------------------------------------


def _qualname(node_id: str) -> str:
    """A node id is `kind:qualname`; the qualname is what a consumer can use."""
    kind, sep, qualname = node_id.partition(":")
    if not sep or not qualname:
        raise ValueError(f"not a node id: {node_id!r}")
    return qualname


def _command(command: Command) -> dict[str, Any]:
    return {
        "display": command.display, "path": list(command.path), "function": _qualname(command.function_id), "module": command.module,
        "file": command.file, "line": command.line, "claim": claim_json(command.claim, "current"),
    }


def _step(step: CallStep) -> dict[str, Any]:
    return {
        "depth": step.depth, "caller": _qualname(step.caller_id), "callee": step.callee, "kind": step.kind, "module": step.module,
        "group": step.group, "claim": claim_json(step.claim, "current"),
    }


def entrypoints_json(eps: Entrypoints, traces: list[tuple[Command, tuple[CallStep, ...]]] | None) -> dict[str, Any]:
    """The `data` part of `explore entrypoints --json`. `traces` is empty unless `--command` was given."""
    return {
        "scripts": [
            {"name": x.name, "module": x.module, "attr": x.attr, "claim": claim_json(x.claim, "current")} for x in eps.scripts
        ],
        "commands": [_command(c) for c in eps.commands],
        "main_guards": [claim_json(c, "current") for c in eps.main_guards],
        "unknowns": [claim_json(c, "current") for c in eps.unknowns],
        "traces": [{"command": _command(c), "steps": [_step(s) for s in steps]} for c, steps in traces or []],
    }


def component_json(view: DetailView) -> dict[str, Any]:
    """The `data` part of `explore component --json`."""
    comp = view.component
    return {
        "label": view.label, "modules": list(comp.modules), "purpose": claim_json(comp.purpose, "current"),
        "imports_out": [{"other": d.other, "count": d.count, "claim": claim_json(d.claim, "current")} for d in comp.imports_out],
        "imported_by": [{"other": d.other, "count": d.count, "claim": claim_json(d.claim, "current")} for d in comp.imported_by],
        "symbols": [
            {"module": x.module, "name": x.name, "kind": x.kind, "claim": claim_json(x.claim, "current")} for x in view.symbols
        ],
        "own_commands": [_command(c) for c in view.own_commands],
        "reached_from": [{"command": _command(r.command), "chains": [[_step(s) for s in chain] for chain in r.chains]} for r in view.reached_from],
        "unknowns": [claim_json(c, "current") for c in view.unknowns],
    }
