"""Breakage check, part B: does each call to a changed function still fit its new inputs?

Reads code and runs nothing. A call is checked by binding placeholder arguments to the new parameter list with
`inspect.Signature.bind`. Anything that cannot be decided from the text (`*args` at the call, an unknown decorator, a call
through the class that passes `self` by hand, a call the call site cannot be told apart on its line) is UNKNOWN, never guessed.
A verdict is an INFERENCE at best: the caller link itself is INFERRED, so a result is never a FACT and never "safe".
"""

from __future__ import annotations

import ast
import inspect
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from argus.explore.claims import Claim, Tier, evidence_at
from argus.explore.grouping import is_production_node
from argus.model.schema import Confidence, Edge, Node, NodeKind

_KINDS = {
    "positional-only": inspect.Parameter.POSITIONAL_ONLY,
    "positional": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "vararg": inspect.Parameter.VAR_POSITIONAL,
    "keyword-only": inspect.Parameter.KEYWORD_ONLY,
    "kwarg": inspect.Parameter.VAR_KEYWORD,
}
_FIRST_PARAMETER_KINDS = ("positional-only", "positional")
MAX_PARAMETERS = 255  # Python itself allows more, but binding a function with thousands of parameters per call site is a denial of service
RECEIVERS = ("self", "cls")


class Verdict(StrEnum):
    FITS = "FITS"
    BREAKS = "BREAKS"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class CallFit:
    """One call site of a function whose inputs changed, and whether the call still fits the new inputs."""

    caller: str
    callee: str
    file: str
    line: int
    verdict: Verdict
    reason: str
    claim: Claim
    already_broken: bool  # BREAKS under the new inputs and the same call also does not fit the old ones
    is_test: bool
    target_id: str  # id of the function whose inputs changed (a call to a class is checked against its `__init__`)


def fit_tier(confidence: Confidence, verdict: Verdict) -> Tier:
    """Never above INFERENCE: the caller link is a resolution, even when the scanner marks the edge EXTRACTED. UNKNOWN stays UNKNOWN."""
    return Tier.UNKNOWN if verdict is Verdict.UNKNOWN else Tier.INFERENCE


def fit_call(parameters: list[dict[str, Any]], call: ast.Call, drop_first: bool) -> tuple[Verdict, str]:
    """Whether `call` binds to `parameters` (the scanner's stored parameter list). `drop_first` skips `self` / `cls`."""
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
        return Verdict.UNKNOWN, "the call passes *args or **kwargs, so its arguments cannot be counted"
    if len(parameters) > MAX_PARAMETERS:
        return Verdict.UNKNOWN, f"the function has more than {MAX_PARAMETERS} parameters"
    if drop_first:
        if not parameters:
            return Verdict.UNKNOWN, "the method has no first parameter to receive the object"
        if parameters[0].get("kind") in _FIRST_PARAMETER_KINDS:
            parameters = parameters[1:]
    try:
        signature = inspect.Signature([
            inspect.Parameter(
                p["name"], _KINDS[p["kind"]], default=... if p["has_default"] else inspect.Parameter.empty,
            )
            for p in parameters
        ])
    except (KeyError, ValueError):
        return Verdict.UNKNOWN, "the parameter list could not be read"
    try:
        signature.bind(*[None] * len(call.args), **{kw.arg: None for kw in call.keywords if kw.arg})
    except TypeError as err:
        return Verdict.BREAKS, _reason(str(err), signature, call)
    return Verdict.FITS, "fits the inputs"


def _reason(message: str, signature: inspect.Signature, call: ast.Call) -> str:
    """The binding error, plus any keyword the function no longer has (`bind` reports a missing argument first)."""
    accepted = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return message
    unknown = [kw.arg for kw in call.keywords if kw.arg not in accepted or accepted[kw.arg].kind is inspect.Parameter.POSITIONAL_ONLY]
    if not unknown or "unexpected keyword" in message:
        return message
    return f"{message}; no parameter named {', '.join(repr(name) for name in unknown)}"


# -- finding the call and the inputs it must fit ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Target:
    """What a call must fit: the function whose inputs changed, its parameters before / after, whether the first one is the receiver,
    and any reason not to know."""

    target_id: str
    now: list[dict[str, Any]]
    was: list[dict[str, Any]] | None
    drop_first: bool
    unknown: str | None = None
    by_receiver: bool = False  # a plain method: only a call through `self` / `cls` shows that the object is passed for us


def _parameters(node: Node | None) -> list[dict[str, Any]] | None:
    payload = node.attrs.get("signature") if node else None
    return payload["parameters"] if payload else None


def _init_id(cls: Node) -> str:
    return f"{NodeKind.METHOD.value}:{cls.qualname}.__init__"


def _function_target(function: Node, base_nodes: dict[str, Node]) -> _Target:
    now = _parameters(function)
    if now is None:
        return _Target(function.id, [], None, False, unknown="the function has no stored signature")
    was = _parameters(base_nodes.get(function.id))
    decorators = tuple(function.attrs.get("decorators") or ())
    is_plain_method = function.kind is NodeKind.METHOD and function.name != "__init__"
    if is_plain_method and decorators in (("staticmethod",), ("classmethod",)):
        return _Target(function.id, now, was, drop_first=decorators == ("classmethod",))
    drop_first = function.kind is NodeKind.METHOD
    if decorators:
        return _Target(function.id, now, was, drop_first, unknown=f"the function has a decorator ({', '.join(decorators)}) that may change its inputs")
    return _Target(function.id, now, was, drop_first, by_receiver=is_plain_method)


def _inherited_init(cls: Node, parents: Mapping[str, tuple[str, ...]], head_nodes: dict[str, Node]) -> Node | None:
    """The `__init__` a class without its own would use, found by walking its bases breadth first (not Python's full lookup order)."""
    queue, seen = deque(parents.get(cls.id, ())), {cls.id}
    while queue:
        base_id = queue.popleft()
        if base_id in seen or base_id not in head_nodes:
            continue
        seen.add(base_id)
        init = head_nodes.get(_init_id(head_nodes[base_id]))
        if init is not None:
            return init
        queue.extend(parents.get(base_id, ()))
    return None


def _class_target(cls: Node, base_nodes: dict[str, Node], head_nodes: dict[str, Node], parents: Mapping[str, tuple[str, ...]], changed_ids: frozenset[str]) -> _Target | None:
    init = head_nodes.get(_init_id(cls))
    if init is not None:
        if init.id not in changed_ids:
            return None
        target = _function_target(init, base_nodes)
        if f"{NodeKind.METHOD.value}:{cls.qualname}.__new__" in head_nodes and target.unknown is None:
            return replace(target, unknown="the class also defines __new__, which can change what a call must pass")
        return target
    inherited = _inherited_init(cls, parents, head_nodes)
    if inherited is None or inherited.id not in changed_ids:
        return None
    return _Target(inherited.id, [], None, True, unknown=f"{cls.name} inherits __init__ from a class whose __init__ changed; subclass constructors are not followed")


def _target(dst: Node, base_nodes: dict[str, Node], head_nodes: dict[str, Node], parents: Mapping[str, tuple[str, ...]], changed_ids: frozenset[str]) -> _Target | None:
    """The inputs a call to `dst` must fit, or None when its inputs did not change or `dst` is not something this check reads."""
    if dst.kind is NodeKind.CLASS:
        return _class_target(dst, base_nodes, head_nodes, parents, changed_ids)
    if dst.kind in (NodeKind.FUNCTION, NodeKind.METHOD) and dst.id in changed_ids:
        return _function_target(dst, base_nodes)
    return None


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _CallSiteReader:
    """The `ast.Call` at a line of a head file; one parse per file."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._calls: dict[str, dict[int, list[ast.Call]] | str] = {}

    def at(self, file: str, line: int, name: str) -> ast.Call | str:
        """The one call to `name` on `line`, or the reason it cannot be told which call is meant."""
        parsed = self._calls.get(file)
        if parsed is None:
            parsed = self._parse(file)
            self._calls[file] = parsed
        if isinstance(parsed, str):
            return parsed
        found = [c for c in parsed.get(line, []) if _called_name(c.func) == name]
        if len(found) == 1:
            return found[0]
        return "no call to that name was found on the line" if not found else "the line calls that name more than once"

    def _parse(self, file: str) -> dict[int, list[ast.Call]] | str:
        try:
            tree = ast.parse((self._root / file).read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError, RecursionError, MemoryError) as err:
            return f"the file could not be read ({type(err).__name__})"
        by_line: dict[int, list[ast.Call]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                by_line.setdefault(node.lineno, []).append(node)
        return by_line


def check_signature_changes(
    edges: Iterable[Edge], base_nodes: dict[str, Node], head_nodes: dict[str, Node], changed_ids: frozenset[str], head_root: Path,
    parents: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[CallFit, ...]:
    """Check every head-side call edge whose callee changed its inputs.

    `changed_ids` holds the ids of functions and methods with SIGNATURE_CHANGED; a call to a class is checked against that class's
    `__init__`, and a call to a subclass that inherits a changed `__init__` is reported UNKNOWN (`parents`: class id -> base class ids).
    `edges` are the head side's resolved CALLS edges, already stripped of files that did not parse.
    """
    reader = _CallSiteReader(head_root)
    seen: set[tuple[str, str, str, int]] = set()
    found: list[CallFit] = []
    for edge in edges:
        src, dst = head_nodes.get(edge.src_id), head_nodes.get(edge.dst_id)
        if src is None or dst is None:
            continue
        target = _target(dst, base_nodes, head_nodes, parents or {}, changed_ids)
        key = (edge.src_id, edge.dst_id, edge.provenance.file, edge.provenance.line)
        if target is None or key in seen:
            continue
        seen.add(key)
        found.append(_one_fit(edge, src, dst, target, reader, head_root))
    return tuple(sorted(found, key=lambda f: (f.file, f.line, f.caller, f.callee)))


def _through_receiver(site: ast.Call) -> bool:
    func = site.func
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in RECEIVERS


def _one_fit(edge: Edge, src: Node, dst: Node, target: _Target, reader: _CallSiteReader, head_root: Path) -> CallFit:
    file, line = edge.provenance.file, edge.provenance.line
    site = reader.at(file, line, dst.name)
    verdict, reason, broken_before = Verdict.UNKNOWN, "", False
    if isinstance(site, str):
        reason = site
    elif target.unknown:
        reason = target.unknown
    elif target.by_receiver and not _through_receiver(site):
        reason = "the call is not made through self or cls, so it cannot be told whether the object is passed by hand"
    else:
        verdict, reason = fit_call(target.now, site, target.drop_first)
        if verdict is Verdict.BREAKS and target.was is not None:
            broken_before = fit_call(target.was, site, target.drop_first)[0] is Verdict.BREAKS
    claim = Claim(f"{src.qualname} calls {dst.qualname}: {reason}", fit_tier(edge.confidence, verdict), (evidence_at(head_root, file, line),))
    return CallFit(src.qualname, dst.qualname, file, line, verdict, reason, claim, broken_before, not is_production_node(src), target.target_id)
