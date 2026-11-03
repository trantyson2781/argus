"""RESOLVE_V2: two extra call-resolution rules, each applied only when the source text proves the target.

Rule A (module alias): `mod.func()` where `mod` is an imported internal module that defines exactly one top-level
function `func`. Rule B (self method): `self.name()` inside a method of class C where `name` is defined exactly once,
directly on C. Anything else keeps the existing resolver's answer, or stays unresolved.

Both rules need to know which names are rebound locally, so a parameter or assignment that shadows an import is never
read as the import. Deliberately not done: inherited methods (needs MRO), types of variables, `cls.x()`, closures.

Three further guards only ever remove or withhold a link; they never add one:
  * a `self.x()` target that is a property-like descriptor (`@property`, `@cached_property`, `@x.setter`, ...) is not linked,
    because the call runs the *returned value*, not the decorated function;
  * a name that a package `__init__` rebinds (function, class, variable, or an import of something other than the
    submodule itself) is not read as that submodule, because `from pkg import name` may yield the rebinding; a
    re-export *from* that submodule (`from .name import name`), or an `__init__` that cannot be read, keeps today's
    module link but is never upgraded;
  * a call in a class body is not linked through a name the class body rebinds (class scope hides module names there,
    though not inside the class's methods).

On by default. `ARGUS_RESOLVE_V2=0` (or `false`, `no`, `off`) or `scan(root, resolve_v2=False)` selects the previous
resolver (V1) exactly; an explicit argument beats the environment. The call graph is the only thing that changes: nodes
and IMPORTS edges are identical in both modes. Outside this file the switch is read by the scanner (one call site), the
Explorer's `symbol` command and `argus scan`/`diff`, which only report which mode was used.
"""

from __future__ import annotations

import ast
import os
import warnings
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from argus.extract.facts import FileFacts, RawCall
from argus.extract.resolve import SymbolTable, Target
from argus.model.ids import node_id
from argus.model.schema import Confidence, NodeKind

ENV_VAR = "ARGUS_RESOLVE_V2"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})
METHOD_DECORATORS_BLOCKING_SELF = ("staticmethod", "classmethod")
DESCRIPTOR_DECORATORS = frozenset({"property", "cached_property", "abstractproperty", "setter", "getter", "deleter"})
SELF = "self"
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def resolve_v2_enabled(explicit: bool | None = None) -> bool:
    """True unless an explicit argument or ARGUS_RESOLVE_V2 turns V2 off. An unrecognised value keeps the default and warns."""
    if explicit is not None:
        return explicit
    raw = os.environ.get(ENV_VAR, "").strip().lower()
    if raw in _FALSY:
        return False
    if raw and raw not in _TRUTHY:
        warnings.warn(
            f"{ENV_VAR}={raw!r} is not recognised (use 0 to disable, 1 to enable); using the default, V2 on.",
            RuntimeWarning, stacklevel=2,
        )
    return True


@dataclass(frozen=True, slots=True)
class Refinement:
    """The outcome for one call: `target=None` means "no edge" (unresolved)."""

    target: Target | None
    dst_id: str | None = None
    attrs: dict[str, object] = field(default_factory=dict)


# -- scope analysis ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Func:
    class_local: str | None
    first_param: str | None
    decorators: tuple[str, ...]
    params: frozenset[str]
    hard: frozenset[str]      # assignments, loop/with/except targets, nested defs, lambda and comprehension names
    imports: frozenset[str]   # names bound by import statements inside the function


@dataclass(frozen=True, slots=True)
class _Class:
    methods: Counter[str]
    stored: frozenset[str]        # names assigned in the class body
    self_assigned: frozenset[str]  # `self.x = ...` anywhere in its methods
    descriptors: frozenset[str]   # methods decorated as property-like descriptors


@dataclass(frozen=True, slots=True)
class _ClassBody:
    """The lines where a class body (not one of its methods) executes, and every name that body binds."""

    start: int
    end: int
    names: frozenset[str]
    span_starts: tuple[int, ...]  # method and nested-function bodies inside the class, merged and sorted by start line
    span_ends: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Scopes:
    funcs: dict[str, _Func]
    classes: dict[str, _Class]
    module_hard: frozenset[str]
    module_imports: Counter[str]
    class_bodies: tuple[_ClassBody, ...] = ()


def _nested_defs(node: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """Function and class definitions directly inside `node`, not crossing into other definitions."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (*_DEFS, ast.ClassDef)):
            yield child
        else:
            yield from _nested_defs(child)


def _collect_bindings(nodes: list[ast.stmt] | list[ast.AST], hard: set[str], imports: Counter[str]) -> None:
    for node in nodes:
        _collect(node, hard, imports)


def _collect(node: ast.AST, hard: set[str], imports: Counter[str]) -> None:
    if isinstance(node, (*_DEFS, ast.ClassDef)):
        hard.add(node.name)
        return
    if isinstance(node, ast.Import):
        for alias in node.names:
            imports[alias.asname or alias.name.partition(".")[0]] += 1
        return
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name != "*":
                imports[alias.asname or alias.name] += 1
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        hard.add(node.id)
    elif isinstance(node, ast.arg):  # lambda parameters
        hard.add(node.arg)
    elif isinstance(node, ast.ExceptHandler) and node.name:
        hard.add(node.name)
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        hard.update(node.names)
    elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        hard.add(node.name)
    elif isinstance(node, ast.MatchMapping) and node.rest:
        hard.add(node.rest)
    for child in ast.iter_child_nodes(node):
        _collect(child, hard, imports)


def _param_names(args: ast.arguments) -> list[str]:
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    names += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
    return names


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    return tuple(ast.unparse(d) for d in node.decorator_list)


def _is_descriptor_decorator(decorator: ast.expr) -> bool:
    """`@property`, `@functools.cached_property`, `@name.setter`, `@abc.abstractproperty(...)` and the like."""
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    last = target.attr if isinstance(target, ast.Attribute) else target.id if isinstance(target, ast.Name) else ""
    return last in DESCRIPTOR_DECORATORS


def _class_bodies(tree: ast.Module) -> tuple[_ClassBody, ...]:
    bodies = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            hard: set[str] = set()
            imports: Counter[str] = Counter()
            _collect_bindings(node.body, hard, imports)
            spans = sorted((f.body[0].lineno, f.end_lineno or f.body[0].lineno) for f in ast.walk(node) if isinstance(f, _DEFS))
            merged: list[list[int]] = []
            for lo, hi in spans:
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            bodies.append(_ClassBody(
                node.lineno, node.end_lineno or node.lineno, frozenset(hard | set(imports)),
                tuple(lo for lo, _ in merged), tuple(hi for _, hi in merged),
            ))
    return tuple(bodies)


def _self_assigned(methods: list[ast.FunctionDef | ast.AsyncFunctionDef]) -> frozenset[str]:
    names: set[str] = set()
    for method in methods:
        first = method.args.posonlyargs[0].arg if method.args.posonlyargs else (method.args.args[0].arg if method.args.args else None)
        if first is None:
            continue
        for sub in ast.walk(method):
            if (
                isinstance(sub, ast.Attribute) and isinstance(sub.ctx, (ast.Store, ast.Del))
                and isinstance(sub.value, ast.Name) and sub.value.id == first
            ):
                names.add(sub.attr)
    return frozenset(names)


def _walk_definitions(parent: ast.AST, container: str | None, in_class: bool, funcs: dict[str, _Func], classes: dict[str, _Class]) -> None:
    for node in _nested_defs(parent):
        local = f"{container}.{node.name}" if container else node.name
        if isinstance(node, ast.ClassDef):
            direct = [d for d in _nested_defs(node) if isinstance(d, _DEFS)]
            descriptors = frozenset(d.name for d in direct if any(_is_descriptor_decorator(x) for x in d.decorator_list))
            classes[local] = _Class(Counter(d.name for d in direct), _assigned_in_class_body(node), _self_assigned(direct), descriptors)
            _walk_definitions(node, local, True, funcs, classes)
        else:
            hard: set[str] = set()
            imports: Counter[str] = Counter()
            _collect_bindings(node.body, hard, imports)
            params = _param_names(node.args)
            funcs[local] = _Func(
                class_local=container if in_class else None,
                first_param=params[0] if params else None,
                decorators=_decorator_names(node),
                params=frozenset(params),
                hard=frozenset(hard),
                imports=frozenset(imports),
            )
            _walk_definitions(node, local, False, funcs, classes)


def _assigned_in_class_body(class_node: ast.ClassDef) -> frozenset[str]:
    """Names the class body assigns (not merely defines as methods)."""
    names: set[str] = set()
    for stmt in class_node.body:
        if isinstance(stmt, (*_DEFS, ast.ClassDef)):
            continue
        names.update(n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)))
    return frozenset(names)


def analyze_scopes(tree: ast.Module) -> _Scopes:
    funcs: dict[str, _Func] = {}
    classes: dict[str, _Class] = {}
    _walk_definitions(tree, None, False, funcs, classes)
    hard: set[str] = set()
    imports: Counter[str] = Counter()
    _collect_bindings(tree.body, hard, imports)
    return _Scopes(funcs, classes, frozenset(hard), imports, _class_bodies(tree))


# -- the resolver --------------------------------------------------------------


class CallResolverV2:
    """Refines the existing resolver's answer for one call site."""

    def __init__(self, table: SymbolTable, facts: dict[str, FileFacts]) -> None:
        self._table = table
        self._facts = facts
        self._scopes: dict[str, _Scopes | None] = {}

    def refine(self, fact: FileFacts, call: RawCall, resolved: Target | None) -> Refinement:
        scopes = self._scopes_of(fact)
        if scopes is None:
            return Refinement(resolved)
        parts = call.callee.split(".")
        root = parts[0]
        chain = _chain(scopes, call.caller_local)
        if root == SELF:
            return self._self_method(fact, call, parts, scopes, chain) if resolved is None else Refinement(resolved)
        if root in _class_body_names(scopes, call.line):
            return Refinement(None)  # class scope hides module names for code that runs in the class body itself
        if any(root in f.params or root in f.hard for f in chain):
            return Refinement(None)  # a local binding hides whatever the file-wide import table says
        is_alias = self._table.alias(fact.qualname, root) is not None
        rebinding = self._package_rebinding(fact, root) if is_alias else None
        if rebinding == "other":
            return Refinement(None)  # `from pkg import name` may yield what pkg/__init__ bound to `name`, not the submodule
        if is_alias and root in scopes.module_hard and resolved is not None and resolved.is_internal:
            return Refinement(None)  # the module also rebinds the name; the import no longer says what it means
        if rebinding is not None:
            return Refinement(resolved)  # re-export or unanalysable __init__: today's link stays, but it is never upgraded
        if is_alias and len(parts) == 2 and not self._is_ambiguous_import(scopes, chain, root):
            upgraded = self._module_alias(fact, root, parts[1])
            if upgraded is not None:
                return upgraded
        return Refinement(resolved)

    # -- helpers -----------------------------------------------------------

    def _scopes_of(self, fact: FileFacts) -> _Scopes | None:
        if fact.qualname not in self._scopes:
            try:
                tree = ast.parse(Path(fact.path).read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError, ValueError):
                self._scopes[fact.qualname] = None
            else:
                self._scopes[fact.qualname] = analyze_scopes(tree)
        return self._scopes[fact.qualname]

    def _package_rebinding(self, fact: FileFacts, root: str) -> str | None:
        """How `pkg/__init__` rebinds the submodule name that `root` imports: None, "reexport", "unknown" or "other".

        "other": a function, class or variable defined in `__init__`, or an import of something else. Whether
        `from pkg import name` then yields the module or the rebinding depends on import order.
        "reexport": the package binds the name to an object imported from that very submodule (`from .name import name`).
        "unknown": `__init__` cannot be fully read (it failed to parse, or star-imports names we cannot see).
        A package that merely re-imports the submodule itself, or has no `__init__.py`, is not a rebinding.
        """
        named = self._imported_submodule(fact, root)
        if named is None:
            return None
        package, name = named
        if not self._table.is_module(f"{package}.{name}"):
            return None
        init = self._facts.get(package)
        if init is None:
            return "unknown" if self._table.is_package_init(package) else None
        scopes = self._scopes_of(init)
        if scopes is None:
            return "unknown"
        if name in scopes.module_hard or scopes.module_imports[name] > 1:
            return "other"
        bound = self._bound_by_import(init, name)
        if bound is not None and bound != f"{package}.{name}":
            return "reexport" if bound.startswith(f"{package}.{name}.") else "other"
        return "unknown" if self._table.has_star_import(package) else None  # a star import could rebind the name

    def _imported_submodule(self, fact: FileFacts, root: str) -> tuple[str, str] | None:
        """(package, name) when `root` is bound by an import that names `package.name`; None for imports that cannot name a submodule."""
        alias = self._table.alias(fact.qualname, root)
        raw = next((r for r in fact.imports if r.local_name == root), None)
        if alias is None or raw is None:
            return None
        if raw.module is not None or raw.level > 0:           # from pkg import name / from . import name
            return (alias.module_ref, raw.name) if raw.name != "*" else None
        package, _, name = raw.name.rpartition(".")           # import pkg.sub as alias (unaliased `import pkg.sub` binds `pkg`)
        return (package, name) if raw.asname and package else None

    def _bound_by_import(self, fact: FileFacts, name: str) -> str | None:
        """The dotted path of the object an import statement binds to `name` in `fact`, or None if no import binds it."""
        alias = self._table.alias(fact.qualname, name)
        raw = next((r for r in fact.imports if r.local_name == name), None)
        if alias is None or raw is None:
            return None
        if raw.module is None and raw.level > 0:               # from . import name
            return f"{alias.module_ref}.{raw.name}"
        return alias.module_ref if alias.is_module_import else f"{alias.module_ref}.{alias.name}"

    @staticmethod
    def _is_ambiguous_import(scopes: _Scopes, chain: list[_Func], root: str) -> bool:
        """The name is imported more than once, or re-imported inside a function: which import applies is unclear."""
        return scopes.module_imports[root] > 1 or any(root in f.imports for f in chain)

    def _module_alias(self, fact: FileFacts, root: str, name: str) -> Refinement | None:
        module = self._aliased_module(fact.qualname, root)
        if module is None:
            return None
        defining = self._unique_function_home(module, name)
        if defining is None:
            return None
        target = Target(
            ref=f"{defining}.{name}", module=defining, symbol=name, is_internal=True, confidence=Confidence.INFERRED,
            resolved_via="module_alias",
        )
        return Refinement(target, node_id(NodeKind.FUNCTION, f"{defining}.{name}"), {"resolved_via": "module_alias"})

    def _aliased_module(self, importing: str, root: str) -> str | None:
        alias = self._table.alias(importing, root)
        if alias is None:
            return None
        if alias.is_module_import:
            unaliased_dotted = "." in alias.module_ref and root == alias.module_ref.partition(".")[0]
            return alias.module_ref if not unaliased_dotted and self._table.is_module(alias.module_ref) else None
        target = self._table.canonicalize(alias.module_ref, alias.name)
        return target.module if target.is_internal and target.symbol is None else None

    def _unique_function_home(self, module: str, name: str) -> str | None:
        """The module that defines exactly one top-level function `name`, following one re-export if needed."""
        for _ in range(2):
            fact = self._facts.get(module)
            if fact is None:
                return None
            defs = [s for s in fact.symbols if s.parent_local is None and s.name == name]
            if defs:
                return module if len(defs) == 1 and defs[0].kind is NodeKind.FUNCTION else None
            alias = self._table.alias(module, name)
            if alias is None or alias.is_module_import:
                return None
            target = self._table.canonicalize(alias.module_ref, alias.name)
            if not (target.is_internal and target.module and target.symbol == name):
                return None
            module = target.module
        return None

    def _self_method(self, fact: FileFacts, call: RawCall, parts: list[str], scopes: _Scopes, chain: list[_Func]) -> Refinement:
        if len(parts) != 2 or call.caller_local is None or len(chain) != 1:
            return Refinement(None)
        func = scopes.funcs.get(call.caller_local)
        if func is None or func.class_local is None or func.first_param != SELF:
            return Refinement(None)
        if any(d.rpartition(".")[2] in METHOD_DECORATORS_BLOCKING_SELF for d in func.decorators) or SELF in func.hard:
            return Refinement(None)
        cls, name = scopes.classes.get(func.class_local), parts[1]
        if cls is None or cls.methods[name] != 1 or name in cls.stored or name in cls.self_assigned or name in cls.descriptors:
            return Refinement(None)
        symbol = f"{func.class_local}.{name}"
        target = Target(
            ref=f"{fact.qualname}.{symbol}", module=fact.qualname, symbol=symbol, is_internal=True,
            confidence=Confidence.INFERRED, resolved_via="self_method",
        )
        attrs = {"resolved_via": "self_method", "override_possible": True}
        return Refinement(target, node_id(NodeKind.METHOD, f"{fact.qualname}.{symbol}"), attrs)


def _class_body_names(scopes: _Scopes, line: int) -> frozenset[str]:
    """Names bound by the innermost class whose own body (not a method body) contains `line`; empty outside class bodies."""
    best: _ClassBody | None = None
    for body in scopes.class_bodies:
        if body.start <= line <= body.end and (best is None or body.start >= best.start) and not _in_function(body, line):
            best = body
    return best.names if best is not None else frozenset()


def _in_function(body: _ClassBody, line: int) -> bool:
    i = bisect_right(body.span_starts, line) - 1
    return i >= 0 and line <= body.span_ends[i]


def _chain(scopes: _Scopes, caller_local: str | None) -> list[_Func]:
    """The function scopes enclosing a call, innermost last."""
    if caller_local is None:
        return []
    parts = caller_local.split(".")
    found = (scopes.funcs.get(".".join(parts[: i + 1])) for i in range(len(parts)))
    return [f for f in found if f is not None]
