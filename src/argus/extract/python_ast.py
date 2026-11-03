"""Parse one Python file into `FileFacts` using the stdlib `ast`.

Chosen over tree-sitter deliberately: for a Python-only tool, `ast` is exact,
version-matched to the interpreter and zero-dependency (PYTHON-EXTRACTION §1).

The walk tracks a context stack so an import can be classified by *where* it
appears — under `if TYPE_CHECKING`, inside a function body, or guarded by
`try/except ImportError` — because those are three different kinds of coupling.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path

from argus.extract.facts import (
    FileFacts,
    RawBlindSpot,
    RawCall,
    RawImport,
    RawParameter,
    RawProtocolImpl,
    RawSignature,
    RawSymbol,
)
from argus.model.schema import ImportFlag, NodeKind

EXTRACTOR = "python_ast"

DYNAMIC_IMPORT_FUNCS = frozenset({"import_module", "__import__"})
DYNAMIC_ATTR_FUNCS = frozenset({"getattr", "setattr", "delattr"})
EVAL_FUNCS = frozenset({"eval", "exec", "compile"})


@dataclass(frozen=True, slots=True)
class _Context:
    """Where in the file's structure the walker currently is."""

    type_checking: bool = False
    in_function: bool = False
    in_try_import_guard: bool = False
    in_conditional: bool = False
    enclosing: str | None = None      # local qualname of nearest function
    container: str | None = None      # local qualname of nearest class/function
    in_class: bool = False

    def import_flags(self) -> frozenset[ImportFlag]:
        flags: set[ImportFlag] = set()
        if self.in_function:
            flags.add(ImportFlag.LAZY)
        if self.in_try_import_guard:
            flags.add(ImportFlag.OPTIONAL)
        if self.in_conditional and not self.type_checking:
            flags.add(ImportFlag.CONDITIONAL)
        return frozenset(flags)


def _unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive; unparse is total in 3.12+
        return None


def _is_type_checking_test(test: ast.expr) -> bool:
    """Match `TYPE_CHECKING`, `typing.TYPE_CHECKING` and their negations."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return False
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _guards_import_error(handlers: list[ast.ExceptHandler]) -> bool:
    for handler in handlers:
        names = handler.type
        if names is None:
            return True
        candidates = names.elts if isinstance(names, ast.Tuple) else [names]
        for candidate in candidates:
            text = _unparse(candidate) or ""
            if "ImportError" in text or "ModuleNotFoundError" in text:
                return True
    return False


def _signature_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> RawSignature:
    args = node.args
    parameters: list[RawParameter] = []

    def add(arg: ast.arg, kind: str, default: ast.expr | None, has_default: bool) -> None:
        parameters.append(
            RawParameter(
                name=arg.arg,
                kind=kind,
                annotation=_unparse(arg.annotation),
                default=_unparse(default),
                has_default=has_default,
            )
        )

    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    pad = len(positional) - len(defaults)
    for index, arg in enumerate(positional):
        kind = "positional-only" if index < len(args.posonlyargs) else "positional"
        default = defaults[index - pad] if index >= pad else None
        add(arg, kind, default, index >= pad)

    if args.vararg:
        add(args.vararg, "vararg", None, False)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        add(arg, "keyword-only", default, default is not None)
    if args.kwarg:
        add(args.kwarg, "kwarg", None, False)

    return RawSignature(
        parameters=tuple(parameters),
        returns=_unparse(node.returns),
        is_async=isinstance(node, ast.AsyncFunctionDef),
    )


def _pair_unpacking(
    targets: list[ast.expr], value: ast.expr | None
) -> list[tuple[ast.expr, ast.expr | None]]:
    """Line up unpacking targets with their values, where that is knowable."""
    if (
        isinstance(value, ast.Tuple | ast.List)
        and len(value.elts) == len(targets)
        and not any(isinstance(t, ast.Starred) for t in targets)
    ):
        return list(zip(targets, value.elts, strict=True))
    return [(target, None) for target in targets]


def _string_elements(node: ast.expr) -> tuple[str, ...] | None:
    """Extract a tuple/list of string literals, or None if it isn't one."""
    if not isinstance(node, ast.List | ast.Tuple | ast.Set):
        return None
    names: list[str] = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            names.append(element.value)
        else:
            return None
    return tuple(names)


class _FileVisitor(ast.NodeVisitor):
    """Single pass over one module's AST."""

    def __init__(self, qualname: str, path: str) -> None:
        self.qualname = qualname
        self.path = path
        self.context = _Context()
        self.imports: list[RawImport] = []
        self.symbols: list[RawSymbol] = []
        self.calls: list[RawCall] = []
        self.blind_spots: list[RawBlindSpot] = []
        self.protocol_impls: list[RawProtocolImpl] = []
        self.dunder_all: tuple[str, ...] | None = None
        self.dunder_all_line: int | None = None

    # -- context helpers -------------------------------------------------

    def _descend(self, **changes: object) -> _Context:
        previous = self.context
        self.context = replace(previous, **changes)  # type: ignore[arg-type]
        return previous

    def _restore(self, previous: _Context) -> None:
        self.context = previous

    def _local_qualname(self, name: str) -> str:
        return f"{self.context.container}.{name}" if self.context.container else name

    # -- imports ---------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        flags = self.context.import_flags()
        for alias in node.names:
            self.imports.append(
                RawImport(
                    module=None,
                    name=alias.name,
                    asname=alias.asname,
                    level=0,
                    line=node.lineno,
                    type_only=self.context.type_checking,
                    flags=flags,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        flags = set(self.context.import_flags())
        if node.level:
            flags.add(ImportFlag.RELATIVE)
        for alias in node.names:
            alias_flags = set(flags)
            if alias.name == "*":
                alias_flags.add(ImportFlag.STAR)
                self.blind_spots.append(
                    RawBlindSpot(
                        kind="star_import",
                        detail=f"from {'.' * node.level}{node.module or ''} import *",
                        line=node.lineno,
                    )
                )
            self.imports.append(
                RawImport(
                    module=node.module,
                    name=alias.name,
                    asname=alias.asname,
                    level=node.level,
                    line=node.lineno,
                    type_only=self.context.type_checking,
                    flags=frozenset(alias_flags),
                )
            )

    # -- definitions -----------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        local = self._local_qualname(node.name)
        bases_text = tuple(_unparse(b) or "" for b in node.bases)
        self.symbols.append(
            RawSymbol(
                kind=NodeKind.CLASS,
                name=node.name,
                local_qualname=local,
                line=node.lineno,
                end_line=node.end_lineno,
                parent_local=self.context.container,
                decorators=tuple(d for d in map(_unparse, node.decorator_list) if d),
                bases=tuple(b for b in map(_unparse, node.bases) if b)
                + tuple(f"{k.arg}={_unparse(k.value)}" for k in node.keywords if k.arg),
                docstring=ast.get_docstring(node),
            )
        )

        for base in bases_text:
            if base in ("Protocol", "typing.Protocol", "typing_extensions.Protocol", "ABC", "abc.ABC") or "Protocol[" in base:
                self.protocol_impls.append(RawProtocolImpl(class_local=local, protocol_name=base))

        previous = self._descend(container=local, in_class=True, in_conditional=False)
        self.generic_visit(node)
        self._restore(previous)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        local = self._local_qualname(node.name)
        kind = NodeKind.METHOD if self.context.in_class else NodeKind.FUNCTION
        signature = _signature_of(node)
        self.symbols.append(
            RawSymbol(
                kind=kind,
                name=node.name,
                local_qualname=local,
                line=node.lineno,
                end_line=node.end_lineno,
                parent_local=self.context.container,
                decorators=tuple(d for d in map(_unparse, node.decorator_list) if d),
                signature=signature,
                docstring=ast.get_docstring(node),
            )
        )

        previous = self._descend(
            container=local, enclosing=local, in_function=True, in_class=False
        )
        self.generic_visit(node)
        self._restore(previous)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- module/class level data -----------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_target(target, node.lineno, node.value, node.end_lineno)
        self.generic_visit(node)

    def _record_target(
        self, target: ast.expr, line: int, value: ast.expr | None, end_line: int | None
    ) -> None:
        """Bind one assignment target, destructuring tuple/list unpacking.

        `A, B = "a", "b"` defines two module-level names. Handling only bare
        `Name` targets silently drops them, which both loses the symbols and
        makes every module importing them look unresolvable.
        """
        if isinstance(target, ast.Name):
            self._record_binding(target.id, line, value, end_line)
            return
        if isinstance(target, ast.Starred):
            self._record_target(target.value, line, None, end_line)
            return
        if not isinstance(target, ast.Tuple | ast.List):
            return  # attribute or subscript target: not a module-level name

        paired = _pair_unpacking(target.elts, value)
        for element, element_value in paired:
            self._record_target(element, line, element_value, end_line)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._record_binding(
                node.target.id,
                node.lineno,
                node.value,
                node.end_lineno,
                annotation=_unparse(node.annotation),
            )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # `__all__ += [...]` is a common re-export idiom in package inits.
        if (
            isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and (extra := _string_elements(node.value)) is not None
        ):
            self.dunder_all = (self.dunder_all or ()) + extra
            self.dunder_all_line = self.dunder_all_line or node.lineno
        self.generic_visit(node)

    def _record_binding(
        self,
        name: str,
        line: int,
        value: ast.expr | None,
        end_line: int | None,
        annotation: str | None = None,
    ) -> None:
        if name == "__all__" and value is not None and not self.context.in_function:
            elements = _string_elements(value)
            if elements is None:
                self.blind_spots.append(
                    RawBlindSpot(
                        kind="computed_dunder_all",
                        detail="__all__ is computed at runtime; public surface unknown",
                        line=line,
                    )
                )
            else:
                self.dunder_all = elements
                self.dunder_all_line = line
            return
        if self.context.in_function:
            return
        kind = NodeKind.CONSTANT if name.isupper() else NodeKind.ATTRIBUTE
        self.symbols.append(
            RawSymbol(
                kind=kind,
                name=name,
                local_qualname=self._local_qualname(name),
                line=line,
                end_line=end_line,
                parent_local=self.context.container,
                attrs={"annotation": annotation, "value": _unparse(value)},
            )
        )

    # -- control flow ----------------------------------------------------

    def visit_If(self, node: ast.If) -> None:
        type_checking = _is_type_checking_test(node.test)
        previous = self._descend(
            type_checking=self.context.type_checking or type_checking,
            in_conditional=not type_checking,
        )
        for child in node.body:
            self.visit(child)
        self._restore(previous)

        previous = self._descend(in_conditional=True)
        for child in node.orelse:
            self.visit(child)
        self._restore(previous)

    def visit_Try(self, node: ast.Try) -> None:
        guarded = _guards_import_error(node.handlers)
        previous = self._descend(in_try_import_guard=self.context.in_try_import_guard or guarded)
        for child in node.body:
            self.visit(child)
        self._restore(previous)
        for handler in node.handlers:
            previous = self._descend(in_try_import_guard=True, in_conditional=True)
            self.generic_visit(handler)
            self._restore(previous)
        for child in [*node.orelse, *node.finalbody]:
            self.visit(child)

    # -- calls and dynamic behaviour --------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        callee = _unparse(node.func)
        if callee:
            self.calls.append(
                RawCall(callee=callee, line=node.lineno, caller_local=self.context.enclosing)
            )
            self._check_dynamic(node, callee)
        self.generic_visit(node)

    def _check_dynamic(self, node: ast.Call, callee: str) -> None:
        leaf = callee.rpartition(".")[2]
        first = node.args[0] if node.args else None
        literal_first = isinstance(first, ast.Constant) and isinstance(first.value, str)

        if leaf in DYNAMIC_IMPORT_FUNCS and not literal_first:
            self._blind(node, "dynamic_import", f"{callee}(...) with a non-literal module name")
        elif leaf in DYNAMIC_ATTR_FUNCS and len(node.args) >= 2:
            second = node.args[1]
            if not (isinstance(second, ast.Constant) and isinstance(second.value, str)):
                self._blind(node, "dynamic_attribute", f"{callee}(...) with a computed name")
        elif leaf in EVAL_FUNCS and callee in EVAL_FUNCS:
            self._blind(node, "eval_exec", f"{callee}(...) executes code Argus cannot read")
        elif callee.endswith("sys.path.append") or callee.endswith("sys.path.insert"):
            self._blind(node, "sys_path_mutation", f"{callee}(...) changes import resolution")

    def _blind(self, node: ast.AST, kind: str, detail: str) -> None:
        self.blind_spots.append(RawBlindSpot(kind=kind, detail=detail, line=getattr(node, "lineno", 0)))


def _disambiguate_symbols(symbols: list[RawSymbol]) -> tuple[RawSymbol, ...]:
    """Give mutually-exclusive redefinitions of the same name distinct identities.

    Static extraction visits every branch of an `if`/`else`, `try`/`except`, or
    version guard unconditionally — it cannot evaluate which one runs — so two
    `def where():` under `if sys.version_info >= (3, 11): ... else: ...` are both
    real, coexisting facts about the source (this exact pattern is how
    `certifi.core.where` and several `click` methods are written). Node identity
    is `kind + qualname` (see `model/ids.py`), so leaving both definitions with
    the identical qualname collides two distinct nodes onto one id.

    The first occurrence in source order keeps its plain qualname — the
    overwhelming common case of one definition per name is entirely unaffected.
    Later occurrences get a stable `#N` suffix, deterministic as long as the
    file's own source order doesn't change.
    """
    occurrence: dict[tuple[str | None, NodeKind, str], int] = {}
    result: list[RawSymbol] = []
    for symbol in symbols:
        key = (symbol.parent_local, symbol.kind, symbol.name)
        count = occurrence.get(key, 0) + 1
        occurrence[key] = count
        if count == 1:
            result.append(symbol)
        else:
            result.append(replace(symbol, local_qualname=f"{symbol.local_qualname}#{count}"))
    return tuple(result)


def parse_file(path: Path, qualname: str, source: str | None = None) -> FileFacts:
    """Parse one file. A syntax error yields facts carrying `parse_error`."""
    text = source if source is not None else path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        return FileFacts(
            qualname=qualname,
            path=str(path),
            parse_error=f"{error.msg} (line {error.lineno})",
        )

    visitor = _FileVisitor(qualname, str(path))
    visitor.visit(tree)
    return FileFacts(
        qualname=qualname,
        path=str(path),
        docstring=ast.get_docstring(tree),
        dunder_all=visitor.dunder_all,
        dunder_all_line=visitor.dunder_all_line,
        imports=tuple(visitor.imports),
        symbols=_disambiguate_symbols(visitor.symbols),
        calls=tuple(visitor.calls),
        blind_spots=tuple(visitor.blind_spots),
        protocol_impls=tuple(visitor.protocol_impls),
    )

