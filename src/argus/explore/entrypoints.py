"""Deterministic entry-point discovery: pyproject scripts, Typer command trees, `__main__` guards, web routes (FastAPI, Flask) and plain programs
(a script that points at a function, a main guard that calls repo functions).

Only syntax is read; nothing is imported or executed. Anything that cannot be resolved
statically (a sub-app from a third-party module, a computed name) becomes an UNKNOWN claim.
"""

from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from argus.explore.claims import Claim, Evidence, Tier, evidence_at
from argus.explore.grouping import is_production_node
from argus.model.schema import ArchModel, NodeKind

COMMAND_ATTR = "command"
ADD_TYPER_ATTR = "add_typer"
TYPER_CLASS = "Typer"
WEB_CLASSES = ("FastAPI", "APIRouter", "Flask", "Blueprint")  # the classes whose variables carry route decorators
BLUEPRINT = "Blueprint"  # a prefix given when it is registered replaces its own; every other router's prefix is added to it
HTTP_METHODS = ("get", "post", "put", "delete", "patch", "head", "options", "trace")
ROUTE_ATTRS = (*HTTP_METHODS, "route", "api_route", "websocket")
MOUNT_ATTRS = ("include_router", "register_blueprint")
PREFIX_KEYWORDS = ("prefix", "url_prefix")
ROUTER_KINDS = ("APIRouter", "Blueprint")  # their routes live below whatever mounts them; a FastAPI or Flask app is its own root
UNREAD_ATTRS = ("add_api_route", "add_api_websocket_route", "add_url_rule", "add_route", "add_websocket_route")  # routes registered by call: not read
SOURCE_FOLDERS = ("src", "lib")  # an import may name a module below one of these folders without the folder: `pkg.x` for `apps.api.src.pkg.x`


@dataclass(frozen=True, slots=True)
class Script:
    name: str
    module: str
    attr: str
    claim: Claim


@dataclass(frozen=True, slots=True)
class Command:
    display: str
    path: tuple[str, ...]
    function_id: str
    module: str
    file: str
    line: int
    claim: Claim


@dataclass(frozen=True, slots=True)
class Entrypoints:
    scripts: tuple[Script, ...]
    commands: tuple[Command, ...]
    main_guards: tuple[Claim, ...]
    unknowns: tuple[Claim, ...]
    routes: tuple[Command, ...] = ()  # web routes, kept apart from the CLI commands: "GET /pay" is not a command of a command tree
    programs: tuple[Command, ...] = ()  # plain programs: a pyproject script that points at a function, the repo functions a main guard calls

    def all_claims(self) -> tuple[Claim, ...]:
        return (
            *(s.claim for s in self.scripts), *(c.claim for c in self.commands), *(r.claim for r in self.routes), *(p.claim for p in self.programs), *self.main_guards, *self.unknowns,
        )


@dataclass(frozen=True, slots=True)
class _App:
    var: str
    line: int
    name: str | None


@dataclass(frozen=True, slots=True)
class _Add:
    parent: str
    child: ast.expr
    name: str | None
    line: int
    name_is_dynamic: bool = False


@dataclass(frozen=True, slots=True)
class _Cmd:
    app: str
    name: str | None
    function: str
    decorator_line: int
    def_line: int


@dataclass(frozen=True, slots=True)
class _Web:
    kind: str  # FastAPI, APIRouter, Flask or Blueprint
    line: int
    prefix: str  # the literal prefix given when it is built
    prefix_dynamic: bool  # a prefix was given but is not a literal string


@dataclass(frozen=True, slots=True)
class _Route:
    var: str
    attr: str
    methods: tuple[str, ...] | None  # None: not a literal list
    path: str | None  # None: not a literal string
    function: str
    decorator_line: int
    def_line: int
    known: bool  # the variable is a web app or router Argus saw being built


@dataclass(frozen=True, slots=True)
class _Mount:
    parent: str  # variable the router is mounted on; "" when it is not a module-level web variable
    child: ast.expr
    prefix: str | None  # None: no prefix given
    prefix_text: str | None  # source text of a prefix that is not a literal
    line: int
    optional: bool = False  # `app.mount(path, x)`: x may be static files or a foreign app, so an unresolved one is not worth a note


@dataclass(frozen=True, slots=True)
class _Guard:
    line: int
    callees: tuple[ast.expr, ...]  # the function part of every call the guard's body can run (nested definitions and dead branches left out)
    bound: frozenset[str]  # names the body binds itself: such a name no longer means what the module imported or defined under it


@dataclass(slots=True)
class _FileInfo:
    module: str
    file: str
    apps: dict[str, _App]
    adds: list[_Add]
    cmds: list[_Cmd]
    imports: dict[str, tuple[str, str]]
    guard_lines: list[int]
    webs: dict[str, _Web] = field(default_factory=dict)
    routes: list[_Route] = field(default_factory=list)
    mounts: list[_Mount] = field(default_factory=list)
    unread: list[int] = field(default_factory=list)  # lines of route registrations Argus does not read (inside functions or classes, add_url_rule, ...)
    guards: list[_Guard] = field(default_factory=list)
    funcs: dict[str, int] = field(default_factory=dict)  # module-level function name -> line of its `def`
    rebound: set[str] = field(default_factory=set)  # names assigned at module level: `main = other` after `def main`


def find_entrypoints(root: Path, model: ArchModel) -> Entrypoints:
    root = root.resolve()
    unknowns: list[Claim] = []
    infos = _parse_files(root, model, unknowns)
    scripts = _read_scripts(root, unknowns)
    by_module = {info.module: info for info in infos}
    roots = _roots(scripts, by_module, unknowns)
    commands: list[Command] = []
    reached_apps: set[tuple[str, str]] = set()
    for display, module, var in roots:
        _walk(root, by_module, module, var, (display,), (), (), reached_apps, commands, unknowns)
    for info in infos:
        for var, app in info.apps.items():
            if (info.module, var) not in reached_apps:
                unknowns.append(Claim(
                    f"Typer app {info.module}:{var} is not reachable from any root app", Tier.UNKNOWN,
                    (evidence_at(root, info.file, app.line),),
                ))
        for cmd in info.cmds:
            if cmd.app not in info.apps:
                unknowns.append(Claim(
                    f"{info.module}.{cmd.function} is decorated with `{cmd.app}.command` but `{cmd.app}` is not a Typer app Argus can see",
                    Tier.UNKNOWN, (evidence_at(root, info.file, cmd.decorator_line),),
                ))
    guards = tuple(
        Claim(f"{info.module} has an `if __name__ == \"__main__\"` block", Tier.FACT, (evidence_at(root, info.file, line),))
        for info in infos for line in info.guard_lines
    )
    ordered = tuple(sorted(commands, key=lambda c: c.display))
    routes = tuple(sorted(_place_routes(root, infos, unknowns), key=lambda c: (c.module, c.line, c.display)))
    programs = tuple(sorted(_place_programs(root, infos, scripts), key=lambda c: (c.file, c.function_id, c.display)))
    for info in infos:
        if info.unread:
            n = len(info.unread)
            unknowns.append(Claim(
                f"{info.file}: {n} route registration{'' if n == 1 else 's'} not read by Argus (inside a function or class, add_url_rule, add_api_route), so a route may be missing",
                Tier.UNKNOWN, (evidence_at(root, info.file, min(info.unread)),),
            ))
    return Entrypoints(tuple(scripts), ordered, guards, tuple(unknowns), routes, programs)


def _parse_files(root: Path, model: ArchModel, unknowns: list[Claim]) -> list[_FileInfo]:
    modules = {
        n.provenance.file: n.qualname
        for n in model.nodes
        if n.kind in (NodeKind.MODULE, NodeKind.PACKAGE) and n.provenance.file.endswith(".py") and is_production_node(n)
    }
    infos = []
    for file in sorted(modules):
        qualname, is_package = modules[file], Path(file).name == "__init__.py"
        try:
            tree = ast.parse((root / file).read_text(encoding="utf-8"), filename=file)
        except (OSError, SyntaxError, UnicodeDecodeError) as err:
            unknowns.append(Claim(f"{file} could not be parsed for entry points: {type(err).__name__}", Tier.UNKNOWN))
            continue
        infos.append(_collect(tree, qualname, file, is_package))
    return infos


def _collect(tree: ast.Module, module: str, file: str, is_package: bool) -> _FileInfo:
    info = _FileInfo(module, file, {}, [], [], {}, [])
    for stmt in _flatten(tree.body, info):
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            info.rebound |= _bound([stmt])
        if isinstance(stmt, ast.ImportFrom):
            _collect_import(stmt, info, is_package)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None and _is_typer_call(stmt.value):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    info.apps[target.id] = _App(target.id, stmt.lineno, _literal_kw(stmt.value, "name"))
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None and _web_class(stmt.value):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    info.webs[target.id] = _web(stmt.value, stmt.lineno)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            _collect_add(stmt.value, info)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_commands(stmt, info)
            _collect_routes(stmt, info)
            if not _makes_a_context_manager(stmt):
                info.funcs.setdefault(stmt.name, stmt.lineno)
    _collect_mounts(tree, info)
    return info


_NOT_RUN = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)  # a definition is not a call of what it contains


def _never(test: ast.expr) -> bool:
    """A test that is false before the program runs: `False`, `0`, `TYPE_CHECKING`."""
    return (isinstance(test, ast.Constant) and not test.value) or (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _executed(nodes):
    """The nodes the statements run, in order: nested definitions are not entered (defining is not calling), nor is the body of an `if` that cannot be true."""
    for node in nodes:
        if isinstance(node, _NOT_RUN):
            continue
        if isinstance(node, ast.If) and _never(node.test):
            yield from _executed(node.orelse)
            continue
        yield node
        yield from _executed(ast.iter_child_nodes(node))


def _bound(nodes) -> set[str]:
    """Every name the statements bind: assigned, imported, defined, a loop or `with` target, an `except` name."""
    names: set[str] = set()
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                names.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                names.add(n.name)
    return names


def _makes_a_context_manager(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """`@contextmanager`: calling the function runs nothing until a `with` enters it."""
    return any((d.id if isinstance(d, ast.Name) else d.attr if isinstance(d, ast.Attribute) else "") in ("contextmanager", "asynccontextmanager") for d in fn.decorator_list)


def _flatten(body: list[ast.stmt], info: _FileInfo):
    """Module-level statements, looking through `if` / `try` / `with` blocks; `__main__` guards are recorded, not entered."""
    for stmt in body:
        if isinstance(stmt, ast.If) and _is_main_guard(stmt.test):
            info.guard_lines.append(stmt.lineno)
            info.guards.append(_Guard(stmt.lineno, tuple(n.func for n in _executed(stmt.body) if isinstance(n, ast.Call)), frozenset(_bound(stmt.body))))
        elif isinstance(stmt, ast.If):
            yield from _flatten([*stmt.body, *stmt.orelse], info)
        elif isinstance(stmt, ast.Try):
            handlers = [h_stmt for h in stmt.handlers for h_stmt in h.body]
            yield from _flatten([*stmt.body, *handlers, *stmt.orelse, *stmt.finalbody], info)
        elif isinstance(stmt, ast.With):
            yield from _flatten(stmt.body, info)
        else:
            yield stmt


def _collect_import(stmt: ast.ImportFrom, info: _FileInfo, is_package: bool) -> None:
    base = _import_base(stmt, info.module, is_package)
    if base is None:
        return
    for alias in stmt.names:
        info.imports[alias.asname or alias.name] = (base, alias.name)


def _import_base(stmt: ast.ImportFrom, module: str, is_package: bool) -> str | None:
    if stmt.level == 0:
        return stmt.module
    parts = module.split(".") if is_package else module.split(".")[:-1]
    drop = stmt.level - 1
    if drop > len(parts):
        return None
    parts = parts[: len(parts) - drop]
    if stmt.module:
        parts = [*parts, *stmt.module.split(".")]
    return ".".join(parts) or None


def _collect_add(call: ast.Call, info: _FileInfo) -> None:
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == ADD_TYPER_ATTR and isinstance(func.value, ast.Name)):
        return
    child = call.args[0] if call.args else next((kw.value for kw in call.keywords if kw.arg == "typer_instance"), None)
    if child is not None:
        info.adds.append(_Add(func.value.id, child, _literal_kw(call, "name"), call.lineno, _has_kw(call, "name")))


def _collect_commands(fn: ast.FunctionDef | ast.AsyncFunctionDef, info: _FileInfo) -> None:
    for deco in fn.decorator_list:
        if not isinstance(deco, ast.Call):
            continue
        f = deco.func
        if not (isinstance(f, ast.Attribute) and f.attr == COMMAND_ATTR and isinstance(f.value, ast.Name)):
            continue
        explicit = _literal_arg(deco, 0) or _literal_kw(deco, "name")
        dynamic = (bool(deco.args) or _has_kw(deco, "name")) and explicit is None
        name = None if dynamic else explicit or fn.name.lower().replace("_", "-")
        info.cmds.append(_Cmd(f.value.id, name, fn.name, deco.lineno, fn.lineno))


def _web_class(node: ast.expr) -> str | None:
    """The web class a call builds (`FastAPI(...)`, `fastapi.APIRouter(...)`), or None."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
    return name if name in WEB_CLASSES else None


def _prefix_of(call: ast.Call) -> tuple[str | None, str | None]:
    """(literal prefix, source text of a prefix that is not a literal): (None, None) when no prefix is given."""
    for kw in call.keywords:
        if kw.arg in PREFIX_KEYWORDS:
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value, None
            return None, ast.unparse(kw.value)
    unpacked = next((kw for kw in call.keywords if kw.arg is None), None)  # `**options` may carry the prefix
    return (None, "**" + ast.unparse(unpacked.value)) if unpacked is not None else (None, None)


def _web(call: ast.expr, line: int) -> _Web:
    assert isinstance(call, ast.Call)
    kind = _web_class(call) or ""
    prefix, dynamic = _prefix_of(call) if kind in ROUTER_KINDS else (None, None)  # a FastAPI or Flask app has no prefix of its own
    return _Web(kind, line, prefix or "", dynamic is not None)


def _route_methods(attr: str, deco: ast.Call) -> tuple[str, ...] | None:
    if attr in HTTP_METHODS:
        return (attr.upper(),)
    if attr == "websocket":
        return ("WS",)
    for kw in deco.keywords:
        if kw.arg == "methods":
            if isinstance(kw.value, (ast.List, ast.Tuple, ast.Set)) and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in kw.value.elts):
                return tuple(e.value.upper() for e in kw.value.elts)  # type: ignore[union-attr]
            return None
    return ("GET",)  # Flask's `route` and FastAPI's `api_route` answer GET when no methods are given


def _collect_routes(fn: ast.FunctionDef | ast.AsyncFunctionDef, info: _FileInfo) -> None:
    for deco in fn.decorator_list:
        f = deco.func if isinstance(deco, ast.Call) else None
        if not (isinstance(deco, ast.Call) and isinstance(f, ast.Attribute) and f.attr in ROUTE_ATTRS and isinstance(f.value, ast.Name)):
            continue
        path = next((p for p in (_literal_arg(deco, 0), _literal_kw(deco, "path"), _literal_kw(deco, "rule")) if p is not None), None)  # "" is a path
        known = f.value.id in info.webs
        if known or f.value.id in info.imports or (path is not None and path.startswith("/")):  # `store.get(key)` is not a route; `x.get("/a")` on an unseen app is worth saying
            info.routes.append(_Route(f.value.id, f.attr, _route_methods(f.attr, deco), path, fn.name, deco.lineno, fn.lineno, known))


def _collect_mounts(tree: ast.Module, info: _FileInfo) -> None:
    """Every `x.include_router(r, prefix=...)`, `x.register_blueprint(b, url_prefix=...)` and `x.mount(path, app)`, in any scope; and the route
    registrations that are not read (a decorator inside a function or class, `add_url_rule`, `add_api_route`)."""
    module_level = {id(fn) for fn in tree.body if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and id(node) not in module_level:
            if any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in ROUTE_ATTRS
                   and (_literal_arg(d, 0) or "").startswith("/") for d in node.decorator_list):
                info.unread.append(node.lineno)
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr in UNREAD_ATTRS:
            info.unread.append(node.lineno)
            continue
        is_mount = node.func.attr == "mount" and len(node.args) >= 2
        if not ((node.func.attr in MOUNT_ATTRS and node.args) or is_mount):
            continue
        parent = node.func.value.id if isinstance(node.func.value, ast.Name) and node.func.value.id in info.webs else ""
        if is_mount:
            literal = _literal_arg(node, 0)
            prefix, text, child = literal, None if literal is not None else ast.unparse(node.args[0]), node.args[1]
        else:
            (prefix, text), child = _prefix_of(node), node.args[0]
        info.mounts.append(_Mount(parent, child, prefix, text, node.lineno, optional=is_mount))


def _is_typer_call(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Attribute) and f.attr == TYPER_CLASS) or (isinstance(f, ast.Name) and f.id == TYPER_CLASS)


def _is_main_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
        and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _literal_kw(call: ast.Call, key: str) -> str | None:
    for kw in call.keywords:
        if kw.arg == key and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _has_kw(call: ast.Call, key: str) -> bool:
    return any(kw.arg == key for kw in call.keywords)


def _literal_arg(call: ast.Call, index: int) -> str | None:
    if len(call.args) > index and isinstance(call.args[index], ast.Constant) and isinstance(call.args[index].value, str):
        return call.args[index].value
    return None


def _read_scripts(root: Path, unknowns: list[Claim]) -> list[Script]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
        project = tomllib.loads(text).get("project", {})
        table = project.get("scripts", {}) if isinstance(project, dict) else None  # `project = "x"` is not a table: reported below
    except (OSError, tomllib.TOMLDecodeError) as err:
        unknowns.append(Claim(f"pyproject.toml could not be read: {type(err).__name__}", Tier.UNKNOWN))
        return []
    if not isinstance(table, dict):
        unknowns.append(Claim("[project.scripts] is not a table of name = target pairs", Tier.UNKNOWN))
        return []
    lines = text.splitlines()
    scripts = []
    for name, target in sorted(table.items()):
        if not isinstance(target, str):
            unknowns.append(Claim(f"script `{name}` has a non-string target and was skipped", Tier.UNKNOWN))
            continue
        module, _, attr = target.partition(":")
        line = _script_line(lines, name)
        scripts.append(Script(name, module, attr, Claim(
            f"script `{name}` -> {module}:{attr}", Tier.FACT, (evidence_at(root, "pyproject.toml", line),)
        )))
    return scripts


def _script_line(lines: list[str], name: str) -> int:
    """Line of `name = ...` inside `[project.scripts]`; the table header if the key is spelled unusually."""
    header = 1
    inside = False
    for number, text in enumerate(lines, 1):
        stripped = text.strip()
        if stripped.startswith("["):
            inside = stripped == "[project.scripts]"
            header = number if inside else header
        elif inside and re.match(rf"""^["']?{re.escape(name)}["']?\s*=""", stripped):
            return number
    return header


def _roots(scripts: list[Script], by_module: dict[str, _FileInfo], unknowns: list[Claim]) -> list[tuple[str, str, str]]:
    roots = []
    for script in scripts:
        info = by_module.get(script.module)
        if info and script.attr in info.apps:
            roots.append((script.name, script.module, script.attr))
        elif _function_named(by_module, script.module, script.attr) is not None:
            continue  # a plain function: placed as a program
        else:
            unknowns.append(Claim(
                f"script `{script.name}` targets {script.module}:{script.attr}, which is not a Typer app Argus can see",
                Tier.UNKNOWN, script.claim.evidence,
            ))
    if scripts:
        return roots
    children = {_child_key(info, add) for info in by_module.values() for add in info.adds}
    return [
        (f"{info.module}:{var}", info.module, var)
        for info in by_module.values() for var in info.apps if (info.module, var) not in children
    ]


def _child_key(info: _FileInfo, add: _Add) -> tuple[str, str] | None:
    """(module, variable) of the registered sub-app: a local name, `from m import app`, or `module.app`."""
    child = add.child
    if isinstance(child, ast.Name):
        if child.id in info.apps:
            return info.module, child.id
        return info.imports.get(child.id)
    if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
        imported = info.imports.get(child.value.id)  # `from pkg import sub` -> sub.app is (pkg.sub, app)
        if imported:
            return f"{imported[0]}.{imported[1]}", child.attr
    return None


def _walk(
    root: Path, by_module: dict[str, _FileInfo], module: str, var: str, display: tuple[str, ...],
    registrations: tuple[Evidence, ...], ancestors: tuple[tuple[str, str], ...],
    reached: set[tuple[str, str]], out: list[Command], unknowns: list[Claim],
) -> None:
    """Place the commands of one app; `ancestors` only guards against registration cycles."""
    if (module, var) in ancestors:
        return
    reached.add((module, var))
    info = by_module[module]
    for cmd in info.cmds:
        if cmd.app != var:
            continue
        site = evidence_at(root, info.file, cmd.decorator_line)
        if cmd.name is None:
            unknowns.append(Claim(
                f"{module}.{cmd.function} has a non-literal command name; it is not placed in the tree", Tier.UNKNOWN, (site,)
            ))
            continue
        shown = (*display, cmd.name)
        out.append(Command(
            " ".join(shown), shown[1:], f"function:{module}.{cmd.function}", module, info.file, cmd.def_line,
            Claim(f"`{' '.join(shown)}` runs {module}.{cmd.function}", Tier.FACT, (*registrations, site)),
        ))
    for add in info.adds:
        if add.parent == var:
            _walk_child(root, by_module, info, add, display, registrations, (*ancestors, (module, var)), reached, out, unknowns)


def _walk_child(
    root: Path, by_module: dict[str, _FileInfo], info: _FileInfo, add: _Add, display: tuple[str, ...],
    registrations: tuple[Evidence, ...], ancestors: tuple[tuple[str, str], ...],
    reached: set[tuple[str, str]], out: list[Command], unknowns: list[Claim],
) -> None:
    site = evidence_at(root, info.file, add.line)
    label = ast.unparse(add.child)
    key = _child_key(info, add)
    child = by_module[key[0]].apps.get(key[1]) if key and key[0] in by_module else None
    if key is None or child is None:
        unknowns.append(Claim(f"sub-app `{label}` registered in {info.module} cannot be resolved to source in this repo", Tier.UNKNOWN, (site,)))
        return
    name = None if add.name_is_dynamic and add.name is None else (add.name or child.name)
    if name is None:
        unknowns.append(Claim(f"sub-app `{label}` in {info.module} has no literal name; its commands are not placed", Tier.UNKNOWN, (site,)))
        return
    _walk(root, by_module, key[0], key[1], (*display, name), (*registrations, site), ancestors, reached, out, unknowns)


# -- web routes ------------------------------------------------------------------------------------------------------


def _web_key(info: _FileInfo, expr: ast.expr) -> tuple[str, str] | None:
    """(module, variable) of a router or blueprint named in a mount: a local name, `from m import r`, or `module.r`."""
    if isinstance(expr, ast.Name):
        return (info.module, expr.id) if expr.id in info.webs else info.imports.get(expr.id)
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        imported = info.imports.get(expr.value.id)
        if imported:
            return f"{imported[0]}.{imported[1]}", expr.attr
    return None


def _join(prefix: str, path: str) -> str:
    joined = prefix.rstrip("/") + "/" + path.lstrip("/") if path else prefix
    return joined or "/"


def _module_named(modules, name: str) -> str | None:
    """An import names `pkg.routers`; the scan calls it `apps.api.src.pkg.routers` when packages sit below a source folder. Only that is accepted:
    any other module that merely ends the same way may be an installed package."""
    if name in modules:
        return name
    ends = [m for m in modules if m.endswith("." + name) and m[: -len(name) - 1].rsplit(".", 1)[-1] in SOURCE_FOLDERS]
    return ends[0] if len(ends) == 1 else None


def _callee_key(info: _FileInfo, func: ast.expr, guard: _Guard) -> tuple[str, str] | None:
    """(module, function) a call in `info` names: a function of the same module, `from m import f` then `f()`, or `from p import m` then `m.f()`.
    A name that the module or the guard assigns again no longer means what it was imported or defined as."""
    if isinstance(func, ast.Name):
        if func.id in info.rebound or func.id in guard.bound:
            return None
        return (info.module, func.id) if func.id in info.funcs else info.imports.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in info.rebound or func.value.id in guard.bound:
            return None
        imported = info.imports.get(func.value.id)
        return (f"{imported[0]}.{imported[1]}", func.attr) if imported else None
    return None


def _function_named(by_module: dict[str, _FileInfo], module: str, name: str) -> tuple[_FileInfo, str] | None:
    """The module and name of the function `module:name` really is: defined there, or imported there from another module of the repository (one hop)."""
    info = by_module.get(_module_named(by_module, module) or "")
    if info is None:
        return None
    if name in info.funcs:
        return info, name
    imported = info.imports.get(name)
    target = by_module.get(_module_named(by_module, imported[0]) or "") if imported else None
    return (target, imported[1]) if target is not None and imported and imported[1] in target.funcs else None


def _place_programs(root: Path, infos: list[_FileInfo], scripts: list[Script]) -> list[Command]:
    by_module = {info.module: info for info in infos}
    out: dict[tuple[str, str], Command] = {}
    for info in infos:
        for guard in info.guards:
            for func in guard.callees:
                key = _callee_key(info, func, guard)
                module = _module_named(by_module, key[0]) if key else None
                if key is None or module is None or key[1] not in by_module[module].funcs:
                    continue  # a method, a library call, something Argus cannot follow to a function of this repository
                display, function_id = f"python {info.file}", f"function:{module}.{key[1]}"
                out.setdefault((display, function_id), Command(
                    display, (info.file,), function_id, info.module, info.file, guard.line,
                    Claim(f"`{display}` runs {module}.{key[1]}", Tier.FACT, (evidence_at(root, info.file, guard.line),)),
                ))
    for script in scripts:
        found = _function_named(by_module, script.module, script.attr)
        script_module = by_module.get(_module_named(by_module, script.module) or "")
        if found is None or (script_module is not None and script.attr in script_module.apps):
            continue
        info, name = found
        function_id = f"function:{info.module}.{name}"
        out.setdefault((script.name, function_id), Command(
            script.name, (script.name,), function_id, info.module, info.file, info.funcs[name],
            Claim(f"script `{script.name}` runs {info.module}.{name}", Tier.FACT, script.claim.evidence),
        ))
    return list(out.values())


def _place_routes(root: Path, infos: list[_FileInfo], unknowns: list[Claim]) -> list[Command]:
    by_module = {info.module: info for info in infos}

    def module_named(name: str) -> str | None:
        return _module_named(by_module, name)

    mounts: dict[tuple[str, str], list[tuple[_FileInfo, _Mount]]] = {}
    for info in infos:
        for mount in info.mounts:
            key = _web_key(info, mount.child)
            module = module_named(key[0]) if key else None
            if key is not None and module is not None and key[1] in by_module[module].webs:
                mounts.setdefault((module, key[1]), []).append((info, mount))
            elif not mount.optional:
                unknowns.append(Claim(
                    f"router {ast.unparse(mount.child)} is mounted here but cannot be resolved to a router in this repository; "
                    f"the URLs of the routes it carries may have a prefix Argus does not know", Tier.UNKNOWN, (evidence_at(root, info.file, mount.line),),
                ))
    reported: set[tuple] = set()

    def prefixes(key: tuple[str, str], seen: frozenset[tuple[str, str]]) -> list[tuple[str, bool, tuple[Evidence, ...]]]:
        """Every way the router is mounted: (path in front of its routes, whether a part of it is unknown, the mount lines)."""
        web = by_module[key[0]].webs[key[1]]
        if key in seen:
            return [(web.prefix, True, ())]
        if key not in mounts:
            if web.kind in ROUTER_KINDS and ("unmounted", key) not in reported:  # its URLs start with whatever mounts it, and nothing visible does
                reported.add(("unmounted", key))
                info = by_module[key[0]]
                unknowns.append(Claim(
                    f"router {key[0]}:{key[1]} is not mounted anywhere Argus can see; the URLs of its routes start with something unknown", Tier.UNKNOWN,
                    (evidence_at(root, info.file, web.line),),
                ))
            return [(web.prefix, web.prefix_dynamic or web.kind in ROUTER_KINDS, ())]
        found = []
        for where, mount in mounts[key]:
            above = prefixes((where.module, mount.parent), seen | {key}) if mount.parent else [("", False, ())]
            site = evidence_at(root, where.file, mount.line)
            unknown = mount.prefix_text is not None
            if unknown and ("prefix", where.file, key, mount.line) not in reported:
                reported.add(("prefix", where.file, key, mount.line))
                unknowns.append(Claim(
                    f"router {ast.unparse(mount.child)} is mounted with a prefix Argus cannot read ({mount.prefix_text}); "
                    f"the URLs of its routes may start with something else", Tier.UNKNOWN, (site,),
                ))
            own = web.prefix
            base = (mount.prefix if mount.prefix is not None else own) if web.kind == BLUEPRINT else (mount.prefix or "") + own
            found += [(upper + base, upper_unknown or unknown or web.prefix_dynamic, (*lines, site)) for upper, upper_unknown, lines in above]
        return found

    out: dict[tuple[str, str], Command] = {}
    orphans: dict[tuple[str, str], list[tuple[_Route, Evidence]]] = {}
    for info in infos:
        for route in info.routes:
            site = evidence_at(root, info.file, route.decorator_line)
            key = (info.module, route.var) if route.known else None
            if key is None and route.var in info.imports:  # `from app.main import app` ... `@app.get("/x")`
                base, name = info.imports[route.var]
                module = module_named(base)
                key = (module, name) if module is not None and name in by_module[module].webs else None
            if key is None:
                orphans.setdefault((info.file, route.var), []).append((route, site))
                continue
            if route.path is None or not route.methods:
                unknowns.append(Claim(f"{info.module}.{route.function} has a route decorator with a non-literal path or methods; it is not placed", Tier.UNKNOWN, (site,)))
                continue
            for prefix, unknown, lines in prefixes(key, frozenset()):
                for method in route.methods:
                    display = f"{method} {'…' if unknown else ''}{_join(prefix, route.path)}"  # "…" marks a URL that starts with something Argus could not read
                    out.setdefault((display, f"function:{info.module}.{route.function}"), Command(
                        display, (method, _join(prefix, route.path)), f"function:{info.module}.{route.function}", info.module, info.file, route.def_line,
                        Claim(f"`{display}` runs {info.module}.{route.function}", Tier.FACT, (*lines, site)),
                    ))
    for (file, var), found in orphans.items():  # one note per file and name, not one per route
        first, site = found[0]
        n = len(found)
        unknowns.append(Claim(
            f"{file}: {n} route{'' if n == 1 else 's'} decorated with `{var}.{first.attr}` but `{var}` is not a web app Argus can see; not placed", Tier.UNKNOWN, (site,),
        ))
    return list(out.values())
