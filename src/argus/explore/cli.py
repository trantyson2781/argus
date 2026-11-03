"""`argus explore ...`: read-only architecture exploration. Always scans fresh; writes nothing."""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import click
import typer
from rich.console import Console
from rich.markup import escape

from argus.explore.ascii import render_deps, render_system
from argus.explore.components import resolve_name
from argus.explore.compare import compare_refs
from argus.explore.diagram import build_deps, build_levels, build_system
from argus.explore.entrypoints import Command, find_entrypoints
from argus.explore.jsonout import component_json, diff_json, dumps, entrypoints_json, envelope, error_json, symbol_json
from argus.explore.reach import CallStep, call_tree
from argus.explore.render import print_detail, print_entrypoints, print_map
from argus.explore.render_diff import DEFAULT_CALLER_LIMIT, DEFAULT_SYMBOL_LIMIT, print_diff
from argus.explore.render_symbol import DEFAULT_LIMIT, print_symbol
from argus.explore.sample import sample_report
from argus.explore.sides import DiffError
from argus.explore.symbols import build_symbol_view, resolve_symbol
from argus.record.observed import latest_recording
from argus.explore.views import EmptyComponentError, build_detail, build_map
from argus.explore.watch import open_watch, run_watch
from argus.extract import scan
from argus.extract.resolve_calls import resolve_v2_enabled
from argus.model.schema import ArchModel

explore = typer.Typer(help="Read-only architecture exploration (fresh scan, nothing is written).", no_args_is_help=True)
console = Console(highlight=False, soft_wrap=True)
err_console = Console(stderr=True, highlight=False, soft_wrap=True)  # errors never share stdout with data (a --json pipe must stay valid)

PathArg = Annotated[Path, typer.Argument(help="Repository root.")]
DEFAULT_TRACE_DEPTH = 2
NESTED_MESSAGE = "A file is nested too deeply to analyse (the scanner ran out of stack); the analysis could not be completed."


def _scan(path: Path, resolve_v2: bool | None = None, command: str = "", as_json: bool = False) -> tuple[Path, ArchModel]:
    root = path.resolve()
    if not root.is_dir():
        _fail(command, "not_a_directory", f"Not a directory: {path if as_json else root}", as_json)  # JSON never carries the resolved absolute path
    return root, scan(root, resolve_v2=resolve_v2)


class MapView(StrEnum):
    SYSTEM = "system"
    DEPS = "deps"
    LIST = "list"
    LEVELS = "levels"


@explore.command("map")
def map_(
    path: PathArg = Path("."),
    view: Annotated[MapView, typer.Option("--view", help="system: subsystems and containment. deps: import layers. levels: every component in import levels, with purposes and entry points. list: ranked list.")] = MapView.SYSTEM,
    open_: Annotated[str | None, typer.Option("--open", help="Expand a subsystem or component (e.g. engine, engine.data).")] = None,
    refs: Annotated[bool, typer.Option("--refs", help="Show file:line sources for relationships.")] = False,
) -> None:
    """ASCII architecture maps: system (containment) and deps (imports). No execution order is implied."""
    root, model = _scan(path)
    eps = find_entrypoints(root, model)
    if view is MapView.LIST:
        print_map(console, build_map(model, root, eps), str(root))
        return
    try:
        if view is MapView.LEVELS and open_:
            raise ValueError("--open does not apply to --view levels: it already shows every component")
        if view is MapView.SYSTEM:
            lines = render_system(build_system(model, root, eps, open_), refs)
        elif view is MapView.LEVELS:
            lines = render_deps(build_levels(model, root, eps), refs)
        else:
            lines = render_deps(build_deps(model, root, open_), refs)
    except (LookupError, ValueError) as err:
        err_console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(code=2) from None
    console.print("\n".join(lines), markup=False, highlight=False, emoji=False, soft_wrap=True)


@explore.command()
def entrypoints(
    path: PathArg = Path("."),
    command: Annotated[str | None, typer.Option("--command", help='Show what one command calls, e.g. "bench run".')] = None,
    depth: Annotated[int, typer.Option(min=1, help="Call depth for --command.")] = DEFAULT_TRACE_DEPTH,
    json_: Annotated[bool, typer.Option("--json", help="Print one complete JSON document instead of the text report.")] = False,
) -> None:
    """Scripts, CLI command tree and `__main__` guards. Use --command to see what a command calls."""
    document: str | None = None
    with _json_guard("entrypoints", json_):
        root, model = _scan(path, command="entrypoints", as_json=json_)
        eps = find_entrypoints(root, model)
        trace: list[tuple[Command, tuple[CallStep, ...]]] | None = None
        if command is not None:
            wanted = command.split()
            matches = [c for c in eps.commands if list(c.path) == wanted]
            if not matches:
                known = ", ".join(" ".join(c.path) for c in eps.commands)
                _fail("entrypoints", "command_not_found", f"No command {command!r}. Known: {known}", json_)
            trace = [(c, call_tree(model, root, c.function_id, depth)) for c in matches]
        if json_:
            document = dumps(envelope(
                "entrypoints", entrypoints_json(eps, trace), resolve_v2=resolve_v2_enabled(), options={"command": command, "depth": depth},
            ))
    if document is not None:
        typer.echo(document)
        return
    print_entrypoints(console, eps, trace)


@explore.command()
def component(
    name: Annotated[str, typer.Argument(help="Component (e.g. argus.benchmark), module, or unique suffix.")],
    path: PathArg = Path("."),
    json_: Annotated[bool, typer.Option("--json", help="Print one complete JSON document instead of the text report.")] = False,
) -> None:
    """One component or module: purpose, public symbols, dependencies, and how CLI entry points reach it."""
    document: str | None = None
    with _json_guard("component", json_):
        root, model = _scan(path, command="component", as_json=json_)
        try:
            label, group_of = resolve_name(model, name)
        except LookupError as err:
            _fail("component", "component_lookup", str(err), json_)
        try:
            view = build_detail(model, root, find_entrypoints(root, model), label, group_of)
        except EmptyComponentError as err:  # only the intended error: a KeyError in there is a bug and stays an internal error
            _fail("component", "component_lookup", str(err), json_)
        if json_:
            document = dumps(envelope("component", component_json(view), resolve_v2=resolve_v2_enabled(), options={"name": name}))
    if document is not None:
        typer.echo(document)
        return
    print_detail(console, view)


@explore.command()
def symbol(
    query: Annotated[str, typer.Argument(help="Function, class, method or module: a name (run_benchmark), Class.method, or a full dotted path.")],
    path: PathArg = Path("."),
    limit: Annotated[int | None, typer.Option("--limit", min=1, help=f"Max rows per section (default {DEFAULT_LIMIT}).")] = None,
    include_tests: Annotated[bool, typer.Option("--include-tests", help="Count callers that live in test code.")] = False,
    brief: Annotated[bool, typer.Option("--brief", help="Print only the summary (definition, purpose, CLI reach, callers, tests, size); skip the long caller and callee lists.")] = False,
    json_: Annotated[bool, typer.Option("--json", help="Print one complete JSON document instead of the text report (never capped).")] = False,
) -> None:
    """Find one symbol and show its definition, purpose, which commands reach it, who calls it, which tests call it, and its callees."""
    if json_ and (limit is not None or brief):
        _fail_json("symbol", "bad_option", "--json prints everything: --limit and --brief do not apply to it.")
    resolve_v2 = resolve_v2_enabled()
    document: str | None = None
    with _json_guard("symbol", json_):
        root, model = _scan(path, resolve_v2, "symbol", json_)
        try:
            node = resolve_symbol(model, query)
        except LookupError as err:
            _fail("symbol", "symbol_lookup", str(err), json_)
        recording, recording_error = latest_recording(root)
        view = build_symbol_view(model, root, node, include_tests, resolve_v2, recording, recording_error)
        if json_:
            document = dumps(envelope("symbol", symbol_json(view), resolve_v2=resolve_v2, options={"query": query, "include_tests": include_tests}))
    if document is not None:
        typer.echo(document)
        return
    print_symbol(console, view, limit if limit is not None else DEFAULT_LIMIT, brief)


@explore.command("diff")
def diff_(
    path: PathArg = Path("."),
    base: Annotated[str, typer.Option("--base", help="Git ref to compare from.")] = "HEAD",
    head: Annotated[str | None, typer.Option("--head", help="Git ref to compare to. Default: the working tree (staged, unstaged and untracked .py files).")] = None,
    include_tests: Annotated[bool, typer.Option("--include-tests", help="Show callers that live in test code.")] = False,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help=f"Max rows per capped section: caller groups (default {DEFAULT_CALLER_LIMIT}) and Symbols (default {DEFAULT_SYMBOL_LIMIT}).")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Show every caller group and every symbol.")] = False,
    json_: Annotated[bool, typer.Option("--json", help="Print one complete JSON document instead of the text report (never capped).")] = False,
) -> None:
    """What changed in structure between two states of a git repository, and who points at it. Writes nothing."""
    if json_ and (all_ or limit is not None):
        _fail_json("diff", "bad_option", "--json prints everything: --all and --limit do not apply to it.")
    if all_ and limit is not None:
        err_console.print("[red]--all and --limit cannot be used together.[/red]")
        raise typer.Exit(code=2)
    resolve_v2 = resolve_v2_enabled()
    document: str | None = None
    with _json_guard("diff", json_):
        try:
            view = compare_refs(path, base, head, include_tests, resolve_v2)
        except DiffError as err:
            _fail("diff", "diff_error", str(err), json_)
        if json_:
            options = {"base": base, "head": head, "include_tests": include_tests}
            document = dumps(envelope("diff", diff_json(view), resolve_v2=resolve_v2, options=options))
    if document is not None:
        typer.echo(document)
        return
    print_diff(console, view, *_display_caps(limit, all_))


@explore.command()
def sample(
    path: PathArg = Path("."),
    count: Annotated[int, typer.Option("--count", min=1, help="How many edges to draw for the hand check.")] = 50,
    seed: Annotated[int, typer.Option("--seed", help="Same seed, same edges.")] = 0,
) -> None:
    """The Phase 0 edge-quality gates: the AMBIGUOUS rate, and a random sample of edges to verify by hand. Exit 1 when the rate gate fails."""
    _, model = _scan(path, command="sample")
    lines, passed = sample_report(model, count, seed)
    console.print("\n".join(lines), markup=False, highlight=False, emoji=False, soft_wrap=True)
    if not passed:
        raise typer.Exit(code=1)


def _fail_json(command: str, kind: str, message: str, code: int = 2) -> NoReturn:
    """One JSON error object on stdout (nothing else): a pipeline never receives half a document."""
    typer.echo(dumps(error_json(command, kind, message)))
    raise typer.Exit(code=code)


def _fail(command: str, kind: str, message: str, as_json: bool) -> NoReturn:
    if as_json:
        _fail_json(command, kind, message)
    err_console.print(f"[red]{escape(message)}[/red]")
    raise typer.Exit(code=2)


@contextmanager
def _json_guard(command: str, as_json: bool) -> Iterator[None]:
    """Turn a deep-nesting failure into the usual error, and in --json mode any unexpected error into a JSON error object (exit 1).
    Only the exception type is reported: its message can hold paths or source text. In text mode an unexpected error still raises."""
    try:
        yield
    except (typer.Exit, click.exceptions.ClickException, click.exceptions.Abort):
        raise
    except (RecursionError, MemoryError):
        _fail(command, "too_deeply_nested", NESTED_MESSAGE, as_json)
    except Exception as err:
        if not as_json:
            raise
        _fail_json(command, "internal_error", type(err).__name__, code=1)


@explore.command()
def watch(
    path: PathArg = Path("."),
    base: Annotated[str | None, typer.Option("--base", help="Compare with this exact commit (no merge-base, unlike `explore diff`). Default: the working tree as it is at launch; edits made before launch are then not shown.")] = None,
    include_tests: Annotated[bool, typer.Option("--include-tests", help="Show callers that live in test code.")] = False,
    details: Annotated[bool, typer.Option("--details", help="Show the long report (callers, calls, imports, everything Argus cannot check) instead of one short row per change.")] = False,
) -> None:
    """Show what changed, and repaint whenever a Python file changes (poll, no extra dependency). Writes nothing; Ctrl-C quits."""
    restore_signals = _install_stop_handlers()  # so the temporary directories are removed on `kill` and a closed terminal, not only on Ctrl-C
    try:
        with _json_guard("watch", False):
            try:
                with open_watch(path, base, include_tests) as settings:
                    run_watch(settings, console, details=details)
            except DiffError as err:
                _fail("watch", "diff_error", str(err), False)
    except KeyboardInterrupt:
        console.print("stopped.")
    except BrokenPipeError:
        pass  # whatever was reading the output has gone away
    finally:
        restore_signals()


STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def _stop_on_signal(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _install_stop_handlers() -> Callable[[], None]:
    """Make SIGTERM and SIGHUP behave like Ctrl-C. Returns the function that puts the previous handlers back."""
    previous: dict[signal.Signals, object] = {}
    for sig in STOP_SIGNALS:
        try:
            previous[sig] = signal.signal(sig, _stop_on_signal)
        except ValueError:  # not the main thread: no handler can be installed
            continue

    def restore() -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler if handler is not None else signal.SIG_DFL)

    return restore


def _display_caps(limit: int | None, show_all: bool) -> tuple[int | None, int | None]:
    """(caller groups per section, symbols) to print: everything, one shared number, or the defaults."""
    if show_all:
        return None, None
    if limit is not None:
        return limit, limit
    return DEFAULT_CALLER_LIMIT, DEFAULT_SYMBOL_LIMIT
