"""`argus` command line: the terminal UI (bare `argus`), `explore` (read-only architecture views and the diff review) and `bench` (the Phase 0 benchmark harness).

The older architecture-first commands (scan, sample, snapshots, health, diff) were removed on 2026-10-17; their library code is still
in the package and is recoverable from git history.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Annotated

import typer

from argus.benchmark.cli import bench
from argus.explore.cli import explore
from argus.record.check import check
from argus.record.cli import record

# Argus parses other people's code, and a warning about it would name a temporary path. One filter for the whole process:
# `warnings.catch_warnings()` is not thread-safe, and the TUI scans from worker threads.
warnings.filterwarnings("ignore", category=SyntaxWarning)

app = typer.Typer(
    name="argus",
    help="Architecture understanding for Python codebases with AI agents in the loop.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    path: Annotated[Path, typer.Option("--path", "-C", help="Repository to open (default: the current directory).")] = Path("."),
    base: Annotated[str | None, typer.Option("--base", help="Compare the Changes view with this commit instead of the state at launch.")] = None,
    include_tests: Annotated[bool, typer.Option("--include-tests", help="Count callers that live in test code.")] = False,
) -> None:
    """With no command: open the terminal UI (changes, map, symbol, diff) for the repository in the current directory."""
    if ctx.invoked_subcommand is not None:
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):  # a pipe or a script cannot show a full-screen app
        typer.echo(ctx.get_help())
        raise typer.Exit()
    from argus.tui.app import run_tui  # imported here: every other command stays as fast as before

    run_tui(path, base, include_tests)


app.add_typer(bench, name="bench")
app.add_typer(explore, name="explore")
app.command("record")(record)
app.command("check")(check)

if __name__ == "__main__":
    app()
