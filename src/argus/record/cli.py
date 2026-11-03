"""`argus record -- COMMAND...`: run a command and save which functions of the repository ran."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Annotated

import typer

from argus.explore.render_diff import printable
from argus.record.run import RecordError, record_command
from argus.record.store import Recording


def summary_lines(rec: Recording, saved_as: str) -> list[str]:
    """What the run recorded, and every reason the recording may be incomplete."""
    ran = [f for f in rec.functions if f.calls > 0]  # a function that was only called (never awaited, failed to bind) did not run
    matched = sum(1 for f in ran if f.node_id)
    p = rec.processes
    lines = [
        f"Recording saved: {saved_as}",
        f"  The command exited with {rec.exit_code} after {rec.seconds} s; {p.saved} process{'' if p.saved == 1 else 'es'} recorded.",
        f"  {len(ran)} functions ran: {matched} matched to the scan, {len(ran) - matched} not (lambdas, comprehensions, code the scan did not read, files changed during the run).",
    ]
    if p.saved == 0:
        lines.append("  ! Nothing was recorded: the command may run Python in isolated mode (-I or -S), or it does not run Python.")
    if p.lost:
        lines.append(f"  ! {p.lost} process{'' if p.lost == 1 else 'es'} did not save (killed, or left through os._exit): their calls are missing.")
    if p.errors:
        lines.append(f"  ! The recorder failed {p.errors} time{'' if p.errors == 1 else 's'} (your program was not affected): some calls may be missing.")
    changed = sum(1 for _, digest in rec.files if digest is None)
    if changed:
        lines.append(f"  ! {changed} file{'' if changed == 1 else 's'} changed or vanished during the run: what ran in them will show as stale.")
    return [printable(line) for line in lines]


def record(
    command: Annotated[list[str], typer.Argument(help="The command to run, after --  (for example: -- pytest -q).", show_default=False)],
    path: Annotated[Path, typer.Option("--path", "-C", help="The repository to record (default: the current directory).")] = Path("."),
) -> None:
    """Run COMMAND in the repository and save which of its functions ran, who called them and which tests reached them.

    This EXECUTES your code (the same as running the command yourself), so use it only on a repository you wrote. The exit code is the command's.
    """
    root = path.resolve()
    if not root.is_dir():
        typer.echo(f"Error: {printable(str(path))} is not a directory.", err=True)
        raise typer.Exit(2)
    typer.echo(printable(f"Argus runs this command in {root}: {shlex.join(command)}"))
    typer.echo("It executes your code. Only the repository's own files are watched; nothing is changed except the .argus folder.")
    try:
        result = record_command(root, command, save_empty=False)
    except (ValueError, RecordError) as exc:
        typer.echo(f"Error: {printable(str(exc))}", err=True)
        raise typer.Exit(2) from exc
    if result.path is None:
        rec = result.recording
        typer.echo(printable(
            "Nothing was recorded; no file was written, so the earlier recordings stay the newest.\n"
            f"  The command exited with {rec.exit_code} after {rec.seconds} s, but no Python process saved a recording: its Python may be older than 3.12, "
            "it may run Python in isolated mode (-I or -S), or it may not run Python at all."
        ))
        raise typer.Exit(rec.exit_code)
    typer.echo("\n".join(summary_lines(result.recording, result.path.relative_to(root).as_posix())))
    raise typer.Exit(result.recording.exit_code)
