"""`argus bench ...` commands."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from argus.benchmark.config import RunConfig
from argus.benchmark.evaluator import HARNESS_TIMEOUT_SECONDS, SWEBENCH_VERSION, SweBenchEvaluator
from argus.benchmark.provenance import code_fingerprint, git_revision
from argus.benchmark.report import render_report
from argus.benchmark.runner import run_benchmark
from argus.benchmark.tasks import Split, assign_split, load_tasks, select_subset
from argus.benchmark.traces import TraceWriter, read_traces
from argus.benchmark.workspace import prepare_workspace
from argus.executor.aider_executor import DEFAULT_TIMEOUT_SECONDS, AiderExecutor, aider_version

bench = typer.Typer(help="Benchmark harness: tasks, runs, reports.", no_args_is_help=True)
console = Console()
DEFAULT_DATASET = "SWE-bench/SWE-bench_Lite"
# Outside the Argus checkout on purpose: an agent in a workspace must not be able to reach the
# tasks file (which holds gold and test patches) or earlier evaluation output by walking up.
BENCH_HOME = Path.home() / ".cache" / "argus" / "bench"
DEFAULT_CACHE = BENCH_HOME / "repos"
# Workspaces live apart from the mirrors (full upstream history, fixes included) and the harness output,
# so `..` from a workspace does not lead to them. Private to this user: /tmp is shared.
DEFAULT_WORK = Path(tempfile.gettempdir()) / f"argus-bench-work-{os.getuid()}"
DEFAULT_EVAL = BENCH_HOME / "eval"
DEFAULT_LEDGER = BENCH_HOME / "heldout_ledger.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@bench.command()
def fetch(
    out: Annotated[Path, typer.Option(help="Where to write the tasks JSONL.")] = Path("benchmark/swebench_lite.jsonl"),
    dataset: Annotated[str, typer.Option(help="Hugging Face dataset name.")] = DEFAULT_DATASET,
) -> None:
    """Download a SWE-bench dataset to JSONL. Needs the `bench` extra (`uv run --extra bench ...`)."""
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]Install the extra: uv run --extra bench argus bench fetch[/red]")
        raise typer.Exit(code=2) from None
    rows = load_dataset(dataset, split="test")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row)) + "\n")
    console.print(f"wrote {len(rows)} tasks to {out}")


@bench.command()
def subset(
    tasks_file: Annotated[Path, typer.Argument(help="Tasks JSONL.")],
    out: Annotated[Path, typer.Option(help="Frozen manifest to write.")],
    split: Annotated[Split, typer.Option(help="dev, validation or heldout.")] = Split.DEV,
    n: Annotated[int, typer.Option("-n", help="Number of tasks.")] = 30,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 0,
) -> None:
    """Freeze a task subset. Refuses to overwrite: a changed subset invalidates comparisons."""
    if out.exists():
        console.print(f"[red]{out} already exists; subsets are frozen. Choose a new name.[/red]")
        raise typer.Exit(code=2)
    chosen = select_subset(load_tasks(tasks_file), split, n, seed)
    manifest = {"split": split.value, "seed": seed, "n": n, "tasks_sha256": _sha256(tasks_file),
                "task_ids": [t.task_id for t in chosen]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    console.print(f"froze {len(chosen)} {split.value} tasks to {out}")


def _load_manifest(manifest: Path, tasks_file: Path, allow_heldout: bool) -> tuple[dict, list]:
    """Validate a frozen manifest against the tasks file and the split rules."""
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    missing = [k for k in ("split", "seed", "n", "tasks_sha256", "task_ids") if k not in spec]
    if missing:
        raise typer.BadParameter(f"manifest is missing {missing}")
    if spec["tasks_sha256"] != _sha256(tasks_file):
        raise typer.BadParameter("tasks file changed since the subset was frozen")
    # The label is not trusted: the split is re-derived from the ids themselves.
    actual = {assign_split(i) for i in spec["task_ids"]}
    if actual != {Split(spec["split"])}:
        raise typer.BadParameter(f"manifest says {spec['split']} but its ids belong to {sorted(s.value for s in actual)}")
    if Split.HELDOUT in actual and not allow_heldout:
        raise typer.BadParameter("refusing to run the heldout split without --allow-heldout")
    if set(spec["task_ids"]) & set(spec.get("excluded", {})):
        raise typer.BadParameter("manifest lists the same task as both included and excluded")
    wanted = set(spec["task_ids"])
    tasks = [t for t in load_tasks(tasks_file) if t.task_id in wanted]
    if len(tasks) != len(wanted):
        raise typer.BadParameter(f"{len(wanted) - len(tasks)} frozen task id(s) are missing from the tasks file")
    return spec, tasks


def _check_isolation(cache_dir: Path, work_dir: Path, eval_dir: Path) -> None:
    """Refuse a layout where a workspace can reach the mirrors or evaluation output by walking up.

    This keeps the obvious paths apart; it is not a sandbox (Aider does not confine the paths it edits).
    """
    work = work_dir.resolve()
    for name, other in (("cache", cache_dir), ("eval", eval_dir)):
        if work.parent in other.resolve().parents or work.parent == other.resolve():
            raise typer.BadParameter(
                f"--{name}-dir {other} is reachable from a workspace by walking up from {work}; "
                "keep --work-dir under a different parent."
            )
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = work.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise typer.BadParameter(f"--work-dir {work} must be private to this user (owner only, mode 700).")


def _write_or_check_sidecar(out: Path, config: RunConfig) -> None:
    """Keep the full config next to the traces; refuse to append a different config to the file."""
    sidecar = out.with_name(out.name + ".config.json")
    payload = {"config_hash": config.config_hash, **asdict(config)}
    if sidecar.exists():
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        # config_hash ignores the split so a run can grow, which means only this check keeps dev and held-out apart.
        if existing.get("split") != config.split:
            raise typer.BadParameter(
                f"{out} holds {existing.get('split')} split traces; this run is the {config.split} split. "
                "Use a new --out file."
            )
        if existing.get("config_hash") != config.config_hash:
            raise typer.BadParameter(
                f"{out} already holds config {existing.get('config_hash')}; this run is {config.config_hash}. "
                "Use a new --out file for a different configuration."
            )
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _record_heldout_run(ledger: Path, config: RunConfig, manifest: Path, n_tasks: int) -> None:
    """Append one line per held-out run attempt, so how often held-out was looked at is on record."""
    row = {"time": time.time(), "split": config.split, "config_hash": config.config_hash,
           "model": config.model, "manifest_sha256": _sha256(manifest), "n_tasks": n_tasks}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


@bench.command()
def derive(
    parent: Annotated[Path, typer.Argument(help="Frozen manifest to derive from.")],
    out: Annotated[Path, typer.Option(help="New frozen manifest to write.")],
    exclude: Annotated[list[str], typer.Option("--exclude", help="Task id to drop (repeatable).")],
    reason: Annotated[str, typer.Option(help="Why, with the evidence (recorded in the manifest).")],
) -> None:
    """Freeze a new subset with documented exclusions (e.g. tasks the environment cannot evaluate)."""
    if out.exists():
        console.print(f"[red]{out} already exists; subsets are frozen.[/red]")
        raise typer.Exit(code=2)
    spec = json.loads(parent.read_text(encoding="utf-8"))
    unknown = [i for i in exclude if i not in spec["task_ids"]]
    if unknown:
        console.print(f"[red]not in {parent}: {unknown}[/red]")
        raise typer.Exit(code=2)
    derived = {
        **spec,
        "task_ids": [i for i in spec["task_ids"] if i not in exclude],
        "excluded": {**spec.get("excluded", {}), **{i: reason for i in exclude}},
        "derived_from": {"file": parent.name, "sha256": _sha256(parent)},
    }
    out.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    console.print(f"froze {len(derived['task_ids'])} tasks ({len(exclude)} excluded) to {out}")


@bench.command()
def run(
    tasks_file: Annotated[Path, typer.Argument(help="Tasks JSONL.")],
    manifest: Annotated[Path, typer.Argument(help="Frozen subset manifest.")],
    model: Annotated[str, typer.Option(help="LiteLLM model name, e.g. deepseek/deepseek-chat.")],
    out: Annotated[Path, typer.Option(help="Trace JSONL (appended; resumes).")],
    limit: Annotated[int | None, typer.Option(help="Run at most this many new tasks.")] = None,
    allow_heldout: Annotated[bool, typer.Option(help="Permit the held-out split (major evaluations only).")] = False,
    map_tokens: Annotated[int | None, typer.Option(help="Aider repo-map budget; default is Aider's.")] = None,
    agent_timeout: Annotated[int, typer.Option(help="Seconds before the agent is stopped.")] = DEFAULT_TIMEOUT_SECONDS,
    cache_dir: Path = DEFAULT_CACHE,
    work_dir: Path = DEFAULT_WORK,
    eval_dir: Path = DEFAULT_EVAL,
    ledger: Annotated[Path, typer.Option(help="Where held-out runs are logged.")] = DEFAULT_LEDGER,
) -> None:
    """Run vanilla Aider on a frozen subset and evaluate with the SWE-bench harness."""
    spec, tasks = _load_manifest(manifest, tasks_file, allow_heldout)
    _check_isolation(cache_dir, work_dir, eval_dir)
    executor = AiderExecutor(model, map_tokens=map_tokens, timeout_seconds=agent_timeout)
    config = RunConfig(
        agent=executor.name, model=model, split=spec["split"], n=spec["n"], seed=spec["seed"],
        extra=(
            ("aider", aider_version()), ("map_tokens", str(map_tokens)), ("tasks_sha256", spec["tasks_sha256"]),
            ("swebench", SWEBENCH_VERSION), ("agent_timeout", str(agent_timeout)),
            ("harness_timeout", str(HARNESS_TIMEOUT_SECONDS)), ("code", code_fingerprint()),
            ("argus_rev", git_revision(Path(__file__).parent)),
        ),
    )
    _write_or_check_sidecar(out, config)
    if spec["split"] == Split.HELDOUT.value:
        _record_heldout_run(ledger, config, manifest, len(tasks))
    writer = TraceWriter(out)
    pending = [t for t in tasks if t.task_id not in writer.completed_ids(config.config_hash)]
    if limit is not None:
        pending = pending[:limit]
    evaluator = SweBenchEvaluator(tasks_file.resolve(), eval_dir.resolve())
    run_benchmark(
        pending, executor, evaluator, writer, config,
        workspace_factory=lambda task: prepare_workspace(task, cache_dir, work_dir),
        on_trace=lambda t: console.print(
            f"{t.task_id}: {'PASS' if t.success else 'fail'} ${t.cost:.3f} {t.total_tokens:,} tok"
            + (" [timeout]" if t.timed_out else "")
            + (f" [yellow]{t.error_kind} error: {t.error[:80]}[/yellow]" if t.error else "")
        ),
    )
    console.print(f"config {config.config_hash}; traces in {out}")


@bench.command() 
def report(
    traces: Annotated[list[Path], typer.Argument(help="Trace JSONL files; the first is the baseline.")],
) -> None:
    """Summarise trace logs and compare them pairwise against the first."""
    runs: dict[str, list] = {}
    for path in traces:
        label = path.stem if path.stem not in runs else f"{path.parent.name}/{path.stem}"
        if label in runs:
            raise typer.BadParameter(f"two trace files would share the label {label!r}")
        runs[label] = read_traces(path)
    console.print(render_report(runs), markup=False)
