from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from argus.cli import app
from argus.benchmark.traces import TraceWriter
from tests.benchmark.test_tasks import _task, _write
from tests.benchmark.test_traces import _trace

runner = CliRunner()


def test_subset_writes_frozen_manifest_with_task_hash(tmp_path: Path) -> None:
    tasks = _write(tmp_path / "t.jsonl", [_task(i) for i in range(200)])
    out = tmp_path / "dev.json"
    result = runner.invoke(app, ["bench", "subset", str(tasks), "--split", "dev", "-n", "5", "--out", str(out)])
    assert result.exit_code == 0, result.output
    manifest = json.loads(out.read_text())
    assert manifest["split"] == "dev" and len(manifest["task_ids"]) == 5 and len(manifest["tasks_sha256"]) == 64


def test_subset_refuses_to_overwrite_a_frozen_manifest(tmp_path: Path) -> None:
    tasks = _write(tmp_path / "t.jsonl", [_task(i) for i in range(200)])
    out = tmp_path / "dev.json"
    args = ["bench", "subset", str(tasks), "--split", "dev", "-n", "5", "--out", str(out)]
    assert runner.invoke(app, args).exit_code == 0
    assert runner.invoke(app, args).exit_code != 0


def test_run_refuses_heldout_without_explicit_flag(tmp_path: Path) -> None:
    tasks = _write(tmp_path / "t.jsonl", [_task(i) for i in range(200)])
    out = tmp_path / "held.json"
    assert runner.invoke(app, ["bench", "subset", str(tasks), "--split", "heldout", "-n", "3", "--out", str(out)]).exit_code == 0
    result = runner.invoke(app, ["bench", "run", str(tasks), str(out), "--model", "m", "--out", str(tmp_path / "r.jsonl")])
    assert result.exit_code != 0 and "heldout" in result.output.lower()


def _frozen(tmp_path: Path, split: str = "dev", n: int = 3):
    tasks = _write(tmp_path / "t.jsonl", [_task(i) for i in range(200)])
    out = tmp_path / f"{split}.json"
    assert runner.invoke(app, ["bench", "subset", str(tasks), "--split", split, "-n", str(n), "--out", str(out)]).exit_code == 0
    return tasks, out


def test_run_rejects_a_manifest_whose_split_label_was_edited(tmp_path: Path) -> None:
    tasks, manifest = _frozen(tmp_path, "heldout")
    spec = json.loads(manifest.read_text())
    spec["split"] = "dev"
    manifest.write_text(json.dumps(spec))
    result = runner.invoke(app, ["bench", "run", str(tasks), str(manifest), "--model", "m", "--out", str(tmp_path / "r.jsonl")])
    assert result.exit_code != 0 and "belong to" in result.output


def test_run_rejects_manifest_with_missing_keys_and_missing_tasks(tmp_path: Path) -> None:
    tasks, manifest = _frozen(tmp_path)
    spec = json.loads(manifest.read_text())
    broken = tmp_path / "b.json"
    broken.write_text(json.dumps({k: v for k, v in spec.items() if k != "seed"}))
    args = ["bench", "run", str(tasks), str(broken), "--model", "m", "--out", str(tmp_path / "r.jsonl")]
    assert "missing" in runner.invoke(app, args).output
    spec["task_ids"] = spec["task_ids"] + ["ghost-1"]
    broken.write_text(json.dumps(spec))
    assert runner.invoke(app, args).exit_code != 0


def test_run_rejects_a_changed_tasks_file(tmp_path: Path) -> None:
    tasks, manifest = _frozen(tmp_path)
    tasks.write_text(tasks.read_text() + "\n")
    result = runner.invoke(app, ["bench", "run", str(tasks), str(manifest), "--model", "m", "--out", str(tmp_path / "r.jsonl")])
    assert result.exit_code != 0 and "changed" in result.output


def test_sidecar_refuses_a_different_config_in_the_same_trace_file(tmp_path: Path) -> None:
    import typer
    from argus.benchmark.cli import _write_or_check_sidecar
    from argus.benchmark.config import RunConfig
    out = tmp_path / "runs.jsonl"
    _write_or_check_sidecar(out, RunConfig("a", "m1", "dev", 1, 0))
    _write_or_check_sidecar(out, RunConfig("a", "m1", "dev", 5, 0))
    with __import__("pytest").raises(typer.BadParameter, match="new --out"):
        _write_or_check_sidecar(out, RunConfig("a", "m2", "dev", 1, 0))


def test_sidecar_refuses_a_different_split_in_the_same_trace_file(tmp_path: Path) -> None:
    import pytest
    import typer
    from argus.benchmark.cli import _write_or_check_sidecar
    from argus.benchmark.config import RunConfig
    out = tmp_path / "runs.jsonl"
    _write_or_check_sidecar(out, RunConfig("a", "m", "dev", 5, 0))
    # the config hash ignores the split on purpose, so this is the only thing keeping dev and held-out apart
    assert RunConfig("a", "m", "dev", 5, 0).config_hash == RunConfig("a", "m", "heldout", 5, 0).config_hash
    with pytest.raises(typer.BadParameter, match="split"):
        _write_or_check_sidecar(out, RunConfig("a", "m", "heldout", 5, 0))


def _fake_run(monkeypatch) -> None:
    import argus.benchmark.cli as cli
    monkeypatch.setattr(cli, "aider_version", lambda: "0.0")
    monkeypatch.setattr(cli, "run_benchmark", lambda *a, **k: [])


def _run_args(tasks: Path, manifest: Path, tmp_path: Path, ledger: Path, *extra: str) -> list[str]:
    return ["bench", "run", str(tasks), str(manifest), "--model", "m", "--out", str(tmp_path / "r.jsonl"),
            "--ledger", str(ledger), *extra]


def test_heldout_run_is_recorded_in_the_ledger(tmp_path: Path, monkeypatch) -> None:
    _fake_run(monkeypatch)
    tasks, manifest = _frozen(tmp_path, "heldout")
    ledger = tmp_path / "ledger.jsonl"
    args = _run_args(tasks, manifest, tmp_path, ledger, "--allow-heldout")
    assert runner.invoke(app, args).exit_code == 0
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["split"] == "heldout" and rows[0]["n_tasks"] == 3 and rows[0]["model"] == "m"
    assert len(rows[0]["manifest_sha256"]) == 64 and rows[0]["config_hash"] and rows[0]["time"] > 0
    assert runner.invoke(app, args).exit_code == 0
    assert len(ledger.read_text().splitlines()) == 2  # every attempt counts, not just the first


def test_dev_run_leaves_no_ledger_entry(tmp_path: Path, monkeypatch) -> None:
    _fake_run(monkeypatch)
    tasks, manifest = _frozen(tmp_path, "dev")
    ledger = tmp_path / "ledger.jsonl"
    assert runner.invoke(app, _run_args(tasks, manifest, tmp_path, ledger)).exit_code == 0
    assert not ledger.exists()


def test_refused_heldout_run_leaves_no_ledger_entry(tmp_path: Path, monkeypatch) -> None:
    _fake_run(monkeypatch)
    tasks, manifest = _frozen(tmp_path, "heldout")
    ledger = tmp_path / "ledger.jsonl"
    assert runner.invoke(app, _run_args(tasks, manifest, tmp_path, ledger)).exit_code != 0
    assert not ledger.exists()


def _is_inside(path: Path, parent: Path) -> bool:
    return parent == path or parent in path.parents


def test_default_work_dir_does_not_share_a_parent_with_mirrors_or_eval_output() -> None:
    from argus.benchmark.cli import DEFAULT_CACHE, DEFAULT_EVAL, DEFAULT_WORK
    for other in (DEFAULT_CACHE, DEFAULT_EVAL):
        assert not _is_inside(other, DEFAULT_WORK.parent), f"{other} is reachable from {DEFAULT_WORK} by walking up"


def test_isolation_check_refuses_workspaces_beside_the_mirrors(tmp_path: Path) -> None:
    import pytest
    import typer
    from argus.benchmark.cli import _check_isolation
    with pytest.raises(typer.BadParameter, match="walking up"):
        _check_isolation(tmp_path / "repos", tmp_path / "work", tmp_path / "eval")


def test_isolation_check_accepts_separate_roots_and_makes_the_work_dir_private(tmp_path: Path) -> None:
    from argus.benchmark.cli import _check_isolation
    work = tmp_path / "w" / "work"
    _check_isolation(tmp_path / "b" / "repos", work, tmp_path / "b" / "eval")
    assert work.is_dir() and work.stat().st_mode & 0o077 == 0


def test_isolation_check_refuses_a_work_dir_others_can_read(tmp_path: Path) -> None:
    import pytest
    import typer
    from argus.benchmark.cli import _check_isolation
    work = tmp_path / "w" / "work"
    work.mkdir(parents=True)
    work.chmod(0o755)
    with pytest.raises(typer.BadParameter, match="private"):
        _check_isolation(tmp_path / "b" / "repos", work, tmp_path / "b" / "eval")


def test_run_refuses_a_layout_that_exposes_the_mirrors_and_leaves_no_trace(tmp_path: Path, monkeypatch) -> None:
    _fake_run(monkeypatch)
    tasks, manifest = _frozen(tmp_path, "heldout")
    ledger = tmp_path / "ledger.jsonl"
    args = _run_args(tasks, manifest, tmp_path, ledger, "--allow-heldout", "--cache-dir", str(tmp_path / "repos"),
                     "--work-dir", str(tmp_path / "work"), "--eval-dir", str(tmp_path / "eval"))
    result = runner.invoke(app, args)
    assert result.exit_code != 0 and "walking up" in result.output
    assert not ledger.exists() and not (tmp_path / "r.jsonl.config.json").exists()


def test_report_command_renders_markdown(tmp_path: Path) -> None:
    path = tmp_path / "base.jsonl"
    TraceWriter(path).write(_trace("a", True))
    result = runner.invoke(app, ["bench", "report", str(path)])
    assert result.exit_code == 0 and "1/1" in result.output


def test_derive_drops_tasks_records_reasons_and_parent_hash(tmp_path: Path) -> None:
    tasks, manifest = _frozen(tmp_path, n=5)
    ids = json.loads(manifest.read_text())["task_ids"]
    out = tmp_path / "derived.json"
    result = runner.invoke(app, ["bench", "derive", str(manifest), "--exclude", ids[0], "--reason", "gold hangs", "--out", str(out)])
    assert result.exit_code == 0, result.output
    spec = json.loads(out.read_text())
    assert spec["task_ids"] == ids[1:] and spec["excluded"] == {ids[0]: "gold hangs"}
    assert len(spec["derived_from"]["sha256"]) == 64
    # a derived manifest still passes run-time validation (fails later only for lack of aider/harness here)
    from argus.benchmark.cli import _load_manifest
    _, loaded = _load_manifest(out, tasks, allow_heldout=False)
    assert len(loaded) == 4


def test_derive_rejects_unknown_ids_and_overwrites(tmp_path: Path) -> None:
    _, manifest = _frozen(tmp_path, n=3)
    out = tmp_path / "d.json"
    base = ["bench", "derive", str(manifest), "--reason", "x", "--out", str(out)]
    assert runner.invoke(app, [*base, "--exclude", "ghost"]).exit_code != 0
    ids = json.loads(manifest.read_text())["task_ids"]
    assert runner.invoke(app, [*base, "--exclude", ids[0]]).exit_code == 0
    assert runner.invoke(app, [*base, "--exclude", ids[1]]).exit_code != 0
