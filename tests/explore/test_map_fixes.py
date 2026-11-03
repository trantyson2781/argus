"""Architecture map: single-file scripts, the 78-column promise for long names, and one-package repositories."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from argus.cli import app
from argus.explore.ascii import MAX_WIDTH

runner = CliRunner()

SOLO = {
    "solo/__init__.py": '"""Solo: the only package."""\n',
    "solo/core/__init__.py": '"""Core logic."""\n',
    "solo/core/engine.py": "def run():\n    return 1\n",
    "solo/ui/__init__.py": '"""User interface."""\n',
    "solo/ui/view.py": "from solo.core.engine import run\n\n\ndef show():\n    return run()\n",
    "solo/cli.py": '"""Command line."""\nfrom solo.ui.view import show\n\n\ndef main():\n    show()\n',
}


def _repo(tmp_path: Path, files: dict[str, str], name: str = "demo") -> Path:
    root = tmp_path / name
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _map(root: Path, *args: str, code: int = 0) -> str:
    result = runner.invoke(app, ["explore", "map", str(root), *args])
    assert result.exit_code == code, result.output
    return result.output


def _box(out: str, title: str) -> str:
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"+- {title} "))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("+-") or not lines[i].strip())
    return "\n".join(lines[start:end])


# -- a single-file script is not a package ---------------------------------------------------------------------------------------


def test_a_single_file_script_is_not_shown_as_holding_an_init_module(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"pkg/__init__.py": '"""A package."""\n', "pkg/a.py": "x = 1\n", "bot.py": '"""A script."""\nprint(1)\n'})

    out = _map(root, "--view", "system")

    assert "__init__" not in _box(out, "bot")
    assert "(single module)" in _box(out, "bot")
    assert "__init__" in _box(out, "pkg")  # a real package keeps its real __init__


# -- the header never breaks the width promise -------------------------------------------------------------------------------------


def test_a_long_directory_name_cannot_push_any_line_past_the_width_limit(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"}, name="a_very_long_project_directory_name_" * 3)

    for view in ("system", "deps"):
        out = _map(root, "--view", view)
        assert max(len(line) for line in out.splitlines()) <= MAX_WIDTH, view
        assert out.startswith("SYSTEM MAP  a_very_long" if view == "system" else "DEPENDENCY MAP  a_very_long")


def test_a_short_directory_name_is_shown_in_full(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"}, name="shortname")

    assert "SYSTEM MAP  shortname  |  " in _map(root, "--view", "system")


# -- one package: its parts become the subsystems -----------------------------------------------------------------------------------


def test_the_parts_of_a_lone_package_get_their_own_boxes_and_arrows(tmp_path: Path) -> None:
    out = _map(_repo(tmp_path, SOLO), "--view", "system")

    for title in ("solo.core", "solo.ui", "solo.cli"):
        assert f"+- {title} " in out
    assert "solo.ui ---> solo.core" in out and "solo.cli ---> solo.ui" in out
    assert "(none between subsystems" not in out
    assert "engine" in _box(out, "solo.core")  # modules are listed inside their part
    assert max(len(line) for line in out.splitlines()) <= MAX_WIDTH


def test_a_lone_package_whose_parts_do_not_import_each_other_stays_one_box(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"solo/__init__.py": "", "solo/a.py": "x = 1\n", "solo/b.py": "y = 2\n"})

    out = _map(root, "--view", "system")

    assert "+- solo " in out and "+- solo.a " not in out


def test_two_top_level_packages_are_unchanged(tmp_path: Path) -> None:
    files = {**SOLO, "other/__init__.py": "", "other/m.py": "from solo.core.engine import run\n"}

    out = _map(_repo(tmp_path, files), "--view", "system")

    assert "+- solo " in out and "+- other " in out and "+- solo.core " not in out


def test_opening_a_part_of_a_lone_package_expands_it_and_opening_the_wrapper_expands_every_part(tmp_path: Path) -> None:
    root = _repo(tmp_path, SOLO)

    one = _map(root, "--view", "system", "--open", "solo.core")
    everything = _map(root, "--view", "system", "--open", "solo")

    assert "Core logic." in _box(one, "solo.core") and "engine" in _box(one, "solo.core")
    assert "Core logic." in _box(everything, "solo.core") and "User interface." in _box(everything, "solo.ui")
