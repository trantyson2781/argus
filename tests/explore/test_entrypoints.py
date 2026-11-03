from pathlib import Path

from argus.explore.claims import Tier
from argus.explore.entrypoints import find_entrypoints
from argus.extract import scan


def _find(root: Path):
    return find_entrypoints(root, scan(root))


def test_script_target_is_read_from_pyproject(mini_repo: Path) -> None:
    eps = _find(mini_repo)
    assert [(s.name, s.module, s.attr) for s in eps.scripts] == [("mini", "mini.cli", "app")]
    assert eps.scripts[0].claim.tier is Tier.FACT
    assert eps.scripts[0].claim.evidence[0].file == "pyproject.toml"


def test_command_tree_follows_add_typer_and_name_overrides(mini_repo: Path) -> None:
    eps = _find(mini_repo)
    by_path = {c.display: c for c in eps.commands}
    assert set(by_path) == {"mini hello-world", "mini sub renamed"}
    hello = by_path["mini hello-world"]
    assert hello.function_id == "function:mini.cli.hello_world"
    assert (hello.claim.tier, hello.file) == (Tier.FACT, "src/mini/cli.py")
    assert by_path["mini sub renamed"].function_id == "function:mini.sub.go"  # relative import resolved


def test_command_evidence_points_at_decorator_and_registration(mini_repo: Path) -> None:
    claim = {c.display: c for c in _find(mini_repo).commands}["mini sub renamed"].claim
    lines = {(e.file, e.line) for e in claim.evidence}
    assert ("src/mini/sub.py", 9) in lines  # @sub_app.command("renamed")
    assert ("src/mini/cli.py", 8) in lines  # app.add_typer(sub_app, name='sub')


def test_unresolvable_subapp_is_unknown_not_guessed(mini_repo: Path) -> None:
    eps = _find(mini_repo)
    assert any("ext_app" in c.text and c.tier is Tier.UNKNOWN for c in eps.unknowns)
    assert not any("ext" in c.display.split() for c in eps.commands)


def test_main_guard_detected(mini_repo: Path) -> None:
    guards = _find(mini_repo).main_guards
    assert len(guards) == 1 and guards[0].evidence[0] == guards[0].evidence[0].__class__("src/mini/__main__.py", 3, guards[0].evidence[0].source_hash)


def test_no_scripts_falls_back_to_root_apps(mini_repo: Path) -> None:
    (mini_repo / "pyproject.toml").write_text('[project]\nname = "mini"\nversion = "0"\n')
    eps = _find(mini_repo)
    assert eps.scripts == ()
    assert {c.display for c in eps.commands} >= {"mini.cli:app hello-world"}


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


SCRIPT = '[project]\nname = "p"\nversion = "0"\n[project.scripts]\n'


def test_annotated_assignment_and_if_block_apps_are_found(tmp_path: Path) -> None:
    _write(tmp_path, {
        "pyproject.toml": SCRIPT + 'p = "p.cli:app"\n',
        "p/__init__.py": "",
        "p/cli.py": "import typer\nif True:\n    app: typer.Typer = typer.Typer()\n\n\n@app.command()\ndef go():\n    pass\n",
    })
    assert [c.display for c in _find(tmp_path).commands] == ["p go"]


def test_module_attribute_subapp_and_relative_import_levels(tmp_path: Path) -> None:
    _write(tmp_path, {
        "pyproject.toml": SCRIPT + 'p = "p.cli:app"\n',
        "p/__init__.py": "",
        "p/cli.py": "import typer\nfrom . import sub\n\napp = typer.Typer()\napp.add_typer(sub.app, name='s')\n",
        "p/sub/__init__.py": "import typer\nfrom ..deep import inner\n\napp = typer.Typer()\napp.add_typer(inner.app, name='in')\n\n\n@app.command()\ndef top():\n    pass\n",
        "p/deep/__init__.py": "",
        "p/deep/inner.py": "import typer\napp = typer.Typer()\n\n\n@app.command()\ndef leaf():\n    pass\n",
    })
    assert {c.display for c in _find(tmp_path).commands} == {"p s top", "p s in leaf"}


def test_non_literal_names_and_orphan_decorators_are_unknown(tmp_path: Path) -> None:
    _write(tmp_path, {
        "pyproject.toml": SCRIPT + 'p = "p.cli:app"\n',
        "p/__init__.py": "",
        "p/cli.py": "import typer\nfrom other import mystery\n\nNAME = 'x'\napp = typer.Typer()\n\n\n@app.command(NAME)\ndef dyn():\n    pass\n\n\n@mystery.command()\ndef orphan():\n    pass\n",
    })
    eps = _find(tmp_path)
    assert eps.commands == ()
    texts = " ".join(c.text for c in eps.unknowns)
    assert "non-literal command name" in texts and "mystery" in texts


def test_same_subapp_under_two_parents_is_listed_twice_and_cycles_terminate(tmp_path: Path) -> None:
    _write(tmp_path, {
        "pyproject.toml": SCRIPT + 'p = "p.cli:app"\n',
        "p/__init__.py": "",
        "p/cli.py": (
            "import typer\nfrom p.sub import sub_app\n\napp = typer.Typer()\nother = typer.Typer()\n"
            "app.add_typer(sub_app, name='a')\napp.add_typer(other, name='o')\nother.add_typer(sub_app, name='b')\n"
            "sub_app.add_typer(app, name='loop')\n"
        ),
        "p/sub.py": "import typer\nsub_app = typer.Typer()\n\n\n@sub_app.command()\ndef run():\n    pass\n",
    })
    assert {c.display for c in _find(tmp_path).commands} == {"p a run", "p o b run"}


def test_script_evidence_line_matches_exact_key_not_a_prefix_sibling(tmp_path: Path) -> None:
    _write(tmp_path, {
        "pyproject.toml": SCRIPT + 'p-dev = "p.cli:app"\np = "p.cli:app"\n',
        "p/__init__.py": "",
        "p/cli.py": "import typer\napp = typer.Typer()\n",
    })
    lines = {s.name: s.claim.evidence[0].line for s in _find(tmp_path).scripts}
    assert lines == {"p": 6, "p-dev": 5}


def test_tests_directory_apps_are_not_reported(mini_repo: Path) -> None:
    _write(mini_repo, {"tests/test_app.py": "import typer\nfake = typer.Typer()\n"})
    assert not any("fake" in c.text for c in _find(mini_repo).unknowns)
