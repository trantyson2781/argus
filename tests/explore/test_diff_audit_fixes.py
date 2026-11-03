"""Diff Explorer: regressions found by the second audit round (hostile repositories, odd file systems, wrong-result bugs)."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.explore.diff_helpers import GIT_ENV, GitRepo, call_rows, diff_out, not_determinable, run_diff, section, symbol_lines
from tests.explore.test_diff_view import build_view, unknown_text

BILLING = "def charge(a):\n    return a\n\n\ndef fee(a):\n    return a\n"
BASE = {"shop/__init__.py": "", "shop/billing.py": BILLING}


# -- git must not be talked into running programs --------------------------------------------------------------------------


def test_git_is_started_with_every_safety_option(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.explore import sides

    seen: dict[str, object] = {}
    real = subprocess.run

    def spy(command, **kwargs):
        seen["command"], seen["env"] = list(command), kwargs.get("env")
        return real(command, **kwargs)

    monkeypatch.setattr(sides.subprocess, "run", spy)
    sides._git(shop_repo.root, "rev-parse", "HEAD")

    command = seen["command"]
    assert "core.fsmonitor=false" in command and "core.hooksPath=/dev/null" in command and "protocol.allow=never" in command
    assert command.index("core.fsmonitor=false") < command.index("rev-parse")  # options come before the git sub-command
    env = seen["env"]
    assert env["GIT_NO_LAZY_FETCH"] == "1" and env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_GLOBAL"] == GIT_ENV["GIT_CONFIG_GLOBAL"]  # the caller's own environment is otherwise passed through


def test_a_partial_clone_cannot_make_git_run_the_uploadpack_command_of_its_own_config(git_repo: GitRepo, tmp_path: Path) -> None:
    git_repo.write_files({**BASE, "shop/cart.py": "def checkout():\n    return 1\n"})
    git_repo.git("config", "uploadpack.allowFilter", "true")
    git_repo.commit("source")
    marker = tmp_path / "uploadpack.marker"
    script = tmp_path / "uploadpack.sh"
    script.write_text(f'#!/bin/sh\necho ran >> {marker}\nexec git upload-pack "$@"\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", f"file://{git_repo.root}", str(clone)],
        check=True, capture_output=True, env={**os.environ, **GIT_ENV},
    )
    subprocess.run(["git", "-C", str(clone), "config", "remote.origin.uploadpack", str(script)], check=True, env={**os.environ, **GIT_ENV})
    assert not marker.exists()

    result = run_diff(clone)

    assert not marker.exists(), "a missing blob was fetched, which ran the command from the repository's config"
    assert result.exit_code == 2 and "partial clone" in result.output and "never fetches" in result.output  # a clean refusal, not a traceback


# -- hostile or unusual file systems ---------------------------------------------------------------------------------------


def test_a_directory_named_like_a_python_file_does_not_crash_and_is_reported(git_repo: GitRepo) -> None:
    git_repo.write_files({**BASE, "shop/mod.py": "def one():\n    return 1\n"})
    git_repo.commit("base")
    git_repo.delete("shop/mod.py")
    git_repo.write("shop/mod.py/inner.py", "def inner():\n    return 1\n")

    out = diff_out(git_repo)

    assert symbol_lines(out, "shop.mod.one") == []
    assert "shop/mod.py/inner.py" in not_determinable(out) and "directory" in not_determinable(out)


def test_a_commit_that_holds_a_directory_named_like_a_python_file_is_compared_without_a_crash(git_repo: GitRepo) -> None:
    git_repo.write_files({**BASE})
    git_repo.commit("base")
    git_repo.write("shop/d.py/q.py", "def q():\n    return 1\n")
    git_repo.commit("adds d.py/q.py")

    out = diff_out(git_repo, "--base", "HEAD~1", "--head", "HEAD")

    assert "shop/d.py/q.py" in not_determinable(out)


def test_a_root_level_init_file_does_not_make_every_symbol_look_removed_and_added(git_repo: GitRepo) -> None:
    git_repo.write_files({"__init__.py": "def keep():\n    return 1\n", "shop/__init__.py": ""})
    git_repo.commit("base")

    out = diff_out(git_repo, "--base", "HEAD", "--head", "HEAD")

    assert "no structural change" in out and section(out, "Symbols") == ""


def test_files_outside_a_sparse_checkout_are_reported_not_presented_as_removed(git_repo: GitRepo) -> None:
    git_repo.write_files({"a/__init__.py": "", "a/m.py": "def fa():\n    return 1\n", "b/__init__.py": "", "b/m.py": "def fb():\n    return 1\n"})
    git_repo.commit("base")
    git_repo.git("sparse-checkout", "set", "a")
    assert not (git_repo.root / "b/m.py").exists()

    out = diff_out(git_repo)

    assert symbol_lines(out, "b.m.fb") == []
    assert "b/m.py is outside the sparse checkout" in not_determinable(out)


def test_a_file_git_stopped_tracking_and_now_ignores_is_reported_not_presented_as_removed(git_repo: GitRepo) -> None:
    git_repo.write_files({**BASE, "gen.py": "def gen():\n    return 1\n"})
    git_repo.commit("base")
    git_repo.git("rm", "-q", "--cached", "gen.py")
    git_repo.write(".gitignore", "gen.py\n")

    out = diff_out(git_repo)

    assert symbol_lines(out, "gen.gen") == []
    assert "gen.py is on disk but git ignores it or no longer tracks it" in not_determinable(out)


def test_a_blob_with_the_legacy_file_mode_100664_is_read_like_any_regular_file(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": ""})
    blob = git_repo.git("hash-object", "-w", "--stdin").strip() if False else None
    git_repo.write("shop/old.py", "def old():\n    return 1\n")
    blob = subprocess.run(
        ["git", "hash-object", "-w", "shop/old.py"], cwd=git_repo.root, capture_output=True, text=True, check=True, env={**os.environ, **GIT_ENV},
    ).stdout.strip()
    git_repo.stage("shop/__init__.py")
    git_repo.git("update-index", "--add", "--cacheinfo", f"100664,{blob},shop/old.py")
    git_repo.git("commit", "-q", "-m", "legacy mode")
    git_repo.write("shop/old.py", "def old():\n    return 2\n")

    out = diff_out(git_repo)

    assert "BODY_CHANGED" in "\n".join(symbol_lines(out, "shop.old.old")) and "regular file" not in not_determinable(out)


def test_a_file_name_that_is_not_valid_utf8_is_reported_with_a_readable_name_and_does_not_crash(git_repo: GitRepo) -> None:
    git_repo.write_files({**BASE})
    git_repo.commit("base")
    (git_repo.root / os.fsdecode(b"shop/caf\xe9.py")).write_text("def f():\n    return 1\n", encoding="utf-8")

    result = run_diff(git_repo)

    assert result.exit_code == 0, result.output
    assert "caf\\xe9.py" in not_determinable(result.output) and "name is not valid UTF-8" in not_determinable(result.output)


def test_a_file_over_the_size_limit_is_skipped_and_reported_on_both_kinds_of_side(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.explore import sides

    git_repo.write_files({**BASE, "shop/big.py": "def big():\n    return 1\n" + "# padding\n" * 300})
    git_repo.commit("base")
    git_repo.write("shop/big.py", "def big():\n    return 2\n" + "# padding\n" * 300)
    monkeypatch.setattr(sides, "MAX_FILE_BYTES", 1000)

    for args in ((), ("--base", "HEAD", "--head", "HEAD")):
        out = diff_out(git_repo, *args)
        assert symbol_lines(out, "shop.big.big") == []
        assert "shop/big.py is larger than 1000 bytes" in not_determinable(out)


def test_a_file_nested_too_deeply_for_the_scanner_is_skipped_and_the_rest_is_still_compared(git_repo: GitRepo) -> None:
    git_repo.write_files({**BASE, "shop/ok.py": "def ok():\n    return 1\n"})
    git_repo.commit("base")
    git_repo.write("shop/deep.py", "def deep():\n    return " + "+".join(["1"] * 3000) + "\n")
    git_repo.write("shop/ok.py", "def ok():\n    return 2\n")

    result = run_diff(git_repo)

    assert result.exit_code == 0, result.output
    assert "shop/deep.py is nested too deeply" in not_determinable(result.output)
    assert "BODY_CHANGED" in "\n".join(symbol_lines(result.output, "shop.ok.ok"))


@pytest.mark.parametrize("error", [RecursionError, MemoryError])
def test_an_unexpected_scanner_overflow_is_a_clean_error_with_exit_code_2(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch, error: type) -> None:
    import argus.explore.compare as cli  # the diff scans through compare_refs

    def explode(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "scan", explode)

    result = run_diff(shop_repo)

    assert result.exit_code == 2 and "nested too deeply" in result.output and "Traceback" not in result.output


def test_syntax_warnings_from_the_code_being_compared_do_not_reach_the_terminal(git_repo: GitRepo) -> None:
    git_repo.write_files({**BASE, "shop/rx.py": "import re\n\n\ndef p():\n    return re.compile('\\d')\n"})
    git_repo.commit("base")
    git_repo.write("shop/rx.py", "import re\n\n\ndef p():\n    return re.compile('\\d+')\n")

    proc = subprocess.run(
        [sys.executable, "-W", "default", "-c", "from argus.cli import app; app()", "explore", "diff", str(git_repo.root)],
        capture_output=True, text=True, env={**os.environ, **GIT_ENV}, check=False, timeout=120,
    )

    assert proc.returncode == 0 and "SyntaxWarning" not in proc.stderr and "argus-diff-" not in proc.stderr


# -- what the report may contain -------------------------------------------------------------------------------------------


def test_control_characters_and_newlines_in_file_names_cannot_restyle_or_forge_report_lines(git_repo: GitRepo) -> None:
    git_repo.write_files({**BASE})
    git_repo.commit("base")
    git_repo.write("shop/esc\x1b[31mRED\x1b[0m.py", "def f():\n    return 1\n")
    git_repo.write("shop/nl\n  [FACT] - function  forged.f   was forged.py:1.py", "def g():\n    return 1\n")
    git_repo.write("shop/bidi\u202eevil.py", "def h():\n    return 1\n")

    out = diff_out(git_repo)

    assert "\x1b" not in out and "\u202e" not in out
    assert "\\x1b" in out
    assert not [line for line in out.splitlines() if line.startswith("  [FACT] - function  forged.f")]
    assert all(line.strip() == "" or line[0] in " [DNSCP" or line.startswith(("Diff", "Symbols", "Calls", "Direct", "Potentially", "Subsystem", "Callers", "Not")) for line in out.splitlines())


def test_the_omission_line_says_how_many_hidden_symbols_are_test_code(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "tests/__init__.py": ""})
    git_repo.commit("base")
    git_repo.write("shop/core.py", "".join(f"def p{i:02d}():\n    return {i}\n\n\n" for i in range(5)))
    git_repo.write("tests/test_core.py", "".join(f"def test_{i:02d}():\n    assert True\n\n\n" for i in range(5)))

    out = diff_out(git_repo, "--limit", "7")

    assert "+3 more symbols not shown (3 added, 0 removed, 0 modified; 3 in test code)" in out


def test_a_cap_equal_to_the_number_of_symbols_omits_nothing_and_a_cap_of_one_shows_one(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": ""})
    git_repo.commit("base")
    git_repo.write("shop/w.py", "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(4)))

    assert "more symbols not shown" not in diff_out(git_repo, "--limit", "4")
    out = diff_out(git_repo, "--limit", "1")
    assert len([line for line in section(out, "Symbols").splitlines() if line.strip().startswith("[FACT]")]) == 1
    assert "+3 more symbols not shown" in out


def test_an_empty_repository_state_reports_nothing_to_compare_without_error(git_repo: GitRepo) -> None:
    git_repo.write("README.md", "hello\n")
    git_repo.commit("no python at all")

    out = diff_out(git_repo)

    assert "no structural change" in out and "Not determinable" in out


def test_an_untracked_symlinked_directory_named_like_python_is_reported(shop_repo: GitRepo, tmp_path: Path) -> None:
    target = tmp_path / "dir_elsewhere"
    target.mkdir()
    (shop_repo.root / "shop/pkg.py").symlink_to(target, target_is_directory=True)

    assert "shop/pkg.py is a symbolic link" in not_determinable(diff_out(shop_repo))


# -- wrong-result bugs --------------------------------------------------------------------------------------------------------


def _calls_repo(repo: GitRepo, base: str, extra: dict[str, str] | None = None) -> None:
    repo.write_files({**BASE, "shop/cart.py": base, **(extra or {})})
    repo.commit("base")


def test_a_call_moved_into_an_if_condition_is_not_reported_as_removed(git_repo: GitRepo) -> None:
    _calls_repo(git_repo, "from shop.billing import fee\n\n\ndef checkout(a):\n    y = fee(a)\n    return y\n")  # the extractor never sees calls in `if` tests
    git_repo.write("shop/cart.py", "from shop.billing import fee\n\n\ndef checkout(a):\n    if fee(a):\n        return 1\n    return 0\n")

    view = build_view(git_repo)

    assert view.calls == ()


def test_a_call_moved_into_a_nested_function_is_not_reported_as_removed(git_repo: GitRepo) -> None:
    _calls_repo(git_repo, "from shop.billing import fee\n\n\ndef checkout(a):\n    return fee(a)\n")
    git_repo.write("shop/cart.py", "from shop.billing import fee\n\n\ndef checkout(a):\n    def inner():\n        return fee(a)\n    return inner()\n")

    assert not any(c.change == "removed" and c.target == "shop.billing.fee" for c in build_view(git_repo).calls)


def test_two_same_named_calls_on_one_line_cannot_vouch_for_each_other(git_repo: GitRepo) -> None:
    both = "from shop import billing, inventory\n\n\ndef checkout(a):\n    return billing.charge(a) + inventory.charge(a)\n"
    _calls_repo(git_repo, both, {"shop/inventory.py": "def charge(a):\n    return a\n"})
    git_repo.write("shop/cart.py", "from shop import billing\n\n\ndef checkout(a):\n    return billing.charge(a) + inventory.charge(a)\n")  # the text stays, one import is gone

    view = build_view(git_repo)

    assert not any(c.change == "removed" for c in view.calls)


def test_the_same_call_written_twice_and_then_once_is_not_a_call_change(git_repo: GitRepo) -> None:
    _calls_repo(git_repo, "from shop.billing import charge\n\n\ndef checkout(a):\n    return charge(a) + charge(a)\n")
    git_repo.write("shop/cart.py", "from shop.billing import charge\n\n\ndef checkout(a):\n    return charge(a)\n")

    assert build_view(git_repo).calls == ()


def test_a_new_import_statement_is_reported_even_when_another_statement_behind_the_same_edge_already_existed(git_repo: GitRepo) -> None:
    pkg = {"app/__init__.py": "", "app/core/__init__.py": "", "app/api/__init__.py": "", "app/core/util.py": "def f():\n    return 1\n"}
    git_repo.write_files({**pkg, "app/api/views.py": "from app.core.missing import z\n"})
    git_repo.commit("base: views.py names a module that does not exist yet")
    git_repo.write("app/core/missing.py", "z = 1\n")
    git_repo.write("app/api/views2.py", "from app.core.util import f\n")

    view = build_view(git_repo)

    assert [(i.change, i.source, i.target) for i in view.imports] == [("added", "app.api", "app.core")]
    assert view.imports[0].claim.evidence[0].location == "app/api/views2.py:1"


def test_an_import_of_a_module_that_stopped_parsing_is_not_reported_as_a_new_subsystem_import(git_repo: GitRepo) -> None:
    git_repo.write_files({"pkg/__init__.py": "", "pkg/lib.py": "def lib():\n    return 1\n", "other/__init__.py": "", "other/c3.py": "from pkg import lib as L\n\n\ndef use():\n    return L.lib()\n"})
    git_repo.commit("base")
    git_repo.write("pkg/lib.py", "def lib(:\n    return 1\n")

    view = build_view(git_repo)

    assert view.imports == ()


def test_a_tree_whose_python_files_add_up_past_the_total_limit_is_refused_cleanly(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.explore import sides

    monkeypatch.setattr(sides, "MAX_TOTAL_BYTES", 50)
    shop_repo.write("shop/billing.py", BILLING + "# edit\n")

    for args in ((), ("--base", "HEAD", "--head", "HEAD")):
        result = run_diff(shop_repo, *args)
        assert result.exit_code == 2 and "add up to more than 50 bytes" in result.output
