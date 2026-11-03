"""Diff Explorer: display caps and determinism. The view always holds everything; only the printed report is capped."""

from __future__ import annotations

import os
import subprocess
import sys

from tests.explore.diff_helpers import GitRepo, diff_out, run_diff, section
from tests.explore.test_diff_view import build_view

MANY = 60


def _many_added(repo: GitRepo, count: int = MANY) -> None:
    repo.write_files({"shop/__init__.py": "", "shop/wide.py": ""})  # wide.py exists already: a brand-new big file is shown as one row
    repo.commit("base")
    repo.write("shop/wide.py", "".join(f"def f{i:03d}():\n    return {i}\n\n\n" for i in range(count)))


def _symbol_rows(out: str) -> list[str]:
    return [line for line in section(out, "Symbols").splitlines() if line.strip().startswith("[FACT]")]


def test_the_symbols_section_is_capped_by_default_and_reports_the_exact_omitted_count(git_repo: GitRepo) -> None:
    _many_added(git_repo)

    out = diff_out(git_repo)

    rows = _symbol_rows(out)
    assert len(rows) == 50
    assert f"Symbols ({MANY})" in out  # the title keeps the true total
    assert "+10 more symbols not shown (10 added, 0 removed, 0 modified)" in section(out, "Symbols")


def test_the_cap_keeps_the_first_symbols_in_the_view_order_so_the_output_is_a_deterministic_prefix(git_repo: GitRepo) -> None:
    _many_added(git_repo)
    view = build_view(git_repo)

    rows = _symbol_rows(diff_out(git_repo))

    assert len(view.symbols) == MANY  # the view model is complete
    assert [r.split()[3] for r in rows] == [s.qualname for s in view.symbols][:50]


def test_when_the_cap_bites_production_symbols_are_kept_before_test_symbols_and_rows_stay_in_file_order(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "tests/__init__.py": "", "tests/test_a.py": "", "aaa/__init__.py": "", "aaa/test_first.py": "", "shop/core.py": ""})
    git_repo.commit("base")
    git_repo.write("tests/test_a.py", "".join(f"def test_{i:02d}():\n    assert True\n\n\n" for i in range(30)))  # sorts before shop/? no: after, but see below
    git_repo.write("aaa/__init__.py", "")
    git_repo.write("aaa/test_first.py", "".join(f"def test_x{i:02d}():\n    assert True\n\n\n" for i in range(30)))  # sorts first, but is test code
    git_repo.write("shop/core.py", "".join(f"def prod_{i:02d}():\n    return {i}\n\n\n" for i in range(30)))

    out = diff_out(git_repo, "--limit", "30")
    rows = [r.split()[3] for r in _symbol_rows(out)]

    assert rows == [f"shop.core.prod_{i:02d}" for i in range(30)]  # all 30 shown rows are production; 60 test symbols are omitted
    assert "+60 more symbols not shown (60 added, 0 removed, 0 modified; 60 in test code)" in out


def test_all_shows_every_symbol_and_no_omission_line(git_repo: GitRepo) -> None:
    _many_added(git_repo)

    out = diff_out(git_repo, "--all")

    assert len(_symbol_rows(out)) == MANY and "more symbols not shown" not in out


def test_limit_sets_the_cap_for_symbols_as_well_as_callers(git_repo: GitRepo) -> None:
    _many_added(git_repo)

    out = diff_out(git_repo, "--limit", "5")

    assert len(_symbol_rows(out)) == 5 and "+55 more symbols not shown" in out


def test_omitted_symbols_are_counted_by_kind_of_change(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/new.py": "", "shop/old.py": "".join(f"def g{i:03d}():\n    return {i}\n\n\n" for i in range(30))})
    git_repo.commit("base")
    git_repo.write("shop/old.py", "".join(f"def g{i:03d}():\n    return {i + 1}\n\n\n" for i in range(30)))  # 30 modified
    git_repo.write("shop/new.py", "".join(f"def h{i:03d}():\n    return {i}\n\n\n" for i in range(30)))  # 30 added

    out = diff_out(git_repo, "--limit", "40")  # review order puts the 30 modified first, so the cap keeps them and cuts 20 of the 30 added

    assert "+20 more symbols not shown (20 added, 0 removed, 0 modified)" in out


def test_the_cap_never_hides_the_caller_sections_or_the_not_determinable_section(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/wide.py": "", "shop/core.py": "def target():\n    return 1\n", "shop/use.py": "from shop.core import target\n\n\ndef user():\n    return target()\n"})
    git_repo.commit("base")
    git_repo.write("shop/core.py", "def target(x):\n    return 1\n")
    git_repo.write("shop/wide.py", "".join(f"def f{i:03d}():\n    return {i}\n\n\n" for i in range(MANY)))
    git_repo.write("shop/odd.py", b"# caf\xe9\ndef odd():\n    return 1\n")

    out = diff_out(git_repo)

    assert "shop.use.user" in section(out, "Direct callers of changed interfaces")
    assert "more symbols not shown" in section(out, "Symbols")
    assert "shop/odd.py is not valid UTF-8" in section(out, "Not determinable")


def test_the_printed_report_is_identical_across_hash_seeds(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/cart/__init__.py": "", "shop/audit/__init__.py": "", "shop/a.py": "def one(x):\n    return x\n"})
    git_repo.commit("base")
    git_repo.write("shop/wide.py", "".join(f"def f{i:03d}(x):\n    return one(x)\n\n\n" for i in range(40)))
    git_repo.write("shop/a.py", "def one(x, y):\n    return x\n")
    git_repo.write("shop/cart/m.py", "from shop.audit import X\nfrom shop.a import one\n\n\ndef c():\n    return one(1, 2)\n")
    outputs = set()
    for seed in ("0", "1", "42", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", "from argus.cli import app; app()", "explore", "diff", str(git_repo.root), "--all"],
            capture_output=True, text=True, env=env, check=False, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Symbols" in proc.stdout and "Not determinable" in proc.stdout
        outputs.add(proc.stdout)
    assert len(outputs) == 1


def test_every_symbol_in_the_view_is_either_printed_or_counted_as_omitted(git_repo: GitRepo) -> None:
    _many_added(git_repo)
    view = build_view(git_repo)

    out = diff_out(git_repo, "--limit", "7")

    assert len(_symbol_rows(out)) + 53 == len(view.symbols)
    assert "+53 more symbols not shown" in out


def test_caller_coverage_warning_is_always_in_not_determinable(shop_repo: GitRepo) -> None:
    for out in (diff_out(shop_repo), _after_body_edit(shop_repo)):
        text = section(out, "Not determinable")
        assert "The caller list may be incomplete" in text
        assert "a missing caller does not mean there is none" in text


def _after_body_edit(repo: GitRepo) -> str:
    repo.write("shop/billing.py", "def charge(amount):\n    return amount + 5\n\n\ndef fee(amount):\n    return amount\n")
    return diff_out(repo)


def test_limit_zero_and_conflicting_flags_still_exit_with_code_2(shop_repo: GitRepo) -> None:
    assert run_diff(shop_repo, "--limit", "0").exit_code == 2
    assert run_diff(shop_repo, "--all", "--limit", "3").exit_code == 2
