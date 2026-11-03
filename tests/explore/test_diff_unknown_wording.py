"""Diff Explorer: the Not determinable lines about caller coverage say what to do next (a ready-to-run `git grep`)."""

from __future__ import annotations

import shlex
import subprocess

from tests.explore.diff_helpers import GitRepo, diff_out, not_determinable

GREP_PREFIX = "To look for callers Argus missed, search the code for the changed names (from the repository root): git grep -n"


def _grep_line(out: str) -> str | None:
    return next((row.strip() for row in not_determinable(out).splitlines() if GREP_PREFIX in row), None)


def test_a_signature_change_gives_a_git_grep_for_the_changed_name(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount, currency=None):\n    return amount + 1\n\n\ndef fee(amount):\n    return amount\n")

    line = _grep_line(diff_out(shop_repo))

    assert line is not None and line.startswith("[UNKNOWN]")
    assert line.endswith("git grep -n -e 'charge('")  # -e per pattern: a bare second pattern would be read as a path


def test_a_removed_function_is_included_and_names_are_sorted_and_unique(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount, currency=None):\n    return amount + 1\n")

    line = _grep_line(diff_out(shop_repo))

    assert line is not None and line.endswith("git grep -n -e 'charge(' -e 'fee('")


def test_a_body_only_change_gives_no_git_grep_line(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount + 5\n\n\ndef fee(amount):\n    return amount\n")

    assert _grep_line(diff_out(shop_repo)) is None


def test_more_than_five_names_are_cut_with_an_exact_count(git_repo: GitRepo) -> None:
    before = "".join(f"def f{i}(a):\n    return a\n\n\n" for i in range(7))
    git_repo.write_files({"pkg/__init__.py": "", "pkg/m.py": before})
    git_repo.commit("base")
    git_repo.write("pkg/m.py", before.replace("(a)", "(a, b=None)"))

    line = _grep_line(diff_out(git_repo))

    assert line is not None
    assert line.count(" -e ") == 5 and line.endswith("(and 2 more names)")


def test_a_changed_constructor_searches_for_the_class_name(git_repo: GitRepo) -> None:
    git_repo.write_files({"pkg/__init__.py": "", "pkg/m.py": "class Cart:\n    def __init__(self):\n        self.x = 1\n"})
    git_repo.commit("base")
    git_repo.write("pkg/m.py", "class Cart:\n    def __init__(self, owner):\n        self.x = 1\n")

    line = _grep_line(diff_out(git_repo))

    assert line is not None and line.endswith("git grep -n -e 'Cart('")


def test_the_printed_git_grep_runs_and_finds_the_known_caller(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount, currency=None):\n    return amount + 1\n\n\ndef fee(amount):\n    return amount\n")
    line = _grep_line(diff_out(shop_repo))
    assert line is not None
    command = shlex.split(line.split("git grep", 1)[1])

    hits = subprocess.run(["git", "grep", *command], cwd=shop_repo.root, capture_output=True, text=True, check=True).stdout

    assert "shop/cart.py" in hits  # cart.checkout calls charge(


def test_the_caller_coverage_line_is_short_plain_and_names_the_if_condition_gap(shop_repo: GitRepo) -> None:
    text = not_determinable(diff_out(shop_repo))

    assert "The caller list may be incomplete" in text
    assert "inside an if/elif condition" in text
    assert "a missing caller does not mean there is none" in text
    assert "by design" not in text and "scanner gap" not in text


def test_the_unresolved_calls_line_says_what_it_risks(git_repo: GitRepo) -> None:
    git_repo.write_files({
        "shop/__init__.py": "", "shop/cart.py": "def checkout(client, amount):\n    return client.charge(amount)\n",
    })
    git_repo.commit("base")
    git_repo.write("shop/cart.py", "def checkout(client, amount):\n    return client.charge(amount) + 1\n")

    line = next(r.strip() for r in not_determinable(diff_out(git_repo)).splitlines() if "could not be traced" in r)

    assert line == (
        "[UNKNOWN] 1 call made by the changed functions could not be traced (e.g. client.charge): "
        "if it reaches a changed function, that caller is missing from the lists above"
    )
