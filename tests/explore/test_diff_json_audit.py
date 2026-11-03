"""`explore diff --json`, audit round 1: evidence sides of calls_claim, the crash path, and parts of the document the first tests left unchecked."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest

from argus.explore import cli, compare
from argus.explore.jsonout import file_change_side
from tests.explore.diff_helpers import GIT_ENV, GitRepo, run_diff

BASE = "def f():\n    return 1\n\n\ndef g():\n    return 2\n\n\ndef h():\n    x = 0\n    return f()\n"


def _doc(repo: GitRepo, *args: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = run_diff(repo, "--json", *args, env=env)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _evidence_lines(repo: GitRepo, doc: dict[str, Any], qualname: str) -> list[tuple[str, str]]:
    """(side, the source line the evidence points at) for every evidence item of one symbol's calls_claim, read from the real file of that side."""
    (symbol,) = [s for s in doc["data"]["symbols"] if s["qualname"] == qualname]
    out = []
    for ev in symbol["calls_claim"]["evidence"]:
        text = repo.git("show", f"HEAD:{ev['file']}") if ev["side"] == "base" else (repo.root / ev["file"]).read_text(encoding="utf-8")
        out.append((ev["side"], text.splitlines()[ev["line"] - 1].strip()))
    return out


@pytest.fixture
def calls_repo(git_repo: GitRepo) -> GitRepo:
    git_repo.write_files({"pkg/__init__.py": "", "pkg/m.py": BASE})
    git_repo.commit("base")
    return git_repo


@pytest.mark.parametrize(
    ("new_h", "expected"),
    [
        ("def h():\n    x = 0\n    return f() + g()\n", [("head", "return f() + g()")]),  # an added call: read from the head file
        ("def h():\n    x = 0\n    return 5\n", [("base", "return f()")]),  # a removed call: read from the base file
        ("def h():\n    x = 0\n    return g()\n", [("base", "return f()"), ("head", "return g()")]),  # both in one claim
    ],
    ids=["added", "removed", "mixed"],
)
def test_calls_claim_evidence_is_labelled_with_the_side_it_was_read_from(calls_repo: GitRepo, new_h: str, expected: list[tuple[str, str]]) -> None:
    calls_repo.write("pkg/m.py", BASE.split("def h():")[0] + new_h)

    got = _evidence_lines(calls_repo, _doc(calls_repo), "pkg.m.h")

    assert sorted(got) == sorted(expected)  # the line each label points at really holds the call


def test_a_removed_call_row_names_the_call_and_not_only_the_side(calls_repo: GitRepo) -> None:
    calls_repo.write("pkg/m.py", BASE.replace("return f()", "return 5"))

    rows = [(c["change"], c["source"], c["target"], c["claim"]["evidence"][0]["side"]) for c in _doc(calls_repo)["data"]["calls"]]

    assert rows == [("removed", "pkg.m.h", "pkg.m.f", "base")]


# -- the crash path ----------------------------------------------------------------------------------------------------------------


def test_an_unexpected_error_in_json_mode_is_a_json_error_without_the_message_text(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("secret detail /home/someone/key")

    monkeypatch.setattr(compare, "build_diff_view", boom)

    result = run_diff(shop_repo, "--json")

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["ok"] is False and doc["error"] == {"kind": "internal_error", "message": "RuntimeError"}
    assert "secret" not in result.stdout


@pytest.mark.parametrize("error", [RecursionError, MemoryError])
def test_deep_nesting_in_json_mode_is_a_json_error(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise error()

    monkeypatch.setattr(compare, "build_diff_view", boom)

    result = run_diff(shop_repo, "--json")

    assert result.exit_code == 2 and json.loads(result.stdout)["error"]["kind"] == "too_deeply_nested"


def test_an_unexpected_error_in_text_mode_is_not_swallowed(shop_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(compare, "build_diff_view", boom)

    assert isinstance(run_diff(shop_repo).exception, RuntimeError)


def test_json_and_a_limit_that_the_cli_library_rejects_leave_stdout_empty(shop_repo: GitRepo) -> None:
    result = run_diff(shop_repo, "--json", "--limit", "0")  # usage errors come from Typer as plain text on stderr; documented in the schema

    assert result.exit_code == 2 and result.stdout == ""


def test_an_unknown_file_change_kind_is_rejected_not_guessed() -> None:
    with pytest.raises(ValueError, match="mystery"):
        file_change_side("mystery", "a.py", frozenset())  # type: ignore[arg-type]


# -- parts of the document the first tests did not check ---------------------------------------------------------------------------


def test_a_body_only_change_puts_the_callers_in_affected_callers_with_only_the_body_flag(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount + 5\n\n\ndef fee(amount):\n    return amount\n")

    data = _doc(shop_repo)["data"]

    assert data["interface_callers"] == []
    edges = [e for g in data["affected_callers"] for e in g["edges"]]
    assert edges and all(e["callee_flags"] == ["BODY_CHANGED"] for e in edges)
    assert {g["caller"] for g in data["affected_callers"]} >= {"shop.cart.checkout"}


def test_an_untracked_file_is_marked_and_listed_in_new_files(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/extra.py", "def extra():\n    return 1\n")

    data = _doc(shop_repo)["data"]

    assert data["untracked_files"] == 1 and data["new_files"] == ["shop/extra.py"]
    (symbol,) = [s for s in data["symbols"] if s["qualname"] == "shop.extra.extra"]
    assert symbol["untracked"] is True and symbol["claim"]["evidence"][0]["side"] == "head"


def test_an_added_import_cites_the_head_side(git_repo: GitRepo) -> None:
    from tests.explore.test_diff_imports import AUDIT_LOG, PKG

    git_repo.write_files({**PKG, "shop/cart/a.py": "X = 1\n", "shop/audit/log.py": AUDIT_LOG})
    git_repo.commit("base")
    git_repo.write("shop/cart/a.py", "X = 1\nfrom shop.audit.log import record\n")

    (item,) = _doc(git_repo)["data"]["imports"]

    assert (item["change"], item["source"], item["target"]) == ("added", "shop.cart", "shop.audit")
    assert item["claim"]["evidence"][0]["side"] == "head"


def test_an_explicit_head_ref_and_the_v1_resolver_are_echoed(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount + 5\n\n\ndef fee(amount):\n    return amount\n")
    shop_repo.commit("edit")

    doc = _doc(shop_repo, "--base", "HEAD~1", "--head", "HEAD", env={"ARGUS_RESOLVE_V2": "0"})

    assert doc["options"] == {"base": "HEAD~1", "head": "HEAD", "include_tests": False}
    assert doc["resolver"] == "calls-v1" and doc["data"]["head_label"].startswith("HEAD (")


def test_the_output_does_not_depend_on_the_hash_seed(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount, currency=None):\n    return amount + 5\n")
    outputs = set()
    for seed in ("1", "2", "3"):
        proc = subprocess.run(
            [sys.executable, "-c", "from argus.cli import app; app()", "explore", "diff", str(shop_repo.root), "--json"],
            capture_output=True, text=True, check=True, timeout=120, env={**os.environ, **GIT_ENV, "PYTHONHASHSEED": seed},
        )
        outputs.add(proc.stdout)

    assert len(outputs) == 1
