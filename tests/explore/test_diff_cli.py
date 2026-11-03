"""Diff Explorer v0.1, slices 7 and 8: the printed report and the `argus explore diff` command (contract: see diff_helpers.py)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.explore.diff_helpers import BILLING, GitRepo, diff_out, remove_fee_scenario, run_diff, section
from tests.explore.test_diff_view import CALLERS, BILLING3, build_view

EDIT = BILLING.replace("amount + 1", "amount + 2")


# -- command behaviour ------------------------------------------------------------------------------------------------------


def test_unknown_base_and_head_refs_exit_with_code_2_and_a_message(shop_repo: GitRepo) -> None:
    for args in (["--base", "no-such-ref"], ["--head", "no-such-head"]):
        result = run_diff(shop_repo, *args)
        assert result.exit_code == 2 and args[1] in result.output


@pytest.mark.parametrize("args", [["--all", "--limit", "5"], ["--limit", "0"], ["--limit", "-3"]])
def test_conflicting_or_invalid_limit_options_exit_with_code_2(shop_repo: GitRepo, args: list[str]) -> None:
    assert run_diff(shop_repo, *args).exit_code == 2


def test_a_directory_that_is_not_a_git_repository_or_does_not_exist_exits_with_code_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    (tmp_path / "plain").mkdir()
    for target in (tmp_path / "plain", tmp_path / "missing"):
        result = run_diff(target)
        assert result.exit_code == 2 and "Traceback" not in result.output


def test_options_that_look_like_git_flags_are_refused_not_passed_to_git(shop_repo: GitRepo, tmp_path: Path) -> None:
    result = run_diff(shop_repo, "--base", f"--output={tmp_path / 'pwned'}")
    assert result.exit_code == 2 and not (tmp_path / "pwned").exists()


def test_the_command_writes_nothing_into_the_repository_and_leaves_no_temporary_files(shop_repo: GitRepo, scratch_tmp: Path) -> None:
    shop_repo.write("shop/billing.py", EDIT)
    shop_repo.write("shop/new.py", "def n():\n    return 1\n")
    before = (shop_repo.git("status", "--porcelain", "--ignored"), sorted(p.name for p in shop_repo.root.rglob("*")), (shop_repo.root / "shop/billing.py").read_bytes())

    diff_out(shop_repo)

    after = (shop_repo.git("status", "--porcelain", "--ignored"), sorted(p.name for p in shop_repo.root.rglob("*")), (shop_repo.root / "shop/billing.py").read_bytes())
    assert before == after and list(scratch_tmp.iterdir()) == []


def test_two_runs_on_the_same_state_are_byte_identical(git_repo: GitRepo) -> None:
    remove_fee_scenario(git_repo)

    assert diff_out(git_repo) == diff_out(git_repo)


def test_head_ref_compares_two_commits(shop_repo: GitRepo) -> None:
    first = shop_repo.sha()
    shop_repo.write("shop/billing.py", EDIT)
    shop_repo.commit("second")
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 9"))

    out = diff_out(shop_repo, "--base", first, "--head", "HEAD")

    assert f"{first} ({first[:7]}) -> HEAD (" in out.splitlines()[0] and "shop.billing.charge" in out


def test_output_has_no_terminal_escape_codes_and_keeps_literal_square_brackets(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", EDIT)

    out = diff_out(shop_repo)

    assert "\x1b" not in out and "[FACT] ~ BODY_CHANGED" in out and "[UNKNOWN]" in out


# -- limits and test callers -----------------------------------------------------------------------------------------------


MANY = {"shop/__init__.py": "", "shop/billing.py": BILLING3, **{f"shop/c{i}.py": f"from shop.billing import fee\n\n\ndef caller{i}(a):\n    return fee(a)\n" for i in range(1, 5)}}


def _many_callers(repo: GitRepo) -> None:
    repo.write_files(MANY)
    repo.commit("base")
    repo.write("shop/billing.py", BILLING3.replace("def fee(amount):\n    return amount", "def fee(amount):\n    return amount + 1"))


def _group_names(out: str, title: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"^  (shop\.\S+)", section(out, title), re.MULTILINE)]


def test_caller_groups_are_limited_per_section_with_an_exact_remainder_and_all_shows_everything(git_repo: GitRepo) -> None:
    _many_callers(git_repo)

    limited = diff_out(git_repo, "--limit", "3")
    everything = diff_out(git_repo, "--all")
    default = diff_out(git_repo)

    assert _group_names(limited, "Potentially affected") == ["shop.c1.caller1", "shop.c2.caller2", "shop.c3.caller3"]
    assert "+1 more" in section(limited, "Potentially affected")
    assert len(_group_names(everything, "Potentially affected")) == 4 and "more" not in section(everything, "Potentially affected")
    assert default == everything  # 4 groups are under the default cap of 20


def test_test_callers_are_hidden_with_a_count_and_shown_with_include_tests(shop_repo: GitRepo) -> None:
    shop_repo.write("tests/test_billing.py", "from shop.billing import charge\n\n\ndef test_charge():\n    assert charge(1) == 2\n")
    shop_repo.commit("add a test")
    shop_repo.write("shop/billing.py", EDIT)

    hidden, shown = diff_out(shop_repo), diff_out(shop_repo, "--include-tests")

    assert "1 test caller group hidden (use --include-tests)" in hidden and "test_charge" not in hidden
    assert "tests.test_billing.test_charge (test code)" in shown and "test caller group" not in shown


# -- notes and wording -----------------------------------------------------------------------------------------------------


def test_body_changed_is_described_as_structural_detection_never_as_behavior(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", EDIT)

    out = diff_out(shop_repo)

    assert "BODY_CHANGED means the code structure (syntax tree) differs. It does not show that behavior changed" in section(out, "Symbols")
    assert "structural change detection" in section(out, "Not determinable") and "not evidence that behavior changed" in section(out, "Not determinable")
    assert not re.search(r"(?i)behaviou?r(al)?ly (changed|different)|breaks|safe to", section(out, "Symbols") + section(out, "Potentially affected"))


def test_the_body_note_is_absent_when_no_body_changed(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("def charge(amount):", "def charge(amount, extra=0):"))

    assert "BODY_CHANGED means" not in diff_out(shop_repo)


def test_call_deltas_are_documented_as_covering_functions_present_on_both_sides(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/api.py", "def view(amount):\n    return amount\n")
    shop_repo.write("shop/extra.py", "from shop.billing import charge\n\n\ndef added_fn(a):\n    return charge(a)\n")

    out = diff_out(shop_repo)

    assert "compared only for functions present on both sides" in section(out, "Calls")
    assert "an added function still appears as a caller" in section(out, "Not determinable")
    assert not any("added_fn" in line for line in section(out, "Calls").splitlines() if "[INFERENCE]" in line)


def test_a_docstring_only_edit_prints_the_syntactic_note_once_per_symbol(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", 'def charge(amount):\n    """Charge."""\n    return amount + 1\n\n\ndef fee(amount):\n    return amount\n')

    assert diff_out(shop_repo).count("docstring-only edit: a syntactic change, not necessarily a behavioral one") == 1


def test_subsystem_import_rows_are_rendered_with_evidence(git_repo: GitRepo) -> None:
    git_repo.write_files({
        "shop/__init__.py": "", "shop/cart/__init__.py": "", "shop/cart/a.py": "X = 1\n", "shop/audit/__init__.py": "", "shop/audit/log.py": "def record():\n    return 1\n",
    })
    git_repo.commit("base")
    git_repo.write("shop/cart/a.py", "from shop.audit.log import record\n\nX = 1\n")

    assert "[FACT] + shop.cart -> shop.audit   shop/cart/a.py:1" in section(diff_out(git_repo), "Subsystem imports")


# -- the printed report equals the view ------------------------------------------------------------------------------------


def test_the_printed_report_is_exactly_the_view_model_rendered(git_repo: GitRepo) -> None:
    from argus.explore.render_diff import diff_lines

    git_repo.write_files({"shop/__init__.py": "", "shop/billing.py": BILLING3, "shop/callers.py": CALLERS})
    git_repo.commit("base")
    git_repo.write("shop/billing.py", "def charge(amount, currency):\n    return amount\n\n\ndef fee(amount):\n    return amount + 1\n")
    git_repo.write("shop/callers.py", CALLERS.replace("return charge(a)", "return charge(a) + 1"))
    git_repo.write("shop/audit.py", "def record():\n    return 1\n")

    out = diff_out(git_repo)
    view = build_view(git_repo)

    assert out.rstrip("\n").splitlines() == diff_lines(view, 20)


def test_every_claim_location_and_hash_in_the_view_appears_in_the_output_and_every_count_matches(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/billing.py": BILLING3, "shop/callers.py": CALLERS})
    git_repo.commit("base")
    git_repo.write("shop/billing.py", "def charge(amount, currency):\n    return amount\n\n\ndef fee(amount):\n    return amount + 1\n")

    out = diff_out(git_repo)
    view = build_view(git_repo)

    assert f"Symbols ({len(view.symbols)})" in out
    for group in (*view.interface_callers, *view.affected_callers):
        for edge in group.edges:
            site = edge.claim.evidence[0]
            assert any(site.location in line and f"hash {site.source_hash}" in line and edge.callee in line and edge.confidence in line for line in out.splitlines())
    for claim in view.claims():
        assert f"[{claim.tier.value}]" in out
    assert [line.strip() for line in section(out, "Not determinable").splitlines() if line.strip()] == [f"[UNKNOWN] {c.text}" for c in view.not_determinable]
