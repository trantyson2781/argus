"""`argus explore diff --json`: one complete, deterministic JSON document built from the same DiffView the text report prints."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import pytest

from argus.explore.jsonout import SCHEMA_VERSION, callee_flags, claim_json, dumps, file_change_side
from tests.explore.diff_helpers import GitRepo, remove_fee_scenario, run_diff
from tests.explore.test_diff_view import build_view

PKG = {"shop/__init__.py": "", "shop/cart/__init__.py": "", "shop/audit/__init__.py": ""}
SIGNATURE_EDIT = "def charge(amount, currency=None):\n    return amount + 1\n\n\ndef fee(amount):\n    return amount\n"


def _doc(repo: GitRepo, *args: str) -> dict[str, Any]:
    result = run_diff(repo, "--json", *args)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _claims(node: Any) -> list[dict[str, Any]]:
    """Every claim object anywhere in the document (an object with a tier and evidence)."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "tier" in node and "evidence" in node:
            found.append(node)
        for value in node.values():
            found.extend(_claims(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_claims(value))
    return found


# -- envelope, determinism, safety ---------------------------------------------------------------------------------------------------


def test_the_document_has_a_versioned_envelope_and_the_data(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", SIGNATURE_EDIT)

    doc = _doc(shop_repo)

    assert doc["schema_version"] == SCHEMA_VERSION and doc["command"] == "diff" and doc["ok"] is True
    assert doc["resolver"] == "calls-v2" and doc["options"] == {"base": "HEAD", "head": None, "include_tests": False}
    assert doc["data"]["head_label"] == "working tree" and doc["data"]["base_label"].startswith("HEAD (")


def test_two_runs_on_the_same_state_are_byte_identical(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", SIGNATURE_EDIT)

    assert run_diff(shop_repo, "--json").stdout == run_diff(shop_repo, "--json").stdout


def test_no_temporary_or_absolute_path_appears_in_the_document(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", SIGNATURE_EDIT)

    text = run_diff(shop_repo, "--json").stdout

    assert str(shop_repo.root) not in text and "/tmp" not in text and "argus-diff" not in text


def test_every_fact_and_inference_claim_has_evidence_and_the_not_determinable_lines_have_none(shop_repo: GitRepo) -> None:
    remove_fee_scenario(shop_repo)

    claims = _claims(_doc(shop_repo))

    assert claims
    assert all(c["evidence"] == [] for c in _doc(shop_repo)["data"]["not_determinable"])
    for claim in claims:
        assert claim["tier"] == "UNKNOWN" or claim["evidence"], claim  # FACT and INFERENCE always carry evidence
        for ev in claim["evidence"]:
            assert ev["side"] in ("base", "head") and ev["line"] >= 1 and len(ev["source_hash"]) == 16 and not ev["file"].startswith("/")


def test_every_claim_of_the_view_is_in_the_document_unchanged(shop_repo: GitRepo) -> None:
    remove_fee_scenario(shop_repo)
    view = build_view(shop_repo)

    expected = sorted((c.tier.value, c.text, tuple((e.file, e.line, e.source_hash) for e in c.evidence)) for c in view.claims())
    got = sorted(
        (c["tier"], c["text"], tuple((e["file"], e["line"], e["source_hash"]) for e in c["evidence"])) for c in _claims(_doc(shop_repo))
    )

    assert got == expected


def test_claim_json_escapes_control_characters_so_the_raw_output_is_terminal_safe() -> None:
    from argus.explore.claims import Claim, Tier

    text = dumps(claim_json(Claim("x\x1b[2J\x7f\u202e\n", Tier.UNKNOWN), "head"))

    assert all(ch == "\n" or (ch.isascii() and ch.isprintable()) for ch in text)  # the only raw control character is the layout newline
    assert json.loads(text)["text"] == "x\x1b[2J\x7f\u202e\n"  # escaped, not lost


# -- content -----------------------------------------------------------------------------------------------------------------------


def test_symbols_carry_flags_in_declaration_order_and_the_interface_before_and_after(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", SIGNATURE_EDIT.replace("amount + 1", "amount + 2"))

    (charge,) = [s for s in _doc(shop_repo)["data"]["symbols"] if s["qualname"] == "shop.billing.charge"]

    assert charge["change"] == "modified" and charge["flags"] == ["SIGNATURE_CHANGED", "BODY_CHANGED"]
    assert charge["interface_was"] != charge["interface_now"] and "currency" in charge["interface_now"]
    assert charge["claim"]["tier"] == "FACT" and charge["file"] == "shop/billing.py"


def test_a_jq_query_lists_the_symbols_whose_interface_changed(shop_repo: GitRepo) -> None:
    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")
    shop_repo.write("shop/billing.py", SIGNATURE_EDIT)
    out = run_diff(shop_repo, "--json").stdout

    names = subprocess.run(
        ["jq", "-r", '.data.symbols[] | select(.flags | index("SIGNATURE_CHANGED")) | .qualname'],
        input=out, capture_output=True, text=True, check=True,
    ).stdout.split()

    assert names == ["shop.billing.charge"]


def test_caller_groups_list_each_edge_with_flags_confidence_and_the_call_site(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", SIGNATURE_EDIT)

    data = _doc(shop_repo)["data"]

    groups = {g["caller"]: g for g in data["interface_callers"]}
    edge = groups["shop.cart.checkout"]["edges"][0]
    assert edge["callee"] == "shop.billing.charge" and "SIGNATURE_CHANGED" in edge["callee_flags"]
    assert edge["confidence"] == "INFERRED" and edge["was"] is False and edge["override_possible"] is False
    assert edge["claim"]["tier"] == "INFERENCE" and edge["claim"]["evidence"][0]["file"] == "shop/cart.py"


def test_the_summary_counts_match_the_text_report(shop_repo: GitRepo) -> None:
    remove_fee_scenario(shop_repo)

    summary = _doc(shop_repo)["data"]["summary"]
    text = run_diff(shop_repo).output

    assert summary["removed"] >= 1 and summary["interface_caller_groups"] >= 1
    assert f"{summary['removed']} removed" in text
    assert f"{summary['interface_caller_groups']} of changed interfaces" in text


def test_nothing_is_cut_by_the_display_caps(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/wide.py": ""})
    git_repo.commit("base")
    git_repo.write("shop/wide.py", "".join(f"def f{i:03d}():\n    return {i}\n\n\n" for i in range(80)))

    data = _doc(git_repo)["data"]

    assert len(data["symbols"]) == 80 and data["summary"]["added"] == 80


def test_hidden_test_callers_are_counted_and_include_tests_is_an_explicit_option(shop_repo: GitRepo) -> None:
    shop_repo.write("tests/__init__.py", "")
    shop_repo.write("tests/test_billing.py", "from shop.billing import charge\n\n\ndef test_charge():\n    assert charge(1)\n")
    shop_repo.commit("add test")
    shop_repo.write("shop/billing.py", SIGNATURE_EDIT)

    hidden = _doc(shop_repo)
    shown = _doc(shop_repo, "--include-tests")

    assert hidden["data"]["hidden_test_callers"] >= 1 and shown["options"]["include_tests"] is True
    assert any(g["is_test"] for g in shown["data"]["interface_callers"])
    assert not any(g["is_test"] for g in hidden["data"]["interface_callers"])


# -- which side each piece of evidence was read from ---------------------------------------------------------------------------------


def test_a_removed_symbol_and_the_callers_of_it_cite_the_base_side_and_added_ones_the_head_side(shop_repo: GitRepo) -> None:
    remove_fee_scenario(shop_repo)

    data = _doc(shop_repo)["data"]

    (fee,) = [s for s in data["symbols"] if s["qualname"] == "shop.billing.fee"]
    assert fee["change"] == "removed" and fee["claim"]["evidence"][0]["side"] == "base"
    edges = [e for g in data["interface_callers"] for e in g["edges"] if e["callee_flags"] == ["REMOVED"]]
    assert edges and all(e["was"] is True and e["claim"]["evidence"][0]["side"] == "base" for e in edges)


def test_added_and_modified_symbols_and_signature_edges_cite_the_head_side(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", SIGNATURE_EDIT + "\n\ndef extra():\n    return 1\n")

    data = _doc(shop_repo)["data"]

    assert len(data["symbols"]) >= 2
    for symbol in data["symbols"]:
        assert symbol["claim"]["evidence"][0]["side"] == "head"
    edges = [e for g in data["interface_callers"] for e in g["edges"]]
    assert edges and all(e["claim"]["evidence"][0]["side"] == "head" for e in edges)


def test_a_removed_import_cites_the_base_side(git_repo: GitRepo) -> None:
    from tests.explore.test_diff_imports import AUDIT_LOG

    git_repo.write_files({**PKG, "shop/cart/a.py": "X = 1\nfrom shop.audit.log import record\n", "shop/audit/log.py": AUDIT_LOG})
    git_repo.commit("base")
    git_repo.write("shop/cart/a.py", "X = 1\n")

    (item,) = _doc(git_repo)["data"]["imports"]

    assert (item["change"], item["source"], item["target"]) == ("removed", "shop.cart", "shop.audit")
    assert item["claim"]["evidence"][0]["side"] == "base"


def test_a_removed_call_cites_the_base_side_and_an_added_call_the_head_side(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/cart.py", (
        "from shop.billing import charge, fee\n\n\ndef checkout(amount):\n    return charge(amount) + fee(amount)\n\n\n"
        "def checkout_twice(amount):\n    return charge(amount)\n"
    ))
    shop_repo.write("shop/api.py", "from shop.cart import checkout\n\n\ndef view(amount):\n    return amount\n")

    rows = {(c["change"], c["source"], c["target"]): c["claim"]["evidence"][0]["side"] for c in _doc(shop_repo)["data"]["calls"]}

    assert rows == {("added", "shop.cart.checkout", "shop.billing.fee"): "head", ("removed", "shop.api.view", "shop.cart.checkout"): "base"}


def test_a_pure_rename_and_a_removed_file_have_the_right_sides(git_repo: GitRepo) -> None:
    core = "RATE = 5\n\n\ndef price(x):\n    return x * RATE\n"
    git_repo.write_files({"shop/__init__.py": "", "shop/core.py": core, "shop/old.py": "X = 1\n"})
    git_repo.commit("base")
    git_repo.git("mv", "shop/core.py", "shop/pricing.py")
    git_repo.delete("shop/old.py")

    data = _doc(git_repo)["data"]

    (rename,) = data["renames"]
    assert (rename["old"], rename["new"]) == ("shop/core.py", "shop/pricing.py") and rename["claim"]["evidence"][0]["side"] == "head"
    (removed,) = [f for f in data["file_changes"] if f["file"] == "shop/old.py"]
    assert removed["claim"]["evidence"][0]["side"] == "base"
    assert data["new_files"] == sorted(data["new_files"])


@pytest.mark.parametrize(
    ("kind", "file", "new_files", "side"),
    [
        ("added", "a.py", {"a.py"}, "head"), ("added_empty", "a.py", {"a.py"}, "head"), ("outside_symbols", "a.py", set(), "head"),
        ("formatting_only", "a.py", set(), "head"), ("removed", "a.py", set(), "base"), ("removed_empty", "a.py", set(), "base"),
        ("relocated", "new.py", {"new.py"}, "head"), ("relocated", "gone.py", {"new.py"}, "base"),
    ],
)
def test_file_change_side_by_kind(kind: str, file: str, new_files: set[str], side: str) -> None:
    assert file_change_side(kind, file, frozenset(new_files)) == side


def test_callee_flags_split_the_label_and_reject_anything_unknown() -> None:
    assert callee_flags("REMOVED") == ["REMOVED"]
    assert callee_flags("SIGNATURE_CHANGED, BODY_CHANGED") == ["SIGNATURE_CHANGED", "BODY_CHANGED"]
    with pytest.raises(ValueError, match="BOGUS"):
        callee_flags("BODY_CHANGED, BOGUS")


# -- errors ------------------------------------------------------------------------------------------------------------------------


def test_a_bad_ref_in_json_mode_is_one_json_error_object_on_stdout_with_exit_2(shop_repo: GitRepo) -> None:
    result = run_diff(shop_repo, "--json", "--base", "no-such-ref")

    assert result.exit_code == 2
    doc = json.loads(result.stdout)
    assert doc["ok"] is False and doc["command"] == "diff" and doc["error"]["kind"] == "diff_error" and "no-such-ref" in doc["error"]["message"]
    assert "data" not in doc and str(shop_repo.root) not in result.stdout


def test_json_with_limit_or_all_is_refused_as_a_json_error(shop_repo: GitRepo) -> None:
    for flag in (("--limit", "3"), ("--all",)):
        result = run_diff(shop_repo, "--json", *flag)

        assert result.exit_code == 2
        assert json.loads(result.stdout)["error"]["kind"] == "bad_option"


def test_a_text_mode_error_goes_to_stderr_not_stdout(shop_repo: GitRepo) -> None:
    result = run_diff(shop_repo, "--base", "no-such-ref")

    assert result.exit_code == 2
    assert result.stdout == "" and "no-such-ref" in result.stderr
