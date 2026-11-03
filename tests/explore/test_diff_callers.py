"""Diff Explorer v0.1: caller sections, grouping, removed symbols and evidence (contract: see diff_helpers.py)."""

from __future__ import annotations

from tests.explore.diff_helpers import (
    BILLING, FEE_CART, FEE_LEGACY, FEE_REPORTS, GitRepo, affected_groups, before_notes, call_rows, diff_out, direct_groups, flags_in,
    group_names, not_determinable, remove_fee_scenario, section, source_hash, symbol_lines,
)

OVERRIDE_NOTE = "via self; a subclass may override it"


def test_signature_and_body_change_lists_callers_once_under_direct_callers_and_never_under_potentially_affected(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("def charge(amount):\n    return amount + 1", "def charge(amount, currency):\n    return amount * 2"))

    out = diff_out(shop_repo)

    assert group_names(direct_groups(out)) == ["shop.cart.checkout", "shop.cart.checkout_twice"]
    assert affected_groups(out) == []
    twice = next(g for g in direct_groups(out) if g.name == "shop.cart.checkout_twice")
    assert [e.location for e in twice.edges] == ["shop/cart.py:9", "shop/cart.py:10"]
    assert all("SIGNATURE_CHANGED" in e.label and e.callee == "shop.billing.charge" for e in twice.edges)


# -- removed symbols: callers come from the base graph ------------------------------------------------------------------


def test_removed_function_lists_base_graph_callers_with_was_evidence_and_the_base_file_hash(git_repo: GitRepo) -> None:
    remove_fee_scenario(git_repo)

    out = diff_out(git_repo)

    removed = symbol_lines(out, "shop.billing.fee")
    assert len(removed) == 1 and "[FACT]" in removed[0] and "- function" in removed[0] and "was shop/billing.py:5" in removed[0]
    groups = {g.name: g for g in direct_groups(out)}
    assert sorted(groups) == ["shop.cart.checkout", "shop.legacy.old_report", "shop.reports.monthly"]
    assert affected_groups(out) == []
    expected = {
        "shop.cart.checkout": ("shop/cart.py", FEE_CART), "shop.legacy.old_report": ("shop/legacy.py", FEE_LEGACY),
        "shop.reports.monthly": ("shop/reports.py", FEE_REPORTS),
    }
    for name, (file, base_source) in expected.items():
        (edge,) = groups[name].edges
        assert (edge.verb, edge.callee, edge.label, edge.was, edge.confidence) == ("called", "shop.billing.fee", "REMOVED", True, "INFERRED")
        assert edge.location == f"{file}:5"
        assert edge.hash == source_hash(base_source)  # the base version of the caller's file, not the current one
        assert "(before-snapshot)" in edge.rest


def test_removed_symbol_callers_say_whether_the_caller_still_exists_changed_or_was_removed(git_repo: GitRepo) -> None:
    remove_fee_scenario(git_repo)

    groups = {g.name: g.header for g in direct_groups(diff_out(git_repo))}

    assert "(caller still exists, unchanged)" in groups["shop.cart.checkout"]
    assert "(caller also changed)" in groups["shop.reports.monthly"] and "unchanged" not in groups["shop.reports.monthly"]
    assert "(caller removed)" in groups["shop.legacy.old_report"]


def test_changed_caller_of_a_removed_symbol_carries_the_hash_of_its_base_file_not_its_current_file(git_repo: GitRepo) -> None:
    remove_fee_scenario(git_repo)

    monthly = next(g for g in direct_groups(diff_out(git_repo)) if g.name == "shop.reports.monthly")

    assert monthly.edges[0].hash == source_hash(FEE_REPORTS) != source_hash(FEE_REPORTS.replace("+ 1", "+ 2"))


def test_removed_symbols_say_it_cannot_be_told_whether_a_surviving_caller_still_calls_the_removed_name(git_repo: GitRepo) -> None:
    remove_fee_scenario(git_repo)

    notes = not_determinable(diff_out(git_repo))

    line = next(row for row in notes.splitlines() if "still calls the removed name" in row)
    assert line.strip().startswith("[UNKNOWN]")


def test_a_removed_symbol_with_no_callers_lists_no_caller_group(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount + 1\n")

    out = diff_out(shop_repo)

    assert direct_groups(out) == [] and symbol_lines(out, "shop.billing.fee")


# -- evidence is carried over, never rebuilt --------------------------------------------------------------------------

EVIDENCE_CART = (
    "from shop.billing import charge\n\n\n"
    "class Cart:\n    def total(self, amount):\n        return self.price(amount)\n\n"
    "    def price(self, amount):\n        return amount\n\n\n"
    "def checkout(amount):\n    return charge(amount)\n\n\n"
    "def checkout_twice(amount):\n    charge(amount)\n    return charge(amount)\n"
)


def test_every_caller_line_matches_the_scanned_model_in_file_line_confidence_override_note_and_hash(git_repo: GitRepo) -> None:
    from argus.explore.claims import evidence_at
    from argus.extract import scan
    from argus.model.schema import EdgeKind

    git_repo.write_files({"shop/__init__.py": "", "shop/billing.py": BILLING, "shop/cart.py": EVIDENCE_CART})
    git_repo.commit("base")
    git_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))
    git_repo.write("shop/cart.py", EVIDENCE_CART.replace("        return amount\n", "        return amount + 1\n", 1))

    out = diff_out(git_repo)

    model = scan(git_repo.root, resolve_v2=True)
    nodes = {n.id: n for n in model.nodes}
    changed = {"function:shop.billing.charge", "method:shop.cart.Cart.price"}
    expected = {
        (
            nodes[e.src_id].qualname, nodes[e.dst_id].qualname, f"{e.provenance.file}:{e.provenance.line}", e.confidence.value,
            evidence_at(git_repo.root, e.provenance.file, e.provenance.line).source_hash, bool(e.attrs.get("override_possible")),
        )
        for e in model.edges if e.kind is EdgeKind.CALLS and e.dst_id in changed
    }
    actual = {
        (g.name, edge.callee, edge.location, edge.confidence, edge.hash, OVERRIDE_NOTE in edge.rest)
        for g in affected_groups(out) for edge in g.edges
    }
    assert len(expected) == 4 and any(row[5] for row in expected)
    assert actual == expected
    assert direct_groups(out) == []


def test_two_calls_from_one_caller_to_the_same_callee_on_different_lines_are_two_edges_under_one_group(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))

    twice = [g for g in affected_groups(diff_out(shop_repo)) if g.name == "shop.cart.checkout_twice"]

    assert len(twice) == 1
    assert [(e.callee, e.location) for e in twice[0].edges] == [("shop.billing.charge", "shop/cart.py:9"), ("shop.billing.charge", "shop/cart.py:10")]


def test_the_override_note_is_printed_for_self_calls_and_confidence_is_never_upgraded(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/cart.py": EVIDENCE_CART.replace("from shop.billing import charge\n\n\n", "").replace("charge(amount)", "amount")})
    git_repo.commit("base")
    git_repo.write("shop/cart.py", (git_repo.root / "shop/cart.py").read_text().replace("        return amount\n", "        return amount + 1\n", 1))

    out = diff_out(git_repo)

    (group,) = affected_groups(out)
    (edge,) = group.edges
    assert (group.name, edge.callee, edge.confidence) == ("shop.cart.Cart.total", "shop.cart.Cart.price", "INFERRED")
    assert OVERRIDE_NOTE in edge.rest and edge.raw.startswith("[INFERENCE]")


# -- one group per caller, interface classification first --------------------------------------------------------------

BILLING3 = "def charge(amount):\n    return amount\n\n\ndef fee(amount):\n    return amount\n\n\ndef tax(amount):\n    return amount\n"
CALLERS = (
    "from shop.billing import charge, fee, tax\n\n\n"
    "def both(a):\n    charge(a)\n    return fee(a)\n\n\n"
    "def only_body(a):\n    fee(a)\n    return tax(a)\n\n\n"
    "def changed_caller(a):\n    return charge(a)\n"
)


def _dedupe_repo(repo: GitRepo) -> None:
    repo.write_files({"shop/__init__.py": "", "shop/billing.py": BILLING3, "shop/callers.py": CALLERS})
    repo.commit("base")
    repo.write("shop/billing.py", (
        "def charge(amount, currency):\n    return amount\n\n\ndef fee(amount):\n    return amount + 1\n\n\ndef tax(amount):\n    return amount + 2\n"
    ))
    repo.write("shop/callers.py", CALLERS.replace("return charge(a)", "return charge(a) + 1"))


def test_a_caller_reaching_interface_and_body_changes_appears_once_in_the_first_section_with_every_edge_labeled(git_repo: GitRepo) -> None:
    _dedupe_repo(git_repo)

    out = diff_out(git_repo)

    direct = {g.name: g for g in direct_groups(out)}
    assert sorted(direct) == ["shop.callers.both", "shop.callers.changed_caller"]
    assert [(e.callee, e.label, e.location) for e in direct["shop.callers.both"].edges] == [
        ("shop.billing.charge", "SIGNATURE_CHANGED", "shop/callers.py:5"), ("shop.billing.fee", "BODY_CHANGED", "shop/callers.py:6"),
    ]
    assert "shop.callers.both" not in group_names(affected_groups(out))
    assert group_names(direct_groups(out)).count("shop.callers.both") == 1


def test_a_caller_that_only_reaches_body_changed_symbols_is_in_the_second_section_only(git_repo: GitRepo) -> None:
    _dedupe_repo(git_repo)

    out = diff_out(git_repo)

    (group,) = affected_groups(out)
    assert group.name == "shop.callers.only_body"
    assert [(e.callee, e.label) for e in group.edges] == [("shop.billing.fee", "BODY_CHANGED"), ("shop.billing.tax", "BODY_CHANGED")]
    assert "shop.callers.only_body" not in group_names(direct_groups(out))


def test_a_caller_that_was_itself_changed_is_marked_and_an_unchanged_one_is_not(git_repo: GitRepo) -> None:
    _dedupe_repo(git_repo)

    headers = {g.name: g.header for g in direct_groups(diff_out(git_repo))}

    assert "(caller also changed)" in headers["shop.callers.changed_caller"]
    assert "also changed" not in headers["shop.callers.both"] and "removed" not in headers["shop.callers.both"]


# -- module-level callers (D3) ----------------------------------------------------------------------------------------


def test_module_level_callers_are_kept_labeled_with_original_evidence_and_never_enter_the_calls_list(shop_repo: GitRepo) -> None:
    boot = "from shop.billing import charge, fee\n\ncharge(1)\n"
    shop_repo.write("shop/boot.py", boot)
    shop_repo.commit("module level call")
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))
    shop_repo.write("shop/boot.py", boot + "fee(2)\n")  # a new module-level call: not compared in v0.1

    out = diff_out(shop_repo)

    group = next(g for g in affected_groups(out) if g.name == "shop.boot")
    assert "(module level)" in group.header
    (edge,) = group.edges
    assert (edge.callee, edge.location, edge.confidence, edge.hash) == ("shop.billing.charge", "shop/boot.py:3", "INFERRED", source_hash(boot + "fee(2)\n"))
    assert call_rows(out) == []
    assert symbol_lines(out, "shop.boot") == [] and flags_in(before_notes(out).splitlines()) == {"BODY_CHANGED"}


def test_a_module_level_caller_of_a_removed_function_uses_the_base_evidence(shop_repo: GitRepo) -> None:
    boot = "from shop.billing import fee\n\nfee(1)\n"
    shop_repo.write("shop/boot.py", boot)
    shop_repo.commit("module level call")
    shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount + 1\n")

    out = diff_out(shop_repo)

    group = next(g for g in direct_groups(out) if g.name == "shop.boot")
    assert "(module level)" in group.header
    assert (group.edges[0].was, group.edges[0].location, group.edges[0].hash) == (True, "shop/boot.py:3", source_hash(boot))
    assert section(out, "Calls") == ""
