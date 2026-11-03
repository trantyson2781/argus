"""Diff Explorer v0.1, slices 2 to 6: the comparison itself, checked on `DiffView` (no rendering or CLI involved).

The CLI-level tests in this folder describe the same behaviours through printed output and go green with slices 7 and 8.
"""

from __future__ import annotations

import pytest

from tests.explore.diff_helpers import BILLING, FEE_CART, FEE_REPORTS, GitRepo, remove_fee_scenario, source_hash

CHARGE = "shop.billing.charge"
RULES = (
    "import functools\n\n\n"
    "def plain(x):\n    return x\n\n\n"
    "@functools.lru_cache(maxsize=8)\ndef cached(x):\n    return x\n\n\n"
    "def with_default(x, rate=1):\n    return x * rate\n\n\n"
    "def annotated(x: int) -> int:\n    return x\n\n\n"
    "def make_async(x):\n    return x\n\n\n"
    "def renamed(x):\n    return 0\n"
)


def build_view(repo: GitRepo, resolve_v2: bool = True, include_tests: bool = False, head: str | None = None):
    from argus.explore.changes import build_diff_view
    from argus.explore.sides import materialize_sides
    from argus.extract import scan

    with materialize_sides(repo.root, "HEAD", head) as (base, head_side):
        return build_diff_view(base, head_side, scan(base.root, resolve_v2=resolve_v2), scan(head_side.root, resolve_v2=resolve_v2), resolve_v2, include_tests)


def change(view, qualname: str):
    found = [s for s in view.symbols if s.qualname == qualname]
    assert len(found) == 1, f"{qualname}: {[s.qualname for s in view.symbols]}"
    return found[0]


def flags(view, qualname: str) -> set[str]:
    return {str(f) for f in change(view, qualname).flags}


def names(groups) -> list[str]:
    return [g.caller for g in groups]


def unknown_text(view) -> str:
    return "\n".join(c.text for c in view.not_determinable)


# -- slice 2: added and removed symbols ------------------------------------------------------------------------------------


def test_added_and_removed_symbols_are_facts_with_head_and_base_locations(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount + 1\n\n\ndef refund(amount):\n    return -amount\n")
    shop_repo.write("shop/audit.py", "class Log:\n    pass\n")

    view = build_view(shop_repo)

    added, removed, cls = change(view, "shop.billing.refund"), change(view, "shop.billing.fee"), change(view, "shop.audit.Log")
    assert (added.change, added.kind, added.file, added.line, added.untracked) == ("added", "function", "shop/billing.py", 5, False)
    assert (removed.change, removed.file, removed.line) == ("removed", "shop/billing.py", 5)
    assert (cls.change, cls.kind, cls.untracked) == ("added", "class", True)
    assert all(c.claim.tier == "FACT" and c.claim.evidence for c in (added, removed, cls))
    assert (added.claim.evidence[0].file, added.claim.evidence[0].line) == ("shop/billing.py", 5)
    assert view.untracked_files == 1 and view.files_changed == 2


def test_symbols_are_sorted_by_file_line_and_qualname_and_the_view_is_deterministic(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2").replace("return amount\n", "return amount + 3\n"))
    shop_repo.write("shop/api.py", "def view(x):\n    return x\n")

    first, second = build_view(shop_repo), build_view(shop_repo)

    assert [(s.file, s.line) for s in first.symbols] == sorted((s.file, s.line) for s in first.symbols)
    assert first == second


def test_a_clean_tree_has_no_changes_and_only_the_standing_unknowns(shop_repo: GitRepo) -> None:
    view = build_view(shop_repo)

    assert (view.symbols, view.calls, view.interface_callers, view.affected_callers, view.imports) == ((), (), (), (), ())
    assert (view.files_changed, view.untracked_files, view.hidden_test_callers) == (0, 0, 0)
    text = unknown_text(view)
    assert "Behavioral correctness and runtime effects" in text and "Dynamic dependencies" in text and "class-level code" in text
    assert all(c.tier == "UNKNOWN" for c in view.not_determinable)


def test_the_view_labels_both_sides_and_holds_no_path_into_a_temporary_directory(shop_repo: GitRepo, scratch_tmp) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))

    view = build_view(shop_repo)

    assert view.base_label == f"HEAD ({shop_repo.short()})" and view.head_label == "working tree" and view.resolve_v2 is True
    assert str(scratch_tmp) not in repr(view)
    assert list(scratch_tmp.iterdir()) == []


# -- slices 3 and 4: signature and body flags -----------------------------------------------------------------------------------


def test_body_only_change_is_body_changed_alone_with_direct_callers_potentially_affected(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))

    view = build_view(shop_repo)

    assert flags(view, CHARGE) == {"BODY_CHANGED"} and change(view, CHARGE).claim.tier == "FACT" and change(view, CHARGE).docstring_only is False
    assert view.interface_callers == () and names(view.affected_callers) == ["shop.cart.checkout", "shop.cart.checkout_twice"]
    assert view.calls == () and len(view.symbols) == 1  # no transitive impact: api.view is not listed


def test_formatting_only_edits_set_no_flag(git_repo: GitRepo) -> None:
    git_repo.write_files({
        "shop/__init__.py": "",
        "shop/billing.py": "def charge(amount, rate):\n    label = 'total'\n    return (amount + 1) * rate, label\n",
        "shop/cart.py": "from shop.billing import charge\n\n\ndef checkout(a):\n    return charge(a, 2)\n",
    })
    git_repo.commit("base")
    git_repo.write("shop/billing.py", (
        "# a new header comment\n\n\n\n"
        'def charge(\n    amount,\n    rate,\n):  # trailing comment\n    # explain\n    label = "total"\n\n'
        "    return (\n        (amount + 1)\n        * rate\n    ), label\n"
    ))
    git_repo.write("shop/cart.py", "\n\n# shifted down\nfrom shop.billing import charge\n\n\ndef checkout(a):\n    return  charge( a,  2 )\n")

    view = build_view(git_repo)

    assert view.symbols == () and view.calls == () and view.interface_callers == () and view.affected_callers == ()
    assert view.files_changed == 2


def test_docstring_only_edit_is_body_changed_marked_as_a_docstring_only_edit(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", 'def charge(amount):\n    """Charge the amount."""\n    return amount + 1\n\n\ndef fee(amount):\n    return amount\n')

    view = build_view(shop_repo)

    assert flags(view, CHARGE) == {"BODY_CHANGED"} and change(view, CHARGE).docstring_only is True
    assert names(view.affected_callers) == ["shop.cart.checkout", "shop.cart.checkout_twice"]


def test_a_docstring_edit_together_with_a_code_edit_is_not_docstring_only(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", 'def charge(amount):\n    """Doc."""\n    return amount + 2\n\n\ndef fee(amount):\n    return amount\n')

    assert change(build_view(shop_repo), CHARGE).docstring_only is False


@pytest.mark.parametrize(
    ("qualname", "old", "new"),
    [
        ("shop.rules.cached", "@functools.lru_cache(maxsize=8)", "@functools.lru_cache(maxsize=16)"),
        ("shop.rules.plain", "def plain(x):", "@functools.cache\ndef plain(x):"),
        ("shop.rules.with_default", "rate=1", "rate=2"),
        ("shop.rules.annotated", "def annotated(x: int)", "def annotated(x: str)"),
        ("shop.rules.annotated", "-> int:", "-> float:"),
        ("shop.rules.make_async", "def make_async(x):", "async def make_async(x):"),
        ("shop.rules.renamed", "def renamed(x):", "def renamed(y):"),
    ],
    ids=["decorator-argument", "decorator-added", "default-value", "param-annotation", "return-annotation", "async-toggle", "param-rename"],
)
def test_interface_only_edits_set_signature_changed_and_never_body_changed(git_repo: GitRepo, qualname: str, old: str, new: str) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/rules.py": RULES})
    git_repo.commit("base")
    git_repo.write("shop/rules.py", RULES.replace(old, new, 1))

    view = build_view(git_repo)

    assert flags(view, qualname) == {"SIGNATURE_CHANGED"}
    assert [s.qualname for s in view.symbols] == [qualname]
    assert change(view, qualname).interface_was != change(view, qualname).interface_now


def test_combined_signature_and_body_change_carries_both_flags_and_the_signature_text(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount, currency):\n    return amount * 2\n\n\ndef fee(amount):\n    return amount\n")

    view = build_view(shop_repo)

    symbol = change(view, CHARGE)
    assert flags(view, CHARGE) == {"SIGNATURE_CHANGED", "BODY_CHANGED"}
    assert (symbol.interface_was, symbol.interface_now) == ("(amount)", "(amount, currency)")
    assert "SIGNATURE_CHANGED" in symbol.claim.text and "BODY_CHANGED" in symbol.claim.text


def test_editing_a_nested_function_flags_inner_and_enclosing_with_calls_changed_on_inner_only(git_repo: GitRepo) -> None:
    base = "def g():\n    return 1\n\n\ndef outer():\n    def inner():\n        return 1\n    return inner()\n"
    git_repo.write_files({"shop/__init__.py": "", "shop/nest.py": base})
    git_repo.commit("base")
    git_repo.write("shop/nest.py", base.replace("        return 1\n    return inner()", "        return g()\n    return inner()"))

    view = build_view(git_repo)

    assert flags(view, "shop.nest.outer.inner") == {"BODY_CHANGED", "CALLS_CHANGED"} and flags(view, "shop.nest.outer") == {"BODY_CHANGED"}
    assert [(c.change, c.source, c.target) for c in view.calls] == [("added", "shop.nest.outer.inner", "shop.nest.g")]


def test_classes_are_only_added_or_removed(git_repo: GitRepo) -> None:
    base = "class Legacy:\n    pass\n\n\nclass Base1:\n    pass\n\n\nclass Base2:\n    pass\n\n\nclass Model(Base1):\n    pass\n"
    git_repo.write_files({"shop/__init__.py": "", "shop/models.py": base})
    git_repo.commit("base")
    git_repo.write("shop/models.py", base.replace("class Legacy:\n    pass\n\n\n", "class Fresh:\n    pass\n\n\n").replace("Model(Base1)", "Model(Base2)"))

    view = build_view(git_repo)

    assert sorted((s.qualname, s.change) for s in view.symbols) == [("shop.models.Fresh", "added"), ("shop.models.Legacy", "removed")]


DUPLICATES = {
    "property": ("class Box:\n    @property\n    def size(self):\n        return 1\n\n    @size.setter\n    def size(self, v):\n        pass\n", "Box.size"),
    "if-else": ("import os\n\nif os.name:\n    def pick():\n        return 1\nelse:\n    def pick():\n        return 2\n", "pick"),
    "overload": ("from typing import overload\n\n\n@overload\ndef ov(x: int) -> int: ...\n@overload\ndef ov(x: str) -> str: ...\ndef ov(x):\n    return x\n", "ov"),
    "class-defined-twice": ("class A:\n    def m(self):\n        return 1\n\n\nclass A:\n    def m(self):\n        return 2\n", "A.m"),
}


@pytest.mark.parametrize("case", sorted(DUPLICATES))
def test_symbols_defined_more_than_once_are_skipped_and_listed_as_unknown(git_repo: GitRepo, case: str) -> None:
    source, local = DUPLICATES[case]
    git_repo.write_files({"shop/__init__.py": "", "shop/dup.py": source})
    git_repo.commit("base")
    git_repo.write("shop/dup.py", source.replace("return 1", "return 11").replace("return 2", "return 22").replace("pass", "return 5").replace("return x", "return x + 1"))

    view = build_view(git_repo)

    assert view.symbols == ()
    lines = [c.text for c in view.not_determinable if "defined more than once" in c.text]
    assert lines and f"shop.dup.{local}" in " ".join(lines) and "#" not in " ".join(lines)


def test_duplicates_in_a_file_that_did_not_change_are_not_reported(git_repo: GitRepo) -> None:
    source, _ = DUPLICATES["if-else"]
    git_repo.write_files({"shop/__init__.py": "", "shop/dup.py": source, "shop/other.py": "X = 1\n"})
    git_repo.commit("base")
    git_repo.write("shop/other.py", "X = 2\n")

    assert "defined more than once" not in unknown_text(build_view(git_repo))


# -- slice 5: call comparison -----------------------------------------------------------------------------------------------


def test_resolved_calls_added_or_removed_set_calls_changed_but_a_changed_call_count_does_not(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/cart.py", (
        "from shop.billing import charge, fee\n\n\n"
        "def checkout(amount):\n    return charge(amount) + fee(amount)\n\n\n"
        "def checkout_twice(amount):\n    return charge(amount)\n"
    ))
    shop_repo.write("shop/api.py", "from shop.cart import checkout\n\n\ndef view(amount):\n    return amount\n")

    view = build_view(shop_repo)

    rows = sorted((c.change, c.source, c.target, c.claim.evidence[0].location) for c in view.calls)
    assert rows == [
        ("added", "shop.cart.checkout", "shop.billing.fee", "shop/cart.py:5"),
        ("removed", "shop.api.view", "shop.cart.checkout", "shop/api.py:5"),
    ]
    assert all(c.claim.tier == "INFERENCE" for c in view.calls)
    assert flags(view, "shop.cart.checkout") == {"BODY_CHANGED", "CALLS_CHANGED"} and flags(view, "shop.api.view") == {"BODY_CHANGED", "CALLS_CHANGED"}
    assert flags(view, "shop.cart.checkout_twice") == {"BODY_CHANGED"}
    assert change(view, "shop.cart.checkout").calls_claim.tier == "INFERENCE" and change(view, "shop.cart.checkout_twice").calls_claim is None


def test_calls_to_external_code_and_builtins_are_not_compared(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "import os\n\n\ndef charge(amount):\n    os.getcwd()\n    return len([amount]) + amount\n\n\ndef fee(amount):\n    return amount\n")

    view = build_view(shop_repo)

    assert flags(view, CHARGE) == {"BODY_CHANGED"} and view.calls == ()


def test_loss_of_resolution_is_not_a_removed_call(git_repo: GitRepo) -> None:
    remove_fee_scenario(git_repo)

    view = build_view(git_repo)

    assert view.calls == ()
    assert [s.qualname for s in view.symbols if s.qualname == "shop.cart.checkout"] == []
    assert flags(view, "shop.reports.monthly") == {"BODY_CHANGED"}


def test_module_level_calls_are_not_compared(shop_repo: GitRepo) -> None:
    boot = "from shop.billing import charge, fee\n\ncharge(1)\n"
    shop_repo.write("shop/boot.py", boot)
    shop_repo.commit("module level call")
    shop_repo.write("shop/boot.py", boot + "fee(2)\n")

    view = build_view(shop_repo)

    assert view.calls == () and view.symbols == ()


def test_a_call_reached_only_through_obj_method_is_never_guessed_and_is_counted_as_unresolved(git_repo: GitRepo) -> None:
    git_repo.write_files({
        "shop/__init__.py": "", "shop/billing.py": BILLING, "shop/cart.py": "def checkout(client, amount):\n    return client.charge(amount)\n",
    })
    git_repo.commit("base")
    git_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))
    git_repo.write("shop/cart.py", "def checkout(client, amount):\n    return client.charge(amount) + 1\n")

    view = build_view(git_repo)

    assert view.interface_callers == () and view.affected_callers == () and view.calls == ()
    (line,) = [c.text for c in view.not_determinable if "made by the changed functions" in c.text]
    assert line.startswith("1 call made by the changed functions could not be traced (e.g. client.charge)") and "that caller is missing from the lists above" in line


# -- slice 5: files that do not parse -----------------------------------------------------------------------------------------

A_OK = "def helper():\n    return 1\n\n\ndef other():\n    return 2\n"
B = "from shop.a import helper\nfrom shop.e import thing\n\n\ndef use():\n    thing()\n    return helper()\n"
C = "from shop.e import thing\n\n\ndef top():\n    return thing()\n"
E = "def thing():\n    return 0\n\n\ndef thing2():\n    return 1\n"
BROKEN = "def helper(:\n    return 1\n"


def _parse_failure_repo(repo: GitRepo) -> None:
    repo.write_files({"shop/__init__.py": "", "shop/a.py": A_OK, "shop/b.py": B, "shop/c.py": C, "shop/e.py": E})
    repo.commit("base")
    repo.write("shop/a.py", BROKEN)


def test_a_file_that_stops_parsing_yields_no_removed_symbols_calls_or_callers(git_repo: GitRepo) -> None:
    _parse_failure_repo(git_repo)

    view = build_view(git_repo)

    assert view.symbols == () and view.calls == () and view.interface_callers == () and view.affected_callers == ()
    (line,) = [c.text for c in view.not_determinable if "shop/a.py does not parse" in c.text]
    assert line.startswith("shop/a.py does not parse in the working tree") and "compared" in line and "callers located in it are missing" in line


def test_call_comparison_is_suppressed_for_importers_of_a_failed_module_but_not_for_other_files(git_repo: GitRepo) -> None:
    _parse_failure_repo(git_repo)
    git_repo.write("shop/b.py", B.replace("    thing()\n", "    thing()\n    thing2()\n").replace("import thing", "import thing, thing2"))
    git_repo.write("shop/c.py", C.replace("return thing()", "return thing() + thing2()").replace("import thing", "import thing, thing2"))

    view = build_view(git_repo)

    assert [(c.change, c.source, c.target) for c in view.calls] == [("added", "shop.c.top", "shop.e.thing2")]
    assert flags(view, "shop.b.use") == {"BODY_CHANGED"}
    (line,) = [c.text for c in view.not_determinable if "call comparison" in c.text]
    assert "suppressed" in line and "shop/b.py" in line and "shop/c.py" not in line


def test_a_file_that_failed_to_parse_in_the_base_is_reported_against_the_base_label(git_repo: GitRepo) -> None:
    git_repo.write_files({"shop/__init__.py": "", "shop/a.py": BROKEN, "shop/e.py": E})
    git_repo.commit("base with a broken file")
    git_repo.write("shop/a.py", A_OK)

    view = build_view(git_repo)

    assert view.symbols == ()
    (line,) = [c.text for c in view.not_determinable if "shop/a.py does not parse in" in c.text]
    assert f"HEAD ({git_repo.short()})" in line


# -- slice 6: caller sections ---------------------------------------------------------------------------------------------------


def test_signature_and_body_change_lists_callers_once_under_direct_callers_only(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("def charge(amount):\n    return amount + 1", "def charge(amount, currency):\n    return amount * 2"))

    view = build_view(shop_repo)

    assert names(view.interface_callers) == ["shop.cart.checkout", "shop.cart.checkout_twice"] and view.affected_callers == ()
    twice = view.interface_callers[1]
    assert [e.claim.evidence[0].location for e in twice.edges] == ["shop/cart.py:9", "shop/cart.py:10"]
    assert all(e.callee == CHARGE and "SIGNATURE_CHANGED" in e.callee_label and e.was is False for e in twice.edges)


def test_removed_symbol_callers_come_from_the_base_graph_with_base_evidence_and_state(git_repo: GitRepo) -> None:
    remove_fee_scenario(git_repo)

    view = build_view(git_repo)

    groups = {g.caller: g for g in view.interface_callers}
    assert sorted(groups) == ["shop.cart.checkout", "shop.legacy.old_report", "shop.reports.monthly"] and view.affected_callers == ()
    assert {name: g.caller_state for name, g in groups.items()} == {
        "shop.cart.checkout": "unchanged", "shop.reports.monthly": "changed", "shop.legacy.old_report": "removed",
    }
    for name, source in (("shop.cart.checkout", FEE_CART), ("shop.reports.monthly", FEE_REPORTS)):
        (edge,) = groups[name].edges
        assert (edge.callee, edge.callee_label, edge.was, edge.confidence) == ("shop.billing.fee", "REMOVED", True, "INFERRED")
        assert edge.claim.evidence[0].source_hash == source_hash(source) and edge.claim.tier == "INFERENCE"
    assert groups["shop.reports.monthly"].edges[0].claim.evidence[0].source_hash != source_hash(FEE_REPORTS.replace("+ 1", "+ 2"))
    assert any("still calls the removed name" in c.text for c in view.not_determinable)


def test_a_removed_symbol_with_no_callers_has_no_caller_group_and_no_rename_note_without_an_addition(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount + 1\n")

    view = build_view(shop_repo)

    assert view.interface_callers == () and change(view, "shop.billing.fee").change == "removed"
    assert "rename" not in unknown_text(view)


def test_a_pure_file_rename_keeps_every_symbol_in_the_view_and_adds_one_rename_record(shop_repo: GitRepo) -> None:
    shop_repo.git("mv", "shop/billing.py", "shop/pricing.py")

    view = build_view(shop_repo)

    assert {(s.qualname, s.change) for s in view.symbols} == {
        ("shop.billing.charge", "removed"), ("shop.billing.fee", "removed"), ("shop.pricing.charge", "added"), ("shop.pricing.fee", "added"),
    }
    (rename,) = view.renames
    assert (rename.old, rename.new, rename.symbols) == ("shop/billing.py", "shop/pricing.py", 2)
    assert "shows as a removal plus an addition" not in unknown_text(view)  # the pair was recognised, so no caveat is needed


EVIDENCE_CART = (
    "from shop.billing import charge\n\n\n"
    "class Cart:\n    def total(self, amount):\n        return self.price(amount)\n\n"
    "    def price(self, amount):\n        return amount\n\n\n"
    "def checkout(amount):\n    return charge(amount)\n\n\n"
    "def checkout_twice(amount):\n    charge(amount)\n    return charge(amount)\n"
)


def test_every_caller_edge_equals_the_scanned_model_in_location_confidence_override_note_and_hash(git_repo: GitRepo) -> None:
    from argus.explore.claims import evidence_at
    from argus.extract import scan
    from argus.model.schema import EdgeKind

    git_repo.write_files({"shop/__init__.py": "", "shop/billing.py": BILLING, "shop/cart.py": EVIDENCE_CART})
    git_repo.commit("base")
    git_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))
    git_repo.write("shop/cart.py", EVIDENCE_CART.replace("        return amount\n", "        return amount + 1\n", 1))

    view = build_view(git_repo)

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
        (g.caller, e.callee, e.claim.evidence[0].location, e.confidence, e.claim.evidence[0].source_hash, e.override_possible)
        for g in view.affected_callers for e in g.edges
    }
    assert len(expected) == 4 and any(row[5] for row in expected) and actual == expected
    assert view.interface_callers == ()


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
    repo.write("shop/billing.py", "def charge(amount, currency):\n    return amount\n\n\ndef fee(amount):\n    return amount + 1\n\n\ndef tax(amount):\n    return amount + 2\n")
    repo.write("shop/callers.py", CALLERS.replace("return charge(a)", "return charge(a) + 1"))


def test_a_caller_appears_once_in_the_first_matching_section_with_every_edge_labeled(git_repo: GitRepo) -> None:
    _dedupe_repo(git_repo)

    view = build_view(git_repo)

    direct = {g.caller: g for g in view.interface_callers}
    assert sorted(direct) == ["shop.callers.both", "shop.callers.changed_caller"]
    assert [(e.callee, e.callee_label, e.claim.evidence[0].line) for e in direct["shop.callers.both"].edges] == [
        ("shop.billing.charge", "SIGNATURE_CHANGED", 5), ("shop.billing.fee", "BODY_CHANGED", 6),
    ]
    (only_body,) = view.affected_callers
    assert only_body.caller == "shop.callers.only_body" and [(e.callee, e.callee_label) for e in only_body.edges] == [
        ("shop.billing.fee", "BODY_CHANGED"), ("shop.billing.tax", "BODY_CHANGED"),
    ]
    assert direct["shop.callers.changed_caller"].caller_state == "changed" and direct["shop.callers.both"].caller_state == "unchanged"


def test_two_calls_to_the_same_callee_on_different_lines_stay_two_edges(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))

    twice = next(g for g in build_view(shop_repo).affected_callers if g.caller == "shop.cart.checkout_twice")

    assert [(e.callee, e.claim.evidence[0].location) for e in twice.edges] == [(CHARGE, "shop/cart.py:9"), (CHARGE, "shop/cart.py:10")]


def test_self_calls_keep_the_override_flag_and_inferred_confidence(git_repo: GitRepo) -> None:
    cart = EVIDENCE_CART.replace("from shop.billing import charge\n\n\n", "").replace("charge(amount)", "amount")
    git_repo.write_files({"shop/__init__.py": "", "shop/cart.py": cart})
    git_repo.commit("base")
    git_repo.write("shop/cart.py", cart.replace("        return amount\n", "        return amount + 1\n", 1))

    (group,) = build_view(git_repo).affected_callers
    (edge,) = group.edges

    assert (group.caller, edge.callee, edge.confidence, edge.override_possible) == ("shop.cart.Cart.total", "shop.cart.Cart.price", "INFERRED", True)
    assert "via self; a subclass may override it" in edge.claim.text and edge.claim.tier == "INFERENCE"


def test_module_level_callers_are_kept_and_marked(shop_repo: GitRepo) -> None:
    boot = "from shop.billing import charge, fee\n\ncharge(1)\nfee(1)\n"
    shop_repo.write("shop/boot.py", boot)
    shop_repo.commit("module level calls")
    shop_repo.write("shop/billing.py", "def charge(amount):\n    return amount + 2\n")

    view = build_view(shop_repo)

    group = next(g for g in (*view.interface_callers, *view.affected_callers) if g.caller == "shop.boot")
    assert group.module_level is True and group in view.interface_callers  # fee was removed, charge body changed: one group, first section
    assert sorted((e.callee, e.callee_label, e.was, e.claim.evidence[0].location) for e in group.edges) == [
        ("shop.billing.charge", "BODY_CHANGED", False, "shop/boot.py:3"), ("shop.billing.fee", "REMOVED", True, "shop/boot.py:4"),
    ]
    assert group.caller_state == "unchanged"


def test_test_callers_are_hidden_but_counted_unless_include_tests_is_set(shop_repo: GitRepo) -> None:
    shop_repo.write("tests/test_billing.py", "from shop.billing import charge\n\n\ndef test_charge():\n    assert charge(1) == 2\n")
    shop_repo.commit("add a test")
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))

    hidden, shown = build_view(shop_repo), build_view(shop_repo, include_tests=True)

    assert names(hidden.affected_callers) == ["shop.cart.checkout", "shop.cart.checkout_twice"] and hidden.hidden_test_callers == 1
    assert "tests.test_billing.test_charge" in names(shown.affected_callers) and shown.hidden_test_callers == 0
    assert next(g for g in shown.affected_callers if g.caller == "tests.test_billing.test_charge").is_test is True


def test_an_added_caller_of_a_changed_interface_is_marked_added(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", BILLING.replace("def charge(amount):", "def charge(amount, currency):"))
    shop_repo.write("shop/extra.py", "from shop.billing import charge\n\n\ndef new_caller(a):\n    return charge(a, 'x')\n")

    group = next(g for g in build_view(shop_repo).interface_callers if g.caller == "shop.extra.new_caller")

    assert group.caller_state == "added"


# -- subsystem imports and the head ref ---------------------------------------------------------------------------------------


def test_new_subsystem_imports_are_reported_and_imports_within_a_subsystem_are_not(git_repo: GitRepo) -> None:
    git_repo.write_files({
        "shop/__init__.py": "", "shop/cart/__init__.py": "", "shop/cart/a.py": "X = 1\n", "shop/cart/b.py": "Y = 1\n",
        "shop/audit/__init__.py": "", "shop/audit/log.py": "def record():\n    return 1\n",
    })
    git_repo.commit("base")
    git_repo.write("shop/cart/a.py", "from shop.audit.log import record\nfrom shop.cart.b import Y\n\nX = 1\n")

    view = build_view(git_repo)

    assert [(i.change, i.source, i.target) for i in view.imports] == [("added", "shop.cart", "shop.audit")]
    assert view.imports[0].claim.tier == "FACT" and view.imports[0].claim.evidence[0].location == "shop/cart/a.py:1"


def test_head_ref_compares_two_commits_and_ignores_the_working_tree(shop_repo: GitRepo) -> None:
    first = shop_repo.sha()
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 2"))
    shop_repo.commit("second")
    shop_repo.write("shop/billing.py", BILLING.replace("amount + 1", "amount + 3"))  # dirty tree must be invisible

    from argus.explore.changes import build_diff_view
    from argus.explore.sides import materialize_sides
    from argus.extract import scan

    with materialize_sides(shop_repo.root, first, "HEAD") as (base, head):
        view = build_diff_view(base, head, scan(base.root), scan(head.root), True, False)

    assert flags(view, CHARGE) == {"BODY_CHANGED"} and view.head_label.startswith("HEAD (")


def test_claims_returns_every_claim_and_never_an_evidence_free_fact_or_inference(shop_repo: GitRepo) -> None:
    shop_repo.write("shop/billing.py", "def charge(amount, currency):\n    return amount * 2\n")

    view = build_view(shop_repo)

    claims = view.claims()
    assert len(claims) >= len(view.symbols) + sum(len(g.edges) for g in (*view.interface_callers, *view.affected_callers)) + len(view.not_determinable)
    assert all(c.evidence for c in claims if c.tier != "UNKNOWN")


# -- removed calls: a vanished callsite is not the same as a call Argus can no longer resolve ----------------------------------

FEE_BASE = {
    "shop/__init__.py": "", "shop/billing.py": "def charge(a):\n    return a\n\n\ndef fee(a):\n    return a\n",
    "shop/cart.py": "from shop.billing import fee\n\n\ndef checkout(a):\n    return fee(a)\n",
}


def _fee_repo(repo: GitRepo) -> None:
    repo.write_files(FEE_BASE)
    repo.commit("base")


def _fee_removed_calls(view) -> list[tuple[str, str, str]]:
    return [(c.change, c.source, c.target) for c in view.calls]


def test_callsite_removed_while_the_target_remains_is_a_removed_call(git_repo: GitRepo) -> None:
    _fee_repo(git_repo)
    git_repo.write("shop/cart.py", "def checkout(a):\n    return a\n")

    view = build_view(git_repo)

    assert _fee_removed_calls(view) == [("removed", "shop.cart.checkout", "shop.billing.fee")]
    assert flags(view, "shop.cart.checkout") == {"BODY_CHANGED", "CALLS_CHANGED"}


def test_target_removed_while_the_callsite_remains_is_lost_resolution_not_a_removed_call(git_repo: GitRepo) -> None:
    _fee_repo(git_repo)
    git_repo.write("shop/billing.py", "def charge(a):\n    return a\n")

    view = build_view(git_repo)

    assert view.calls == () and view.symbols[0].qualname == "shop.billing.fee"
    (group,) = view.interface_callers
    assert (group.caller, group.caller_state) == ("shop.cart.checkout", "unchanged") and group.edges[0].was is True
    assert any("still calls the removed name" in c.text for c in view.not_determinable)


def test_callsite_and_target_both_removed_reports_no_call_row_but_keeps_the_caller_with_before_evidence(git_repo: GitRepo) -> None:
    _fee_repo(git_repo)
    git_repo.write("shop/billing.py", "def charge(a):\n    return a\n")
    git_repo.write("shop/cart.py", "def checkout(a):\n    return a\n")

    view = build_view(git_repo)

    assert view.calls == ()  # the pair cannot be told apart from lost resolution by the graph alone, so no call row is claimed
    (group,) = view.interface_callers
    (edge,) = group.edges
    assert (group.caller, group.caller_state, edge.callee, edge.callee_label, edge.was) == ("shop.cart.checkout", "changed", "shop.billing.fee", "REMOVED", True)
    assert edge.claim.evidence[0].location == "shop/cart.py:5"
    assert flags(view, "shop.cart.checkout") == {"BODY_CHANGED"}  # no CALLS_CHANGED: nothing is claimed about the call itself
    assert any("still calls the removed name" in c.text for c in view.not_determinable)


def test_callsite_still_in_the_source_but_no_longer_resolved_is_not_a_removed_call(git_repo: GitRepo) -> None:
    _fee_repo(git_repo)
    git_repo.write("shop/cart.py", "from shop.billing import fee\n\n\ndef checkout(a, fee):\n    return fee(a)\n")  # the parameter hides the import

    view = build_view(git_repo)

    assert view.calls == ()
    assert flags(view, "shop.cart.checkout") == {"SIGNATURE_CHANGED"}  # the body text is identical; the call is not claimed to be removed


def test_a_call_that_resolves_to_a_different_function_is_reported_as_one_removal_and_one_addition(git_repo: GitRepo) -> None:
    _fee_repo(git_repo)
    git_repo.write("shop/pricing.py", "def fee(a):\n    return a + 1\n")
    git_repo.write("shop/cart.py", "from shop.pricing import fee\n\n\ndef checkout(a):\n    return fee(a)\n")

    view = build_view(git_repo)

    assert sorted(_fee_removed_calls(view)) == [
        ("added", "shop.cart.checkout", "shop.pricing.fee"), ("removed", "shop.cart.checkout", "shop.billing.fee"),
    ]
