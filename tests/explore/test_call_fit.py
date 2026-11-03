"""Breakage check, part B: does each call still fit the new inputs of a changed function? Reads code, runs nothing.

Written first (RED). The pure binding rules are checked on `fit_call`; the wiring is checked on `DiffView.call_fits`.
"""

from __future__ import annotations

import ast
import json

import pytest

from argus.explore.call_fit import Verdict, fit_call
from argus.explore.claims import Tier
from tests.explore.diff_helpers import GitRepo, diff_out, section
from argus.explore.render_panel import panel_lines
from tests.explore.test_diff_view import build_view, change


def params(*specs: str) -> list[dict]:
    """Parameters from the source of a def's argument list, in the same shape the scanner stores them."""
    tree = ast.parse(f"def f({', '.join(specs)}): pass").body[0]
    from argus.extract.python_ast import _signature_of

    return _signature_of(tree).as_payload()["parameters"]


def call(source: str) -> ast.Call:
    node = ast.parse(source).body[0].value
    assert isinstance(node, ast.Call)
    return node


# -- the binding rules ---------------------------------------------------------


def test_call_with_every_required_argument_fits():
    assert fit_call(params("amount", "currency"), call("charge(1, 'EUR')"), drop_first=False)[0] is Verdict.FITS


def test_missing_required_argument_breaks_and_names_it():
    verdict, reason = fit_call(params("amount", "currency"), call("charge(1)"), drop_first=False)
    assert verdict is Verdict.BREAKS
    assert "currency" in reason


def test_default_makes_an_argument_optional():
    assert fit_call(params("amount", "currency='EUR'"), call("charge(1)"), drop_first=False)[0] is Verdict.FITS


def test_renamed_parameter_breaks_a_keyword_call():
    verdict, reason = fit_call(params("total"), call("charge(amount=1)"), drop_first=False)
    assert verdict is Verdict.BREAKS
    assert "amount" in reason


def test_too_many_positional_arguments_break():
    assert fit_call(params("amount"), call("charge(1, 2)"), drop_first=False)[0] is Verdict.BREAKS


def test_keyword_only_parameter_cannot_be_passed_by_position():
    assert fit_call(params("amount", "*", "currency"), call("charge(1, 'EUR')"), drop_first=False)[0] is Verdict.BREAKS


def test_star_arguments_at_the_call_are_unknown_not_guessed():
    assert fit_call(params("amount", "currency"), call("charge(*args)"), drop_first=False)[0] is Verdict.UNKNOWN


def test_double_star_arguments_at_the_call_are_unknown_not_guessed():
    assert fit_call(params("amount", "currency"), call("charge(**options)"), drop_first=False)[0] is Verdict.UNKNOWN


def test_vararg_and_kwarg_accept_anything():
    assert fit_call(params("*args", "**kwargs"), call("charge(1, 2, x=3)"), drop_first=False)[0] is Verdict.FITS


def test_dropping_the_first_parameter_skips_self():
    assert fit_call(params("self", "item"), call("cart.add(1)"), drop_first=True)[0] is Verdict.FITS
    assert fit_call(params("self", "item", "qty"), call("cart.add(1)"), drop_first=True)[0] is Verdict.BREAKS


def test_nothing_to_drop_when_the_first_parameter_is_a_vararg():
    assert fit_call(params("*args"), call("cart.add(1)"), drop_first=True)[0] is Verdict.FITS


# -- the wiring, on a real diff ------------------------------------------------

BILLING_OLD = "def charge(amount):\n    return amount + 1\n"
BILLING_NEW = "def charge(amount, currency):\n    return amount + 1\n"
CART = (
    "from shop.billing import charge\n\n\n"
    "def checkout(amount):\n    return charge(amount)\n\n\n"
    "def checkout_twice(amount):\n    charge(amount)\n    return charge(amount)\n"
)


def repo_with(tmp_path, files: dict[str, str], edits: dict[str, str]) -> GitRepo:
    repo = GitRepo(tmp_path / "repo")
    repo.write_files({"shop/__init__.py": "", **files})
    repo.commit("base")
    repo.write_files(edits)
    return repo


def fits(view, caller: str):
    return [f for f in view.call_fits if f.caller == caller]


def test_call_that_no_longer_fits_is_reported_with_its_line(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART}, {"shop/billing.py": BILLING_NEW})

    found = fits(build_view(repo), "shop.cart.checkout")

    assert len(found) == 1
    assert found[0].verdict is Verdict.BREAKS
    assert found[0].callee == "shop.billing.charge"
    assert (found[0].file, found[0].line) == ("shop/cart.py", 5)
    assert "currency" in found[0].reason
    assert found[0].already_broken is False


def test_every_call_site_is_checked_not_only_the_first(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART}, {"shop/billing.py": BILLING_NEW})

    assert sorted(f.line for f in fits(build_view(repo), "shop.cart.checkout_twice")) == [9, 10]


def test_claim_is_an_inference_with_the_call_site_as_evidence(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART}, {"shop/billing.py": BILLING_NEW})

    claim = fits(build_view(repo), "shop.cart.checkout")[0].claim

    assert claim.tier is Tier.INFERENCE  # the caller link is INFERRED, so the verdict is never a FACT
    assert [(e.file, e.line) for e in claim.evidence] == [("shop/cart.py", 5)]


def test_caller_updated_in_the_same_change_fits(tmp_path):
    repo = repo_with(
        tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART},
        {"shop/billing.py": BILLING_NEW, "shop/cart.py": CART.replace("charge(amount)", "charge(amount, 'EUR')")},
    )

    assert {f.verdict for f in fits(build_view(repo), "shop.cart.checkout")} == {Verdict.FITS}


def test_call_that_did_not_fit_the_old_signature_either_says_so(tmp_path):
    cart = CART.replace("return charge(amount)\n\n\ndef checkout_twice", "return charge(amount, 1, 2)\n\n\ndef checkout_twice")
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": cart}, {"shop/billing.py": BILLING_NEW})

    found = fits(build_view(repo), "shop.cart.checkout")

    assert found[0].verdict is Verdict.BREAKS  # 3 arguments fit neither (amount) nor (amount, currency)
    assert found[0].already_broken is True


def test_two_calls_to_the_same_name_on_one_line_are_unknown(tmp_path):
    cart = "from shop.billing import charge\n\n\ndef checkout(a):\n    return charge(a) + charge(a)\n"
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": cart}, {"shop/billing.py": BILLING_NEW})

    found = fits(build_view(repo), "shop.cart.checkout")

    assert {f.verdict for f in found} == {Verdict.UNKNOWN}


def test_no_signature_change_means_nothing_to_check(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART}, {"shop/billing.py": BILLING_OLD + "# note\n"})

    assert build_view(repo).call_fits == ()


METHODS = (
    "class Cart:\n"
    "    def add(self, item):\n        return item\n\n"
    "    @staticmethod\n    def parse(raw):\n        return raw\n\n"
    "    @classmethod\n    def build(cls, raw):\n        return cls()\n\n"
    "    @some_decorator\n    def wrapped(self, a):\n        return a\n\n"
    "    def run(self):\n        self.add(1)\n        self.parse(1)\n        self.build(1)\n        self.wrapped(1)\n"
)


def methods_after(tmp_path, old: str, new: str):
    repo = repo_with(tmp_path, {"shop/cart.py": METHODS}, {"shop/cart.py": METHODS.replace(old, new)})
    return fits(build_view(repo), "shop.cart.Cart.run")


def test_bound_method_call_skips_self(tmp_path):
    found = methods_after(tmp_path, "def add(self, item):", "def add(self, item, qty):")

    assert [(f.callee, f.verdict) for f in found] == [("shop.cart.Cart.add", Verdict.BREAKS)]
    assert "qty" in found[0].reason


def test_staticmethod_keeps_its_first_parameter(tmp_path):
    found = methods_after(tmp_path, "def parse(raw):", "def parse(raw, strict):")

    assert [(f.callee, f.verdict) for f in found] == [("shop.cart.Cart.parse", Verdict.BREAKS)]


def test_classmethod_skips_cls(tmp_path):
    found = methods_after(tmp_path, "def build(cls, raw):", "def build(cls, raw, strict):")

    assert [(f.callee, f.verdict) for f in found] == [("shop.cart.Cart.build", Verdict.BREAKS)]


def test_unknown_decorator_means_unknown(tmp_path):
    found = methods_after(tmp_path, "def wrapped(self, a):", "def wrapped(self, a, b):")

    assert [(f.callee, f.verdict) for f in found] == [("shop.cart.Cart.wrapped", Verdict.UNKNOWN)]


def test_call_through_the_class_never_gets_a_fits_or_breaks_verdict(tmp_path):
    cart = METHODS.replace("self.add(1)", "Cart.add(self, 1)")
    repo = repo_with(tmp_path, {"shop/cart.py": cart}, {"shop/cart.py": cart.replace("def add(self, item):", "def add(self, item, qty):")})

    found = [f for f in fits(build_view(repo), "shop.cart.Cart.run") if f.callee.endswith(".add")]

    assert [f.verdict for f in found] in ([], [Verdict.UNKNOWN])  # the receiver is passed by hand: counting it would be a guess


def test_constructor_call_is_checked_against_init(tmp_path):
    models = "class Order:\n    def __init__(self, total):\n        self.total = total\n"
    cart = "from shop.models import Order\n\n\ndef make(total):\n    return Order(total)\n"
    repo = repo_with(
        tmp_path, {"shop/models.py": models, "shop/cart.py": cart},
        {"shop/models.py": models.replace("(self, total)", "(self, total, currency)")},
    )

    found = fits(build_view(repo), "shop.cart.make")

    assert [(f.callee, f.verdict) for f in found] == [("shop.models.Order", Verdict.BREAKS)]
    assert "currency" in found[0].reason


def test_nothing_is_written_to_the_repository(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART}, {"shop/billing.py": BILLING_NEW})
    before = {p: p.read_bytes() for p in repo.root.rglob("*.py")}

    build_view(repo)

    assert {p: p.read_bytes() for p in repo.root.rglob("*.py")} == before


@pytest.mark.parametrize("verdict", list(Verdict))
def test_verdict_values_are_plain_words(verdict):
    assert verdict.value.isupper()


# -- what the reader sees ------------------------------------------------------

TITLE = "Calls checked against the new inputs"


def test_diff_lists_a_call_that_does_not_fit_with_its_place_and_reason(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART}, {"shop/billing.py": BILLING_NEW})

    body = section(diff_out(repo.root), TITLE)

    assert "DOES NOT FIT" in body
    assert "shop.cart.checkout" in body and "shop/cart.py:5" in body
    assert "currency" in body
    assert "[INFERENCE]" in body and "[FACT]" not in body


def test_diff_says_how_many_calls_fit_and_never_calls_them_safe(tmp_path):
    repo = repo_with(
        tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART},
        {"shop/billing.py": BILLING_NEW, "shop/cart.py": CART.replace("charge(amount)", "charge(amount, 'EUR')")},
    )

    body = section(diff_out(repo.root), TITLE)

    assert "3 calls checked fit the new inputs" in body
    assert "DOES NOT FIT" not in body
    assert "safe" not in body.lower()


def test_diff_names_calls_it_could_not_check(tmp_path):
    cart = "from shop.billing import charge\n\n\ndef checkout(a, rest):\n    return charge(*rest)\n"
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": cart}, {"shop/billing.py": BILLING_NEW})

    body = section(diff_out(repo.root), TITLE)

    assert "NOT CHECKED" in body and "shop/cart.py:5" in body


def test_diff_says_when_a_call_was_already_broken(tmp_path):
    cart = CART.replace("return charge(amount)\n\n\ndef checkout_twice", "return charge(amount, 1, 2)\n\n\ndef checkout_twice")
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": cart}, {"shop/billing.py": BILLING_NEW})

    assert "did not fit before" in section(diff_out(repo.root), TITLE)


def test_diff_has_no_section_without_a_changed_signature(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART}, {"shop/billing.py": BILLING_OLD + "# note\n"})

    assert TITLE not in diff_out(repo.root)


def test_json_carries_every_checked_call_with_its_evidence(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART}, {"shop/billing.py": BILLING_NEW})

    rows = json.loads(diff_out(repo.root, "--json"))["data"]["call_fits"]

    row = next(r for r in rows if r["caller"] == "shop.cart.checkout")
    assert (row["verdict"], row["callee"], row["already_broken"]) == ("BREAKS", "shop.billing.charge", False)
    assert row["claim"]["tier"] == "INFERENCE"
    assert [(e["file"], e["line"], e["side"]) for e in row["claim"]["evidence"]] == [("shop/cart.py", 5, "head")]
    assert len(rows) == 3


# -- the panel ---------------------------------------------------------------


def panel(view, qualname: str) -> str:
    return "\n".join(panel_lines(change(view, qualname), None))


def test_symbol_carries_the_calls_checked_against_it(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART}, {"shop/billing.py": BILLING_NEW})

    assert len(change(build_view(repo), "shop.billing.charge").call_fits) == 3


def test_class_calls_belong_to_the_init_that_changed(tmp_path):
    models = "class Order:\n    def __init__(self, total):\n        self.total = total\n"
    cart = "from shop.models import Order\n\n\ndef make(total):\n    return Order(total)\n"
    repo = repo_with(
        tmp_path, {"shop/models.py": models, "shop/cart.py": cart},
        {"shop/models.py": models.replace("(self, total)", "(self, total, currency)")},
    )

    assert len(change(build_view(repo), "shop.models.Order.__init__").call_fits) == 1


def test_panel_counts_calls_that_do_not_fit_and_names_them(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": CART}, {"shop/billing.py": BILLING_NEW})

    text = panel(build_view(repo), "shop.billing.charge")

    assert "Calls checked" in text and "3 do not fit" in text
    assert "shop/cart.py:5" in text and "currency" in text


def test_panel_says_when_there_was_no_call_to_check(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD}, {"shop/billing.py": BILLING_NEW})

    assert "Calls checked  no known call to check" in panel(build_view(repo), "shop.billing.charge")


def test_panel_has_no_calls_line_when_the_inputs_did_not_change(tmp_path):
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD}, {"shop/billing.py": BILLING_OLD.replace("+ 1", "+ 2")})

    assert "Calls checked" not in panel(build_view(repo), "shop.billing.charge")


# -- audit round: the tier, dropped calls, constructors ------------------------


def test_a_verdict_is_never_tagged_fact_even_for_an_extracted_edge():
    from argus.explore.call_fit import fit_tier
    from argus.model.schema import Confidence

    assert fit_tier(Confidence.EXTRACTED, Verdict.BREAKS) is Tier.INFERENCE
    assert fit_tier(Confidence.EXTRACTED, Verdict.FITS) is Tier.INFERENCE
    assert fit_tier(Confidence.INFERRED, Verdict.UNKNOWN) is Tier.UNKNOWN


MODELS = "class Base:\n    def __init__(self, total):\n        self.total = total\n"


def test_call_to_a_subclass_that_inherits_a_changed_init_is_unknown_not_silent(tmp_path):
    use = "from shop.models import Base\n\n\nclass Sub(Base):\n    pass\n\n\ndef make():\n    return Sub(1)\n"
    repo = repo_with(
        tmp_path, {"shop/models.py": MODELS, "shop/use.py": use}, {"shop/models.py": MODELS.replace("(self, total)", "(self, total, currency)")},
    )
    view = build_view(repo)

    found = fits(view, "shop.use.make")

    assert [(f.callee, f.verdict) for f in found] == [("shop.use.Sub", Verdict.UNKNOWN)]
    assert "inherit" in found[0].reason
    assert change(view, "shop.models.Base.__init__").call_fits == tuple(found)


def test_class_with_its_own_new_is_unknown(tmp_path):
    models = MODELS + "\n    def __new__(cls, *args):\n        return super().__new__(cls)\n"
    cart = "from shop.models import Base\n\n\ndef make():\n    return Base(1)\n"
    repo = repo_with(
        tmp_path, {"shop/models.py": models, "shop/cart.py": cart}, {"shop/models.py": models.replace("(self, total)", "(self, total, currency)")},
    )

    found = fits(build_view(repo), "shop.cart.make")

    assert [f.verdict for f in found] == [Verdict.UNKNOWN] and "__new__" in found[0].reason


def test_constructor_calls_stay_on_init_when_the_class_itself_also_changed(tmp_path):
    models = MODELS
    cart = "from shop.models import Base\n\n\ndef make():\n    return Base(1)\n"
    new_models = models.replace("(self, total)", "(self, total, currency)") + "\n    def extra(self):\n        return 1\n"
    repo = repo_with(tmp_path, {"shop/models.py": models, "shop/cart.py": cart}, {"shop/models.py": new_models})

    assert len(change(build_view(repo), "shop.models.Base.__init__").call_fits) == 1


def test_a_function_called_at_module_level_is_still_checked(tmp_path):
    cart = "from shop.billing import charge\n\nTOTAL = charge(1)\n"
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": cart}, {"shop/billing.py": BILLING_NEW})

    assert [(f.caller, f.verdict) for f in build_view(repo).call_fits] == [("shop.cart", Verdict.BREAKS)]


def test_import_alias_is_unknown_and_module_attribute_is_checked(tmp_path):
    alias = "from shop.billing import charge as ch\n\n\ndef f():\n    return ch(1)\n"
    attr = "import shop.billing as b\n\n\ndef g():\n    return b.charge(1)\n"
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/a.py": alias, "shop/b.py": attr}, {"shop/billing.py": BILLING_NEW})
    view = build_view(repo)

    assert [f.verdict for f in fits(view, "shop.a.f")] == [Verdict.UNKNOWN]
    assert [f.verdict for f in fits(view, "shop.b.g")] == [Verdict.BREAKS]


def test_call_split_over_several_lines_is_reported_at_the_line_of_the_call(tmp_path):
    cart = "from shop.billing import charge\n\n\ndef f():\n    return charge(\n        1,\n    )\n"
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": cart}, {"shop/billing.py": BILLING_NEW})

    assert [(f.line, f.verdict) for f in fits(build_view(repo), "shop.cart.f")] == [(5, Verdict.BREAKS)]


def test_async_function_with_a_changed_signature_is_checked(tmp_path):
    billing = "async def charge(amount):\n    return amount\n"
    cart = "from shop.billing import charge\n\n\nasync def f():\n    return await charge(1)\n"
    repo = repo_with(
        tmp_path, {"shop/billing.py": billing, "shop/cart.py": cart}, {"shop/billing.py": billing.replace("(amount)", "(amount, currency)")},
    )

    assert [f.verdict for f in fits(build_view(repo), "shop.cart.f")] == [Verdict.BREAKS]


def test_caller_in_test_code_is_marked_as_test(tmp_path):
    test = "from shop.billing import charge\n\n\ndef test_charge():\n    charge(1)\n"
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "tests/test_billing.py": test}, {"shop/billing.py": BILLING_NEW})

    assert [(f.caller, f.is_test) for f in build_view(repo).call_fits] == [("tests.test_billing.test_charge", True)]


def test_more_than_the_parameter_limit_is_unknown_not_a_slow_bind():
    many = params(*[f"a{i}=0" for i in range(300)])

    verdict, reason = fit_call(many, call("f()"), drop_first=False)

    assert verdict is Verdict.UNKNOWN and "parameters" in reason


def test_keyword_only_without_default_is_reported_after_a_bare_star_and_kwargs():
    verdict, reason = fit_call(params("a", "*", "x", "**kw"), call("f(1)"), drop_first=False)

    assert verdict is Verdict.BREAKS and "'x'" in reason


def test_positional_only_parameter_cannot_be_passed_by_keyword():
    assert fit_call(params("a", "/", "b"), call("f(a=1, b=2)"), drop_first=False)[0] is Verdict.BREAKS


def test_keyword_only_default_is_optional():
    assert fit_call(params("a", "*", "k=1"), call("f(1)"), drop_first=False)[0] is Verdict.FITS
    assert fit_call(params("a", "*", "k=1"), call("f(1, k=2)"), drop_first=False)[0] is Verdict.FITS


def test_method_without_a_receiver_parameter_is_unknown():
    assert fit_call([], call("c.m()"), drop_first=True)[0] is Verdict.UNKNOWN


def test_unreadable_caller_file_is_a_reason_not_a_crash(tmp_path):
    from argus.explore.call_fit import _CallSiteReader

    (tmp_path / "bad.py").write_text("def (:\n", encoding="utf-8")
    (tmp_path / "latin.py").write_bytes(b"x = '\xe9'\n")

    reader = _CallSiteReader(tmp_path)

    assert isinstance(reader.at("bad.py", 1, "f"), str)
    assert isinstance(reader.at("latin.py", 1, "f"), str)
    assert isinstance(reader.at("missing.py", 1, "f"), str)


def test_calls_that_could_not_be_looked_at_are_counted_in_not_determinable(tmp_path):
    cart = (
        "from shop.billing import charge\n\n\n"
        "def helper(a):\n    return charge(a)\n\n\n"
        "def helper(a):\n    return a\n"  # defined twice: the scanner cannot tell which one the call belongs to
    )
    repo = repo_with(tmp_path, {"shop/billing.py": BILLING_OLD, "shop/cart.py": cart}, {"shop/billing.py": BILLING_NEW})

    view = build_view(repo)

    notes = [c.text for c in view.not_determinable if "not checked" in c.text]
    assert view.call_fits == () and len(notes) == 1 and "1 call" in notes[0]
