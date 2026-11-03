"""Gate for part B of the breakage check: the four code mutations, the reader of pytest tracebacks and the scoring.

Written first (RED). The mutations are pure text-to-text functions; each must give valid Python or raise ValueError, never a file that
does not parse.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from argus.record.call_fit_gate import (
    BindingError,
    KindResult,
    Pooled,
    Scores,
    add_required_parameter,
    binding_errors,
    delete_function,
    expand_statements,
    kind_verdict,
    line_map,
    overall_verdict,
    pool,
    remove_first_default,
    rename_first_parameter,
    score_sites,
)


def src(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def def_line(source: str, name: str) -> int:
    return next(n.lineno for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)


def params_of(source: str, name: str) -> list[str]:
    func = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    a = func.args
    return [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]


# -- add a required parameter --------------------------------------------------


def test_add_required_parameter_is_keyword_only_so_it_is_valid_after_defaults():
    source = src("""
        def charge(amount, rate=1):
            return amount * rate
    """)

    new = add_required_parameter(source, def_line(source, "charge"))

    assert params_of(new, "charge") == ["amount", "rate", "argus_new_required"]
    ast.parse(new)


def test_add_required_parameter_after_a_bare_star_or_vararg_adds_no_second_star():
    source = src("""
        def a(x, *rest):
            return x

        def b(x, *, key):
            return x
    """)

    for name in ("a", "b"):
        new = add_required_parameter(source, def_line(source, name))
        assert params_of(new, name)[-1] == "argus_new_required"
        assert new.count("*,") <= 1


def test_add_required_parameter_to_a_function_without_parameters():
    source = src("""
        def run():
            return 1
    """)

    assert params_of(add_required_parameter(source, 1), "run") == ["argus_new_required"]


def test_add_required_parameter_handles_a_multiline_signature_with_a_trailing_comma():
    source = src("""
        def charge(
            amount,
            rate=1,
        ):
            return amount
    """)

    new = add_required_parameter(source, 1)

    assert params_of(new, "charge") == ["amount", "rate", "argus_new_required"]


def test_add_required_parameter_refuses_a_function_with_kwargs():
    source = "def f(a, **kw):\n    return a\n"

    with pytest.raises(ValueError):
        add_required_parameter(source, 1)


# -- rename the first parameter ------------------------------------------------


def test_rename_first_parameter_keeps_the_body_working_with_an_alias():
    source = src('''
        def charge(amount, rate=1):
            """Charge."""
            return amount * rate
    ''')

    new = rename_first_parameter(source, 1)

    assert params_of(new, "charge") == ["amount_renamed", "rate"]
    assert "amount = amount_renamed" in new
    assert new.index('"""Charge."""') < new.index("amount = amount_renamed")
    ns: dict = {}
    exec(compile(new, "<t>", "exec"), ns)  # the renamed function still works when called by position
    assert ns["charge"](2, 3) == 6


def test_rename_skips_self():
    source = src("""
        class Cart:
            def add(self, item):
                return item
    """)

    new = rename_first_parameter(source, 2)

    assert params_of(new, "add") == ["self", "item_renamed"]


def test_rename_refuses_a_function_with_nothing_to_rename():
    with pytest.raises(ValueError):
        rename_first_parameter("def f():\n    return 1\n", 1)


# -- remove the first default --------------------------------------------------


def test_remove_first_default_makes_that_parameter_required():
    source = src("""
        def charge(amount, rate: float = 1.0, fee=2):
            return amount * rate + fee
    """)

    new = remove_first_default(source, 1)

    assert "def charge(amount, rate: float, fee=2):" in new
    ast.parse(new)


def test_remove_first_default_works_for_keyword_only_and_multiline_defaults():
    source = src("""
        def f(a, *, key=dict(
            x=1,
        )):
            return a
    """)

    new = remove_first_default(source, 1)

    assert "key=" not in new
    ast.parse(new)


def test_remove_first_default_refuses_a_function_without_defaults():
    with pytest.raises(ValueError):
        remove_first_default("def f(a):\n    return a\n", 1)


# -- delete ---------------------------------------------------------------------


def test_delete_function_removes_decorators_and_body():
    source = src("""
        import functools

        @functools.cache
        def gone(x):
            return x


        def kept(x):
            return x
    """)

    new = delete_function(source, def_line(source, "gone"))

    assert "gone" not in new and "kept" in new
    ast.parse(new)


def test_delete_function_refuses_when_it_would_empty_a_class():
    source = src("""
        class Only:
            def method(self):
                return 1
    """)

    with pytest.raises(ValueError):
        delete_function(source, 2)


# -- reading pytest tracebacks --------------------------------------------------

TRACE = src("""
    ____ test_checkout ____

    tests/test_cart.py:8: in test_checkout
        checkout(5)
    shop/cart.py:5: in checkout
        return charge(amount)
    E   TypeError: charge() missing 1 required keyword-only argument: 'argus_new_required'

    ____ test_other ____

    shop/util.py:3: in helper
        return {}["k"]
    E   KeyError: 'k'

    ____ test_gone ____

    shop/cart.py:9: in checkout_twice
        return fee(amount)
    E   NameError: name 'fee' is not defined
""")


def test_binding_errors_report_the_innermost_frame_of_each_failure():
    errors = binding_errors(TRACE)

    assert errors == [
        BindingError("shop/cart.py", 5, "checkout", "TypeError", "charge() missing 1 required keyword-only argument: 'argus_new_required'"),
        BindingError("shop/cart.py", 9, "checkout_twice", "NameError", "name 'fee' is not defined"),
    ]


def test_binding_errors_ignore_other_exceptions():
    assert binding_errors("shop/util.py:3: in helper\n    x\nE   ValueError: bad\n") == []


# -- scoring ---------------------------------------------------------------------

A, B, C, D = ("f.py", 1), ("f.py", 2), ("f.py", 3), ("f.py", 4)


def test_score_counts_flagged_sites_that_failed_and_those_that_did_not():
    scores = score_sites(flagged={A, B, C}, observed={A, D}, resolved={A, B, C}, executed={A, B})

    assert scores == Scores(flagged_executed=2, confirmed=1, observed_resolved=1, caught=1, observed=2)
    # B ran and nothing failed there: a wrong flag. C never ran: it is neither confirmed nor refuted. D failed but Argus has no link there.


def test_pooled_precision_and_recall():
    pooled = pool([Scores(2, 1, 1, 1, 2), Scores(2, 2, 3, 2, 4)])

    assert pooled.precision == pytest.approx(3 / 4)
    assert pooled.recall == pytest.approx(3 / 4)
    assert pooled.resolved_share == pytest.approx(4 / 6)


def test_pool_of_nothing_has_no_rates():
    pooled = pool([])

    assert pooled.precision is None and pooled.recall is None and pooled.resolved_share is None


# -- audit round: mutations always parse -----------------------------------------


def test_parenthesised_default_is_refused_with_value_error():
    with pytest.raises(ValueError):
        remove_first_default("def f(a=(1)):\n    return a\n", 1)


def test_generic_function_never_yields_code_that_does_not_parse():
    source = "def f[T](x: T):\n    return x\n"

    try:
        new = add_required_parameter(source, 1)
    except ValueError:
        return
    ast.parse(new)


def test_add_required_parameter_refuses_a_line_that_is_not_a_def():
    with pytest.raises(ValueError):
        add_required_parameter("x = 1\n", 1)


def test_positional_only_marker_is_followed_by_a_keyword_only_parameter():
    new = add_required_parameter("def f(a, /):\n    return a\n", 1)

    assert "def f(a, /, *, argus_new_required):" in new


def test_non_ascii_parameter_names_survive_remove_first_default():
    assert "def f(é, b=2):" in remove_first_default("def f(é=1, b=2):\n    return é\n", 1)


def test_multiline_signature_rename_still_parses():
    source = "def f(\n    amount,\n    rate=1,\n):\n    return amount\n"

    ast.parse(rename_first_parameter(source, 1))


def test_two_error_lines_in_one_block_both_use_the_innermost_frame():
    out = "a.py:3: in f\n    x\nE   TypeError: f() missing 1 required positional argument: 'b'\nE   NameError: name 'g' is not defined\n"

    assert [(e.exc, e.line) for e in binding_errors(out)] == [("TypeError", 3), ("NameError", 3)]


def test_an_error_line_before_any_frame_is_ignored():
    assert binding_errors("E   TypeError: f() missing 1 required positional argument: 'b'\n") == []


# -- audit round: line numbers after a mutation ------------------------------------


def test_line_map_follows_lines_that_moved():
    original = "a\nb\nc\n"
    mutated = "a\nNEW\nb\nc\n"

    assert line_map(original, mutated) == {1: 1, 2: 3, 3: 4}


def test_line_map_drops_lines_that_were_changed_or_removed():
    assert line_map("a\nb\nc\n", "a\nX\nc\n") == {1: 1, 3: 3}


def test_expand_statements_marks_every_line_of_a_multiline_statement():
    source = "def f():\n    return g(\n        1,\n    )\n"

    assert expand_statements(source, {2}) == {2, 3, 4}


def test_expand_statements_does_not_mark_a_whole_compound_statement():
    source = "def f():\n    if x:\n        a()\n        b()\n"

    assert expand_statements(source, {2}) == {2}


# -- audit round: verdicts -----------------------------------------------------------


def result(kind="add_required", attempted=20, skipped=0, precision=1.0, recall=1.0, wrong_fact=0):
    return KindResult(kind, attempted, skipped, Pooled(precision, recall, 1.0), wrong_fact)


def test_a_kind_that_meets_every_threshold_continues():
    assert kind_verdict(result()) == "CONTINUE"


def test_precision_between_the_thresholds_means_fix():
    assert kind_verdict(result(precision=0.92)) == "FIX"


def test_low_recall_or_a_wrong_fact_means_fix():
    assert kind_verdict(result(recall=0.5)) == "FIX"
    assert kind_verdict(result(wrong_fact=1)) == "FIX"


def test_precision_under_ninety_percent_stops():
    assert kind_verdict(result(precision=0.85)) == "STOP"


def test_too_many_skipped_runs_make_a_good_looking_kind_inconclusive():
    assert kind_verdict(result(skipped=5)) == "INCONCLUSIVE"
    assert kind_verdict(result(skipped=4)) == "CONTINUE"


def test_a_stop_is_not_hidden_by_skips():
    assert kind_verdict(result(precision=0.5, skipped=15)) == "STOP"


def test_no_data_is_reported_as_no_data():
    assert kind_verdict(result(precision=None, recall=None)) == "NO DATA"


def test_overall_verdict_needs_every_kind_with_data_to_pass_on_its_own():
    good, bad = result(), result("delete", precision=0.92)

    assert overall_verdict([good, good]) == "CONTINUE"
    assert overall_verdict([good, bad]) == "FIX"
    assert overall_verdict([good, result("delete", precision=0.5)]) == "STOP"
    assert overall_verdict([good, result("rename", precision=None, recall=None)]) == "CONTINUE"
    assert overall_verdict([good, result("delete", skipped=10)]) == "INCONCLUSIVE"
    assert overall_verdict([result(precision=None, recall=None)]) == "NO DATA"
