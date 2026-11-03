"""Gate for part A of the breakage check (`argus check`): the return-value mutation, the scoring and the verdict.

The runner (`scripts/check_gate/run_gate.py`) only wires these pure parts to real test runs.
"""

from __future__ import annotations

import ast

import pytest

from argus.record.call_fit_gate import nullify_returns
from argus.record.check_gate import Row, pool_rows, verdict


def def_line(source: str, name: str) -> int:
    return next(n.lineno for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)


# -- the return-value mutation ----------------------------------------------------


def test_every_return_value_of_the_function_becomes_none():
    source = "def f(x):\n    if x:\n        return x + 1\n    return {'a': x}\n"

    assert nullify_returns(source, 1) == "def f(x):\n    if x:\n        return None\n    return None\n"


def test_returns_of_nested_functions_and_other_functions_are_left_alone():
    source = "def f(x):\n    def inner():\n        return 5\n    return inner()\n\n\ndef g():\n    return 7\n"

    new = nullify_returns(source, 1)

    assert "return 5" in new and "return 7" in new and "return inner()" not in new


def test_multiline_and_non_ascii_return_values_are_replaced_whole():
    new = nullify_returns("def f(é):\n    return (\n        é,\n        2,\n    )\n", 1)

    assert new == "def f(é):\n    return None\n"


def test_a_function_that_returns_no_value_cannot_be_mutated():
    with pytest.raises(ValueError):
        nullify_returns("def f():\n    return\n", 1)
    with pytest.raises(ValueError):
        nullify_returns("def f():\n    pass\n", 1)


def test_a_line_that_is_not_a_def_is_refused():
    with pytest.raises(ValueError):
        nullify_returns("x = 1\n", 1)


# -- scoring -------------------------------------------------------------------------


def row(truth=(), selected=(), static=(), newly=(), ratio=0.3, skipped=False, partial=()):
    return Row("f", "raise", frozenset(truth), frozenset(selected), frozenset(static), frozenset(newly), ratio, skipped, frozenset(partial))


def test_recall_is_the_share_of_really_failing_tests_the_selection_contained():
    pooled = pool_rows([row(truth={"a", "b", "c"}, selected={"a", "b", "x"}, static={"a"}), row(truth={"d"}, selected={"d"}, static={"d"})])

    assert pooled.recall == pytest.approx(3 / 4)
    assert pooled.static_recall == pytest.approx(2 / 4)


def test_partial_recording_recall_is_scored_apart():
    pooled = pool_rows([row(truth={"a", "b"}, selected={"a", "b"}, partial={"a"}), row(truth={"c", "d"}, selected={"c", "d"}, partial={"c", "d"})])

    assert pooled.partial_recall == pytest.approx(3 / 4)


def test_blame_recall_counts_only_tests_reported_as_newly_failing():
    pooled = pool_rows([row(truth={"a", "b"}, selected={"a", "b"}, newly={"a"})])

    assert pooled.recall == 1.0 and pooled.blame_recall == pytest.approx(1 / 2)  # picked is not the same as blamed


def test_a_newly_failing_test_that_is_not_a_real_failure_is_a_false_alarm():
    pooled = pool_rows([row(truth={"a"}, selected={"a"}, newly={"a", "z"}), row(truth=set(), newly={"q"})])

    assert pooled.false_newly_failing == 2


def test_time_is_scored_only_where_tests_were_selected_or_really_failed():
    pooled = pool_rows([row(truth={"a"}, selected={"a"}, ratio=0.4), row(truth={"b"}, selected={"b"}, ratio=0.6), row(ratio=0.0), row(ratio=0.0)])

    assert pooled.median_ratio == pytest.approx(0.5)
    assert pooled.untested_rows == 2


def test_skipped_runs_are_counted_and_do_not_enter_the_rates():
    pooled = pool_rows([row(truth={"a"}, selected={"a"}), row(truth={"b"}, selected=set(), skipped=True)])

    assert pooled.skipped == 1 and pooled.recall == 1.0


def test_changes_the_selection_only_partly_covered_are_counted():
    pooled = pool_rows([row(truth={"a", "b"}, selected={"a"}), row(truth={"c"}, selected={"c"}), row(truth={"d"}, selected=set())])

    assert pooled.partly_missed_changes == 2


def test_nothing_failing_gives_no_recall():
    assert pool_rows([row()]).recall is None


# -- the verdict ------------------------------------------------------------------------


def rows_of(n=10, **kw):
    base = {"truth": {"a", "b"}, "selected": {"a", "b"}, "static": {"a"}, "newly": {"a", "b"}, "ratio": 0.2}
    return [row(**{**base, **kw}) for _ in range(n)]


def test_all_thresholds_met_continues():
    assert verdict(pool_rows(rows_of())) == "CONTINUE"


def test_a_false_newly_failing_test_means_fix():
    assert verdict(pool_rows([*rows_of(10), row(truth=set(), newly={"z"})])) == "FIX"


def test_low_recall_means_fix_and_very_low_recall_means_stop():
    assert verdict(pool_rows(rows_of(selected={"a"}))) == "FIX"
    assert verdict(pool_rows(rows_of(truth={"a", "b", "c"}, selected={"a"}))) == "STOP"


def test_picked_but_not_blamed_means_fix():
    assert verdict(pool_rows(rows_of(newly={"a"}))) == "FIX"


def test_a_slow_check_means_fix():
    assert verdict(pool_rows(rows_of(ratio=0.8))) == "FIX"


def test_too_many_skipped_runs_is_inconclusive():
    assert verdict(pool_rows([*rows_of(6), *[row(skipped=True)] * 4])) == "INCONCLUSIVE"


def test_exactly_the_skip_limit_is_still_allowed():
    assert verdict(pool_rows([*rows_of(10), *[row(skipped=True)] * 2])) == "CONTINUE"


def test_the_threshold_values_are_inclusive_where_the_plan_says_at_least():
    assert verdict(pool_rows(rows_of(n=20, selected={"a", "b"}, ratio=0.5))) == "CONTINUE"  # ratio exactly 0.5
    assert verdict(pool_rows([row(truth={"a", "b"}, selected={"a"}, newly={"a"})] * 10)) == "FIX"  # recall exactly 0.5 is not a stop


def test_a_sample_with_too_few_really_failing_tests_is_no_data():
    assert verdict(pool_rows(rows_of(n=3))) == "NO DATA"


def test_no_failing_test_anywhere_is_no_data():
    assert verdict(pool_rows([row()] * 5)) == "NO DATA"
