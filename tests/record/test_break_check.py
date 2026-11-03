"""Break check, pure helpers (`argus.record.break_check`): sampling, inserting the raise, reading pytest output, rank statistics, precision and recall.

Written test-first: every number below is worked out by hand, not taken from the code under test.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

from argus.explore.blast import build_blast_tree
from argus.extract import scan
from argus.record.break_check import (
    BREAK_MESSAGE,
    Candidate,
    PrecisionRecall,
    baseline_flaky,
    drop_flaky,
    failed_tests,
    insert_raise,
    kendall_tau_b,
    permutation_p,
    pooled_precision_recall,
    precision_recall,
    sample_functions,
    reaching_tests,
    tier_of,
)
from argus.record.store import ObservedFunction, ObservedTest, Processes, Recording

RAISE = f'raise RuntimeError("{BREAK_MESSAGE}")'


# --- tier_of -------------------------------------------------------------------------------------------------------------------------

MOD = '''\
def core():
    return 1


def a():
    return core()


def b():
    return core()


def c():
    return core()


class K:
    def meth(self):
        return 1


def outer():
    def inner():
        return 2
    return inner()


def one(): return 1
'''
TEST_MOD = "from pkg.mod import core, outer\n\n\ndef test_core():\n    assert core() == 1\n"


def test_tier_of_reads_tier_radius_and_lines_of_function_and_method_leaves_only(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text(MOD, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mod.py").write_text(TEST_MOD, encoding="utf-8")
    tree = build_blast_tree(scan(tmp_path), tmp_path)

    # Act
    found = {c.qualname: c for c in tier_of(tree, {"pkg/mod.py": MOD})}

    # Assert
    assert set(found) == {"pkg.mod.core", "pkg.mod.a", "pkg.mod.b", "pkg.mod.c", "pkg.mod.K.meth", "pkg.mod.outer", "pkg.mod.one"}  # no inner, no test
    core = found["pkg.mod.core"]
    assert (core.node_id, core.path, core.line, core.end_line) == ("function:pkg.mod.core", "pkg/mod.py", 1, 2)
    assert (core.radius, core.tier) == (3, "MED")  # a, b and c reach it; MED starts at 3
    assert (found["pkg.mod.a"].radius, found["pkg.mod.a"].tier) == (0, "LOW")
    assert found["pkg.mod.K.meth"].node_id == "method:pkg.mod.K.meth"
    assert found["pkg.mod.one"].one_line and not core.one_line


def test_tier_of_an_empty_tree_gives_no_candidates(tmp_path: Path) -> None:
    tree = build_blast_tree(scan(tmp_path), tmp_path)

    assert tier_of(tree, {}) == []


# --- sample_functions ----------------------------------------------------------------------------------------------------------------

def cand(name: str, tier: str, radius: int = 0, one_line: bool = False) -> Candidate:
    return Candidate(f"function:m.{name}", f"m.{name}", "m.py", 1, 1 if one_line else 3, tier, radius)  # type: ignore[arg-type]


POOL = (
    [cand(f"h{i}", "HIGH", 20 + i) for i in range(10)] + [cand(f"m{i}", "MED", 4) for i in range(10)] + [cand(f"l{i}", "LOW") for i in range(3)]
    + [cand("tiny_h", "HIGH", 30, one_line=True), cand("tiny_l", "LOW", one_line=True)]
)


def test_sampling_takes_per_tier_from_each_tier_and_all_of_a_small_tier() -> None:
    picked = sample_functions(POOL, per_tier=4, seed=0)

    by_tier = {t: [c for c in picked if c.tier == t] for t in ("HIGH", "MED", "LOW")}
    assert [len(by_tier[t]) for t in ("HIGH", "MED", "LOW")] == [4, 4, 3]  # LOW has only three usable functions, so all of them
    assert {c.qualname for c in by_tier["LOW"]} == {"m.l0", "m.l1", "m.l2"}
    assert len({c.node_id for c in picked}) == len(picked)


def test_sampling_never_picks_a_one_line_function() -> None:
    picked = sample_functions(POOL, per_tier=100, seed=0)

    assert len(picked) == 23
    assert not any(c.one_line for c in picked)


def test_the_same_seed_gives_the_same_sample_whatever_the_input_order() -> None:
    first = sample_functions(POOL, per_tier=4, seed=7)
    shuffled = list(POOL)
    random.Random(1).shuffle(shuffled)

    assert sample_functions(POOL, per_tier=4, seed=7) == first
    assert sample_functions(shuffled, per_tier=4, seed=7) == first


def test_a_different_seed_changes_the_sample() -> None:
    base = {c.node_id for c in sample_functions(POOL, per_tier=4, seed=0)}

    others = [{c.node_id for c in sample_functions(POOL, per_tier=4, seed=s)} for s in range(1, 6)]

    assert any(other != base for other in others)  # ten choose four is 210: five seeds all equal to seed 0 would mean the seed is ignored


def test_sampling_nothing_or_zero_per_tier_gives_an_empty_list() -> None:
    assert sample_functions([], per_tier=5, seed=0) == []
    assert sample_functions(POOL, per_tier=0, seed=0) == []


# --- insert_raise --------------------------------------------------------------------------------------------------------------------

def test_a_plain_function_gets_the_raise_as_its_first_statement() -> None:
    source = "def f(x):\n    y = x\n    return y\n"

    assert insert_raise(source, 1) == f"def f(x):\n    {RAISE}\n    y = x\n    return y\n"


def test_only_the_function_on_the_given_line_is_changed() -> None:
    source = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"

    assert insert_raise(source, 5) == f"def a():\n    return 1\n\n\ndef b():\n    {RAISE}\n    return 2\n"


def test_a_method_gets_the_raise_at_the_depth_of_its_body() -> None:
    source = "class K:\n    x = 1\n\n    def go(self):\n        return self.x\n"

    assert insert_raise(source, 4) == f"class K:\n    x = 1\n\n    def go(self):\n        {RAISE}\n        return self.x\n"


def test_an_async_function_gets_the_raise() -> None:
    source = "async def f():\n    return await g()\n"

    assert insert_raise(source, 1) == f"async def f():\n    {RAISE}\n    return await g()\n"


def test_a_generator_gets_the_raise_and_stays_a_generator() -> None:
    source = "def gen():\n    yield 1\n    yield 2\n"

    result = insert_raise(source, 1)

    assert result == f"def gen():\n    {RAISE}\n    yield 1\n    yield 2\n"
    ast.parse(result)


def test_a_decorated_function_is_found_by_its_def_line_and_the_decorator_stays_above() -> None:
    source = "import functools\n\n\n@functools.cache\ndef f(x):\n    return x\n"

    assert insert_raise(source, 5) == f"import functools\n\n\n@functools.cache\ndef f(x):\n    {RAISE}\n    return x\n"


def test_the_raise_goes_after_a_one_line_docstring_so_it_stays_a_docstring() -> None:
    source = 'def f():\n    """Doc."""\n    return 1\n'

    result = insert_raise(source, 1)

    assert result == f'def f():\n    """Doc."""\n    {RAISE}\n    return 1\n'
    assert ast.get_docstring(ast.parse(result).body[0]) == "Doc."  # type: ignore[arg-type]


def test_the_raise_goes_after_a_multi_line_docstring() -> None:
    source = 'def f():\n    """Doc.\n\n    More text.\n    """\n    return 1\n'

    result = insert_raise(source, 1)

    assert result == f'def f():\n    """Doc.\n\n    More text.\n    """\n    {RAISE}\n    return 1\n'
    assert ast.get_docstring(ast.parse(result).body[0]) == "Doc.\n\nMore text."  # type: ignore[arg-type]


def test_a_function_whose_body_is_only_a_docstring_gets_the_raise_after_it() -> None:
    source = 'def f():\n    """Doc."""\n'

    result = insert_raise(source, 1)

    assert result == f'def f():\n    """Doc."""\n    {RAISE}\n'


def test_a_function_inside_a_function_inside_a_class_keeps_the_nested_indentation() -> None:
    source = "class K:\n    def outer(self):\n        def inner(x):\n            return x\n        return inner(1)\n"

    result = insert_raise(source, 3)

    assert result == f"class K:\n    def outer(self):\n        def inner(x):\n            {RAISE}\n            return x\n        return inner(1)\n"


def test_a_signature_over_several_lines_gets_the_raise_in_the_body_not_in_the_signature() -> None:
    source = "def f(\n    a,\n    b,\n):\n    return a + b\n"

    result = insert_raise(source, 1)

    assert result == f"def f(\n    a,\n    b,\n):\n    {RAISE}\n    return a + b\n"


def test_a_function_with_a_two_space_body_keeps_its_own_indentation() -> None:
    source = "def f():\n  return 1\n"

    assert insert_raise(source, 1) == f"def f():\n  {RAISE}\n  return 1\n"


def test_a_source_without_a_final_newline_still_parses_after_the_insert() -> None:
    source = "def f():\n    return 1"

    ast.parse(insert_raise(source, 1))


@pytest.mark.parametrize("source", ["def f(): return x\n", "async def f(): return x\n", "class K:\n    def m(self): pass\n"])
def test_a_one_line_function_is_refused(source: str) -> None:
    line = 2 if source.startswith("class") else 1

    with pytest.raises(ValueError):
        insert_raise(source, line)


@pytest.mark.parametrize("line", [0, 3, 99])
def test_a_line_that_is_not_a_def_is_refused(line: int) -> None:
    source = "x = 1\n\n\ndef f():\n    return x\n"  # line 3 is blank, 0 and 99 are outside

    with pytest.raises(ValueError):
        insert_raise(source, line)


# --- failed_tests --------------------------------------------------------------------------------------------------------------------

def test_a_plain_failure_gives_the_test_function_id() -> None:
    out = "=== short test summary info ===\nFAILED tests/test_a.py::test_x - assert 1 == 2\n=== 1 failed, 3 passed in 0.1s ===\n"

    result = failed_tests(out)

    assert result.tests == {"tests/test_a.py::test_x"}
    assert result.collection_error is False


def test_a_failure_in_a_class_is_named_with_the_class() -> None:
    out = "FAILED tests/test_a.py::TestK::test_y - boom\nFAILED tests/test_a.py::Outer::Inner::test_z\n"

    assert failed_tests(out).tests == {"tests/test_a.py::TestK.test_y", "tests/test_a.py::Outer.Inner.test_z"}


def test_parametrized_cases_merge_into_one_test_function() -> None:
    out = (
        "FAILED tests/test_a.py::test_p[1-2] - assert False\n"
        "FAILED tests/test_a.py::test_p[3-4] - assert False\n"
        "FAILED tests/test_a.py::test_p[a - b] - assert False\n"  # the id itself contains ' - '
        "FAILED tests/test_a.py::test_p[x::y]\n"  # and '::'
        "FAILED tests/test_a.py::TestK::test_q[0]\n"
    )

    assert failed_tests(out).tests == {"tests/test_a.py::test_p", "tests/test_a.py::TestK.test_q"}


def test_an_error_on_a_fixture_counts_as_that_test_failing() -> None:
    out = "ERROR tests/test_b.py::test_f - fixture 'db' not found\nERROR tests/test_b.py::TestK::test_g[1] - RuntimeError\n"

    result = failed_tests(out)

    assert result.tests == {"tests/test_b.py::test_f", "tests/test_b.py::TestK.test_g"}
    assert result.collection_error is False


def test_a_collection_error_is_a_flag_and_not_a_failing_test() -> None:
    out = "FAILED tests/test_a.py::test_x\nERROR tests/test_c.py - ImportError: cannot import name 'gone'\n"

    result = failed_tests(out)

    assert result.collection_error is True
    assert result.tests == {"tests/test_a.py::test_x"}


def test_output_with_no_failures_gives_an_empty_result() -> None:
    out = "....\n=== 4 passed in 0.02s ===\n"

    result = failed_tests(out)

    assert result.tests == frozenset()
    assert result.collection_error is False


def test_empty_output_gives_an_empty_result() -> None:
    assert failed_tests("") == failed_tests("....\n4 passed\n")


def test_a_line_in_the_middle_of_captured_text_that_only_mentions_failed_is_not_a_failure() -> None:
    out = "some log: the FAILED tests/test_a.py::test_x was retried\n"

    assert failed_tests(out).tests == frozenset()


# --- reaching_tests ------------------------------------------------------------------------------------------------------------------

def test_reaching_tests_gives_file_and_qualname_of_each_test_that_reached_the_node() -> None:
    # Arrange
    def fn(file: str, qual: str, node_id: str | None, line: int = 1) -> ObservedFunction:
        return ObservedFunction(f"{file}:{line}:{qual}", file, line, qual, node_id, None, (), 1)

    target = fn("pkg/m.py", "core", "function:pkg.m.core")
    other = fn("pkg/m.py", "other", "function:pkg.m.other", 5)
    t1 = fn("tests/test_m.py", "test_one", "function:tests.test_m.test_one")
    t2 = fn("tests/test_m.py", "TestK.test_two", "method:tests.test_m.TestK.test_two", 9)
    t3 = fn("tests/test_m.py", "test_three", "function:tests.test_m.test_three", 20)
    rec = Recording(
        "0.1.0", "3.13", ("t",), "2026-10-24T10:00:00Z", 1.0, 0, (), (target, other, t1, t2, t3), (), (),
        (ObservedTest(t1.key, (target.key,)), ObservedTest(t2.key, (other.key, target.key)), ObservedTest(t3.key, (other.key,))), Processes(1, 0),
    )

    # Act / Assert
    assert reaching_tests(rec, "function:pkg.m.core") == {"tests/test_m.py::test_one", "tests/test_m.py::TestK.test_two"}
    assert reaching_tests(rec, "function:pkg.m.missing") == frozenset()


# --- kendall_tau_b and permutation_p -------------------------------------------------------------------------------------------------

def test_a_perfect_increasing_order_is_plus_one() -> None:
    assert kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_a_perfect_decreasing_order_is_minus_one() -> None:
    assert kendall_tau_b([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_when_one_side_is_all_ties_tau_is_undefined() -> None:
    assert kendall_tau_b([1, 2, 3], [5, 5, 5]) is None
    assert kendall_tau_b([2, 2, 2], [1, 2, 3]) is None


def test_fewer_than_two_points_is_undefined() -> None:
    assert kendall_tau_b([], []) is None
    assert kendall_tau_b([1], [1]) is None


def test_tau_b_with_ties_is_corrected_for_them() -> None:
    # xs = [1, 2, 2, 3], ys = [1, 1, 2, 3]; six pairs:
    #   (1,2) x differs, y tied   -> tied in y only
    #   (1,3) concordant          (1,4) concordant
    #   (2,3) x tied, y differs   -> tied in x only
    #   (2,4) concordant          (3,4) concordant
    # C = 4, D = 0, n0 = 6, pairs tied in x = 1, pairs tied in y = 1
    # tau-b = (C - D) / sqrt((n0 - 1) * (n0 - 1)) = 4 / 5 = 0.8
    assert kendall_tau_b([1, 2, 2, 3], [1, 1, 2, 3]) == pytest.approx(0.8)


def test_tau_b_with_one_discordant_pair() -> None:
    # xs = [1, 2, 3], ys = [1, 3, 2]: pairs (1,2) C, (1,3) C, (2,3) D -> (2 - 1) / 3 = 1/3
    assert kendall_tau_b([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)


def test_tau_b_does_not_change_when_both_sides_swap() -> None:
    xs, ys = [1, 2, 2, 5, 7], [3, 1, 4, 4, 9]

    assert kendall_tau_b(xs, ys) == pytest.approx(kendall_tau_b(ys, xs))  # type: ignore[arg-type]


def test_the_permutation_p_value_is_the_same_for_the_same_seed() -> None:
    xs, ys = [1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 4, 3, 8, 5, 7, 6]

    first = permutation_p(xs, ys, seed=3, n=500)

    assert first == permutation_p(xs, ys, seed=3, n=500)
    assert first is not None and 0 < first <= 1


def test_a_perfect_order_of_six_values_has_a_small_p_value() -> None:
    # exactly 2 of the 720 orders are as extreme (+1 and -1), so p is about 0.003; with 999 shuffles and the +1 smoothing it stays under 0.02
    p = permutation_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], seed=0, n=999)

    assert p is not None and p < 0.02


def test_when_every_order_is_at_least_as_extreme_p_is_exactly_one() -> None:
    # xs = [1, 2, 3], ys = [1, 3, 2]: tau = 1/3, and all six orders of ys give |tau| >= 1/3, so hits = n and p = (n + 1) / (n + 1)
    assert permutation_p([1, 2, 3], [1, 3, 2], seed=0, n=200) == pytest.approx(1.0)


def test_p_value_is_none_when_tau_is_undefined() -> None:
    assert permutation_p([1, 2, 3], [4, 4, 4], seed=0, n=100) is None


def test_the_permutation_does_not_change_the_inputs() -> None:
    xs, ys = [1, 2, 3, 4], [4, 3, 1, 2]

    permutation_p(xs, ys, seed=0, n=50)

    assert (xs, ys) == ([1, 2, 3, 4], [4, 3, 1, 2])


# --- precision_recall ----------------------------------------------------------------------------------------------------------------

def test_precision_and_recall_of_partly_overlapping_sets() -> None:
    # R = {a, b, c, d}, F = {c, d, e}: R & F = {c, d}; precision 2/4, recall 2/3
    result = precision_recall({"a", "b", "c", "d"}, {"c", "d", "e"})

    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(2 / 3)


def test_identical_sets_give_one_and_one() -> None:
    assert precision_recall({"a", "b"}, {"a", "b"}) == PrecisionRecall(1.0, 1.0)


def test_nothing_reached_makes_precision_undefined_and_recall_zero() -> None:
    assert precision_recall(set(), {"a", "b"}) == PrecisionRecall(None, 0.0)


def test_nothing_failing_makes_recall_undefined_and_precision_zero() -> None:
    assert precision_recall({"a"}, set()) == PrecisionRecall(0.0, None)


def test_both_empty_makes_both_undefined() -> None:
    assert precision_recall(set(), set()) == PrecisionRecall(None, None)


def test_lists_are_accepted_and_duplicates_do_not_count_twice() -> None:
    assert precision_recall(["a", "a", "b"], ["a"]) == PrecisionRecall(0.5, 1.0)


def test_pooled_precision_and_recall_add_the_counts_over_all_functions() -> None:
    # core: R = F = 3 tests; helper: R = F = 1; risky: R = 1, F = 0; unused: both empty
    pairs = [({"a", "b", "c"}, {"a", "b", "c"}), ({"d"}, {"d"}), ({"e"}, set()), (set(), set())]
    # hits = 3 + 1 + 0 + 0 = 4; sum |R| = 5; sum |F| = 4 -> precision 4/5, recall 4/4
    result = pooled_precision_recall(pairs)

    assert result.precision == pytest.approx(0.8)
    assert result.recall == pytest.approx(1.0)


def test_pooled_over_nothing_or_only_empty_pairs_is_undefined() -> None:
    assert pooled_precision_recall([]) == PrecisionRecall(None, None)
    assert pooled_precision_recall([(set(), set())]) == PrecisionRecall(None, None)


def test_pooled_recall_is_defined_even_when_no_test_was_reached_anywhere() -> None:
    assert pooled_precision_recall([(set(), {"a"})]) == PrecisionRecall(None, 0.0)


# --- baseline_flaky ------------------------------------------------------------------------------------------------------------------

def test_a_test_failing_in_either_clean_run_is_flaky() -> None:
    assert baseline_flaky({"t1", "t2"}, {"t2", "t3"}) == {"t1", "t2", "t3"}
    assert baseline_flaky(set(), set()) == frozenset()


def test_flaky_tests_are_dropped_from_a_count() -> None:
    flaky = baseline_flaky({"t1"}, {"t3"})

    assert drop_flaky({"t1", "t2", "t3"}, flaky) == {"t2"}
    assert drop_flaky(["t2"], frozenset()) == {"t2"}


# --- audit fixes (2026-10-24) --------------------------------------------------------------------------------------------------------

def test_a_decorated_nested_function_as_first_statement_gets_the_raise_above_its_decorator() -> None:
    source = "def f():\n    @dec\n    def g():\n        return 1\n    return g\n"

    result = insert_raise(source, 1)

    assert result == f"def f():\n    {RAISE}\n    @dec\n    def g():\n        return 1\n    return g\n"
    ast.parse(result)


def test_a_log_line_that_starts_with_error_is_not_a_collection_error() -> None:
    out = "ERROR    root:app.py:12 something was logged\nFAILED tests/test_a.py::test_x\n"

    result = failed_tests(out)

    assert result.collection_error is False
    assert result.tests == {"tests/test_a.py::test_x"}
