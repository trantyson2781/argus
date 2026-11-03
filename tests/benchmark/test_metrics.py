from __future__ import annotations

import math

import pytest

from argus.benchmark.metrics import (
    bootstrap_ci,
    compare_paired,
    mcnemar_exact,
    summarize,
)
from tests.benchmark.test_traces import _trace


def test_summarize_reports_success_and_cost_per_success() -> None:
    traces = [
        _trace("a", True, cost=0.10, input_tokens=900, output_tokens=100),
        _trace("b", False, cost=0.30, input_tokens=1800, output_tokens=200),
    ]
    s = summarize(traces)
    assert s.n == 2 and s.successes == 1
    assert s.success_rate == 0.5
    assert s.cost_per_success == pytest.approx(0.40)
    assert s.tokens_per_success == 3000


def test_summarize_with_no_successes_has_no_cost_per_success() -> None:
    s = summarize([_trace("a", False)])
    assert s.cost_per_success is None and s.tokens_per_success is None


def test_summarize_empty_is_safe() -> None:
    assert summarize([]).n == 0


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) * 0.5**10)


def test_compare_paired_counts_four_outcomes_on_shared_tasks_only() -> None:
    base = [_trace("a", True), _trace("b", False), _trace("c", True), _trace("d", False), _trace("only-base", True)]
    cand = [_trace("a", True), _trace("b", True), _trace("c", False), _trace("d", False)]
    p = compare_paired(base, cand)
    assert (p.both_pass, p.candidate_only, p.baseline_only, p.both_fail) == (1, 1, 1, 1)
    assert p.n == 4 and p.p_value == 1.0 and p.excluded_errors == 0


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = bootstrap_ci(values, seed=0)
    assert (lo, hi) == bootstrap_ci(values, seed=0)
    assert lo <= 3.0 <= hi
    assert math.isnan(bootstrap_ci([], seed=0)[0])


def test_compare_paired_excludes_errored_tasks() -> None:
    base = [_trace("a", True), _trace("b", False, error="infra")]
    cand = [_trace("a", True), _trace("b", True)]
    p = compare_paired(base, cand)
    assert p.n == 1 and p.excluded_errors == 1 and p.candidate_only == 0


def test_summarize_spend_includes_retries_but_not_reused_patch_copies() -> None:
    final = [_trace("a", True, cost=0.5)]
    spend = [_trace("a", False, cost=1.0, error="x"), _trace("a", True, cost=0.5), _trace("a", True, cost=0.5, reused_patch=True)]
    s = summarize(final, spend=spend)
    assert s.total_cost == pytest.approx(1.5) and s.cost_per_success == pytest.approx(1.5)


def test_wilson_interval_known_values() -> None:
    from argus.benchmark.metrics import wilson_interval
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert math.isnan(wilson_interval(0, 0)[0])
