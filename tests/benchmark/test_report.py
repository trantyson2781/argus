from __future__ import annotations

from argus.benchmark.report import render_report
from tests.benchmark.test_traces import _trace


def test_report_summarises_each_run_and_counts_infrastructure_errors() -> None:
    runs = {"base": [_trace("a", True, cost=0.2), _trace("b", False, error="api down")]}
    text = render_report(runs)
    assert "base" in text and "1/2" in text and "50.0%" in text
    assert "1 unresolved infrastructure error" in text


def test_report_uses_only_the_latest_trace_per_task() -> None:
    runs = {"base": [_trace("a", False, error="x"), _trace("a", True)]}
    assert "1/1" in render_report(runs)


def test_report_pairs_candidate_against_first_run() -> None:
    runs = {
        "base": [_trace("a", True), _trace("b", False)],
        "cand": [_trace("a", True), _trace("b", True)],
    }
    text = render_report(runs)
    assert "cand vs base" in text and "candidate-only 1" in text and "baseline-only 0" in text


def test_report_warns_when_runs_use_different_models() -> None:
    runs = {"base": [_trace("a", model="m1")], "cand": [_trace("a", model="m2")]}
    assert "different models" in render_report(runs).lower()


def test_report_flags_small_samples() -> None:
    assert "small sample" in render_report({"base": [_trace("a")]}).lower()


def test_report_handles_empty_input() -> None:
    assert "no traces" in render_report({}).lower()


def test_report_counts_spend_across_all_attempts_but_success_from_the_latest() -> None:
    runs = {"base": [_trace("a", False, cost=1.0, error="x"), _trace("a", True, cost=0.5, attempts=2)]}
    text = render_report(runs)
    assert "1/1" in text and "$1.500" in text and "1 task(s) needed a retry" in text


def test_report_does_not_double_count_reused_patch_usage() -> None:
    runs = {"base": [_trace("a", False, cost=1.0, error="docker", error_kind="evaluator"),
                     _trace("a", True, cost=1.0, reused_patch=True, attempts=2)]}
    assert "$1.000" in render_report(runs)


def test_report_flags_mixed_configs_in_one_file() -> None:
    runs = {"base": [_trace("a", config_hash="h1"), _trace("b", config_hash="h2")]}
    assert "2 different configs" in render_report(runs)


def test_report_flags_mixed_splits_in_one_file() -> None:
    text = render_report({"base": [_trace("a", split="dev"), _trace("b", split="heldout")]})
    assert "WARNING" in text and "dev" in text and "heldout" in text and "split" in text.lower()


def test_report_is_quiet_when_a_file_has_one_split() -> None:
    assert "split" not in render_report({"base": [_trace("a"), _trace("b")]}).lower()


def test_report_warns_when_compared_runs_use_different_splits() -> None:
    text = render_report({"base": [_trace("a", split="dev")], "cand": [_trace("a", split="heldout")]})
    assert "different splits" in text


def test_report_excludes_errored_tasks_from_the_paired_test_and_says_so() -> None:
    runs = {
        "base": [_trace("a", True), _trace("b", False, error="x")],
        "cand": [_trace("a", True), _trace("b", True)],
    }
    text = render_report(runs)
    assert "paired on 1 tasks (1 excluded for errors)" in text and "only 1 tasks are shared" in text


def test_report_flags_unknown_cost_and_wilson_interval() -> None:
    text = render_report({"base": [_trace("a", True, cost_unknown=True)]})
    assert "lower bound" in text and "Wilson" in text
