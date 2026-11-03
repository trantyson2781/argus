"""Markdown report over one or more trace logs (plan §15, §19)."""

from __future__ import annotations

import math
from statistics import median

from argus.benchmark.metrics import bootstrap_ci, compare_paired, summarize, wilson_interval
from argus.benchmark.traces import RunTrace, latest_by_task

SMALL_SAMPLE = 50
_SEED = 0


def _money(value: float | None) -> str:
    return "n/a" if value is None or math.isnan(value) else f"${value:.3f}"


def _num(value: float, digits: int = 0) -> str:
    return "n/a" if math.isnan(value) else f"{value:,.{digits}f}"


def _run_section(label: str, all_traces: list[RunTrace]) -> list[str]:
    latest = latest_by_task(all_traces)
    s = summarize(latest, spend=all_traces)
    lines = [f"### {label}", ""]
    hashes = {t.config_hash for t in all_traces}
    if len(hashes) > 1:
        lines += [f"- WARNING: {len(hashes)} different configs in this file ({', '.join(sorted(hashes))}); "
                  "numbers below mix them and are not a valid result", ""]
    splits = {t.split for t in all_traces}
    if len(splits) > 1:
        lines += [f"- WARNING: traces from different splits in this file ({', '.join(sorted(splits))}); "
                  "dev and held-out must not be pooled", ""]
    lo, hi = wilson_interval(s.successes, s.n)
    cost_lo, cost_hi = bootstrap_ci([t.cost for t in latest], _SEED)
    tok_lo, tok_hi = bootstrap_ci([float(t.total_tokens) for t in latest], _SEED)
    rt_lo, rt_hi = bootstrap_ci([t.runtime_seconds for t in latest], _SEED)
    runtimes = [t.runtime_seconds for t in latest]
    lines += [
        f"- success: {s.successes}/{s.n} ({s.success_rate:.1%}); 95% Wilson CI [{lo:.1%}, {hi:.1%}]",
        f"- tokens (all attempts): {s.total_tokens:,}; per success: "
        + ("n/a" if s.tokens_per_success is None else f"{s.tokens_per_success:,.0f}")
        + f"; per task 95% CI [{_num(tok_lo)}, {_num(tok_hi)}]",
        f"- cost (all attempts): {_money(s.total_cost)}; per success: {_money(s.cost_per_success)}; "
        f"per task 95% CI [{_money(cost_lo)}, {_money(cost_hi)}]",
        f"- agent runtime: mean {s.mean_runtime_seconds:.1f}s, median {median(runtimes) if runtimes else math.nan:.1f}s, "
        f"95% CI [{_num(rt_lo, 1)}, {_num(rt_hi, 1)}]",
    ]
    errors = [t for t in latest if t.error]
    timeouts = sum(t.timed_out for t in latest)
    retried = sum(1 for t in latest if t.attempts > 1)
    lines.append(f"- errors: {len(errors)} unresolved infrastructure error(s) (excluded from paired tests); "
                 f"{retried} task(s) needed a retry; {timeouts} agent timeout(s) (counted as failures)")
    if any(t.cost_unknown for t in all_traces):
        lines.append("- WARNING: the model has no price in LiteLLM for some runs; cost is a lower bound")
    if s.n < SMALL_SAMPLE:
        lines.append(f"- small sample (n={s.n} < {SMALL_SAMPLE}): do not read differences as conclusions")
    return [*lines, ""]


def _comparison(baseline: str, label: str, base: list[RunTrace], cand: list[RunTrace]) -> list[str]:
    out = [f"### {label} vs {baseline}", ""]
    if {t.model for t in base} != {t.model for t in cand}:
        out.append("- WARNING: different models; this is not a controlled comparison")
    if {t.split for t in base} != {t.split for t in cand}:
        out.append("- WARNING: different splits; this is not a controlled comparison")
    p = compare_paired(base, cand)
    sizes = (len(base), len(cand))
    if p.n < min(sizes):
        out.append(f"- WARNING: only {p.n} tasks are shared and clean (runs have {sizes[0]} and {sizes[1]}); "
                   "headline rates above cover different task sets")
    out.append(f"- paired on {p.n} tasks ({p.excluded_errors} excluded for errors): both pass {p.both_pass}, "
               f"candidate-only {p.candidate_only}, baseline-only {p.baseline_only}, both fail {p.both_fail}")
    out.append(f"- McNemar exact p = {p.p_value:.3f}")
    return [*out, ""]


def render_report(runs: dict[str, list[RunTrace]]) -> str:
    """`runs` maps a label to its full trace log (all attempts); the first entry is the baseline."""
    if not runs:
        return "# Argus benchmark report\n\nNo traces found.\n"
    out = ["# Argus benchmark report", ""]
    for label, traces in runs.items():
        out += _run_section(label, traces)
    labels = list(runs)
    for label in labels[1:]:
        out += _comparison(labels[0], label, latest_by_task(runs[labels[0]]), latest_by_task(runs[label]))
    return "\n".join(out)
