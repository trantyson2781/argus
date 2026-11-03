"""Aggregate metrics and paired statistics over run traces (plan §15, §19)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean

from argus.benchmark.traces import RunTrace

BOOTSTRAP_RESAMPLES = 2000


@dataclass(frozen=True, slots=True)
class Summary:
    n: int
    successes: int
    success_rate: float
    total_tokens: int
    total_cost: float
    cost_per_success: float | None
    tokens_per_success: float | None
    mean_runtime_seconds: float


def summarize(traces: list[RunTrace], spend: list[RunTrace] | None = None) -> Summary:
    """`traces` are the final outcome per task. `spend` is every attempt (retries cost money
    too); re-evaluations of a stored patch are excluded because their usage is a copy."""
    n = len(traces)
    successes = sum(t.success for t in traces)
    paid = [t for t in (traces if spend is None else spend) if not t.reused_patch]
    tokens = sum(t.total_tokens for t in paid)
    cost = sum(t.cost for t in paid)
    return Summary(
        n=n,
        successes=successes,
        success_rate=successes / n if n else 0.0,
        total_tokens=tokens,
        total_cost=cost,
        cost_per_success=cost / successes if successes else None,
        tokens_per_success=tokens / successes if successes else None,
        mean_runtime_seconds=fmean(t.runtime_seconds for t in traces) if n else 0.0,
    )


def mcnemar_exact(candidate_only: int, baseline_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    low = min(candidate_only, baseline_only)
    tail = sum(math.comb(discordant, k) for k in range(low + 1)) * 0.5**discordant
    return min(1.0, 2 * tail)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True, slots=True)
class PairedComparison:
    n: int
    excluded_errors: int
    both_pass: int
    candidate_only: int
    baseline_only: int
    both_fail: int
    p_value: float


def compare_paired(baseline: list[RunTrace], candidate: list[RunTrace]) -> PairedComparison:
    """Compare two runs on the tasks both attempted cleanly.

    A task with an unresolved infrastructure error in either run is excluded and counted, so
    infrastructure noise cannot leak into the discordant pairs.
    """
    base = {t.task_id: t.success for t in baseline if not t.error}
    cand = {t.task_id: t.success for t in candidate if not t.error}
    attempted = {t.task_id for t in baseline} & {t.task_id for t in candidate}
    shared = sorted(base.keys() & cand.keys())
    both_pass = sum(base[i] and cand[i] for i in shared)
    cand_only = sum(cand[i] and not base[i] for i in shared)
    base_only = sum(base[i] and not cand[i] for i in shared)
    both_fail = len(shared) - both_pass - cand_only - base_only
    return PairedComparison(
        n=len(shared),
        excluded_errors=len(attempted) - len(shared),
        both_pass=both_pass,
        candidate_only=cand_only,
        baseline_only=base_only,
        both_fail=both_fail,
        p_value=mcnemar_exact(cand_only, base_only),
    )


def bootstrap_ci(values: list[float], seed: int, confidence: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    means = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_RESAMPLES))
    alpha = (1 - confidence) / 2
    return (means[int(alpha * BOOTSTRAP_RESAMPLES)], means[int((1 - alpha) * BOOTSTRAP_RESAMPLES) - 1])
