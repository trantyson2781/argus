"""Gate for part A of the breakage check (`argus check`): score what `check_change` picked and reported against what really fails.

Pure scoring and the verdict; `scripts/check_gate/run_gate.py` makes the changes, runs the whole suite for the truth and calls `check_change`.
Thresholds written before the first run (handoff section 0.5, with the additions of the 2026-10-25 audit): CONTINUE when selection recall
>= 0.95 with a fresh recording, blame recall (tests reported newly failing out of the really failing ones) >= 0.95, 0 false "newly failing"
tests and the median time of the check <= 0.5 of the full suite, over at least 20 really failing tests; STOP below recall 0.5; FIX
otherwise; INCONCLUSIVE when more than 20% of the runs were skipped. Static-only and partial-recording recall are reported apart.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass

RECALL_CONTINUE, RECALL_STOP = 0.95, 0.5
RATIO_CONTINUE = 0.5
MAX_SKIP_RATE = 0.2
MIN_TRUTH = 20  # really failing tests the sample must hold before a rate means anything


@dataclass(frozen=True, slots=True)
class Row:
    """One changed function. `truth`: tests that really fail with it (clean failures and flaky tests removed); `selected`: tests `argus check`
    picked; `static`: tests it would pick without a recording; `partial`: tests it would pick with a recording of half the test files;
    `newly_failing`: what it reported; `ratio`: seconds of the check over seconds of the clean full suite. A run that could not be
    judged (the truth run unreadable) is `skipped`; a check that did not decide is NOT skipped: it picked nothing and reported nothing."""

    function: str
    kind: str
    truth: frozenset[str]
    selected: frozenset[str]
    static: frozenset[str]
    newly_failing: frozenset[str]
    ratio: float
    skipped: bool = False
    partial: frozenset[str] = frozenset()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Pooled:
    runs: int
    skipped: int
    truth_total: int
    recall: float | None
    static_recall: float | None
    partial_recall: float | None
    blame_recall: float | None
    false_newly_failing: int
    median_ratio: float | None
    untested_rows: int  # runs where nothing was selected and nothing really failed: left out of the time
    partly_missed_changes: int  # changes with a really failing test the selection did not contain


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def pool_rows(rows: Iterable[Row]) -> Pooled:
    all_rows = list(rows)
    used = [r for r in all_rows if not r.skipped]
    truth = sum(len(r.truth) for r in used)
    timed = [r.ratio for r in used if r.selected or r.truth]
    return Pooled(
        len(all_rows), len(all_rows) - len(used), truth,
        _share(sum(len(r.truth & r.selected) for r in used), truth),
        _share(sum(len(r.truth & r.static) for r in used), truth),
        _share(sum(len(r.truth & r.partial) for r in used), truth),
        _share(sum(len(r.truth & r.newly_failing) for r in used), truth),
        sum(len(r.newly_failing - r.truth) for r in used),
        statistics.median(timed) if timed else None,
        len(used) - len(timed),
        sum(1 for r in used if r.truth - r.selected),
    )


def verdict(pooled: Pooled) -> str:
    if pooled.runs and pooled.skipped / pooled.runs > MAX_SKIP_RATE:
        return "INCONCLUSIVE"
    if pooled.recall is None or pooled.blame_recall is None or pooled.median_ratio is None or pooled.truth_total < MIN_TRUTH:
        return "NO DATA"
    if pooled.recall < RECALL_STOP:
        return "STOP"
    ok = (
        pooled.recall >= RECALL_CONTINUE and pooled.blame_recall >= RECALL_CONTINUE and pooled.false_newly_failing == 0
        and pooled.median_ratio <= RATIO_CONTINUE
    )
    return "CONTINUE" if ok else "FIX"
