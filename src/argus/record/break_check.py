"""Break check (handoff backlog 24): pure helpers that judge two Argus claims by breaking functions on a throwaway copy of a repository.

H1: a higher blast-radius tier means more failing test functions. H2: the tests the recording says reached a function are the tests that
fail when it is broken. Standard library only, no file or process access here; the running is in scripts/break_check/run_break_check.py.
"""

from __future__ import annotations

import ast
import io
import itertools
import math
import random
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from argus.explore.blast import BlastNode, BlastTree, Tier
from argus.record.store import Recording

BREAK_MESSAGE = "argus-break-check"
TIERS: tuple[Tier, ...] = ("HIGH", "MED", "LOW")
_FLOAT_SLACK = 1e-12  # two taus equal on paper can differ in the last bits


@dataclass(frozen=True, slots=True)
class Candidate:
    """One function or method leaf of the blast tree that could be broken."""

    node_id: str  # "function:pkg.mod.name" or "method:pkg.mod.Class.name"
    qualname: str
    path: str  # repo-relative file
    line: int  # line of the `def`
    end_line: int  # last line of the function
    tier: Tier
    radius: int

    @property
    def one_line(self) -> bool:
        return self.line == self.end_line


@dataclass(frozen=True, slots=True)
class FailedTests:
    tests: frozenset[str]  # test FUNCTION ids "tests/test_a.py::Class.test_name" (no "[param]"), failed or errored
    collection_error: bool  # an ERROR line with no "::": the run could not even collect, so `tests` says nothing


@dataclass(frozen=True, slots=True)
class PrecisionRecall:
    precision: float | None  # |R & F| / |R|; None when R is empty
    recall: float | None  # |R & F| / |F|; None when F is empty


def _leaves(node: BlastNode) -> Iterator[BlastNode]:
    if node.kind in ("function", "method"):
        yield node
    for child in node.children:
        yield from _leaves(child)


def _end_lines(source: str) -> dict[int, int]:
    """`def` line -> last line of the function, for every function in `source`."""
    return {
        n.lineno: n.end_lineno
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.end_lineno is not None
    }


def tier_of(tree: BlastTree, sources: Mapping[str, str]) -> list[Candidate]:
    """Every function and method leaf of `tree` (nested functions are not in the tree) with its tier and radius.
    `sources` maps repo-relative path -> file text and is read only to find where each function ends."""
    ends: dict[str, dict[int, int]] = {}
    found = []
    for root in tree.roots:
        for leaf in _leaves(root):
            if leaf.qualname is None or leaf.line is None:
                raise ValueError(f"blast node {leaf.name} in {leaf.path} has no qualname or line")
            if leaf.path not in ends:
                ends[leaf.path] = _end_lines(sources[leaf.path])
            end = ends[leaf.path].get(leaf.line)
            if end is None:  # never guess where a function ends: a wrong range would break the wrong code
                raise ValueError(f"no function starts at {leaf.path}:{leaf.line} ({leaf.qualname})")
            found.append(Candidate(f"{leaf.kind}:{leaf.qualname}", leaf.qualname, leaf.path, leaf.line, end, leaf.tier, leaf.radius))
    return found


def sample_functions(candidates: Sequence[Candidate], per_tier: int, seed: int) -> list[Candidate]:
    """Up to `per_tier` candidates from each tier, drawn with random.Random(seed): repeatable, a smaller tier returns all of it,
    one-line functions are never drawn. Result order is stable: HIGH, MED, LOW, then by node_id."""
    rng = random.Random(seed)
    picked: list[Candidate] = []
    for tier in TIERS:
        pool = sorted((c for c in candidates if c.tier == tier and not c.one_line), key=lambda c: c.node_id)  # sorted: input order must not matter
        picked += sorted(rng.sample(pool, min(per_tier, len(pool))), key=lambda c: c.node_id)
    return picked


def insert_raise(source: str, line: int) -> str:
    """`source` with `raise RuntimeError("argus-break-check")` as the first statement of the function whose `def` is on `line`."""
    return insert_first_statement(source, line, f'raise RuntimeError("{BREAK_MESSAGE}")')


def insert_first_statement(source: str, line: int, statement: str) -> str:
    """`source` with `statement` as the first statement of the function whose `def` is on `line` (1-based), same indentation, after
    a docstring. Raises ValueError for a one-line function or a line that is not a `def`."""
    func = next(
        (n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.lineno == line), None,
    )
    if func is None:
        raise ValueError(f"line {line} is not a function definition")
    lines = io.StringIO(source, newline="").readlines()  # splits where the parser counts lines
    first = func.body[0]
    indent = lines[first.lineno - 1][: first.col_offset] if not getattr(first, "decorator_list", ()) else lines[first.decorator_list[0].lineno - 1][: first.col_offset]
    if indent.strip():
        raise ValueError(f"the function on line {line} has its body on the line of its signature")
    eol = "\r\n" if "\r\n" in source else "\n"
    new = f"{indent}{statement}"
    is_docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    decorators = getattr(first, "decorator_list", ())
    start = min([first.lineno, *(d.lineno for d in decorators)])  # a decorated first statement starts at its decorator, not at its `def`
    at = (first.end_lineno or first.lineno) if is_docstring else start - 1
    if at == len(lines) and not lines[-1].endswith(("\n", "\r")):  # nothing follows and the file has no final newline: keep it that way
        lines[-1] += eol
        return "".join([*lines, new])
    return "".join([*lines[:at], new + eol, *lines[at:]])


def _test_id(rest: str) -> str:
    """The `file::Class::test` part of a pytest summary line: ends at ' - message', brackets (a parameter id may hold ' - ' and '::') are cut."""
    depth, end = 0, len(rest)
    for i, ch in enumerate(rest):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0 and rest.startswith(" - ", i):
            end = i
            break
    return rest[:end].split("[", 1)[0]


def failed_tests(pytest_output: str) -> FailedTests:
    """Parse the FAILED and ERROR lines of `pytest -rfE --tb=no` into test function ids."""
    tests: set[str] = set()
    collection_error = False
    for text in pytest_output.splitlines():
        prefix = next((p for p in ("FAILED ", "ERROR ") if text.startswith(p)), None)
        if prefix is None or text[len(prefix):len(prefix) + 1].isspace() or len(text) == len(prefix):  # a log line reads "ERROR    root:..."
            continue
        file, sep, qual = _test_id(text[len(prefix):]).partition("::")
        if not sep:  # no test named: the file did not even load, or the line is not understood; either way `tests` is incomplete
            collection_error = True
        else:
            tests.add(f"{file}::{qual.replace('::', '.')}")
    return FailedTests(frozenset(tests), collection_error)


def reaching_tests(rec: Recording, node_id: str) -> frozenset[str]:
    """Test function ids (same form as `failed_tests`: file + code qualname) of the tests the recording says reached `node_id`."""
    by_key = {f.key: f for f in rec.functions}
    mine = {f.key for f in rec.functions if f.node_id == node_id}
    return frozenset(
        f"{by_key[t.test].file}::{by_key[t.test].code_qualname}" for t in rec.tests if mine & set(t.reached)
    )


def kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Kendall rank correlation tau-b (ties handled); None when either side is constant or there are fewer than 2 points."""
    pairs = list(itertools.combinations(zip(xs, ys, strict=True), 2))
    concordant = discordant = tied_x = tied_y = 0
    for (x1, y1), (x2, y2) in pairs:
        tied_x += x1 == x2
        tied_y += y1 == y2
        if x1 != x2 and y1 != y2:
            same_direction = (x1 < x2) == (y1 < y2)
            concordant += same_direction
            discordant += not same_direction
    denominator = (len(pairs) - tied_x) * (len(pairs) - tied_y)
    return (concordant - discordant) / math.sqrt(denominator) if denominator > 0 else None


def permutation_p(xs: Sequence[float], ys: Sequence[float], seed: int, n: int) -> float | None:
    """Two-sided permutation p-value of tau-b: (hits + 1) / (n + 1) over `n` shuffles of `ys` from random.Random(seed).
    None when tau-b is undefined."""
    observed = kendall_tau_b(xs, ys)
    if observed is None:
        return None
    rng = random.Random(seed)
    shuffled = list(ys)
    hits = 0
    for _ in range(n):
        rng.shuffle(shuffled)
        tau = kendall_tau_b(xs, shuffled)  # a shuffle of a non-constant side is non-constant, so tau exists
        hits += tau is not None and abs(tau) >= abs(observed) - _FLOAT_SLACK
    return (hits + 1) / (n + 1)


def precision_recall(reached: Collection[str], failed: Collection[str]) -> PrecisionRecall:
    """Precision and recall of R = tests the recording says reached, against F = tests that fail. An undefined side is None."""
    r, f = set(reached), set(failed)
    hits = len(r & f)
    return PrecisionRecall(hits / len(r) if r else None, hits / len(f) if f else None)


def pooled_precision_recall(pairs: Iterable[tuple[Collection[str], Collection[str]]]) -> PrecisionRecall:
    """Micro average over many (R, F) pairs: sum of |R & F| over sum of |R| (precision) and over sum of |F| (recall)."""
    hits = reached = failed = 0
    for r, f in pairs:
        r, f = set(r), set(f)
        hits += len(r & f)
        reached += len(r)
        failed += len(f)
    return PrecisionRecall(hits / reached if reached else None, hits / failed if failed else None)


def baseline_flaky(run1_failed: Collection[str], run2_failed: Collection[str]) -> frozenset[str]:
    """Tests that failed in either of two clean runs; they are excluded from every count."""
    return frozenset(run1_failed) | frozenset(run2_failed)


def drop_flaky(tests: Collection[str], flaky: Collection[str]) -> frozenset[str]:
    """`tests` without the flaky ones."""
    return frozenset(tests) - frozenset(flaky)
