"""Gate for part B of the breakage check (`explore/call_fit.py`): inject a change into a throwaway copy, run the tests, and compare the
calls Argus says no longer fit with the `TypeError` / `NameError` the tests really raise.

Pure helpers only (text to text, text to numbers); `scripts/call_fit_gate/run_gate.py` does the copying and running. Every mutation gives
Python that parses or raises ValueError, so a copy is never left in a state nobody can read.
"""

from __future__ import annotations

import ast
import difflib
import io
import re
import tokenize
from collections.abc import Collection, Iterable
from dataclasses import dataclass

from argus.record.break_check import insert_first_statement

NEW_PARAMETER = "argus_new_required"
RENAME_SUFFIX = "_renamed"
RECEIVER_NAMES = ("self", "cls")

PRECISION_CONTINUE, RECALL_CONTINUE, PRECISION_STOP = 0.95, 0.8, 0.9  # written down before the first run (handoff section 0.5)
MAX_SKIP_RATE = 0.2  # more skipped runs than this and a kind says nothing: the breaks that stop a whole test module from loading are the real ones

_COMPOUND_FIELDS = ("body", "orelse", "handlers", "finalbody", "cases")
_FRAME = re.compile(r"^(?P<file>\S+\.py):(?P<line>\d+): in (?P<func>\S+)\s*$")
_ERROR = re.compile(r"^E\s+(?P<exc>TypeError|NameError|AttributeError): (?P<message>.*)$")

Site = tuple[str, int]  # (file, line) of a call


@dataclass(frozen=True, slots=True)
class BindingError:
    """A `TypeError` / `NameError` / `AttributeError` a test run raised, at the innermost frame pytest printed for it."""

    file: str
    line: int
    func: str
    exc: str
    message: str


@dataclass(frozen=True, slots=True)
class Scores:
    """One mutated function. `flagged`: sites Argus says do not fit; `observed`: sites where a test really failed; `resolved`: sites
    Argus has a call link for; `executed`: sites the clean test run reached (a failure also proves its site was reached)."""

    flagged_executed: int  # flagged sites a test reached: the only ones the tests can confirm or refute
    confirmed: int  # flagged sites where a test really failed
    observed_resolved: int  # real failures at sites Argus has a link for
    caught: int  # of those, the ones Argus flagged
    observed: int  # real failures, linked or not


@dataclass(frozen=True, slots=True)
class Pooled:
    precision: float | None
    recall: float | None
    resolved_share: float | None  # real failures at sites Argus can see at all, over all real failures


def score_sites(flagged: Collection[Site], observed: Collection[Site], resolved: Collection[Site], executed: Collection[Site]) -> Scores:
    flagged_set, observed_set, resolved_set = set(flagged), set(observed), set(resolved)
    observed_resolved = observed_set & resolved_set
    return Scores(
        len(flagged_set & (set(executed) | observed_set)), len(flagged_set & observed_set), len(observed_resolved),
        len(flagged_set & observed_resolved), len(observed_set),
    )


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def pool(scores: Iterable[Scores]) -> Pooled:
    rows = list(scores)
    total = Scores(*(sum(getattr(r, f) for r in rows) for f in Scores.__slots__))
    return Pooled(
        _rate(total.confirmed, total.flagged_executed), _rate(total.caught, total.observed_resolved), _rate(total.observed_resolved, total.observed),
    )


@dataclass(frozen=True, slots=True)
class KindResult:
    """The pooled numbers of one kind of change. `attempted` counts every sampled function, `skipped` those whose run could not be read."""

    kind: str
    attempted: int
    skipped: int
    pooled: Pooled
    wrong_fact: int


def kind_verdict(result: KindResult) -> str:
    """STOP, INCONCLUSIVE, FIX, CONTINUE or NO DATA for one kind, by the thresholds written before the first run."""
    precision, recall = result.pooled.precision, result.pooled.recall
    if precision is None:
        return "NO DATA"
    if precision < PRECISION_STOP:
        return "STOP"
    if result.attempted and result.skipped / result.attempted > MAX_SKIP_RATE:
        return "INCONCLUSIVE"
    if recall is None:
        return "NO DATA"
    return "CONTINUE" if precision >= PRECISION_CONTINUE and recall >= RECALL_CONTINUE and result.wrong_fact == 0 else "FIX"


def overall_verdict(results: Iterable[KindResult]) -> str:
    """Every kind with data must pass on its own; one good kind never props up another."""
    verdicts = {kind_verdict(r) for r in results} - {"NO DATA"}
    for worst in ("STOP", "FIX", "INCONCLUSIVE"):
        if worst in verdicts:
            return worst
    return "CONTINUE" if verdicts else "NO DATA"


def line_map(original: str, mutated: str) -> dict[int, int]:
    """Line number in `original` -> line number in `mutated`, for the lines that are unchanged (a mutation may add or remove lines)."""
    old, new = _lines(original), _lines(mutated)
    mapping: dict[int, int] = {}
    for tag, i1, i2, j1, _ in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            mapping.update({i + 1: j1 + (i - i1) + 1 for i in range(i1, i2)})
    return mapping


def expand_statements(source: str, lines: Collection[int]) -> set[int]:
    """`lines` plus every line of the simple statements they start. Coverage reports the first line of a statement; a call can sit on a later one."""
    marked = set(lines)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return marked
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and node.lineno in lines and not any(hasattr(node, f) for f in _COMPOUND_FIELDS):
            marked.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return marked


def binding_errors(pytest_output: str) -> list[BindingError]:
    """The call-binding failures in `pytest --tb=short` output, each with the innermost frame printed above it."""
    found: list[BindingError] = []
    frame: re.Match[str] | None = None
    for text in pytest_output.splitlines():
        if text.startswith("___"):
            frame = None
        elif (match := _FRAME.match(text)):
            frame = match
        elif frame and (error := _ERROR.match(text)):
            found.append(BindingError(frame["file"], int(frame["line"]), frame["func"], error["exc"], error["message"]))
    return found


# -- mutations -------------------------------------------------------------------------------------------------------


def _function_at(source: str, line: int) -> ast.FunctionDef | ast.AsyncFunctionDef:
    func = next((n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.lineno == line), None)
    if func is None:
        raise ValueError(f"line {line} is not a function definition")
    return func


def _lines(source: str) -> list[str]:
    return io.StringIO(source, newline="").readlines()  # splits where the parser counts lines


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    """Character offset of an `ast` position (its column counts UTF-8 bytes)."""
    text = lines[lineno - 1]
    return sum(len(x) for x in lines[: lineno - 1]) + len(text.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


def _signature_close(source: str, func: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[tuple[int, int], str]:
    """((row, column) of the `)` that ends the parameter list, text of the last token before it)."""
    start = (func.lineno, len(_lines(source)[func.lineno - 1].encode("utf-8")[: func.col_offset].decode("utf-8", errors="ignore")))
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    begin = next(i for i, t in enumerate(tokens) if t.string == "def" and t.start >= start)
    depth, last = 0, "("
    for token in tokens[begin + 2 :]:  # past `def` and the name
        if token.type in (tokenize.NL, tokenize.COMMENT, tokenize.NEWLINE):
            continue
        if token.string in "([{" and token.type == tokenize.OP:
            depth += 1
        elif token.string in ")]}" and token.type == tokenize.OP:
            depth -= 1
            if depth == 0:
                return token.start, last
        if depth > 0 and not (depth == 1 and token.string == "("):
            last = token.string
    raise ValueError("the parameter list is not closed")


def _splice(source: str, start: int, end: int, text: str) -> str:
    return source[:start] + text + source[end:]


def _parsed(source: str) -> str:
    """`source` itself when it parses; ValueError otherwise, so a mutation never leaves a file nobody can read."""
    try:
        ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as err:
        raise ValueError("the changed code does not parse") from err
    return source


def add_required_parameter(source: str, line: int) -> str:
    """The function on `line` with one more required, keyword-only parameter: valid after defaults, and no call that omits it fits."""
    func = _function_at(source, line)
    if func.args.kwarg:
        raise ValueError("a function with **kwargs takes the new parameter anywhere only if it is placed before them")
    try:
        (row, col), last = _signature_close(source, func)
    except (StopIteration, tokenize.TokenError) as err:
        raise ValueError("the parameter list could not be located") from err
    item = ("" if func.args.vararg or func.args.kwonlyargs else "*, ") + NEW_PARAMETER
    glue = "" if last == "(" else " " if last == "," else ", "
    at = sum(len(x) for x in _lines(source)[: row - 1]) + col
    return _parsed(_splice(source, at, at, glue + item))


def rename_first_parameter(source: str, line: int) -> str:
    """The first ordinary parameter (not `self` / `cls`) renamed in the signature; the body keeps working through `old = new`."""
    func = _function_at(source, line)
    candidates = list(func.args.args)
    if candidates and candidates[0].arg in RECEIVER_NAMES and not func.args.posonlyargs:
        candidates = candidates[1:]
    if not candidates:
        raise ValueError("no parameter can be renamed without changing what a positional call means")
    arg = candidates[0]
    renamed = insert_first_statement(source, line, f"{arg.arg} = {arg.arg}{RENAME_SUFFIX}")  # below the signature, so `arg`'s position holds
    at = _offset(_lines(renamed), arg.lineno, arg.col_offset)
    return _parsed(_splice(renamed, at, at + len(arg.arg), arg.arg + RENAME_SUFFIX))


def remove_first_default(source: str, line: int) -> str:
    """The first parameter that has a default, made required. Raises ValueError when none has one or the default is parenthesised."""
    func = _function_at(source, line)
    args = func.args
    positional = [*args.posonlyargs, *args.args]
    paired = [*zip(positional[len(positional) - len(args.defaults) :], args.defaults), *((k, d) for k, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None)]
    if not paired:
        raise ValueError("the function has no default to remove")
    arg, default = paired[0]
    lines = _lines(source)
    new = _splice(source, _offset(lines, arg.end_lineno or arg.lineno, arg.end_col_offset or 0), _offset(lines, default.end_lineno or default.lineno, default.end_col_offset or 0), "")
    return _parsed(new)  # a parenthesised default leaves a stray `)`: refused


def delete_function(source: str, line: int) -> str:
    """The function on `line` removed, decorators included. Raises ValueError when what is left does not parse (an emptied class)."""
    func = _function_at(source, line)
    first = min([func.lineno, *(d.lineno for d in func.decorator_list)])
    lines = _lines(source)
    new = "".join([*lines[: first - 1], *lines[(func.end_lineno or func.lineno) :]])
    try:
        ast.parse(new)
    except SyntaxError as err:
        raise ValueError("removing the function leaves code that does not parse") from err
    return new


def _own_returns(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Return]:
    """The `return` statements of `func` itself, not those of functions, lambdas or classes defined inside it."""
    found: list[ast.Return] = []
    stack: list[ast.AST] = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            found.append(node)
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            stack.extend(ast.iter_child_nodes(node))
    return found


def nullify_returns(source: str, line: int) -> str:
    """Every returned value of the function on `line` replaced by `None`: the function still runs, but what it hands back is wrong.
    Raises ValueError when it returns no value."""
    returns = [r for r in _own_returns(_function_at(source, line)) if r.value is not None]
    if not returns:
        raise ValueError("the function returns no value")
    lines = _lines(source)
    new = source
    for node in sorted(returns, key=lambda r: (r.value.lineno, r.value.col_offset), reverse=True):  # from the end, so earlier offsets stay valid
        value = node.value
        start, end = _offset(lines, value.lineno, value.col_offset), _offset(lines, value.end_lineno or value.lineno, value.end_col_offset or 0)
        new = _splice(new, start, end, "None")
    return _parsed(new)

