"""What a function edit touched, read from the syntax trees of the old and the new function (nothing is run).

Three kinds, strongest first: a CONDITION (an `if`, `while`, `assert`, conditional expression, comprehension filter or `match` guard),
a RESULT (a returned or raised value) and other STATEMENTS (assignments, calls, loop headers, `with` items, `except` types, `match`
patterns, `break` / `continue`, imports, and the order of all of these). "" means none of the three changed (a docstring, a name, a
decorator, a default value); the edit stays unclassified, never guessed. The kind says where an edit reaches, not whether it is right:
a condition rewritten into an equal one still reads as a condition edit.
"""

from __future__ import annotations

import ast
from collections import Counter

KINDS = ("condition", "result", "statements")
FUNCTION = ast.FunctionDef | ast.AsyncFunctionDef
_SIMPLE = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr, ast.Delete, ast.Break, ast.Continue, ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom)
_NONE = ast.dump(ast.Constant(value=None))  # a bare `return` and `return None` are the same result


def _statement(node: ast.AST) -> str | None:
    """The text that stands for a plain statement or a statement header, None for anything else."""
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return None  # a bare string is a docstring or a note, not a statement that does anything
    if isinstance(node, _SIMPLE):
        return ast.dump(node)
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return ast.dump(node.iter)
    if isinstance(node, ast.withitem):
        return ast.dump(node)
    if isinstance(node, ast.ExceptHandler):
        return ast.dump(node.type) if node.type is not None else "bare except"
    if isinstance(node, ast.match_case):
        return ast.dump(node.pattern)
    return None


def _parts(function: ast.AST) -> tuple[Counter[str], Counter[str], list[str]]:
    """(conditions, results, statements in walk order): the walk is breadth first, so moving a statement changes the list."""
    conditions: Counter[str] = Counter()
    results: Counter[str] = Counter()
    statements: list[str] = []
    for node in ast.walk(function):
        if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            conditions[ast.dump(node.test)] += 1
        elif isinstance(node, ast.comprehension):
            conditions.update(ast.dump(test) for test in node.ifs)
        elif isinstance(node, ast.match_case) and node.guard is not None:
            conditions[ast.dump(node.guard)] += 1
        elif isinstance(node, ast.Return):
            results[ast.dump(node.value) if node.value is not None else _NONE] += 1
        elif isinstance(node, ast.Raise):
            results[ast.dump(node.exc) if node.exc is not None else "re-raise"] += 1
        if isinstance(node, ast.Raise) and node.cause is not None:
            statements.append(ast.dump(node.cause))
        elif isinstance(node, ast.Assert) and node.msg is not None:
            statements.append(ast.dump(node.msg))
        text = _statement(node)
        if text is not None:
            statements.append(text)
    return conditions, results, statements


def edit_kind(old: FUNCTION, new: FUNCTION) -> str:
    """The strongest kind that differs between the two functions, or "" when none does (or the function is too deep to read)."""
    try:
        before, after = _parts(old), _parts(new)
    except RecursionError:  # an absurdly nested expression: say nothing rather than crash the whole comparison
        return ""
    return next((kind for kind, a, b in zip(KINDS, before, after, strict=True) if a != b), "")
