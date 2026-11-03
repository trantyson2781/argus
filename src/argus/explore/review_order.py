"""Suggested review order of changed symbols, and the attention label each one gets.

Kind of change comes first because it is the one thing static analysis reports reliably: a removed or interface-changed symbol can break
its callers, a body edit alone does not. Inside a level, the part of the body that was edited (a condition before a returned value before the rest) orders
symbols, then the known-caller count. It is a lower bound (calls through
`obj.method()`, aliases and inheritance are not linked), so it never stands alone. A label says where to look first, never that a change is right.
"""

from __future__ import annotations

from enum import IntEnum

from argus.analysis.code_type import classify_file, is_production
from argus.explore.changes import Flag, SymbolChange


class Attention(IntEnum):
    """How soon a reader should look at a change. The value is the sort order."""

    CHECK = 0  # removed, or its inputs or outputs changed: other code may break
    LOOK = 1  # its logic was edited
    LOW = 2  # new, or only its description changed


def attention(change: SymbolChange) -> Attention:
    if change.change == "removed" or Flag.SIGNATURE_CHANGED in change.flags:
        return Attention.CHECK
    if change.change == "added" or change.docstring_only:
        return Attention.LOW
    return Attention.LOOK


# What the edit touched. An edit that could not be classified ("") stays above plain statements: unread must never sink below a known small edit.
EDIT_RANK = {"condition": 0, "result": 1, "": 2, "statements": 3}


def review_key(change: SymbolChange) -> tuple[bool, int, int, int, str, int, str, str]:
    """Sort key: production before test code, then attention, then what the edit touched (a condition, a returned or raised value, the rest),
    more known callers first (unknown counts as 0), then the stable (file, line, name, change) order."""
    return (
        not is_production(classify_file(change.file)), attention(change), EDIT_RANK.get(change.edit_kind, EDIT_RANK['']), -(change.known_callers or 0),
        change.file, change.line, change.qualname, change.change,
    )


def order_symbols(symbols: tuple[SymbolChange, ...]) -> tuple[SymbolChange, ...]:
    return tuple(sorted(symbols, key=review_key))
