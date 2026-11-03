"""The lines that differ between the old and the new text of one function: what it said before and what it says now.

Pure and standard library only. Lines are compared exactly apart from trailing spaces, because a different indentation changes what
Python code does. Nothing here says whether the edit is right or what it does to behaviour.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

MAX_LINES = 60  # per function; a bigger rewrite is cut here (the panel shows far fewer)
MAX_COMPARE = 2000  # old + new lines; above this difflib can take minutes on repetitive text, so the function is not compared
MAX_TEXT = 300  # characters kept of one line (a minified line is not stored whole)
NOTE_UNREADABLE = "the old or the new text could not be read"
NOTE_TOO_LARGE = "this function is too large to compare here (git diff shows what changed)"
NOTE_SAME = "the lines look the same; the code differs in a way the text cannot show (a space inside a string, for example)"


@dataclass(frozen=True, slots=True)
class ChangedLine:
    kind: Literal["-", "+"]  # "-" was there before and is gone, "+" is there now
    text: str  # tabs shown as 4 spaces; indentation relative to the function's own `def` line
    line: int  # 1-based line in its own file: the old file for "-", the new file for "+"


def _shown(line: str) -> str:
    text = line.expandtabs(4)
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "…"


def diff_lines(before: Sequence[str], first_before: int, after: Sequence[str], first_after: int) -> tuple[ChangedLine, ...]:
    """The differing lines of two texts, in order, old lines of a changed block before its new lines. `first_*` is the file line of `[0]`."""
    old, new = [line.rstrip() for line in before], [line.rstrip() for line in after]
    found: list[ChangedLine] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        found += [ChangedLine("-", _shown(old[i]), first_before + i) for i in range(i1, i2)]
        found += [ChangedLine("+", _shown(new[j]), first_after + j) for j in range(j1, j2)]
    return tuple(found[:MAX_LINES])


def function_diff(old: tuple[int, Sequence[str]] | None, new: tuple[int, Sequence[str]] | None) -> tuple[tuple[ChangedLine, ...], str]:
    """(changed lines, note) for two `(file line of the def, function lines)` pairs. The note says why there are no lines to show
    (unreadable, too large, or the text is the same), and is empty when there are lines."""
    if old is None or new is None:
        return (), NOTE_UNREADABLE
    if len(old[1]) + len(new[1]) > MAX_COMPARE:
        return (), NOTE_TOO_LARGE
    lines = diff_lines(old[1], old[0], new[1], new[0])
    return (lines, "") if lines else ((), NOTE_SAME)
