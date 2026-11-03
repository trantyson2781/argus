"""One plain sentence for the OBSERVED facts of a symbol, shared by `explore symbol` and the TUI panel.

The `[O]` tag marks a fact seen while the code ran. It is never a FACT or an INFERENCE (those are read from the code), it covers only the
code paths that ran in one recording, and it says STALE when a file it rests on changed afterwards.
"""

from __future__ import annotations

from argus.explore.render import short_name
from argus.explore.render_diff import printable
from argus.record.observed import ObservedFacts, SeenCaller

TAG = "[O]"
MAX_TESTS = 3
MAX_CALLERS = 3
MAX_COMMAND = 40
MAX_FILES = 2


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _command(text: str) -> str:
    return text if len(text) <= MAX_COMMAND else text[: MAX_COMMAND - 3] + "..."


def _caller(c: SeenCaller) -> str:
    where = f" (line {c.lines[0]})" if len(c.lines) == 1 else f" (lines {', '.join(map(str, c.lines[:3]))})" if c.lines else ""
    return f"{short_name(c.name)}{where}"


def _more(shown: int, total: int) -> str:
    return f", +{total - shown} more" if total > shown else ""


def observed_text(facts: ObservedFacts) -> str:
    when = f"{facts.recorded_at}, {_command(facts.command)}"
    parts = [f"{_plural(facts.calls, 'call')} (last recording {when})" if facts.calls else f"did not run in the last recording ({when})"]
    if facts.tests:
        names = ", ".join(short_name(t) for t in facts.tests[:MAX_TESTS])
        parts.append(f"run by {_plural(len(facts.tests), 'test')}: {names}{_more(MAX_TESTS, len(facts.tests))}")
    also = [c for c in facts.callers if c.also]
    if also:
        parts.append("also called by " + ", ".join(_caller(c) for c in also[:MAX_CALLERS]) + _more(MAX_CALLERS, len(also)))
    if facts.outside_calls:
        parts.append(f"{_plural(facts.outside_calls, 'call')} from outside the repo")
    text = " · ".join(parts)
    if facts.stale:
        text = f"STALE ({', '.join(facts.stale_files[:MAX_FILES])}{_more(MAX_FILES, len(facts.stale_files))} changed after the recording) " + text
    if facts.incomplete:
        text += f" (incomplete: {'; '.join(facts.incomplete)})"
    return printable(f"{text}  {TAG}")
