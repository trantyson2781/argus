"""The `argus record` summary: it must name every reason the recording may be incomplete (`argus.record.cli.summary_lines`)."""

from __future__ import annotations

from argus.record.cli import summary_lines
from argus.record.store import ObservedFunction, Processes, Recording


def rec(processes: Processes, files=(("a.py", "h1"),), functions=()) -> Recording:
    return Recording("0.1.0", "3.13", ("pytest",), "2026-10-23T10:00:00Z", 1.0, 0, files, functions, (), (), (), processes)


def text(recording: Recording) -> str:
    return "\n".join(summary_lines(recording, ".argus/recordings/x.json"))


def test_a_clean_run_has_no_warning() -> None:
    fn = (ObservedFunction("a.py:1:f", "a.py", 1, "f", "function:a.f", None, (), 1),)

    out = text(rec(Processes(1, 0), functions=fn))

    assert "!" not in out and "1 functions ran: 1 matched" in out and "x.json" in out


def test_each_reason_for_an_incomplete_recording_is_named() -> None:
    out = text(rec(Processes(0, 2, 3), files=(("a.py", None), ("b.py", None))))

    assert "Nothing was recorded" in out
    assert "2 processes did not save" in out
    assert "failed 3 times" in out
    assert "2 files changed or vanished" in out


def test_unmatched_functions_are_counted_apart() -> None:
    fns = (
        ObservedFunction("a.py:1:f", "a.py", 1, "f", "function:a.f", None, (), 1),
        ObservedFunction("a.py:2:<lambda>", "a.py", 2, "<lambda>", None, "anonymous", (), 1),
    )

    assert "2 functions ran: 1 matched" in text(rec(Processes(1, 0), functions=fns))


def test_a_function_that_was_only_called_and_never_started_did_not_run() -> None:
    fns = (
        ObservedFunction("a.py:1:f", "a.py", 1, "f", "function:a.f", None, (), 1),
        ObservedFunction("a.py:5:coro", "a.py", 5, "coro", "function:a.coro", None, (), 0),  # an async function called but never awaited
    )

    assert "1 functions ran: 1 matched" in text(rec(Processes(1, 0), functions=fns))
