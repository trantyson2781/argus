"""The text of the ? help screen: every key, the Python that R and C use, and the tmux setting of s."""

from __future__ import annotations

from argus.explore.render_diff import printable
from argus.tui.pythonchoice import ENV, FILE, MIN_RECORDING, PythonChoice

KEYS = (
    ("1 2 3 4", "Changes, Map, Symbol, Diff"),
    ("t", "in the Map: switch between the level map and the blast-radius tree"),
    ("d", "the long report of the changes"),
    ("r", "start fresh from the current state"),
    ("R", "record the repository's tests (asks first); it fills in the 'Seen running' lines"),
    ("C", "run argus check on your uncommitted change (asks first): which selected tests newly fail"),
    ("s", "type a note about the highlighted change into the Claude Code tmux pane (never presses Enter)"),
    ("e", "open the editor at the highlighted line"),
    ("/", "search for a function, class or method"),
    ("b  h", "base and head ref of the Diff view"),
    ("Esc", "leave a text box, or close this help"),
    ("?", "this help"),
    ("q", "quit"),
)


def help_lines(choice: PythonChoice) -> list[str]:
    lines = ["Keys"]
    lines += [f"  {key:<8}{what}" for key, what in KEYS]
    lines += ["", "Python used by R and C"]
    if choice.error:
        lines += [f"  {choice.error} R and C will not run until that is fixed."]
    else:
        lines += [f"  {choice.path}   (from {choice.source})"]
    lines += [
        f"  To change it: set {ENV}=/path/to/python, or write that path in {FILE} in the repository.",
        f"  Recording needs Python {MIN_RECORDING[0]}.{MIN_RECORDING[1]} or newer with the repository's packages installed; Argus never builds the environment.",
        "", "Agent pane used by s",
        "  The only pane running claude in this tmux window; set ARGUS_AGENT_PANE=%N to name another pane.",
    ]
    return [printable(line) for line in lines]
