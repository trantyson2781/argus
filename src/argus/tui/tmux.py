"""The s key: type a note about the selected finding into the Claude Code tmux pane. It never presses Enter: the user reads the note and sends it.

The pane is the one running `claude` in the window Argus is in (never Argus's own pane, never a pane running anything else), or the one named in
ARGUS_AGENT_PANE. The note is one line, so even a pane that does not understand bracketed paste cannot submit it.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping

from argus.explore.changes import Flag, SymbolChange
from argus.explore.render_diff import printable
from argus.explore.render_panel import ReachNote

AGENT_PANE_ENV = "ARGUS_AGENT_PANE"
AGENT_COMMANDS = ("claude",)  # `node` would also match a dev server: typing a note into that is worse than asking the user to name the pane
PANE_ID = re.compile(r"%[0-9]+")  # [0-9], not \\d: other scripts' digits are not pane numbers
MAX_NOTE = 600
MAX_LINE = 100
MAX_REACH = 4
TIMEOUT = 5
SCREEN_LINES = 15  # the part of the agent pane that is read before typing into it
PROMPT_MARKERS = ("do you want to", "esc to cancel", "(esc to")  # Claude Code asking for permission: digits typed there can answer it
EDIT_TEXT = {"condition": "a condition was edited", "result": "a returned or raised value was edited"}


def agent_pane(env: Mapping[str, str], run=subprocess.run) -> tuple[str | None, str]:
    """(pane id, message). The id is None when no pane can be named; the message then says why, or names the pane that was chosen."""
    if not env.get("TMUX"):
        return None, "Not inside tmux: run Argus in a tmux pane next to the agent."
    explicit = env.get(AGENT_PANE_ENV)
    if explicit is not None:
        if PANE_ID.fullmatch(explicit) and explicit == env.get("TMUX_PANE"):
            return None, f"{AGENT_PANE_ENV} names Argus's own pane; the note would be typed into Argus itself."
        if PANE_ID.fullmatch(explicit):
            return explicit, f"pane {explicit} ({AGENT_PANE_ENV})"
        return None, printable(f"{AGENT_PANE_ENV} must look like %3, not {explicit!r}.")
    own = env.get("TMUX_PANE", "")
    if not own:
        return None, "TMUX_PANE is not set, so Argus cannot tell which tmux window it is in."
    argv = ["tmux", "list-panes", "-t", own, "-F", "#{pane_id} #{pane_current_command}"]
    try:
        done = run(argv, capture_output=True, text=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as err:
        return None, f"tmux failed ({type(err).__name__}), so the agent pane cannot be found."
    if done.returncode:
        return None, "tmux failed to list the panes of this window, so the agent pane cannot be found."
    agents = [pane for pane, _, command in (line.partition(" ") for line in done.stdout.splitlines()) if pane != own and command.strip() in AGENT_COMMANDS]
    if len(agents) == 1:
        return agents[0], f"pane {agents[0]} (the pane running claude in this window)"
    if not agents:
        return None, f"No pane running claude in this tmux window; set {AGENT_PANE_ENV}=%N to name one."
    return None, f"Several panes run claude ({', '.join(agents)}); set {AGENT_PANE_ENV}=%N to choose."


def _what(change: SymbolChange) -> str:
    if change.change == "removed":
        return "it was deleted"
    if change.change == "added":
        return "it is new"
    if Flag.SIGNATURE_CHANGED in change.flags:
        return "its inputs or outputs changed"
    if change.docstring_only:
        return "only its description changed"
    return EDIT_TEXT.get(change.edit_kind, "its logic was edited")


def finding_note(change: SymbolChange, reach: ReachNote | None) -> str:
    """One line about a finding: where it is, what was edited, who uses it, which tests and entry points reach it, the first old and new line."""
    parts = [f"Please check {change.qualname} ({change.file}:{change.line}): {_what(change)}"]
    if change.known_callers is not None:
        parts.append(f"used in at least {change.known_callers} place{'' if change.known_callers == 1 else 's'}")
    if change.test_callers is not None:
        parts.append(f"{change.test_callers} test{'' if change.test_callers == 1 else 's'} found" if change.test_callers else "no test found")
    if reach is not None and reach.commands:
        more = f" +{len(reach.commands) - MAX_REACH} more" if len(reach.commands) > MAX_REACH else ""
        parts.append(f"reached from {', '.join(reach.commands[:MAX_REACH])}{more}")
    old = next((c.text.strip() for c in change.changed_lines if c.kind == "-"), None)
    new = next((c.text.strip() for c in change.changed_lines if c.kind == "+"), None)
    if old is not None or new is not None:
        parts.append(f"was `{(old or '')[:MAX_LINE]}` now `{(new or '')[:MAX_LINE]}`")
    note = printable("; ".join(parts) + ". Please explain the change and check what it can break.")
    return note if len(note) <= MAX_NOTE else note[: MAX_NOTE - 1] + "…"


def _tmux(run, *args: str) -> str | None:
    """The output of one tmux command, or None when it failed."""
    try:
        done = run(["tmux", *args], capture_output=True, text=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def send_text(pane: str, text: str, run=subprocess.run) -> str | None:
    """Type `text` into the pane (literally: no word is read as a key name) and stop there, Enter is the user's. None on success, else why not.
    Nothing is typed into a pane that is scrolled back (its keys would run as copy-mode commands) or that is asking for permission (a digit can answer it)."""
    mode = _tmux(run, "display", "-p", "-t", pane, "#{pane_in_mode}")
    if mode is None or mode.strip() not in ("0", "1"):
        return f"tmux could not tell what pane {pane} is doing; nothing was sent."
    if mode.strip() == "1":
        return f"Pane {pane} is in copy mode (scrolled back); leave it first (q). Nothing was sent."
    screen = _tmux(run, "capture-pane", "-p", "-t", pane)
    if screen is None:
        return f"tmux could not read pane {pane}; nothing was sent."
    if any(marker in line.lower() for line in screen.splitlines()[-SCREEN_LINES:] for marker in PROMPT_MARKERS):
        return f"Pane {pane} looks like it is asking for permission: answer it first. Nothing was sent."
    sent = _tmux(run, "send-keys", "-t", pane, "-l", "--", text)
    return None if sent is not None else f"tmux failed to send to {pane}; nothing was sent."
