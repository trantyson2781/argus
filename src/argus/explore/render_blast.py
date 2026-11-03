"""Text for the blast-radius tree: a label per node, a detail line for the highlighted one and a legend.

The numbers are minimums (calls Argus cannot link are not counted) and the tier says where to be careful, never that code is wrong.
The ran marks come from one recording: `●` ran, `○` did not run, `?` the file changed since, so the recording no longer says.
"""

from __future__ import annotations

from rich.text import Text

from argus.explore.blast import MED_FLOOR, BlastNode, BlastTree
from argus.explore.render_diff import printable

TIER_STYLES = {"HIGH": "bold magenta", "MED": "cyan", "LOW": "dim"}
RAN, NOT_RAN, CHANGED = "●", "○", "?"
SYMBOLS = ("function", "method")


def _ran_mark(node: BlastNode) -> str:
    if node.ran is None:
        return ""
    if node.kind in SYMBOLS:
        return CHANGED if node.stale else RAN if node.ran else NOT_RAN
    if not node.total:
        return ""
    return f"{node.ran}/{node.total} ran" + (f" {CHANGED}" if node.stale else "")


def blast_label(node: BlastNode) -> Text:
    text = Text(printable(node.name), style="bold" if node.kind in ("folder", "file") else "")
    text.append(f"  {node.tier} {node.radius}", style=TIER_STYLES[node.tier])
    if node.kind in SYMBOLS and node.direct:
        text.append(f" ({node.direct} direct)", style="dim")
    mark = _ran_mark(node)
    if mark:
        text.append(f"  {mark}")
    return text


def _recording_sentence(node: BlastNode) -> str:
    if node.ran is None:
        return ""
    if node.stale:
        return f" The file changed after the recording, so whether it ran is unknown ({CHANGED})."
    if node.kind in SYMBOLS:
        return " It ran in the last recording." if node.ran else " It did not run in the last recording."
    return f" {node.ran} of {node.total} functions inside ran in the last recording."


def blast_detail(node: BlastNode) -> str:
    """Two lines about the highlighted node. Repository names are escaped."""
    if node.qualname and node.line:
        first = f"{node.qualname}   ({node.path}:{node.line})"
        second = f"{node.radius} functions depend on it and {node.direct} call it directly: at least that many, calls Argus cannot link are not counted."
    else:
        first = node.path
        second = f"The most depended-on code inside has {node.radius} dependents (at least) across {node.total} functions."
    return printable(f"{first}\n{second}{_recording_sentence(node)}")


def blast_legend(tree: BlastTree) -> str:
    text = (
        f"Blast radius = functions that reach it through calls Argus can link. HIGH {tree.high_at}+ · MED {MED_FLOOR}+ · LOW fewer. "
        "Most depended-on first. A minimum, and where to be careful, not a verdict."
    )
    if tree.has_recording:
        text += f"  {RAN} ran  {NOT_RAN} did not run  {CHANGED} file changed since the recording (last recording only)"
    return text
