"""ASCII renderers for the system and dependency maps. Pure: data in, lines out; at most MAX_WIDTH columns.

Boxes are containment. `--->` is always an import ("A ---> B": A imports B). Layer order is import depth,
never execution order. Tags: no tag = FACT, [I] = INFERENCE, [?] = UNKNOWN.
"""

from __future__ import annotations

import textwrap
from itertools import zip_longest

from argus.explore.claims import Claim, Tier
from argus.explore.diagram import OPEN_ALL, Arrow, DepsView, Node, Part, Subsystem, SystemView, purpose_text

MAX_WIDTH = 78
BOX_WIDTH = MAX_WIDTH - 4          # full-width box inside the 4-column layer gutter
HALF_BOX_WIDTH = (BOX_WIDTH - 2) // 2
MAX_PARTS_SHOWN = 12
MAX_MODULES_SHOWN = 16
MAX_OUTSIDE_LINES = 12
ARROW = "--->"
_FOLD = {"—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "..."}

LEGEND_SYSTEM = (
    "boxes = containment (what a package holds)     A ---> B = A imports B",
    "[+] collapsed (--open NAME)    no tag = FACT   [I] INFERENCE   [?] UNKNOWN",
    "Box order is import layering only, NOT execution order.",
)
LEGEND_DEPS = (
    "A ---> B = A imports B; n = import edges. L0 = top of the import stack;",
    "arrows point down to a higher L. Layers are import depth, NOT execution order.",
    "no tag = FACT   [I] INFERENCE   [?] UNKNOWN",
)


def ascii_safe(text: str) -> str:
    folded = "".join(_FOLD.get(ch, ch) for ch in text)
    return "".join(ch if ch.isascii() and (ch.isprintable() or ch == " ") else "~" for ch in folded)


def tag(tier: Tier) -> str:
    return {Tier.FACT: "", Tier.INFERENCE: "[I]", Tier.UNKNOWN: "[?]"}[tier]


def fit(text: str, width: int) -> str:
    """Keep the tail (the part that matters in a path); never exceed width."""
    return text if len(text) <= width else "..." + text[len(text) - width + 3:]


def clip(text: str, width: int) -> str:
    """Keep the head (names start there); mark the cut."""
    return text if len(text) <= width else text[: width - 3] + "..."


def wrap(text: str, width: int, first: str = "", rest: str = "") -> list[str]:
    lines = textwrap.wrap(
        ascii_safe(text), width=width, initial_indent=first, subsequent_indent=rest, break_long_words=True,
        break_on_hyphens=False,
    )
    return lines or [first.rstrip()]


def box(title: str, body: list[str], width: int) -> list[str]:
    title = clip(ascii_safe(title), width - 5)
    top = f"+- {title} " + "-" * (width - len(title) - 5) + "+"
    inner = width - 4
    rows = [f"| {fit(line, inner).ljust(inner)} |" for line in body]
    return [top, *rows, "+" + "-" * (width - 2) + "+"]


def _ref(claim_or_arrow: Claim | Arrow) -> str:
    claim = claim_or_arrow.claim if isinstance(claim_or_arrow, Arrow) else claim_or_arrow
    return claim.evidence[0].location if claim.evidence else ""


MAX_PURPOSE_LINES = 3


def _purpose_lines(purpose: Claim, refs: bool, width: int) -> list[str]:
    if purpose.tier is Tier.UNKNOWN:
        lines = wrap(f"[?] {purpose_text(purpose)}", width)
    else:
        lines = wrap(f'"{purpose_text(purpose)}"', width)
        if len(lines) > MAX_PURPOSE_LINES:
            lines = [*lines[: MAX_PURPOSE_LINES - 1], clip(lines[MAX_PURPOSE_LINES - 1], width - 4) + ' ..."']
    if refs and purpose.evidence:
        lines.append(f"  docstring at {_ref(purpose)}")
    return lines


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


# ---------------------------------------------------------------- system map

def render_system(view: SystemView, refs: bool) -> list[str]:
    parts_total = sum(len(s.parts) for s in view.subsystems)
    head, tail = "SYSTEM MAP  ", f"  |  {_count(view.files, 'file')} scanned, {parts_total} production components"
    label = clip(ascii_safe(view.label), max(MAX_WIDTH - len(head) - len(tail), 8))  # a long directory name must not break the width
    out = [head + label + tail, *LEGEND_SYSTEM, ""]
    arrow_of = {(a.src, a.dst): a for a in view.arrows}
    for index, subsystem in enumerate(view.subsystems):
        out += box(subsystem.name, _subsystem_body(subsystem, view.opened, refs), BOX_WIDTH + 4)
        nxt = view.subsystems[index + 1] if index + 1 < len(view.subsystems) else None
        link = arrow_of.get((subsystem.name, nxt.name)) if nxt else None
        mark = f" {tag(link.claim.tier)}" if link and tag(link.claim.tier) else ""
        out += ["", f"      {subsystem.name} ---> {nxt.name}  ({_count(link.count, 'import edge')}){mark}", ""] if link else [""]
    out += _arrow_table(view.arrows, refs)
    out += ["", "UNKNOWN", *_unknown_lines(view.unknowns)]
    out += ["", "Next: --view deps (import layers)   --open NAME (expand)   --refs (sources)"]
    return out


def _unknown_lines(claims: tuple[Claim, ...]) -> list[str]:
    lines: list[str] = []
    for claim in claims:
        wrapped = wrap(claim.text, MAX_WIDTH - 6)
        lines += [f"  [?] {wrapped[0]}", *[f"      {line}" for line in wrapped[1:]]]
    return lines


def _arrow_table(arrows: tuple[Arrow, ...], refs: bool) -> list[str]:
    if not arrows:
        return ["IMPORTS BETWEEN SUBSYSTEMS", "  (none between subsystems; components inside one: --view deps)"]
    out = ["IMPORTS BETWEEN SUBSYSTEMS  (A ---> B: A imports B; not execution order)"]
    for a in sorted(arrows, key=lambda x: (x.src, x.dst)):
        mark = tag(a.claim.tier)
        out.append(f"  {a.src} {ARROW} {a.dst}  {_count(a.count, 'import edge')} {mark}".rstrip())
        if refs:
            out.append(f"      example import at {fit(_ref(a), MAX_WIDTH - 24)}")
    return out


def _subsystem_body(sub: Subsystem, opened: tuple[str, str] | None, refs: bool) -> list[str]:
    inner = BOX_WIDTH
    body = _purpose_lines(sub.purpose, refs, inner)
    for entry in sub.entries:
        what = "no Typer commands detected" if entry.commands == 0 else _count(entry.commands, "command")
        body += wrap(f"ENTRY  $ {entry.label} -> {entry.module}  ({what})  {_ref(entry.claim)}", inner, "", "       ")
    body.append("CONTAINS")
    expanded = opened is not None and opened[0] == "subsystem" and opened[1] in (sub.name, OPEN_ALL)
    if expanded:
        body += _expanded_parts(sub, opened, inner, refs)
    else:
        body += _part_tokens(sub, opened, inner)
        body += _opened_component(sub, opened, inner, refs)
    return body


def _token(part: Part, opened: tuple[str, str] | None) -> str:
    if opened == ("component", part.name):
        return f"[-] {part.short} ({_count(len(part.modules), 'module')})"
    if part.collapsed:
        return f"[+] {part.short} ({_count(len(part.modules), 'module')})"
    return part.short


def _part_tokens(sub: Subsystem, opened: tuple[str, str] | None, inner: int) -> list[str]:
    shown = sub.parts[:MAX_PARTS_SHOWN]
    tokens = [_token(p, opened) for p in shown]
    if len(sub.parts) > len(shown):
        tokens.append(f"+{len(sub.parts) - len(shown)} more (--open {sub.name})")
    return wrap("  ".join(tokens), inner - 2, "  ", "  ")


def _expanded_parts(sub: Subsystem, opened: tuple[str, str], inner: int, refs: bool) -> list[str]:
    lines: list[str] = []
    for part in sub.parts:
        mark = "  [?]" if part.purpose.tier is Tier.UNKNOWN else ""
        lines += wrap(f"{_token(part, opened)}  {purpose_text(part.purpose)}{mark}", inner - 2, "  ", "      ")
        if refs and part.purpose.evidence:
            lines.append(f"      docstring at {_ref(part.purpose)}")
    return lines


def _opened_component(sub: Subsystem, opened: tuple[str, str] | None, inner: int, refs: bool) -> list[str]:
    part = next((p for p in sub.parts if opened == ("component", p.name)), None)
    if part is None:
        return []
    mark = "  [?]" if part.purpose.tier is Tier.UNKNOWN else ""
    lines = ["", *wrap(f"[-] {part.short}: {purpose_text(part.purpose)}{mark}", inner - 2, "  ", "      ")]
    if refs and part.purpose.evidence:
        lines.append(f"      docstring at {_ref(part.purpose)}")
    names = [m[len(part.name) + 1:] if m != part.name else "__init__" for m in part.modules]
    shown = names[:MAX_MODULES_SHOWN]
    more = f"  +{len(names) - len(shown)} more (all: --view deps --open {part.name})" if len(names) > len(shown) else ""
    return lines + wrap("  ".join(shown) + more, inner - 6, "      ", "      ")


# ------------------------------------------------------------ dependency map

LEGEND_LEVELS = (
    "L0 = top of the import stack. A ---> B = A imports B (n = import edges).",
    "Levels are import depth, NOT execution order. ENTRY = CLI commands here.",
    "no tag = FACT   [I] INFERENCE   [?] UNKNOWN",
)
MAX_PURPOSE_BOX_LINES = 2


def render_deps(view: DepsView, refs: bool) -> list[str]:
    title = "LEVEL MAP" if view.detailed else "DEPENDENCY MAP"
    legend = LEGEND_LEVELS if view.detailed else LEGEND_DEPS
    out = [*wrap(f"{title}  {view.label}  |  scope: {view.scope}", MAX_WIDTH, "", "    "), *legend, ""]
    if view.scope_purpose is not None:
        out += [*_purpose_lines(view.scope_purpose, refs, MAX_WIDTH - 2), ""]
    shown = {n.name: n.shown for layer in view.layers for n in layer}
    for index, layer in enumerate(view.layers):
        out += _layer(index, layer, shown, refs)
    out += _outside(view, refs)
    out += ["UNKNOWN", *_unknown_lines(view.unknowns)]
    if not view.opened and not view.detailed:
        names = ", ".join(sorted({n.name for layer in view.layers for n in layer}))
        out += ["", *wrap(f"Expand: --open NAME   (NAME: {names})", MAX_WIDTH, "", "    ")]
    return [fit_line(line) for line in out]


def fit_line(line: str) -> str:
    return line if len(line) <= MAX_WIDTH else line[: MAX_WIDTH - 3] + "..."


def _layer(index: int, layer: tuple[Node, ...], shown: dict[str, str], refs: bool) -> list[str]:
    width = BOX_WIDTH if refs else HALF_BOX_WIDTH
    boxes = [_node_box(n, shown, refs, width) for n in layer]
    step = 1 if refs else 2
    out: list[str] = []
    for start in range(0, len(boxes), step):
        row = boxes[start:start + step]
        blank = " " * width
        merged = ["  ".join(cells) for cells in zip_longest(*row, fillvalue=blank)]
        gutters = [f"L{index}  " if start == 0 and i == 0 else "    " for i in range(len(merged))]
        out += [(gutter + line).rstrip() for gutter, line in zip(gutters, merged)]
    return out + [""]


def _node_box(node: Node, shown: dict[str, str], refs: bool, width: int) -> list[str]:
    inner = width - 4
    body: list[str] = []
    if node.purpose:
        lines = wrap(node.purpose if node.purpose.startswith("no docstring") else f'"{node.purpose}"', inner)
        if len(lines) > MAX_PURPOSE_BOX_LINES:  # the quote is cut: say so instead of dropping the rest silently
            lines = [*lines[:MAX_PURPOSE_BOX_LINES - 1], clip(f"{lines[MAX_PURPOSE_BOX_LINES - 1]} ...", inner)]
        body += lines
    if node.commands:
        body.append(f"ENTRY  {_count(node.commands, 'command')}")
    if node.cycle:
        others = ", ".join(shown.get(n, n) for n in node.cycle if n != node.name)
        body += wrap(f"CYCLE with: {others}", inner)
    if not node.out:
        body.append("(imports nothing in this scope)")
    elif refs:
        for a in node.out:
            body += wrap(f"{ARROW} {shown.get(a.dst, a.dst)} {a.count} {tag(a.claim.tier)}".rstrip(), inner, "", "     ")
            body.append(f"      import at {fit(_ref(a), inner - 16)}")
    else:
        items = ", ".join(f"{shown.get(a.dst, a.dst)} {a.count}{(' ' + tag(a.claim.tier)) if tag(a.claim.tier) else ''}" for a in node.out)
        body += wrap(f"{ARROW} {items}", inner, "", "     ")
    title = f"{node.shown}  ({_count(node.modules, 'module')})"
    return box(title, body, width)


def _outside(view: DepsView, refs: bool) -> list[str]:
    out: list[str] = []
    for title, arrows in (
        ("IMPORTS FROM OUTSIDE THIS SCOPE  (inside ---> outside)", view.outward),
        ("IMPORTED BY OUTSIDE THIS SCOPE  (outside ---> inside)", view.inward),
    ):
        if not arrows:
            continue
        out += [title]
        ordered = sorted(arrows, key=lambda a: (a.src, a.dst))
        for a in ordered[:MAX_OUTSIDE_LINES]:
            out.append(f"  {a.src} {ARROW} {a.dst}  {_count(a.count, 'import edge')} {tag(a.claim.tier)}".rstrip())
            if refs:
                out.append(f"      example import at {fit(_ref(a), MAX_WIDTH - 24)}")
        if len(ordered) > MAX_OUTSIDE_LINES:
            hidden = ordered[MAX_OUTSIDE_LINES:]
            out.append(f"  +{len(hidden)} more arrows ({sum(a.count for a in hidden)} edges); narrow with --open NAME")
        out.append("")
    return out
