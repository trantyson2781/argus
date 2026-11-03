"""Text for `argus explore sample`: the AMBIGUOUS gate verdict and a reproducible sample of edges to check by hand."""

from __future__ import annotations

from argus.explore.render_diff import printable
from argus.model.schema import ArchModel
from argus.report import AMBIGUOUS_RATE_GATE, EXTRACTED_PRECISION_GATE, ambiguous_rate, sample_edges, structural_edges


def sample_report(model: ArchModel, count: int, seed: int) -> tuple[list[str], bool]:
    """(lines to print, whether the AMBIGUOUS gate passed). Repository names are escaped: they are not ours to trust."""
    structural = len(structural_edges(model))
    stats = model.stats
    lines = [f"Edge quality gates   {stats.files_scanned} files scanned, {stats.files_failed} failed to parse, {structural} structural edges"]
    if not structural:
        return [*lines, "no structural edges found: nothing to measure or sample"], True
    rate = ambiguous_rate(model)
    passed = rate <= AMBIGUOUS_RATE_GATE
    lines += [
        f"AMBIGUOUS rate   {rate:.1%}   limit {AMBIGUOUS_RATE_GATE:.0%}   {'PASS' if passed else 'FAIL'}",
        f"Precision of EXTRACTED edges   Argus cannot measure this: check the edges below by hand (open the site, read the line) and count the wrong ones; "
        f"wrong / checked must stay under {1 - EXTRACTED_PRECISION_GATE:.0%}. Only EXTRACTED rows count toward this gate.",
        f"Sample   {min(count, structural)} of {structural} edges, seed {seed}",
    ]
    for number, edge in enumerate(sample_edges(model, count, seed), start=1):
        flags = f"  [{', '.join(edge.flags)}]" if edge.flags else ""
        lines.append(f"{number:>4}  {edge.kind:<18} {edge.confidence:<10} {edge.source} -> {edge.target}   {edge.site}{flags}")
    return [printable(line) for line in lines], passed
