"""Shared helpers for the `--json` tests of the explore commands."""

from __future__ import annotations

import json
from typing import Any

from typer.testing import CliRunner

from argus.cli import app
from argus.explore.claims import Claim

runner = CliRunner()


def claims_in(node: Any) -> list[dict[str, Any]]:
    """Every claim object anywhere in a JSON document (an object with a tier and evidence)."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "tier" in node and "evidence" in node:
            found.append(node)
        for value in node.values():
            found.extend(claims_in(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(claims_in(value))
    return found


def json_keys(doc_claims: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted((c["tier"], c["text"], tuple((e["file"], e["line"], e["source_hash"]) for e in c["evidence"])) for c in doc_claims)


def view_keys(claims: list[Claim]) -> list[tuple[Any, ...]]:
    return sorted((c.tier.value, c.text, tuple((e.file, e.line, e.source_hash) for e in c.evidence)) for c in claims)


def run_json(*args: str, env: dict[str, str] | None = None):
    return runner.invoke(app, ["explore", *args, "--json"], env=env)


def doc_of(*args: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = run_json(*args, env=env)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)
