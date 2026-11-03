"""Runs one Aider session through its Python API. Executed by *Aider's* interpreter.

Kept free of argus imports on purpose: Aider lives in its own virtualenv. Reports exact
token and cost totals as JSON, which stdout scraping cannot (Aider rounds to 0.1k).

The result file is rewritten after every model call, so a timeout or crash still leaves the
usage (and therefore the spend) behind.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Failures of the provider, network, or account: the model never got a fair attempt.
_API_FAILURE = re.compile(
    r"authenticat|api key|rate.?limit|\b429\b|\b5\d\d\b|timeout|timed out|connection|overloaded|"
    r"service.?unavailable|insufficient|quota|billing|api provider|internal.?server|litellm\.\w*Error",
    re.IGNORECASE,
)
# Failures that are the agent's: it asked for more than the model can hold.
_AGENT_FAILURE = re.compile(r"context window|token limit|too many tokens|maximum context", re.IGNORECASE)


def split_tool_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    """(api failures, other tool errors)."""
    api: list[str] = []
    other: list[str] = []
    for message in errors:
        if _API_FAILURE.search(message) and not _AGENT_FAILURE.search(message):
            api.append(message)
        else:
            other.append(message)
    return api, other


def _load_aider() -> SimpleNamespace:
    from aider.coders import Coder
    from aider.io import InputOutput
    from aider.models import Model
    from aider.repo import GitRepo

    return SimpleNamespace(Coder=Coder, InputOutput=InputOutput, Model=Model, GitRepo=GitRepo)


def run_session(args: argparse.Namespace, api: Any) -> dict:
    """Run the session and write the result file; returns the result dict."""
    out = Path(args.out)
    result: dict = {
        "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "llm_calls": 0,
        "api_errors": [], "tool_errors": [], "cost_unknown": False, "error": None,
    }
    tool_errors: list[str] = []
    coder = None

    def flush() -> None:
        if coder is not None:
            result["input_tokens"] = int(coder.total_tokens_sent)
            result["output_tokens"] = int(coder.total_tokens_received)
            result["cost"] = float(coder.total_cost)
        result["api_errors"], result["tool_errors"] = split_tool_errors(tool_errors)
        out.write_text(json.dumps(result), encoding="utf-8")

    try:
        io = api.InputOutput(yes=True, pretty=False, fancy_input=False, root=args.repo)
        original_tool_error = io.tool_error

        def record_tool_error(message: str = "", strip: bool = True) -> object:
            tool_errors.append(str(message))
            return original_tool_error(message, strip)

        io.tool_error = record_tool_error
        fnames = [str(Path(args.repo) / f) for f in args.file]
        kwargs = {} if args.map_tokens is None else {"map_tokens": args.map_tokens}
        coder = api.Coder.create(
            main_model=api.Model(args.model), io=io, repo=api.GitRepo(io, fnames, args.repo), fnames=fnames,
            auto_commits=False, dirty_commits=False, stream=False, auto_lint=False,
            suggest_shell_commands=False, detect_urls=False, **kwargs,
        )
        original_send = coder.send_message

        def counted(inp: str) -> object:
            result["llm_calls"] += 1
            try:
                return original_send(inp)
            finally:
                flush()

        coder.send_message = counted
        coder.run(Path(args.message_file).read_text(encoding="utf-8"))
    except BaseException:  # noqa: BLE001 - incl. SystemExit from Aider; reported to the parent
        result["error"] = traceback.format_exc()[-3000:]
    flush()
    if result["error"] is None and result["input_tokens"] > 0 and result["cost"] == 0.0:
        result["cost_unknown"] = True  # model missing from the price table: cost is NOT zero
        flush()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--message-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--map-tokens", type=int, default=None)
    run_session(parser.parse_args(), _load_aider())
    return 0


if __name__ == "__main__":
    sys.exit(main())
