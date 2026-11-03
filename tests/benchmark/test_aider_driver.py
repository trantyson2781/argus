from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).parents[2] / "src/argus/executor/_aider_driver.py"
_spec = importlib.util.spec_from_file_location("aider_driver_under_test", _PATH)
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)


class FakeIO:
    def __init__(self, **_kw) -> None:
        self.messages: list[str] = []

    def tool_error(self, message="", strip=True) -> None:
        self.messages.append(message)


class FakeCoder:
    def __init__(self, script, io, usage) -> None:
        self.io, self._script, self._usage = io, script, usage
        self.total_tokens_sent = self.total_tokens_received = 0
        self.total_cost = 0.0

    def send_message(self, inp):
        step = self._script.pop(0)
        if step == "ok":
            self.total_tokens_sent += self._usage[0]
            self.total_tokens_received += self._usage[1]
            self.total_cost += self._usage[2]
        elif step == "crash":
            raise SystemExit(3)
        else:
            self.io.tool_error(step)

    def run(self, message):
        for _ in range(len(self._script)):
            self.send_message(message)


def _api(script, usage=(1000, 100, 0.01)):
    holder: dict = {}

    class Coder:
        @staticmethod
        def create(**kw):
            holder["coder"] = FakeCoder(list(script), kw["io"], usage)
            return holder["coder"]

    return SimpleNamespace(
        InputOutput=lambda **kw: holder.setdefault("io", FakeIO()),
        Model=lambda name: name, GitRepo=lambda *a: None, Coder=Coder,
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    msg = tmp_path / "m.txt"
    msg.write_text("do it")
    return argparse.Namespace(model="m", repo=str(tmp_path), message_file=str(msg),
                              out=str(tmp_path / "o.json"), file=[], map_tokens=None)


def test_reports_exact_usage_and_call_count(tmp_path: Path) -> None:
    result = driver.run_session(_args(tmp_path), _api(["ok", "ok"]))
    assert (result["input_tokens"], result["output_tokens"], result["llm_calls"]) == (2000, 200, 2)
    assert result["cost"] == pytest.approx(0.02) and result["error"] is None and not result["cost_unknown"]
    assert json.loads((tmp_path / "o.json").read_text())["llm_calls"] == 2


def test_api_failure_after_a_successful_call_is_reported(tmp_path: Path) -> None:
    result = driver.run_session(_args(tmp_path), _api(["ok", "litellm.RateLimitError: 429 too many requests"]))
    assert result["input_tokens"] == 1000 and result["api_errors"]


def test_context_window_overflow_is_not_an_api_failure(tmp_path: Path) -> None:
    result = driver.run_session(_args(tmp_path), _api(["Input exceeds the model's context window limit"]))
    assert result["api_errors"] == [] and result["tool_errors"]


def test_systemexit_inside_aider_is_captured_and_usage_survives(tmp_path: Path) -> None:
    result = driver.run_session(_args(tmp_path), _api(["ok", "crash"]))
    assert result["error"] and "SystemExit" in result["error"] and result["input_tokens"] == 1000


def test_zero_cost_with_tokens_flags_unknown_price(tmp_path: Path) -> None:
    assert driver.run_session(_args(tmp_path), _api(["ok"], usage=(500, 50, 0.0)))["cost_unknown"] is True


def test_split_tool_errors_classification() -> None:
    api, other = driver.split_tool_errors(["Authentication Fails", "Unable to apply edit", "connection reset"])
    assert len(api) == 2 and other == ["Unable to apply edit"]
