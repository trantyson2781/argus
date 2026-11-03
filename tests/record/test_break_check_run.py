"""Break check, runner (`scripts/break_check/run_break_check.py`): safety of the copy, restoring the broken file, and one end-to-end run
on a tiny repository where the answer is known by construction.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from argus.record.break_check import BREAK_MESSAGE, PrecisionRecall

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "break_check" / "run_break_check.py"
NEEDS_312 = pytest.mark.skipif(sys.version_info < (3, 12), reason="argus record needs Python >= 3.12 in the interpreter that runs the tests")


@pytest.fixture(scope="module", name="runner")  # named apart from the parameter so V1 has no fixture-name links to lose (see test_resolve_calls)
def load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_break_check", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_break_check"] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


# --- the fixture repository ----------------------------------------------------------------------------------------------------------

CORE = '''\
def core(x):
    return x + 1


def helper(x):
    return x * 2


def unused(x):
    return x - 1


def risky(x):
    return x / 1
'''
TESTS = '''\
from pkg.core import core, helper, risky


def test_core_one():
    assert core(1) == 2


def test_core_two():
    assert core(2) == 3


def test_core_three():
    assert core(3) == 4


def test_helper():
    assert helper(2) == 4


def test_swallow():
    try:
        risky(1)
    except Exception:
        pass
'''


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    root = tmp_path / "source_repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "core.py").write_text(CORE, encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_core.py").write_text(TESTS, encoding="utf-8")
    return root


# --- ensure_safe_copy_dir ------------------------------------------------------------------------------------------------------------

def test_a_copy_directory_equal_to_the_source_is_refused(runner: ModuleType, source_repo: Path) -> None:
    with pytest.raises(runner.UnsafeCopyError):
        runner.ensure_safe_copy_dir(source_repo, source_repo)


def test_a_copy_directory_inside_the_source_is_refused(runner: ModuleType, source_repo: Path) -> None:
    with pytest.raises(runner.UnsafeCopyError):
        runner.ensure_safe_copy_dir(source_repo, source_repo / "build" / "copy")


def test_a_copy_directory_that_contains_the_source_is_refused(runner: ModuleType, source_repo: Path) -> None:
    with pytest.raises(runner.UnsafeCopyError):
        runner.ensure_safe_copy_dir(source_repo, source_repo.parent)


def test_a_copy_directory_reaching_the_source_through_dotdot_or_a_link_is_refused(runner: ModuleType, source_repo: Path, tmp_path: Path) -> None:
    link = tmp_path / "link_to_source"
    link.symlink_to(source_repo, target_is_directory=True)

    with pytest.raises(runner.UnsafeCopyError):
        runner.ensure_safe_copy_dir(source_repo, source_repo / "pkg" / "..")
    with pytest.raises(runner.UnsafeCopyError):
        runner.ensure_safe_copy_dir(source_repo, link)


def test_a_separate_sibling_directory_is_accepted(runner: ModuleType, source_repo: Path, tmp_path: Path) -> None:
    runner.ensure_safe_copy_dir(source_repo, tmp_path / "copy")  # does not raise


# --- mutated_file --------------------------------------------------------------------------------------------------------------------

def test_the_file_holds_the_raise_inside_the_block_and_is_byte_identical_after_it(runner: ModuleType, tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "m.py"
    path.write_bytes(CORE.encode())
    before = sha(path)

    # Act / Assert
    with runner.mutated_file(path, 1):
        assert BREAK_MESSAGE in path.read_text(encoding="utf-8")
        assert sha(path) != before
    assert sha(path) == before


def test_the_file_is_restored_even_when_the_block_raises(runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "m.py"
    path.write_bytes(CORE.encode())
    before = sha(path)

    with pytest.raises(KeyError):
        with runner.mutated_file(path, 5):
            raise KeyError("the test run blew up")

    assert sha(path) == before


def test_windows_line_endings_and_a_missing_final_newline_come_back_exactly(runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "m.py"
    original = b"def f():\r\n    return 1"
    path.write_bytes(original)

    with runner.mutated_file(path, 1):
        pass

    assert path.read_bytes() == original


def test_a_function_that_cannot_be_broken_leaves_the_file_untouched(runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "m.py"
    path.write_bytes(b"def f(): return 1\n")

    with pytest.raises(ValueError):
        with runner.mutated_file(path, 1):
            pytest.fail("the block must not run when the function cannot be broken")

    assert path.read_bytes() == b"def f(): return 1\n"


# --- run_break_check, end to end -----------------------------------------------------------------------------------------------------

def test_the_runner_refuses_an_unsafe_copy_before_copying_or_running_anything(runner: ModuleType, source_repo: Path) -> None:
    before = tree_hashes(source_repo)

    with pytest.raises(runner.UnsafeCopyError):
        runner.run_break_check(source_repo, source_repo / "copy", per_tier=5, seed=0)

    assert tree_hashes(source_repo) == before  # nothing written into the source, not even the refused copy directory


@NEEDS_312
def test_end_to_end_failing_sets_and_precision_recall_match_the_hand_computed_values(runner: ModuleType, source_repo: Path, tmp_path: Path) -> None:
    # Arrange: by construction
    #   core   <- test_core_one, test_core_two, test_core_three : breaking it fails exactly those three
    #   helper <- test_helper                                    : breaking it fails exactly that one
    #   unused <- no test                                        : breaking it fails nothing, and the recording lists no test
    #   risky  <- test_swallow, which catches Exception          : the recording says it was reached, but breaking it fails nothing
    t = "tests/test_core.py::"
    before = tree_hashes(source_repo)

    # Act
    report = runner.run_break_check(source_repo, tmp_path / "copy", per_tier=10, seed=0)

    # Assert
    by_name = {r.candidate.qualname: r for r in report.results}
    assert set(by_name) == {"pkg.core.core", "pkg.core.helper", "pkg.core.unused", "pkg.core.risky"}
    core = by_name["pkg.core.core"]
    assert core.reached == core.failed == {t + "test_core_one", t + "test_core_two", t + "test_core_three"}
    helper = by_name["pkg.core.helper"]
    assert helper.reached == helper.failed == {t + "test_helper"}
    unused = by_name["pkg.core.unused"]
    assert unused.reached == frozenset() and unused.failed == frozenset() and not unused.break_not_noticed
    risky = by_name["pkg.core.risky"]
    assert risky.reached == {t + "test_swallow"}
    assert risky.failed == frozenset()
    assert risky.break_not_noticed is True  # reached by a test, yet breaking it fails none
    assert not core.break_not_noticed and not helper.break_not_noticed
    assert not any(r.collection_error for r in report.results)
    assert report.flaky == frozenset()
    # pooled: hits 3 + 1 + 0 + 0 = 4; sum |R| = 3 + 1 + 0 + 1 = 5; sum |F| = 3 + 1 + 0 + 0 = 4 -> precision 0.8, recall 1.0
    assert report.pooled.precision == pytest.approx(0.8) and report.pooled.recall == pytest.approx(1.0)
    assert isinstance(report.pooled, PrecisionRecall)
    assert report.tau is None and report.p_value is None  # all four have radius 0 (tests are not production callers), so no rank order exists
    assert tree_hashes(source_repo) == before  # the source is never touched


@NEEDS_312
def test_end_to_end_the_copy_is_left_clean_after_every_function_was_broken(runner: ModuleType, source_repo: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"

    runner.run_break_check(source_repo, copy, per_tier=10, seed=0)

    assert sha(copy / "pkg" / "core.py") == sha(source_repo / "pkg" / "core.py")  # byte-identical to the original
    assert BREAK_MESSAGE not in (copy / "pkg" / "core.py").read_text(encoding="utf-8")


@NEEDS_312
def test_end_to_end_a_flaky_test_is_excluded_from_every_count(runner: ModuleType, source_repo: Path, tmp_path: Path) -> None:
    # Arrange: a test that fails by itself in the clean run (a stand-in for flaky: it fails in both clean runs)
    flaky_test = source_repo / "tests" / "test_flaky.py"
    flaky_test.write_text("from pkg.core import core\n\n\ndef test_always_red():\n    core(1)\n    assert False\n", encoding="utf-8")

    # Act
    report = runner.run_break_check(source_repo, tmp_path / "copy", per_tier=10, seed=0)

    # Assert
    assert report.flaky == {"tests/test_flaky.py::test_always_red"}
    for result in report.results:
        assert "tests/test_flaky.py::test_always_red" not in result.reached | result.failed
    core = next(r for r in report.results if r.candidate.qualname == "pkg.core.core")
    assert len(core.failed) == 3



# --- audit fixes (2026-10-24) --------------------------------------------------------------------------------------------------------

import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from argus.record.break_check import Candidate  # noqa: E402


@pytest.mark.parametrize(
    ("code", "out", "readable", "tests"),
    [
        (0, "....\n4 passed\n", True, set()),
        (1, "FAILED tests/test_a.py::test_x - boom\n", True, {"tests/test_a.py::test_x"}),
        (1, "1 failed\n", False, set()),  # exit code 1 but no test named: the summary is cut or in another format
        (0, "FAILED tests/test_a.py::test_x\n", False, set()),  # exit code 0 cannot come with a failure
        (2, "", False, set()),  # interrupted
        (4, "", False, set()),  # usage error
        (5, "", False, set()),  # no tests collected
        (1, "ERROR tests/test_c.py - ImportError\n", False, set()),  # collection error
    ],
)
def test_a_pytest_run_is_readable_only_when_its_exit_code_and_its_summary_agree(
    runner: ModuleType, code: int, out: str, readable: bool, tests: set[str],
) -> None:
    result = runner.parse_run(code, out)

    assert (result is not None) is readable
    if result is not None:
        assert result.tests == tests


def test_the_copy_holds_regular_files_where_the_source_has_links_so_breaking_it_cannot_reach_outside(runner: ModuleType, tmp_path: Path) -> None:
    # Arrange
    outside = tmp_path / "outside.py"
    outside.write_text("def out():\n    return 1\n", encoding="utf-8")
    source = tmp_path / "src"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "linked.py").symlink_to(outside)
    (source / "pkg" / "dangling.py").symlink_to(tmp_path / "missing.py")
    (source / ".venv").mkdir()
    (source / ".venv" / "x.py").write_text("x = 1\n", encoding="utf-8")

    # Act
    runner.copy_repo(source, tmp_path / "copy")

    # Assert
    linked = tmp_path / "copy" / "pkg" / "linked.py"
    assert linked.is_file() and not linked.is_symlink()
    assert not (tmp_path / "copy" / ".venv").exists()
    with runner.mutated_file(linked, 1):
        pass
    assert outside.read_text(encoding="utf-8") == "def out():\n    return 1\n"


def test_a_file_that_is_a_link_is_never_broken(runner: ModuleType, tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_bytes(b"def f():\n    return 1\n")
    link = tmp_path / "link.py"
    link.symlink_to(target)

    with pytest.raises(ValueError):
        with runner.mutated_file(link, 1):
            pytest.fail("the block must not run")

    assert target.read_bytes() == b"def f():\n    return 1\n"


def test_a_file_with_a_byte_order_mark_is_broken_and_restored_exactly(runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "m.py"
    original = b"\xef\xbb\xbfdef f():\n    return 1\n"
    path.write_bytes(original)

    with runner.mutated_file(path, 1):
        assert BREAK_MESSAGE in path.read_text(encoding="utf-8")

    assert path.read_bytes() == original


def test_a_copy_directory_that_is_an_existing_file_is_refused(runner: ModuleType, source_repo: Path, tmp_path: Path) -> None:
    target = tmp_path / "afile"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(runner.UnsafeCopyError):
        runner.ensure_safe_copy_dir(source_repo, target)


def test_a_timed_out_run_takes_its_child_processes_with_it(runner: ModuleType, tmp_path: Path) -> None:
    # Arrange: a stand-in for python that starts a long sleep in the background and waits for it
    fake = tmp_path / "fake_python"
    fake.write_text('#!/bin/bash\nsleep 60 &\necho $! > grandchild.pid\nwait\n', encoding="utf-8")
    fake.chmod(0o755)

    # Act
    done, _ = runner.run_pytest(str(fake), tmp_path, timeout=1)

    # Assert
    assert done is None
    pid = int((tmp_path / "grandchild.pid").read_text(encoding="utf-8"))
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail("the background process of the timed-out run is still alive")


def test_a_function_that_cannot_be_broken_is_reported_unreadable_and_does_not_stop_the_run(runner: ModuleType, source_repo: Path, tmp_path: Path) -> None:
    # Arrange: a function whose body starts on the line of its signature but runs on to a second line, so it is not a "one-line" function
    core = source_repo / "pkg" / "core.py"
    core.write_text(core.read_text(encoding="utf-8") + "\n\ndef squeeze(x): return (x,\n                   x)\n", encoding="utf-8")

    # Act
    report = runner.run_break_check(source_repo, tmp_path / "copy", per_tier=10, seed=0)

    # Assert
    by_name = {r.candidate.qualname: r for r in report.results}
    assert by_name["pkg.core.squeeze"].collection_error is True
    assert by_name["pkg.core.core"].collection_error is False
    assert len(by_name["pkg.core.core"].failed) == 3


def test_a_recording_of_a_crashed_run_stops_the_check(runner: ModuleType, source_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = SimpleNamespace(recording=SimpleNamespace(exit_code=2, processes=SimpleNamespace(errors=0, saved=1), functions=()))
    monkeypatch.setattr(runner, "record_command", lambda root, argv: bad)

    with pytest.raises(RuntimeError, match="recording"):
        runner.run_break_check(source_repo, tmp_path / "copy", per_tier=5, seed=0)


# --- the report ----------------------------------------------------------------------------------------------------------------------

def cand(name: str, tier: str, radius: int = 0) -> Candidate:
    return Candidate(f"function:m.{name}", f"m.{name}", "m.py", 1, 3, tier, radius)  # type: ignore[arg-type]


def result(runner: ModuleType, name: str, tier: str, reached: set[str], failed: set[str], unreadable: bool = False):  # noqa: ANN202
    return runner.FunctionResult(cand(name, tier), frozenset(reached), frozenset(failed), unreadable)


def report_of(runner: ModuleType, *results, tau=None, p=None, pooled=PrecisionRecall(None, None)):  # noqa: ANN002, ANN202
    return runner.BreakReport(tuple(results), frozenset(), tau, p, pooled)


@pytest.mark.parametrize(
    ("tau", "p", "word"),
    [(None, None, "UNDECIDED"), (-0.2, 0.01, "STOP"), (0.0, 0.5, "STOP"), (0.1, 0.01, "FIX"), (0.35, 0.01, "CONTINUE"), (0.35, 0.2, "UNCLEAR")],
)
def test_the_tier_verdict_follows_the_thresholds_written_down_in_advance(runner: ModuleType, tau: float | None, p: float | None, word: str) -> None:
    lines = runner.verdicts(report_of(runner, tau=tau, p=p))

    assert lines[0].startswith(f"H1 {word}")


@pytest.mark.parametrize(
    ("pooled", "word"),
    [
        (PrecisionRecall(None, 1.0), "UNDECIDED"),
        (PrecisionRecall(0.85, 0.95), "CONTINUE"),
        (PrecisionRecall(0.85, 0.80), "FIX"),
        (PrecisionRecall(0.40, 0.95), "RENAME"),
    ],
)
def test_the_run_by_verdict_follows_the_thresholds_written_down_in_advance(runner: ModuleType, pooled: PrecisionRecall, word: str) -> None:
    lines = runner.verdicts(report_of(runner, pooled=pooled))

    assert lines[1].startswith(f"H2 {word}")


def test_the_report_names_the_important_functions_no_test_reaches_and_counts_unreadable_runs_per_tier(runner: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    report = report_of(
        runner,
        result(runner, "busy", "HIGH", {"t::a"}, {"t::a"}),
        result(runner, "orphan", "HIGH", set(), set()),
        result(runner, "swallowed", "MED", {"t::b"}, set()),
        result(runner, "broken_import", "HIGH", set(), set(), unreadable=True),
    )

    runner.print_report(report)
    out = capsys.readouterr().out

    assert "Important but untested" in out and "m.orphan" in out
    assert "m.busy" not in out.split("Important but untested")[1].split("\n")[0]
    assert "Unreadable runs by tier: HIGH 1" in out


def test_a_report_with_no_readable_run_is_a_failed_run(runner: ModuleType) -> None:
    assert runner.exit_code_for(report_of(runner)) == 2
    assert runner.exit_code_for(report_of(runner, result(runner, "x", "LOW", set(), set(), unreadable=True))) == 2
    assert runner.exit_code_for(report_of(runner, result(runner, "x", "LOW", set(), set()))) == 0


# --- what the report says about the repository -----------------------------------------------------------------------------------------

def test_the_source_description_says_so_when_there_is_no_git_repository(runner: ModuleType, tmp_path: Path) -> None:
    assert any("not a git repository" in line for line in runner.describe_source(tmp_path))


def test_the_source_description_names_the_commit_and_every_uncommitted_file(runner: ModuleType, tmp_path: Path) -> None:
    # Arrange
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, env=env, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "first")
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")

    # Act
    text = "\n".join(runner.describe_source(tmp_path))

    # Assert
    assert "a.py" in text and "uncommitted" in text
    assert len(subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()) >= 7


# --- from the test-quality audit -----------------------------------------------------------------------------------------------------

@NEEDS_312
def test_the_check_stops_when_no_function_of_the_copy_ran(runner: ModuleType, source_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = SimpleNamespace(recording=SimpleNamespace(exit_code=0, processes=SimpleNamespace(errors=0, saved=1), functions=()))
    monkeypatch.setattr(runner, "record_command", lambda root, argv: empty)

    with pytest.raises(runner.CanaryError):
        runner.run_break_check(source_repo, tmp_path / "copy", per_tier=5, seed=0)


def test_an_unreadable_run_is_left_out_of_tau_and_of_precision_and_recall(runner: ModuleType) -> None:
    # Arrange: tiers [LOW, LOW, HIGH, HIGH] with 0, 0, 1, 2 failing tests: C = 4, D = 0, ties 2 and 1, tau-b = 4 / sqrt(4 * 5)
    readable = [
        result(runner, "l1", "LOW", set(), set()), result(runner, "l2", "LOW", set(), set()),
        result(runner, "h1", "HIGH", {"t::a"}, {"t::a"}), result(runner, "h2", "HIGH", {"t::b", "t::c"}, {"t::b", "t::c"}),
    ]
    noise = result(runner, "h3", "HIGH", {"t::d"}, {"t::x", "t::y", "t::z"}, unreadable=True)  # would change every number if it were counted

    # Act
    report = runner._report((*readable, noise), frozenset(), seed=0)

    # Assert
    assert report.tau == pytest.approx(4 / (20 ** 0.5))
    assert report.pooled == PrecisionRecall(1.0, 1.0)
