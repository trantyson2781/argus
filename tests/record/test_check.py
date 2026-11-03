"""Breakage check, part A (`argus check`): pick the tests that reach a change, run them on throwaway copies, compare with the base.

Written first (RED). The pure rules (classification, test ids, selection) are tested without running anything; the end-to-end tests run
pytest in real throwaway repositories where the answer is known by construction.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus.cli import app
from argus.extract import scan
from argus.record.check import CheckError, Outcome, Selected, check_change, classify, node_id, result_lines, select_tests
from argus.record.runner import scrubbed_env
from argus.record.store import ObservedFunction, ObservedTest, Processes, Recording
from tests.explore.diff_helpers import GitRepo
from tests.explore.test_diff_view import build_view

# -- the rules, without running anything --------------------------------------


@pytest.mark.parametrize(
    ("head1", "head2", "base1", "base2", "expected"),
    [
        (True, True, False, False, Outcome.NEWLY_FAILING),
        (True, True, True, True, Outcome.ALREADY_FAILING),
        (True, False, False, False, Outcome.FLAKY),  # failed once on head, then passed
        (True, True, True, False, Outcome.FLAKY),  # failed on both sides once
        (True, True, False, True, Outcome.FLAKY),
        (True, False, True, False, Outcome.FLAKY),
    ],
)
def test_classify_truth_table(head1, head2, base1, base2, expected):
    assert classify(head1, head2, base1, base2) is expected


def test_node_id_turns_a_dotted_qualname_into_a_pytest_node_id():
    assert node_id("tests/test_a.py::TestCart.test_add") == "tests/test_a.py::TestCart::test_add"
    assert node_id("tests/test_a.py::test_add") == "tests/test_a.py::test_add"


def test_scrubbed_env_drops_keys_and_replaces_home(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("HOME", "/home/real")

    env = scrubbed_env(tmp_path)

    assert env["HOME"] == str(tmp_path)
    assert "ANTHROPIC_API_KEY" not in env and "AWS_SECRET_ACCESS_KEY" not in env
    assert "PATH" in env


# -- fixtures ------------------------------------------------------------------

BILLING = "def charge(amount):\n    return amount * 2\n\n\ndef fee(amount):\n    return amount\n"
CART = "from shop.billing import charge\n\n\ndef checkout(amount):\n    return charge(amount)\n"
TESTS = (
    "from shop.billing import charge, fee\n"
    "from shop.cart import checkout\n\n\n"
    "def test_charge():\n    assert charge(2) == 4\n\n\n"
    "def test_checkout():\n    assert checkout(2) == 4\n\n\n"
    "def test_fee():\n    assert fee(3) == 3\n"
)
BASE_FILES = {"conftest.py": "", "shop/__init__.py": "", "shop/billing.py": BILLING, "shop/cart.py": CART, "tests/test_shop.py": TESTS}


def make_repo(tmp_path: Path, files: dict[str, str] | None = None) -> GitRepo:
    repo = GitRepo(tmp_path / "repo")
    repo.write_files(files or BASE_FILES)
    repo.commit("base")
    return repo


def tree_hash(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
    }


def sources(selection) -> dict[str, frozenset[str]]:
    return {s.test: s.sources for s in selection.tests}


def select(repo: GitRepo, rec=None):
    view = build_view(repo)
    return select_tests(view, scan(repo.root), rec, repo.root)


# -- selection -----------------------------------------------------------------


def test_tests_that_call_the_changed_function_are_selected_as_direct(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    chosen = sources(select(repo))

    assert chosen["tests/test_shop.py::test_charge"] == {"direct"}


def test_tests_that_call_a_caller_of_the_changed_function_are_selected_as_caller(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    assert sources(select(repo))["tests/test_shop.py::test_checkout"] == {"caller"}


def test_a_test_that_reaches_nothing_changed_is_not_selected(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    assert "tests/test_shop.py::test_fee" not in sources(select(repo))


def test_nothing_is_selected_when_no_function_changed_its_behaviour(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING + "# note\n")

    assert select(repo).tests == ()


def recording_for(repo: GitRepo, node: str, test_file: str, test_name: str) -> Recording:
    fn = ObservedFunction("shop/billing.py:4:fee", "shop/billing.py", 4, "fee", node, None, (), 1)
    test = ObservedFunction(f"{test_file}:9:{test_name}", test_file, 9, test_name, f"function:tests.test_shop.{test_name}", None, (), 1)
    from argus.explore.claims import file_hash

    files = tuple((f, file_hash(repo.root, f)) for f in ("shop/billing.py", test_file))
    return Recording("0", "3.13", ("pytest",), "2026-10-25T00:00:00Z", 1.0, 0, files, (fn, test), (), (), (ObservedTest(test.key, (fn.key,)),), Processes(1, 0))


def test_tests_the_recording_saw_reach_the_function_are_selected_as_observed(tmp_path):
    repo = make_repo(tmp_path)
    rec = recording_for(repo, "function:shop.billing.fee", "tests/test_shop.py", "test_fee")
    repo.write("shop/billing.py", BILLING.replace("return amount\n", "return amount + 0\n"))

    chosen = sources(select(repo, rec))

    assert chosen["tests/test_shop.py::test_fee"] >= {"observed"}


def test_a_recording_whose_test_file_changed_since_is_marked_stale(tmp_path):
    repo = make_repo(tmp_path)
    rec = recording_for(repo, "function:shop.billing.fee", "tests/test_shop.py", "test_fee")
    repo.write("shop/billing.py", BILLING.replace("return amount\n", "return amount + 0\n"))
    repo.write("tests/test_shop.py", TESTS + "\n# edited\n")

    assert select(repo, rec).stale == ("tests/test_shop.py",)


# -- running ---------------------------------------------------------------------


def run_check(repo: GitRepo, **kwargs):
    return check_change(repo.root, "HEAD", python=sys.executable, timeout=120, **kwargs)


def run_check_with(repo: GitRepo, rec: Recording, monkeypatch: pytest.MonkeyPatch):
    from argus.record import check as module

    monkeypatch.setattr(module, "latest_recording", lambda root: (rec, None))
    return check_change(repo.root, "HEAD", python=sys.executable, timeout=120)


def test_a_change_that_breaks_a_test_is_newly_failing(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    result = run_check(repo)

    assert set(result.newly_failing) == {"tests/test_shop.py::test_charge", "tests/test_shop.py::test_checkout"}
    assert result.already_failing == () and result.flaky == ()
    assert result.exit_code == 1


def test_a_change_no_test_notices_runs_clean(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount + amount"))

    result = run_check(repo)

    assert result.ran == 2 and result.newly_failing == () and result.exit_code == 0


def test_no_test_found_is_said_as_a_lower_bound_not_as_safe(tmp_path):
    repo = make_repo(tmp_path, {**BASE_FILES, "shop/lonely.py": "def lonely(x):\n    return x\n"})
    repo.write("shop/lonely.py", "def lonely(x):\n    return x + 1\n")

    result = run_check(repo)
    text = "\n".join(result_lines(result))

    assert result.ran == 0 and result.exit_code == 3  # nothing ran: not a green result
    assert "no test found" in text and "safe" not in text.lower()


def test_a_test_that_already_failed_before_the_change_is_not_blamed(tmp_path):
    files = {**BASE_FILES, "tests/test_shop.py": TESTS.replace("assert charge(2) == 4", "assert charge(2) == 999")}
    repo = make_repo(tmp_path, files)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    result = run_check(repo)

    assert "tests/test_shop.py::test_charge" in result.already_failing
    assert "tests/test_shop.py::test_charge" not in result.newly_failing
    assert "tests/test_shop.py::test_checkout" in result.newly_failing


def test_a_test_that_fails_once_and_then_passes_is_flaky_not_newly_failing(tmp_path):
    flaky = (
        "import os\nfrom shop.billing import charge\n\n\n"
        "def test_charge():\n"
        "    marker = os.path.join(os.path.dirname(__file__), '.seen')\n"
        "    first = not os.path.exists(marker)\n    open(marker, 'w').close()\n"
        "    assert not first\n    assert charge(2) is not None\n"
    )
    repo = make_repo(tmp_path, {**BASE_FILES, "tests/test_shop.py": flaky})
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    result = run_check(repo)

    assert result.flaky == ("tests/test_shop.py::test_charge",) and result.newly_failing == ()
    assert result.exit_code == 0


def test_the_repository_is_never_written_to(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))
    before = tree_hash(repo.root)

    run_check(repo)

    assert tree_hash(repo.root) == before


def test_tests_do_not_see_the_users_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGUS_CHECK_SECRET", "hunter2")
    leaky = TESTS.replace("def test_charge():\n    assert charge(2) == 4", "def test_charge():\n    import os\n    assert 'ARGUS_CHECK_SECRET' not in os.environ\n    assert charge(2) == 4")
    repo = make_repo(tmp_path, {**BASE_FILES, "tests/test_shop.py": leaky})
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount + amount"))

    result = run_check(repo)

    assert result.ran == 2 and result.newly_failing == () and result.already_failing == ()


def test_a_test_the_recording_names_but_the_tree_no_longer_has_is_counted_not_a_crash(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    rec = recording_for(repo, "function:shop.billing.fee", "tests/test_shop.py", "test_fee")
    repo.write("shop/billing.py", BILLING.replace("return amount\n", "return amount + 0\n"))
    repo.delete("tests/test_shop.py")

    selection = select(repo, rec)
    result = run_check_with(repo, rec, monkeypatch)

    assert [s.test for s in selection.tests] == ["tests/test_shop.py::test_fee"]
    assert result.exit_code == 3 and result.ran == 0
    assert result.missing == ("tests/test_shop.py::test_fee",)


def test_an_unknown_base_ref_is_an_error(tmp_path):
    repo = make_repo(tmp_path)

    with pytest.raises(CheckError):
        check_change(repo.root, "no-such-ref", python=sys.executable, timeout=60)


def test_a_hung_test_run_is_reported_unreadable_not_as_a_pass(tmp_path):
    hang = "import time\nfrom shop.billing import charge\n\n\ndef test_charge():\n    time.sleep(60)\n    assert charge(2)\n"
    repo = make_repo(tmp_path, {**BASE_FILES, "tests/test_shop.py": hang})
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    result = check_change(repo.root, "HEAD", python=sys.executable, timeout=3)

    assert result.unreadable and result.exit_code == 2


# -- the command -------------------------------------------------------------------


def invoke(repo: GitRepo, *args: str):
    return CliRunner().invoke(app, ["check", str(repo.root), "--python", sys.executable, *args])


def test_command_prints_newly_failing_tests_and_exits_1(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    out = invoke(repo)

    assert out.exit_code == 1
    assert "newly failing" in out.output and "tests/test_shop.py::test_charge" in out.output
    assert "safe" not in out.output.lower()


def test_command_says_it_runs_your_code_before_it_does(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount + amount"))

    assert "executes your code" in invoke(repo).output.lower()


def test_command_exits_2_on_a_bad_ref(tmp_path):
    repo = make_repo(tmp_path)

    out = invoke(repo, "--base", "no-such-ref")

    assert out.exit_code == 2 and "Error" in out.output


def test_command_exits_3_when_nothing_changed_so_no_test_ran(tmp_path):
    repo = make_repo(tmp_path)

    out = invoke(repo)

    assert out.exit_code == 3 and "no test found" in out.output
    assert os.path.exists(repo.root / "shop")


def test_command_exits_2_not_with_a_traceback_when_the_python_does_not_exist(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    out = invoke(repo, "--python", "/no/such/python")

    assert out.exit_code == 2 and "Traceback" not in out.output and "could not be started" in out.output


# -- audit round: what "ran" and "passed" mean ------------------------------------------


def test_tests_that_were_all_skipped_are_not_reported_as_passed(tmp_path):
    skipped = TESTS.replace("def test_charge", "@pytest.mark.skip\ndef test_charge").replace("def test_checkout", "@pytest.mark.skip\ndef test_checkout")
    repo = make_repo(tmp_path, {**BASE_FILES, "tests/test_shop.py": "import pytest\n" + skipped})
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount + amount"))

    result = run_check(repo)
    text = "\n".join(result_lines(result))

    assert result.ran == 0 and result.skipped == 2 and result.exit_code == 3
    assert "all passed" not in text and "skipped" in text


def test_an_unreadable_recording_is_said_not_ignored(tmp_path, monkeypatch):
    from argus.record import check as module

    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount + amount"))
    monkeypatch.setattr(module, "latest_recording", lambda root: (None, "the latest recording cannot be read (damaged)"))

    text = "\n".join(result_lines(check_change(repo.root, "HEAD", python=sys.executable, timeout=120)))

    assert "cannot be read (damaged)" in text


def test_no_recording_is_said_so_the_observed_source_is_not_assumed(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount + amount"))

    text = "\n".join(result_lines(run_check(repo)))

    assert "no recording" in text


def test_results_carry_the_observed_label(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    assert "newly failing [O]" in "\n".join(result_lines(run_check(repo)))


# -- audit round: tests the base does not have, tests pytest cannot select --------------

OLD_TEST = "from shop.billing import charge\n\n\ndef test_old():\n    assert charge(1) == 2\n"


def test_a_new_failing_test_in_a_modified_file_is_reported_and_the_old_one_still_blamed(tmp_path):
    repo = make_repo(tmp_path, {**BASE_FILES, "tests/test_shop.py": OLD_TEST})
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))
    repo.write("tests/test_shop.py", OLD_TEST + "\n\ndef test_new():\n    assert charge(1) == 2\n")

    result = run_check(repo)

    assert "tests/test_shop.py::test_old" in result.newly_failing
    assert result.new_tests_failing == ("tests/test_shop.py::test_new",)
    assert result.unreadable is None and result.exit_code == 1


def test_a_test_pytest_cannot_select_is_dropped_and_named_not_fatal(tmp_path):
    odd = OLD_TEST + "\n\nclass Helper:\n    def test_meth(self):\n        assert charge(1) == 2\n"  # not a Test* class: pytest does not collect it
    repo = make_repo(tmp_path, {**BASE_FILES, "tests/test_shop.py": odd})
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    result = run_check(repo)

    assert result.unreadable is None
    assert "tests/test_shop.py::test_old" in result.newly_failing
    assert result.uncollected == ("tests/test_shop.py::Helper.test_meth",)


def test_a_change_that_stops_a_test_file_from_loading_is_a_failure_not_undecided(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", "def charge(amount):\n    return amount * 2\n")  # fee is gone, tests/test_shop.py imports it

    result = run_check(repo)

    assert result.broken_files == ("tests/test_shop.py",)
    assert result.unreadable is None and result.exit_code == 1


# -- audit round: the copy must be what is tested ----------------------------------------


def test_code_imported_from_outside_the_copy_makes_the_result_not_decided(tmp_path):
    src_files = {
        "conftest.py": "", "src/shop/__init__.py": "", "src/shop/billing.py": BILLING, "src/shop/cart.py": CART,
        "tests/test_shop.py": TESTS,
    }
    repo = make_repo(tmp_path, src_files)
    wrapper = tmp_path / "py"
    wrapper.write_text(f"#!/bin/sh\nexport PYTHONPATH={repo.root}/src\nexec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)
    repo.write("src/shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    result = check_change(repo.root, "HEAD", python=str(wrapper), timeout=120)

    assert result.unreadable and "outside the copy" in result.unreadable and result.exit_code == 2
    assert result.newly_failing == ()


# -- audit round: part B and part A on one change ----------------------------------------


def test_one_change_is_flagged_by_the_call_fit_and_blamed_by_the_check(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("def charge(amount):", "def charge(amount, currency):"))
    before = tree_hash(repo.root)
    view = build_view(repo)

    result = run_check(repo)

    flagged = [(f.caller, f.line) for f in view.call_fits if f.verdict.value == "BREAKS"]
    assert ("shop.cart.checkout", 5) in flagged
    assert "tests/test_shop.py::test_checkout" in result.newly_failing and result.exit_code == 1
    assert tree_hash(repo.root) == before


# -- audit round: reproductions the test review asked for ------------------------------


def test_extract_commit_refuses_an_unknown_ref(tmp_path):
    from argus.explore.sides import DiffError, extract_commit

    repo = make_repo(tmp_path)

    with pytest.raises(DiffError):
        extract_commit(repo.root, "no-such-ref", tmp_path / "out")


def test_extract_commit_refuses_a_member_that_would_escape_the_destination(tmp_path, monkeypatch):
    import io
    import tarfile

    from argus.explore import sides
    from argus.explore.sides import DiffError, extract_commit

    repo = make_repo(tmp_path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        member = tarfile.TarInfo("../escape.txt")
        member.size = 3
        tar.addfile(member, io.BytesIO(b"bad"))
    monkeypatch.setattr(sides, "_git", lambda path, *args, **kw: b"deadbeef\n" if args[0] == "rev-parse" else buffer.getvalue())
    dest = tmp_path / "out"
    dest.mkdir()

    with pytest.raises(DiffError):
        extract_commit(repo.root, "HEAD", dest)

    assert not (tmp_path / "escape.txt").exists()


def test_extract_commit_returns_the_tracked_names_and_the_commit_wins_over_the_copy(tmp_path):
    from argus.explore.sides import extract_commit

    repo = make_repo(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "shop").mkdir()
    (dest / "shop" / "billing.py").write_text("changed", encoding="utf-8")

    names = extract_commit(repo.root, "HEAD", dest)

    assert "shop/billing.py" in names and (dest / "shop" / "billing.py").read_text(encoding="utf-8") == BILLING


def test_a_new_source_file_is_not_part_of_the_base(tmp_path, monkeypatch):
    from argus.record import check as module

    seen: set[tuple[str, bool]] = set()
    original = module._collect

    def spy(python, root, files, timeout, env):
        seen.add((root.name, (root / "shop" / "tax.py").exists()))
        return original(python, root, files, timeout, env)

    monkeypatch.setattr(module, "_collect", spy)
    repo = make_repo(tmp_path)
    repo.write("shop/tax.py", "def tax(x):\n    return x\n")  # untracked, new
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount + amount"))

    run_check(repo)

    assert seen == {("head", True), ("base", False)}


def test_a_parameter_with_a_dot_in_its_id_is_still_run(tmp_path):
    param = (
        "import pytest\nfrom shop.billing import charge\n\n\n"
        "@pytest.mark.parametrize('x', [1.5, 2])\n"
        "def test_param(x):\n    assert charge(x) == x * 2\n"
    )
    repo = make_repo(tmp_path, {**BASE_FILES, "tests/test_shop.py": param})
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    result = run_check(repo)

    assert result.newly_failing == ("tests/test_shop.py::test_param",) and result.unreadable is None


def test_a_base_other_than_head_is_compared(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))
    repo.commit("break it")

    result = check_change(repo.root, "HEAD~1", python=sys.executable, timeout=120)

    assert "tests/test_shop.py::test_charge" in result.newly_failing and result.exit_code == 1
    assert check_change(repo.root, "HEAD", python=sys.executable, timeout=120).exit_code == 3  # nothing changed since HEAD: nothing ran


def test_a_test_method_in_a_test_class_is_run(tmp_path):
    cls = "from shop.billing import charge\n\n\nclass TestCart:\n    def test_add(self):\n        assert charge(2) == 4\n"
    repo = make_repo(tmp_path, {**BASE_FILES, "tests/test_shop.py": cls})
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    assert run_check(repo).newly_failing == ("tests/test_shop.py::TestCart.test_add",)


def test_a_tracked_module_deleted_from_the_working_tree_breaks_what_imports_it(tmp_path):
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))  # something changed, so tests are chosen
    repo.delete("shop/cart.py")

    result = run_check(repo)

    assert result.exit_code == 1 and (result.broken_files == ("tests/test_shop.py",) or "tests/test_shop.py::test_checkout" in result.newly_failing)


def test_a_rerun_that_times_out_is_not_decided(tmp_path, monkeypatch):
    from argus.record import check as module

    real = module.run_pytest
    calls = {"n": 0}

    def flaky_timeout(python, root, timeout, args=(), env=None):
        if "--collect-only" not in args and "-c" not in args:
            calls["n"] += 1
            if calls["n"] == 2:  # the second test run: the rerun on the head
                return None, 3.0
        return real(python, root, timeout, args, env)

    monkeypatch.setattr(module, "run_pytest", flaky_timeout)
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    result = run_check(repo)

    assert result.exit_code == 2 and "did not finish" in result.unreadable


def test_too_many_selected_tests_are_not_decided(tmp_path, monkeypatch):
    from argus.record import check as module

    monkeypatch.setattr(module, "MAX_TESTS", 1)
    repo = make_repo(tmp_path)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    result = run_check(repo)

    assert result.exit_code == 2 and "more than the 1" in result.unreadable


def test_default_python_prefers_the_repository_venv(tmp_path):
    from argus.record.check import default_python

    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("", encoding="utf-8")

    assert default_python(tmp_path) == str(tmp_path / ".venv" / "bin" / "python")
    assert default_python(tmp_path / "nowhere") == sys.executable


def test_command_rejects_a_path_that_is_not_a_directory_and_a_zero_timeout(tmp_path):
    repo = make_repo(tmp_path)

    assert CliRunner().invoke(app, ["check", str(tmp_path / "nope")]).exit_code == 2
    assert CliRunner().invoke(app, ["check", str(repo.root), "--timeout", "0"]).exit_code == 2


def test_a_repository_that_sets_its_own_pytest_verbosity_still_gets_its_tests_collected(tmp_path):
    quiet = {**BASE_FILES, "pyproject.toml": '[tool.pytest.ini_options]\naddopts = "-q"\n'}
    repo = make_repo(tmp_path, quiet)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 3"))

    result = run_check(repo)

    assert result.uncollected == () and result.unreadable is None
    assert "tests/test_shop.py::test_charge" in result.newly_failing


def test_a_repository_that_sets_its_own_pytest_verbosity_still_gets_its_passing_tests_counted(tmp_path):
    quiet = {**BASE_FILES, "pyproject.toml": '[tool.pytest.ini_options]\naddopts = "-q"\n'}
    repo = make_repo(tmp_path, quiet)
    repo.write("shop/billing.py", BILLING.replace("amount * 2", "amount * 2 + 0"))

    result = run_check(repo)

    assert result.ran > 0 and result.newly_failing == () and result.exit_code == 0


def test_collecting_does_not_depend_on_how_pytest_prints(tmp_path):
    from argus.record.check import _collect
    from argus.record.runner import scrubbed_env

    repo = make_repo(tmp_path, {**BASE_FILES, "pyproject.toml": '[tool.pytest.ini_options]\naddopts = "-qq"\n'})

    ids, broken = _collect(sys.executable, repo.root, ["tests/test_shop.py"], 60, scrubbed_env(tmp_path))

    assert ids == {"tests/test_shop.py::test_charge", "tests/test_shop.py::test_checkout", "tests/test_shop.py::test_fee"} and broken == frozenset()
