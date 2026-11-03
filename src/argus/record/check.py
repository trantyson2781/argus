"""`argus check`: run the tests that reach a change, on throwaway copies, and say which ones are newly failing.

EXECUTES the repository's tests (the same as running pytest yourself), so it is only ever started by this explicit command and never by
`explore` or the TUI (`tests/record/test_guard.py`). The repository itself is never written to: the head is a copy of the working tree, the
base is that copy with the files git tracks at the base ref put back (and `.py` files git does not track there removed), and the tests get
an empty HOME and no keys. The disk and the network are not isolated: use it on repositories you wrote.

Which tests: the union of three tagged sources, none of them a complete list ("no test found" is a lower bound, never "nothing covers it"):
  direct    a test function the call graph says calls the changed function
  caller    a test function that calls a production caller of the changed function
  observed  a test the latest recording saw reach the changed function (made before the change; STALE files are named)
State before: failing tests are run again on the head and twice on the base. Failed twice on the head and passed twice on the base is
`newly failing`; failed everywhere is `already failing`; any other mix is `flaky`. A test that does not exist on the base is reported apart
(`new test`), a test file the change stops from loading is a failure of its own, and a result is never worded as safe.
Exit codes: 0 tests ran and none newly failing, 1 a newly failing test or a file that no longer loads, 2 not decided, 3 no test ran.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from argus.explore.changes import DiffView
from argus.explore.compare import compare_refs
from argus.explore.grouping import MODULE_KINDS, is_test_function, owner_module
from argus.explore.render_diff import printable
from argus.explore.sides import DiffError, extract_commit
from argus.extract import scan
from argus.model.schema import ArchModel, EdgeKind, Node
from argus.record.break_check import reaching_tests
from argus.record.observed import changed_files, latest_recording
from argus.record.runner import PYTEST_ARGS, UnsafeCopyError, copy_repo, ensure_safe_copy_dir, parse_run, run_pytest, scrubbed_env
from argus.record.store import Recording

DEFAULT_TIMEOUT = 120.0  # seconds for one pytest run
MAX_TESTS = 5000  # test ids go on the command line
MAX_PACKAGES = 50
FUNCTION_KINDS = ("function", "method")
NO_TEST = "no test found that reaches this change (a lower bound: Argus found none, which does not mean nothing covers it)"
NO_RECORDING = "no recording: the [O] picks are missing (run `argus record -- pytest -q` once to add them)"
# Collecting is done through pytest's own hooks, not by reading what it prints: a repository's `addopts` (a `-q`, a `-qq`) changes that output.
_COLLECT = (
    "import json, sys, pytest\n"
    "class Collector:\n    def __init__(self):\n        self.ids, self.broken = [], []\n"
    "    def pytest_collection_modifyitems(self, items):\n        self.ids = [item.nodeid for item in items]\n"
    "    def pytest_collectreport(self, report):\n        if report.failed:\n            self.broken.append(report.nodeid)\n"
    "c = Collector()\n"
    "code = pytest.main(['--collect-only', '-q', '-p', 'no:cacheprovider', '-p', 'no:randomly', '--continue-on-collection-errors', '--', *sys.argv[1:]], plugins=[c])\n"
    "print('ARGUS-COLLECT ' + json.dumps({'code': int(code), 'ids': c.ids, 'broken': c.broken}))\n"
)
COLLECT_MARK = "ARGUS-COLLECT "
_COUNT = re.compile(r"(\d+) (passed|failed|skipped|deselected|xfailed|xpassed|errors?|warnings?)")
# Where each top-level package of the repository is imported from, as seen from the copy (an editable install would point at the original).
_WHERE = (
    "import importlib.util as u, json, sys\nout = {}\nfor name in sys.argv[1:]:\n    try:\n        spec = u.find_spec(name)\n    except Exception:\n"
    "        spec = None\n    where = list(spec.submodule_search_locations or []) if spec and spec.origin in (None, 'namespace') else [spec.origin] if spec else []\n"
    "    out[name] = where\nprint(json.dumps(out))\n"
)


class CheckError(Exception):
    """The check cannot be set up (bad ref, not a repository, unusable copy). Carries a user-facing message."""


class Outcome(StrEnum):
    NEWLY_FAILING = "newly failing"
    ALREADY_FAILING = "already failing"
    FLAKY = "flaky"


def classify(head1: bool, head2: bool, base1: bool, base2: bool) -> Outcome:
    """Each argument is "the test failed" in one run: the head, the head again, the base, the base again."""
    if head1 and head2 and not base1 and not base2:
        return Outcome.NEWLY_FAILING
    if head1 and head2 and base1 and base2:
        return Outcome.ALREADY_FAILING
    return Outcome.FLAKY


def node_id(test: str) -> str:
    """The pytest node id of a test written `file::Class.test` (the form `failed_tests` and `reaching_tests` use)."""
    file, _, qual = test.partition("::")
    return f"{file}::{qual.replace('.', '::')}"


def _file(test: str) -> str:
    return test.partition("::")[0]


@dataclass(frozen=True, slots=True)
class Selected:
    test: str  # "file::qualname"
    sources: frozenset[str]  # "direct", "caller", "observed"


@dataclass(frozen=True, slots=True)
class Selection:
    tests: tuple[Selected, ...]
    stale: tuple[str, ...]  # test files that changed after the recording the "observed" tests came from
    untraced: int = 0  # call sites that could not be followed to a test (module-level code, test helpers, names the scan cannot match)


@dataclass(frozen=True, slots=True)
class CheckResult:
    selection: Selection
    missing: tuple[str, ...]  # selected tests whose file is not in the working tree (deleted or renamed)
    uncollected: tuple[str, ...]  # selected tests pytest does not collect (a helper class, a renamed test)
    broken_files: tuple[str, ...]  # test files that load on the base and no longer load with the change
    ran: int  # tests pytest executed (passed or failed), not the number selected
    skipped: int
    newly_failing: tuple[str, ...]
    already_failing: tuple[str, ...]
    flaky: tuple[str, ...]
    new_tests_failing: tuple[str, ...]  # fail with the change and do not exist on the base: nothing to compare with
    unreadable: str | None  # why the check could not be decided, if it could not
    base_label: str
    recording_note: str | None

    @property
    def exit_code(self) -> int:
        if self.unreadable:
            return 2
        if self.newly_failing or self.broken_files or self.new_tests_failing:
            return 1
        return 3 if not self.ran else 0


@dataclass(frozen=True, slots=True)
class Run:
    failed: frozenset[str]
    executed: int  # tests that passed, failed, xfailed, xpassed or errored
    skipped: int


# -- choosing the tests -------------------------------------------------------------


def _test_id(node: Node, nodes: dict[str, Node]) -> str | None:
    module = owner_module(node, nodes)
    return f"{node.provenance.file}::{node.qualname[len(module) + 1:]}" if module and node.qualname.startswith(f"{module}.") else None


def select_tests(view: DiffView, model: ArchModel, rec: Recording | None, root: Path) -> Selection:
    """The tests that reach the modified or removed functions of `view`, each tagged with why it was picked."""
    nodes = {n.id: n for n in model.nodes}
    by_name = {n.qualname: n for n in model.nodes}
    calls_into: defaultdict[str, list[str]] = defaultdict(list)
    for edge in model.edges:
        if edge.kind is EdgeKind.CALLS:
            calls_into[edge.dst_id].append(edge.src_id)
    found: defaultdict[str, set[str]] = defaultdict(set)
    untraced = 0
    changed = [s for s in view.symbols if s.change != "added" and s.kind in FUNCTION_KINDS]
    for symbol in changed:
        for user in symbol.users:
            if user.module_level:
                untraced += 1
            elif user.is_test:
                if user.caller.rpartition(".")[2].startswith("test") and user.caller.startswith(f"{user.module}."):
                    found[f"{user.file}::{user.caller[len(user.module) + 1:]}"].add("direct")
                else:
                    untraced += 1
            elif (caller := by_name.get(user.caller)) is None:
                untraced += 1
            else:
                for src_id in calls_into.get(caller.id, ()):
                    test = nodes[src_id]
                    if is_test_function(test) and (test_id := _test_id(test, nodes)):
                        found[test_id].add("caller")
    observed: set[str] = set()
    if rec is not None:
        for symbol in changed:
            observed |= reaching_tests(rec, symbol.node_id)
        for test_id in observed:
            found[test_id].add("observed")
    stale = changed_files(rec, root, {_file(t) for t in observed}) if rec is not None and observed else frozenset()
    return Selection(tuple(Selected(t, frozenset(s)) for t, s in sorted(found.items())), tuple(sorted(stale)), untraced)


# -- running pytest -----------------------------------------------------------------


def _counts(stdout: str) -> dict[str, int]:
    """The numbers of pytest's summary line ("3 passed, 1 skipped in 0.2s"); empty when there is none."""
    for text in reversed(stdout.splitlines()):
        if re.search(r" in [\d.]+s", text):
            return {word.rstrip("s") if word.startswith("error") else word: int(n) for n, word in _COUNT.findall(text)}
    return {}


def _run(python: str, root: Path, tests: Collection[str], timeout: float, env: dict[str, str]) -> Run | str:
    """One pytest run on `tests`, or the reason it could not be read."""
    args = (*PYTEST_ARGS, "--", *sorted(node_id(t) for t in tests))
    try:
        done, seconds = run_pytest(python, root, timeout, args, env)
    except OSError as err:
        return f"pytest could not be started ({err.strerror or type(err).__name__})"
    if done is None:
        return f"a test run did not finish within {timeout:.0f} s and was stopped ({seconds:.0f} s)"
    failed = parse_run(done.returncode, done.stdout)
    if failed is None:
        return f"a test run could not be read (pytest exit code {done.returncode})"
    c = _counts(done.stdout)
    return Run(failed.tests, sum(c.get(k, 0) for k in ("passed", "failed", "xfailed", "xpassed", "error")), c.get("skipped", 0))


def _collect(python: str, root: Path, files: Iterable[str], timeout: float, env: dict[str, str]) -> tuple[frozenset[str], frozenset[str]] | str:
    """(ids of the test functions pytest collects in `files`, files that fail to load), or the reason collecting failed."""
    try:
        done, _ = run_pytest(python, root, timeout, ("-B", "-c", _COLLECT, *sorted(files)), env)
    except OSError as err:
        return f"pytest could not be started ({err.strerror or type(err).__name__})"
    if done is None:
        return f"collecting the tests did not finish within {timeout:.0f} s"
    line = next((t for t in reversed(done.stdout.splitlines()) if t.startswith(COLLECT_MARK)), None)
    try:
        found = json.loads(line[len(COLLECT_MARK):]) if line else None
    except ValueError:
        found = None
    if not isinstance(found, dict) or found.get("code") not in (0, 1, 2, 5):  # 5: nothing collected, 1 or 2: a file failed to load
        return f"the tests could not be collected (pytest exit code {done.returncode})"
    ids = {f"{file}::{rest.split('[')[0].replace('::', '.')}" for file, _, rest in (i.partition("::") for i in found["ids"])}
    return frozenset(ids), frozenset(i.partition("::")[0] for i in found["broken"])


def _outside_copy(python: str, copy: Path, model: ArchModel, timeout: float, env: dict[str, str]) -> str | None:
    """A reason when the repository's own packages are imported from somewhere other than `copy` (an editable install): the copy would
    not be what is tested. None when they all resolve inside it."""
    names = sorted({n.qualname.split(".")[0] for n in model.nodes if n.kind in MODULE_KINDS})[:MAX_PACKAGES]
    if not names:
        return None
    try:
        done, _ = run_pytest(python, copy, timeout, ("-c", _WHERE, *names), env)
    except OSError as err:
        return f"the Python {python} could not be started ({err.strerror or type(err).__name__})"
    try:
        where = json.loads(done.stdout.strip().splitlines()[-1]) if done is not None and done.returncode == 0 else None
    except (ValueError, IndexError):
        where = None
    if not isinstance(where, dict):
        return "It could not be checked where the repository's packages are imported from"
    inside = copy.resolve()
    for name, places in where.items():
        for place in places:
            if not Path(place).resolve().is_relative_to(inside):
                return f"the package {name} is imported from {place}, outside the copy (an editable install?): the copy would not be what is tested; use a Python without that install"
    return None


# -- the check ------------------------------------------------------------------------


def default_python(root: Path) -> str:
    venv = root / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def check_change(root: Path, base: str = "HEAD", python: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> CheckResult:
    """Compare the working tree of `root` with `base`, pick the tests that reach the change and run them (see the module docstring)."""
    root = root.resolve()
    try:
        view = compare_refs(root, base)
    except DiffError as err:
        raise CheckError(str(err)) from None
    model = scan(root)
    rec, rec_problem = latest_recording(root)
    selection = select_tests(view, model, rec, root)
    files = {_file(s.test) for s in selection.tests if (root / _file(s.test)).is_file()}
    missing = tuple(sorted(s.test for s in selection.tests if _file(s.test) not in files))
    result = CheckResult(selection, missing, (), (), 0, 0, (), (), (), (), None, view.base_label, rec_problem or (None if rec else NO_RECORDING))
    if not files:
        return result
    if len(selection.tests) - len(missing) > MAX_TESTS:
        return replace(result, unreadable=f"{len(selection.tests)} tests were selected, more than the {MAX_TESTS} one run can name")
    python = python or default_python(root)
    with tempfile.TemporaryDirectory(prefix="argus-check-") as tmp:
        tmp_path = Path(tmp)

        def env_for(name: str) -> dict[str, str]:  # each run gets its own empty HOME, so runs side by side cannot share files there
            home = tmp_path / f"home-{name}"
            home.mkdir(exist_ok=True)
            return scrubbed_env(home)

        head = tmp_path / "head"
        try:
            ensure_safe_copy_dir(root, head)
            copy_repo(root, head)
        except (UnsafeCopyError, OSError) as err:
            raise CheckError(str(err)) from None
        if (problem := _outside_copy(python, head, model, timeout, env_for("head"))) is not None:
            return replace(result, unreadable=problem)
        on_head = _collect(python, head, files, timeout, env_for("head"))
        if isinstance(on_head, str):
            return replace(result, unreadable=on_head)
        wanted = [s.test for s in selection.tests if _file(s.test) in files]
        runnable = [t for t in wanted if t in on_head[0]]
        with ThreadPoolExecutor(max_workers=2) as pool:  # the base is prepared while the first run on the head goes
            base_future = pool.submit(_prepare_base, root, base, tmp_path / "base", python, files, timeout, env_for("base"))
            first_future = pool.submit(_run, python, head, runnable, timeout, env_for("head")) if runnable else None
            try:
                prepared = base_future.result()
            except (UnsafeCopyError, DiffError, OSError) as err:
                raise CheckError(str(err)) from None
            first = first_future.result() if first_future else None
        if isinstance(prepared, str):
            return replace(result, unreadable=prepared)
        return _execute(result, wanted, on_head, prepared, first, (python, head, tmp_path / "base", timeout, env_for))


def _prepare_base(root: Path, base: str, dest: Path, python: str, files: set[str], timeout: float, env: dict[str, str]) -> tuple[frozenset[str], frozenset[str]] | str:
    """Build the base copy (the working tree with the files git tracks at `base` put back, and `.py` files it does not track removed) and
    collect the tests in it."""
    ensure_safe_copy_dir(root, dest)
    copy_repo(root, dest)
    tracked = extract_commit(root, base, dest)
    for path in dest.rglob("*.py"):  # a source file git does not track at the base was not there
        if path.relative_to(dest).as_posix() not in tracked:
            path.unlink()
    return _collect(python, dest, files, timeout, env)


def _execute(
    result: CheckResult, wanted: list[str], on_head: tuple[frozenset[str], frozenset[str]], on_base: tuple[frozenset[str], frozenset[str]],
    first: Run | str | None, where: tuple[str, Path, Path, float, Callable[[str], dict[str, str]]],
) -> CheckResult:
    python, head, base, timeout, env_for = where
    (head_ids, head_broken), (base_ids, base_broken) = on_head, on_base
    broken = tuple(sorted(head_broken - base_broken))  # loads on the base, does not load with the change
    uncollected = tuple(sorted(t for t in wanted if t not in head_ids and _file(t) not in head_broken))
    result = replace(result, uncollected=uncollected, broken_files=broken)
    if first is None:
        return result
    if isinstance(first, str):
        return replace(result, unreadable=first)
    failing = sorted(first.failed)
    result = replace(result, ran=first.executed, skipped=first.skipped, new_tests_failing=tuple(t for t in failing if t not in base_ids))
    compare = [t for t in failing if t in base_ids]
    if not compare:
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:  # the three re-runs do not depend on each other
        runs = [pool.submit(_run, python, where_, compare, timeout, env_for(name)) for name, where_ in (("head2", head), ("base1", base), ("base2", base))]
        results = [r.result() for r in runs]
    for run in results:
        if isinstance(run, str):
            return replace(result, unreadable=run)
    head2, base1, base2 = (r.failed for r in results if isinstance(r, Run))
    groups: defaultdict[Outcome, list[str]] = defaultdict(list)
    for test in compare:
        groups[classify(True, test in head2, test in base1, test in base2)].append(test)
    return replace(
        result, newly_failing=tuple(groups[Outcome.NEWLY_FAILING]), already_failing=tuple(groups[Outcome.ALREADY_FAILING]), flaky=tuple(groups[Outcome.FLAKY]),
    )


# -- the report ---------------------------------------------------------------------------


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _listed(title: str, tests: Iterable[str]) -> list[str]:
    tests = list(tests)
    return [f"  {title}:", *(f"    {t}" for t in tests)] if tests else []


def result_lines(result: CheckResult) -> list[str]:
    """The report as plain lines (repository names are escaped). What the runs showed carries [O], what was picked by reading carries [I]."""
    selection = result.selection
    lines = [f"Check  {result.base_label} -> working tree   (ran on throwaway copies; your files were not changed)"]
    if result.recording_note:
        lines.append(f"  note: {result.recording_note}")
    if not selection.tests:
        lines.append(f"Result: {NO_TEST}")
        return [printable(line) for line in lines]
    by_source = {tag: sum(1 for s in selection.tests if tag in s.sources) for tag in ("direct", "caller", "observed")}
    lines.append(
        f"Selected {_plural(len(selection.tests), 'test')}: {by_source['direct']} call it directly [I], {by_source['caller']} call a caller of it [I], "
        f"{by_source['observed']} reached it in the latest recording [O]",
    )
    if selection.untraced:
        lines.append(f"  ! {_plural(selection.untraced, 'call site')} could not be followed to a test (module-level code, test helpers, names the scan cannot match)")
    if selection.stale:
        lines.append(f"  ! the recording is older than a change in {', '.join(selection.stale)}: those [O] picks may be out of date")
    if result.missing:
        lines.append(f"  ! {_plural(len(result.missing), 'selected test')} not found in the working tree (deleted or renamed): {', '.join(result.missing)}")
    if result.uncollected:
        lines.append(f"  ! {_plural(len(result.uncollected), 'selected test')} not collected by pytest and not run: {', '.join(result.uncollected)}")
    if result.unreadable:
        lines.append(f"Result: not decided. {result.unreadable[0].upper()}{result.unreadable[1:]}.")
        return [printable(line) for line in lines]
    failures = (result.newly_failing, result.already_failing, result.flaky, result.new_tests_failing, result.broken_files)
    if not result.ran and not any(failures):
        lines.append(f"Result: no selected test ran ({result.skipped} skipped). {NO_TEST}" if result.skipped else f"Result: none of the selected tests could be run. {NO_TEST}")
    elif not any(failures):
        extra = f", {result.skipped} skipped" if result.skipped else ""
        lines.append(f"Result [O]: {result.ran} ran{extra}, none failed. Passing tests show those tests did not notice the change, not that the change is right.")
    else:
        lines.append(f"Result [O]: {result.ran} ran, {len(result.newly_failing)} newly failing, {len(result.already_failing)} already failing, {len(result.flaky)} flaky.")
        lines += _listed("newly failing [O] (passed on the base, fail with your change)", result.newly_failing)
        lines += _listed("test files that no longer load [O] (they load on the base)", result.broken_files)
        lines += _listed("new tests failing [O] (not on the base, nothing to compare with)", result.new_tests_failing)
        lines += _listed("already failing [O] (also fail on the base)", result.already_failing)
        lines += _listed("flaky [O] (results differ between runs)", result.flaky)
    return [printable(line) for line in lines]


def check(
    path: Annotated[Path, typer.Argument(help="The repository (default: the current directory).")] = Path("."),
    base: Annotated[str, typer.Option("--base", help="Compare the working tree with this git ref.")] = "HEAD",
    python: Annotated[str | None, typer.Option("--python", help="The Python that runs the tests (default: .venv/bin/python of the repository, else this one).")] = None,
    timeout: Annotated[float, typer.Option("--timeout", help="Seconds one pytest run may take before it is stopped.", min=1)] = DEFAULT_TIMEOUT,
) -> None:
    """Run the tests that reach your uncommitted change on a copy and say which ones are newly failing.

    This EXECUTES your repository's tests (the same as running pytest yourself) on throwaway copies; your files are not changed. Use it only on a
    repository you wrote. Exit code: 0 tests ran and none newly failing, 1 newly failing tests (or a test file that no longer loads), 2 the check
    could not be done, 3 no test ran.
    """
    root = path.resolve()
    if not root.is_dir():
        typer.echo(f"Error: {printable(str(path))} is not a directory.", err=True)
        raise typer.Exit(2)
    typer.echo(printable(f"Argus runs your tests in copies of {root} (python: {shlex.quote(python or default_python(root))})."))
    typer.echo("It executes your code. Only tests that reach the change are run; nothing in your repository is changed.")
    try:
        result = check_change(root, base, python, timeout)
    except (CheckError, OSError) as exc:
        typer.echo(f"Error: {printable(str(exc))}", err=True)
        raise typer.Exit(2) from exc
    typer.echo("\n".join(result_lines(result)))
    raise typer.Exit(result.exit_code)
