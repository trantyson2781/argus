"""Record mode, slice 3: the gain measure (`argus.record.gate`).

Question it answers: of the call sites that static reading could not link, how many gained an observed target? A gain is claimed only by
NAME: the observed target's name is the last word of the call as written. A target seen on a line whose only unresolved call has another
name (an alias such as `real = add_lot`, or just a neighbouring call) is a CANDIDATE for a hand check and is never counted.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from argus.extract import scan
from argus.record.gate import gain_report, sample_gains
from argus.record.store import ObservedFunction, ObservedSite, Processes, Recording

SRC = '''\
class A:
    def go(self, v):
        return v


def by_name(x):
    return x.go(1)


def by_position(x):
    return x.handle(1)


def two_on_a_line(x):
    return x.first(x.second(1))


def never_ran(x):
    return x.go(2)


def resolved():
    return by_name(None)


def resolved_target(v):
    return v


def mixed(x):
    return x.zap(resolved_target(1))


def double(x, y):
    return x.go(y.go(1))
'''
GO = "method:m.A.go"


def fn(name: str, node_id: str, line: int, calls: int = 1) -> ObservedFunction:
    return ObservedFunction(f"m.py:{line}:{name}", "m.py", line, name, node_id, None, (), calls)


def site(caller: ObservedFunction, line: int, callee: ObservedFunction) -> ObservedSite:
    return ObservedSite(caller.key, line, callee.key, 1)


def build(tmp_path: Path, sites_for) -> tuple:
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    model = scan(tmp_path)
    go = fn("A.go", GO, 2)
    funcs = {
        "by_name": fn("by_name", "function:m.by_name", 6), "by_position": fn("by_position", "function:m.by_position", 10),
        "two": fn("two_on_a_line", "function:m.two_on_a_line", 14), "never": fn("never_ran", "function:m.never_ran", 18, calls=0),
        "go": go,
    }
    sites = tuple(sites_for(funcs))
    rec = Recording("0.1.0", "3.13", ("t",), "2026-10-23T10:00:00Z", 1.0, 0, (), tuple(funcs.values()), sites, (), (), Processes(1, 0))
    return model, rec


def test_a_site_whose_target_has_the_name_written_in_the_call_is_a_gain_by_name(tmp_path: Path) -> None:
    model, rec = build(tmp_path, lambda f: [site(f["by_name"], 7, f["go"])])

    report = gain_report(model, tmp_path, rec)

    (gain,) = report.gains
    assert (gain.caller, gain.line, gain.callee_text, gain.target, gain.basis) == ("m.by_name", 7, "x.go", "m.A.go", "name")


def test_a_different_name_on_a_line_with_one_call_and_one_target_is_only_a_candidate(tmp_path: Path) -> None:
    model, rec = build(tmp_path, lambda f: [site(f["by_position"], 11, f["go"])])

    report = gain_report(model, tmp_path, rec)

    assert report.gains == ()  # hand checks of 37 such matches on two repositories found about four in ten wrong
    (candidate,) = report.candidates
    assert (candidate.callee_text, candidate.target, candidate.basis) == ("x.handle", "m.A.go", "position")


def test_two_unresolved_calls_on_one_line_and_a_target_that_matches_neither_is_no_gain(tmp_path: Path) -> None:
    model, rec = build(tmp_path, lambda f: [site(f["two"], 15, f["go"])])

    assert gain_report(model, tmp_path, rec).gains == ()


def test_counts_separate_all_sites_from_sites_in_functions_that_ran(tmp_path: Path) -> None:
    model, rec = build(tmp_path, lambda f: [site(f["by_name"], 7, f["go"])])

    report = gain_report(model, tmp_path, rec)

    assert report.unresolved == 8  # go, handle, first, second, go again in never_ran, zap, and the two go calls in double
    assert report.in_code_that_ran == 4  # never_ran did not run
    assert len(report.gains) == 1


def test_sites_whose_call_name_matches_no_function_of_the_repo_are_told_apart(tmp_path: Path) -> None:
    model, rec = build(tmp_path, lambda f: [site(f["by_name"], 7, f["go"])])

    report = gain_report(model, tmp_path, rec)

    # in functions that ran: x.go (a repo method), x.handle, x.first and x.second (names the repo does not define, so a library or a
    # dynamic call: these can never gain a repo target by name)
    assert report.in_production_that_ran == 4 and report.internal_looking == 1


def test_a_site_in_a_function_that_did_not_run_is_never_a_gain(tmp_path: Path) -> None:
    model, rec = build(tmp_path, lambda f: [site(f["never"], 19, f["go"])])

    assert gain_report(model, tmp_path, rec).gains == ()


def test_a_target_the_scan_could_not_match_is_not_a_gain(tmp_path: Path) -> None:
    model, rec = build(tmp_path, lambda f: [])
    ghost = ObservedFunction("m.py:99:ghost", "m.py", 99, "ghost", None, "no_node", (), 1)
    by_name = next(f for f in rec.functions if f.code_qualname == "by_name")
    rec = replace(rec, functions=(*rec.functions, ghost), sites=(site(by_name, 7, ghost),))

    assert gain_report(model, tmp_path, rec).gains == ()


def test_a_target_the_scan_already_linked_on_that_line_does_not_explain_an_unresolved_call(tmp_path: Path) -> None:
    model, rec = build(tmp_path, lambda f: [])
    mixed_line = next(n for n, text in enumerate(SRC.splitlines(), 1) if "x.zap" in text)
    mixed = fn("mixed", "function:m.mixed", mixed_line - 1)
    target = fn("resolved_target", "function:m.resolved_target", mixed_line - 5)
    rec = replace(rec, functions=(*rec.functions, mixed, target), sites=(site(mixed, mixed_line, target),))

    assert gain_report(model, tmp_path, rec).gains == ()  # resolved_target(...) is called on that line, x.zap(...) is something else


def test_one_observed_target_explains_one_call_not_two_with_the_same_name(tmp_path: Path) -> None:
    model, rec = build(tmp_path, lambda f: [])
    double_line = next(n for n, text in enumerate(SRC.splitlines(), 1) if "x.go(y.go(1))" in text)
    double = fn("double", "function:m.double", double_line - 1)
    go = next(f for f in rec.functions if f.code_qualname == "A.go")
    rec = replace(rec, functions=(*rec.functions, double), sites=(site(double, double_line, go),))

    assert len(gain_report(model, tmp_path, rec).gains) == 1


def test_gains_in_production_code_are_counted_apart_from_gains_in_tests(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_t.py").write_text("def test_t(x):\n    return x.go(1)\n", encoding="utf-8")
    model, rec = build(tmp_path, lambda f: [site(f["by_name"], 7, f["go"])])
    test_fn = ObservedFunction("tests/test_t.py:1:test_t", "tests/test_t.py", 1, "test_t", "function:tests.test_t.test_t", None, (), 1)
    test_site = ObservedSite(test_fn.key, 2, rec.functions[-1].key, 1)
    rec = replace(rec, functions=(*rec.functions, test_fn), sites=(*rec.sites, test_site))

    report = gain_report(scan(tmp_path), tmp_path, rec)

    assert sorted((g.caller, g.is_test) for g in report.gains) == [("m.by_name", False), ("tests.test_t.test_t", True)]
    assert report.in_production_that_ran == report.in_code_that_ran - 1  # the one site in the test function


def test_the_sample_is_seeded_and_never_larger_than_the_gains(tmp_path: Path) -> None:
    model, rec = build(tmp_path, lambda f: [site(f["by_name"], 7, f["go"])])
    double_line = next(n for n, text in enumerate(SRC.splitlines(), 1) if "x.go(y.go(1))" in text)
    double = fn("double", "function:m.double", double_line - 1)
    go = next(f for f in rec.functions if f.code_qualname == "A.go")
    rec = replace(rec, functions=(*rec.functions, double), sites=(*rec.sites, site(double, double_line, go)))
    gains = gain_report(model, tmp_path, rec).gains

    assert sample_gains(gains, 10, seed=0) == sample_gains(gains, 10, seed=0)
    assert sorted(g.line for g in sample_gains(gains, 10, seed=0)) == [7, double_line]
    assert len(sample_gains(gains, 1, seed=0)) == 1
