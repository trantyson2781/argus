"""Record mode, slice 2: matching a recorded function to a scanned node (`argus.record.match`).

A recorded function (file, first line, `co_qualname`) matches a node only when the file, the qualified name and the position all agree,
and exactly one node fits. Anything else stays unmatched with a reason; a match is never guessed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.extract import scan
from argus.record.match import Match, match_functions

SRC = '''\
"""Module."""


def top(x):
    return x


def deco(f):
    return f


@deco
def decorated(x):
    return x


class K:
    def m(self):
        def inner():
            return 1
        return inner()

    @property
    def p(self):
        return 1

    @p.setter
    def p(self, v):
        pass


f = lambda: 1


if True:
    def twice():
        return 1
else:
    def twice():
        return 2
'''


def line(text: str, nth: int = 1) -> int:
    hits = [n for n, row in enumerate(SRC.splitlines(), 1) if row.strip() == text]
    return hits[nth - 1]


@pytest.fixture
def matches(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text(SRC, encoding="utf-8")
    model = scan(tmp_path)

    def run(qualname: str, first_line: int, file: str = "pkg/mod.py") -> Match:
        key = f"{file}:{first_line}:{qualname}"
        return match_functions(model, [{"key": key, "file": file, "first_line": first_line, "code_qualname": qualname}])[key]

    return run


def test_a_plain_function_matches_its_node(matches) -> None:
    assert matches("top", line("def top(x):")) == Match("function:pkg.mod.top", None)


def test_a_decorated_function_matches_although_its_first_line_is_the_decorator(matches) -> None:
    assert matches("decorated", line("@deco")) == Match("function:pkg.mod.decorated", None)


def test_methods_and_classes_and_module_code_match(matches) -> None:
    assert matches("K.m", line("def m(self):")).node_id == "method:pkg.mod.K.m"
    assert matches("K", line("class K:")).node_id == "class:pkg.mod.K"
    assert matches("<module>", 1).node_id == "module:pkg.mod"


def test_a_nested_function_matches_its_node_without_the_locals_marker(matches) -> None:
    assert matches("K.m.<locals>.inner", line("def inner():")).node_id == "function:pkg.mod.K.m.inner"


def test_a_property_and_its_setter_are_told_apart_by_position(matches) -> None:
    assert matches("K.p", line("@property")).node_id == "method:pkg.mod.K.p"
    assert matches("K.p", line("@p.setter")).node_id == "method:pkg.mod.K.p#2"


def test_two_definitions_of_one_name_are_told_apart_by_position(matches) -> None:
    assert matches("twice", line("def twice():", 1)).node_id == "function:pkg.mod.twice"
    assert matches("twice", line("def twice():", 2)).node_id == "function:pkg.mod.twice#2"


@pytest.mark.parametrize("qualname", ["<lambda>", "K.m.<locals>.<listcomp>", "<genexpr>"])
def test_anonymous_code_is_unmatched_with_a_reason(matches, qualname: str) -> None:
    assert matches(qualname, line("f = lambda: 1")) == Match(None, "anonymous")


def test_a_name_the_scanner_never_saw_is_unmatched(matches) -> None:
    assert matches("ghost", line("def top(x):")) == Match(None, "no_node")


def test_a_position_after_the_last_definition_is_unmatched(matches) -> None:
    assert matches("top", 500) == Match(None, "no_node")  # no node at or after that line


def test_a_file_the_scanner_did_not_read_is_unmatched(matches) -> None:
    assert matches("top", 4, file="elsewhere.py") == Match(None, "file_not_scanned")


def test_a_wrong_name_at_a_right_position_is_not_rescued_by_the_position(matches) -> None:
    assert matches("top", line("def deco(f):")) == Match(None, "no_node")
