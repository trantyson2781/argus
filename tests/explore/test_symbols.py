from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus.cli import app
from argus.explore.claims import Tier
from argus.explore.symbols import build_symbol_view, resolve_symbol
from argus.extract import scan

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]

LIB_FILES = {
    "pyproject.toml": '[project]\nname = "tinylib"\nversion = "0"\n',
    "src/tinylib/__init__.py": "",
    "src/tinylib/shapes.py": (
        '"""Shapes."""\nimport math\n\nfrom tinylib import units\n\n\n'
        'class Circle:\n    """A circle."""\n\n    def __init__(self, r):\n        self.r = r\n\n'
        "    def area(self):\n        return self.scale(math.pi * self.r ** 2)\n\n"
        "    def scale(self, v):\n        return units.convert(v)\n\n"
        "    def describe(self, other):\n        n = len(str(self.r))\n        return other.render(n)\n\n\n"
        "def make(r):\n    return Circle(r)\n\n\n"
        "def twin(r):\n    return make(r), make(r)\n"
    ),
    "src/tinylib/units.py": "def convert(v):\n    return v\n\n\ndef make():\n    pass\n",
    "tests/test_shapes.py": "from tinylib.shapes import make\n\n\ndef test_make():\n    make(1)\n",
}


@pytest.fixture
def lib_repo(tmp_path: Path) -> Path:
    for rel, content in LIB_FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def _view(repo: Path, query: str, include_tests: bool = False, resolve_v2: bool = False):
    """Pinned to V1 unless asked: the older tests below describe V1's gaps (the V2 behaviour has its own tests)."""
    model = scan(repo, resolve_v2=resolve_v2)
    return build_symbol_view(model, repo, resolve_symbol(model, query), include_tests, resolve_v2)


def test_resolve_by_exact_qualname_suffix_and_bare_name(lib_repo: Path) -> None:
    model = scan(lib_repo)
    assert resolve_symbol(model, "tinylib.shapes.make").qualname == "tinylib.shapes.make"
    assert resolve_symbol(model, "shapes.make").qualname == "tinylib.shapes.make"
    assert resolve_symbol(model, "Circle.area").qualname == "tinylib.shapes.Circle.area"
    assert resolve_symbol(model, "twin").qualname == "tinylib.shapes.twin"


def test_ambiguous_name_lists_candidates_and_never_guesses(lib_repo: Path) -> None:
    with pytest.raises(LookupError) as err:
        resolve_symbol(scan(lib_repo), "make")
    assert "tinylib.shapes.make" in str(err.value) and "tinylib.units.make" in str(err.value)


def test_unknown_name_suggests_close_matches(lib_repo: Path) -> None:
    with pytest.raises(LookupError) as err:
        resolve_symbol(scan(lib_repo), "twinn")
    assert "twin" in str(err.value)


def test_definition_signature_and_missing_docstring_is_unknown(lib_repo: Path) -> None:
    view = _view(lib_repo, "twin")
    assert view.definition.tier is Tier.FACT and view.definition.evidence[0].location == "src/tinylib/shapes.py:28"
    assert view.signature == "(r)"
    assert view.purpose.tier is Tier.UNKNOWN


def test_class_docstring_is_a_quoted_fact_and_members_are_listed(lib_repo: Path) -> None:
    view = _view(lib_repo, "Circle")
    assert view.purpose.tier is Tier.FACT and "A circle." in view.purpose.text
    assert [m.text for m in view.members] == [
        "method tinylib.shapes.Circle.__init__", "method tinylib.shapes.Circle.area",
        "method tinylib.shapes.Circle.scale", "method tinylib.shapes.Circle.describe",
    ]


def test_callers_come_with_call_site_evidence_and_tests_are_hidden_by_default(lib_repo: Path) -> None:
    view = _view(lib_repo, "shapes.make")
    assert [c.text for c in view.callers] == ["tinylib.shapes.twin calls tinylib.shapes.make"] * 2
    assert {c.evidence[0].line for c in view.callers} == {29}
    assert view.hidden_test_callers == 1
    with_tests = _view(lib_repo, "shapes.make", include_tests=True)
    assert len(with_tests.callers) == 3 and with_tests.hidden_test_callers == 0


def test_class_callers_are_instantiations(lib_repo: Path) -> None:
    assert [c.text for c in _view(lib_repo, "Circle").callers] == ["tinylib.shapes.make calls tinylib.shapes.Circle"]


def test_callees_are_resolved_with_evidence_and_externals_are_collapsed(lib_repo: Path) -> None:
    view = _view(lib_repo, "Circle.scale")
    [callee] = view.callees
    assert callee.text.startswith("tinylib.shapes.Circle.scale calls tinylib.units (a module")  # module-alias gap, Step 2
    assert callee.tier is Tier.INFERENCE and callee.evidence[0].line == 17
    assert "math.pi" not in view.external_calls


def test_unresolved_calls_are_listed_with_a_reason_never_as_edges(lib_repo: Path) -> None:
    area = _view(lib_repo, "Circle.area")
    assert area.callees == ()
    [unknown] = area.unknown_calls
    assert unknown.tier is Tier.UNKNOWN and "self.scale" in unknown.text and "self" in unknown.text
    assert unknown.evidence[0].line == 14

    describe = _view(lib_repo, "Circle.describe")
    texts = " ".join(c.text for c in describe.unknown_calls)
    assert "other.render" in texts and "variable or parameter" in texts
    assert describe.builtin_call_count == 2  # len, str


def test_v2_links_the_self_call_and_the_module_alias_that_v1_leaves_open(lib_repo: Path) -> None:
    [via_self] = _view(lib_repo, "Circle.area", resolve_v2=True).callees
    assert "Circle.scale" in via_self.text and "a subclass may override it" in via_self.text and via_self.tier is Tier.INFERENCE
    [via_alias] = _view(lib_repo, "Circle.scale", resolve_v2=True).callees
    assert "tinylib.units.convert" in via_alias.text and via_alias.tier is Tier.INFERENCE and via_alias.evidence[0].line == 17


def test_module_symbol_lists_members_without_a_call_view(lib_repo: Path) -> None:
    view = _view(lib_repo, "tinylib.units")
    assert [m.text for m in view.members] == ["function tinylib.units.convert", "function tinylib.units.make"]
    assert view.callees == () and view.unknown_calls == ()


def test_cli_works_with_just_a_name(lib_repo: Path) -> None:
    result = runner.invoke(app, ["explore", "symbol", "twin", str(lib_repo)])
    assert result.exit_code == 0, result.output
    assert "tinylib.shapes.twin" in result.output and "Calls" in result.output and "shapes.py:29" in result.output


def test_cli_ambiguous_name_exits_2_with_candidates(lib_repo: Path) -> None:
    result = runner.invoke(app, ["explore", "symbol", "make", str(lib_repo)])
    assert result.exit_code == 2 and "tinylib.units.make" in result.output


def test_cli_limit_reports_hidden_rows(lib_repo: Path) -> None:
    result = runner.invoke(app, ["explore", "symbol", "shapes.make", str(lib_repo), "--limit", "1"])
    assert result.exit_code == 0 and "+1 more" in result.output


def test_cli_is_read_only(lib_repo: Path) -> None:
    before = sorted((p, p.stat().st_mtime_ns) for p in lib_repo.rglob("*") if p.is_file())
    runner.invoke(app, ["explore", "symbol", "twin", str(lib_repo)])
    assert before == sorted((p, p.stat().st_mtime_ns) for p in lib_repo.rglob("*") if p.is_file())


def test_on_argus_run_benchmark_is_found_with_callers() -> None:
    result = runner.invoke(app, ["explore", "symbol", "run_benchmark", str(ROOT)])
    assert result.exit_code == 0, result.output
    assert "argus.benchmark.runner.run_benchmark" in result.output and "benchmark/cli.py" in result.output


def test_method_view_warns_that_self_callers_may_be_missing(lib_repo: Path) -> None:
    result = runner.invoke(app, ["explore", "symbol", "Circle.scale", str(lib_repo)])
    assert "may be missing" in result.output and "single definition" in " ".join(result.output.split())  # default is V2 on


# -- messages must be true whether or not RESOLVE_V2 is on (Step 6 audit) -----------------------------------------------

INHERIT_FILES = {
    "pyproject.toml": '[project]\nname = "inh"\nversion = "0"\n',
    "src/inh/__init__.py": "",
    "src/inh/m.py": (
        "class Base:\n    def b(self):\n        pass\n\n\n"
        "class C(Base):\n    def a(self):\n        self.b()\n\n    @classmethod\n    def make(cls):\n        cls.helper()\n\n"
        "    def helper(self):\n        pass\n\n    def talk(self):\n        self.session.get()\n"
    ),
}


@pytest.fixture
def inherit_repo(tmp_path: Path) -> Path:
    for rel, content in INHERIT_FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def _text(*args: str, v2: bool) -> str:
    env = {"ARGUS_RESOLVE_V2": "1" if v2 else "0"}
    result = runner.invoke(app, ["explore", "symbol", *args], env=env)
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_footnote_with_v2_off_says_resolution_was_disabled(lib_repo: Path) -> None:
    out = _text("Circle.scale", str(lib_repo), v2=False)
    assert "ARGUS_RESOLVE_V2=0" in out and "disabled" in out and "may be missing" in out
    assert "single definition" not in out


def test_footnote_with_v2_on_does_not_claim_self_calls_are_unresolved(lib_repo: Path) -> None:
    out = _text("Circle.scale", str(lib_repo), v2=True)
    assert "not resolved yet" not in out and "RESOLVE_V2" not in out
    assert "single definition" in out and "inherit" in out and "property" in out and "may be missing" in out


def test_unresolved_self_call_reason_names_the_flag_only_when_it_is_off(inherit_repo: Path) -> None:
    off = _text("C.a", str(inherit_repo), v2=False)
    on = _text("C.a", str(inherit_repo), v2=True)
    assert "self.b" in off and "ARGUS_RESOLVE_V2=0" in off
    assert "self.b" in on and "RESOLVE_V2" not in on and "inherited" in on and "property" in on


def test_cls_call_reason_never_blames_the_flag(inherit_repo: Path) -> None:
    for v2 in (False, True):
        model = scan(inherit_repo, resolve_v2=v2)
        view = build_symbol_view(model, inherit_repo, resolve_symbol(model, "C.make"), resolve_v2=v2)
        [unknown] = view.unknown_calls
        assert "cls.helper" in unknown.text and "RESOLVE_V2" not in unknown.text and "class-method calls are not resolved" in unknown.text


def test_build_symbol_view_takes_the_flag_explicitly(inherit_repo: Path) -> None:
    model = scan(inherit_repo, resolve_v2=True)
    node = resolve_symbol(model, "C.a")
    on = build_symbol_view(model, inherit_repo, node, resolve_v2=True)
    off = build_symbol_view(model, inherit_repo, node, resolve_v2=False)
    assert "RESOLVE_V2" not in on.unknown_calls[0].text and "ARGUS_RESOLVE_V2=0" in off.unknown_calls[0].text


def test_build_symbol_view_without_a_flag_follows_the_same_default_as_scan(inherit_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = scan(inherit_repo)
    node = resolve_symbol(model, "C.a")
    assert "RESOLVE_V2" not in build_symbol_view(model, inherit_repo, node).unknown_calls[0].text
    monkeypatch.setenv("ARGUS_RESOLVE_V2", "0")
    assert "ARGUS_RESOLVE_V2=0" in build_symbol_view(scan(inherit_repo), inherit_repo, node).unknown_calls[0].text


def test_an_import_withheld_by_a_guard_is_not_described_as_a_plain_variable(tmp_path: Path) -> None:
    files = {
        "pyproject.toml": '[project]\nname = "g"\nversion = "0"\n', "src/g/__init__.py": "",
        "src/g/pkg/__init__.py": "def shadowed():\n    pass\n", "src/g/pkg/shadowed.py": "def f():\n    pass\n",
        "src/g/app.py": "from g.pkg import shadowed\n\n\ndef run():\n    shadowed.f()\n",
    }
    for rel, content in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(content, encoding="utf-8")
    out = _text("run", str(tmp_path), v2=True)
    assert "shadowed.f" in out and "ambiguous import" in out


def test_call_on_an_attribute_of_self_is_explained_by_the_unknown_type_not_by_inheritance(inherit_repo: Path) -> None:
    for v2 in (False, True):
        out = _text("C.talk", str(inherit_repo), v2=v2)
        assert "self.session.get" in out and "attribute of self" in out and "type is unknown" in out
        assert "inherited" not in out.split("Unresolved calls")[1] and "RESOLVE_V2 is off" not in out.split("Unresolved calls")[1]
