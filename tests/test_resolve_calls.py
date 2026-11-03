"""RESOLVE_V2: module-alias calls and direct self.method() calls, resolved only when the source text proves the target."""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.extract import scan
from argus.model.schema import ArchModel, Confidence, Edge, EdgeKind

UTIL = "def helper():\n    pass\n"


def _repo(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _calls(model: ArchModel, src_suffix: str) -> list[Edge]:
    return [e for e in model.edges if e.kind is EdgeKind.CALLS and e.src_id.endswith(f".{src_suffix}")]


def _dsts(model: ArchModel, src_suffix: str) -> list[str]:
    return sorted(e.dst_id for e in _calls(model, src_suffix))


def _app(tmp_path: Path, app_source: str, util_source: str = UTIL, v2: bool = True) -> ArchModel:
    files = {"pkg/__init__.py": "", "pkg/util.py": util_source, "pkg/app.py": app_source}
    return scan(_repo(tmp_path, files), resolve_v2=v2)


# -- module alias -----------------------------------------------------------


def test_from_import_module_alias_call_resolves_to_the_function(tmp_path: Path) -> None:
    model = _app(tmp_path, "from pkg import util\n\n\ndef run():\n    util.helper()\n")
    [edge] = _calls(model, "run")
    assert edge.dst_id == "function:pkg.util.helper"
    assert edge.confidence is Confidence.INFERRED and edge.attrs["resolved_via"] == "module_alias"
    assert edge.provenance.line == 5


def test_with_v2_off_the_call_still_points_at_the_module(tmp_path: Path) -> None:
    model = _app(tmp_path, "from pkg import util\n\n\ndef run():\n    util.helper()\n", v2=False)
    assert _dsts(model, "run") == ["module:pkg.util"]


def test_import_as_alias_resolves_to_the_function(tmp_path: Path) -> None:
    model = _app(tmp_path, "import pkg.util as u\n\n\ndef run():\n    u.helper()\n")
    assert _dsts(model, "run") == ["function:pkg.util.helper"]


def test_reexported_function_resolves_to_its_defining_module(tmp_path: Path) -> None:
    files = {
        "pkg/__init__.py": "", "pkg/impl.py": UTIL, "pkg/util.py": "from pkg.impl import helper\n",
        "pkg/app.py": "from pkg import util\n\n\ndef run():\n    util.helper()\n",
    }
    assert _dsts(scan(_repo(tmp_path, files), resolve_v2=True), "run") == ["function:pkg.impl.helper"]


@pytest.mark.parametrize(
    "body",
    [
        "def run(util):\n    util.helper()\n",
        "def run():\n    util = make()\n    util.helper()\n",
        "def run(items):\n    for util in items:\n        util.helper()\n",
        "def run(cm):\n    with cm as util:\n        util.helper()\n",
        "def run():\n    try:\n        pass\n    except Exception as util:\n        util.helper()\n",
        "def run():\n    def util():\n        pass\n    util.helper()\n",
        "def run():\n    from other import util\n    util.helper()\n",
        "def run(items):\n    return [util.helper() for util in items]\n",
        "def run():\n    if (util := make()):\n        util.helper()\n",
        "def run(*util):\n    util.helper()\n",
        "def run():\n    def inner():\n        util.helper()\n    util = make()\n    inner()\n",
    ],
)
def test_a_shadowed_module_alias_is_not_resolved(tmp_path: Path, body: str) -> None:
    model = _app(tmp_path, f"from pkg import util\n\n\n{body}")
    assert not [d for d in _dsts(model, "run") if d.startswith(("function:pkg.util", "module:pkg.util"))]


def test_module_level_rebinding_blocks_resolution(tmp_path: Path) -> None:
    model = _app(tmp_path, "from pkg import util\n\nutil = object()\n\n\ndef run():\n    util.helper()\n")
    assert not [d for d in _dsts(model, "run") if "pkg.util" in d]


def test_optional_external_import_pattern_keeps_its_external_edge(tmp_path: Path) -> None:
    app = "try:\n    from zoneinfo import ZoneInfo\nexcept ImportError:\n    ZoneInfo = None\n\n\ndef run():\n    return ZoneInfo('x')\n"
    assert _dsts(_app(tmp_path, app), "run") == ["unresolved:zoneinfo.ZoneInfo"]


def test_pytest_fixture_name_used_as_a_parameter_is_not_the_fixture_function(tmp_path: Path) -> None:
    src = "def store():\n    pass\n\n\ndef test_x(store):\n    store.count()\n"
    assert _dsts(_app(tmp_path, src), "test_x") == []
    assert _dsts(_app(tmp_path, src, v2=False), "test_x") == ["function:pkg.app.store"]


def test_unshadowed_alias_still_resolves_next_to_a_shadowed_one(tmp_path: Path) -> None:
    model = _app(tmp_path, "from pkg import util\n\n\ndef a(util):\n    util.helper()\n\n\ndef b():\n    util.helper()\n")
    assert _dsts(model, "a") == [] and _dsts(model, "b") == ["function:pkg.util.helper"]


def test_missing_name_keeps_todays_module_edge(tmp_path: Path) -> None:
    model = _app(tmp_path, "from pkg import util\n\n\ndef run():\n    util.nope()\n")
    assert _dsts(model, "run") == ["module:pkg.util"]


def test_function_defined_twice_is_ambiguous_and_not_picked(tmp_path: Path) -> None:
    twice = "if True:\n    def helper():\n        pass\nelse:\n    def helper():\n        pass\n"
    model = _app(tmp_path, "from pkg import util\n\n\ndef run():\n    util.helper()\n", util_source=twice)
    assert "function:pkg.util.helper" not in _dsts(model, "run")


def test_deeper_attribute_chains_are_deferred(tmp_path: Path) -> None:
    model = _app(tmp_path, "from pkg import util\n\n\ndef run():\n    util.helper.attr()\n")
    assert "function:pkg.util.helper" not in _dsts(model, "run")


def test_external_library_calls_are_never_resolved_internally(tmp_path: Path) -> None:
    app = "import requests\nfrom requests import sessions\n\n\ndef run():\n    requests.get('x')\n    sessions.Session()\n"
    edges = _calls(_app(tmp_path, app), "run")
    assert all(e.dst_id.startswith("unresolved:") for e in edges)


def test_external_library_whose_name_matches_an_internal_function_stays_external(tmp_path: Path) -> None:
    model = _app(tmp_path, "import json\n\n\ndef run():\n    json.helper()\n")
    assert all(e.dst_id.startswith("unresolved:") for e in _calls(model, "run"))


# -- self.method() ----------------------------------------------------------


def _cls(tmp_path: Path, source: str) -> ArchModel:
    return scan(_repo(tmp_path, {"m.py": source}), resolve_v2=True)


def test_direct_self_method_resolves_as_inference_with_override_caveat(tmp_path: Path) -> None:
    model = _cls(tmp_path, "class C:\n    def a(self):\n        self.b()\n\n    def b(self):\n        pass\n")
    [edge] = _calls(model, "C.a")
    assert edge.dst_id == "method:m.C.b" and edge.provenance.line == 3
    assert edge.confidence is Confidence.INFERRED
    assert edge.attrs["resolved_via"] == "self_method" and edge.attrs["override_possible"] is True


def test_self_method_defined_after_use_and_in_async_methods(tmp_path: Path) -> None:
    model = _cls(tmp_path, "class C:\n    async def a(self):\n        await self.b()\n\n    async def b(self):\n        pass\n")
    assert _dsts(model, "C.a") == ["method:m.C.b"]


def test_inherited_method_stays_unresolved(tmp_path: Path) -> None:
    model = _cls(tmp_path, "class Base:\n    def b(self):\n        pass\n\n\nclass C(Base):\n    def a(self):\n        self.b()\n")
    assert _dsts(model, "C.a") == []


def test_method_defined_twice_is_ambiguous(tmp_path: Path) -> None:
    src = "class C:\n    def a(self):\n        self.b()\n\n    def b(self):\n        pass\n\n    def b(self, x):\n        pass\n"
    assert _dsts(_cls(tmp_path, src), "C.a") == []


def test_property_setter_pair_is_ambiguous(tmp_path: Path) -> None:
    src = (
        "class C:\n    def a(self):\n        self.b()\n\n    @property\n    def b(self):\n        pass\n\n"
        "    @b.setter\n    def b(self, v):\n        pass\n"
    )
    assert _dsts(_cls(tmp_path, src), "C.a") == []


def test_instance_attribute_with_the_same_name_blocks_resolution(tmp_path: Path) -> None:
    src = "class C:\n    def __init__(self, cb):\n        self.b = cb\n\n    def a(self):\n        self.b()\n\n    def b(self):\n        pass\n"
    assert _dsts(_cls(tmp_path, src), "C.a") == []


@pytest.mark.parametrize(
    "method",
    [
        "    def a(self):\n        self = other()\n        self.b()\n",
        "    @staticmethod\n    def a(self):\n        self.b()\n",
        "    def a(self):\n        def inner():\n            self.b()\n        inner()\n",
        "    def a(this):\n        this.b()\n",
        "    @classmethod\n    def a(cls):\n        cls.b()\n",
        "    def a(self):\n        self.session.b()\n",
        "    def a(self):\n        self.b.c()\n",
        "    def a(self, self2=None):\n        for self in self2:\n            self.b()\n",
    ],
)
def test_unsafe_self_shapes_are_not_resolved(tmp_path: Path, method: str) -> None:
    model = _cls(tmp_path, f"class C:\n{method}\n    def b(self):\n        pass\n")
    assert _dsts(model, "C.a") == []


def test_self_call_to_a_non_method_attribute_is_not_resolved(tmp_path: Path) -> None:
    model = _cls(tmp_path, "class C:\n    b = staticmethod(print)\n\n    def a(self):\n        self.b()\n")
    assert _dsts(model, "C.a") == []


def test_nested_class_resolves_against_its_own_methods(tmp_path: Path) -> None:
    src = (
        "class Outer:\n    def b(self):\n        pass\n\n    class Inner:\n        def a(self):\n            self.b()\n\n"
        "        def b(self):\n            pass\n"
    )
    assert _dsts(_cls(tmp_path, src), "Outer.Inner.a") == ["method:m.Outer.Inner.b"]


# -- switch and invariants --------------------------------------------------


def test_default_is_on_and_matches_explicit_true(tmp_path: Path) -> None:
    files = {"pkg/__init__.py": "", "pkg/util.py": UTIL, "pkg/app.py": "from pkg import util\n\n\ndef run():\n    util.helper()\n"}
    root = _repo(tmp_path, files)
    default, on = scan(root), scan(root, resolve_v2=True)
    assert [(e.id, e.dst_id) for e in default.edges] == [(e.id, e.dst_id) for e in on.edges]
    assert _dsts(default, "run") == ["function:pkg.util.helper"]


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " False "])
def test_environment_variable_opts_out_to_v1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    root = _repo(tmp_path, {"m.py": "class C:\n    def a(self):\n        self.b()\n\n    def b(self):\n        pass\n"})
    monkeypatch.setenv("ARGUS_RESOLVE_V2", value)
    assert _dsts(scan(root), "C.a") == []


@pytest.mark.parametrize("value", ["", "1", "true", "yes", "on", "ON"])
def test_environment_variable_values_that_mean_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    root = _repo(tmp_path, {"m.py": "class C:\n    def a(self):\n        self.b()\n\n    def b(self):\n        pass\n"})
    monkeypatch.setenv("ARGUS_RESOLVE_V2", value)
    assert _dsts(scan(root), "C.a") == ["method:m.C.b"]


def test_an_unrecognised_value_keeps_the_default_but_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.extract.resolve_calls import resolve_v2_enabled

    monkeypatch.setenv("ARGUS_RESOLVE_V2", "flase")
    with pytest.warns(RuntimeWarning, match="flase.*not recognised.*0 to disable"):
        assert resolve_v2_enabled() is True


def test_an_explicit_argument_beats_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path, {"m.py": "class C:\n    def a(self):\n        self.b()\n\n    def b(self):\n        pass\n"})
    monkeypatch.setenv("ARGUS_RESOLVE_V2", "0")
    assert _dsts(scan(root, resolve_v2=True), "C.a") == ["method:m.C.b"]
    monkeypatch.setenv("ARGUS_RESOLVE_V2", "1")
    assert _dsts(scan(root, resolve_v2=False), "C.a") == []


def test_v2_leaves_nodes_and_non_call_edges_untouched(tmp_path: Path) -> None:
    files = {
        "pkg/__init__.py": "", "pkg/util.py": UTIL,
        "pkg/app.py": "from pkg import util\n\n\nclass C:\n    def a(self):\n        self.b()\n        util.helper()\n\n    def b(self):\n        pass\n",
    }
    root = _repo(tmp_path, files)
    off, on = scan(root, resolve_v2=False), scan(root, resolve_v2=True)
    assert off.nodes == on.nodes
    not_calls = lambda m: sorted(e.id for e in m.edges if e.kind is not EdgeKind.CALLS)  # noqa: E731
    assert not_calls(off) == not_calls(on)


def test_on_argus_imports_are_identical_and_resolved_function_edges_do_not_drop() -> None:
    root = Path(__file__).resolve().parents[1]
    off, on = scan(root, resolve_v2=False), scan(root, resolve_v2=True)
    imports = lambda m: sorted(e.id for e in m.edges if e.kind is EdgeKind.IMPORTS)  # noqa: E731
    assert imports(off) == imports(on)

    def internal_callable_edges(m: ArchModel) -> int:
        return sum(1 for e in m.edges if e.kind is EdgeKind.CALLS and e.dst_id.startswith(("function:", "method:")))

    assert internal_callable_edges(on) > internal_callable_edges(off)


# -- hardening: descriptor targets, package-init shadowing, class-body rebinding (Step 6 audit) ------------


def _method_call(tmp_path: Path, target: str, header: str = "") -> ArchModel:
    src = f"{header}class C:\n    def a(self):\n        self.b()\n\n{target}"
    return _cls(tmp_path, src)


@pytest.mark.parametrize(
    "target",
    [
        "    @property\n    def b(self):\n        return lambda: 1\n",
        "    @functools.cached_property\n    def b(self):\n        return lambda: 1\n",
        "    @cached_property\n    def b(self):\n        return lambda: 1\n",
        "    @abc.abstractproperty\n    def b(self):\n        return lambda: 1\n",
        "    @b.setter\n    def b(self, value):\n        pass\n",
        "    @b.getter\n    def b(self):\n        pass\n",
        "    @b.deleter\n    def b(self):\n        pass\n",
    ],
)
def test_a_property_like_target_is_not_resolved(tmp_path: Path, target: str) -> None:
    model = _method_call(tmp_path, target, header="import abc\nimport functools\nfrom functools import cached_property\n\n\n")
    assert _dsts(model, "C.a") == []


@pytest.mark.parametrize(
    "target",
    [
        "    @staticmethod\n    def b():\n        pass\n",
        "    @classmethod\n    def b(cls):\n        pass\n",
        "    @functools.lru_cache(maxsize=None)\n    def b(self):\n        pass\n",
        "    @contextlib.contextmanager\n    def b(self):\n        yield\n",
    ],
)
def test_ordinary_decorated_targets_still_resolve(tmp_path: Path, target: str) -> None:
    model = _method_call(tmp_path, target, header="import contextlib\nimport functools\n\n\n")
    assert _dsts(model, "C.a") == ["method:m.C.b"]


def test_a_property_does_not_block_the_plain_methods_next_to_it(tmp_path: Path) -> None:
    src = (
        "class C:\n    def a(self):\n        self.p()\n        self.m()\n\n"
        "    @property\n    def p(self):\n        return lambda: 1\n\n    def m(self):\n        pass\n"
    )
    assert _dsts(_cls(tmp_path, src), "C.a") == ["method:m.C.m"]


SHADOW_PKG = {"pkg/shadowed.py": "def f():\n    pass\n"}
SHADOW_APP = "from pkg import shadowed\n\n\ndef run():\n    shadowed.f()\n"


def _shadow(tmp_path: Path, init: str, app: str = SHADOW_APP, v2: bool = True) -> ArchModel:
    return scan(_repo(tmp_path, {"pkg/__init__.py": init, **SHADOW_PKG, "pkg/app.py": app}), resolve_v2=v2)


@pytest.mark.parametrize(
    "init",
    [
        "def shadowed():\n    pass\n",
        "class shadowed:\n    pass\n",
        "shadowed = None\n",
        "from . import other as shadowed\n",
        "try:\n    from pkg.other import shadowed\nexcept ImportError:\n    shadowed = None\n",
        "with open('f') as shadowed:\n    pass\n",
        "if (shadowed := object()):\n    pass\n",
        "import sys\nif sys:\n    def shadowed():\n        pass\n",
        "from pkg.other import shadowed\n",
        "from pkg import shadowed\nfrom pkg.other import shadowed\n",
    ],
)
def test_a_package_init_name_that_shadows_a_submodule_is_not_resolved(tmp_path: Path, init: str) -> None:
    assert _dsts(_shadow(tmp_path, init), "run") == []


def test_package_init_shadowing_is_a_v2_only_change(tmp_path: Path) -> None:
    assert _dsts(_shadow(tmp_path, "def shadowed():\n    pass\n", v2=False), "run") == ["module:pkg.shadowed"]


def test_aliased_submodule_import_is_also_ambiguous_when_the_package_shadows_it(tmp_path: Path) -> None:
    app = "import pkg.shadowed as shadowed\n\n\ndef run():\n    shadowed.f()\n"
    assert _dsts(_shadow(tmp_path, "def shadowed():\n    pass\n", app), "run") == []


@pytest.mark.parametrize("init", ["", "from pkg import shadowed\n", "from . import shadowed\n", "import pkg.shadowed as shadowed\n"])
def test_a_package_that_merely_reimports_the_submodule_does_not_block_resolution(tmp_path: Path, init: str) -> None:
    assert _dsts(_shadow(tmp_path, init), "run") == ["function:pkg.shadowed.f"]


def test_a_package_init_binding_of_a_different_name_does_not_block_resolution(tmp_path: Path) -> None:
    assert _dsts(_shadow(tmp_path, "def other():\n    pass\n"), "run") == ["function:pkg.shadowed.f"]


def _class_body(tmp_path: Path, body: str, v2: bool = True) -> ArchModel:
    files = {"pkg/__init__.py": "", "pkg/util.py": UTIL, "pkg/app.py": f"from pkg import util\n\n\n{body}"}
    return scan(_repo(tmp_path, files), resolve_v2=v2)


@pytest.mark.parametrize(
    "body",
    [
        "class K:\n    util = object()\n    x = util.helper()\n",
        "class K:\n    util: object = object()\n    x = util.helper()\n",
        "class K:\n    import other as util\n    x = util.helper()\n",
        "class K:\n    def util(self):\n        pass\n    x = util.helper()\n",
        "class K:\n    for util in range(3):\n        x = util.helper()\n",
        "class K:\n    x = util.helper()\n    util = object()\n",
        "class K:\n    with open('f') as util:\n        x = util.helper()\n",
        "class K:\n    if (util := object()):\n        x = util.helper()\n",
        "class K:\n    util, other = object(), 1\n    x = util.helper()\n",
        "class K:\n    util = 1\n    util += 1\n    x = util.helper()\n",
        "class K:\n    class util:\n        pass\n    x = util.helper()\n",
        "class K:\n    try:\n        pass\n    except Exception as util:\n        x = util.helper()\n",
        "class Outer:\n    class K:\n        util = object()\n        x = util.helper()\n",
        "def factory():\n    class K:\n        util = object()\n        x = util.helper()\n    return K\n",
    ],
)
def test_a_class_body_rebinding_blocks_resolution(tmp_path: Path, body: str) -> None:
    assert _all_dsts(_class_body(tmp_path, body)) == []


def _all_dsts(model: ArchModel) -> list[str]:
    return sorted(e.dst_id for e in model.edges if e.kind is EdgeKind.CALLS)


def test_class_body_rebinding_is_a_v2_only_change(tmp_path: Path) -> None:
    assert _all_dsts(_class_body(tmp_path, "class K:\n    util = object()\n    x = util.helper()\n", v2=False)) == ["module:pkg.util"]


@pytest.mark.parametrize(
    "body",
    [
        "class K:\n    x = util.helper()\n",
        "class K:\n    other = object()\n    x = util.helper()\n",
        "class K:\n    util = object()\n\n    def m(self):\n        util.helper()\n",
        "class Outer:\n    util = object()\n\n    class K:\n        x = util.helper()\n",
        "class K:\n    util = object()\n\n\nx = util.helper()\n",
    ],
)
def test_class_scope_names_only_block_what_python_lets_them_shadow(tmp_path: Path, body: str) -> None:
    assert "function:pkg.util.helper" in _all_dsts(_class_body(tmp_path, body))


def test_unaliased_dotted_import_is_left_to_the_existing_resolver(tmp_path: Path) -> None:
    app = "import pkg.shadowed\n\n\ndef run():\n    pkg.shadowed.f()\n"
    on = _dsts(_shadow(tmp_path, "def shadowed():\n    pass\n", app), "run")
    assert on == _dsts(_shadow(tmp_path / "off", "def shadowed():\n    pass\n", app, v2=False), "run") == ["module:pkg.shadowed"]


@pytest.mark.parametrize("init", ["from pkg.shadowed import shadowed\n", "from .shadowed import shadowed\n", "from .shadowed import f as shadowed\n"])
def test_a_reexport_from_the_submodule_keeps_todays_module_link_and_is_never_upgraded(tmp_path: Path, init: str) -> None:
    """`from .evaluate import evaluate` (stock_analysis workspace/paper): the name is an object defined in that submodule."""
    files_app = "from pkg import shadowed\n\n\ndef run():\n    shadowed(1)\n    shadowed.f()\n"
    on = _dsts(_shadow(tmp_path, init, files_app), "run")
    off = _dsts(_shadow(tmp_path / "off", init, files_app, v2=False), "run")
    assert on == off and all(d == "module:pkg.shadowed" for d in on) and on


# -- audit-panel fixes: no link may reappear, and "cannot tell" must not become a new link --------------------


def test_a_reexport_cannot_bring_back_a_link_the_importing_module_already_rebinds(tmp_path: Path) -> None:
    app = "from pkg import shadowed\n\nshadowed = wrap(shadowed)\n\n\ndef run():\n    shadowed.f()\n"
    assert _dsts(_shadow(tmp_path, "from .shadowed import shadowed\n", app), "run") == []


@pytest.mark.parametrize(
    "init",
    [
        "def shadowed(:\n",                                 # __init__ fails to parse: its bindings are unknown
        "from .shadowed_impl import *\n",                    # a star import may bind any name
        "try:\n    from pkg import shadowed\nexcept ImportError:\n    pass\nfrom .shadowed_impl import *\n",
    ],
)
def test_a_package_init_that_cannot_be_analysed_keeps_todays_link_and_adds_no_upgrade(tmp_path: Path, init: str) -> None:
    files = {"pkg/__init__.py": init, "pkg/shadowed.py": "def f():\n    pass\n", "pkg/shadowed_impl.py": "def shadowed():\n    pass\n", "pkg/app.py": SHADOW_APP}
    on = _dsts(scan(_repo(tmp_path, files), resolve_v2=True), "run")
    off = _dsts(scan(_repo(tmp_path / "off", files), resolve_v2=False), "run")
    assert on == off and "function:pkg.shadowed.f" not in on


def test_a_package_without_an_init_file_is_not_treated_as_unanalysable(tmp_path: Path) -> None:
    files = {"pkg/shadowed.py": "def f():\n    pass\n", "pkg/app.py": SHADOW_APP}
    assert _dsts(scan(_repo(tmp_path, files), resolve_v2=True), "run") == ["function:pkg.shadowed.f"]


def test_class_body_detection_survives_methods_with_nested_functions(tmp_path: Path) -> None:
    body = (
        "class K:\n    def a(self):\n        def inner():\n            return util.helper()\n        return inner\n\n"
        "    def b(self):\n        return util.helper()\n\n    util = object()\n    x = util.helper()\n"
    )
    dsts = _all_dsts(_class_body(tmp_path, body))
    assert dsts == ["function:pkg.util.helper", "function:pkg.util.helper"]  # a/inner and b resolve; the class-body call does not


def test_a_realistic_property_with_a_setter_and_a_plain_neighbour(tmp_path: Path) -> None:
    src = (
        "class C:\n    def a(self):\n        self.size()\n        self.grow()\n\n"
        "    @property\n    def size(self):\n        return self._n\n\n    @size.setter\n    def size(self, value):\n        self._n = value\n\n"
        "    def grow(self):\n        pass\n"
    )
    assert _dsts(_cls(tmp_path, src), "C.a") == ["method:m.C.grow"]
