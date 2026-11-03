"""Web routes (FastAPI, Flask) are entry points like Typer commands: `@app.get("/pay")` is read from the decorator, nothing is imported or run.

Rules under test: a route is found only on a variable Argus saw being built by `FastAPI()`, `APIRouter()`, `Flask()` or `Blueprint()`; the
displayed path is the router's own prefix plus the decorator path, plus the prefix a literal `include_router(..., prefix="/x")` or
`register_blueprint(..., url_prefix="/x")` adds; a prefix Argus cannot read is named as UNKNOWN and never guessed; routes are kept apart from
CLI commands, so every CLI-only project reads exactly as before; "reached from" counts routes too.
"""

from __future__ import annotations

from pathlib import Path

from argus.explore.claims import Tier
from argus.explore.entrypoints import find_entrypoints
from argus.explore.symbols import build_symbol_view
from argus.extract import scan


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _routes(root: Path) -> dict[str, str]:
    """display -> function id"""
    return {r.display: r.function_id for r in find_entrypoints(root, scan(root)).routes}


FASTAPI = "from fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get('/pay')\ndef pay():\n    return 1\n\n\n@app.post('/refund')\nasync def refund():\n    return 2\n"


def test_fastapi_methods_are_read_from_the_decorators(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": FASTAPI})

    assert _routes(tmp_path) == {"GET /pay": "function:p.web.pay", "POST /refund": "function:p.web.refund"}


def test_a_route_is_a_fact_with_the_decorator_line_as_evidence(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": FASTAPI})

    route = next(r for r in find_entrypoints(tmp_path, scan(tmp_path)).routes if r.display == "GET /pay")

    assert route.claim.tier is Tier.FACT and route.claim.evidence[0].line == 6 and route.file == "p/web.py" and route.line == 7


def test_routes_do_not_change_the_cli_commands_of_a_project(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": FASTAPI})

    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert eps.commands == () and len(eps.routes) == 2 and eps.routes[0].claim in eps.all_claims()


def test_a_router_prefix_is_part_of_the_path(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/auth.py": "from fastapi import APIRouter\n\nrouter = APIRouter(prefix='/auth', tags=['a'])\n\n\n@router.post('/login')\ndef login():\n    pass\n"})

    assert _routes(tmp_path) == {"POST …/auth/login": "function:p.auth.login"}  # nobody mounts it where Argus can see: the front is unknown


def test_a_literal_include_prefix_is_added_in_front(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "",
        "p/auth.py": "from fastapi import APIRouter\n\nrouter = APIRouter(prefix='/auth')\n\n\n@router.post('/login')\ndef login():\n    pass\n",
        "p/main.py": "from fastapi import FastAPI\nfrom p import auth\n\napp = FastAPI()\napp.include_router(auth.router, prefix='/api')\n",
    })

    assert _routes(tmp_path) == {"POST /api/auth/login": "function:p.auth.login"}


def test_a_prefix_Argus_cannot_read_is_named_unknown_and_not_guessed(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "",
        "p/auth.py": "from fastapi import APIRouter\n\nrouter = APIRouter(prefix='/auth')\n\n\n@router.post('/login')\ndef login():\n    pass\n",
        "p/main.py": "from fastapi import FastAPI\nfrom p import auth\n\nPREFIX = '/v' + '1'\napp = FastAPI()\napp.include_router(auth.router, prefix=PREFIX)\n",
    })
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert [r.display for r in eps.routes] == ["POST …/auth/login"]  # the dots say the URL starts with something unread
    assert any(c.tier is Tier.UNKNOWN and "prefix" in c.text and "auth.router" in c.text for c in eps.unknowns)


def test_a_router_included_twice_is_listed_twice(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "",
        "p/auth.py": "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n@router.get('/me')\ndef me():\n    pass\n",
        "p/main.py": "from fastapi import FastAPI\nfrom p import auth\n\napp = FastAPI()\napp.include_router(auth.router, prefix='/v1')\napp.include_router(auth.router, prefix='/v2')\n",
    })

    assert sorted(_routes(tmp_path)) == ["GET /v1/me", "GET /v2/me"]


def test_flask_route_with_methods_default_get_and_method_shortcuts(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": (
        "from flask import Flask\n\napp = Flask(__name__)\n\n\n@app.route('/a', methods=['GET', 'POST'])\ndef a():\n    pass\n\n\n"
        "@app.route('/b')\ndef b():\n    pass\n\n\n@app.get('/c')\ndef c():\n    pass\n"
    )})

    assert _routes(tmp_path) == {"GET /a": "function:p.web.a", "POST /a": "function:p.web.a", "GET /b": "function:p.web.b", "GET /c": "function:p.web.c"}


def test_flask_blueprint_prefix_and_the_register_prefix_that_overrides_it(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "",
        "p/bp.py": "from flask import Blueprint\n\nbp = Blueprint('b', __name__, url_prefix='/b')\n\n\n@bp.route('/x')\ndef x():\n    pass\n",
        "p/app.py": "from flask import Flask\nfrom p import bp\n\napp = Flask(__name__)\napp.register_blueprint(bp.bp, url_prefix='/z')\n",
    })

    assert _routes(tmp_path) == {"GET /z/x": "function:p.bp.x"}


def test_a_blueprint_prefix_alone_counts_when_it_is_never_registered_where_argus_can_see(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/bp.py": "from flask import Blueprint\n\nbp = Blueprint('b', __name__, url_prefix='/b')\n\n\n@bp.route('/x')\ndef x():\n    pass\n"})

    assert _routes(tmp_path) == {"GET …/b/x": "function:p.bp.x"}


def test_a_computed_path_is_unknown_and_not_placed(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": "from fastapi import FastAPI\n\nPATH = '/x'\napp = FastAPI()\n\n\n@app.get(PATH)\ndef x():\n    pass\n"})
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert eps.routes == () and any(c.tier is Tier.UNKNOWN and "p.web.x" in c.text for c in eps.unknowns)


def test_a_get_decorator_on_something_that_is_not_a_web_app_is_not_a_route(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": "from cache import store\n\n\n@store.get(some_key)\ndef x():\n    pass\n"})

    assert find_entrypoints(tmp_path, scan(tmp_path)).routes == ()


def test_a_literal_route_on_an_app_argus_cannot_see_is_reported_unknown(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": "from other import app\n\n\n@app.get('/x')\ndef x():\n    pass\n"})
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert eps.routes == () and any(c.tier is Tier.UNKNOWN and "app.get" in c.text for c in eps.unknowns)


def test_apps_in_test_code_are_not_reported(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "tests/test_web.py": FASTAPI})

    assert _routes(tmp_path) == {}


# -- reached from ------------------------------------------------------------------------------------------------------


def _symbol(root: Path, node_id: str):
    model = scan(root)
    node = next(n for n in model.nodes if n.id == node_id)
    return build_symbol_view(model, root, node)


def test_a_function_called_by_a_route_is_reached_from_that_route(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "",
        "p/core.py": "def charge(x):\n    return x\n",
        "p/web.py": "from fastapi import FastAPI\nfrom p.core import charge\n\napp = FastAPI()\n\n\n@app.post('/pay')\ndef pay():\n    return charge(1)\n",
    })

    view = _symbol(tmp_path, "function:p.core.charge")

    assert [r.command for r in view.reached_from] == ["POST /pay"] and view.commands_found == 1


def test_a_route_handler_is_reached_from_its_own_route(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": FASTAPI})

    assert [r.command for r in _symbol(tmp_path, "function:p.web.pay").reached_from] == ["GET /pay"]


def test_the_symbol_text_says_reached_from_when_a_route_is_involved(tmp_path: Path) -> None:
    from argus.explore.render_symbol import print_symbol  # noqa: F401 - the label is checked through the rows below
    from argus.explore.render_symbol import _reach_rows

    _write(tmp_path, {"p/__init__.py": "", "p/web.py": FASTAPI})

    assert _reach_rows(_symbol(tmp_path, "function:p.web.pay"))[0][0] == "Reached from"


# -- found by checking against a real repository (oquizy) ------------------------------------------------------------------


def test_an_empty_path_is_the_prefix_itself(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/stats.py": "from fastapi import APIRouter\n\nrouter = APIRouter(prefix='/stats')\n\n\n@router.get('')\ndef stats():\n    pass\n"})

    assert _routes(tmp_path) == {"GET …/stats": "function:p.stats.stats"}


def test_a_mount_is_resolved_when_the_repository_keeps_its_packages_below_a_source_folder(tmp_path: Path) -> None:
    _write(tmp_path, {
        "apps/api/src/pkg/__init__.py": "",
        "apps/api/src/pkg/routers/__init__.py": "",
        "apps/api/src/pkg/routers/auth.py": "from fastapi import APIRouter\n\nrouter = APIRouter(prefix='/auth')\n\n\n@router.post('/login')\ndef login():\n    pass\n",
        "apps/api/src/pkg/main.py": "from fastapi import FastAPI\nfrom pkg.routers import auth\n\napp = FastAPI()\napp.include_router(auth.router, prefix='/v1')\n",
    })

    assert list(_routes(tmp_path)) == ["POST /v1/auth/login"]


def test_a_mount_inside_a_function_with_a_computed_prefix_is_named_unknown(tmp_path: Path) -> None:
    _write(tmp_path, {
        "apps/api/src/pkg/__init__.py": "",
        "apps/api/src/pkg/auth.py": "from fastapi import APIRouter\n\nrouter = APIRouter(prefix='/auth')\n\n\n@router.post('/login')\ndef login():\n    pass\n",
        "apps/api/src/pkg/main.py": "from fastapi import FastAPI\nfrom pkg import auth\n\n\ndef create(settings):\n    app = FastAPI()\n    app.include_router(auth.router, prefix=settings.v1)\n    return app\n",
    })
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert [r.display for r in eps.routes] == ["POST …/auth/login"]
    assert any(c.tier is Tier.UNKNOWN and "settings.v1" in c.text for c in eps.unknowns)


def test_a_mount_whose_router_cannot_be_found_is_named_unknown(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/main.py": "from fastapi import FastAPI\nfrom elsewhere import users\n\napp = FastAPI()\napp.include_router(users.router, prefix='/u')\n"})
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert any(c.tier is Tier.UNKNOWN and "users.router" in c.text and "cannot be resolved" in c.text for c in eps.unknowns)


def test_explore_entrypoints_lists_the_routes(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from argus.cli import app

    _write(tmp_path, {"p/__init__.py": "", "p/web.py": FASTAPI})

    result = CliRunner().invoke(app, ["explore", "entrypoints", str(tmp_path)])

    assert result.exit_code == 0 and "GET /pay" in result.output and "POST /refund" in result.output and "p.web.pay" in result.output


# -- found by the audit ----------------------------------------------------------------------------------------------------


def test_a_sub_application_mounted_with_mount_gets_its_path(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/sub.py": "from fastapi import FastAPI\n\nsub = FastAPI()\n\n\n@sub.get('/ping')\ndef ping():\n    pass\n",
                      "p/main.py": "from fastapi import FastAPI\nfrom p import sub\n\napp = FastAPI()\napp.mount('/sub', sub.sub)\n"})

    assert list(_routes(tmp_path)) == ["GET /sub/ping"]


def test_mounting_static_files_is_not_a_router_and_makes_no_noise(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/main.py": "from fastapi import FastAPI\nfrom fastapi.staticfiles import StaticFiles\n\napp = FastAPI()\napp.mount('/static', StaticFiles(directory='s'))\n"})

    assert find_entrypoints(tmp_path, scan(tmp_path)).unknowns == ()


def test_a_router_nobody_mounts_where_argus_can_see_has_an_unknown_front_and_says_so(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/items.py": "from fastapi import APIRouter\n\nrouter = APIRouter(prefix='/items')\n\n\n@router.get('/{id}')\ndef one(id):\n    pass\n"})
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert [r.display for r in eps.routes] == ["GET …/items/{id}"]
    assert any(c.tier is Tier.UNKNOWN and "p.items:router" in c.text and "not mounted" in c.text for c in eps.unknowns)


def test_a_mount_with_unpacked_keywords_has_an_unknown_prefix(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "",
        "p/auth.py": "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n@router.post('/login')\ndef login():\n    pass\n",
        "p/main.py": "from fastapi import FastAPI\nfrom p import auth\n\nOPTS = {}\napp = FastAPI()\napp.include_router(auth.router, **OPTS)\n",
    })

    assert [r.display for r in find_entrypoints(tmp_path, scan(tmp_path)).routes] == ["POST …/login"]


def test_a_suffix_match_is_only_accepted_below_a_source_folder(tmp_path: Path) -> None:
    # `api` may be an installed package, not the repository's `svc/api.py`
    _write(tmp_path, {
        "svc/__init__.py": "",
        "svc/api.py": "from flask import Blueprint\n\nbp = Blueprint('b', __name__)\n\n\n@bp.route('/who')\ndef who():\n    pass\n",
        "svc/main.py": "from flask import Flask\nfrom api import bp as third\n\napp = Flask(__name__)\napp.register_blueprint(third, url_prefix='/x')\n",
    })
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert [r.display for r in eps.routes] == ["GET …/who"]
    assert any("cannot be resolved" in c.text for c in eps.unknowns)


def test_the_same_route_reached_through_two_branches_is_listed_once(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "",
        "p/bp.py": "from flask import Blueprint\n\nbp = Blueprint('b', __name__)\n\n\n@bp.route('/login', methods=['POST'])\ndef login():\n    pass\n",
        "p/app.py": "from flask import Flask\nfrom p import bp\nimport os\n\napp = Flask(__name__)\nif os.environ.get('A'):\n    app.register_blueprint(bp.bp, url_prefix='/a')\nelse:\n    app.register_blueprint(bp.bp, url_prefix='/a')\n",
    })

    assert [r.display for r in find_entrypoints(tmp_path, scan(tmp_path)).routes] == ["POST /a/login"]


def test_a_route_on_an_app_imported_from_another_module_is_placed(tmp_path: Path) -> None:
    _write(tmp_path, {
        "p/__init__.py": "",
        "p/main.py": "from flask import Flask\n\napp = Flask(__name__)\n",
        "p/views.py": "from p.main import app\n\n\n@app.get('/v1')\ndef v1():\n    pass\n",
    })
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert [r.display for r in eps.routes] == ["GET /v1"] and eps.unknowns == ()


def test_many_routes_on_an_app_argus_cannot_see_make_one_note_per_file(tmp_path: Path) -> None:
    body = "from other import app\n\n\n" + "".join(f"@app.get('/r{i}')\ndef r{i}():\n    pass\n\n\n" for i in range(5))
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": body})

    unknowns = find_entrypoints(tmp_path, scan(tmp_path)).unknowns

    assert len([c for c in unknowns if "app.get" in c.text or "web app" in c.text]) == 1 and "5" in unknowns[0].text


def test_registrations_that_are_not_read_are_named_not_dropped(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": (
        "from flask import Flask\n\n\ndef create():\n    app = Flask(__name__)\n\n    @app.get('/inner')\n    def inner():\n        pass\n\n"
        "    app.add_url_rule('/rule', view_func=inner)\n    return app\n"
    )})
    unknowns = find_entrypoints(tmp_path, scan(tmp_path)).unknowns

    assert any(c.tier is Tier.UNKNOWN and "p/web.py" in c.text and "not read" in c.text for c in unknowns)


def test_a_route_with_an_empty_methods_list_is_not_placed_and_is_named(tmp_path: Path) -> None:
    _write(tmp_path, {"p/__init__.py": "", "p/web.py": "from flask import Flask\n\napp = Flask(__name__)\n\n\n@app.route('/x', methods=[])\ndef x():\n    pass\n"})
    eps = find_entrypoints(tmp_path, scan(tmp_path))

    assert eps.routes == () and any("p.web.x" in c.text for c in eps.unknowns)
