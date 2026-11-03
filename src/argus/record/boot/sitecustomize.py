"""Starts the call watcher inside a command run by `argus record`.

Python imports a module named `sitecustomize` at start-up, and the runner puts this folder first on PYTHONPATH, so every Python process the
command starts (a test worker, a server child) records too. Standard library only: the command's interpreter may not have Argus installed.
Without ARGUS_RECORD_OUT this file does nothing. A failure to start is printed on stderr and never stops the command.
"""

import atexit
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_recorder = None
_out = ""


def _marker() -> None:
    open(os.path.join(_out, f"{os.getpid()}.start"), "w").close()


def _save() -> None:
    recorder = _recorder
    if recorder is None:
        return
    recorder.stop()
    data = recorder.snapshot()
    target = os.path.join(_out, f"{os.getpid()}.json")
    with open(target + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    os.replace(target + ".tmp", target)  # a half-written file is never mistaken for a saved one


def _after_fork() -> None:
    _recorder.reset_after_fork()
    _marker()


def _start() -> None:
    global _recorder, _out
    out, config = os.environ.get("ARGUS_RECORD_OUT"), os.environ.get("ARGUS_RECORD_CONFIG")
    if not out or not config:
        return
    spec = importlib.util.spec_from_file_location("_argus_monitor", os.path.join(_HERE, "_argus_monitor.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["_argus_monitor"] = module  # registered before it runs, as an import would have done
    spec.loader.exec_module(module)
    with open(config, encoding="utf-8") as handle:
        settings = json.load(handle)
    _out = out
    recorder = module.Recorder(settings["files"], settings["test_files"])
    recorder.start()
    _recorder = recorder
    _marker()
    atexit.register(_save)
    os.register_at_fork(after_in_child=_after_fork)


def _chain() -> None:
    """Run the command's own `sitecustomize`, which this file hides because it comes first on the path."""
    for entry in sys.path:
        if os.path.abspath(entry or ".") == _HERE:
            continue
        candidate = os.path.join(entry or ".", "sitecustomize.py")
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("sitecustomize_user", candidate)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            return


try:
    _start()
except Exception as exc:  # noqa: BLE001 - the recorded command must run even if recording cannot start
    print(f"argus record: could not start recording in this process ({type(exc).__name__}: {exc})", file=sys.stderr)
_chain()
