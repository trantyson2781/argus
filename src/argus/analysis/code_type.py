"""Classifies every node as production, test, generated, external, configuration, or unknown code.

Argus Production vs Test Architecture Separation.md: the default architecture view should
answer "how does the software system work?", not "how is the repository structured?" —
which requires knowing which nodes are verification evidence (tests) rather than runtime
architecture before any other analysis (communities, health, contracts, diffs) runs.

Classification is a pure function of the file path, so it is recomputed on read, never stored.
"""

from __future__ import annotations

import re
from enum import StrEnum

_TEST_DIR_NAMES = frozenset({"tests", "test"})
_TEST_FILE_RE = re.compile(r"^(test_.*|.*_test)\.py$")
_CONFTEST = "conftest.py"

_GENERATED_DIR_NAMES = frozenset({"migrations", "generated", "_generated", "__generated__"})
_GENERATED_FILE_RE = re.compile(r"^generated_.*\.py$|.*_pb2(_grpc)?\.py$")

_CONFIGURATION_FILE_NAMES = frozenset({"settings.py", "config.py", "conf.py"})
_CONFIGURATION_DIR_NAMES = frozenset({"settings"})


class CodeType(StrEnum):
    """What role a node plays in the codebase, independent of its structural kind."""

    PRODUCTION = "production"
    TEST = "test"
    GENERATED = "generated"
    EXTERNAL = "external"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


def classify_file(file_path: str) -> CodeType:
    """Classify a file by path convention alone (Doc: "Code Classification")."""
    if not file_path:
        return CodeType.UNKNOWN

    parts = file_path.replace("\\", "/").split("/")
    filename = parts[-1]
    dirs = parts[:-1]

    if filename == _CONFTEST or any(part in _TEST_DIR_NAMES for part in dirs) or _TEST_FILE_RE.match(filename):
        return CodeType.TEST

    if any(part in _GENERATED_DIR_NAMES for part in dirs) or _GENERATED_FILE_RE.match(filename):
        return CodeType.GENERATED

    if filename in _CONFIGURATION_FILE_NAMES or any(part in _CONFIGURATION_DIR_NAMES for part in dirs):
        return CodeType.CONFIGURATION

    return CodeType.PRODUCTION


def is_production(code_type: CodeType) -> bool:
    return code_type is CodeType.PRODUCTION
