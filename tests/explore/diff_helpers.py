"""Helpers for the Diff Explorer v0.1 tests (written first, before any implementation exists).

CONTRACT the tests assume (Stage 4 should match these names and strings; the design source is
docs/HANDOFF_NEXT_SESSION.md section 4):

argus.explore.sides
    DiffError(Exception)
    Side(label, root, files, untracked): frozen dataclass. `files` is a sorted tuple of POSIX paths (relative to
    Side.root, which mirrors the scan PATH) of regular, tracked-or-untracked-not-ignored .py files only. `untracked` is a
    frozenset subset of `files` (only ever non-empty for the working-tree head side). Labels: "<ref> (<sha7>)" for a
    commit (for example "HEAD (3ad8b6b)") and "working tree" for the head when head is None.
    materialize_sides(path, base, head=None): context manager yielding (base_side, head_side); only module that runs git.

argus.explore.changes
    Flag(StrEnum): SIGNATURE_CHANGED, CALLS_CHANGED, BODY_CHANGED (value == name). BODY_CHANGED is structural change detection
        (the syntax tree differs), never evidence that behavior changed.
    SymbolChange(qualname, kind, change, flags, claim, calls_claim, untracked, file, line, interface_was, interface_now, docstring_only)
        kind: node kind value ("function", "method", "class"); change: "added" | "removed" | "modified";
        flags: frozenset[Flag]; claim: FACT Claim (added/removed/SIGNATURE_CHANGED/BODY_CHANGED), None when CALLS_CHANGED is the only flag;
        calls_claim: INFERENCE Claim, present iff CALLS_CHANGED in flags (never printed on a FACT line).
        file, line: head location (base location for a removed symbol). interface_was / interface_now: decorators, async marker and
        signature text, set iff SIGNATURE_CHANGED. docstring_only: BODY_CHANGED caused only by a docstring edit.
    CallChange(change, source, target, claim): a resolved call present on one side only, compared for functions present on both sides.
    ImportChange(change, source, target, claim): a subsystem import added or removed.
    CallerEdge(callee, callee_label, claim, was, confidence, override_possible): claim is an INFERENCE Claim with one Evidence (call
        site); callee_label: "REMOVED" or the callee's SIGNATURE_CHANGED / BODY_CHANGED flags joined by ", "; was: True when evidence is
        from the base side; confidence: the scanned edge's confidence, never upgraded; override_possible: the "via self" note applies.
    CallerGroup(caller, module_level, caller_state, is_test, edges): caller_state is "unchanged" | "changed" | "added" | "removed"
        ("added": a new function that calls a changed or removed symbol).
    DiffView(base_label, head_label, resolve_v2, files_changed, untracked_files, symbols, calls, interface_callers,
             affected_callers, hidden_test_callers, imports, not_determinable) and a method claims() returning every
             Claim in the view (symbols, calls, callers, imports, not_determinable). hidden_test_callers counts caller groups.
    build_diff_view(base, head, base_model, head_model, resolve_v2, include_tests) -> DiffView
    The view must be self-contained: it holds no Path into a temp directory.

argus.explore.render_diff
    print_diff(console, view, limit): limit None means unlimited. CLI: `argus explore diff [PATH] [--base REF]
    [--head REF] [--include-tests] [--limit N | --all]`; DiffError, bad repo or ref, `--all` with `--limit` and
    `--limit 0` all exit with code 2.

Rendered text (all section titles start at column 0, everything inside a section is indented):
    line 1 "Diff  <base label> -> <head label>   N files changed (M untracked)   calls-v2 on both sides" ("calls-v1" when
    V2 is off). Sections, in order, each omitted when it has no rows: "Symbols (N)", "Calls (N)",
    "Direct callers of changed interfaces", "Potentially affected", "Subsystem imports", and always last
    "Not determinable". A tree with nothing to report prints "no structural change".
    Symbols row:  "  [FACT] ~ SIGNATURE_CHANGED, BODY_CHANGED  <qualname>   <file>:<line>" (continuation line
        "was (amount)   now (amount, currency)"), "  [FACT] + function <qualname>   <file>:<line>" (adds
        "(untracked file)" for untracked files), "  [FACT] - method <qualname>   was <file>:<line>".
        CALLS_CHANGED is its own row: "  [INFERENCE] ~ CALLS_CHANGED  <qualname>   <file>:<line>".
        A BODY_CHANGED row of a docstring-only edit carries the words "syntactic change".
    Calls row:    "  [INFERENCE] + <src> -> <dst>   <file>:<line>" or "  [INFERENCE] - <src> -> <dst>   was <file>:<line>".
    Caller group: "  <caller qualname> [(module level)]   [(caller also changed) | (caller added) | (caller removed)]"
        ("(caller still exists, unchanged)" is printed for unchanged callers of a REMOVED callee only) then edges "    [INFERENCE] calls <callee> [<LABEL>]   <file>:<line>   INFERRED   hash <16 hex>"
        (+ "  via self; a subclass may override it" when override_possible). Edges of a removed callee read
        "called <callee> [REMOVED]   was <file>:<line>   INFERRED   hash <16 hex> (before-snapshot)".
    Overflow: "  +N more" inside the section; "  N test caller groups hidden" for hidden test callers.
    Not determinable rows start with "[UNKNOWN]".
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from typer.testing import CliRunner, Result

from argus.cli import app
from argus.explore.claims import SOURCE_HASH_BYTES

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "author@example.invalid",
    "GIT_COMMITTER_NAME": "Test Committer",
    "GIT_COMMITTER_EMAIL": "committer@example.invalid",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+0000",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+0000",
}

CART = (
    "from shop.billing import charge\n\n\n"
    "def checkout(amount):\n    return charge(amount)\n\n\n"
    "def checkout_twice(amount):\n    charge(amount)\n    return charge(amount)\n"
)
API = "from shop.cart import checkout\n\n\ndef view(amount):\n    return checkout(amount)\n"
BILLING = "def charge(amount):\n    return amount + 1\n\n\ndef fee(amount):\n    return amount\n"
SHOP_FILES = {"shop/__init__.py": "", "shop/billing.py": BILLING, "shop/cart.py": CART, "shop/api.py": API}


FEE_BILLING = "def charge(amount):\n    return amount + 1\n\n\ndef fee(amount):\n    return amount\n"
FEE_CART = "from shop.billing import fee\n\n\ndef checkout(a):\n    return fee(a)\n"
FEE_REPORTS = "from shop.billing import fee\n\n\ndef monthly(a):\n    return fee(a) + 1\n"
FEE_LEGACY = "from shop.billing import fee\n\n\ndef old_report(a):\n    return fee(a)\n"


class GitRepo:
    """A real git repository in a temp dir, driven by scripted commands (no global or system git config)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=False, timeout=60, env={**os.environ, **GIT_ENV},
        )
        assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr}"
        return proc.stdout

    def write(self, rel: str, content: str | bytes) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def write_files(self, files: dict[str, str]) -> None:
        for rel, content in files.items():
            self.write(rel, content)

    def delete(self, rel: str) -> None:
        (self.root / rel).unlink()

    def stage(self, *rels: str) -> None:
        self.git("add", "--", *rels)

    def commit(self, message: str = "commit") -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.sha()

    def sha(self, ref: str = "HEAD") -> str:
        return self.git("rev-parse", ref).strip()

    def short(self, ref: str = "HEAD") -> str:
        return self.sha(ref)[:7]

    def symlink(self, rel: str, target: str) -> None:
        link = self.root / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)

    def add_gitlink(self, rel: str) -> None:
        """Track `rel` as a submodule entry (mode 160000) without needing a real submodule."""
        self.git("update-index", "--add", "--cacheinfo", f"160000,{'a' * 40},{rel}")


def remove_fee_scenario(repo: GitRepo) -> None:
    """Base: three callers of billing.fee (cart.checkout, reports.monthly, legacy.old_report). Working tree: fee is removed
    from billing.py, reports.monthly is edited (still calls fee) and legacy.py is deleted. cart.py is untouched."""
    repo.write_files({
        "shop/__init__.py": "", "shop/billing.py": FEE_BILLING, "shop/cart.py": FEE_CART, "shop/reports.py": FEE_REPORTS,
        "shop/legacy.py": FEE_LEGACY,
    })
    repo.commit("base")
    repo.write("shop/billing.py", "def charge(amount):\n    return amount + 1\n")
    repo.write("shop/reports.py", FEE_REPORTS.replace("+ 1", "+ 2"))
    repo.delete("shop/legacy.py")


def source_hash(content: str | bytes) -> str:
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.blake2b(data, digest_size=SOURCE_HASH_BYTES).hexdigest()


def run_diff(repo: GitRepo | Path, *args: str, env: dict[str, str] | None = None) -> Result:
    path = repo.root if isinstance(repo, GitRepo) else repo
    return CliRunner().invoke(app, ["explore", "diff", str(path), *args], env=env)


def diff_out(repo: GitRepo | Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = run_diff(repo, *args, env=env)
    assert result.exit_code == 0, result.output
    return result.output


# -- reading rendered output ---------------------------------------------------


def section(out: str, title: str) -> str:
    """Body of the section whose title line starts with `title` ('' when the section is absent)."""
    lines = out.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(title)), None)
    if start is None:
        return ""
    body: list[str] = []
    for line in lines[start + 1:]:
        if line.strip() and not line.startswith(" "):
            break
        body.append(line)
    return "\n".join(body)


def before_notes(out: str) -> str:
    """Everything above the Not determinable section (that section mentions every flag name in prose)."""
    return out.split("\nNot determinable", 1)[0]


def name_re(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w.]){re.escape(name)}(?![\w.])")


def symbol_lines(out: str, qualname: str) -> list[str]:
    return [line for line in section(out, "Symbols").splitlines() if name_re(qualname).search(line)]


def flags_in(lines: list[str]) -> set[str]:
    return set(re.findall(r"\b(SIGNATURE_CHANGED|CALLS_CHANGED|BODY_CHANGED)\b", "\n".join(lines)))


EDGE_RE = re.compile(
    r"^\[(?P<tier>\w+)\] (?P<verb>calls|called) (?P<callee>\S+) \[(?P<label>[A-Z_, ]+)\]\s+(?P<was>was )?(?P<loc>\S+:\d+)\s+"
    r"(?P<conf>[A-Z]+)\s+hash (?P<hash>[0-9a-f]{16})(?P<rest>.*)$"
)


@dataclass(frozen=True)
class ParsedEdge:
    verb: str
    callee: str
    label: str
    was: bool
    location: str
    confidence: str
    hash: str
    rest: str
    raw: str


@dataclass(frozen=True)
class ParsedGroup:
    name: str
    header: str
    edges: tuple[ParsedEdge, ...]


def parse_groups(body: str) -> list[ParsedGroup]:
    """Caller groups of a caller section: a 2-space line starting with a name, then 4-space edge lines."""
    groups: list[tuple[str, str, list[ParsedEdge]]] = []
    for line in body.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if not stripped:
            continue
        if indent == 2 and re.match(r"[A-Za-z_]", stripped):
            groups.append((stripped.split()[0], stripped, []))
        elif indent >= 4 and groups:
            match = EDGE_RE.match(stripped)
            assert match, f"unparseable caller edge line: {line!r}"
            groups[-1][2].append(ParsedEdge(
                match["verb"], match["callee"], match["label"], bool(match["was"]), match["loc"], match["conf"], match["hash"],
                match["rest"], stripped,
            ))
    return [ParsedGroup(name, header, tuple(edges)) for name, header, edges in groups]


def direct_groups(out: str) -> list[ParsedGroup]:
    return parse_groups(section(out, "Direct callers of changed interfaces"))


def affected_groups(out: str) -> list[ParsedGroup]:
    return parse_groups(section(out, "Potentially affected"))


def group_names(groups: list[ParsedGroup]) -> list[str]:
    return [g.name for g in groups]


CALL_ROW_RE = re.compile(r"\[INFERENCE\] (?P<sign>[+-]) (?P<src>\S+) -> (?P<dst>\S+)\s+(?:was )?(?P<file>\S+?):(?P<line>\d+)")


def call_rows(out: str) -> list[tuple[str, str, str, str, int]]:
    rows = []
    for line in section(out, "Calls").splitlines():
        match = CALL_ROW_RE.search(line)
        if match:
            rows.append((match["sign"], match["src"], match["dst"], match["file"], int(match["line"])))
    return rows


def not_determinable(out: str) -> str:
    return section(out, "Not determinable")
