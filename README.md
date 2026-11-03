# Argus

Control and understanding layer for coding agents (Python first): an evidence-backed RepoGraph, targeted context selection, and verification, benchmarked against vanilla Aider.

Argus extracts a **provenance-tracked, content-addressed architecture model** from Python source — packages, modules, classes, functions, contracts, and the edges between them — so a human can see what an AI agent actually changed at the structural level, not the line level.

**Status:** RepoGraph extractor (plan Phase 1, partial) and structural diff/health checks exist. The benchmark harness (Phase 0), retrieval, and executor are next. The web explorer was pruned; see the handoff.

# Architecture rules

Argus is an architecture analysis tool.

Prefer:
- deterministic extraction over LLM inference
- explicit provenance for every fact
- confidence levels for uncertain edges
- tests for every extractor change

Never:
- silently invent architecture relationships
- remove ambiguity tracking
- weaken validation to make tests pass
---

## Capabilities

### 1. Static Architecture Extraction (Phase 0)
* **AST Parsing**: Extracts packages, modules, classes, functions, methods, attributes, and constants using the Python standard library `ast`.
* **Public Surface & Signatures**: Detects `__all__` exports and naming conventions (`_private`), type-hinted signatures, parameter defaults, and `async` status.
* **Smart Dependency Resolution**:
  * Distinguishes **runtime imports** from **type-only imports** (`if TYPE_CHECKING:`).
  * Resolves lazy imports inside function bodies, relative imports, and star imports.
  * Canonicalizes re-exports through `__init__.py` back to their defining modules.
* **Provenance & Confidence Tracking**: Every node and edge records exact `file:line` locations, extractor details, and confidence levels (`EXTRACTED`, `INFERRED`, `AMBIGUOUS`).

### 2. Explorer: understand a repo
* `argus explore sample`: the edge-quality gates (AMBIGUOUS rate, plus a reproducible sample of edges to check by hand).
* `argus explore map`: system and dependency maps (ASCII), plus a ranked list. `entrypoints`, `component`, `symbol`: where a command starts, what a package holds, who calls a function.
* Every claim is tagged FACT / INFERENCE / UNKNOWN with `file:line` evidence and a source hash. Unknown stays unknown.

### 3. Diff Explorer: review a change
* `argus explore diff`: what changed in structure between two states of a git repo (default: HEAD against the working tree), and who calls the changed code. Reports added, removed and modified symbols (interface, body, calls), files changed outside functions, pure file renames, and direct callers of changed interfaces.
* `--base REF` compares from the merge-base with `--head REF`, like a pull request. Read-only, about 2 seconds, nothing stored.

### 4. Terminal UI: browse it live
* `argus` with no command opens a full-screen terminal UI on the repository (`-C PATH` to pick one, `--base REF` to compare against a commit, `--include-tests` to count test callers). It repaints when a Python file changes.
* Four views, one key each: **1** Changes (what the agent changed, and who calls it), **2** Map (level map, or the blast-radius tree with **t**), **3** Symbol (one function, class or method with its callers), **4** Diff (base and head refs, set with **b** and **h**).
* **d** long report of the changes · **/** search · **e** open the editor at the highlighted line · **r** start fresh from the current state · **?** help · **q** quit.
* **R** records the repository's tests and fills in the "Seen running" lines (dynamic tracing). **C** runs `argus check` on your uncommitted change and shows which tests newly fail. Both ask first, because they execute your code; use them only on a repository you wrote. They need Python 3.12 or newer with the repository's packages installed; set `ARGUS_PYTHON=/path/to/python` (or write the path in `.argus/python` in the repository) to choose which Python runs them.
* **s** types a note about the highlighted change into the Claude Code pane of your tmux window. It never presses Enter, so you review and send it yourself.

---

## Installation

```bash
uv venv && uv pip install -e .
```

---

## Usage & CLI Commands

```bash
# Maps of a repo (system view, dependency layers, ranked list)
argus explore map path/to/repo --view system
argus explore map path/to/repo --view deps --open NAME

# Entry points, one component, one symbol with its callers
argus explore entrypoints path/to/repo
argus explore component NAME path/to/repo
argus explore symbol QUERY path/to/repo

# Review a change: working tree vs HEAD, one commit, or a branch against main
argus explore diff path/to/repo
argus explore diff path/to/repo --base abc123~1 --head abc123
argus explore diff path/to/repo --base main --head my-branch

# Terminal UI (the default when no command is given)
argus
argus -C path/to/repo --base main

# Run a command and record which functions ran; run the tests your change reaches
argus record -C path/to/repo -- pytest -q
argus check path/to/repo

# Phase 0 benchmark harness
argus bench --help
```

The earlier `scan`, `snapshots`, `health` and `diff` commands, with their SQLite store and health analysis, were removed on 2026-10-21 (recoverable from git history before that date). `explore sample` replaces the old `sample` command.

---

## Quality Gates

| Gate | Target | How it is measured |
|---|---|---|
| Precision on `EXTRACTED` edges | $\ge$ 95% | `argus explore sample` draws a reproducible random sample; you check each edge by hand |
| `AMBIGUOUS` edge rate | $\le$ 15% | Computed by `argus explore sample`, which exits 1 when the rate is above the limit |

Both matter. Precision alone is trivially gamed by dumping everything into `AMBIGUOUS`; tracking coverage alongside it keeps the model honest.

---

## Roadmap

Short version: benchmark baseline, RepoGraph completion (method-call resolution, test links, query API), structural context selector, Aider executor, then the first end-to-end experiment as a decision gate before any semantic layer is built.

The previous GUI/community/PR-comment code is recoverable at git tag `pre-benchmark-prune`.
