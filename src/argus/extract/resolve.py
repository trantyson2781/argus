"""Resolve import statements to the modules and symbols they actually name.

Two jobs, in order of importance:

1. **Re-export canonicalization.** `from myapp import Invoice`, where
   `myapp/__init__.py` does `from .billing.invoice import Invoice`, is a
   dependency on `myapp.billing.invoice` — not on the package root. Without this
   every consumer collapses onto the package root and the real structure
   disappears (PYTHON-EXTRACTION §3.1).

2. **Relative import resolution**, including PEP 420 namespace packages, where
   walking up for `__init__.py` does not work.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from argus.model.schema import Confidence

if TYPE_CHECKING:
    from argus.extract.discovery import SourceFile
    from argus.extract.facts import FileFacts, RawImport, RawSymbol

MAX_ALIAS_HOPS = 16
"""Guard against import cycles between package inits."""

STDLIB_MODULES = frozenset(sys.stdlib_module_names)
"""Never let a local file with a stdlib name capture a stdlib import."""


@dataclass(frozen=True, slots=True)
class Target:
    """What an import statement resolves to."""

    ref: str                    # full dotted path as resolved
    module: str | None          # internal module qualname, if internal
    symbol: str | None          # symbol qualname within that module, if a symbol
    is_internal: bool
    confidence: Confidence
    canonicalized_from: str | None = None  # the re-export path we saw through
    resolved_via: str | None = None        # non-obvious route taken, for provenance

    @property
    def qualname(self) -> str:
        if self.module and self.symbol:
            return f"{self.module}.{self.symbol}"
        return self.module or self.ref


@dataclass(frozen=True, slots=True)
class _Alias:
    """A name bound in a module by an import, before canonicalization."""

    module_ref: str        # dotted path of the module it came from
    name: str              # original name in that module ("*" or a symbol name)
    is_module_import: bool
    line: int


class SymbolTable:
    """Index over all discovered files, used to resolve names to definitions."""

    def __init__(self, files: tuple[SourceFile, ...], facts: dict[str, FileFacts]) -> None:
        self._facts = facts
        self._is_package_init = {f.qualname: f.is_package_init for f in files}
        self.modules: frozenset[str] = frozenset(facts)
        self._top_level: dict[str, frozenset[str]] = {
            qualname: frozenset(
                symbol.name for symbol in fact.symbols if symbol.parent_local is None
            )
            for qualname, fact in facts.items()
        }
        self._aliases: dict[str, dict[str, _Alias]] = {
            qualname: self._alias_map(qualname, fact) for qualname, fact in facts.items()
        }
        self._by_suffix = self._suffix_index()

    def _suffix_index(self) -> dict[str, tuple[str, ...]]:
        """Map every proper dotted suffix of a module name back to that module.

        Scripts that do `sys.path.insert(0, "eval")` import `eval/run_eval.py` as
        plain `run_eval`. Argus cannot execute the path mutation, but it can see
        that exactly one discovered module ends with that name.
        """
        index: dict[str, list[str]] = defaultdict(list)
        for qualname in self.modules:
            parts = qualname.split(".")
            for start in range(1, len(parts)):
                index[".".join(parts[start:])].append(qualname)
        return {suffix: tuple(sorted(names)) for suffix, names in index.items()}

    def remap_by_suffix(self, module_ref: str) -> str | None:
        """Best-effort recovery of a module made importable by a path mutation.

        Returns None unless there is exactly one candidate and the name is not a
        stdlib module — guessing between two candidates would cost precision,
        which is the gate that matters most.
        """
        if module_ref in self.modules or module_ref.partition(".")[0] in STDLIB_MODULES:
            return None
        candidates = self._by_suffix.get(module_ref, ())
        return candidates[0] if len(candidates) == 1 else None

    # -- construction ----------------------------------------------------

    def _alias_map(self, qualname: str, fact: FileFacts) -> dict[str, _Alias]:
        aliases: dict[str, _Alias] = {}
        for raw in fact.imports:
            if raw.name == "*":
                continue
            module_ref = self._absolute_module(qualname, raw)
            if module_ref is None:
                continue
            aliases[raw.local_name] = _Alias(
                module_ref=module_ref,
                name="" if raw.module is None else raw.name,
                is_module_import=raw.module is None,
                line=raw.line,
            )
        return aliases

    def _absolute_module(self, importing: str, raw: RawImport) -> str | None:
        """The absolute dotted path of the module an import reads from."""
        if raw.level == 0:
            return raw.name if raw.module is None else raw.module
        base = self._relative_base(importing, raw.level)
        if base is None:
            return None
        return f"{base}.{raw.module}" if raw.module else base

    def _relative_base(self, importing: str, level: int) -> str | None:
        """Resolve `level` leading dots against the importing module's package."""
        package = importing if self._is_package_init.get(importing) else importing.rpartition(".")[0]
        parts = package.split(".") if package else []
        keep = len(parts) - (level - 1)
        if keep < 0:
            return None  # escaped above the source root — unresolvable
        return ".".join(parts[:keep])

    # -- queries ---------------------------------------------------------

    def is_module(self, ref: str) -> bool:
        return ref in self.modules

    def defines(self, module: str, name: str) -> bool:
        return name in self._top_level.get(module, frozenset())

    def alias(self, module: str, name: str) -> _Alias | None:
        return self._aliases.get(module, {}).get(name)

    def is_package_init(self, qualname: str) -> bool:
        """The module is a package's `__init__.py` (true even when that file failed to parse)."""
        return self._is_package_init.get(qualname, False)

    def has_star_import(self, module: str) -> bool:
        fact = self._facts.get(module)
        return bool(fact) and any(raw.name == "*" for raw in fact.imports)

    def top_level_symbol(self, module: str, name: str) -> RawSymbol | None:
        """The module-level symbol `name` defined in `module`, or None (unknown module, or no such top-level name)."""
        fact = self._facts.get(module)
        return next((s for s in fact.symbols if s.parent_local is None and s.name == name), None) if fact else None

    # -- resolution ------------------------------------------------------

    def resolve(self, importing: str, raw: RawImport) -> Target:
        """Resolve one import to its canonical target."""
        module_ref = self._absolute_module(importing, raw)
        if module_ref is None:
            return Target(
                ref=f"{'.' * raw.level}{raw.module or ''}",
                module=None,
                symbol=None,
                is_internal=False,
                confidence=Confidence.AMBIGUOUS,
            )

        if raw.module is None:
            # `import a.b.c` — always names a module.
            return self._module_target(module_ref)

        if raw.name == "*":
            return self._module_target(module_ref)

        return self.canonicalize(module_ref, raw.name)

    def canonicalize(self, module_ref: str, name: str) -> Target:
        """Follow re-export chains to the module that actually defines `name`."""
        origin = f"{module_ref}.{name}"
        resolved_via = None
        if (remapped := self.remap_by_suffix(module_ref)) is not None:
            module_ref, resolved_via = remapped, "sys_path_suffix"
        seen: set[tuple[str, str]] = set()
        current_module, current_name = module_ref, name

        for _ in range(MAX_ALIAS_HOPS):
            if (current_module, current_name) in seen:
                break
            seen.add((current_module, current_name))

            submodule = f"{current_module}.{current_name}"
            if self.is_module(submodule):
                return self._module_target(submodule, origin)

            if not self.is_module(current_module):
                return Target(
                    ref=submodule,
                    module=None,
                    symbol=None,
                    is_internal=False,
                    confidence=Confidence.EXTRACTED,
                    canonicalized_from=origin if submodule != origin else None,
                )

            if self.defines(current_module, current_name):
                return Target(
                    ref=submodule,
                    module=current_module,
                    symbol=current_name,
                    is_internal=True,
                    confidence=Confidence.INFERRED if resolved_via else Confidence.EXTRACTED,
                    canonicalized_from=origin if submodule != origin else None,
                    resolved_via=resolved_via,
                )

            hop = self.alias(current_module, current_name)
            if hop is None:
                break
            if hop.is_module_import:
                return self._module_target(hop.module_ref, origin)
            current_module, current_name = hop.module_ref, hop.name

        # Name exists at runtime but Argus cannot say where it is defined.
        # A star import in the target module is the usual cause and is honest
        # to report as such rather than guessed at.
        return Target(
            ref=origin,
            module=module_ref if self.is_module(module_ref) else None,
            symbol=None,
            is_internal=self.is_module(module_ref),
            confidence=Confidence.AMBIGUOUS,
            canonicalized_from=None,
        )

    def _module_target(self, module_ref: str, origin: str | None = None) -> Target:
        internal = self.is_module(module_ref)
        resolved_via = None
        if not internal and (remapped := self.remap_by_suffix(module_ref)):
            module_ref, internal, resolved_via = remapped, True, "sys_path_suffix"
        return Target(
            ref=module_ref,
            module=module_ref if internal else None,
            symbol=None,
            is_internal=internal,
            confidence=Confidence.INFERRED if resolved_via else Confidence.EXTRACTED,
            canonicalized_from=origin if origin and origin != module_ref else None,
            resolved_via=resolved_via,
        )
