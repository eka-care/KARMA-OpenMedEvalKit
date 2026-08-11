"""Named, versioned prompt registry.

Prompts are declared as module-level ``Prompt(...)`` instances in this
package's domain files (e.g. ``mcqa_prompts.py``). ``resolve_prompt(name)``
imports those modules once, collects every ``Prompt`` instance, and returns the
one with the requested name — raising ``PromptNotFoundError`` if it is unknown
(callers pass a name explicitly, so an unknown name is a hard error, never a
silent fallback).

Kept dependency-light on purpose: nothing heavy may be pulled in here.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Dict, List

_VALID_ROLES = {"user", "system"}

_REGISTRY: Dict[str, "Prompt"] = {}
_discovered = False


class PromptNotFoundError(Exception):
    """Raised when a referenced prompt name is not registered."""


@dataclass(frozen=True)
class Prompt:
    """A named, versioned prompt body. ``version`` is provenance only (bump on
    edit); the registry is keyed by unique ``name`` — two live wordings need
    two names."""

    name: str
    version: int
    role: str
    body: str

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"prompt {self.name!r}: role must be one of {sorted(_VALID_ROLES)}, "
                f"got {self.role!r}"
            )


def _register(prompt: Prompt) -> None:
    existing = _REGISTRY.get(prompt.name)
    if existing is not None and existing != prompt:
        raise ValueError(f"duplicate prompt name {prompt.name!r}")
    _REGISTRY[prompt.name] = prompt


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    package = importlib.import_module(__name__)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__, __name__ + "."):
        module = importlib.import_module(module_name)
        for value in vars(module).values():
            if isinstance(value, Prompt):
                _register(value)
    _discovered = True


def resolve_prompt(name: str) -> Prompt:
    """Return the registered prompt, or raise ``PromptNotFoundError``."""
    _discover()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise PromptNotFoundError(
            f"unknown prompt {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def list_prompts() -> List[Prompt]:
    _discover()
    return list(_REGISTRY.values())
