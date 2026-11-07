"""Name -> class registries.

Every pluggable component (model adapter, pruning method, dataset, evaluator)
registers itself with a decorator. Configs refer to components by name, so adding
a new one means adding a new module; nothing existing has to be edited.

Modules under ``vtb.models``, ``vtb.methods``, ``vtb.datasets`` and
``vtb.evaluators`` are imported automatically the first time a registry is
queried, so a new file is picked up without touching any ``__init__``.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str, package: str):
        self.kind = kind
        self.package = package
        self._items: dict[str, type[T]] = {}
        self._scanned = False

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if name in self._items and self._items[name] is not cls:
                raise KeyError(f"{self.kind} '{name}' is already registered by {self._items[name]!r}")
            self._items[name] = cls
            cls.registry_name = name  # type: ignore[attr-defined]
            return cls

        return deco

    def _scan(self) -> None:
        if self._scanned:
            return
        self._scanned = True
        pkg = importlib.import_module(self.package)
        for mod in pkgutil.iter_modules(pkg.__path__):
            if not mod.name.startswith("_"):
                importlib.import_module(f"{self.package}.{mod.name}")

    def get(self, name: str) -> type[T]:
        self._scan()
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {self.names()}")
        return self._items[name]

    def build(self, key: str, /, **kwargs: Any) -> T:
        return self.get(key)(**kwargs)

    def names(self) -> list[str]:
        self._scan()
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        self._scan()
        return name in self._items


MODELS: Registry = Registry("model", "vtb.models")
METHODS: Registry = Registry("method", "vtb.methods")
DATASETS: Registry = Registry("dataset", "vtb.datasets")
EVALUATORS: Registry = Registry("evaluator", "vtb.evaluators")
