"""Evaluator interface.

Evaluators score the records written to ``predictions.jsonl`` (dicts with
``output``, ``answers``, ``choices``, ``meta``), so metrics can always be
recomputed from disk without re-running the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class BaseEvaluator(ABC):
    registry_name: ClassVar[str]
    primary_metric: ClassVar[str]  # the number that goes into summary tables (a dataset may override)
    # divisors that map a metric onto 0..100 for cross-dataset averages (e.g. MME perception / 20)
    percent_scale: ClassVar[dict[str, float]] = {}

    def __init__(self, **kwargs: Any):
        self.cfg = kwargs

    @abstractmethod
    def score(self, records: list[dict]) -> dict[str, float]: ...

    def headline(self, scores: dict[str, float], metric: str | None = None) -> dict[str, Any]:
        """Primary metric, its value, and the value on a 0..100 scale."""
        metric = metric or self.primary_metric
        value = scores[metric]
        return {"primary_metric": metric, "score": value, "score_pct": value / self.percent_scale.get(metric, 1.0)}
