"""Exact match after light normalisation (GQA, following LLaVA's convert_gqa_for_eval.py)."""

from __future__ import annotations

from collections import defaultdict

from vtb.core.registry import EVALUATORS
from vtb.evaluators.base import BaseEvaluator


def normalize(text: str) -> str:
    return text.strip().rstrip(".").strip().lower()


@EVALUATORS.register("exact_match")
class ExactMatch(BaseEvaluator):
    primary_metric = "accuracy"

    def score(self, records: list[dict]) -> dict[str, float]:
        per_type: dict[str, list[bool]] = defaultdict(list)
        hits = []
        for r in records:
            ok = normalize(r["output"]) in {normalize(a) for a in r["answers"]}
            hits.append(ok)
            if "type" in r.get("meta", {}):
                per_type[r["meta"]["type"]].append(ok)
        out = {"accuracy": 100 * sum(hits) / max(len(hits), 1), "num_samples": float(len(records))}
        out.update({f"{t}/accuracy": 100 * sum(v) / len(v) for t, v in sorted(per_type.items())})
        return out
