"""POPE scoring, following LLaVA's ``eval_pope.py``.

The answer is the first sentence; it counts as "no" if it contains the word
no/not, otherwise "yes". Metrics are computed per category; the headline ``f1``
is the mean over categories, which is what the pruning papers report.
"""

from __future__ import annotations

from collections import defaultdict

from vtb.core.registry import EVALUATORS
from vtb.evaluators.base import BaseEvaluator


def parse_yes_no(text: str) -> str:
    first = text.split(".")[0].replace(",", " ")
    words = [w.strip().lower() for w in first.split()]
    return "no" if ("no" in words or "not" in words) else "yes"


def binary_metrics(preds: list[str], labels: list[str]) -> dict[str, float]:
    tp = sum(p == "yes" and l == "yes" for p, l in zip(preds, labels))
    fp = sum(p == "yes" and l == "no" for p, l in zip(preds, labels))
    tn = sum(p == "no" and l == "no" for p, l in zip(preds, labels))
    fn = sum(p == "no" and l == "yes" for p, l in zip(preds, labels))
    n = max(len(preds), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": 100 * (tp + tn) / n,
        "precision": 100 * precision,
        "recall": 100 * recall,
        "f1": 100 * f1,
        "yes_ratio": 100 * (tp + fp) / n,
    }


@EVALUATORS.register("pope")
class POPEEvaluator(BaseEvaluator):
    primary_metric = "f1"

    def score(self, records: list[dict]) -> dict[str, float]:
        groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for r in records:
            p, l = groups[r["meta"].get("category", "all")]
            p.append(parse_yes_no(r["output"]))
            l.append(r["answers"][0].strip().lower())

        out: dict[str, float] = {}
        per_cat = {c: binary_metrics(p, l) for c, (p, l) in sorted(groups.items())}
        for c, m in per_cat.items():
            out.update({f"{c}/{k}": v for k, v in m.items()})
        for k in ("accuracy", "f1"):
            out[k] = sum(m[k] for m in per_cat.values()) / len(per_cat)
        out["num_samples"] = float(len(records))
        return out
