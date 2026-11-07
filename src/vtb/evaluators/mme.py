"""MME scoring, following the official ``calculation.py``.

Per sub-task: acc (per question) + acc+ (both questions of an image correct),
each in %, so a sub-task scores up to 200. Perception = sum of 10 sub-tasks
(max 2000), cognition = sum of 4 (max 800).

CDPruner reports perception (e.g. 1506.5 for LLaVA-1.5-7B); VisionZip reports
perception + cognition (1862). Both are returned; pick one with the dataset's
``primary_metric``.
"""

from __future__ import annotations

from collections import defaultdict

from vtb.core.registry import EVALUATORS
from vtb.evaluators.base import BaseEvaluator

PERCEPTION = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]
COGNITION = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]


def parse_pred(text: str) -> str:
    t = text.strip().lower()
    if t in ("yes", "no"):
        return t
    prefix = t[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


@EVALUATORS.register("mme")
class MMEEvaluator(BaseEvaluator):
    primary_metric = "perception"
    percent_scale = {"perception": 20.0, "cognition": 8.0, "total": 28.0}  # score / scale -> 0..100

    def score(self, records: list[dict]) -> dict[str, float]:
        # category -> image -> [correct, ...]
        groups: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            ok = parse_pred(r["output"]) == r["answers"][0].strip().lower()
            groups[r["meta"]["category"]][r["meta"]["image_id"]].append(ok)

        out: dict[str, float] = {}
        for cat, images in groups.items():
            flat = [x for v in images.values() for x in v]
            acc = 100 * sum(flat) / len(flat)
            acc_plus = 100 * sum(all(v) for v in images.values()) / len(images)
            out[f"{cat}/acc"], out[f"{cat}/acc_plus"], out[cat] = acc, acc_plus, acc + acc_plus
        out["perception"] = sum(out.get(c, 0.0) for c in PERCEPTION)
        out["cognition"] = sum(out.get(c, 0.0) for c in COGNITION)
        out["total"] = out["perception"] + out["cognition"]
        out["num_samples"] = float(len(records))
        return out
