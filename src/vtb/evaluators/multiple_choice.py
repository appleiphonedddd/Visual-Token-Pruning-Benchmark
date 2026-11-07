"""Multiple-choice accuracy (ScienceQA-IMG, MMBench, ...).

The predicted letter is extracted rule-based (no LLM judge):
  1. a leading option letter: "B", "(B)", "B.", "B) ...", "B: ..."
  2. "The answer is B" / "Answer: B"
  3. the output equals one option's text
Anything else counts as wrong (recorded as ``failed_parse``).

If records carry ``meta["circular_group"]`` (MMBench CircularEval, options
rotated), ``circular_accuracy`` counts a question correct only when every
rotation is correct.
"""

from __future__ import annotations

import re
import string
from collections import defaultdict

from vtb.core.registry import EVALUATORS
from vtb.evaluators.base import BaseEvaluator

LETTERS = string.ascii_uppercase


def parse_choice(text: str, choices: list[str]) -> str | None:
    letters = LETTERS[: len(choices)]
    t = text.strip()
    m = re.match(rf"^\(?([{letters}])\)?(?:[.:)\s]|$)", t)
    if m:
        return m.group(1)
    m = re.search(rf"answer(?: is|:)\s*\(?([{letters}])\)?\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    norm = t.lower().rstrip(".")
    for letter, choice in zip(letters, choices):
        if norm == str(choice).strip().lower().rstrip("."):
            return letter
    return None


@EVALUATORS.register("multiple_choice")
class MultipleChoice(BaseEvaluator):
    primary_metric = "accuracy"

    def score(self, records: list[dict]) -> dict[str, float]:
        hits, failed = [], 0
        per_cat: dict[str, list[bool]] = defaultdict(list)
        circular: dict[str, list[bool]] = defaultdict(list)
        for r in records:
            pred = parse_choice(r["output"], r["choices"])
            failed += pred is None
            ok = pred == r["answers"][0]
            hits.append(ok)
            meta = r.get("meta", {})
            if "category" in meta:
                per_cat[meta["category"]].append(ok)
            if "circular_group" in meta:
                circular[str(meta["circular_group"])].append(ok)

        out = {
            "accuracy": 100 * sum(hits) / max(len(hits), 1),
            "failed_parse": 100 * failed / max(len(hits), 1),
            "num_samples": float(len(records)),
        }
        if circular:
            out["circular_accuracy"] = 100 * sum(all(v) for v in circular.values()) / len(circular)
        out.update({f"{c}/accuracy": 100 * sum(v) / len(v) for c, v in sorted(per_cat.items())})
        return out
