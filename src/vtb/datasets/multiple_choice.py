"""Multiple-choice datasets. Prompt layout follows LLaVA-1.5's eval scripts:

    [hint]
    question
    A. option
    B. option
    Answer with the option's letter from the given choices directly.
"""

from __future__ import annotations

import math
import string
from typing import Any

import numpy as np

from vtb.core.registry import DATASETS
from vtb.core.types import Sample
from vtb.datasets.base import BaseDataset

LETTERS = string.ascii_uppercase


def _present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return str(v).strip().lower() not in ("", "nan", "none")


class MultipleChoiceDataset(BaseDataset):
    """Base for multiple-choice datasets; subclasses fill ``choices`` and ``meta['hint']``."""

    evaluator = "multiple_choice"
    prompt_suffix = "Answer with the option's letter from the given choices directly."
    hint_template = "{}"

    def format_prompt(self, sample: Sample) -> str:
        parts = []
        hint = sample.meta.get("hint")
        if _present(hint):
            parts.append(self.hint_template.format(hint))
        parts.append(sample.question)
        parts += [f"{l}. {c}" for l, c in zip(LETTERS, sample.choices or [])]
        parts.append(self.prompt_suffix)
        return "\n".join(p for p in parts if p)


@DATASETS.register("scienceqa_img")
class ScienceQAIMG(MultipleChoiceDataset):
    """lmms-lab/ScienceQA, ScienceQA-IMG test: 2017 image questions, 2-5 options."""

    hint_template = "Context: {}"

    def to_sample(self, row: dict, idx: int) -> Sample:
        choices = list(row["choices"])
        return Sample(
            id=str(idx),
            images=[row["image"]],
            question=row["question"],
            answers=[LETTERS[int(row["answer"])]],
            choices=choices,
            meta={"hint": row.get("hint"), "category": row.get("subject"), "grade": row.get("grade")},
        )


@DATASETS.register("mmbench")
class MMBench(MultipleChoiceDataset):
    """lmms-lab/MMBench dev (en / cn): 4329 rows = 1164 questions x circular option rotations.

    Rows sharing ``index % 1_000_000`` are rotations of one question; the evaluator
    reports ``circular_accuracy`` (the official CircularEval) next to per-row
    accuracy. ``limit`` samples whole questions so rotation groups stay complete.
    """

    def subset_indices(self, ds) -> list[int]:
        groups = np.array([int(i) % 1_000_000 for i in ds["index"]])
        uniq = np.unique(groups)
        if self.limit is None or self.limit >= len(uniq):
            return list(range(len(ds)))
        keep = set(np.random.default_rng(self.subset_seed).permutation(uniq)[: self.limit].tolist())
        return [i for i, g in enumerate(groups) if g in keep]

    def to_sample(self, row: dict, idx: int) -> Sample:
        choices = [row[l] for l in "ABCD" if _present(row.get(l))]
        return Sample(
            id=str(row["index"]),
            images=[row["image"]],
            question=row["question"],
            answers=[row["answer"]],
            choices=choices,
            meta={
                "hint": row.get("hint"),
                "category": row.get("category"),
                "circular_group": int(row["index"]) % 1_000_000,
            },
        )
