from __future__ import annotations

from vtb.core.registry import DATASETS
from vtb.core.types import Sample
from vtb.datasets.base import BaseDataset


@DATASETS.register("mme")
class MME(BaseDataset):
    """lmms-lab/MME: 2374 yes/no questions, 2 per image, 14 sub-tasks (10 perception + 4 cognition).

    Questions already end with "Please answer yes or no.", so no suffix is added.
    """

    evaluator = "mme"
    prompt_suffix = ""

    def to_sample(self, row: dict, idx: int) -> Sample:
        return Sample(
            id=f"{row['question_id']}|{idx}",
            images=[row["image"]],
            question=row["question"],
            answers=[row["answer"]],
            meta={"category": row["category"], "image_id": row["question_id"]},
        )
