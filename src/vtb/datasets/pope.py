from __future__ import annotations

from vtb.core.registry import DATASETS
from vtb.core.types import Sample
from vtb.datasets.base import BaseDataset


@DATASETS.register("pope")
class POPE(BaseDataset):
    """lmms-lab/POPE: 9000 yes/no object-existence questions over 3 splits (adversarial/popular/random)."""

    evaluator = "pope"
    prompt_suffix = "Answer the question using a single word or phrase."

    def to_sample(self, row: dict, idx: int) -> Sample:
        return Sample(
            id=str(row["question_id"]),
            images=[row["image"]],
            question=row["question"],
            answers=[row["answer"]],
            meta={"category": row["category"]},
        )
