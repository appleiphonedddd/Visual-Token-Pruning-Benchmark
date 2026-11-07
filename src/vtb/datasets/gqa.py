from __future__ import annotations

from typing import Any

from vtb.core.registry import DATASETS
from vtb.core.types import Sample
from vtb.datasets.base import BaseDataset


@DATASETS.register("gqa")
class GQA(BaseDataset):
    """lmms-lab/GQA test-dev balanced: 12,578 questions over 398 images.

    Questions and images live in separate configs (``<split>_instructions`` and
    ``<split>_images``) and are joined on ``imageId``.
    """

    evaluator = "exact_match"
    prompt_suffix = "Answer the question using a single word or phrase."

    def load_rows(self) -> tuple[Any, list[int]]:
        base = self.cfg.get("subset_prefix", "testdev_balanced")
        questions = self.load_hf(subset=f"{base}_instructions")
        self._images = self.load_hf(subset=f"{base}_images")
        self._image_row = {img_id: i for i, img_id in enumerate(self._images["id"])}
        idx = self.subset_indices(questions)
        return (questions.select(idx) if len(idx) < len(questions) else questions), idx

    def to_sample(self, row: dict, idx: int) -> Sample:
        image = self._images[self._image_row[row["imageId"]]]["image"]
        return Sample(
            id=str(row["id"]),
            images=[image],
            question=row["question"],
            answers=[row["answer"]],
            meta={"image_id": row["imageId"]},
        )
