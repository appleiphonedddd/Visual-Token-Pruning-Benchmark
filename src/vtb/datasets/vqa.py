"""Open-ended VQA datasets scored with VQA accuracy (10 human answers)."""

from __future__ import annotations

from vtb.core.registry import DATASETS
from vtb.core.types import Sample
from vtb.datasets.base import BaseDataset


@DATASETS.register("textvqa")
class TextVQA(BaseDataset):
    """lmms-lab/textvqa validation: 5000 questions about scene text.

    ``ocr_reference: true`` appends "Reference OCR token: ..." to the prompt, which
    is the LLaVA-1.5 evaluation protocol used by most pruning papers. It leaks
    OCR text that the pruned visual tokens may have lost, so it is off by default.
    """

    evaluator = "vqa_accuracy"
    prompt_suffix = "Answer the question using a single word or phrase."

    def to_sample(self, row: dict, idx: int) -> Sample:
        return Sample(
            id=str(row["question_id"]),
            images=[row["image"]],
            question=row["question"],
            answers=list(row["answers"]),
            meta={"ocr_tokens": list(row.get("ocr_tokens") or [])},
        )

    def format_prompt(self, sample: Sample) -> str:
        parts = [sample.question]
        if self.cfg.get("ocr_reference") and sample.meta["ocr_tokens"]:
            parts.append("Reference OCR token: " + ", ".join(sample.meta["ocr_tokens"]))
        parts.append(self.prompt_suffix)
        return "\n".join(p for p in parts if p)


@DATASETS.register("vqav2")
class VQAv2(BaseDataset):
    """lmms-lab/VQAv2. Papers report test-dev, which needs the EvalAI server; local
    scoring uses the validation split (214K questions, ~7 GB, use ``limit``)."""

    evaluator = "vqa_accuracy"
    prompt_suffix = "Answer the question using a single word or phrase."

    def to_sample(self, row: dict, idx: int) -> Sample:
        return Sample(
            id=str(row["question_id"]),
            images=[row["image"]],
            question=row["question"],
            answers=[a["answer"] for a in row["answers"]],
            meta={"answer_type": row["answer_type"]},
        )
