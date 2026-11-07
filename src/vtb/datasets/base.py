"""Dataset interface.

A dataset turns raw rows into ``Sample`` objects and formats the prompt.
Scoring is delegated to the evaluator named in ``evaluator`` so evaluators can
be shared across datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterator

import numpy as np

from vtb.core.types import Sample


class BaseDataset(ABC):
    registry_name: ClassVar[str]
    evaluator: ClassVar[str]
    prompt_suffix: ClassVar[str] = ""

    def __init__(
        self,
        path: str,
        split: str,
        limit: int | None = None,
        subset_seed: int = 0,
        prompt_suffix: str | None = None,
        primary_metric: str | None = None,
        data_files: str | list | dict | None = None,
        name: str | None = None,
        type: str | None = None,  # noqa: A002 - config key
        **kwargs: Any,
    ):
        self.path = path
        self.split = split
        self.limit = limit
        self.subset_seed = subset_seed
        self.primary_metric = primary_metric  # None -> evaluator default
        self.data_files = data_files
        self.name = name or self.registry_name
        if prompt_suffix is not None:
            self.prompt_suffix = prompt_suffix
        self.cfg = kwargs
        self._rows, self._indices = self.load_rows()

    # ---- loading ----------------------------------------------------------------------
    def load_hf(self, subset: str | None = None, split: str | None = None, data_files: Any = None):
        """Load a HF hub dataset. ``data_files`` restricts the download to the needed split."""
        from datasets import load_dataset

        split = split or self.split
        data_files = data_files if data_files is not None else self.data_files
        if isinstance(data_files, (str, list)):
            data_files = {split: data_files}
        return load_dataset(self.path, subset if subset is not None else self.cfg.get("subset"),
                            data_files=data_files, split=split)

    def subset_indices(self, ds) -> list[int]:
        """Rows to evaluate. With ``limit``: a seeded random subset (same members as
        ``ds.shuffle(seed).select(range(limit))``), kept in original order."""
        n = len(ds)
        if self.limit is None or self.limit >= n:
            return list(range(n))
        perm = np.random.default_rng(self.subset_seed).permutation(n)[: self.limit]
        return sorted(int(i) for i in perm)

    def load_rows(self) -> tuple[Any, list[int]]:
        ds = self.load_hf()
        idx = self.subset_indices(ds)
        return (ds.select(idx) if len(idx) < len(ds) else ds), idx

    # ---- iteration --------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Sample]:
        for row, idx in zip(self._rows, self._indices):
            yield self.to_sample(row, idx)

    @abstractmethod
    def to_sample(self, row: dict, idx: int) -> Sample:
        """``idx`` is the row's index in the full split (stable across ``limit`` settings)."""

    def format_prompt(self, sample: Sample) -> str:
        return f"{sample.question}\n{self.prompt_suffix}".strip() if self.prompt_suffix else sample.question
