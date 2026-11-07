"""Data objects shared between models, methods, datasets and evaluators.

These are the *only* things the three axes exchange, which keeps them decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class Feature(str, Enum):
    """Signals a pruning method may need and a model adapter may provide."""

    VISUAL_TOKENS = "visual_tokens"  # [N, D_llm] tokens after the projector (always provided)
    VISION_FEATS = "vision_feats"  # [N, D_vis] encoder features before the projector
    VISION_ATTN = "vision_attn"  # [N] attention each token receives inside the vision encoder
    VISION_KEYS = "vision_keys"  # [N, D_k] head-averaged keys of the encoder's select layer (VisionZip merging)
    VISION_QK = "vision_qk"  # (q [S, D], k [S, D], num_heads) of the select layer, S includes [CLS] at 0
    CLS_TOKEN = "cls_token"  # [D_vis] pre-projector [CLS] feature; VISION_ATTN is then CLS->patch attention
    TEXT_EMBEDS = "text_embeds"  # [D] instruction embedding comparable to VISUAL_TOKENS
    GRID_SHAPE = "grid_shape"  # (H, W) spatial layout of VISUAL_TOKENS in raster order
    PROJECTOR = "projector"  # callable VISION_FEATS [M, D_vis] -> LLM tokens [M, D_llm] (merge before projecting)
    CLIP_IMAGE_EMBEDS = "clip_image_embeds"  # [N, D_clip] patch tokens in the CLIP image-text joint space
    CLIP_TEXT_EMBEDS = "clip_text_embeds"  # [D_clip] pooled CLIP text embedding of the instruction
    CLIP_TEXT_TOKENS = "clip_text_tokens"  # [T, D_clip] per-token CLIP text embeddings of the instruction
    LLM_ATTN = "llm_attn"  # model supports in-LLM pruning (FastV / PDrop / DART)


@dataclass
class Sample:
    id: str
    images: list[Any]  # PIL images
    question: str
    answers: list[str] = field(default_factory=list)
    choices: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PruningContext:
    """Everything a pre-LLM pruner is allowed to look at."""

    budget: int
    features: dict[Feature, Any]
    instruction: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def visual_tokens(self) -> torch.Tensor:
        return self.features[Feature.VISUAL_TOKENS]

    @property
    def num_tokens(self) -> int:
        return self.visual_tokens.shape[0]

    def __getitem__(self, f: Feature) -> Any:
        if f not in self.features:
            raise KeyError(f"Feature {f.value!r} was not provided by the model adapter")
        return self.features[f]


@dataclass
class PruneResult:
    """Output of a pre-LLM pruner.

    ``tokens`` are spliced into the sequence in the given order. ``indices[i]`` is
    the original visual-token index whose position ``tokens[i]`` takes (used to
    rebuild position ids for M-RoPE models). For pure selection it is the selected
    index, sorted ascending; merging methods may use anchor indices in the order
    of their reference implementation (e.g. VisionZip: dominant then contextual).
    """

    tokens: torch.Tensor  # [K, D]
    indices: torch.Tensor  # [K] long
    merged: bool = False

    @staticmethod
    def select(tokens: torch.Tensor, keep: torch.Tensor) -> "PruneResult":
        keep = torch.sort(keep.long()).values
        return PruneResult(tokens=tokens[keep], indices=keep)


@dataclass
class Prediction:
    sample_id: str
    output: str
    num_visual_tokens: int  # before pruning
    num_kept_tokens: int  # fed into the LLM (after pre-LLM pruning)
    seq_len: int = 0  # LLM prefill length entering layer 0 (after pre-LLM pruning)
    avg_layer_seq_len: float = 0.0  # mean sequence length over decoder layers (< seq_len for in-LLM pruning)
    vision_ms: float = 0.0  # preprocessing + vision encoder + projector
    prune_ms: float = 0.0
    prefill_ms: float = 0.0  # LLM prefill only
    decode_ms: float = 0.0
    num_generated: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
