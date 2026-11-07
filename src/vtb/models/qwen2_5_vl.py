"""Qwen2.5-VL adapter.

Positions use M-RoPE: the full 3D position ids of the unpruned sequence come
from the model's own ``get_rope_index``; the columns of kept visual tokens are
gathered, text after the image keeps its original positions, and decoding
continues from ``max(position) + 1`` (what ``generate`` does via ``rope_deltas``).

The vision encoder uses window attention and has no [CLS] token, so attention-
or CLS-based pruners (VisionZip, PruMerge, ...) are reported as unsupported.
Instruction embedding = mean LLM input embedding of the question tokens
(the choice CDPruner makes for models without a paired text encoder).
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from vtb.core.registry import MODELS
from vtb.core.types import Feature
from vtb.models.hf_base import HFVLMAdapter


@MODELS.register("qwen2_5_vl")
class Qwen2_5_VLAdapter(HFVLMAdapter):
    provides = frozenset({Feature.VISUAL_TOKENS, Feature.TEXT_EMBEDS, Feature.GRID_SHAPE, Feature.LLM_ATTN})
    model_cls = Qwen2_5_VLForConditionalGeneration
    reindex_positions = False
    state_positions_dim = 3

    def image_features(self, inputs: dict) -> tuple[torch.Tensor, Any]:
        out = self.core.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
        return out.pooler_output[0].to(self.dtype), out

    def position_ids(self, inputs: dict) -> torch.Tensor:
        pos, _ = self.core.get_rope_index(
            inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"], image_grid_thw=inputs["image_grid_thw"]
        )
        return pos[:, 0]  # [3, L]

    def extra_features(self, inputs: dict, visual_tokens: torch.Tensor, vision_out: Any) -> dict[Feature, Any]:
        merge = self.model.config.vision_config.spatial_merge_size
        _, gh, gw = inputs["image_grid_thw"][0].tolist()
        return {Feature.GRID_SHAPE: (gh // merge, gw // merge)}
