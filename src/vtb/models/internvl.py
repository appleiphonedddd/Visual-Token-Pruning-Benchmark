"""InternVL3 adapter (HF ``OpenGVLab/InternVL3-*-hf`` checkpoints).

Images are split into 448x448 tiles (+ a thumbnail when there is more than one
tile); each tile gives 256 tokens after pixel shuffle. With a single tile
(``processor_call_kwargs: {crop_to_patches: false}``) the tokens form a 16x16
grid, which is exposed as ``grid_shape``. 1D RoPE: positions are re-numbered
after pruning.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from vtb.core.registry import MODELS
from vtb.core.types import Feature
from vtb.models.hf_base import HFVLMAdapter


@MODELS.register("internvl")
class InternVLAdapter(HFVLMAdapter):
    provides = frozenset({Feature.VISUAL_TOKENS, Feature.TEXT_EMBEDS, Feature.GRID_SHAPE, Feature.LLM_ATTN})

    def extra_features(self, inputs: dict, visual_tokens: torch.Tensor, vision_out: Any) -> dict[Feature, Any]:
        n = visual_tokens.shape[0]
        side = math.isqrt(n)
        if inputs["pixel_values"].shape[0] == 1 and side * side == n:
            return {Feature.GRID_SHAPE: (side, side)}
        return {}  # multi-tile layout: no single grid (grid-based pruners will raise a clear KeyError)
