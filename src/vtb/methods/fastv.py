"""Attention-based in-LLM pruning: FastV and PyramidDrop.

Both rank the remaining visual tokens by the attention the last prompt token
pays to them (head-averaged) in the layer *before* the pruning point.

* FastV (Chen et al., ECCV 2024): keep ``budget`` tokens from layer ``K`` (default 2) on.
* PyramidDrop (Xing et al., CVPR 2025): prune at several layers, keeping
  ``budget`` tokens from the first stage layer and multiplying by ``ratio`` at
  each later stage. Default stages [2, 6, 12] with ratio 0.5 give
  576 -> 64 -> 32 -> 16 at budget 64. The paper's LLaVA setting is stages
  [8, 16, 24] with ratio 0.5 from all 576 tokens; set ``layers`` / ``ratio`` /
  budget in the config to reproduce it.
"""

from __future__ import annotations

import torch

from vtb.core.registry import METHODS
from vtb.methods.base import InLLMPruner, LLMLayerContext


@METHODS.register("fastv")
class FastV(InLLMPruner):
    def schedule(self, num_visual: int, num_layers: int) -> list[tuple[int, int]]:
        if self.budget is None:
            return []
        return [(int(self.params.get("K", 2)), self.budget)]

    def select(self, ctx: LLMLayerContext) -> torch.Tensor:
        return torch.topk(ctx.visual_attn.float(), ctx.budget).indices


@METHODS.register("pdrop")
class PyramidDrop(InLLMPruner):
    def schedule(self, num_visual: int, num_layers: int) -> list[tuple[int, int]]:
        if self.budget is None:
            return []
        layers = list(self.params.get("layers", [2, 6, 12]))
        ratio = float(self.params.get("ratio", 0.5))
        out, keep = [], float(self.budget)
        for layer in layers:
            out.append((int(layer), max(1, round(keep))))
            keep *= ratio
        return out

    def select(self, ctx: LLMLayerContext) -> torch.Tensor:
        return torch.topk(ctx.visual_attn.float(), ctx.budget).indices
