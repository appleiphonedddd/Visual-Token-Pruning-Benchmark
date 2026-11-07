"""VisionZip (Yang et al., CVPR 2025), training-free variant, following the official code.

1. Dominant tokens: [CLS] + the top-(D-1) patches by [CLS]->patch attention
   (summed over heads) at the encoder's select layer (-2).
2. Contextual tokens: the remaining patches are split uniformly into C targets
   and the rest; each remaining patch is assigned to its most similar target by
   the L2-normalised, head-averaged keys of the same layer, and
   contextual = target + mean(assigned).
All tokens are pre-projector features and go through the projector, ordered
[CLS, dominant patches (original order), contextual (target order)].

``dominant_ratio`` = 54/64 reproduces the paper's LLaVA-1.5 splits
(54+10 / 108+20 / 162+30 for 64 / 128 / 192; the 54 includes [CLS]).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vtb.core.registry import METHODS
from vtb.core.types import Feature, PruneResult, PruningContext
from vtb.methods.base import PreLLMPruner


def contextual_merge(feats: torch.Tensor, keys: torch.Tensor, num_targets: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge ``feats`` [M, D] into ``num_targets`` tokens. Returns (merged [C, D], target local idx [C])."""
    m = feats.shape[0]
    step = max(1, m // num_targets)
    target = torch.arange(0, m, step, device=feats.device)[:num_targets]
    is_target = torch.zeros(m, dtype=torch.bool, device=feats.device)
    is_target[target] = True
    others = (~is_target).nonzero().squeeze(-1)
    merged = feats[target].float().clone()
    if others.numel():
        k = F.normalize(keys.float(), dim=-1)
        assign = (k[others] @ k[target].T).argmax(dim=1)  # [M - C]
        sums = torch.zeros_like(merged).index_add_(0, assign, feats[others].float())
        counts = torch.zeros(target.numel(), device=feats.device).index_add_(
            0, assign, torch.ones_like(assign, dtype=torch.float))
        merged = merged + sums / counts.clamp(min=1)[:, None]
    return merged.to(feats.dtype), target


@METHODS.register("visionzip")
class VisionZip(PreLLMPruner):
    requires = frozenset({Feature.VISUAL_TOKENS, Feature.VISION_ATTN, Feature.VISION_KEYS,
                          Feature.VISION_FEATS, Feature.PROJECTOR, Feature.CLS_TOKEN})

    def prune(self, ctx: PruningContext) -> PruneResult:
        n, budget = ctx.num_tokens, ctx.budget
        ratio = float(self.params.get("dominant_ratio", 54 / 64))
        n_dom = min(budget, max(1, round(budget * ratio)))  # includes [CLS]
        n_ctx = budget - n_dom
        feats = ctx[Feature.VISION_FEATS]

        dom = torch.sort(torch.topk(ctx[Feature.VISION_ATTN].float(), n_dom - 1).indices).values
        rest = torch.ones(n, dtype=torch.bool, device=feats.device)
        rest[dom] = False
        rest = rest.nonzero().squeeze(-1)

        parts = [ctx[Feature.CLS_TOKEN].reshape(1, -1).to(feats.dtype), feats[dom]]
        anchors = [torch.zeros(1, dtype=torch.long, device=feats.device), dom]  # [CLS] sits before the patches
        if n_ctx > 0 and rest.numel():
            merged, local = contextual_merge(feats[rest], ctx[Feature.VISION_KEYS][rest], n_ctx)
            parts.append(merged)
            anchors.append(rest[local])
        tokens = ctx[Feature.PROJECTOR](torch.cat(parts)).to(ctx.visual_tokens.dtype)
        return PruneResult(tokens=tokens, indices=torch.cat(anchors), merged=True)
