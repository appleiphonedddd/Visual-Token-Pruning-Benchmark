"""TRIM (Song et al., COLING 2025), fixed-budget variant of the official code.

Official scoring (``token_reduction`` in the TRIM repo): CLIP image embeddings of
the select-layer patches (post_layernorm + visual_projection, not normalised)
dotted with the pooled CLIP text embedding, negated, then softmax over tokens.
The top tokens are kept (original order) and the mean of all other pre-projector
features is appended as one aggregated token.

TRIM chooses the kept count adaptively with an IQR rule; for a fixed budget we
keep the top ``budget - 1`` tokens plus the aggregated token.
"""

from __future__ import annotations

import torch

from vtb.core.registry import METHODS
from vtb.core.types import Feature, PruneResult, PruningContext
from vtb.methods.base import PreLLMPruner


def trim_scores(img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
    return torch.softmax(-(img.float() @ txt.float().reshape(-1)), dim=0)  # [N]


@METHODS.register("trim")
class TRIM(PreLLMPruner):
    requires = frozenset({Feature.VISUAL_TOKENS, Feature.CLIP_IMAGE_EMBEDS, Feature.CLIP_TEXT_EMBEDS,
                          Feature.VISION_FEATS, Feature.PROJECTOR})

    def prune(self, ctx: PruningContext) -> PruneResult:
        scores = trim_scores(ctx[Feature.CLIP_IMAGE_EMBEDS], ctx[Feature.CLIP_TEXT_EMBEDS])
        k = ctx.budget - 1
        if k <= 0:
            return PruneResult.select(ctx.visual_tokens, torch.topk(scores, 1).indices)
        keep = torch.sort(torch.topk(scores, k).indices).values
        rest = torch.ones_like(scores, dtype=torch.bool)
        rest[keep] = False
        rest = rest.nonzero().squeeze(-1)
        feats = ctx[Feature.VISION_FEATS]
        agg = ctx[Feature.PROJECTOR](feats[rest].float().mean(0, keepdim=True).to(feats.dtype))
        tokens = torch.cat([ctx.visual_tokens[keep], agg.to(ctx.visual_tokens.dtype)])
        return PruneResult(tokens=tokens, indices=torch.cat([keep, rest.max().view(1)]), merged=True)
