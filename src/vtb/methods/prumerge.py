"""LLaVA-PruMerge+ (Shang et al., 2024), fixed-budget variant of the official code.

Official procedure (``token_prune_merge_advanced_plus``):
  * [CLS]->patch attention from the select layer's q/k used as ONE head over the
    full width: softmax(q_cls k^T / sqrt(D)).
  * keep IQR outliers (> Q3 + 1.5 IQR) of that attention, ordered by attention,
    plus spatially uniform samples;
  * each kept token x_i += sum_j a_j x_j over its 32 nearest tokens (key cosine,
    among all other tokens), with a_j the [CLS] attention (no normalisation);
  * one extra token = sum of the pruned tokens weighted by attention.

Fixed budget (to compare at 64 / 128 / 192): ``budget - 1`` kept tokens (outliers,
top-by-attention if there are more, then uniform fill) + the extra token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vtb.core.registry import METHODS
from vtb.core.types import Feature, PruneResult, PruningContext
from vtb.methods.base import PreLLMPruner


def iqr_outliers(scores: torch.Tensor) -> torch.Tensor:
    q1, q3 = torch.quantile(scores.float(), torch.tensor([0.25, 0.75], device=scores.device))
    return (scores > q3 + 1.5 * (q3 - q1)).nonzero().squeeze(-1)


@METHODS.register("prumerge_plus")
class PruMergePlus(PreLLMPruner):
    requires = frozenset({Feature.VISUAL_TOKENS, Feature.VISION_QK, Feature.VISION_FEATS,
                          Feature.PROJECTOR, Feature.CLS_TOKEN})

    def prune(self, ctx: PruningContext) -> PruneResult:
        n, budget = ctx.num_tokens, ctx.budget
        q, k, _ = ctx[Feature.VISION_QK]  # [S, D] with [CLS] at 0
        q, k = q.float(), k.float()
        attn = torch.softmax((q[:1] @ k.T) * q.shape[-1] ** -0.5, dim=-1)[0, 1:]  # [N]
        keys = F.normalize(k[1:], dim=-1)
        feats = ctx[Feature.VISION_FEATS].float()

        n_keep = budget - 1
        out = iqr_outliers(attn)
        out = out[torch.argsort(attn[out], descending=True)][:n_keep]  # by attention, like topk
        chosen = torch.zeros(n, dtype=torch.bool, device=attn.device)
        chosen[out] = True
        need = n_keep - out.numel()
        fill = out.new_empty(0)
        if need > 0:  # spatially uniform fill among the remaining tokens
            rest = (~chosen).nonzero().squeeze(-1)
            fill = rest[torch.linspace(0, rest.numel() - 1, need, device=attn.device).round().long()]
            chosen[fill] = True
        keep = torch.cat([out, fill])
        pruned = (~chosen).nonzero().squeeze(-1)

        parts, anchors = [], []
        if keep.numel():
            sim = keys[keep] @ keys.T  # [K, N]
            sim[torch.arange(keep.numel(), device=sim.device), keep] = -float("inf")  # exclude self
            knn = min(int(self.params.get("k", 32)), n - 1)
            nbr = torch.topk(sim, knn, dim=1).indices  # [K, knn]
            parts.append(feats[keep] + (attn[nbr][..., None] * feats[nbr]).sum(1))
            anchors.append(keep)
        if pruned.numel():
            parts.append((attn[pruned][:, None] * feats[pruned]).sum(0, keepdim=True))
            anchors.append(pruned.max().view(1))
        dtype = ctx[Feature.VISION_FEATS].dtype
        tokens = ctx[Feature.PROJECTOR](torch.cat(parts).to(dtype)).to(ctx.visual_tokens.dtype)
        return PruneResult(tokens=tokens, indices=torch.cat(anchors), merged=True)
