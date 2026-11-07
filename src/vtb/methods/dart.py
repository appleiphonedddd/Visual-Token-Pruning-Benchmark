"""DART (Wen et al., 2025): "duplication matters more than importance", following the official code.

Right before LLM layer ``K`` (default 2), using the outputs of layer ``K - 1``
(the official loop checks ``layer_idx == K`` before calling the layer):
  * pivots: the ``pivot_image_token`` visual tokens and ``pivot_text_token``
    instruction tokens with the largest L1 norm of the post-RoPE keys;
  * for each pivot, keep the visual tokens least cosine-similar to it, using the
    hidden states passed through the LLM's final norm; image pivots are kept.

Budget: the official code keeps ``P_img + (P_img + P_txt) * (budget // (P_img + P_txt))``
tokens (about budget + 4). We split ``budget - P_img`` across the pivots instead, so
exactly ``budget`` tokens remain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vtb.core.registry import METHODS
from vtb.methods.base import InLLMPruner, LLMLayerContext


def dart_select(keys: torch.Tensor, normed: torch.Tensor, vis: torch.Tensor, txt: torch.Tensor,
                budget: int, p_img: int, p_txt: int) -> torch.Tensor:
    """Indices into ``vis`` to keep."""
    n = vis.numel()
    p_img = min(p_img, budget, n)
    p_txt = min(p_txt, txt.numel())
    img_piv = torch.topk(keys[vis].float().abs().sum(-1), p_img).indices  # local
    txt_piv = txt[torch.topk(keys[txt].float().abs().sum(-1), p_txt).indices] if p_txt else txt[:0]
    pivots = torch.cat([normed[vis[img_piv]], normed[txt_piv]])  # [P, D]

    taken = torch.zeros(n, dtype=torch.bool, device=vis.device)
    taken[img_piv] = True
    vis_feat = F.normalize(normed[vis].float(), dim=-1)
    piv = F.normalize(pivots.float(), dim=-1)
    remaining = budget - p_img
    num_p = piv.shape[0]
    for j in range(num_p):
        quota = remaining // (num_p - j)
        if quota <= 0:
            continue
        cos = (vis_feat @ piv[j]).masked_fill(taken, float("inf"))
        pick = torch.topk(cos, quota, largest=False).indices
        taken[pick] = True
        remaining -= quota
    return taken.nonzero().squeeze(-1)


@METHODS.register("dart")
class DART(InLLMPruner):
    needs_attention = False
    needs_keys = True

    def schedule(self, num_visual: int, num_layers: int) -> list[tuple[int, int]]:
        if self.budget is None:
            return []
        return [(int(self.params.get("K", 2)), self.budget)]

    def select(self, ctx: LLMLayerContext) -> torch.Tensor:
        return dart_select(ctx.key_states, ctx.normed_hidden, ctx.visual_positions, ctx.text_positions,
                           ctx.budget, int(self.params.get("pivot_image_token", 4)),
                           int(self.params.get("pivot_text_token", 4)))
