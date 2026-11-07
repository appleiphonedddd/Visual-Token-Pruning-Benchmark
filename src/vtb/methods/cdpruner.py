"""CDPruner (Zhang et al., NeurIPS 2025): maximise conditional diversity with a DPP.

    L_ij   = cos(H_i, H_j)                               (eq. 3)
    r_i    = cos(H_i, H_q), min-max normalised -> r~     (eq. 5-6)
    L~     = diag(r~) L diag(r~)                         (eq. 7)
    or, with balance factor theta (appendix C.4):
    L~'    = diag(exp(a r~)) L diag(exp(a r~)),  a = theta / (2 (1 - theta))

The subset is found with the fast greedy MAP algorithm of Chen et al. 2018
(appendix Algorithm 1), O(N K^2).

``dppruner`` is the paper's ablation without the instruction condition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vtb.core.registry import METHODS
from vtb.core.types import Feature, PruneResult, PruningContext
from vtb.methods.base import PreLLMPruner


def dpp_greedy_map(kernel: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    """Fast greedy MAP inference for a DPP with PSD ``kernel`` [N, N].

    If the kernel's rank is exhausted before ``k`` picks (every remaining marginal
    gain d_i^2 <= eps, e.g. more tokens requested than feature dimensions), the
    remaining slots are filled by largest diagonal (quality) so the result always
    has ``k`` distinct indices.
    """
    n = kernel.shape[0]
    cis = torch.zeros(k, n, dtype=kernel.dtype, device=kernel.device)
    di2s = torch.diagonal(kernel).clone()
    taken = torch.zeros(n, dtype=torch.bool, device=kernel.device)
    selected: list[torch.Tensor] = []
    for i in range(k):
        j = torch.argmax(di2s)
        if not di2s[j] > eps:
            break
        selected.append(j)
        taken[j] = True
        eis = (kernel[j] - cis[:i, j] @ cis[:i]) / torch.sqrt(di2s[j])  # eq. 16
        cis[i] = eis
        di2s = (di2s - eis.square()).masked_fill(taken, -float("inf"))  # eq. 17
    if len(selected) < k:
        quality = torch.diagonal(kernel).masked_fill(taken, -float("inf"))
        selected.extend(torch.topk(quality, k - len(selected)).indices)
    return torch.stack(selected)


def conditional_kernel(
    x: torch.Tensor,
    q: torch.Tensor | None,
    theta: float | None,
    rel_x: torch.Tensor | None = None,
    negate: bool = False,
) -> torch.Tensor:
    """diag(r~) L diag(r~) with L = cos-sim of ``x``; relevance from ``rel_x`` (default ``x``) vs. ``q``.

    ``negate`` flips the relevance before normalisation, as the official LLaVA code does
    for CLIP patch embeddings (``relevance = (-relevance).mean(-1)``).
    """
    xn = F.normalize(x.float(), dim=-1)
    sim = xn @ xn.T
    if q is None:
        return sim
    rn = F.normalize((x if rel_x is None else rel_x).float(), dim=-1)
    qn = F.normalize(q.float().reshape(-1, q.shape[-1]), dim=-1)  # [T, D]
    rel = rn @ qn.T  # [N, T]
    rel = (-rel if negate else rel).mean(dim=-1)
    rel = (rel - rel.min() + 1e-6) / (rel.max() - rel.min() + 1e-6)
    if theta is not None:
        alpha = theta / (2 * (1 - theta))
        rel = torch.exp(alpha * rel)
    return rel[:, None] * sim * rel[None, :]


@METHODS.register("cdpruner")
class CDPruner(PreLLMPruner):
    """The similarity kernel always uses projector tokens (as the official code does).

    Relevance (``relevance_space``):
      * ``auto``: CLIP joint space when the model provides it (the paper's LLaVA setting),
        otherwise projector tokens vs. mean LLM embedding of the instruction (eq. 5).
      * ``clip`` / ``llm``: force one.
    ``negate_relevance``: ``auto`` = True in CLIP space (official LLaVA code), False in LLM
    space (eq. 5; there is no official non-CLIP implementation to follow). Or true / false.
    """

    requires = frozenset({Feature.VISUAL_TOKENS, Feature.TEXT_EMBEDS})
    optional = frozenset({Feature.CLIP_IMAGE_EMBEDS, Feature.CLIP_TEXT_EMBEDS})

    def relevance_inputs(self, ctx: PruningContext) -> tuple[torch.Tensor, torch.Tensor, bool]:
        space = self.params.get("relevance_space", self.params.get("embedding_space", "auto"))
        has_clip = Feature.CLIP_IMAGE_EMBEDS in ctx.features and Feature.CLIP_TEXT_EMBEDS in ctx.features
        use_clip = space == "clip" or (space == "auto" and has_clip)
        negate = self.params.get("negate_relevance", "auto")
        negate = use_clip if negate == "auto" else bool(negate)
        if use_clip:
            return ctx[Feature.CLIP_IMAGE_EMBEDS], ctx[Feature.CLIP_TEXT_EMBEDS], negate
        return ctx.visual_tokens, ctx[Feature.TEXT_EMBEDS], negate

    def prune(self, ctx: PruningContext) -> PruneResult:
        rel_x, q, negate = self.relevance_inputs(ctx)
        kernel = conditional_kernel(ctx.visual_tokens, q, self.params.get("theta"), rel_x=rel_x, negate=negate)
        keep = dpp_greedy_map(kernel, ctx.budget)
        return PruneResult.select(ctx.visual_tokens, keep)


@METHODS.register("dppruner")
class DPPruner(PreLLMPruner):
    def prune(self, ctx: PruningContext) -> PruneResult:
        keep = dpp_greedy_map(conditional_kernel(ctx.visual_tokens, None, None), ctx.budget)
        return PruneResult.select(ctx.visual_tokens, keep)
