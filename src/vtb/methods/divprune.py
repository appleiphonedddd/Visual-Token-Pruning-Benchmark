"""DivPrune (Alvar et al., CVPR 2025): Max-Min Diversity Problem via greedy selection.

Distance is 1 - cosine similarity. The first token is the one whose nearest
neighbour is farthest; each next token maximises its minimum distance to the
already-selected set.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vtb.core.registry import METHODS
from vtb.core.types import PruneResult, PruningContext
from vtb.methods.base import PreLLMPruner


def divprune_select(x: torch.Tensor, k: int) -> torch.Tensor:
    xn = F.normalize(x.float(), dim=-1)
    dist = 1.0 - xn @ xn.T  # [N, N]
    n = dist.shape[0]
    selected = torch.empty(k, dtype=torch.long, device=x.device)

    nn_dist = dist.clone()
    nn_dist.fill_diagonal_(float("inf"))
    first = torch.argmax(nn_dist.min(dim=1).values)
    selected[0] = first

    min_dist = dist[first].clone()
    taken = torch.zeros(n, dtype=torch.bool, device=x.device)
    taken[first] = True
    for i in range(1, k):
        scores = min_dist.masked_fill(taken, -float("inf"))
        j = torch.argmax(scores)
        selected[i] = j
        taken[j] = True
        min_dist = torch.minimum(min_dist, dist[j])
    return selected


@METHODS.register("divprune")
class DivPrune(PreLLMPruner):
    def prune(self, ctx: PruningContext) -> PruneResult:
        keep = divprune_select(ctx.visual_tokens, ctx.budget)
        return PruneResult.select(ctx.visual_tokens, keep)
