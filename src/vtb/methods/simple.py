"""Reference baselines: no pruning, random selection, spatial average pooling."""

from __future__ import annotations

import math
import zlib

import torch
import torch.nn.functional as F

from vtb.core.registry import METHODS
from vtb.core.types import Feature, PruneResult, PruningContext
from vtb.methods.base import PreLLMPruner


@METHODS.register("vanilla")
class Vanilla(PreLLMPruner):
    """Upper bound: keep every token (budget is ignored)."""

    def __call__(self, ctx: PruningContext) -> PruneResult:
        n = ctx.num_tokens
        return PruneResult(tokens=ctx.visual_tokens, indices=torch.arange(n, device=ctx.visual_tokens.device))

    def prune(self, ctx: PruningContext) -> PruneResult:  # pragma: no cover - never reached
        raise NotImplementedError


@METHODS.register("random")
class RandomPruner(PreLLMPruner):
    """Uniformly random subset; seeded per sample so results don't depend on order."""

    def prune(self, ctx: PruningContext) -> PruneResult:
        sid = str(ctx.meta.get("sample_id", ""))
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + zlib.crc32(sid.encode()))
        keep = torch.randperm(ctx.num_tokens, generator=g)[: ctx.budget].to(ctx.visual_tokens.device)
        return PruneResult.select(ctx.visual_tokens, keep)


@METHODS.register("pooling")
class PoolingPruner(PreLLMPruner):
    """Adaptive 2D average pooling of the token grid (aspect ratio preserved)."""

    requires = frozenset({Feature.VISUAL_TOKENS, Feature.GRID_SHAPE})

    def prune(self, ctx: PruningContext) -> PruneResult:
        h, w = ctx[Feature.GRID_SHAPE]
        x = ctx.visual_tokens
        scale = math.sqrt(ctx.budget / (h * w))
        oh, ow = max(1, int(h * scale)), max(1, int(w * scale))
        grid = x.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0).float()
        pooled = F.adaptive_avg_pool2d(grid, (oh, ow))[0].permute(1, 2, 0).reshape(oh * ow, -1).to(x.dtype)
        # anchor each pooled cell at the original token nearest its centre (used for positions)
        rows = ((torch.arange(oh) + 0.5) * h / oh).long().clamp(max=h - 1)
        cols = ((torch.arange(ow) + 0.5) * w / ow).long().clamp(max=w - 1)
        idx = (rows[:, None] * w + cols[None, :]).reshape(-1).to(x.device)
        return PruneResult(tokens=pooled, indices=idx, merged=True)
