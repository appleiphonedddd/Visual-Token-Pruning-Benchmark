"""Pruning method interfaces.

A method declares *where* it runs (``stage``) and *what* it needs (``requires``).
Model adapters declare what they provide; the runner skips incompatible pairs.
Methods only touch tensors, never model objects, so they are unit-testable with
synthetic inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import torch

from vtb.core.types import Feature, PruneResult, PruningContext


class BasePruner(ABC):
    registry_name: ClassVar[str]
    stage: ClassVar[Literal["pre_llm", "in_llm"]]
    requires: ClassVar[frozenset[Feature]] = frozenset({Feature.VISUAL_TOKENS})
    # used when the model provides them, never a reason to skip (e.g. CDPruner's CLIP embeddings)
    optional: ClassVar[frozenset[Feature]] = frozenset()

    def __init__(self, budget: int | None = None, seed: int = 0, **params: Any):
        self.budget = budget
        self.seed = seed
        self.params = params

    def missing_features(self, provided: set[Feature]) -> set[Feature]:
        return set(self.requires) - set(provided)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(budget={self.budget}, params={self.params})"


class PreLLMPruner(BasePruner):
    """Reduces visual tokens between the projector and the LLM."""

    stage = "pre_llm"

    def __call__(self, ctx: PruningContext) -> PruneResult:
        n = ctx.num_tokens
        if ctx.budget is None or ctx.budget >= n:
            return PruneResult(tokens=ctx.visual_tokens, indices=torch.arange(n, device=ctx.visual_tokens.device))
        out = self.prune(ctx)
        assert out.tokens.shape[0] == out.indices.shape[0], "tokens/indices length mismatch"
        assert out.tokens.shape[0] <= ctx.budget, f"{self} returned {out.tokens.shape[0]} > budget {ctx.budget}"
        return out

    @abstractmethod
    def prune(self, ctx: PruningContext) -> PruneResult:
        """Return at most ``ctx.budget`` tokens. Only called when pruning is needed."""


@dataclass
class LLMLayerContext:
    """What an in-LLM pruner sees right before decoder layer ``layer_idx`` during prefill."""

    layer_idx: int  # pruning happens before this layer runs
    budget: int  # visual tokens to keep from here on
    hidden_states: torch.Tensor  # [L, D] output of layer ``layer_idx - 1``
    visual_positions: torch.Tensor  # [n] where the remaining visual tokens sit in the current sequence
    last_token_attn: torch.Tensor | None  # [L] head-averaged attention of the last token in layer ``layer_idx - 1``
    meta: dict[str, Any]
    key_states: torch.Tensor | None = None  # [L, kv_heads * hd] post-RoPE keys of layer ``layer_idx - 1``
    normed_hidden: torch.Tensor | None = None  # [L, D] ``hidden_states`` after the LLM's final norm

    @property
    def text_positions(self) -> torch.Tensor:
        """Positions after the visual tokens (instruction + generation prompt)."""
        last = int(self.visual_positions.max())
        return torch.arange(last + 1, self.hidden_states.shape[0], device=self.hidden_states.device)

    @property
    def visual_attn(self) -> torch.Tensor:
        """[n] attention the last token pays to each remaining visual token."""
        return self.last_token_attn[self.visual_positions]

    @property
    def visual_hidden(self) -> torch.Tensor:
        return self.hidden_states[self.visual_positions]


class InLLMPruner(BasePruner):
    """Drops visual tokens inside the LLM during prefill (FastV, PDrop, DART, ...).

    The adapter runs the decoder layer by layer; before each scheduled layer it
    calls ``select`` and removes the dropped tokens from the hidden states, the
    position ids and every later layer's KV cache. Kept tokens keep their
    original position ids (as in the FastV reference implementation).
    """

    stage = "in_llm"
    requires = frozenset({Feature.VISUAL_TOKENS, Feature.LLM_ATTN})
    needs_attention: ClassVar[bool] = True  # False -> the adapter skips computing ``last_token_attn``
    needs_keys: ClassVar[bool] = False  # True -> ``key_states`` and ``normed_hidden`` are filled in

    @abstractmethod
    def schedule(self, num_visual: int, num_layers: int) -> list[tuple[int, int]]:
        """[(layer_idx, visual tokens to keep from that layer on), ...] in increasing layer order."""

    @abstractmethod
    def select(self, ctx: LLMLayerContext) -> torch.Tensor:
        """``ctx.budget`` indices into ``ctx.visual_positions`` to keep."""
