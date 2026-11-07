"""Model adapter interface.

An adapter wraps one HF model family and owns everything model-specific:
preprocessing, which features it can expose to pruners, how to splice pruned
visual tokens back into the sequence (position ids included), and the LLM
forward call. The generation loop itself is shared (``generate``), so every
model is timed and decoded the same way.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from vtb.core.types import Feature, Prediction, PruneResult, PruningContext, Sample
from vtb.methods.base import BasePruner, InLLMPruner, PreLLMPruner

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class Prepared:
    """Output of ``prepare``: features for the pruner plus adapter-private state."""

    features: dict[Feature, Any]
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def num_visual(self) -> int:
        return self.features[Feature.VISUAL_TOKENS].shape[0]


@dataclass
class Assembled:
    inputs_embeds: torch.Tensor  # [1, L, D]
    position_ids: torch.Tensor  # model-specific shape
    next_position: int  # position id of the first generated token


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class BaseVLMAdapter(ABC):
    registry_name: ClassVar[str]
    provides: ClassVar[frozenset[Feature]] = frozenset({Feature.VISUAL_TOKENS})

    def __init__(
        self,
        pretrained: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        name: str | None = None,
        type: str | None = None,  # noqa: A002 - config key, unused here
        **kwargs: Any,
    ):
        self.pretrained = pretrained
        self.name = name or pretrained
        self.dtype = DTYPES[dtype]
        self.device = torch.device(device)
        self.attn_implementation = attn_implementation
        self.cfg = kwargs
        self.provides = self.features_for({"pretrained": pretrained, **kwargs})
        self._wanted: set[Feature] = set(self.provides)  # narrowed per pruner in generate()
        self.load()

    # ---- to implement per model family -------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load weights/processor onto ``self.device``."""

    @abstractmethod
    def prepare(self, sample: Sample, prompt: str) -> Prepared:
        """Tokenise, run the vision encoder + projector, collect the features in ``provides``."""

    @abstractmethod
    def assemble(self, prepared: Prepared, result: PruneResult) -> Assembled:
        """Splice (pruned) visual tokens into the text embeddings and build position ids."""

    @abstractmethod
    def lm_step(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, cache: Any) -> torch.Tensor:
        """One LLM forward with KV cache; return logits of the last position, shape [V]."""

    @abstractmethod
    def new_cache(self) -> Any: ...

    @abstractmethod
    def embed_token(self, token_id: int) -> torch.Tensor:
        """[1, 1, D] input embedding for a generated token."""

    @abstractmethod
    def decode_position_ids(self, position: int) -> torch.Tensor:
        """Position ids for a single decode step at ``position``."""

    @property
    @abstractmethod
    def eos_token_ids(self) -> set[int]: ...

    @abstractmethod
    def decode_text(self, token_ids: list[int]) -> str: ...

    # ---- shared logic ---------------------------------------------------------------------
    @classmethod
    def features_for(cls, cfg: dict) -> frozenset[Feature]:
        """Features an instance built from ``cfg`` will provide (checked before loading weights)."""
        return cls.provides

    def supports(self, pruner: BasePruner) -> set[Feature]:
        """Features ``pruner`` needs that this adapter cannot provide (empty set == OK)."""
        return pruner.missing_features(set(self.provides))

    def needs(self, pruner: BasePruner) -> set[Feature]:
        """Features to actually compute for this pruner (avoid e.g. attention output when unused)."""
        return (set(pruner.requires) | set(pruner.optional)) & set(self.provides)

    # ---- in-LLM pruning (optional; adapters that provide Feature.LLM_ATTN implement these) ------
    def prefill_in_llm(self, prepared: Prepared, asm: Assembled, pruner: InLLMPruner, cache: Any) -> tuple[torch.Tensor, dict]:
        """Layer-by-layer prefill that lets ``pruner`` drop visual tokens.

        Returns last-position logits and ``{"num_kept": int, "avg_layer_seq_len": float}``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support in-LLM pruning")

    def manual_step(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, cache: Any) -> torch.Tensor:
        """Decode step compatible with the per-layer KV cache left by ``prefill_in_llm``."""
        raise NotImplementedError

    @torch.inference_mode()
    def generate(self, sample: Sample, prompt: str, pruner: BasePruner, max_new_tokens: int = 16) -> Prediction:
        self._wanted = self.needs(pruner)
        in_llm = isinstance(pruner, InLLMPruner)

        _sync(self.device)
        t0 = time.perf_counter()
        prepared = self.prepare(sample, prompt)
        _sync(self.device)
        t1 = time.perf_counter()

        n = prepared.num_visual
        if isinstance(pruner, PreLLMPruner):
            ctx = PruningContext(budget=pruner.budget, features=prepared.features,
                                 instruction=sample.question, meta={"sample_id": sample.id})
            result = pruner(ctx)
        else:  # in-LLM methods see every visual token at layer 0
            vt = prepared.features[Feature.VISUAL_TOKENS]
            result = PruneResult(tokens=vt, indices=torch.arange(n, device=vt.device))
        asm = self.assemble(prepared, result)
        _sync(self.device)
        t2 = time.perf_counter()

        cache = self.new_cache()
        seq_len = int(asm.inputs_embeds.shape[1])
        if in_llm:
            logits, info = self.prefill_in_llm(prepared, asm, pruner, cache)
            step = self.manual_step
            kept, avg_len = info["num_kept"], info["avg_layer_seq_len"]
        else:
            logits = self.lm_step(asm.inputs_embeds, asm.position_ids, cache)
            step = self.lm_step
            kept, avg_len = int(result.tokens.shape[0]), float(seq_len)
        token = int(torch.argmax(logits))
        _sync(self.device)
        t3 = time.perf_counter()

        generated: list[int] = []
        pos = asm.next_position
        eos = self.eos_token_ids
        while token not in eos and len(generated) < max_new_tokens:
            generated.append(token)
            if len(generated) == max_new_tokens:
                break
            logits = step(self.embed_token(token), self.decode_position_ids(pos), cache)
            token = int(torch.argmax(logits))
            pos += 1
        _sync(self.device)
        t4 = time.perf_counter()

        return Prediction(
            sample_id=sample.id,
            output=self.decode_text(generated).strip(),
            num_visual_tokens=n,
            num_kept_tokens=kept,
            seq_len=seq_len,
            avg_layer_seq_len=avg_len,
            vision_ms=(t1 - t0) * 1e3,
            prune_ms=(t2 - t1) * 1e3,
            prefill_ms=(t3 - t2) * 1e3,
            decode_ms=(t4 - t3) * 1e3,
            num_generated=len(generated),
        )

    def llm_dims(self) -> dict[str, int]:
        """num_layers / hidden / intermediate, used for the FLOPs estimate."""
        return {}
