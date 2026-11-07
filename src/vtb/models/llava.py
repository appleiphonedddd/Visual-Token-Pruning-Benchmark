"""LLaVA family adapters (HF ``llava-hf/*`` checkpoints).

* ``llava``            LLaVA-1.5 (CLIP ViT-L/336 + [CLS], 576 tokens on a 24x24 grid).
                       Exposes encoder internals of the select layer (``vision_feature_layer``,
                       -2 by default): [CLS]->patch attention, head-averaged keys, pre-projector
                       features and the projector itself, which VisionZip / PruMerge+ / TRIM use.
                       With ``clip_text_encoder: openai/clip-vit-large-patch14-336`` it also
                       exposes CLIP joint-space image/text embeddings (TRIM, CDPruner's LLaVA setting).
* ``llava_next``       LLaVA-NeXT (anyres tiles + unpadding + newline tokens).
* ``llava_onevision``  LLaVA-OneVision (SigLIP, anyres_max_9).

The anyres models pack several tiles and newline tokens into one sequence, so
only projector tokens and the instruction embedding are exposed for them.
All use 1D RoPE: positions are re-numbered after pre-LLM pruning.

Note: HF chat templates omit the LLaVA-1.5 "A chat between a curious human..."
system prompt, so absolute scores can differ slightly from the original repo.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from vtb.core.registry import MODELS
from vtb.core.types import Feature
from vtb.models.hf_base import AttentionCapture, HFVLMAdapter, find_encoder_layers

CLIP_FEATURES = frozenset({Feature.CLIP_IMAGE_EMBEDS, Feature.CLIP_TEXT_EMBEDS, Feature.CLIP_TEXT_TOKENS})


@MODELS.register("llava")
class LlavaAdapter(HFVLMAdapter):
    provides = frozenset({
        Feature.VISUAL_TOKENS, Feature.TEXT_EMBEDS, Feature.GRID_SHAPE, Feature.LLM_ATTN, Feature.PROJECTOR,
        Feature.VISION_FEATS, Feature.VISION_ATTN, Feature.VISION_KEYS, Feature.VISION_QK, Feature.CLS_TOKEN,
    })

    _capture: AttentionCapture | None = None

    @classmethod
    def features_for(cls, cfg: dict) -> frozenset[Feature]:
        return cls.provides | CLIP_FEATURES if cfg.get("clip_text_encoder") else cls.provides

    def load(self) -> None:
        super().load()
        self.clip = None
        if name := self.cfg.get("clip_text_encoder"):
            from transformers import CLIPModel, CLIPTokenizer

            clip = CLIPModel.from_pretrained(name, dtype=self.dtype)
            clip.vision_model.encoder = None  # LLaVA's own tower provides the hidden states; keep only the heads
            self.clip = clip.to(self.device).eval()
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(name)

    @property
    def _layer(self) -> int:
        layer = self.model.config.vision_feature_layer
        if not isinstance(layer, int):
            raise NotImplementedError("multi-layer vision features are not supported for encoder-attention features")
        return layer

    @property
    def _has_cls(self) -> bool:
        return self.model.config.vision_feature_select_strategy == "default"

    def before_vision(self) -> None:
        if self._wanted & {Feature.VISION_ATTN, Feature.VISION_KEYS, Feature.VISION_QK}:
            layers = find_encoder_layers(self.core.vision_tower)
            self._capture = AttentionCapture(layers[self._layer].self_attn)

    def after_vision(self) -> None:
        if self._capture is not None:
            self._capture.remove()

    def extra_features(self, inputs: dict, visual_tokens: torch.Tensor, vision_out: Any) -> dict[Feature, Any]:
        n = visual_tokens.shape[0]
        off = 1 if self._has_cls else 0
        feats: dict[Feature, Any] = {}
        if self._has_cls:
            feats[Feature.CLS_TOKEN] = vision_out.hidden_states[self._layer][0, 0]  # pre-projector [CLS]
        side = math.isqrt(n)
        if side * side == n:
            feats[Feature.GRID_SHAPE] = (side, side)
        if Feature.VISION_FEATS in self._wanted:
            feats[Feature.VISION_FEATS] = vision_out.hidden_states[self._layer][0, off:]
        if Feature.PROJECTOR in self._wanted:
            proj = self.core.multi_modal_projector
            feats[Feature.PROJECTOR] = lambda x: proj(x[None].to(self.dtype))[0]

        cap, self._capture = self._capture, None
        if cap is not None:
            if Feature.VISION_ATTN in self._wanted:
                attn = cap.attention()  # [heads, S, S]
                # [CLS] -> patch attention if there is a [CLS]; otherwise mean attention received
                received = attn[:, 0, off:] if self._has_cls else attn.mean(dim=1)
                feats[Feature.VISION_ATTN] = received.mean(dim=0)  # [N]
            if Feature.VISION_KEYS in self._wanted:
                _, k = cap.qk()  # [heads, S, hd]
                feats[Feature.VISION_KEYS] = k.mean(dim=0)[off:]  # head-averaged, as VisionZip's metric
            if Feature.VISION_QK in self._wanted:
                feats[Feature.VISION_QK] = cap.raw_qk()  # PruMerge computes single-head attention from these

        if self.clip is not None and self._wanted & CLIP_FEATURES:
            feats.update(self._clip_features(vision_out, off))
        return feats

    def _clip_features(self, vision_out: Any, off: int) -> dict[Feature, Any]:
        """Patch tokens and instruction mapped into CLIP's joint space.

        Image: ``visual_projection(post_layernorm(h))`` of the patch tokens of the
        select layer (``clip_image_layer``, default ``vision_feature_layer`` = -2), as in
        the official CDPruner and TRIM code.
        """
        clip = self.clip
        layer = self.cfg.get("clip_image_layer", self._layer)
        h = vision_out.hidden_states[layer][0, off:]
        img = clip.visual_projection(clip.vision_model.post_layernorm(h))
        tok = self.clip_tokenizer(self._question, return_tensors="pt", truncation=True, max_length=77).to(self.device)
        out = clip.text_model(**tok)
        return {
            Feature.CLIP_IMAGE_EMBEDS: img,
            Feature.CLIP_TEXT_TOKENS: clip.text_projection(out.last_hidden_state[0]),
            Feature.CLIP_TEXT_EMBEDS: clip.text_projection(out.pooler_output[0]),
        }

    def prepare(self, sample, prompt):
        self._question = sample.question
        return super().prepare(sample, prompt)


@MODELS.register("llava_next")
class LlavaNextAdapter(HFVLMAdapter):
    provides = frozenset({Feature.VISUAL_TOKENS, Feature.TEXT_EMBEDS, Feature.LLM_ATTN})


@MODELS.register("llava_onevision")
class LlavaOnevisionAdapter(HFVLMAdapter):
    provides = frozenset({Feature.VISUAL_TOKENS, Feature.TEXT_EMBEDS, Feature.LLM_ATTN})
