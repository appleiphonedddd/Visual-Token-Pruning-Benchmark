"""Shared adapter for HF ``*ForConditionalGeneration`` VLMs.

All supported HF VLMs share one layout: ``model.model.get_image_features(...)``
returns projector outputs, ``model.model.language_model`` is the decoder, and
``model.lm_head`` produces logits. The processor expands the image placeholder
into one token per visual feature, so pruning = replacing that span.

Families override only what differs:
  * ``position_ids`` / ``reindex_positions``: 1D RoPE (re-numbered after pruning) vs. M-RoPE
  * ``extra_features``: encoder attention, [CLS], grid shape, ...
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, DynamicCache

from vtb.core.types import Feature, PruneResult, Sample
from vtb.models.base import Assembled, BaseVLMAdapter, Prepared

_NON_VISION_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}


class HFVLMAdapter(BaseVLMAdapter):
    model_cls: Any = AutoModelForImageTextToText
    # True: positions are re-numbered 0..L'-1 after pruning (1D RoPE models, as in FastV/CDPruner on LLaVA).
    reindex_positions: bool = True

    # ---- loading --------------------------------------------------------------------------
    def load(self) -> None:
        kwargs: dict[str, Any] = {"dtype": self.dtype, "attn_implementation": self.attn_implementation}
        quant = self.cfg.get("quantization")
        if quant:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant == "4bit",
                load_in_8bit=quant == "8bit",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
                # keep the vision side + head in full precision; only the LLM decoder is quantised
                llm_int8_skip_modules=["vision_tower", "visual", "multi_modal_projector", "lm_head"],
            )
            kwargs["device_map"] = {"": self.device.index or 0}
        self.model = self.model_cls.from_pretrained(self.pretrained, **kwargs).eval()
        if not quant:
            self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.pretrained, **(self.cfg.get("processor_kwargs") or {}))
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = self.model.config.image_token_id
        size = self.cfg.get("image_size")  # [W, H] or None for native resolution
        self.image_size = tuple(size) if size else None

        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self._eos = {e for e in [*eos, self.tokenizer.eos_token_id] if e is not None}

    @property
    def core(self):
        return self.model.model

    # ---- preprocessing --------------------------------------------------------------------
    def build_inputs(self, sample: Sample, prompt: str) -> dict:
        if len(sample.images) != 1:
            raise NotImplementedError(f"{type(self).__name__} currently handles exactly one image per sample")
        image = sample.images[0].convert("RGB")
        if self.image_size:
            image = image.resize(self.image_size)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        call_kwargs = self.cfg.get("processor_call_kwargs") or {}  # e.g. InternVL {crop_to_patches: false}
        return self.processor(text=[text], images=[image], return_tensors="pt", **call_kwargs).to(self.device)

    def image_features(self, inputs: dict) -> tuple[torch.Tensor, Any]:
        """Projector output [N, D] for the (single) image, plus the raw vision outputs."""
        vision_kwargs = {k: v for k, v in inputs.items() if k not in _NON_VISION_KEYS}
        if "pixel_values" in vision_kwargs:
            vision_kwargs["pixel_values"] = vision_kwargs["pixel_values"].to(self.dtype)
        out = self.core.get_image_features(**vision_kwargs, return_dict=True)
        feats = out.pooler_output
        feats = torch.cat(list(feats), dim=0) if isinstance(feats, (list, tuple)) else feats.reshape(-1, feats.shape[-1])
        return feats.to(self.dtype), out

    def position_ids(self, inputs: dict) -> torch.Tensor:
        """[P, L] position ids of the unpruned sequence (P = 1 for 1D RoPE)."""
        return torch.arange(inputs["input_ids"].shape[1], device=self.device)[None]

    def extra_features(self, inputs: dict, visual_tokens: torch.Tensor, vision_out: Any) -> dict[Feature, Any]:
        return {}

    def before_vision(self) -> None:
        """Hook to install capture hooks (e.g. encoder attention) before the vision forward."""

    def after_vision(self) -> None:
        """Remove anything installed in ``before_vision``."""

    def prepare(self, sample: Sample, prompt: str) -> Prepared:
        inputs = self.build_inputs(sample, prompt)
        input_ids = inputs["input_ids"]
        self.before_vision()
        try:
            visual_tokens, vision_out = self.image_features(inputs)
        finally:
            self.after_vision()

        embed = self.core.get_input_embeddings()
        text_embeds = embed(input_ids)[0]  # [L, D]

        img_pos = (input_ids[0] == self.image_token_id).nonzero().squeeze(-1)
        start, end = int(img_pos[0]), int(img_pos[-1]) + 1
        if end - start != visual_tokens.shape[0] or len(img_pos) != end - start:
            raise RuntimeError(f"image placeholder span {end - start} does not match {visual_tokens.shape[0]} features")

        features: dict[Feature, Any] = {Feature.VISUAL_TOKENS: visual_tokens}
        if Feature.TEXT_EMBEDS in self._wanted:
            q_ids = self.tokenizer(sample.question, return_tensors="pt", add_special_tokens=False).input_ids
            features[Feature.TEXT_EMBEDS] = embed(q_ids.to(self.device))[0].mean(dim=0)
        features.update(self.extra_features(inputs, visual_tokens, vision_out))
        return Prepared(
            features=features,
            state={"text_embeds": text_embeds, "position_ids": self.position_ids(inputs), "span": (start, end)},
        )

    def assemble(self, prepared: Prepared, result: PruneResult) -> Assembled:
        st = prepared.state
        start, end = st["span"]
        emb, pos = st["text_embeds"], st["position_ids"]  # [L, D], [P, L]
        new_emb = torch.cat([emb[:start], result.tokens.to(emb.dtype), emb[end:]], dim=0)
        if self.reindex_positions:
            new_pos = torch.arange(new_emb.shape[0], device=pos.device).expand(pos.shape[0], -1)
            next_pos = new_emb.shape[0]
        else:
            idx = result.indices.to(pos.device)
            new_pos = torch.cat([pos[:, :start], pos[:, start:end][:, idx], pos[:, end:]], dim=1)
            next_pos = int(pos.max()) + 1
        return Assembled(inputs_embeds=new_emb.unsqueeze(0), position_ids=self.format_positions(new_pos),
                         next_position=next_pos)

    def format_positions(self, pos: torch.Tensor) -> torch.Tensor:
        """[P, L] -> what the decoder expects: [1, L] for 1D, [3, 1, L] for M-RoPE."""
        return pos if pos.shape[0] == 1 else pos.unsqueeze(1)

    # ---- LLM ------------------------------------------------------------------------------
    def new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.model.config.get_text_config())

    def lm_step(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, cache: Any) -> torch.Tensor:
        out = self.core.language_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, past_key_values=cache, use_cache=True
        )
        return self.model.lm_head(out.last_hidden_state[:, -1])[0].float()

    def embed_token(self, token_id: int) -> torch.Tensor:
        return self.core.get_input_embeddings()(torch.tensor([[token_id]], device=self.device))

    def decode_position_ids(self, position: int) -> torch.Tensor:
        p = self.state_positions_dim
        return self.format_positions(torch.full((p, 1), position, dtype=torch.long, device=self.device))

    state_positions_dim: int = 1

    @property
    def eos_token_ids(self) -> set[int]:
        return self._eos

    def decode_text(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def llm_dims(self) -> dict[str, int]:
        c = self.model.config.get_text_config()
        return {"num_layers": c.num_hidden_layers, "hidden": c.hidden_size, "intermediate": c.intermediate_size}

    # ---- in-LLM pruning: our own layer loop ------------------------------------------------
    # With batch size 1, no padding and SDPA, ``attention_mask=None`` gives a causal prefill
    # and full attention for single-token decode steps, so each layer's KV cache may hold a
    # different number of tokens after pruning.
    def _check_attn_impl(self) -> None:
        if self.attn_implementation not in ("sdpa", "flash_attention_2"):
            raise ValueError("in-LLM pruning needs attn_implementation sdpa or flash_attention_2")

    def manual_step(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, cache: Any) -> torch.Tensor:
        lm = self.core.language_model
        h = inputs_embeds
        pe = lm.rotary_emb(h, position_ids)
        for layer in lm.layers:
            h = layer(h, attention_mask=None, position_embeddings=pe, past_key_values=cache, use_cache=True)
        return self.model.lm_head(lm.norm(h)[:, -1])[0].float()

    def prefill_in_llm(self, prepared, asm, pruner, cache) -> tuple[torch.Tensor, dict]:
        from vtb.methods.base import LLMLayerContext

        self._check_attn_impl()
        lm = self.core.language_model
        num_layers = len(lm.layers)
        start, end = prepared.state["span"]
        vis = torch.arange(start, end, device=self.device)  # visual positions in the current sequence
        schedule = {l: k for l, k in pruner.schedule(end - start, num_layers) if 0 < l < num_layers}

        h, pos = asm.inputs_embeds, asm.position_ids
        pe = lm.rotary_emb(h, pos)
        attn_prev: torch.Tensor | None = None
        keys_prev: torch.Tensor | None = None
        lens: list[int] = []
        for i, layer in enumerate(lm.layers):
            if i in schedule and schedule[i] < vis.numel():
                ctx = LLMLayerContext(layer_idx=i, budget=schedule[i], hidden_states=h[0], visual_positions=vis,
                                      last_token_attn=attn_prev, meta={}, key_states=keys_prev,
                                      normed_hidden=lm.norm(h)[0] if pruner.needs_keys else None)
                keep_vis = torch.sort(pruner.select(ctx).to(self.device)).values
                keep = torch.ones(h.shape[1], dtype=torch.bool, device=self.device)
                keep[vis] = False
                keep[vis[keep_vis]] = True
                new_index = torch.cumsum(keep, 0) - 1
                vis = new_index[vis[keep_vis]]
                idx = keep.nonzero().squeeze(-1)
                h, pos = h[:, idx], pos[..., idx]
                pe = lm.rotary_emb(h, pos)
            attn_prev = (self._last_token_attention(layer, h, pe)
                         if (i + 1) in schedule and pruner.needs_attention else None)
            keys_prev = self._key_states(layer, h, pe) if (i + 1) in schedule and pruner.needs_keys else None
            h = layer(h, attention_mask=None, position_embeddings=pe, past_key_values=cache, use_cache=True)
            lens.append(h.shape[1])
        logits = self.model.lm_head(lm.norm(h)[:, -1])[0].float()
        return logits, {"num_kept": int(vis.numel()), "avg_layer_seq_len": sum(lens) / len(lens)}

    @staticmethod
    def _key_states(layer, h: torch.Tensor, pe: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Post-RoPE keys of ``layer`` for every position: [L, kv_heads * hd]."""
        return _key_states_impl(layer, h, pe)

    @staticmethod
    def _last_token_attention(layer, h: torch.Tensor, pe: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Head-averaged attention of the last position over the whole sequence in ``layer``: [L]."""
        attn = layer.self_attn
        x = layer.input_layernorm(h)
        hd = attn.head_dim
        q = attn.q_proj(x[:, -1:]).view(1, 1, -1, hd)
        k = attn.k_proj(x).view(1, x.shape[1], -1, hd)
        if hasattr(attn, "q_norm"):  # Qwen3-style per-head norms
            q, k = attn.q_norm(q), attn.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)  # [1, heads, len, hd]
        cos, sin = (t.unsqueeze(1) for t in pe)  # [1, 1, L, hd]
        q = q * cos[:, :, -1:] + _rotate_half(q) * sin[:, :, -1:]
        k = k * cos + _rotate_half(k) * sin
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)  # GQA
        scale = getattr(attn, "scaling", hd ** -0.5)
        w = torch.softmax((q.float() @ k.float().transpose(-1, -2)) * scale, dim=-1)  # [1, heads, 1, L]
        return w[0, :, 0].mean(dim=0)


def _key_states_impl(layer, h: torch.Tensor, pe: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    attn = layer.self_attn
    x = layer.input_layernorm(h)
    k = attn.k_proj(x).view(1, x.shape[1], -1, attn.head_dim)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)
    k = k.transpose(1, 2)  # [1, kv_heads, L, hd]
    cos, sin = (t.unsqueeze(1) for t in pe)
    k = k * cos + _rotate_half(k) * sin
    return k[0].transpose(0, 1).reshape(x.shape[1], -1)  # [L, kv_heads * hd]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# ---- vision-encoder attention capture (shared by CLIP / SigLIP / InternViT towers) ------------------
def find_encoder_layers(vision_tower: torch.nn.Module) -> torch.nn.ModuleList:
    for path in ("vision_model.encoder.layers", "encoder.layers", "encoder.layer", "blocks"):
        mod: Any = vision_tower
        try:
            for part in path.split("."):
                mod = getattr(mod, part)
        except AttributeError:
            continue
        return mod
    raise AttributeError(f"cannot locate encoder layers in {type(vision_tower).__name__}")


class AttentionCapture:
    """Records the input of one encoder layer's self-attention and recomputes q/k.

    Works with SDPA/flash attention (no ``output_attentions`` needed).
    """

    def __init__(self, attn_module: torch.nn.Module):
        self.attn = attn_module
        self.hidden: torch.Tensor | None = None
        self.handle = attn_module.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        self.hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]

    def remove(self) -> None:
        self.handle.remove()

    def raw_qk(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Un-split q, k [S, D] of the first image, plus the number of heads."""
        a, h = self.attn, self.hidden[:1]
        nh = getattr(a, "num_heads", None) or a.config.num_attention_heads
        return a.q_proj(h)[0], a.k_proj(h)[0], nh

    def qk(self) -> tuple[torch.Tensor, torch.Tensor]:
        """q, k of the first image in the batch: [heads, S, head_dim]."""
        a, h = self.attn, self.hidden[:1]
        nh = getattr(a, "num_heads", None) or a.config.num_attention_heads
        q = a.q_proj(h).view(h.shape[0], h.shape[1], nh, -1).transpose(1, 2)[0]
        k = a.k_proj(h).view(h.shape[0], h.shape[1], nh, -1).transpose(1, 2)[0]
        return q, k

    def attention(self) -> torch.Tensor:
        q, k = self.qk()
        scale = getattr(self.attn, "scale", q.shape[-1] ** -0.5)
        return torch.softmax((q.float() @ k.float().transpose(-1, -2)) * scale, dim=-1)  # [heads, S, S]
