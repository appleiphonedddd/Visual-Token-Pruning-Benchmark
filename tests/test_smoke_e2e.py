"""GPU end-to-end checks (run with ``pytest -m gpu``).

Every model config that fits a 12 GB GPU is checked; weights + lmms-lab/POPE must
be downloadable or cached. Select one model with ``-k qwen`` etc.
"""

import gc

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

LOCAL_MODELS = ["qwen2_5_vl_3b", "llava_ov_0_5b", "internvl3_2b", "llava_1_5_7b_4bit",
                "llava_1_5_7b_clip_4bit", "llava_next_7b_4bit"]


@pytest.fixture(scope="module", params=LOCAL_MODELS)
def adapter(request):
    from vtb.core.config import load_component
    from vtb.core.registry import MODELS

    cfg = load_component("models", request.param)
    a = MODELS.build(cfg.pop("type"), **cfg)
    yield a
    del a
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def samples():
    from vtb.core.registry import DATASETS

    ds = DATASETS.build("pope", path="lmms-lab/POPE", split="test", limit=4)
    return ds, list(ds)


@torch.inference_mode()
def test_vanilla_matches_hf_generate(adapter, samples):
    """Our splice + manual decode loop must reproduce HF generate() token-for-token."""
    from vtb.core.registry import METHODS

    ds, items = samples
    vanilla = METHODS.build("vanilla")
    for s in items:
        prompt = ds.format_prompt(s)
        ours = adapter.generate(s, prompt, vanilla, max_new_tokens=12).output
        inputs = adapter.build_inputs(s, prompt)
        out = adapter.model.generate(**inputs, max_new_tokens=12, do_sample=False)
        ref = adapter.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        assert ours == ref, f"{adapter.name} {s.id}: ours={ours!r} hf={ref!r}"


@torch.inference_mode()
def test_full_budget_random_equals_vanilla(adapter, samples):
    from vtb.core.registry import METHODS

    ds, items = samples
    s = items[0]
    n = adapter.prepare(s, ds.format_prompt(s)).num_visual
    a = adapter.generate(s, ds.format_prompt(s), METHODS.build("vanilla"), 12)
    b = adapter.generate(s, ds.format_prompt(s), METHODS.build("random", budget=n), 12)
    assert a.output == b.output


@torch.inference_mode()
def test_declared_features_are_provided(adapter, samples):
    """Every feature in ``provides`` must actually be produced (when requested)."""
    from vtb.core.types import Feature

    ds, items = samples
    adapter._wanted = set(adapter.provides)
    prep = adapter.prepare(items[0], ds.format_prompt(items[0]))
    n = prep.num_visual
    for f in adapter.provides - {Feature.LLM_ATTN}:  # LLM_ATTN is a capability, not a prepared tensor
        assert f in prep.features, f"{adapter.name} declares {f.value} but did not provide it"
    if Feature.GRID_SHAPE in prep.features:
        h, w = prep.features[Feature.GRID_SHAPE]
        assert h * w == n
    for f in (Feature.VISION_ATTN, Feature.VISION_KEYS, Feature.VISION_FEATS):
        if f in prep.features:
            assert prep.features[f].shape[0] == n, f"{f.value}: {tuple(prep.features[f].shape)} vs {n} tokens"
    if Feature.VISION_ATTN in prep.features:
        assert torch.all(prep.features[Feature.VISION_ATTN] >= 0)


@torch.inference_mode()
@pytest.mark.parametrize("method", ["random", "pooling", "divprune", "cdpruner", "visionzip", "prumerge_plus",
                                    "trim", "fastv", "pdrop", "dart"])
def test_pruned_runs(adapter, samples, method):
    from vtb.core.registry import METHODS

    ds, items = samples
    pruner = METHODS.build(method, budget=64)
    if adapter.supports(pruner):
        pytest.skip(f"{adapter.name} lacks {adapter.supports(pruner)}")
    pred = adapter.generate(items[0], ds.format_prompt(items[0]), pruner, 8)
    assert pred.num_kept_tokens <= 64
    if pruner.stage == "pre_llm":
        assert pred.seq_len < pred.num_visual_tokens
    else:  # full sequence enters layer 0, shrinks from layer K on
        assert pred.avg_layer_seq_len < pred.seq_len
    assert pred.output


@torch.inference_mode()
@pytest.mark.parametrize("method", ["fastv", "dart"])
def test_in_llm_loop_without_pruning_equals_vanilla(adapter, samples, method):
    """Our layer-by-layer prefill/decode (budget >= #visual tokens -> nothing dropped) must match vanilla."""
    from vtb.core.registry import METHODS

    ds, items = samples
    for s in items[:2]:
        prompt = ds.format_prompt(s)
        ref = adapter.generate(s, prompt, METHODS.build("vanilla"), 12)
        ours = adapter.generate(s, prompt, METHODS.build(method, budget=10**6), 12)
        assert ours.output == ref.output, f"{adapter.name} {s.id}: {ours.output!r} vs {ref.output!r}"
        assert ours.num_kept_tokens == ref.num_visual_tokens


@torch.inference_mode()
def test_last_token_attention_matches_eager(adapter, samples):
    """Recomputed head-averaged attention (GQA + RoPE) == HF eager attention weights."""
    ds, items = samples
    s = items[0]
    adapter._wanted = set()
    from vtb.core.types import PruneResult

    prep = adapter.prepare(s, ds.format_prompt(s))
    vt = prep.features[next(iter(prep.features))]
    asm = adapter.assemble(prep, PruneResult(tokens=vt, indices=torch.arange(vt.shape[0], device=vt.device)))
    if asm.inputs_embeds.shape[1] > 2000:
        pytest.skip("eager attention for every layer does not fit in 12 GB at this length "
                    "(the same LLM is covered by a shorter-sequence config)")
    lm = adapter.core.language_model
    layer_idx = 1
    adapter.model.set_attn_implementation("eager")
    try:
        out = lm(inputs_embeds=asm.inputs_embeds, position_ids=asm.position_ids,
                 output_attentions=True, output_hidden_states=True)
    finally:
        adapter.model.set_attn_implementation(adapter.attn_implementation)
    ref = out.attentions[layer_idx][0, :, -1].float().mean(0)  # [L]
    h = out.hidden_states[layer_idx]
    pe = lm.rotary_emb(h, asm.position_ids)
    ours = adapter._last_token_attention(lm.layers[layer_idx], h, pe)
    # eager runs QK^T in bf16, we use fp32: compare distributions, not exact values.
    # (dropping RoPE, for reference, gives cos ~0.86 and ~40/64 top-64 overlap)
    cos = torch.nn.functional.cosine_similarity(ours, ref, dim=0)
    k = min(64, ours.numel())
    overlap = len(set(torch.topk(ours, k).indices.tolist()) & set(torch.topk(ref, k).indices.tolist()))
    assert cos > 0.999 and overlap >= 0.9 * k, (float(cos), overlap)
