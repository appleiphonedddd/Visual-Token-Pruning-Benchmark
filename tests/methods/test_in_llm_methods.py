"""Contract tests for in-LLM pruners on a synthetic layer context."""

import pytest
import torch

from vtb.core.registry import METHODS
from vtb.methods.base import InLLMPruner, LLMLayerContext

IN_LLM = [n for n in METHODS.names() if issubclass(METHODS.get(n), InLLMPruner)]
L, D, N, START = 200, 32, 144, 20


def make_ctx(budget, seed=0):
    g = torch.Generator().manual_seed(seed)
    return LLMLayerContext(
        layer_idx=2, budget=budget,
        hidden_states=torch.randn(L, D, generator=g),
        visual_positions=torch.arange(START, START + N),
        last_token_attn=torch.softmax(torch.randn(L, generator=g), 0),
        meta={},
        key_states=torch.randn(L, 16, generator=g),
        normed_hidden=torch.randn(L, D, generator=g),
    )


def test_registered():
    assert {"fastv", "pdrop", "dart"} <= set(IN_LLM)


@pytest.mark.parametrize("name", IN_LLM)
@pytest.mark.parametrize("budget", [1, 16, 64])
def test_select_contract(name, budget):
    keep = METHODS.build(name, budget=budget).select(make_ctx(budget))
    assert keep.numel() == budget
    assert len(set(keep.tolist())) == budget
    assert keep.min() >= 0 and keep.max() < N


@pytest.mark.parametrize("name", IN_LLM)
def test_schedule(name):
    sched = METHODS.build(name, budget=64).schedule(576, 32)
    layers = [l for l, _ in sched]
    keeps = [k for _, k in sched]
    assert layers == sorted(layers) and all(0 < l < 32 for l in layers)
    assert keeps[0] == 64 and keeps == sorted(keeps, reverse=True)
    assert METHODS.build(name).schedule(576, 32) == []  # no budget -> no pruning


def test_fastv_picks_top_attention():
    ctx = make_ctx(16)
    keep = METHODS.build("fastv", budget=16).select(ctx)
    assert set(keep.tolist()) == set(torch.topk(ctx.visual_attn, 16).indices.tolist())


def test_pdrop_stages():
    assert METHODS.build("pdrop", budget=64).schedule(576, 32) == [(2, 64), (6, 32), (12, 16)]


def test_dart_keeps_key_norm_pivots():
    ctx = make_ctx(32)
    keep = set(METHODS.build("dart", budget=32).select(ctx).tolist())
    pivots = torch.topk(ctx.key_states[ctx.visual_positions].abs().sum(-1), 4).indices.tolist()  # L1 norm
    assert set(pivots) <= keep


def test_dart_prunes_before_layer_k():
    assert METHODS.build("dart", budget=64).schedule(576, 32) == [(2, 64)]


def test_text_positions():
    ctx = make_ctx(8)
    assert ctx.text_positions[0] == START + N and ctx.text_positions[-1] == L - 1
