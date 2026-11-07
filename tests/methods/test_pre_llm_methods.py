"""Generic contract every pre-LLM pruner must satisfy, run on synthetic tensors.

New methods are picked up automatically through the registry; give them the
features they need in ``make_ctx``.
"""

import itertools

import pytest
import torch

from vtb.core.registry import METHODS
from vtb.core.types import Feature, PruningContext
from vtb.methods.base import PreLLMPruner
from vtb.methods.cdpruner import conditional_kernel, dpp_greedy_map

H, W, D = 12, 12, 32
N = H * W


D_VIS, D_CLIP = 16, 12
_W = torch.randn(D_VIS, D, generator=torch.Generator().manual_seed(123))


def make_ctx(budget, seed=0, sample_id="s0"):
    """Synthetic context carrying every feature any pre-LLM method may require."""
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(N, D_VIS, generator=g)
    return PruningContext(
        budget=budget,
        features={
            Feature.VISUAL_TOKENS: feats @ _W,
            Feature.TEXT_EMBEDS: torch.randn(D, generator=g),
            Feature.GRID_SHAPE: (H, W),
            Feature.VISION_FEATS: feats,
            Feature.PROJECTOR: lambda x: x @ _W,
            Feature.VISION_ATTN: torch.softmax(3 * torch.randn(N, generator=g), 0),
            Feature.VISION_KEYS: torch.randn(N, 8, generator=g),
            Feature.CLS_TOKEN: torch.randn(D_VIS, generator=g),
            Feature.VISION_QK: (torch.randn(N + 1, D_VIS, generator=g), torch.randn(N + 1, D_VIS, generator=g), 2),
            Feature.CLIP_IMAGE_EMBEDS: torch.randn(N, D_CLIP, generator=g),
            Feature.CLIP_TEXT_EMBEDS: torch.randn(D_CLIP, generator=g),
            Feature.CLIP_TEXT_TOKENS: torch.randn(5, D_CLIP, generator=g),
        },
        instruction="Is there a dog?",
        meta={"sample_id": sample_id},
    )


PRE_LLM = [n for n in METHODS.names() if issubclass(METHODS.get(n), PreLLMPruner) and n != "vanilla"]


@pytest.mark.parametrize("name,budget", list(itertools.product(PRE_LLM, [1, 16, 64])))
def test_budget_and_indices(name, budget):
    out = METHODS.build(name, budget=budget)(make_ctx(budget))
    k = out.tokens.shape[0]
    assert 1 <= k <= budget
    assert out.indices.shape == (k,)
    assert out.tokens.shape[1] == D
    assert out.indices.min() >= 0 and out.indices.max() < N
    if not out.merged:
        assert torch.all(out.indices[1:] > out.indices[:-1]), "selection indices must be sorted and unique"
        assert len(set(out.indices.tolist())) == k
        assert torch.equal(out.tokens, make_ctx(budget).visual_tokens[out.indices])


@pytest.mark.parametrize("name", PRE_LLM)
def test_deterministic(name):
    a = METHODS.build(name, budget=32)(make_ctx(32))
    b = METHODS.build(name, budget=32)(make_ctx(32))
    assert torch.equal(a.indices, b.indices)


@pytest.mark.parametrize("name", PRE_LLM + ["vanilla"])
def test_full_budget_is_identity(name):
    ctx = make_ctx(N)
    out = METHODS.build(name, budget=N)(ctx)
    assert torch.equal(out.indices, torch.arange(N))
    assert torch.equal(out.tokens, ctx.visual_tokens)


def test_dpp_greedy_beats_random_logdet():
    ctx = make_ctx(16)
    kernel = conditional_kernel(ctx.visual_tokens, ctx[Feature.TEXT_EMBEDS], None).double()
    kernel = kernel + 1e-6 * torch.eye(N, dtype=kernel.dtype)
    greedy = dpp_greedy_map(kernel, 16)
    ld = lambda idx: torch.logdet(kernel[idx][:, idx])  # noqa: E731
    g = torch.Generator().manual_seed(1)
    rand = [ld(torch.randperm(N, generator=g)[:16]) for _ in range(50)]
    assert ld(greedy) >= max(rand)


def test_dpp_greedy_matches_bruteforce_first_steps():
    # greedy step i must maximise det of the selected submatrix among one-element extensions
    ctx = make_ctx(4)
    kernel = conditional_kernel(ctx.visual_tokens, None, None).double()
    sel = dpp_greedy_map(kernel, 4).tolist()
    for i in range(1, 4):
        prev = sel[:i]
        best = max((j for j in range(N) if j not in prev), key=lambda j: torch.det(kernel[prev + [j]][:, prev + [j]]))
        assert sel[i] == best


def test_random_depends_on_sample_id():
    a = METHODS.build("random", budget=16)(make_ctx(16, sample_id="a"))
    b = METHODS.build("random", budget=16)(make_ctx(16, sample_id="b"))
    assert not torch.equal(a.indices, b.indices)


def test_cdpruner_embedding_space():
    ctx = make_ctx(16)
    clip = METHODS.build("cdpruner", budget=16, relevance_space="clip")(ctx).indices
    llm = METHODS.build("cdpruner", budget=16, relevance_space="llm")(ctx).indices
    auto = METHODS.build("cdpruner", budget=16)(ctx).indices
    assert torch.equal(auto, clip) and not torch.equal(clip, llm)


def test_visionzip_keeps_top_attention():
    ctx = make_ctx(64)
    out = METHODS.build("visionzip", budget=64)(ctx)
    dom = torch.sort(torch.topk(ctx[Feature.VISION_ATTN], 53).indices).values  # + [CLS] = 54 dominant
    assert torch.equal(out.indices[1:54], dom)
    assert out.tokens.shape[0] == 64
    # first token is the projected [CLS]
    assert torch.allclose(out.tokens[0], ctx[Feature.PROJECTOR](ctx[Feature.CLS_TOKEN][None])[0])


def test_cdpruner_negation_flips_relevance():
    ctx = make_ctx(16)
    a = METHODS.build("cdpruner", budget=16, relevance_space="clip", negate_relevance=True)(ctx).indices
    b = METHODS.build("cdpruner", budget=16, relevance_space="clip", negate_relevance=False)(ctx).indices
    assert not torch.equal(a, b)


def test_trim_keeps_lowest_clip_similarity():
    ctx = make_ctx(9)
    out = METHODS.build("trim", budget=9)(ctx)
    dots = ctx[Feature.CLIP_IMAGE_EMBEDS] @ ctx[Feature.CLIP_TEXT_EMBEDS]
    assert set(out.indices[:-1].tolist()) == set(torch.topk(-dots, 8).indices.tolist())

