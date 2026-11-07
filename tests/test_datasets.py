"""Load every dataset config and check samples + prompts (run with ``pytest -m data``).

Needs the datasets in the Hugging Face cache (or network access).
"""

from pathlib import Path

import pytest

from vtb.core.config import load_component
from vtb.core.registry import DATASETS, EVALUATORS

pytestmark = pytest.mark.data

CONFIGS = sorted(p.stem for p in (Path(__file__).resolve().parents[1] / "configs/datasets").glob("*.yaml"))
CONFIGS = [c for c in CONFIGS if c != "vqav2"]  # ~7 GB; covered by the textvqa code path


@pytest.mark.parametrize("name", CONFIGS)
def test_dataset_loads(name):
    cfg = load_component("datasets", name)
    cfg["limit"] = 4
    ds = DATASETS.build(cfg.pop("type"), **cfg)
    samples = list(ds)
    assert 1 <= len(samples) and len({s.id for s in samples}) == len(samples)
    for s in samples:
        assert s.images and s.question and s.answers
        prompt = ds.format_prompt(s)
        assert s.question in prompt
        if s.choices:
            assert "A. " in prompt and s.answers[0] in "ABCDE"[: len(s.choices)]
    # evaluator accepts the records it will see
    recs = [{"output": s.answers[0], "answers": s.answers, "choices": s.choices, "meta": s.meta} for s in samples]
    ev = EVALUATORS.build(ds.evaluator)
    ev.headline(ev.score(recs), ds.primary_metric)


def test_subset_matches_hf_shuffle():
    cfg = load_component("datasets", "pope")
    cfg.update(limit=50)
    ds = DATASETS.build(cfg.pop("type"), **cfg)
    from datasets import load_dataset

    ref = load_dataset("lmms-lab/POPE", split="test").shuffle(seed=0).select(range(50))
    assert {s.id for s in ds} == {str(q) for q in ref["question_id"]}


def test_mmbench_limit_keeps_rotation_groups():
    cfg = load_component("datasets", "mmbench_en")
    cfg.update(limit=10)
    ds = DATASETS.build(cfg.pop("type"), **cfg)
    groups = {s.meta["circular_group"] for s in ds}
    assert len(groups) == 10
    assert len(ds) > 10  # rotations included
