from pathlib import Path

import pytest

from vtb.core.config import load_experiment
from vtb.core.registry import DATASETS, EVALUATORS, METHODS, MODELS, Registry
from vtb.evaluators.pope import parse_yes_no

ROOT = Path(__file__).resolve().parents[1]


def test_registries_autodiscover():
    assert {"vanilla", "random", "pooling", "divprune", "cdpruner", "dppruner"} <= set(METHODS.names())
    assert "qwen2_5_vl" in MODELS
    assert "pope" in DATASETS
    assert "pope" in EVALUATORS


def test_duplicate_registration_rejected():
    reg = Registry("thing", "vtb.methods")

    @reg.register("a")
    class A: ...

    with pytest.raises(KeyError):
        @reg.register("a")
        class B: ...


def test_unknown_name_lists_available():
    with pytest.raises(KeyError, match="Available"):
        METHODS.get("does-not-exist")


def test_experiment_expansion():
    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    names = [r.run_name for r in cfg.runs]
    assert names == ["vanilla_full", "random_64", "divprune_64", "cdpruner_64"]
    assert all(r.dataset["limit"] == 60 for r in cfg.runs)


def test_experiment_overrides():
    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml", ["dataset_overrides.limit=5"])
    assert cfg.runs[0].dataset["limit"] == 5


@pytest.mark.parametrize("text,expected", [
    ("Yes", "yes"), ("No", "no"), ("no.", "no"), ("Yes, there is a dog.", "yes"),
    ("There is not a dog in the image.", "no"), ("No, there isn't. Yes.", "no"),
])
def test_pope_parse(text, expected):
    assert parse_yes_no(text) == expected


def test_pope_metrics():
    recs = [
        {"output": "Yes", "answers": ["yes"], "meta": {"category": "random"}},
        {"output": "Yes", "answers": ["no"], "meta": {"category": "random"}},
        {"output": "No", "answers": ["no"], "meta": {"category": "random"}},
        {"output": "No", "answers": ["yes"], "meta": {"category": "random"}},
        {"output": "Yes", "answers": ["yes"], "meta": {"category": "adversarial"}},
        {"output": "No", "answers": ["no"], "meta": {"category": "adversarial"}},
    ]
    s = EVALUATORS.build("pope").score(recs)
    assert s["random/f1"] == pytest.approx(50.0)
    assert s["adversarial/f1"] == pytest.approx(100.0)
    assert s["f1"] == pytest.approx(75.0)
