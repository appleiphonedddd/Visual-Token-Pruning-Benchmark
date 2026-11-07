"""YAML config loading and sweep expansion.

Layout::

    configs/models/<name>.yaml    -> {type: <registered adapter>, ...adapter kwargs}
    configs/methods/<name>.yaml   -> {type: <registered method>, budgets: [...], params: {...}}
    configs/datasets/<name>.yaml  -> {type: <registered dataset>, ...dataset kwargs}
    configs/experiments/<x>.yaml  -> lists of the above names (or inline dicts) + overrides

An experiment expands into the cartesian product models x datasets x methods x budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def _to_dict(cfg: Any) -> dict:
    return OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg


def load_component(kind: str, ref: str | dict, root: Path = CONFIG_ROOT) -> dict:
    """Resolve ``ref`` (a config name or an inline dict with optional ``config:`` base)."""
    if isinstance(ref, str):
        ref = {"config": ref}
    ref = dict(ref)
    base_name = ref.pop("config", None)
    base: dict = {}
    if base_name is not None:
        path = root / kind / f"{base_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No {kind} config '{base_name}' at {path}")
        base = _to_dict(OmegaConf.load(path))
        base.setdefault("name", base_name)
    merged = _to_dict(OmegaConf.merge(base, ref))
    if "type" not in merged:
        raise ValueError(f"{kind} config {merged} has no 'type'")
    merged.setdefault("name", merged["type"])
    return merged


@dataclass
class RunSpec:
    model: dict
    dataset: dict
    method: dict
    budget: int | None
    generation: dict = field(default_factory=dict)

    @property
    def run_name(self) -> str:
        b = "full" if self.budget is None else str(self.budget)
        return f"{self.method['name']}_{b}"


@dataclass
class ExperimentConfig:
    name: str
    output_dir: Path
    seed: int
    runs: list[RunSpec]


def load_experiment(path: str | Path, overrides: list[str] | None = None, root: Path = CONFIG_ROOT) -> ExperimentConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg = _to_dict(cfg)

    generation = cfg.get("generation", {})
    dataset_overrides = cfg.get("dataset_overrides", {})
    model_overrides = cfg.get("model_overrides", {})

    runs: list[RunSpec] = []
    for m_ref in cfg["models"]:
        model = load_component("models", m_ref, root)
        model.update(model_overrides)
        for d_ref in cfg["datasets"]:
            dataset = load_component("datasets", d_ref, root)
            dataset.update(dataset_overrides)
            for meth_ref in cfg["methods"]:
                method = load_component("methods", meth_ref, root)
                budgets = method.get("budgets") or [None]
                for b in budgets:
                    runs.append(RunSpec(model=model, dataset=dataset, method=method, budget=b, generation=generation))

    return ExperimentConfig(
        name=cfg.get("name", Path(path).stem),
        output_dir=Path(cfg.get("output_dir", "outputs")),
        seed=int(cfg.get("seed", 0)),
        runs=runs,
    )
