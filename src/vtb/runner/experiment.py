"""Experiment runner: models x datasets x methods x budgets.

Each model is loaded once and reused for all its runs. Every run writes

    <output_dir>/<experiment>/<model>/<dataset>/<method>_<budget>/
        predictions.jsonl   one record per sample (resumable)
        metrics.json        scores + efficiency + resolved config, or status=skipped
"""

from __future__ import annotations

import gc
import random
import statistics
import time
from pathlib import Path

import torch

from vtb.core.config import ExperimentConfig, RunSpec
from vtb.core.registry import DATASETS, EVALUATORS, METHODS, MODELS
from vtb.datasets.base import BaseDataset
from vtb.models.base import BaseVLMAdapter
from vtb.profiling.flops import prefill_flops
from vtb.runner.io import PredictionStore, write_json

_TIMING_KEYS = ("vision_ms", "prune_ms", "prefill_ms", "decode_ms")


def _kwargs(cfg: dict, drop: tuple[str, ...] = ()) -> dict:
    return {k: v for k, v in cfg.items() if k not in drop}


def build_pruner(spec: RunSpec, seed: int):
    m = spec.method
    return METHODS.build(m["type"], budget=spec.budget, seed=seed, **(m.get("params") or {}))


def score_run(run_dir: Path, evaluator_name: str) -> dict[str, float]:
    records = PredictionStore(run_dir / "predictions.jsonl").records()
    return EVALUATORS.build(evaluator_name).score(records)


def efficiency_summary(records: list[dict], model: BaseVLMAdapter | None) -> dict[str, float]:
    if not records:
        return {}
    out: dict[str, float] = {}
    for k in ("num_visual_tokens", "num_kept_tokens", "seq_len", "avg_layer_seq_len", "num_generated", *_TIMING_KEYS):
        # records written before a field existed fall back to the pre-LLM sequence length
        out[f"avg_{k}"] = statistics.fmean(r[k] if k in r else r.get("seq_len", 0) for r in records)
    dims = model.llm_dims() if model is not None else {}
    if dims:
        out["avg_prefill_tflops"] = statistics.fmean(
            prefill_flops(r.get("avg_layer_seq_len") or r["seq_len"], **dims) for r in records
        ) / 1e12
    return out


class Experiment:
    def __init__(self, cfg: ExperimentConfig, log_every: int = 50):
        self.cfg = cfg
        self.log_every = log_every
        self.root = cfg.output_dir / cfg.name

    def run_dir(self, spec: RunSpec) -> Path:
        return self.root / spec.model["name"] / spec.dataset["name"] / spec.run_name

    def run(self) -> None:
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        by_model: dict[str, list[RunSpec]] = {}
        for spec in self.cfg.runs:
            by_model.setdefault(spec.model["name"], []).append(spec)

        for model_name, specs in by_model.items():
            model: BaseVLMAdapter | None = None
            datasets: dict[str, BaseDataset] = {}
            for spec in specs:
                pruner = build_pruner(spec, self.cfg.seed)
                out_dir = self.run_dir(spec)
                missing = pruner.missing_features(set(MODELS.get(spec.model["type"]).features_for(spec.model)))
                if missing:
                    print(f"[skip] {model_name} / {spec.method['name']}: needs {sorted(f.value for f in missing)}")
                    write_json(out_dir / "metrics.json", self._meta(spec, status="skipped",
                                                                   reason=f"model lacks {sorted(f.value for f in missing)}"))
                    continue
                if model is None:
                    print(f"[load] {model_name} ({spec.model['pretrained']})")
                    model = MODELS.build(spec.model["type"], **_kwargs(spec.model, ("type",)))
                ds_name = spec.dataset["name"]
                if ds_name not in datasets:
                    datasets[ds_name] = DATASETS.build(spec.dataset["type"], **_kwargs(spec.dataset, ("type",)))
                self._run_one(spec, model, pruner, datasets[ds_name], out_dir)
            del model
            gc.collect()  # quantised models hold reference cycles; free them before loading the next one
            torch.cuda.empty_cache()

    def _meta(self, spec: RunSpec, **extra) -> dict:
        return {
            "experiment": self.cfg.name,
            "model": spec.model["name"],
            "dataset": spec.dataset["name"],
            "method": spec.method["name"],
            "budget": spec.budget,
            "config": {"model": spec.model, "dataset": spec.dataset, "method": spec.method, "generation": spec.generation},
            **extra,
        }

    def _run_one(self, spec: RunSpec, model: BaseVLMAdapter, pruner, dataset: BaseDataset, out_dir: Path) -> None:
        store = PredictionStore(out_dir / "predictions.jsonl")
        done = store.done_ids()
        max_new = int(spec.generation.get("max_new_tokens", 16))
        n_todo = len(dataset) - len(done)
        print(f"[run ] {spec.model['name']} | {dataset.name} | {spec.run_name}: {n_todo} todo, {len(done)} done")
        todo = (s for s in dataset if s.id not in done)  # lazy: don't hold every image in memory

        if n_todo and not getattr(model, "_warmed_up", False):
            first = next(iter(dataset))
            model.generate(first, dataset.format_prompt(first), pruner, max_new)  # CUDA warmup, not recorded
            model._warmed_up = True

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i, sample in enumerate(todo, 1):
            pred = model.generate(sample, dataset.format_prompt(sample), pruner, max_new)
            store.append({
                "id": sample.id,
                "question": sample.question,
                "output": pred.output,
                "answers": sample.answers,
                "choices": sample.choices,
                "meta": sample.meta,
                **{k: getattr(pred, k) for k in (
                    "num_visual_tokens", "num_kept_tokens", "seq_len", "avg_layer_seq_len", "num_generated", *_TIMING_KEYS)},
            })
            if i % self.log_every == 0 or i == n_todo:
                rate = i / (time.time() - t0)
                print(f"       {i}/{n_todo}  ({rate:.2f} it/s)")

        records = store.records()
        evaluator = EVALUATORS.build(dataset.evaluator)
        scores = evaluator.score(records)
        metrics = self._meta(
            spec,
            status="ok",
            **evaluator.headline(scores, dataset.primary_metric),
            scores=scores,
            efficiency={
                **efficiency_summary(records, model),
                **({"peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30} if torch.cuda.is_available() and n_todo else {}),
            },
        )
        write_json(out_dir / "metrics.json", metrics)
        print(f"       {metrics['primary_metric']} = {metrics['score']:.2f}")
