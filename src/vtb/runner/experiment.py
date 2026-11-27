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
from vtb.profiling.hardware import device_info
from vtb.runner.io import PredictionStore, write_json

_COUNT_KEYS = ("num_visual_tokens", "num_kept_tokens", "seq_len", "avg_layer_seq_len", "num_generated")
_TIMING_KEYS = ("vision_ms", "prune_ms", "prefill_ms", "decode_ms")
_MEMORY_KEYS = ("prefill_mem_mb", "kv_cache_mb")
# per-sample fields copied from Prediction into predictions.jsonl
_RECORD_KEYS = (*_COUNT_KEYS, *_TIMING_KEYS, "num_decode_steps", "peak_mem_mb", *_MEMORY_KEYS)


def _kwargs(cfg: dict, drop: tuple[str, ...] = ()) -> dict:
    return {k: v for k, v in cfg.items() if k not in drop}


def build_pruner(spec: RunSpec, seed: int):
    m = spec.method
    return METHODS.build(m["type"], budget=spec.budget, seed=seed, **(m.get("params") or {}))


def score_run(run_dir: Path, evaluator_name: str) -> dict[str, float]:
    records = PredictionStore(run_dir / "predictions.jsonl").records()
    return EVALUATORS.build(evaluator_name).score(records)


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    v = sorted(values)
    pos = (len(v) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def efficiency_summary(records: list[dict], model: BaseVLMAdapter | None) -> dict[str, float]:
    """Per-run efficiency block of metrics.json.

    Latencies are means (``avg_*``) plus median / p90 for prefill and time-to-first-token.
    Decode throughput is pooled (total decode steps / total decode time), so samples that
    stop after one token do not get the same weight as long answers. Records written
    before a field existed are left out of that field's statistics.
    """
    if not records:
        return {}
    out: dict[str, float] = {}
    for k in (*_COUNT_KEYS, *_TIMING_KEYS):
        # records written before avg_layer_seq_len existed fall back to the pre-LLM sequence length
        out[f"avg_{k}"] = statistics.fmean(r[k] if k in r else r.get("seq_len", 0) for r in records)

    ttft = [r["vision_ms"] + r["prune_ms"] + r["prefill_ms"] for r in records]
    out["avg_ttft_ms"] = statistics.fmean(ttft)
    for name, vals in (("prefill_ms", [r["prefill_ms"] for r in records]), ("ttft_ms", ttft)):
        out[f"p50_{name}"] = _percentile(vals, 50)
        out[f"p90_{name}"] = _percentile(vals, 90)

    dec = [r for r in records if r.get("num_decode_steps")]
    steps = sum(r["num_decode_steps"] for r in dec)
    dec_ms = sum(r["decode_ms"] for r in dec)
    if any("num_decode_steps" in r for r in records):
        out["avg_num_decode_steps"] = statistics.fmean(r.get("num_decode_steps", 0) for r in records)
    if steps and dec_ms > 0:
        out["decode_tokens_per_s"] = steps / dec_ms * 1e3
        out["avg_tpot_ms"] = dec_ms / steps  # time per output token

    peaks = [r["peak_mem_mb"] for r in records if r.get("peak_mem_mb")]
    if peaks:
        out["peak_memory_gb"] = max(peaks) / 1024
    for k in _MEMORY_KEYS:
        vals = [r[k] for r in records if k in r]
        if vals and any(vals):
            out[f"avg_{k}"] = statistics.fmean(vals)
            out[f"max_{k}"] = max(vals)
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

        # Warm up every run, not once per model: each method has its own cold-start cost
        # (cuBLAS/cuSOLVER handles for DPP, the in-LLM layer loop, new kernel shapes).
        if n_todo:
            first = next(iter(dataset))
            for _ in range(int(spec.generation.get("warmup", 1))):
                model.generate(first, dataset.format_prompt(first), pruner, max_new)  # not recorded

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
                **{k: getattr(pred, k) for k in _RECORD_KEYS},
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
            efficiency=efficiency_summary(records, model),
            hardware=device_info(model.device),
        )
        write_json(out_dir / "metrics.json", metrics)
        print(f"       {metrics['primary_metric']} = {metrics['score']:.2f}")
