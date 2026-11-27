"""Efficiency summary and aggregate table on synthetic records (no GPU)."""

import json

import pytest
import torch

from vtb.profiling.hardware import MemoryProbe, device_info
from vtb.runner.aggregate import build_table
from vtb.runner.experiment import _percentile, efficiency_summary


def _rec(prefill, decode_ms, steps, peak=0.0, kv=0.0, **extra):
    return {"num_visual_tokens": 576, "num_kept_tokens": 64, "seq_len": 100, "avg_layer_seq_len": 100.0,
            "num_generated": steps + 1, "vision_ms": 10.0, "prune_ms": 2.0, "prefill_ms": prefill,
            "decode_ms": decode_ms, "num_decode_steps": steps, "peak_mem_mb": peak,
            "prefill_mem_mb": kv * 3, "kv_cache_mb": kv, **extra}


def test_percentile():
    assert _percentile([5.0], 90) == 5.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert _percentile(list(map(float, range(11))), 90) == pytest.approx(9.0)


def test_efficiency_summary_pools_decode_throughput():
    # 1 step in 10 ms and 9 steps in 90 ms -> 10 ms/token overall, not a mean of per-sample rates
    recs = [_rec(20.0, 10.0, 1, peak=6000, kv=20), _rec(40.0, 90.0, 9, peak=6100, kv=30), _rec(30.0, 0.5, 0)]
    out = efficiency_summary(recs, None)
    assert out["decode_tokens_per_s"] == pytest.approx(100.0)
    assert out["avg_tpot_ms"] == pytest.approx(10.0)
    assert out["avg_prefill_ms"] == pytest.approx(30.0)
    assert out["p50_prefill_ms"] == pytest.approx(30.0)
    assert out["avg_ttft_ms"] == pytest.approx(42.0)
    assert out["peak_memory_gb"] == pytest.approx(6100 / 1024)
    assert out["max_kv_cache_mb"] == 30
    assert out["avg_num_decode_steps"] == pytest.approx(10 / 3)


def test_efficiency_summary_old_records_have_no_hardware_fields():
    old = {k: v for k, v in _rec(20.0, 10.0, 1).items()
           if k not in ("num_decode_steps", "peak_mem_mb", "prefill_mem_mb", "kv_cache_mb")}
    out = efficiency_summary([old], None)
    assert out["avg_prefill_ms"] == 20.0
    for k in ("decode_tokens_per_s", "peak_memory_gb", "avg_kv_cache_mb", "avg_num_decode_steps"):
        assert k not in out


def test_aggregate_efficiency_columns(tmp_path):
    def write(method, budget, dataset, score, eff):
        d = tmp_path / "m" / dataset / f"{method}_{budget}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({
            "model": "m", "dataset": dataset, "method": method, "budget": budget, "status": "ok",
            "score": score, "efficiency": eff}))

    write("vanilla", None, "a", 80.0, {"avg_prefill_ms": 70.0, "decode_tokens_per_s": 40.0,
                                       "avg_kv_cache_mb": 30.0, "peak_memory_gb": 7.5})
    write("vanilla", None, "b", 60.0, {"avg_prefill_ms": 50.0, "decode_tokens_per_s": 50.0,
                                       "avg_kv_cache_mb": 20.0, "peak_memory_gb": 7.0})
    write("cdpruner", 64, "a", 78.0, {"avg_prefill_ms": 20.0})  # metrics from an older version
    header, table, _ = build_table(tmp_path)
    row = dict(zip(header, table[0]))
    assert row["method"] == "vanilla"
    assert (row["prefill_ms"], row["decode_tok/s"], row["kv_cache_mb"], row["peak_mem_gb"]) == \
        ("60.0", "45.0", "25.0", "7.50")
    row = dict(zip(header, table[1]))
    assert (row["prefill_ms"], row["decode_tok/s"], row["peak_mem_gb"]) == ("20.0", "-", "-")


def test_memory_probe_on_cpu_is_zero():
    probe = MemoryProbe(torch.device("cpu"))
    probe.reset_peak()
    assert probe.allocated() == 0.0 and probe.phase() == 0.0
    assert device_info(torch.device("cpu"))["device"] == "cpu"
