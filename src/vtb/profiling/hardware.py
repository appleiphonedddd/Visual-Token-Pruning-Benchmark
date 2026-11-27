"""Measured hardware metrics: CUDA allocator probes and device info.

``flops.py`` is an analytic estimate; this module records what the GPU actually
did. ``MemoryProbe`` reads the PyTorch caching allocator (``memory_allocated`` /
``max_memory_allocated``), i.e. bytes held by live tensors, not the larger
``nvidia-smi`` figure that also counts cached blocks and the CUDA context. On a
CPU device every reading is 0.
"""

from __future__ import annotations

import torch

_MiB = 2**20


class MemoryProbe:
    """Allocator readings for one ``generate`` call, in MiB.

    ``phase()`` closes the current phase: it returns the phase's peak allocation and
    starts a new peak window, so each phase's peak is measured on its own.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == "cuda" and torch.cuda.is_available()

    def allocated(self) -> float:
        return torch.cuda.memory_allocated(self.device) / _MiB if self.enabled else 0.0

    def reset_peak(self) -> None:
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def phase(self) -> float:
        """Peak allocation since the last ``reset_peak``/``phase`` call; then reset."""
        if not self.enabled:
            return 0.0
        peak = torch.cuda.max_memory_allocated(self.device) / _MiB
        torch.cuda.reset_peak_memory_stats(self.device)
        return peak


def device_info(device: torch.device) -> dict:
    """What the timings were measured on; written to metrics.json next to the efficiency block."""
    info: dict = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(device)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2**30, 2),
                    compute_capability=f"{p.major}.{p.minor}")
    return info
