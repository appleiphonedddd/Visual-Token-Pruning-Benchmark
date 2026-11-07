"""Prefill FLOPs estimate using the formula from VisionZip (Sec. 2.1):

    FLOPs = T * (4 n d^2 + 2 n^2 d + 2 n d m)

T layers, n sequence length, d hidden size, m FFN intermediate size. It is an
approximation (ignores GQA, SwiGLU's third matrix, the LM head) but it is what
the pruning papers use, so numbers stay comparable across methods.
"""

from __future__ import annotations


def prefill_flops(seq_len: int, num_layers: int, hidden: int, intermediate: int) -> float:
    n, d, m = seq_len, hidden, intermediate
    return float(num_layers * (4 * n * d * d + 2 * n * n * d + 2 * n * d * m))
