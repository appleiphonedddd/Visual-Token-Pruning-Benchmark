# syntax=docker/dockerfile:1
# Visual-Token-Pruning-Benchmark
# The base variant is enough: the PyTorch cu128 wheel ships its own CUDA runtime,
# cuBLAS and cuDNN as pip packages, and nothing here needs nvcc (attention is SDPA).
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Python 3.11 (Ubuntu 24.04 ships 3.12)
RUN uv venv /opt/venv --python 3.11

# PyTorch first, in its own layer, so code changes don't trigger a reinstall
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace

# Project dependencies (torch is already satisfied and is not reinstalled)
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e ".[dev,quant]"

COPY . .

# Run as the host user (docker run --user "$(id -u):$(id -g)") so files written to
# mounted dirs (HF cache, outputs) are owned by you, not root.
# HF cache lives outside /root so any uid can use it; mount the host cache here.
ENV HF_HOME=/hf-cache \
    HOME=/tmp
RUN mkdir -m 1777 /hf-cache

CMD ["bash"]
