# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Pre-decoder training image.
#
# Build:
#   docker build -t predecoder-train .
#   docker build -t predecoder-train --build-arg TORCH_CUDA=cu124 .   # different CUDA
#
# Run:
#   docker run --rm --gpus all \
#     -v $(pwd):/app:ro -v $HOME/predecoder_outputs:/data \
#     -e SHARED_OUTPUT_DIR=/data \
#     predecoder-train
#
# See TRAINING.md for the full environment variable reference.

ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.11
ARG TORCH_CUDA=cu121

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PREDECODER_PYTHON=/opt/venv/bin/python

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
        curl git coreutils build-essential cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY code/requirements_public_inference.txt /tmp/requirements_public_inference.txt
COPY code/requirements_public_train.txt /tmp/requirements_public_train.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        -r /tmp/requirements_public_train.txt \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
        --extra-index-url https://pypi.org/simple && \
    python -c "import torch; print('PyTorch', torch.__version__, '(CUDA build:', torch.version.cuda, ')')"

WORKDIR /app
CMD ["bash", "code/scripts/cluster_container_install_and_train.sh"]
