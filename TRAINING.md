# Cluster / Remote Training Guide

This document covers running pre-decoder training on remote GPU nodes
using Docker containers, with optional SLURM integration.
For local single-machine usage, see `README.md`.

## Prerequisites

- Docker with NVIDIA GPU support (`nvidia-docker` / `--gpus`)
- One or more NVIDIA GPUs (H100, A100, or similar)
- A persistent directory for checkpoints and logs

## Quick start (Docker — recommended)

### Option A: build the image once, reuse everywhere

```bash
# Build (once, from repo root)
docker build -t predecoder-train .

# Optionally, for a different CUDA version:
docker build -t predecoder-train --build-arg TORCH_CUDA=cu124 .

# Train
docker run --rm --gpus all \
  -v $(pwd):/app:ro \
  -v $HOME/predecoder_outputs:/data \
  -e SHARED_OUTPUT_DIR=/data \
  predecoder-train
```

The image includes Python 3.11, PyTorch with CUDA, and all training dependencies.
Dependencies are baked in, so startup is fast and no internet access is needed at
runtime.

### Option B: install deps at runtime from a CUDA base image

If you cannot pre-build the image (e.g. in a locked-down environment):

```bash
docker run --rm --gpus all \
  -v $(pwd):/app:ro \
  -v $HOME/predecoder_outputs:/data \
  -e SHARED_OUTPUT_DIR=/data \
  -e INSTALL_DIR=/opt/predecoder_env \
  nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 \
  bash -c 'apt-get update -qq && apt-get install -y -qq python3.11 python3.11-venv python3.11-dev curl git build-essential cmake >/dev/null 2>&1; bash /app/code/scripts/cluster_container_install_and_train.sh'
```

This installs dependencies on every run, so it is slower. Use Option A when possible.

## Quick start (SLURM + Docker)

1. Build the image on a machine with Docker access:
   ```bash
   docker build -t predecoder-train .
   ```

2. Edit the `#SBATCH` directives in `code/scripts/sbatch_train.sh` for your cluster
   (partition name, GPU count, memory, time limit).

3. Submit:
   ```bash
   export SHARED_OUTPUT_DIR=$HOME/predecoder_outputs
   sbatch code/scripts/sbatch_train.sh
   ```

4. Monitor:
   ```bash
   tail -f predecoder_train_<jobid>.out
   ```

The sbatch script auto-detects: pre-built image > base CUDA image > bare-metal fallback.

## Script overview

| Script | Purpose |
|--------|---------|
| `Dockerfile` | Builds `predecoder-train` image with all dependencies. |
| `code/scripts/local_run.sh` | Core runner. Handles Hydra config, GPU detection, logging, checkpoints. Works everywhere. |
| `code/scripts/cluster_install_deps.sh` | Installs Python 3.11+ and training dependencies into an isolated environment. |
| `code/scripts/cluster_train.sh` | Sets up output dirs, exports env, then calls `local_run.sh`. Expects `SHARED_OUTPUT_DIR`. |
| `code/scripts/cluster_container_install_and_train.sh` | Runs inside a Docker container: install deps (if needed), then train. |
| `code/scripts/sbatch_train.sh` | SLURM submission script (template). Edit `#SBATCH` directives for your cluster. |

### Call chain

```
sbatch_train.sh  (or: docker run ... predecoder-train)
  ├─ (pre-built image)   → cluster_container_install_and_train.sh
  │                           └─ cluster_train.sh → local_run.sh
  ├─ (base CUDA image)   → cluster_container_install_and_train.sh
  │                           ├─ cluster_install_deps.sh
  │                           └─ cluster_train.sh → local_run.sh
  └─ (no Docker)         → cluster_install_deps.sh
                            → cluster_train.sh → local_run.sh
```

## Quick start (bare-metal node, no Docker)

If Docker is unavailable, you can install directly on the node:

```bash
# Install deps once
export INSTALL_DIR=$HOME/predecoder_env
bash code/scripts/cluster_install_deps.sh

# Train
export SHARED_OUTPUT_DIR=$HOME/predecoder_outputs
export PREDECODER_PYTHON=$INSTALL_DIR/venv/bin/python
bash code/scripts/cluster_train.sh
```

## Available training configs

| Config file | Model | R | Noise |
|-------------|-------|---|-------|
| `conf/config_qec_decoder_r9_fp8.yaml` | Model 1 | 9 | Depolarizing p=0.006 |
| `conf/config_qec_decoder_r13_fp8.yaml` | Model 4 | 13 | Depolarizing p=0.006 |
| `conf/config_public.yaml` | Any | Varies | User-defined |

Select a config by setting `CONFIG_NAME` (without the `.yaml` extension):
```bash
export CONFIG_NAME=config_qec_decoder_r13_fp8
```

## Environment variable reference

### Core variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHARED_OUTPUT_DIR` | *(required for cluster)* | Persistent directory for outputs, logs, checkpoints. |
| `EXPERIMENT_NAME` | `qec-decoder-depolarizing-r9-fp8` | Subdirectory under `outputs/` for this run. Change this when changing configs. |
| `CONFIG_NAME` | `config_qec_decoder_r9_fp8` | Hydra config name (file in `conf/` without `.yaml`). |
| `WORKFLOW` | `train` | `train` or `inference`. |
| `GPUS` | auto-detect | Number of GPUs. Must match SLURM `--gres=gpu:N`. |
| `FRESH_START` | `0` | Set `1` to ignore existing checkpoints and start from scratch. |

### Training overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDECODER_TRAIN_EPOCHS` | `100` | Total number of training epochs. |
| `PREDECODER_TRAIN_SAMPLES` | config-defined | Samples per epoch. Bypasses auto-scaling when set explicitly. |
| `PREDECODER_LR_MILESTONES` | config-defined | Comma-separated LR schedule milestone fractions (e.g. `0.25,0.5,1.0`). |
| `PREDECODER_TIMING_RUN` | unset | Set `1` for timing/benchmarking mode (disables some overhead). |
| `PREDECODER_TORCH_COMPILE` | `0` when run via `sbatch_train.sh`, otherwise unset | `0` to disable `torch.compile`, `1` to enable. |
| `PREDECODER_DISABLE_SDR` | `1` when run via `sbatch_train.sh`, otherwise unset | `1` to skip Syndrome Density Reduction computation (saves time on cluster). |
| `TORCH_COMPILE` | unset | Alternative way to control `torch.compile` (`0`/`1`). |
| `TORCH_COMPILE_MODE` | unset | `default`, `reduce-overhead`, or `max-autotune`. |

### Infrastructure variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INSTALL_DIR` | `$HOME/predecoder_env` | Where `cluster_install_deps.sh` creates the Python environment. |
| `PREDECODER_PYTHON` | auto-detect | Explicit path to the Python binary. |
| `TORCH_CUDA` | `cu121` | PyTorch CUDA wheel tag (e.g. `cu121`, `cu124`, `cu130`). |
| `DOCKER_IMAGE` | `predecoder-train` | Pre-built Docker image name. |
| `DOCKER_BASE_IMAGE` | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` | Fallback CUDA base image. |
| `SHARED_LOG_DIR` | `$SHARED_OUTPUT_DIR/logs` | Override the logs root directory (advanced). |
| `PREDECODER_BASE_OUTPUT_DIR` | `$SHARED_OUTPUT_DIR/outputs` | Override the outputs root (advanced). |
| `PREDECODER_LOG_BASE_DIR` | `$SHARED_OUTPUT_DIR/logs` | Override the logs root (advanced, set by `cluster_train.sh` from `SHARED_LOG_DIR`). |

## Example SLURM configurations

### R=9, 1 GPU (Model 1)

```bash
export SHARED_OUTPUT_DIR=$HOME/predecoder_outputs
sbatch code/scripts/sbatch_train.sh
```

### R=13, 1 GPU (Model 4)

```bash
export SHARED_OUTPUT_DIR=$HOME/predecoder_outputs
EXPERIMENT_NAME=qec-decoder-depolarizing-r13-fp8 \
CONFIG_NAME=config_qec_decoder_r13_fp8 \
  sbatch code/scripts/sbatch_train.sh
```

### R=13, 4 GPUs (Model 4)

Override SLURM resources on the command line:

```bash
export SHARED_OUTPUT_DIR=$HOME/predecoder_outputs
EXPERIMENT_NAME=qec-decoder-depolarizing-r13-fp8-4gpu \
CONFIG_NAME=config_qec_decoder_r13_fp8 \
GPUS=4 FRESH_START=1 \
  sbatch --partition=<your-4gpu-partition> \
         --nodes=1 --gres=gpu:4 --cpus-per-task=80 --mem=240G \
         code/scripts/sbatch_train.sh
```

### Resume a 1-GPU checkpoint on 4 GPUs

When moving from 1 to N GPUs mid-training, fix the sample count and LR milestones
so the schedule matches the original trajectory:

```bash
export SHARED_OUTPUT_DIR=$HOME/predecoder_outputs
EXPERIMENT_NAME=qec-decoder-depolarizing-r13-fp8 \
CONFIG_NAME=config_qec_decoder_r13_fp8 \
GPUS=4 \
PREDECODER_TRAIN_SAMPLES=8388608 \
PREDECODER_LR_MILESTONES="1.0,2.0,4.0" \
  sbatch --partition=<your-4gpu-partition> \
         --nodes=1 --gres=gpu:4 --cpus-per-task=80 --mem=240G \
         code/scripts/sbatch_train.sh
```

The milestone rescaling formula: if original milestones are `[m1, m2, m3]` and you
increase GPU count by factor `k`, new milestones are `[m1*k, m2*k, m3*k]`.

## Multi-GPU training

- Training uses PyTorch DDP (`torch.distributed.run`). Set `GPUS=N` and ensure N GPUs are visible.
- Auto-scaling: with N GPUs, each GPU processes `num_samples / N` samples per epoch.
  To keep the *total* samples identical to a 1-GPU run, set `PREDECODER_TRAIN_SAMPLES` explicitly.
- LR milestones are expressed as fractions of total steps. Changing GPU count changes total steps,
  so milestones may need rescaling (see the resume example above).
- The `MASTER_PORT` is auto-selected if not set. Override it to avoid port conflicts
  when running multiple jobs on the same node.

## Resuming training

Training auto-resumes from the latest checkpoint found in
`$SHARED_OUTPUT_DIR/outputs/$EXPERIMENT_NAME/models/`.

- Same experiment name = resume. Different experiment name = fresh run.
- To force a clean restart on the same experiment: `export FRESH_START=1`.
- A lock file prevents two SLURM jobs from writing to the same experiment directory concurrently.

## Output structure

```
$SHARED_OUTPUT_DIR/
├── outputs/
│   └── <experiment_name>/
│       ├── models/          # Checkpoints and final model
│       ├── tensorboard/     # TensorBoard logs
│       ├── config/          # Config snapshots per run
│       └── run.log          # Latest run log
└── logs/
    └── <experiment_name>_<timestamp>/
        └── train.log        # Full stdout/stderr
```

## Adapting to your cluster

1. **Edit `#SBATCH` directives** in `sbatch_train.sh`:
   - `--partition=` your cluster's GPU partition
   - `--gres=gpu:N` matching your GPU count
   - `--cpus-per-task=`, `--mem=`, `--time=` as appropriate

2. **CUDA version**: set `TORCH_CUDA=cuXXX` to match your driver
   (e.g. `cu121` for CUDA 12.1, `cu124` for CUDA 12.4).

3. **Docker base image**: set `DOCKER_BASE_IMAGE` if your cluster uses a different CUDA runtime.

4. **File systems**: `SHARED_OUTPUT_DIR` should be on a shared/persistent filesystem
   visible from all nodes (NFS, Lustre, etc.). The sbatch script sets `chmod 1777` for
   NFS compatibility when using Docker.

5. **No Docker?** The scripts fall back to bare-metal install automatically.
   Ensure the node has internet access (for pip) or pre-install deps via `cluster_install_deps.sh`.

## Troubleshooting

- **`SHARED_OUTPUT_DIR is not set`**: export it before running cluster scripts.
- **Lock file conflict**: if a previous job crashed, remove `$SHARED_OUTPUT_DIR/.lock_<experiment>`.
- **`steps_per_epoch=0`**: samples too low for the batch size. Increase `PREDECODER_TRAIN_SAMPLES`.
- **torch.compile segfaults**: set `PREDECODER_TORCH_COMPILE=0`.
- **pip install fails in container**: ensure the base image has `python3.11-dev` and `build-essential`.
