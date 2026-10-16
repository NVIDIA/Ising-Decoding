# Cluster / Remote Training Guide

This document covers running pre-decoder training on remote GPU nodes
using Docker containers, with optional SLURM integration.
For local single-machine usage, see `README.md`.

## Prerequisites

- Docker with NVIDIA GPU support (`nvidia-docker` / `--gpus`)
- One or more NVIDIA GPUs (B200, H200 or similar)
- A persistent directory for checkpoints and logs

## Quick start (Docker — recommended)

### Option A: build the image once, reuse everywhere

```bash
# Build (once, from repo root)
docker build -t predecoder-train .

# Optionally, for a different CUDA version:
docker build -t predecoder-train --build-arg TORCH_CUDA=cu128 .

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

| Config file | Code | Model | R | Noise |
|-------------|------|-------|---|-------|
| `conf/config_qec_decoder_r9_fp8.yaml` | Surface | Model 1 | 9 | Depolarizing p=0.006 |
| `conf/config_qec_decoder_r13_fp8.yaml` | Surface | Model 4 | 13 | Depolarizing p=0.006 |
| `conf/config_public.yaml` | Surface | Any | Varies | User-defined |
| `conf/config_color_model_1_s_LR3e-4.yaml` | Color | Model 1-shaped | 9 | Superdense color-code circuit (`p_min/p_max ≈ 0.003`) |
| `conf/config_color_threshold_model_1_d13.yaml` | Color | n/a (eval) | 13 | Threshold sweep against a trained color checkpoint |
| `conf/config_inference_color_model_5.yaml` | Color | n/a (inference) | 9 | Inference with a trained Model-5-shaped color checkpoint via `workflow.task=inference` (set `model_checkpoint_file`) |
| `conf/config_color_accurate_d17.yaml` | Color | Cascade, 25.4M parameters | 17 | Superdense color-code circuit (`p ≈ 0.004`). Reference recipe for 8 nodes × 4 GPUs with superbatching, SyncBatchNorm and bf16 autocast, see [Large training runs](#large-training-runs-superbatching-syncbatchnorm-multi-node) |

Select a config by setting `CONFIG_NAME` (without the `.yaml` extension):
```bash
export CONFIG_NAME=config_qec_decoder_r13_fp8
```

### Color-code training

Color-code training runs through the same `local_run.sh` launcher as
surface; the launcher bypasses the public-config validator for
`code: color` configs and `run_color` dispatches to the training entry
point. Color-code **inference** (on a trained color-code checkpoint) is
documented in **Color code support** in `README.md`. Before launching training,
generate the augmented DEM bundle once per
`(distance, n_rounds, basis)` combination:

```bash
PYTHONPATH=code python code/qec/precompute_dem.py \
    --code color --distance 9 --n_rounds 9 --basis X \
    --dem_output_dir frames_data
PYTHONPATH=code python code/qec/precompute_dem.py \
    --code color --distance 9 --n_rounds 9 --basis Z \
    --dem_output_dir frames_data
```

Then launch training:

```bash
CONFIG_NAME=config_color_model_1_s_LR3e-4 \
    WORKFLOW=train \
    EXTRA_PARAMS="data.precomputed_frames_dir=$(pwd)/frames_data" \
    bash code/scripts/local_run.sh
```

See **Color code support** in `README.md` for current limitations
(fixed-p augmented DEM, no Y decomposition).

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
| `PREDECODER_TRAIN_EPOCHS` | `100` | Total number of training epochs. For models 1, 4, 5 (uncorrelated matching), use at least **100** epochs; fewer epochs will yield under-trained models. |
| `PREDECODER_TRAIN_SAMPLES` | config-defined | Samples per epoch. Bypasses auto-scaling when set explicitly. For best results with 8 GPUs, use **67 million** shots per epoch (`67108864`); fewer shots per epoch will produce worse results. |
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
| `TORCH_CUDA` | `cu121` | PyTorch CUDA wheel tag (e.g. `cu121`, `cu128`, `cu130`). |
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

## Large training runs (superbatching, SyncBatchNorm, multi-node)

These options are off by default. Existing configs behave as before unless you
enable them. `conf/config_color_accurate_d17.yaml` uses all of them.

### Superbatching (color generator)

Direct generation produces one minibatch per step. Superbatching generates a
large batch per basis on the GPU and serves minibatches from it. The next
superbatch is built in a background thread while the current one is consumed.

| Config key | Default | Description |
|------------|---------|-------------|
| `data.superbatch_size` | `0` | Samples per superbatch. `0` keeps direct generation. Must be a multiple of the per-device batch size. |
| `data.superbatch_prefetch` | `false` | Build the next superbatch in the background. |
| `data.superbatch_interleave_bases` | `false` | Serve half-X/half-Z minibatches. Keep `false` to reproduce the basis schedule of direct generation. |

Buffers are per basis, so every minibatch stays single-basis as with direct
generation. With `superbatch_size` equal to the batch size the samples are
identical to direct generation. Speed-up per microbatch on one node with 4 GPUs:
3.0× for the Model-1-shaped color config and 1.26× for the 25.4M-parameter
cascade, where the forward and backward passes take most of the step. The
surface-code generator does not have this path.

### SyncBatchNorm

Under DDP each rank normalizes with its own BatchNorm statistics, and the
checkpoint carries the running statistics of rank 0 only. `sync_batchnorm: true`
synchronizes them across all ranks. The conversion happens before the DDP wrap
and does not change state-dict keys, so checkpoints stay interchangeable with
single-GPU runs.

### Mixed precision

`enable_bf16: true` or `enable_fp16: true` applies autocast to the forward pass
only. Parameters, BatchNorm statistics and optimizer state stay in fp32.

### Multi-node training

`local_run.sh` starts one `torch.distributed.run` per node. Launch one task per
node. Every task needs the node count, its own node rank and a shared master
address. Inside a SLURM allocation these default to `SLURM_NNODES`,
`SLURM_NODEID` and the first host of `SLURM_JOB_NODELIST`:

```bash
#SBATCH --nodes=8 --ntasks-per-node=1 --gres=gpu:4
srun --export=ALL bash -c '
  GPUS=4 CONFIG_NAME=config_color_accurate_d17 WORKFLOW=train \
  EXTRA_PARAMS="data.precomputed_frames_dir=$FRAMES output=$OUT resume_dir=$OUT/models" \
  bash code/scripts/local_run.sh'
```

| Variable | Default | Description |
|----------|---------|-------------|
| `NNODES` | `SLURM_NNODES`, else `1` | Number of nodes. |
| `NODE_RANK` | `SLURM_NODEID`, else `0` | Rank of this node. Always `0` when `NNODES=1`. |
| `MASTER_ADDR` | first host of `SLURM_JOB_NODELIST` | Rendezvous host. Required outside SLURM when `NNODES > 1`. |
| `MASTER_PORT` | `29500` for multi-node | Rendezvous port. |

`generate_stim_data` always runs as one serial process. Keep `output` on shared
storage, or set `PREDECODER_BASE_OUTPUT_DIR`, so every node resumes from the
same checkpoints.

Per-rank data streams are keyed on the global rank. Each rank owns a fixed range
of seeds, `data.rank_seed_stride` generator steps long, default 1,000,003.
Training fails at start-up in three cases: the run would leave its range, the
training ranges would reach the seeds of the validation generator, or the
ranges would wrap at 2**31. The error message names the limit. Few ranks take
more steps each, so a long run on few ranks may need a larger
`data.rank_seed_stride`. Changing it changes the data stream of every rank.
Reducing epochs or samples per epoch is the alternative. On resume only the
remaining epochs are budgeted.

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
   (e.g. `cu121` for CUDA 12.1, `cu128` for CUDA 12.8).

3. **Docker base image**: set `DOCKER_BASE_IMAGE` if your cluster uses a different CUDA runtime.

4. **File systems**: `SHARED_OUTPUT_DIR` should be on a shared/persistent filesystem
   visible from all nodes (NFS, Lustre, etc.). When using Docker, the training
   process runs as the submitting user's UID/GID (the base-image path installs
   dependencies as root, then drops privileges), so no special directory
   permissions are needed.

5. **No Docker?** The scripts fall back to bare-metal install automatically.
   Ensure the node has internet access (for pip) or pre-install deps via `cluster_install_deps.sh`.

## Troubleshooting

- **`SHARED_OUTPUT_DIR is not set`**: export it before running cluster scripts.
- **Lock file conflict**: if a previous job crashed, remove `$SHARED_OUTPUT_DIR/.lock_<experiment>`.
- **`steps_per_epoch=0`**: samples too low for the batch size. Increase `PREDECODER_TRAIN_SAMPLES`.
- **torch.compile segfaults**: set `PREDECODER_TORCH_COMPILE=0`.
- **pip install fails in container**: ensure the base image has `python3.11-dev` and `build-essential`.
