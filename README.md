# Ising Decoding

[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](./LICENSE)
[![Release](https://img.shields.io/badge/Release-v0.1.0-brightgreen)](https://github.com/NVIDIA/Ising-Decoding/tree/releases/v0.1.0)
[![Paper](https://img.shields.io/badge/Paper-NVIDIA%20Research-76b900)](https://research.nvidia.com/publication/2026-04_fast-ai-based-pre-decoders-surface-codes)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Model: Fast](https://img.shields.io/badge/🤗%20HuggingFace-Fast%20Model-ffd21e)](https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast)
[![Model: Accurate](https://img.shields.io/badge/🤗%20HuggingFace-Accurate%20Model-ffd21e)](https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Accurate)

This repo offers AI training frameworks and recipes to build, customize and deploy scalable quantum error correction **decoders**:

- A neural network consumes detector syndromes across space **and** time
- It predicts corrections that reduce syndrome density / improve decoding
- A standard decoder (PyMatching) produces the final logical decision

The public release exposes a **single user-facing config** and a **single runner script**.

![Pre-decoder pipeline](images/predecoder_pipeline.png)

## Table of Contents

- [High-level workflow](#high-level-workflow)
- [Quick start (train + inference)](#quick-start-train--inference)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Inference (pre-trained models)](#inference-pre-trained-models)
- [Model export and downstream tools](#model-export-and-downstream-tools)
  - [Converting .pt checkpoints to SafeTensors](#converting-pt-checkpoints-to-safetensors-optional-post-training)
  - [ONNX export and quantization](#onnx-export-and-quantization-optional-post-training)
  - [Generating data for CUDA-Q QEC](#generating-data-for-cuda-q-qec-realtime-predecoder-test-application)
  - [Decoder ablation study with cudaq-qec](#decoder-ablation-study-with-cudaq-qec-optional)
- [Configuration and advanced usage](#configuration-and-advanced-usage)
  - [GPU selection](#gpu-selection)
  - [Public configuration](#public-configuration-confconfig_publicyaml)
  - [Precomputed frames](#precomputed-frames-recommended)
  - [Resuming training and running inference](#resuming-training-and-running-inference-on-a-trained-model)
- [Logging and outputs](#logging-and-outputs)
  - [What gets written where](#what-gets-written-where)
  - [Evaluation defaults](#evaluation-defaults-public-release)
- [Testing and CI](#testing-and-ci)
  - [Testing (CPU + GPU)](#testing-cpu--gpu)
  - [CI (GitHub Actions)](#ci-github-actions)
- [Results](#results)
- [License](#license)

## High-level workflow

```text
 ┌────────────────────────────────────────┐  Uses:
 │ 1. Train or Download Model             │  - Ising-Decoding repo (train)
 │                                        │  - Hugging Face (download)
 └──────────────────┬─────────────────────┘
                    │
                    ▼
 ┌────────────────────────────────────────┐  Uses:
 │ 2. Assess Performance                  │  - Ising-Decoding repo
 │    (Run inference tests)               │
 └──────────────────┬─────────────────────┘
                    │
 ┌──────────────────▼─────────────────────┐  Uses:
 │ 3. Investigate Realtime Performance    │  - Ising-Decoding repo (3a, 3b)
 │                                        │  - CUDA-Q QEC (3c)
 │   ┌────────────────────────────────┐   │
 │   │ 3a. Enable ONNX_WORKFLOW &     │   │
 │   │     choose quantization format │   │
 │   └──────────────┬─────────────────┘   │
 │                  │                     │
 │   ┌──────────────▼─────────────────┐   │
 │   │ 3b. Run generate_test_data.py  │   │
 │   └──────────────┬─────────────────┘   │
 │                  │                     │
 │   ┌──────────────▼─────────────────┐   │
 │   │ 3c. Take .onnx and .bin files  │   │
 │   │     into CUDA-Q QEC            │   │
 │   └────────────────────────────────┘   │
 └────────────────────────────────────────┘
```

## Quick start (train + inference)

From the repo root:

- `code/scripts/local_run.sh`

This script runs the Hydra workflow locally (no SLURM required) and reads **one** user-facing config file:

- `conf/config_public.yaml`

## Dependencies

Target Python versions: **3.11, 3.12, 3.13**.

Two minimal requirements files are provided:

- `code/requirements_public_inference.txt` (Stim + PyTorch path)
- `code/requirements_public_train-cuXY.txt` (training path, where XY = 12 or 13)

Install examples (virtual environment is optional but recommended):

```bash
# Optional: create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Optional: install CUDA-enabled PyTorch (example: pick any available cuXXX)
# Pick one that matches your CUDA runtime; cu130 is known to work.
export TORCH_CUDA=cu130

# Inference-only (training install is a superset)
pip install -r code/requirements_public_inference.txt

# Training (includes inference deps, adjust to cu13 as appropriate)
pip install -r code/requirements_public_train-cu12.txt

bash code/scripts/check_python_compat.sh
```

Tip: To force CUDA-enabled PyTorch, set `TORCH_CUDA=cuXXX` (recommended `cu13x`) or
`TORCH_WHL_INDEX=https://download.pytorch.org/whl/cuXXX` before running installs.

Quick start:

```bash
# Train (reads conf/config_public.yaml)
bash code/scripts/local_run.sh

# Inference (loads a saved model from outputs/<exp>/models/*)
WORKFLOW=inference bash code/scripts/local_run.sh
```

Inference note:

- On bare metal, keep the default DataLoader workers.
- In containers, set a larger shared-memory size (e.g., `docker run --shm-size=1g ...`).
- If you cannot change `--shm-size`, set `PREDECODER_INFERENCE_NUM_WORKERS=0` to avoid shared-memory worker crashes.
- Default evaluation is heavy (`cfg.test.num_samples=262144` shots per basis); expect inference to take time.

## Troubleshooting

- **Avoid `steps_per_epoch=0` on short runs**:
  - Keep `PREDECODER_TRAIN_SAMPLES >= per_device_batch_size * accumulate_steps * world_size`.
  - Note: the batch schedule jumps to 2048 after epoch 0, so epoch 1 uses
    `2048 * 2 * world_size` effective batch size.
  - For quick short runs, use `GPUS=1` and `PREDECODER_TRAIN_SAMPLES >= 4096`.
- **Segfaults during training startup (torch.compile)**:
  - Some environments crash during `torch.compile`.
  - Disable compile: `TORCH_COMPILE=0 bash code/scripts/local_run.sh`.
  - Or try a safer mode: `TORCH_COMPILE=1 TORCH_COMPILE_MODE=reduce-overhead bash code/scripts/local_run.sh`.
- **Blackwell GPUs (RTX 5080/5090, GB200/GB300)**:
  - Stable PyTorch wheels (`cu124`) do not ship SM 12.0 kernels.
    Install the nightly build with the `cu128` index:
    ```bash
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
    ```
- **Windows (Git Bash / WSL)**:
  - Triton is not supported on native Windows, which causes `torch.compile` to
    fail. Disable it before running:
    ```bash
    export TORCH_COMPILE_DISABLE=1   # PyTorch-level flag
    # or, equivalently for the repo scripts:
    export PREDECODER_TORCH_COMPILE=0
    ```
  - When running scripts directly (outside the notebook or `local_run.sh`),
    set the Python path so that repo modules are importable:
    ```bash
    export PYTHONPATH="code"
    ```
- **Pre-trained model not found during inference**:
  - `find_best_model` searches inside `{output}/models/best_model/` first,
    then falls back to `{output}/models/`. If you placed the downloaded
    `.pt` file elsewhere, either move it into one of those directories or
    point to it directly:
    ```bash
    PREDECODER_MODEL_CHECKPOINT_FILE=path/to/Ising-Decoder-SurfaceCode-1-Accurate.pt \
      WORKFLOW=inference bash code/scripts/local_run.sh
    ```

## Inference (pre-trained models)

If you are not training locally, you can run inference using pre-trained models.

1. **(Optional) create a venv and install inference deps**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r code/requirements_public_inference.txt
   ```

2. **Get the pre-trained models**
   This repo ships two pre-trained model files (tracked with Git LFS):
   - `models/Ising-Decoder-SurfaceCode-1-Fast.pt` (receptive field R=9)
   - `models/Ising-Decoder-SurfaceCode-1-Accurate.pt` (receptive field R=13)

   Clones get the files via `git lfs pull`. Optionally, set `PREDECODER_MODEL_URL` to the LFS/raw URL to fetch files when not in the working tree (e.g. in a minimal checkout or CI).

3. Set:

   - `EXPERIMENT_NAME=predecoder_model_1`
   - `model_id: 1` in `conf/config_public.yaml`

4. **Run inference**:

   ```bash
   WORKFLOW=inference EXPERIMENT_NAME=predecoder_model_1 bash code/scripts/local_run.sh
   ```

Inference output is written to `outputs/<EXPERIMENT_NAME>/` with a full log in
`outputs/<EXPERIMENT_NAME>/run.log`.

## Model export and downstream tools

### Converting .pt checkpoints to SafeTensors (optional, post-training)

By default, training produces `.pt` checkpoints under `outputs/<EXPERIMENT_NAME>/models/` and inference loads them directly. SafeTensors export is optional — use it when downstream tooling requires the SafeTensors format.

**Step 1 — convert the best trained checkpoint:**

```bash
PYTHONPATH=code python code/export/checkpoint_to_safetensors.py \
    --checkpoint outputs/<EXPERIMENT_NAME>/models/<checkpoint>.pt \
    --model-id <MODEL_ID> [--fp16]
```

Output is written next to the checkpoint (e.g. `<checkpoint>_fp16.safetensors`).

**Step 2 — run inference from the SafeTensors file:**

```bash
PREDECODER_SAFETENSORS_CHECKPOINT=outputs/<EXPERIMENT_NAME>/models/<checkpoint>_fp16.safetensors \
WORKFLOW=inference bash code/scripts/local_run.sh
```

`MODEL_ID` is the public model identifier (1–5); see `model/registry.py` for the mapping.
The pre-trained public models use `--model-id 1` (R=9) and `--model-id 4` (R=13).

### ONNX export and quantization (optional, post-training)

After training (or starting from the shipped `.safetensors` files), you can export the model to
ONNX and optionally apply INT8 or FP8 post-training quantization for deployment.

You may also change the surface code distance and number of rounds at inference
time. That is - you are not required retrain a new model when changing either
one of these parameters; since the model is a 3D convolutional neural network,
the model will simply be run over a new decoding volume.

- To run with a new distance, simply add `DISTANCE=<your distance>` to the commands below.
- To run with a new number of rounds, simply add `N_ROUNDS=<your number of rounds>` to the commands below.

Set the `ONNX_WORKFLOW` and (optionally) (`QUANT_FORMAT`, `DISTANCE`,
`N_ROUNDS`) environment variables before running inference with `local_run.sh`:

| `ONNX_WORKFLOW` | Behavior |
|---|---|
| `0` (default) | PyTorch inference only, no ONNX export |
| `1` | Export ONNX model and run inference with PyTorch |
| `2` | Export ONNX model and run inference via TensorRT |
| `3` | Load a pre-existing TensorRT engine file and run inference |

```bash
# Export ONNX only (no TensorRT)
ONNX_WORKFLOW=1 WORKFLOW=inference bash code/scripts/local_run.sh

# Export ONNX + apply INT8 quantization + run TensorRT inference
ONNX_WORKFLOW=2 QUANT_FORMAT=int8 WORKFLOW=inference bash code/scripts/local_run.sh

# Export ONNX + apply FP8 quantization + run TensorRT inference
ONNX_WORKFLOW=2 QUANT_FORMAT=fp8 WORKFLOW=inference bash code/scripts/local_run.sh

# Use a pre-built TensorRT engine (skip export)
ONNX_WORKFLOW=3 WORKFLOW=inference bash code/scripts/local_run.sh
```

**Quantization variables:**

| Variable | Default | Description |
|---|---|---|
| `QUANT_FORMAT` | unset | `int8` or `fp8`. Unset means no quantization (FP32 ONNX). |
| `QUANT_CALIB_SAMPLES` | `256` | Calibration samples for INT8/FP8 post-training quantization. |

**Circuit variables:**

| Variable | Default | Description |
|---|---|---|
| `CONFIG_NAME` | `config_public` | Use the defaults from the `conf/$CONFIG_NAME.yaml` file |
| `DISTANCE` | Use the distance specified in the `conf/$CONFIG_NAME.yaml` file | surface code distance |
| `N_ROUNDS` | Calibration samples for INT8/FP8 post-training quantization. | number of rounds in memory experiment |

Notes:

- TensorRT workflows (`ONNX_WORKFLOW=2` or `3`) require `tensorrt` and `modelopt`.
- FP8 quantization failure is fatal. INT8 failure falls back to the FP32 ONNX model silently.
- ONNX and engine files are written to the current working directory.
- `ONNX_WORKFLOW` is also honoured by the `decoder_ablation` workflow — see below.

### Generating data for CUDA-Q QEC realtime predecoder test application

When evaluating the neural pre-decoder in an end-to-end downstream system like
CUDA-Q Realtime, you will need a test harness with valid inputs—both the
exported neural network model and the corresponding syndrome data.

The utility script `code/export/generate_test_data.py` is provided to generate
this exact data (both an `.onnx` file and several `.bin` files) so you can
easily consume it in the CUDA-Q QEC realtime AI decoder.

> **Important:** The `--distance` and `--n-rounds` arguments provided to this
script **must match** the values used in the preceding section when running the
ONNX export (e.g. `ONNX_WORKFLOW=2`).

For a detailed walkthrough on how to ingest these files into the CUDA-Q Realtime
C++ pipeline, see the downstream documentation here: [Realtime AI Predecoder
Pipeline](https://nvidia.github.io/cudaqx/examples_rst/qec/realtime_predecoder_pymatching.html).

```bash
python3 code/export/generate_test_data.py --distance 13 --n-rounds 104 --num-samples 10000 --basis X --p-error=0.003 --simple-noise
```

**Example output:**

```text
Building circuit: D=13, T=104, basis=X, rotation=XV, p=0.003
  Circuit built in 0.007s
Building detector error model and PyMatching matcher...
  DEM + matcher built in 0.083s
  Detectors: 17472, Observables: 1
Extracting check matrices (beliefmatching)...
  H shape: (17472, 93864), O shape: (1, 93864), priors shape: (93864,)
Sampling 10000 shots...
  Sampled in 1.006s
Decoding with PyMatching (baseline)...
  Errors: 30/10000, LER: 0.0030
  Decode time: 5.439s (543.9 µs/shot)
Writing outputs to test_data/d13_T104_X/
Done.
  H_csr.bin                           808,944 bytes
  O_csr.bin                             2,932 bytes
  detectors.bin                   698,880,008 bytes
  metadata.txt                            162 bytes
  observables.bin                      40,008 bytes
  priors.bin                          750,916 bytes
  pymatching_predictions.bin           40,008 bytes
```

### Decoder ablation study with cudaq-qec (optional)

The `decoder_ablation` workflow compares multiple global decoders on the residual syndromes left
by the neural pre-decoder. It supports both PyTorch and TensorRT backends for the pre-decoder
and GPU-accelerated global decoders from the `cudaq-qec` package (`cudaq_qec`).

**PyTorch pre-decoder + cudaq-qec global decoders:**

```bash
# Requires: cudaq-qec (cudaq_qec), ldpc, beliefmatching, scipy
WORKFLOW=decoder_ablation bash code/scripts/local_run.sh
```

**TRT pre-decoder + cudaq-qec global decoders (full GPU pipeline):**

The same `ONNX_WORKFLOW` variable used for `inference` also applies here. When a TRT engine is
active, the neural pre-decoder runs via TensorRT (fast, quantised inference) while `cudaq-qec`
decoders handle the residual syndromes on GPU — combining fast TRT inference with
GPU-accelerated global decoding end-to-end.

```bash
# Export ONNX, build TRT engine, run ablation (TRT pre-decoder + cudaq-qec)
ONNX_WORKFLOW=2 WORKFLOW=decoder_ablation bash code/scripts/local_run.sh

# INT8 quantized TRT pre-decoder + cudaq-qec
ONNX_WORKFLOW=2 QUANT_FORMAT=int8 WORKFLOW=decoder_ablation bash code/scripts/local_run.sh

# Load a previously built engine, then run ablation
ONNX_WORKFLOW=3 WORKFLOW=decoder_ablation bash code/scripts/local_run.sh
```

The ablation study reports per-decoder logical error rates, convergence statistics for
`cudaq-qec` BP variants, residual syndrome weight distributions, and timing breakdowns.
Results are written to `outputs/<EXPERIMENT_NAME>/plots/`.

**Decoder variants benchmarked:**

| Decoder | Source | Notes |
|---|---|---|
| No-op | — | Pre-decoder output only, no global correction |
| Union-Find | `ldpc` | Fast, sub-optimal LER (Logical Error Rate) |
| BP-only | `ldpc` | Belief propagation, no OSD |
| BP+LSD-0 | `ldpc` | BP with localized statistics decoding |
| Uncorr-PM | PyMatching | Uncorrelated minimum-weight perfect matching |
| Corr-PM | PyMatching | Correlated MWPM (best classical baseline) |
| cudaq-BP | `cudaq-qec` | Sum-product BP on GPU |
| cudaq-MinSum | `cudaq-qec` | Min-sum BP on GPU |
| cudaq-BP+OSD-0/7 | `cudaq-qec` | BP + ordered statistics decoding |
| cudaq-MemBP | `cudaq-qec` | Memory-based min-sum BP |
| cudaq-MemBP+OSD | `cudaq-qec` | Memory BP + OSD |
| cudaq-RelayBP | `cudaq-qec` | Sequential relay composition |

`cudaq-qec` decoders are loaded automatically when `cudaq_qec` is importable; the study
degrades gracefully to the non-cudaq decoders if the package is absent.

## Configuration and advanced usage

### GPU selection

- **Defaults**: if you do not set `CUDA_VISIBLE_DEVICES` or `GPUS`, all GPUs are used.

- **Use one specific GPU** (recommended for precise selection):

```bash
CUDA_VISIBLE_DEVICES=1 GPUS=1 bash code/scripts/local_run.sh
```

- **Use multiple GPUs** (first N visible devices):

```bash
GPUS=4 bash code/scripts/local_run.sh
```

- **Explicit multi-GPU selection** (more granular than `GPUS`):

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 bash code/scripts/local_run.sh
```

### Public configuration (`conf/config_public.yaml`)

External users should only edit `conf/config_public.yaml`.
If you change any config settings, also change the experiment name so outputs are not mixed.

#### Model selection

- `model_id`: one of **{1,2,3,4,5}**

Each `model_id` has a fixed receptive field \(R\):

- **model 1**: \(R=9\)
- **model 2**: \(R=9\)
- **model 3**: \(R=17\)
- **model 4**: \(R=13\)
- **model 5**: \(R=13\)

#### Distance / rounds semantics

- Top-level `distance` / `n_rounds` are the **evaluation targets** (what you care about in inference).
- Training runs on the model receptive field: **distance = n_rounds = R**.

#### Code orientation

- `data.code_rotation`: **O1, O2, O3, O4**

For a concrete picture, here are the **distance-3** layouts and the corresponding **logical operator supports** (● = in the logical, · = not in the logical).

```text
============
O1
============
CODE LAYOUT:
      (z)
    D     D     D
      [X]   [Z]   (x)
    D     D     D
(x)   [Z]   [X]
    D     D     D
            (z)

LOGICAL X (lx):
 ●  ●  ●
 ·  ·  ·
 ·  ·  ·

LOGICAL Z (lz):
 ●  ·  ·
 ●  ·  ·
 ●  ·  ·

============
O2
============
CODE LAYOUT:
            (x)
    D     D     D
(z)   [X]   [Z]
    D     D     D
      [Z]   [X]   (z)
    D     D     D
      (x)

LOGICAL X (lx):
 ●  ·  ·
 ●  ·  ·
 ●  ·  ·

LOGICAL Z (lz):
 ●  ●  ●
 ·  ·  ·
 ·  ·  ·

============
O3
============
CODE LAYOUT:
      (x)
    D     D     D
      [Z]   [X]   (z)
    D     D     D
(z)   [X]   [Z]
    D     D     D
            (x)

LOGICAL X (lx):
 ●  ·  ·
 ●  ·  ·
 ●  ·  ·

LOGICAL Z (lz):
 ●  ●  ●
 ·  ·  ·
 ·  ·  ·

============
O4
============
CODE LAYOUT:
            (z)
    D     D     D
(x)   [Z]   [X]
    D     D     D
      [X]   [Z]   (x)
    D     D     D
      (z)

LOGICAL X (lx):
 ●  ●  ●
 ·  ·  ·
 ·  ·  ·

LOGICAL Z (lz):
 ●  ·  ·
 ●  ·  ·
 ●  ·  ·
```

#### Noise model (public default)

- `data.noise_model`: a **25-parameter circuit-level** noise model (SPAM, idles, and CNOT Pauli channels).

#### Training noise upscaling (surface code)

When training a surface-code pre-decoder the noise parameters you specify may be very small (e.g. `p = 1e-4`), which produces extremely sparse syndromes and slow convergence. To address this, the training pipeline **automatically upscales** all 25 noise-model parameters so that the largest grouped total `max(P_prep, P_meas, P_idle_cnot, P_idle_spam, P_cnot)` equals a fixed target of **6 × 10⁻³** (just below the surface-code threshold of ~7.5 × 10⁻³).

The five grouped totals are:

| Group | Sum of |
|-------|--------|
| P_prep | `p_prep_X + p_prep_Z` |
| P_meas | `p_meas_X + p_meas_Z` |
| P_idle_cnot | `p_idle_cnot_X + p_idle_cnot_Y + p_idle_cnot_Z` |
| P_idle_spam | `p_idle_spam_X + p_idle_spam_Y + p_idle_spam_Z` |
| P_cnot | sum of all 15 `p_cnot_*` |

**Upscaling rules:**

- If `max_group < 6e-3`: all 25 p's are multiplied by `6e-3 / max_group` for training data generation only. Evaluation always uses the original user-specified noise model as-is.
- If `max_group >= 6e-3`: parameters are **not** modified (the training log emits a warning in case this indicates a configuration error).
- Non-surface-code types (`code_type != "surface_code"`) are never upscaled.

We have found that training on denser syndromes and then evaluating on sparser data produces better results than training directly on sparse data.

#### Skipping noise upscaling

If you need to train with your **exact** noise parameters (e.g. for benchmarking or controlled experiments), you can disable upscaling via config or environment variable:

**Config** (`conf/config_public.yaml`):

```yaml
data:
  skip_noise_upscaling: true
  noise_model:
    p_prep_X: 0.002
    # ... rest of 25 params
```

**Environment variable:**

```bash
PREDECODER_SKIP_NOISE_UPSCALING=1 bash code/scripts/local_run.sh
```

Either method causes the training pipeline to use the user-specified noise model verbatim — no scaling is applied. The training log will confirm:

```
[Train] noise_model upscaling SKIPPED (skip_noise_upscaling=true or PREDECODER_SKIP_NOISE_UPSCALING=1).
```

### Precomputed frames (recommended)

Training/validation data generation can load precomputed frames from:

- `frames_data/`

If frames are missing, the code can fall back to on-the-fly generation, but it is slower. To precompute frames:

```bash
python3 code/data/precompute_frames.py --distance 13 --n_rounds 13 --basis X Z --rotation O1
```

### Resuming training and running inference on a trained model

- **Inference uses the trained model from `outputs/<experiment_name>/models/`**, so keep the same `EXPERIMENT_NAME` when you switch from training to inference.
- **Training auto-resumes**: if a run is interrupted, launching the same training command again (same `EXPERIMENT_NAME`) will automatically load the latest checkpoint it finds and continue training (up to the fixed 100 epochs). To force a clean restart, set `FRESH_START=1`, although we recommend changing `EXPERIMENT_NAME` instead.

## Logging and outputs

### What gets written where

Runs are organized under:

- `outputs/<experiment_name>/`
  - `models/` (checkpoints + model files)
  - `tensorboard/`
  - `config/` (a snapshot of the config used for each run)
  - `run.log` (copy of the latest run’s log)
- `logs/<experiment_name>_<timestamp>/`
  - `<workflow>.log` (full stdout/stderr)

`code/scripts/local_run.sh` automatically snapshots the config into:

- `outputs/<experiment_name>/config/<config_name>_<timestamp>.yaml`
- `outputs/<experiment_name>/config/<config_name>_<timestamp>.overrides.txt`

#### TensorBoard (training metrics)

TensorBoard logs live under `outputs/<experiment_name>/tensorboard/`.

Key scalars (as shown in TensorBoard):

- **`Loss/train_step`**: Training loss (BCEWithLogits) logged every optimization step. Lower is better.
- **`LearningRate/train`**: The current learning rate (after warmup/schedule) per training step.
- **`BatchSize`**: The **effective** batch size per epoch: `per_device_batch_size * accumulate_steps * world_size`. We accumulate 2 steps: one for X basis circuit, and another one for Z basis.
- **`Metrics/LER`**: Logical Error Rate on the evaluation target (computed during training-time evaluation). Lower is better.
  - Averaging: computed over `cfg.test.num_samples` Monte Carlo shots **per basis** (X and Z).
  - Default: `cfg.test.num_samples = 262144` (hardcoded for the current public release).
  - Distributed: each rank uses `cfg.test.num_samples // world_size` shots per basis (any remainder is dropped).
- **`Metrics/LER_Reduction_Factor`**: Ratio of post-predecoder LER to baseline LER (a “relative improvement” factor). `>1` means improvement. If both are 0, we log `1.0`.
  - Averaging: derived from the same LER evaluation run (same shot count as `Metrics/LER`).
- **`Metrics/PyMatching_Speedup`**: Average PyMatching speedup from the pre-decoder: `latency_baseline / latency_after`. `>1` means faster decoding of PyMatching after pre-decoding.
  - Averaging: latencies are measured on a small subset (`cfg.test.latency_num_samples`, default `10000`) using **single-shot** PyMatching (`batch_size=1`, `matcher.decode`) and reported as microseconds/round.
- **`Metrics/SDR`**: Syndrome Density Reduction factor: `syndrome_density_before / syndrome_density_after`. `>1` means the pre-decoder reduced syndrome density.
- **`EarlyStopping/epochs_since_best`**: How many epochs since the best validation metric (we use LER as the validation metric).
- **`EarlyStopping/best_metric`**: The best (lowest) validation loss observed so far.

### Evaluation defaults (public release)

- **Validation loss** during training uses the on-the-fly generator.
- **Testing / inference metrics** (LER / SDR / latency) default to the **Stim** path.

## Testing and CI

### Testing (CPU + GPU)

CPU-only tests are fast and recommended for quick validation:

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -p "test_*.py"
```

GPU tests are automatically skipped when no GPU is available. On a GPU machine
all tests run, including those gated behind `torch.cuda.is_available()`:

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -p "test_*.py"
```

Useful env vars for noise model tests:

- `RUN_SLOW=1` enables >=100k-shot statistical tests
- `NOISEMODEL_FAST_SHOTS` controls fast-tier shots (default 10000)
- `NOISEMODEL_SLOW_SHOTS` controls slow-tier shots (default 100000)

Example fast GPU run:

```bash
NOISEMODEL_FAST_SHOTS=2000 PYTHONPATH=code python -m unittest code/tests/test_noise_model.py
```

**Test coverage (local):** To see which code is exercised by tests and get a report:

```bash
pip install -r code/requirements_public_inference.txt -r code/requirements_ci.txt
PYTHONPATH=code coverage run -m unittest discover -s code/tests -p "test_*.py"
coverage report
coverage html -d htmlcov   # open htmlcov/index.html in a browser
```

CI runs the same suite with coverage and publishes `htmlcov/` and `coverage.xml` as
job artifacts.

### CI (GitHub Actions)

CI is defined in `.github/workflows/ci.yml` and runs on pushes to `main`,
`pull-request/*` branches (via copy-pr-bot), merge-group checks, and manual
dispatch:

| Job | Runner | What it checks |
|-----|--------|----------------|
| `spdx-header-check` | CPU | SPDX licence headers on all source files |
| `unit-tests` | CPU | Full `unittest discover` suite (GPU tests auto-skip) |
| `unit-tests-coverage` | CPU | Same suite with `coverage` reporting |
| `python-compat` | CPU | Import/install check across Python 3.11 / 3.12 / 3.13 |
| `gpu-tests` | GPU | Full test suite on a self-hosted GPU runner |
| `gpu-tests` (train+inference) | GPU | Short train + inference with LER check |

## Results

Logical error rate (LER) vs. time for X-basis decoding at physical error rates p = 0.003 and 0.006:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/ler_vs_time_model_card_p0.003_0.006_X_dark.svg">
  <img src="images/ler_vs_time_model_card_p0.003_0.006_X_light.svg" alt="LER vs time (X basis, p=0.003–0.006)">
</picture>

## License

This project is released under the [Apache License 2.0](LICENSE).

Every source file in this repository carries an [SPDX](https://spdx.dev/) copyright and license
header of the form:

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Presence of these headers is enforced automatically by the `spdx-header-check` CI job (see
`.github/workflows/ci.yml`).

Third-party open source components bundled with or required by this project are listed with their
respective copyright notices and license texts in [NOTICE](NOTICE).
