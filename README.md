# AI pre-decoder for surface-code memory circuits

This repo implements a **pre-decoder** for surface-code memory experiments:

- A neural network consumes detector syndromes across space **and** time
- It predicts corrections that reduce syndrome density / improve decoding
- A standard decoder (PyMatching) produces the final logical decision

The public release exposes a **single user-facing config** and a **single runner script**.

### Quick start (train + inference)

From the repo root:

- `code/scripts/local_run.sh`

This script runs the Hydra workflow locally (no SLURM required) and reads **one** user-facing config file:

- `conf/config_public.yaml`

## Dependencies

Target Python versions: **3.11, 3.12, 3.13**.

Two minimal requirements files are provided:

- `code/requirements_public_inference.txt` (Stim + PyTorch path)
- `code/requirements_public_train.txt` (training path)

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

# Training (includes inference deps)
pip install -r code/requirements_public_train.txt

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

### Troubleshooting

- **Avoid `steps_per_epoch=0` on short runs**:
  - Keep `PREDECODER_TRAIN_SAMPLES >= per_device_batch_size * accumulate_steps * world_size`.
  - Note: the batch schedule jumps to 2048 after epoch 0, so epoch 1 uses
    `2048 * 2 * world_size` effective batch size.
  - For quick smoke runs, use `GPUS=1` and `PREDECODER_TRAIN_SAMPLES >= 4096`.
- **Segfaults during training startup (torch.compile)**:
  - Some environments crash during `torch.compile`.
  - Disable compile: `TORCH_COMPILE=0 bash code/scripts/local_run.sh`.
  - Or try a safer mode: `TORCH_COMPILE=1 TORCH_COMPILE_MODE=reduce-overhead bash code/scripts/local_run.sh`.

### Inference (pre-trained models)

If you are not training locally, you can run inference using pre-trained models.

1. **(Optional) create a venv and install inference deps**:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r code/requirements_public_inference.txt
```

2. **Get the pre-trained model**  
   **Package users** receive model files via your distribution channel (e.g. release package or archive), not from this repo. Place the model file so the loader can find it:
   - Either extract the provided archive into the experiment models folder:
     ```bash
     mkdir -p outputs/predecoder_model_1/models
     tar xzf /path/to/PreDecoderModelMemory_v1.0.94.pt.tar.gz -C outputs/predecoder_model_1/models
     ```
     This creates `outputs/predecoder_model_1/models/PreDecoderModelMemory_v1.0.94.pt`.
   - Or, if your package includes a download URL for the model, set `PREDECODER_MODEL_URL` (or `MODEL_URL`) and run with `EXTRA_PARAMS="+test.use_model_checkpoint=94"`; the loader will download to the experiment models folder when the file is missing.

   **Repository users**: This repo tracks the model with Git LFS. Clones get the file via `git lfs pull`. Optionally, set `PREDECODER_MODEL_URL` to the LFS/raw URL to fetch the file when not in the working tree (e.g. in a minimal checkout or CI).

3. Set:
- `EXPERIMENT_NAME=predecoder_model_1`
- `model_id: 1` in `conf/config_public.yaml`

4. **Run inference**:
```bash
WORKFLOW=inference EXPERIMENT_NAME=predecoder_model_1 bash code/scripts/local_run.sh
```

Inference output is written to `outputs/<EXPERIMENT_NAME>/` with a full log in
`outputs/<EXPERIMENT_NAME>/run.log`.

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

Training may apply a **training-only** “noise floor” sparsity guard (scales probabilities up if the grouped totals are too small). Evaluation uses the user-specified noise model **as-is**. We have seen that it is preferrable to train on data that is more dense and then apply it to sparser data that training on the sparse data.

### Precomputed frames (recommended)

Training/validation data generation can load precomputed frames from:

- `frames_data/`

If frames are missing, the code can fall back to on-the-fly generation, but it is slower. To precompute frames:

```bash
python3 code/data/precompute_frames.py --distance 13 --n_rounds 13 --basis X Z --rotation O1
```

### Resuming training & running inference on a trained model


- **Inference uses the trained model from `outputs/<experiment_name>/models/`**, so keep the same `EXPERIMENT_NAME` when you switch from training to inference.
- **Training auto-resumes**: if a run is interrupted, launching the same training command again (same `EXPERIMENT_NAME`) will automatically load the latest checkpoint it finds and continue training (up to the fixed 100 epochs). To force a clean restart, set `FRESH_START=1`, although we recommend changing `EXPERIMENT_NAME` instead.


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
  - Default: `cfg.test.num_samples = 262144` (hardcoded for the current public release). If the training noise “floor” rescaling triggers, we increase this to at least `1048576` for a cleaner evaluation signal.
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
`pull-request/*` branches (via copy-pr-bot), merge-group checks, and manual dispatch:

| Job | Runner | What it checks |
|-----|--------|----------------|
| `spdx-header-check` | CPU | SPDX licence headers on all source files |
| `unit-tests` | CPU | Full `unittest discover` suite (GPU tests auto-skip) |
| `unit-tests-coverage` | CPU | Same suite with `coverage` reporting |
| `python-compat` | CPU | Import/install check across Python 3.11 / 3.12 / 3.13 |
| `gpu-tests` | GPU | Full test suite on a GPU runner (GPU tests included) |
| `smoke-test-gpu` | GPU | Short train + inference end-to-end smoke run |