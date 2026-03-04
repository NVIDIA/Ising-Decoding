# Test tiers: short, mid, and long

Tests and validation runs are grouped into three tiers by runtime and when they run.

## Tier 1: Short (pre-merge required)

- **Runtime:** Typically under a few minutes total.
- **When:** Run in CI before every merge; required to pass.
- **Purpose:** Fast feedback on correctness: config validation, unit tests, boundary detectors, integration, latency checks. No full training or long inference.

**How to run (repo root):**

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -p "test_*.py"
```

Or use the tier script:

```bash
bash code/scripts/run_tests_tier.sh short
```

CI uses the same discovery pattern; no per-file registration.

**CI jobs:** `unit-tests`, `unit-tests-coverage`, `gpu-tests` (in `.github/workflows/ci.yml`).

---

## Tier 2: Mid (~5-10 minutes, pre-merge GPU)

- **Runtime:** On the order of 5-10 minutes (depends on GPU hardware).
- **When:** Run in CI on every push/PR, after the smoke test passes. Requires GPU.
- **Purpose:** Sanity-check that training converges and inference works with larger sample
  counts than the minimal smoke test. Catches regressions that need more than a few
  hundred samples to surface (e.g. training instability, LER order of magnitude).

**Typical contents:**

- Extended smoke: 2 epochs, 32k training samples, 4k val/test samples.
- Training + inference end-to-end with the public config.

**How to run:**

```bash
bash code/scripts/run_tests_tier.sh mid
```

Or directly with custom parameters:

```bash
EXPERIMENT_NAME=my_mid \
PREDECODER_TRAIN_SAMPLES=32768 \
PREDECODER_TRAIN_EPOCHS=2 \
bash code/scripts/smoke_run.sh
```

**CI job:** `mid-gpu-tests` (in `.github/workflows/ci.yml`).

---

## Tier 3: Long (30 min+, scheduled / on-demand)

- **Runtime:** 30 minutes to several hours per job.
- **When:** Daily scheduled runs and manual dispatch. **Not** triggered by push/PR.
- **Purpose:** Full statistical validation, multi-orientation coverage, LER regression
  checks, and production-scale training. Answers "does the full pipeline work at scale?"
  and "are the LER numbers stable?".

**CI workflow:** `.github/workflows/long-running-tests.yml`

| Job | Runtime | What it validates |
|-----|---------|-------------------|
| `statistical-noise-model` | ~15 min | 100k+ shot noise model tests (`RUN_SLOW=1`) |
| `orientation-inference` | ~30-60 min | Inference over all 4 orientations (O1-O4) |
| `ler-regression` | ~30-60 min | LER quality at d=9 and d=13 with pre-trained models |
| `full-epoch-training` | ~30-60 min | 1 epoch with 2M samples + LER validation |

**How to run locally:**

```bash
# Full training for all four orientations
ORIENTATIONS_LONG_TASK=train bash code/scripts/run_orientations_long.sh

# Full inference for all four orientations
ORIENTATIONS_LONG_TASK=inference bash code/scripts/run_orientations_long.sh

# Or use the tier script
bash code/scripts/run_tests_tier.sh long
```

**Manual dispatch:** Go to Actions > "Long-running tests" > "Run workflow". Optionally
specify a comma-separated list of job names to run a subset (e.g.
`statistical-noise-model,ler-regression`).

---

## Summary

| Tier | Runtime | When | CI file | Purpose |
|------|---------|------|---------|---------|
| Short | < few min | Pre-merge | `ci.yml` | Correctness; required for merge |
| Mid | ~5-10 min | Pre-merge (GPU) | `ci.yml` | Extended smoke; training convergence |
| Long | 30 min - hours | Daily / on-demand | `long-running-tests.yml` | Full matrix; regression / benchmark |
