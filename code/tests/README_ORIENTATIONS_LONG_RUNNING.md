# Long-running tests for all four surface code orientations

For the three-tier test model (short / mid / long), see **README_TEST_TIERS.md**.

The pre-decoder supports **four surface code orientations** (measurement layouts): **O1, O2, O3, O4** (internal names: XV, XH, ZV, ZH). Training and inference are **per orientation**: each run uses a single `data.code_rotation` from config.

## Short-term tests (CI)

Short tests that cover all four orientations run in normal CI:

- **test_public_config.py**: `test_code_rotation_o1_to_o4_accepted_and_normalized`, `test_code_rotation_internal_aliases_accepted`
- **test_boundary_detectors.py**: `TestCodeRotations.test_all_rotations_dem_builds`, `test_all_rotations_detector_count_consistent`, `TestLERComparison.test_ler_improves_with_bd_all_orientations`
- **test_boundary_detector_integration.py**: detector count match, decoding with appended boundary detectors, and boundary-detector consistency for all four orientations

CI discovers all `code/tests/test_*.py` automatically. Run from repo root:

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -p "test_*.py"
```

## Long-running tests (30 minutes - several hours)

These are **not** intended for standard pre-merge CI. They run in the scheduled
`long-running-tests.yml` workflow or on dedicated servers.

### 1. Training per orientation

Train one run per orientation (O1..O4). Each run uses the same config except
`data.code_rotation`. Full training can take hours per orientation depending on
model and hardware.

From repo root:

```bash
ORIENTATIONS_LONG_TASK=train bash code/scripts/run_orientations_long.sh
```

Or manually, for each orientation:

```bash
# O1 (default in config_public.yaml)
WORKFLOW=train EXPERIMENT_NAME=orient_O1 bash code/scripts/local_run.sh

# O2, O3, O4
WORKFLOW=train EXPERIMENT_NAME=orient_O2 EXTRA_PARAMS="data.code_rotation=O2" bash code/scripts/local_run.sh
WORKFLOW=train EXPERIMENT_NAME=orient_O3 EXTRA_PARAMS="data.code_rotation=O3" bash code/scripts/local_run.sh
WORKFLOW=train EXPERIMENT_NAME=orient_O4 EXTRA_PARAMS="data.code_rotation=O4" bash code/scripts/local_run.sh
```

### 2. Inference / LER evaluation per orientation

Run inference (pre-decoder + PyMatching LER) for each orientation. With default test
sample counts (e.g. 262k), each orientation can take on the order of tens of minutes to
several hours depending on distance, rounds, and hardware.

```bash
ORIENTATIONS_LONG_TASK=inference bash code/scripts/run_orientations_long.sh
```

Or manually:

```bash
WORKFLOW=inference EXPERIMENT_NAME=orient_O1 bash code/scripts/local_run.sh
WORKFLOW=inference EXPERIMENT_NAME=orient_O2 EXTRA_PARAMS="data.code_rotation=O2" bash code/scripts/local_run.sh
# ... O3, O4
```

### 3. Running in CI (scheduled)

The `orientation-inference` job in `.github/workflows/long-running-tests.yml` runs
inference over all 4 orientations daily. Manual dispatch is also available.

### 4. Running on a remote server (SSH)

Copy the repo (or clone), install dependencies, then run the same commands:

```bash
ssh user@remote-server "cd /path/to/quantum-predecoder && ORIENTATIONS_LONG_TASK=train bash code/scripts/run_orientations_long.sh"
```

## Orientation mapping

| Public name | Internal name | Description |
|-------------|---------------|-------------|
| O1 | XV | X-type first bulk syndrome, vertical boundary |
| O2 | XH | X-type first bulk syndrome, horizontal boundary |
| O3 | ZV | Z-type first bulk syndrome, vertical boundary |
| O4 | ZH | Z-type first bulk syndrome, horizontal boundary |

Config accepts either public (O1..O4) or internal (XV, XH, ZV, ZH) names; they are
normalized internally.
