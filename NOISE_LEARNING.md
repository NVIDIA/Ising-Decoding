# Learning the noise model from detector data

Pre-decoder training generates its data from the 25-parameter circuit-level noise model in
`data.noise_model` (see `README.md`, "Noise model"). A model trained on rates that match the
device decodes better than one trained on assumed rates. When the device rates are unknown,
CUDA-Q QEC can estimate them from recorded detector data, without logical labels and without
prior knowledge of the rates. This guide shows how to turn that estimate into a
`data.noise_model` block and train on it.

The noise learning itself is part of CUDA-Q QEC (`cudaq_qec.plugins.nv_qldpc`), not of this
repository. For the method's options, limits and measured performance, see the CUDA-Q QEC
documentation:

- [Noise Learning](https://nvidia.github.io/cudaq-qec/components/qec/noise_learning.html)
- [Learning Noise and Decoding with Learned Rates](https://nvidia.github.io/cudaq-qec/examples_rst/qec/noise_learning.html)

## Workflow

1. Run surface-code memory experiments on the device (X and Z basis) and record the detector
   data.
2. Learn the 25 parameters from that data with `learn_noise(mode="parametric")`.
3. Write them as a `data.noise_model` block.
4. Train the pre-decoder with that block (`README.md`, "Quick start").

## Requirements

- CUDA-Q QEC 0.9.0 or later with the `nv-qldpc-decoder` plugin, and an NVIDIA GPU.
- The inference requirements of this repository (the script imports the circuit builder).

```bash
pip install -r code/requirements_public_inference.txt
pip install cudaq-qec-cu12 pyyaml   # or cudaq-qec-cu13, to match your CUDA runtime
```

## Example

Run from the repository root. The script learns the noise model of a distance-7 surface-code
memory experiment from 262,144 shots per basis and writes `learned_noise_model.yaml`. It takes
about one minute on one GPU.

The detector data here is simulated from a noise model that the learning step never sees, so
that the result can be compared with the truth. For a device, replace `dets` with the recorded
data: a `uint8` array of shape `[shots, detectors]`, with detectors in the order of the
circuit built by `MemoryCircuit` for the same distance, rounds, basis and orientation.

```python
import sys

import numpy as np
import yaml

sys.path.insert(0, "code")  # run from the repository root
from qec.noise_model import NoiseModel
from qec.surface_code.memory_circuit import MemoryCircuit
from cudaq_qec.plugins import nv_qldpc
from cudaq_qec.plugins.nv_qldpc import p25, tying

DISTANCE, ROUNDS, ROTATION = 7, 7, "XV"  # XV/XH/ZV/ZH = O1/O2/O3/O4
SHOTS = 262144

# cudaq-qec names a prep/meas parameter by the Pauli that is applied, this
# repository by the basis that fails; the four keys are exchanged at the boundary.
SWAP = {"p_prep_X": "p_prep_Z", "p_prep_Z": "p_prep_X",
        "p_meas_X": "p_meas_Z", "p_meas_Z": "p_meas_X"}


def swap(d):
    return {SWAP.get(k, k): v for k, v in d.items()}


def memory_circuit(noise, basis):
    c = MemoryCircuit(distance=DISTANCE, n_rounds=ROUNDS, basis=basis,
                      code_rotation=ROTATION, noise_model=NoiseModel.from_config_dict(noise),
                      idle_error=1e-3, sqgate_error=1e-3, tqgate_error=1e-3, spam_error=1e-3,
                      add_boundary_detectors=True)
    c.set_error_rates()
    return c.stim_circuit


# Stand-in for the device: a noise model the learning step never sees.
rng = np.random.default_rng(7)
true_noise = {k: float(v * rng.uniform(0.5, 2.0))
              for k, v in NoiseModel.from_single_p(0.003).to_config_dict().items()}

tags = p25.tag_noise_model()
learned, identifiable = {}, {}
tcbp_model = None
for basis in ("Z", "X"):
    # Any valid noise model gives the same detector error matrix H; only its structure is used.
    circuit = memory_circuit(NoiseModel.from_single_p(0.003).to_config_dict(), basis)
    H, _, _ = tying.dem_arrays(tying.circuit_dem(circuit))
    design = p25.design_map(memory_circuit(swap(tags), basis), tags, target_circuit=circuit)
    if tcbp_model is None:
        tcbp_model = nv_qldpc.init_tcbp_model(H, seed=1)  # once per code family

    # Detector data, uint8 [shots, detectors]. Replace with data recorded on the device.
    device = memory_circuit(true_noise, basis)
    dets = device.compile_detector_sampler(seed=1).sample(SHOTS).astype(np.uint8)

    _, report = nv_qldpc.learn_noise(H, dets, tcbp_model, mode="parametric",
                                     design=design, passes=16, tol=0)
    learned[basis] = swap(p25.export_p25(report["q_hat"], design["names"]))
    identifiable[basis] = swap(dict(zip(design["names"], report["q_identifiable"])))
    print(f"{basis} memory: rank {report['design_rank']} of {report['n_params']}")

# Preparation and measurement parameters come from the experiment in their own basis,
# every other parameter from the runs that identify it on its own.
noise_model = {}
for k in p25.P25_PARAM_NAMES:
    if k in SWAP:
        runs = [k[-1]]
    else:
        runs = [b for b in learned if identifiable[b][k]] or list(learned)
    noise_model[k] = float(np.mean([learned[b][k] for b in runs]))

NoiseModel.from_config_dict(noise_model)  # validates the 25 keys
with open("learned_noise_model.yaml", "w") as f:
    yaml.safe_dump({"data": {"noise_model": noise_model}}, f, sort_keys=False)

for k in p25.P25_PARAM_NAMES:
    print(f"{k:14s} learned {noise_model[k]:.3e}  true {true_noise[k]:.3e}")
```

## Training with the learned noise model

Copy the `noise_model` block of `learned_noise_model.yaml` over `data.noise_model` in
`conf/config_public.yaml`, then train as usual:

```bash
bash code/scripts/local_run.sh
```

Training upscales the noise model by default (`README.md`, "Training noise upscaling");
evaluation uses the learned values as written. Set `data.skip_noise_upscaling: true` to train on
the learned values exactly.

## Notes

- **Parameter names.** In this repository `p_prep_Z` and `p_meas_Z` are the failure
  probabilities of a Z-basis preparation and measurement (an X flip). `cudaq_qec`'s `p25`
  helpers name the same parameters by the applied Pauli (`p_prep_X`, `p_meas_X`). The `swap`
  step in the example converts between the two; without it `p25.design_map` rejects the
  circuit. Because of this, the `--tag-circuit` and `--yaml` options of the
  `cudaq_qec.plugins.nv_qldpc.cli.learn_physical_params` command-line tool write the four
  prep/meas keys exchanged for this repository.
- **Use both bases.** One memory basis fixes 23 of the 25 parameter combinations.
  `report["q_identifiable"]` marks the parameters that one experiment does not determine on
  their own: the four preparation and measurement parameters and two CNOT parameters, a
  different pair in each basis. A Z-basis experiment sees `p_prep_X` and `p_meas_X` only as
  a sum and returns the same value for both; an X-basis experiment does the same for
  `p_prep_Z` and `p_meas_Z`. The example therefore takes each preparation and measurement
  parameter from the experiment in its own basis, and every other parameter from the runs
  that identify it.
- **Shots.** Accuracy improves with the number of shots. The CNOT parameters are the smallest
  and need the most data. In the example (distance 7, 262,144 shots per basis) the
  preparation, measurement and idle parameters come out within 3% of the true values and 14
  of the 15 CNOT parameters within 20%.
- **Model assumption.** The parametric mode assumes the device follows the 25-parameter model
  with the same rates in every round and on every qubit. If that does not hold, use the
  per-mechanism or tied modes described in the CUDA-Q QEC documentation; their output is a
  rate per error mechanism, which a decoder can use as a prior but which cannot be written as
  `data.noise_model`.
- **Surface code only.** The example covers the surface-code memory circuits of this
  repository.
