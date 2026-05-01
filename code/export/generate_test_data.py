# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate test data for the pre-decoder evaluation pipeline.

Builds a Stim memory circuit, samples syndrome data, extracts the detector
error model (H, O, priors) via beliefmatching, decodes with PyMatching, and
optionally runs an ONNX pre-decoder model.  All artefacts are written to a
directory in a custom binary format so that residual syndromes can be fed
into PyMatching for evaluation.

Output files
------------
All files use little-endian byte order.

  metadata.txt              Plain text, one key=value per line.  Records the
                            circuit parameters used for generation.

  detectors.bin             Stim detector values (syndrome bits).
                            Header: (num_samples: uint32, num_detectors: uint32)
                            Body:   num_samples × num_detectors int32 values
                                    (row-major, each value 0 or 1).

  observables.bin           Ground-truth logical observable for each shot.
                            Header: (num_samples: uint32, num_observables: uint32)
                            Body:   num_samples × num_observables int32 values.

  pymatching_predictions.bin
                            Baseline PyMatching decode of the raw detectors
                            (no pre-decoder).  Same format as observables.bin.

  H_csr.bin                 Parity check matrix from the detector error model
                            (binary CSR — all non-zero entries are 1).
                            Header: (rows: uint32, cols: uint32, nnz: uint32)
                            Body:   (rows + 1) int32 indptr values,
                                    nnz int32 column-index values.

  O_csr.bin                 Observable matrix from the detector error model.
                            Same binary-CSR format as H_csr.bin.

  priors.bin                Edge error probabilities from the detector error
                            model (one per column of H / O).
                            Header: (n: uint32)
                            Body:   n float64 values.

  predecoder_outputs.bin    (only written when --onnx-model is supplied)
                            Raw output of the ONNX pre-decoder model.
                            Header: (num_samples: uint32, 1 + num_detectors: uint32)
                            Body:   num_samples × (1 + num_detectors) uint8 values.
                            Column 0 is pre_L; columns 1.. are residual detectors.

Usage
-----
    python generate_test_data.py --distance 13 --n-rounds 104 --basis X \\
        --num-samples 1000 --output-dir ../../test_data/d13_T104_X

    # Optionally include ONNX pre-decoder outputs:
    python generate_test_data.py --distance 13 --n-rounds 104 --basis X \\
        --num-samples 1000 --onnx-model /path/to/predecoder.onnx \\
        --output-dir ../../test_data/d13_T104_X
"""

import sys
import struct
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pymatching
from beliefmatching.belief_matching import detector_error_model_to_check_matrices

# Import MemoryCircuit without triggering qec/surface_code/__init__.py,
# which pulls in data_mapping → torch.  We only need memory_circuit.py
# (numpy + stim) and noise_model.py, so we bypass the package __init__.
import importlib, types

_sc_pkg = types.ModuleType("qec.surface_code")
_sc_pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "qec" / "surface_code")]
_sc_pkg.__package__ = "qec.surface_code"
sys.modules.setdefault("qec.surface_code", _sc_pkg)

from qec.surface_code.memory_circuit import MemoryCircuit
from qec.noise_model import NoiseModel

# Default 25-parameter noise model matching config_public.yaml at p=0.003
DEFAULT_NOISE_PARAMS = {
    "p_prep_X": 0.002,
    "p_prep_Z": 0.002,
    "p_meas_X": 0.002,
    "p_meas_Z": 0.002,
    "p_idle_cnot_X": 0.001,
    "p_idle_cnot_Y": 0.001,
    "p_idle_cnot_Z": 0.001,
    "p_idle_spam_X": 0.001996,
    "p_idle_spam_Y": 0.001996,
    "p_idle_spam_Z": 0.001996,
    "p_cnot_IX": 0.0002,
    "p_cnot_IY": 0.0002,
    "p_cnot_IZ": 0.0002,
    "p_cnot_XI": 0.0002,
    "p_cnot_XX": 0.0002,
    "p_cnot_XY": 0.0002,
    "p_cnot_XZ": 0.0002,
    "p_cnot_YI": 0.0002,
    "p_cnot_YX": 0.0002,
    "p_cnot_YY": 0.0002,
    "p_cnot_YZ": 0.0002,
    "p_cnot_ZI": 0.0002,
    "p_cnot_ZX": 0.0002,
    "p_cnot_ZY": 0.0002,
    "p_cnot_ZZ": 0.0002,
}

_ROTATION_ALIASES = {"O1": "XV", "O2": "XH", "O3": "ZV", "O4": "ZH"}

# ---------------------------------------------------------------------------
# Binary I/O helpers
# ---------------------------------------------------------------------------


def save_dense_bin(path: str, arr: np.ndarray) -> None:
    """Save a 2-D array with an 8-byte header: (rows: u32, cols: u32)."""
    rows, cols = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", rows, cols))
        f.write(arr.tobytes())


def save_csr_bin(path: str, mat) -> None:
    """Save a scipy CSR matrix: (rows: u32, cols: u32, nnz: u32, indptr, indices).

    Values are not stored — the matrix is assumed binary (all non-zeros are 1).
    """
    from scipy import sparse
    csr = sparse.csr_matrix(mat)
    rows, cols = csr.shape
    nnz = csr.nnz
    with open(path, "wb") as f:
        f.write(struct.pack("<III", rows, cols, nnz))
        f.write(csr.indptr.astype(np.int32).tobytes())
        f.write(csr.indices.astype(np.int32).tobytes())


def save_priors_bin(path: str, priors: np.ndarray) -> None:
    """Save a 1-D float64 array: (n: u32, data as float64)."""
    n = len(priors)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(priors.astype(np.float64).tobytes())


def save_metadata(path: str, **kwargs) -> None:
    with open(path, "w") as f:
        for k, v in kwargs.items():
            f.write(f"{k}={v}\n")


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def generate_test_data(
    distance: int = 13,
    n_rounds: int = 104,
    basis: str = "X",
    p_error: float = 0.003,
    code_rotation: str = "XV",
    noise_model_params: dict | None = None,
    num_samples: int = 1000,
    onnx_model: str | None = None,
    output_dir: str = "test_data",
):
    code_rotation = _ROTATION_ALIASES.get(code_rotation.upper(), code_rotation.upper())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Noise model ----
    noise_model = None
    if noise_model_params is not None:
        noise_model = NoiseModel.from_config_dict(noise_model_params)

    p_placeholder = float(noise_model.get_max_probability()) if noise_model else float(p_error)

    # ---- Build Stim circuit ----
    print(
        f"Building circuit: D={distance}, T={n_rounds}, basis={basis}, "
        f"rotation={code_rotation}, p={p_error}"
    )
    t0 = time.perf_counter()
    circ = MemoryCircuit(
        distance=distance,
        idle_error=p_placeholder,
        sqgate_error=p_placeholder,
        tqgate_error=p_placeholder,
        spam_error=(2.0 / 3.0) * p_placeholder,
        n_rounds=n_rounds,
        basis=basis.upper(),
        code_rotation=code_rotation,
        noise_model=noise_model,
        add_boundary_detectors=True,
    )
    circ.set_error_rates()
    stim_circuit = circ.stim_circuit
    print(f"  Circuit built in {time.perf_counter() - t0:.3f}s")

    # ---- Detector error model + PyMatching ----
    print("Building detector error model and PyMatching matcher...")
    t0 = time.perf_counter()
    det_model = stim_circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    matcher = pymatching.Matching.from_detector_error_model(det_model)
    print(f"  DEM + matcher built in {time.perf_counter() - t0:.3f}s")
    print(f"  Detectors: {det_model.num_detectors}, Observables: {det_model.num_observables}")

    # ---- Extract H, O, priors via beliefmatching ----
    print("Extracting check matrices (beliefmatching)...")
    matrices = detector_error_model_to_check_matrices(det_model)
    H = matrices.edge_check_matrix
    O = matrices.edge_observables_matrix
    edge_probs = matrices.hyperedge_to_edge_matrix @ matrices.priors
    eps = 1e-14
    edge_probs[edge_probs > 1 - eps] = 1 - eps
    edge_probs[edge_probs < eps] = eps
    priors = edge_probs
    print(f"  H shape: {H.shape}, O shape: {O.shape}, priors shape: {priors.shape}")

    # ---- Sample syndrome data ----
    print(f"Sampling {num_samples} shots...")
    t0 = time.perf_counter()
    meas = stim_circuit.compile_sampler().sample(shots=num_samples)
    converter = stim_circuit.compile_m2d_converter()
    dets_and_obs = converter.convert(measurements=meas, append_observables=True)

    stim_dets = np.asarray(dets_and_obs[:, :-stim_circuit.num_observables], dtype=np.int32)
    stim_obs = np.asarray(dets_and_obs[:, -stim_circuit.num_observables:], dtype=np.int32)
    print(f"  Sampled in {time.perf_counter() - t0:.3f}s")
    assert stim_dets.shape[1] == det_model.num_detectors, (
        f"Detector width {stim_dets.shape[1]} != DEM {det_model.num_detectors}"
    )

    # ---- PyMatching baseline decode ----
    print("Decoding with PyMatching (baseline)...")
    t0 = time.perf_counter()
    predictions = matcher.decode_batch(np.asarray(stim_dets, dtype=np.uint8))
    decode_time = time.perf_counter() - t0
    predictions = np.asarray(predictions, dtype=np.int32).reshape(-1, stim_circuit.num_observables)
    num_errors = int((predictions != stim_obs).sum())
    ler = num_errors / num_samples
    print(f"  Errors: {num_errors}/{num_samples}, LER: {ler:.4f}")
    print(f"  Decode time: {decode_time:.3f}s "
          f"({decode_time / num_samples * 1e6:.1f} µs/shot)")

    # ---- Optional ONNX pre-decoder inference ----
    predecoder_outputs = None
    if onnx_model and Path(onnx_model).is_file():
        print(f"Running ONNX pre-decoder: {onnx_model}")
        try:
            import onnxruntime as ort
            t0 = time.perf_counter()
            sess = ort.InferenceSession(
                onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            dets_uint8 = np.asarray(stim_dets, dtype=np.uint8)
            result = sess.run(None, {"dets": dets_uint8})
            predecoder_outputs = np.asarray(result[0], dtype=np.uint8)
            print(
                f"  Pre-decoder ran in {time.perf_counter() - t0:.3f}s, "
                f"output shape: {predecoder_outputs.shape}"
            )
        except Exception as e:
            print(f"  ONNX inference failed: {e}")

    # ---- Save everything ----
    print(f"Writing outputs to {out}/")

    save_dense_bin(str(out / "detectors.bin"), stim_dets)
    save_dense_bin(str(out / "observables.bin"), stim_obs)
    save_dense_bin(str(out / "pymatching_predictions.bin"), predictions)
    save_csr_bin(str(out / "H_csr.bin"), H)
    save_csr_bin(str(out / "O_csr.bin"), O)
    save_priors_bin(str(out / "priors.bin"), priors)

    if predecoder_outputs is not None:
        save_dense_bin(str(out / "predecoder_outputs.bin"), predecoder_outputs)

    noise_label = "25-param" if noise_model_params else "simple"
    save_metadata(
        str(out / "metadata.txt"),
        distance=distance,
        n_rounds=n_rounds,
        basis=basis.upper(),
        code_rotation=code_rotation,
        p_error=p_error,
        num_samples=num_samples,
        num_detectors=det_model.num_detectors,
        num_observables=det_model.num_observables,
        H_shape=H.shape,
        noise_model=noise_label,
        **({
            "onnx_model": onnx_model
        } if onnx_model else {}),
    )

    print("Done.")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:30s} {f.stat().st_size:>12,d} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test data for the pre-decoder evaluation pipeline"
    )
    parser.add_argument("--distance", type=int, default=13)
    parser.add_argument("--n-rounds", type=int, default=104)
    parser.add_argument("--basis", type=str, default="X", choices=["X", "Z"])
    parser.add_argument(
        "--code-rotation", type=str, default="XV", help="XV, XH, ZV, ZH or public aliases O1-O4"
    )
    parser.add_argument("--p-error", type=float, default=0.003)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument(
        "--onnx-model", type=str, default=None, help="Path to ONNX pre-decoder model (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: test_data/d{D}_T{T}_{basis})"
    )
    parser.add_argument(
        "--simple-noise",
        action="store_true",
        help="Use simple p_error instead of 25-parameter noise model"
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"test_data/d{args.distance}_T{args.n_rounds}_{args.basis}"

    noise_params = None if args.simple_noise else DEFAULT_NOISE_PARAMS

    generate_test_data(
        distance=args.distance,
        n_rounds=args.n_rounds,
        basis=args.basis,
        p_error=args.p_error,
        code_rotation=args.code_rotation,
        noise_model_params=noise_params,
        num_samples=args.num_samples,
        onnx_model=args.onnx_model,
        output_dir=args.output_dir,
    )
