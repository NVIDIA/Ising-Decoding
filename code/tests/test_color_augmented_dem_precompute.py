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
"""Color-code augmented DEM precompute tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_color_precompute_p_vector_uses_25p_values(basis):
    """Color DEM precompute should build p from the explicit 25p model, not scalar p/3 or p/K."""
    from qec.noise_model import CNOT_ERROR_TYPES, NoiseModel
    from qec.precompute_dem import precompute_dem_bundle_color_code

    cnot_probs = {f"p_cnot_{k}": 0.00011 + i * 0.00001 for i, k in enumerate(CNOT_ERROR_TYPES)}
    nm = NoiseModel(
        p_prep_X=0.0011,
        p_prep_Z=0.0022,
        p_meas_X=0.0033,
        p_meas_Z=0.0044,
        p_idle_cnot_X=0.0051,
        p_idle_cnot_Y=0.0052,
        p_idle_cnot_Z=0.0053,
        p_idle_spam_X=0.0061,
        p_idle_spam_Y=0.0062,
        p_idle_spam_Z=0.0063,
        **cnot_probs
    )

    artifacts = precompute_dem_bundle_color_code(
        distance=3,
        n_rounds=3,
        basis=basis,
        schedule="nearest-neighbor",
        p_scalar=0.1234,
        dem_output_dir=None,
        device=torch.device("cpu"),
        export=False,
        return_artifacts=True,
        enable_z_feedforward=True,
        apply_data_x_override=True,
        use_decomposed_errors=False,
        chunk_size=64,
        buffer_size=1,
        noise_model=nm,
    )
    p_values = artifacts["p"].cpu().numpy()

    expected_values = [
        nm.p_idle_cnot_X,
        nm.p_idle_cnot_Y,
        nm.p_idle_cnot_Z,
        nm.p_cnot_IX,
        nm.p_cnot_ZZ,
    ]
    for expected in expected_values:
        assert np.any(np.isclose(p_values, expected, rtol=0.0, atol=1e-9)
                     ), (f"Expected 25p probability {expected} in color DEM p vector")

    scalar_derived_values = [0.1234 / 3.0, 0.1234 / 15.0, 2.0 * 0.1234 / 3.0]
    for scalar_value in scalar_derived_values:
        assert not np.any(
            np.isclose(p_values, scalar_value, rtol=0.0, atol=1e-9)
        ), (f"Unexpected scalar-derived probability {scalar_value} in 25p color DEM p vector")


def test_color_precompute_p_vector_scalar_path_unchanged_when_noise_model_none():
    """Passing noise_model=None must reproduce the legacy scalar p vector bit-for-bit."""
    from qec.precompute_dem import precompute_dem_bundle_color_code

    common_kwargs = dict(
        distance=3,
        n_rounds=2,
        basis="X",
        schedule="nearest-neighbor",
        p_scalar=0.01,
        dem_output_dir=None,
        device=torch.device("cpu"),
        export=False,
        return_artifacts=True,
        enable_z_feedforward=True,
        apply_data_x_override=True,
        use_decomposed_errors=False,
        chunk_size=256,
        buffer_size=1,
    )
    baseline = precompute_dem_bundle_color_code(**common_kwargs)
    with_none = precompute_dem_bundle_color_code(**common_kwargs, noise_model=None)
    np.testing.assert_array_equal(baseline["p"].cpu().numpy(), with_none["p"].cpu().numpy())
    np.testing.assert_array_equal(
        baseline["H"].cpu().numpy(),
        with_none["H"].cpu().numpy(),
    )


def test_color_precompute_export_writes_noise_metadata(tmp_path):
    """Exported color DEM should record noise_mode + sha256 + canonical_parameters."""
    import json

    from qec.noise_model import NoiseModel
    from qec.precompute_dem import (
        DEM_ARTIFACT_METADATA_KEY,
        get_color_augmented_dem_paths,
        precompute_dem_bundle_color_code,
    )

    nm = NoiseModel.from_single_p(0.005)

    distance, n_rounds, basis, schedule = 3, 2, "X", "nearest-neighbor"
    precompute_dem_bundle_color_code(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
        p_scalar=0.123,
        dem_output_dir=str(tmp_path),
        device=torch.device("cpu"),
        export=True,
        enable_z_feedforward=True,
        apply_data_x_override=True,
        use_decomposed_errors=False,
        chunk_size=64,
        buffer_size=1,
        noise_model=nm,
    )

    paths = get_color_augmented_dem_paths(
        tmp_path,
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
    )
    with np.load(paths["p"], allow_pickle=False) as z:
        raw = z[DEM_ARTIFACT_METADATA_KEY]
        meta = json.loads(str(raw.item() if raw.shape == () else raw.reshape(-1)[0]))

    assert meta["noise_mode"] == "noise_model"
    assert meta["noise_model_sha256"] == nm.sha256()
    assert meta["noise_model"] == nm.canonical_parameters()
    assert meta["code"] == "color"


def test_color_precompute_export_scalar_metadata_unchanged(tmp_path):
    """When noise_model is omitted, exported metadata should still record noise_mode=scalar."""
    import json

    from qec.precompute_dem import (
        DEM_ARTIFACT_METADATA_KEY,
        get_color_augmented_dem_paths,
        precompute_dem_bundle_color_code,
    )

    distance, n_rounds, basis, schedule = 3, 2, "Z", "nearest-neighbor"
    precompute_dem_bundle_color_code(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
        p_scalar=0.004,
        dem_output_dir=str(tmp_path),
        device=torch.device("cpu"),
        export=True,
        enable_z_feedforward=True,
        apply_data_x_override=True,
        use_decomposed_errors=False,
        chunk_size=64,
        buffer_size=1,
    )

    paths = get_color_augmented_dem_paths(
        tmp_path,
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
    )
    with np.load(paths["p"], allow_pickle=False) as z:
        raw = z[DEM_ARTIFACT_METADATA_KEY]
        meta = json.loads(str(raw.item() if raw.shape == () else raw.reshape(-1)[0]))

    assert meta["noise_mode"] == "scalar"
    assert "noise_model_sha256" not in meta
    assert float(meta["p_scalar"]) == pytest.approx(0.004)


def test_color_augmented_dem_rejects_nonproduction_feedforward_modes():
    from qec.precompute_dem import precompute_dem_bundle_color_code

    common = dict(
        distance=3,
        n_rounds=3,
        basis="X",
        schedule="nearest-neighbor",
        p_scalar=0.01,
        dem_output_dir=None,
        device=torch.device("cpu"),
        export=False,
        return_artifacts=True,
        use_decomposed_errors=False,
    )
    with pytest.raises(ValueError, match="enable_z_feedforward=True"):
        precompute_dem_bundle_color_code(
            **common,
            enable_z_feedforward=False,
            apply_data_x_override=True,
        )
    with pytest.raises(ValueError, match="apply_data_x_override=True"):
        precompute_dem_bundle_color_code(
            **common,
            enable_z_feedforward=True,
            apply_data_x_override=False,
        )
