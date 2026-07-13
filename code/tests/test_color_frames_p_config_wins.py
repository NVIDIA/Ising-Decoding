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
"""Regression test: for the color path, the *configured* noise p wins over the
precomputed bundle's baked-in p.

Background: a precomputed augmented-DEM bundle caches the (expensive) DEM
*structure* together with a per-error probability vector built at some p. The
color circuit used to adopt the bundle's ``p_nominal`` wholesale and overwrite the
configured ``p_min``/``p_max`` -- so pointing training at a bundle built at a
different p than the config requested would silently train at the bundle's p
(a silent train/eval noise mismatch). The surface path already avoids this by
recomputing the probability vector at the configured p; this test pins the same
behaviour for color: structure is reused from the bundle, probabilities follow the
configured p, and the bundle's p is honoured only when no p is configured (legacy).

Construction-only (no sampling) so it runs on CPU without cuStabilizer.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch

from qec.color_code.memory_circuit_torch import ColorMemoryCircuitTorch
from qec.precompute_dem import precompute_dem_bundle_color_code

_D = 3
_R = 3
_BASIS = "X"
_SCHEDULE = "nearest-neighbor"
_BUNDLE_P = 0.001  # the p the frames are built at
_CONFIG_P = 0.004  # the p the config requests (4x the bundle's)


def _build_bundle(frames_dir: str, p_scalar: float) -> None:
    precompute_dem_bundle_color_code(
        distance=_D,
        n_rounds=_R,
        basis=_BASIS,
        schedule=_SCHEDULE,
        p_scalar=p_scalar,
        dem_output_dir=frames_dir,
        device=torch.device("cpu"),
        export=True,
    )


def _circuit(frames_dir: str, p_scalar):
    return ColorMemoryCircuitTorch(
        distance=_D,
        n_rounds=_R,
        basis=_BASIS,
        schedule=_SCHEDULE,
        precomputed_frames_dir=frames_dir,
        apply_spacelike_he=False,  # skip HE cache build; not needed for p check
        device=torch.device("cpu"),
        p_scalar=p_scalar,
    )


class TestColorFramesConfigPWins(unittest.TestCase):

    def test_configured_p_overrides_bundle_p(self):
        """p_scalar from config must drive p_nominal/p_min/p_max and rebuild self.p."""
        with tempfile.TemporaryDirectory() as frames_dir:
            _build_bundle(frames_dir, _BUNDLE_P)

            # Sanity: bundle really was built at the (different) bundle p.
            legacy = _circuit(frames_dir, None)
            self.assertAlmostEqual(legacy.bundle_p_nominal, _BUNDLE_P, places=9)
            self.assertAlmostEqual(legacy.p_nominal, _BUNDLE_P, places=9)
            self.assertAlmostEqual(legacy.p_min, _BUNDLE_P, places=9)
            self.assertAlmostEqual(legacy.p_max, _BUNDLE_P, places=9)

            # Configured p must win, structure reused from the bundle.
            cfg = _circuit(frames_dir, _CONFIG_P)
            self.assertAlmostEqual(cfg.bundle_p_nominal, _BUNDLE_P, places=9)
            self.assertAlmostEqual(cfg.p_nominal, _CONFIG_P, places=9)
            self.assertAlmostEqual(cfg.p_min, _CONFIG_P, places=9)
            self.assertAlmostEqual(cfg.p_max, _CONFIG_P, places=9)
            self.assertAlmostEqual(cfg.active_p_nominal, _CONFIG_P, places=9)

            # Same DEM structure (column count unchanged), but probabilities rescaled.
            self.assertEqual(cfg.p.shape, legacy.p.shape)
            # To leading order p_err scales with p_scalar, so a 4x config p gives a
            # ~4x larger probability vector than the legacy (bundle-p) one.
            ratio = float(cfg.p.sum() / legacy.p.sum())
            self.assertAlmostEqual(ratio, _CONFIG_P / _BUNDLE_P, delta=0.5)

    def test_matching_p_is_a_noop(self):
        """When the configured p equals the bundle p, behaviour is unchanged."""
        with tempfile.TemporaryDirectory() as frames_dir:
            _build_bundle(frames_dir, _BUNDLE_P)
            same = _circuit(frames_dir, _BUNDLE_P)
            self.assertAlmostEqual(same.p_nominal, _BUNDLE_P, places=9)
            legacy = _circuit(frames_dir, None)
            self.assertTrue(torch.allclose(same.p, legacy.p))

    def test_noise_model_drives_p_while_config_p_sets_nominal(self):
        """The reviewer's case: in practice users train by specifying a ``noise_model``.

        When one is configured the per-error probabilities come from the 25-param
        model (the bundle's baked-in p is never used for them), while the configured
        ``p`` still drives the nominal ``p_nominal``/``p_min``/``p_max`` and
        ``active_p_nominal`` reflects the model's grouped totals -- so neither the
        probabilities nor the reported noise follow the bundle.
        """
        from qec.noise_model import NoiseModel, get_grouped_totals

        with tempfile.TemporaryDirectory() as frames_dir:
            _build_bundle(frames_dir, _BUNDLE_P)
            legacy = _circuit(frames_dir, None)  # scalar, bundle p, no noise_model

            nm = NoiseModel.from_single_p(_CONFIG_P)
            mc = ColorMemoryCircuitTorch(
                distance=_D,
                n_rounds=_R,
                basis=_BASIS,
                schedule=_SCHEDULE,
                precomputed_frames_dir=frames_dir,
                apply_spacelike_he=False,
                device=torch.device("cpu"),
                p_scalar=_CONFIG_P,
                noise_model=nm,
            )

            # Same DEM structure reused from the bundle...
            self.assertEqual(mc.p.shape, legacy.p.shape)
            # ...but the probabilities come from the noise_model, not the bundle's p.
            self.assertFalse(torch.allclose(mc.p, legacy.p))
            # Reported (active) noise tracks the model's grouped totals, not the bundle.
            self.assertAlmostEqual(
                mc.active_p_nominal, float(get_grouped_totals(nm)["max_group"]), places=9
            )
            # The configured p still sets the nominal fields (config wins over bundle p).
            self.assertAlmostEqual(mc.p_nominal, _CONFIG_P, places=9)
            self.assertAlmostEqual(mc.p_min, _CONFIG_P, places=9)
            self.assertAlmostEqual(mc.p_max, _CONFIG_P, places=9)
            self.assertAlmostEqual(mc.bundle_p_nominal, _BUNDLE_P, places=9)


if __name__ == "__main__":
    unittest.main()
