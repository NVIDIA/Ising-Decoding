# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

# Ensure repo's code/ is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import resolve_precomputed_frames_dir


class TestPrecomputedFramesFallback(unittest.TestCase):
    def test_missing_frames_dir_falls_back(self):
        with TemporaryDirectory() as tmp:
            result = resolve_precomputed_frames_dir(tmp, 9, 9, "both", rank=0)
            self.assertIsNone(result)

    def test_existing_frames_dir_is_used(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            for basis in ("X", "Z"):
                prefix = f"surface_d9_r9_{basis}_frame_predecoder"
                (d / f"{prefix}.X.npz").touch()
                (d / f"{prefix}.Z.npz").touch()
                (d / f"{prefix}.p.npz").touch()
            result = resolve_precomputed_frames_dir(str(d), 9, 9, "both", rank=0)
            self.assertEqual(result, str(d))


if __name__ == "__main__":
    unittest.main()
