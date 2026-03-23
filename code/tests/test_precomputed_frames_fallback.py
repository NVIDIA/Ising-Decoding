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
