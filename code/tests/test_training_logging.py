# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Tests for training.logging (PythonLogger)."""

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from training.logging import PythonLogger


class TestPythonLogger(unittest.TestCase):

    def test_init_and_info(self):
        log = PythonLogger("test_training_logging")
        log.info("info message")

    def test_warning(self):
        log = PythonLogger("test_training_logging_warn")
        log.warning("warning message")

    def test_success(self):
        log = PythonLogger("test_training_logging_ok")
        log.success("success message")
