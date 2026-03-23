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
