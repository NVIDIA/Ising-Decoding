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
Minimal static capture placeholder to replace physicsnemo.utils.capture._StaticCapture.
"""

from __future__ import annotations


class _StaticCapture:
    _amp_scalers = {}
    _amp_scaler_checkpoints = {}

    @classmethod
    def state_dict(cls):
        return {
            "amp_scalers": cls._amp_scalers,
            "amp_scaler_checkpoints": cls._amp_scaler_checkpoints,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        if not state_dict:
            return
        cls._amp_scalers = state_dict.get("amp_scalers", {})
        cls._amp_scaler_checkpoints = state_dict.get("amp_scaler_checkpoints", {})
