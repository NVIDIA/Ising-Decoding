# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
