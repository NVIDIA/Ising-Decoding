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
Neural network model definitions.

Contains:
- factory: Factory for creating models from config
- predecoder: Pre-decoder model architectures (PreDecoderModelMemory_v1, PreDecoderModelMemory_v2)
"""
from model.factory import ModelFactory

# Import predecoder models lazily to avoid hard dependency on optional training
# stacks (e.g., physicsnemo) during lightweight config validation.
try:
    from model.predecoder import PreDecoderModelMemory_v1, PreDecoderModelMemory_v2
except ModuleNotFoundError:
    PreDecoderModelMemory_v1 = None
    PreDecoderModelMemory_v2 = None
