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
Neural network model definitions.

Contains:
- factory: Factory for creating models from config
- predecoder: Pre-decoder model architectures (PreDecoderModelMemory_v1)
"""
from model.factory import ModelFactory

# Import predecoder models lazily to avoid hard dependency on optional training
# stacks (e.g., physicsnemo) during lightweight config validation.
try:
    from model.predecoder import PreDecoderModelMemory_v1
except ModuleNotFoundError:
    PreDecoderModelMemory_v1 = None
