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
Data generation and loading modules.

Contains:
- factory: Factory for creating data loaders and datapipes
- generator_torch: Torch on-the-fly surface-code data generator (QCDataGeneratorTorch)
- generator_torch_color: Torch on-the-fly color-code data generator
- generator_torch_multi: Multi-pair (distance, n_rounds) round-robin Torch generator
- datapipe_stim: Stim-based datapipe for inference
"""
from data.factory import DatapipeFactory
