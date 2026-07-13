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
Factory module for creating models.

Provides ModelFactory for instantiating pre-decoder models from config.
"""


class ModelFactory:

    @staticmethod
    def create_model(cfg):
        # Phase A: allow training the same predecoder model on either surface or color data.
        # The architectures are grid-based Conv3D and don't inherently require a square (d x d) lattice.
        code_name = str(getattr(cfg, "code", "surface")).lower()
        if code_name == "surface" or code_name == "surface_partition" or code_name.startswith(
            "color"
        ):
            return ModelFactory._create_surface_model(cfg)
        raise ValueError(f"Invalid cfg.code={cfg.code!r} for model creation")

    @staticmethod
    def _create_surface_model(cfg):
        if cfg.model.version == "predecoder_memory_v1":
            from model.predecoder import PreDecoderModelMemory_v1
            model = PreDecoderModelMemory_v1(cfg)
            return model
        elif cfg.model.version == "predecoder_memory_v2":
            from model.predecoder import PreDecoderModelMemory_v2
            model = PreDecoderModelMemory_v2(cfg)
            return model
        elif cfg.model.version == "predecoder_memory_v3_natten":
            from model.predecoder import PreDecoderModelMemory_v3
            model = PreDecoderModelMemory_v3(cfg)
            return model
        elif cfg.model.version == "predecoder_memory_cascade":
            from model.predecoder import PreDecoderModelMemory_Cascade
            model = PreDecoderModelMemory_Cascade(cfg)
            return model
        else:
            raise ValueError(f"Invalid model version: {cfg.model.version}")
