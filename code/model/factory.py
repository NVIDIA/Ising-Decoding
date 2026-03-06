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
Factory module for creating models.

Provides ModelFactory for instantiating pre-decoder models from config.
"""


class ModelFactory:

    @staticmethod
    def create_model(cfg):
        if cfg.code == "surface":
            return ModelFactory._create_surface_model(cfg)
        else:
            raise ValueError("Invalid model name")

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
        else:
            raise ValueError(f"Invalid model version: {cfg.model.version}")
