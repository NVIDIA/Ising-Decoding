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
Minimal logging helpers to replace physicsnemo.launch.logging.
"""

from __future__ import annotations

import logging


class PythonLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def success(self, msg: str) -> None:
        self._logger.info(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)
