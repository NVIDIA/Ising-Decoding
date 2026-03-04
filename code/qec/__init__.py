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
Quantum Error Correction modules.

Contains:
- surface_code/: Surface code specific modules
  - memory_circuit: Stim-based circuit generation (MemoryCircuit, SurfaceCode)
  - homological_equivalence: Error simplification via homological equivalence
  - data_mapping: Stabilizer-to-data qubit mappings
  - stim_utils: Stim circuit utilities
"""
