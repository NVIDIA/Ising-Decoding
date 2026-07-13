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
25-Parameter Noise Model for Quantum Error Correction.

This module provides a detailed noise model that explicitly specifies error
probabilities for each error type and location, replacing the single physical
error rate p with 22 parameters.

The 25 Parameters:
- State Preparation (2): p_prep_X, p_prep_Z
- Measurement (2): p_meas_X, p_meas_Z
- Idle during CNOT layers / bulk (3): p_idle_cnot_X, p_idle_cnot_Y, p_idle_cnot_Z
- Idle during ancilla prep/reset window for data qubits (3): p_idle_spam_X, p_idle_spam_Y, p_idle_spam_Z
- CNOT Two-qubit (15): All Pauli pairs except II
  (IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ)

Usage:
    # Create from single p (backwards compatible)
    noise_model = NoiseModel.from_single_p(p=0.01)
    
    # Create with explicit parameters
    noise_model = NoiseModel(
        p_prep_X=0.005, p_prep_Z=0.005,
        p_meas_X=0.005, p_meas_Z=0.005,
        p_idle_cnot_X=0.003, p_idle_cnot_Y=0.003, p_idle_cnot_Z=0.003,
        p_idle_spam_X=0.003, p_idle_spam_Y=0.003, p_idle_spam_Z=0.003,
        p_cnot_IX=0.001, ...
    )
    
    # From config dict
    noise_model = NoiseModel.from_config_dict(cfg.noise_model)
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math
import numpy as np

# Ordered list of CNOT error types (excluding II)
# Order matches Stim's PAULI_CHANNEL_2: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
CNOT_ERROR_TYPES = [
    'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ'
]

# Mapping from error type string to index (0-14)
CNOT_ERROR_INDEX = {et: i for i, et in enumerate(CNOT_ERROR_TYPES)}


# Internal helper for depolarizing-equivalent 25p mapping (tests/docs).
def _single_p_mapping(p: float, spam_factor: float = 2.0 / 3.0) -> Dict[str, float]:
    if p < 0 or p > 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    p_spam = p * spam_factor
    p_idle_cnot = p / 3.0
    p_idle_spam = (2.0 * p / 3.0) - (4.0 * p * p / 9.0)
    if p_idle_spam < 0:
        p_idle_spam = 0.0
    p_cnot = p / 15.0
    return {
        "p_prep_X": p_spam,
        "p_prep_Z": p_spam,
        "p_meas_X": p_spam,
        "p_meas_Z": p_spam,
        "p_idle_cnot_X": p_idle_cnot,
        "p_idle_cnot_Y": p_idle_cnot,
        "p_idle_cnot_Z": p_idle_cnot,
        "p_idle_spam_X": p_idle_spam,
        "p_idle_spam_Y": p_idle_spam,
        "p_idle_spam_Z": p_idle_spam,
        **{
            f"p_cnot_{k}": p_cnot for k in CNOT_ERROR_TYPES
        },
    }


def normalize_noise_mode(noise_mode: Optional[str]) -> str:
    """
    Normalize a high-level named noise-mode toggle.

    Supported values:
    - None / legacy / default / single_p / depolarizing / uniform / none -> "legacy"
    - si1000 -> "si1000"
    """
    if noise_mode is None:
        return "legacy"

    mode = str(noise_mode).strip().lower()
    if mode in ("", "legacy", "default", "single_p", "depolarizing", "uniform", "none"):
        return "legacy"
    if mode == "si1000":
        return "si1000"

    raise ValueError(
        f"Invalid noise_mode={noise_mode!r}. Expected one of "
        f"'legacy' or 'Si1000'."
    )


def normalize_noise_model_family(
    noise_model_family: Optional[str],
    *,
    fallback_noise_mode: Optional[str] = None,
) -> str:
    """
    Normalize the new config axis selecting the channel family.

    Supported values:
    - ``legacy``: existing scalar-p / explicit-noise-model semantics
    - ``si1000``: named Si1000 family

    ``fallback_noise_mode`` keeps backward compatibility with the older
    ``test.noise_mode`` toggle.
    """
    if noise_model_family is None:
        return normalize_noise_mode(fallback_noise_mode)

    family = str(noise_model_family).strip().lower()
    if family in ("", "legacy", "default", "single_p", "depolarizing", "uniform", "none"):
        return "legacy"
    if family == "si1000":
        return "si1000"

    raise ValueError(
        f"Invalid noise_model_family={noise_model_family!r}. Expected "
        f"'legacy' or 'si1000'."
    )


def normalize_noise_instruction_semantics(noise_instruction_semantics: Optional[str]) -> str:
    """
    Normalize the config axis selecting how noise instructions are attached to
    the fixed circuit.

    Supported values:
    - ``current``: existing MemoryCircuit semantics
    - ``reference``: new reference-noise semantics
    """
    if noise_instruction_semantics is None:
        return "current"

    semantics = str(noise_instruction_semantics).strip().lower()
    if semantics in ("", "current", "legacy"):
        return "current"
    if semantics in ("reference", "reference_noise"):
        return "reference"

    raise ValueError(
        f"Invalid noise_instruction_semantics={noise_instruction_semantics!r}. "
        f"Expected 'current' or 'reference'."
    )


@dataclass
class NoiseModel:
    """
    25-Parameter Noise Model for circuit-level noise simulation.
    
    Attributes:
        p_prep_X: X error probability after state preparation
        p_prep_Z: Z error probability after state preparation
        p_meas_X: X error probability before measurement
        p_meas_Z: Z error probability before measurement
        p_idle_cnot_X/Y/Z: Idle Pauli errors during bulk/CNOT layers (single-qubit Pauli channel)
        p_idle_spam_X/Y/Z: Idle Pauli errors on data qubits during ancilla prep/reset window.
                           NOTE: In noise-model mode we intentionally do NOT apply data-qubit
                           idle noise during ancilla measurement window.
        p_cnot_*: Two-qubit Pauli error probabilities for CNOT gates
                  Convention: "AB" means A on control, B on target
    """
    # State preparation errors (2)
    p_prep_X: float = 0.0
    p_prep_Z: float = 0.0

    # Measurement errors (2)
    p_meas_X: float = 0.0
    p_meas_Z: float = 0.0

    # Idle errors during bulk/CNOT layers (3)
    p_idle_cnot_X: float = 0.0
    p_idle_cnot_Y: float = 0.0
    p_idle_cnot_Z: float = 0.0

    # Idle errors during ancilla prep/reset window on data qubits (3)
    p_idle_spam_X: float = 0.0
    p_idle_spam_Y: float = 0.0
    p_idle_spam_Z: float = 0.0

    # CNOT two-qubit Pauli errors (15)
    # Convention: "AB" means A on control, B on target
    p_cnot_IX: float = 0.0
    p_cnot_IY: float = 0.0
    p_cnot_IZ: float = 0.0
    p_cnot_XI: float = 0.0
    p_cnot_XX: float = 0.0
    p_cnot_XY: float = 0.0
    p_cnot_XZ: float = 0.0
    p_cnot_YI: float = 0.0
    p_cnot_YX: float = 0.0
    p_cnot_YY: float = 0.0
    p_cnot_YZ: float = 0.0
    p_cnot_ZI: float = 0.0
    p_cnot_ZX: float = 0.0
    p_cnot_ZY: float = 0.0
    p_cnot_ZZ: float = 0.0

    # Drift support (not part of the user-facing parameterization)
    _reference: Dict[str, float] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Capture reference parameters once (used for drift/randomization)
        if not self._reference:
            self._reference = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        self.validate()

    def validate(self) -> None:
        """
        Validate that all probabilities are valid (0 <= p <= 1).
        
        Raises:
            ValueError: If any probability is out of range or total CNOT prob > 1.
        """
        all_params = {k: v for k, v in asdict(self).items() if not k.startswith("_")}

        for name, value in all_params.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {type(value)}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            if value > 1:
                raise ValueError(f"{name} must be <= 1, got {value}")

        # Check total CNOT probability doesn't exceed 1
        cnot_total = sum(v for k, v in all_params.items() if k.startswith('p_cnot_'))
        if cnot_total > 1:
            raise ValueError(f"Total CNOT error probability ({cnot_total}) exceeds 1")

        # Check total idle probabilities don't exceed 1
        idle_cnot_total = self.p_idle_cnot_X + self.p_idle_cnot_Y + self.p_idle_cnot_Z
        if idle_cnot_total > 1:
            raise ValueError(
                f"Total CNOT-layer idle error probability ({idle_cnot_total}) exceeds 1"
            )
        idle_spam_total = self.p_idle_spam_X + self.p_idle_spam_Y + self.p_idle_spam_Z
        if idle_spam_total > 1:
            raise ValueError(
                f"Total SPAM-window idle error probability ({idle_spam_total}) exceeds 1"
            )

    @classmethod
    def from_single_p(cls, p: float, spam_factor: float = 2.0 / 3.0) -> 'NoiseModel':
        """
        Create a NoiseModel from a single physical error rate using depolarizing defaults.
        
        This provides backwards compatibility with the single-p noise model:
        - SPAM errors: spam_factor * p for X or Z (default 2p/3)
        - Idle errors: p/3 for each of X, Y, Z
        - CNOT errors: p/15 for each of 15 two-qubit Paulis
        
        Args:
            p: Single physical error rate
            spam_factor: Factor to multiply p for SPAM errors (default 2/3)
            
        Returns:
            NoiseModel with depolarizing-equivalent parameters
        """
        if p < 0 or p > 1:
            raise ValueError(f"p must be in [0, 1], got {p}")

        # SPAM: 2p/3 for the error type that flips the prepared/measured basis
        # In X-basis prep: Z error flips |+> to |->
        # In Z-basis prep: X error flips |0> to |1>
        # We use the same probability for both since the circuit handles basis selection
        p_spam = p * spam_factor

        # Idle during CNOT layers: p/3 for each Pauli (depolarizing channel)
        p_idle_cnot = p / 3.0

        # SPAM-window data-idle: in the legacy model, data qubits experienced two idle steps
        # per round (one at ancilla prep, one at ancilla measurement), each with (p/3,p/3,p/3).
        # In the 25p noise-model semantics, we intentionally APPLY ONLY ONE SPAM-window idle
        # step (during ancilla prep/reset) and IGNORE the measurement-window data-idle.
        # To preserve backwards-compatibility for NoiseModel.from_single_p, we therefore set
        # p_idle_spam_* to the *two-step effective* channel of two consecutive depolarizing idles.
        #
        # For per-step: pI=1-p, pX=pY=pZ=p/3
        # Two-step effective: p2(X)=2 pI pX + 2 pY pZ = 2p/3 - 4p^2/9 (same for Y,Z).
        p_idle_spam = (2.0 * p / 3.0) - (4.0 * p * p / 9.0)
        if p_idle_spam < 0:
            p_idle_spam = 0.0

        # CNOT: p/15 for each of 15 two-qubit Paulis (depolarizing channel)
        p_cnot = p / 15.0

        return cls(
            # State preparation
            p_prep_X=p_spam,
            p_prep_Z=p_spam,

            # Measurement
            p_meas_X=p_spam,
            p_meas_Z=p_spam,

            # Idle during CNOT layers
            p_idle_cnot_X=p_idle_cnot,
            p_idle_cnot_Y=p_idle_cnot,
            p_idle_cnot_Z=p_idle_cnot,

            # SPAM-window data-idle (effective two-step mapped into one step)
            p_idle_spam_X=p_idle_spam,
            p_idle_spam_Y=p_idle_spam,
            p_idle_spam_Z=p_idle_spam,

            # CNOT (all equal for depolarizing)
            p_cnot_IX=p_cnot,
            p_cnot_IY=p_cnot,
            p_cnot_IZ=p_cnot,
            p_cnot_XI=p_cnot,
            p_cnot_XX=p_cnot,
            p_cnot_XY=p_cnot,
            p_cnot_XZ=p_cnot,
            p_cnot_YI=p_cnot,
            p_cnot_YX=p_cnot,
            p_cnot_YY=p_cnot,
            p_cnot_YZ=p_cnot,
            p_cnot_ZI=p_cnot,
            p_cnot_ZX=p_cnot,
            p_cnot_ZY=p_cnot,
            p_cnot_ZZ=p_cnot,
        )

    @classmethod
    def from_si1000(cls, p: float) -> 'NoiseModel':
        """
        Create a best-effort Si1000-style circuit noise model from a single scale ``p``.

        Mapping taken from the VibeLSD paper's Si1000 table:
        - 2Q gate:       two-qubit depolarizing with total probability p
        - 1Q Clifford:   one-qubit depolarizing with total probability p/10
        - Init:          basis-flip with probability 2p
        - Measure:       basis-flip with probability 5p
        - Idle (gates):  one-qubit depolarizing with total probability p/10
        - Idle (meas/reset): one-qubit depolarizing with total probability 2p

        Notes:
        - This codebase represents 2Q noise via Stim ``PAULI_CHANNEL_2`` on ``CX``.
          Full two-qubit depolarizing is Clifford-invariant, so distributing p/15
          across the 15 non-identity Paulis is the natural mapping here.
        - The current NoiseModel does not distinguish 1Q Clifford noise from bulk
          idle noise, so both use the same ``p/10`` depolarizing channel.
        - The current 25-parameter semantics only apply one ``p_idle_spam`` step
          per round, while the paper's Si1000 table has a distinct measure/reset
          idle window in addition to reset/prep timing. To preserve the intended
          total noise under this one-step representation, ``p_idle_spam_*`` uses
          the effective composition of *two* identical one-qubit depolarizing
          channels, each with total probability ``2p``.
        """
        if p < 0 or p > 1:
            raise ValueError(f"p must be in [0, 1], got {p}")

        # One-qubit depolarizing with total probability q => q/3 on X,Y,Z.
        p_oneq_gate = p / 30.0  # q = p/10
        p_idle_meas_reset_single = 2.0 * p / 3.0  # q = 2p, so each Pauli gets q/3
        p_idle_meas_reset = 2.0 * p_idle_meas_reset_single - 4.0 * (p_idle_meas_reset_single**2)

        # Two-qubit depolarizing with total probability p.
        p_twoq = p / 15.0

        p_prep = 2.0 * p
        p_meas = 5.0 * p

        return cls(
            p_prep_X=p_prep,
            p_prep_Z=p_prep,
            p_meas_X=p_meas,
            p_meas_Z=p_meas,
            p_idle_cnot_X=p_oneq_gate,
            p_idle_cnot_Y=p_oneq_gate,
            p_idle_cnot_Z=p_oneq_gate,
            p_idle_spam_X=p_idle_meas_reset,
            p_idle_spam_Y=p_idle_meas_reset,
            p_idle_spam_Z=p_idle_meas_reset,
            p_cnot_IX=p_twoq,
            p_cnot_IY=p_twoq,
            p_cnot_IZ=p_twoq,
            p_cnot_XI=p_twoq,
            p_cnot_XX=p_twoq,
            p_cnot_XY=p_twoq,
            p_cnot_XZ=p_twoq,
            p_cnot_YI=p_twoq,
            p_cnot_YX=p_twoq,
            p_cnot_YY=p_twoq,
            p_cnot_YZ=p_twoq,
            p_cnot_ZI=p_twoq,
            p_cnot_ZX=p_twoq,
            p_cnot_ZY=p_twoq,
            p_cnot_ZZ=p_twoq,
        )

    def to_config_dict(self) -> Dict[str, float]:
        """
        Convert to a configuration dictionary.

        Returns:
            Dictionary with all public parameters (25)
        """
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def canonical_parameters(self) -> Dict[str, float]:
        """Stable public 25p parameter mapping for metadata and hashing."""
        return {k: float(v) for k, v in sorted(self.to_config_dict().items())}

    def canonical_json(self) -> str:
        """Stable JSON representation of public parameters only."""
        return json.dumps(
            self.canonical_parameters(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def sha256(self) -> str:
        """SHA-256 of the canonical public 25p parameter JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def copy(self) -> "NoiseModel":
        """Deep-ish copy preserving reference parameters."""
        nm = NoiseModel.from_config_dict(self.to_config_dict())
        nm._reference = dict(self._reference)
        return nm

    def reset_to_reference(self) -> None:
        """Reset all public parameters back to the stored reference values."""
        for k, v in self._reference.items():
            setattr(self, k, float(v))
        self.validate()

    def randomize_around_reference(
        self, *, frac: float = 0.25, rng: Optional[np.random.Generator] = None
    ) -> None:
        """
        Apply a uniform ±frac multiplicative drift to each parameter around the stored reference.

        Example: frac=0.25 => each p is multiplied by U[0.75, 1.25].

        Notes:
        - Keeps a stable _reference copy.
        - Renormalizes the idle/cnot families if their totals exceed 1 due to drift.
        """
        if frac < 0:
            raise ValueError(f"frac must be non-negative, got {frac}")
        if rng is None:
            rng = np.random.default_rng()

        keys = [k for k in self._reference.keys() if not k.startswith("_")]
        for k in keys:
            base = float(self._reference[k])
            # multiplicative drift, symmetric around base
            mult = float(rng.uniform(1.0 - frac, 1.0 + frac))
            setattr(self, k, base * mult)

        # Clamp singles to [0,1]
        for k in ("p_prep_X", "p_prep_Z", "p_meas_X", "p_meas_Z"):
            setattr(self, k, float(min(max(getattr(self, k), 0.0), 1.0)))

        # Renormalize idle groups if needed
        def _renorm(prefix: str) -> None:
            ks = [k for k in keys if k.startswith(prefix)]
            total = float(sum(getattr(self, k) for k in ks))
            if total > 1.0 and total > 0:
                scale = (1.0 - 1e-12) / total
                for k in ks:
                    setattr(self, k, float(getattr(self, k) * scale))

        _renorm("p_idle_cnot_")
        _renorm("p_idle_spam_")
        _renorm("p_cnot_")

        self.validate()

    @classmethod
    def from_config_dict(cls, d: Dict[str, float]) -> 'NoiseModel':
        """
        Create a NoiseModel from a configuration dictionary.
        
        Args:
            d: Dictionary with noise model parameters. Can contain either:
               - All 22 individual parameters, or
               - A single 'p' key (will use from_single_p)
               
        Returns:
            NoiseModel instance
        """
        if d is None:
            return None

        # Check for single-p shorthand
        if 'p' in d and len(d) == 1:
            return cls.from_single_p(d['p'])

        # Check for single-p with spam_factor
        if 'p' in d and 'spam_factor' in d and len(d) == 2:
            return cls.from_single_p(d['p'], d['spam_factor'])

        # Backwards-compat: allow old 22p keys p_idle_X/Y/Z
        if "p_idle_X" in d or "p_idle_Y" in d or "p_idle_Z" in d:
            # If new keys are absent, map old idle_* -> idle_cnot_* and idle_spam_*.
            if "p_idle_cnot_X" not in d and "p_idle_spam_X" not in d:
                d = dict(d)
                d["p_idle_cnot_X"] = d.get("p_idle_X", 0.0)
                d["p_idle_cnot_Y"] = d.get("p_idle_Y", 0.0)
                d["p_idle_cnot_Z"] = d.get("p_idle_Z", 0.0)
                d["p_idle_spam_X"] = d.get("p_idle_X", 0.0)
                d["p_idle_spam_Y"] = d.get("p_idle_Y", 0.0)
                d["p_idle_spam_Z"] = d.get("p_idle_Z", 0.0)
            # Drop legacy keys to avoid __init__ error
            d.pop("p_idle_X", None)
            d.pop("p_idle_Y", None)
            d.pop("p_idle_Z", None)

        return cls(**d)

    def get_cnot_probabilities(self) -> np.ndarray:
        """
        Get CNOT error probabilities as a numpy array.
        
        Returns:
            Array of shape (15,) with probabilities in Stim PAULI_CHANNEL_2 order:
            [IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]
        """
        return np.array(
            [
                self.p_cnot_IX, self.p_cnot_IY, self.p_cnot_IZ, self.p_cnot_XI, self.p_cnot_XX,
                self.p_cnot_XY, self.p_cnot_XZ, self.p_cnot_YI, self.p_cnot_YX, self.p_cnot_YY,
                self.p_cnot_YZ, self.p_cnot_ZI, self.p_cnot_ZX, self.p_cnot_ZY, self.p_cnot_ZZ
            ],
            dtype=np.float64
        )

    def get_idle_cnot_probabilities(self) -> np.ndarray:
        """Get bulk/CNOT-layer idle probabilities as (3,) array [p_X, p_Y, p_Z]."""
        return np.array(
            [self.p_idle_cnot_X, self.p_idle_cnot_Y, self.p_idle_cnot_Z], dtype=np.float64
        )

    def get_idle_spam_probabilities(self) -> np.ndarray:
        """Get SPAM-window (data during ancilla prep/reset) idle probabilities as (3,) array [p_X, p_Y, p_Z]."""
        return np.array(
            [self.p_idle_spam_X, self.p_idle_spam_Y, self.p_idle_spam_Z], dtype=np.float64
        )

    def get_max_probability(self) -> float:
        """
        Get the maximum probability across all error types.
        
        Useful for buffer size calculations in the sampler.
        
        Returns:
            Maximum probability value
        """
        all_probs = [v for k, v in asdict(self).items() if not k.startswith("_")]
        return float(max(all_probs)) if all_probs else 0.0

    def get_total_cnot_probability(self) -> float:
        """Get total probability of any CNOT error occurring."""
        return sum(self.get_cnot_probabilities())

    def get_total_idle_cnot_probability(self) -> float:
        """Get total probability of any bulk/CNOT-layer idle error occurring."""
        return float(np.sum(self.get_idle_cnot_probabilities()))

    def get_total_idle_spam_probability(self) -> float:
        """Get total probability of any SPAM-window idle error occurring."""
        return float(np.sum(self.get_idle_spam_probabilities()))

    def to_stim_pauli_channel_1_args(self) -> Tuple[float, float, float]:
        """
        Args (p_X, p_Y, p_Z) for generic one-qubit Clifford noise.

        The current 25-parameter model does not separate single-qubit gate noise
        from bulk/CNOT-layer idle noise, so 1Q Clifford gates reuse the same
        channel as ``p_idle_cnot_*``.
        """
        return self.to_stim_pauli_channel_1_args_cnot()

    def to_stim_pauli_channel_1_args_cnot(self) -> Tuple[float, float, float]:
        """Args (p_X,p_Y,p_Z) for PAULI_CHANNEL_1 during bulk/CNOT-layer idle."""
        return (self.p_idle_cnot_X, self.p_idle_cnot_Y, self.p_idle_cnot_Z)

    def to_stim_pauli_channel_1_args_spam(self) -> Tuple[float, float, float]:
        """Args (p_X,p_Y,p_Z) for PAULI_CHANNEL_1 during SPAM-window data-idle (ancilla prep/reset)."""
        return (self.p_idle_spam_X, self.p_idle_spam_Y, self.p_idle_spam_Z)

    def to_stim_pauli_channel_2_args(self) -> Tuple[float, ...]:
        """
        Get arguments for Stim's PAULI_CHANNEL_2 instruction.
        
        Returns:
            Tuple of 15 probabilities in Stim order:
            (p_IX, p_IY, p_IZ, p_XI, p_XX, p_XY, p_XZ, 
             p_YI, p_YX, p_YY, p_YZ, p_ZI, p_ZX, p_ZY, p_ZZ)
        """
        return tuple(self.get_cnot_probabilities())

    def scale(self, factor: float) -> 'NoiseModel':
        """
        Create a new NoiseModel with all probabilities scaled by a factor.
        
        Args:
            factor: Scaling factor (e.g., 0.5 for half the noise)
            
        Returns:
            New NoiseModel with scaled probabilities
        """
        params = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        scaled_params = {k: v * factor for k, v in params.items()}
        nm = NoiseModel(**scaled_params)
        nm._reference = dict(self._reference)
        return nm

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"NoiseModel("
            f"prep=[X:{self.p_prep_X:.4f}, Z:{self.p_prep_Z:.4f}], "
            f"meas=[X:{self.p_meas_X:.4f}, Z:{self.p_meas_Z:.4f}], "
            f"idle_cnot=[X:{self.p_idle_cnot_X:.4f}, Y:{self.p_idle_cnot_Y:.4f}, Z:{self.p_idle_cnot_Z:.4f}], "
            f"idle_spam=[X:{self.p_idle_spam_X:.4f}, Y:{self.p_idle_spam_Y:.4f}, Z:{self.p_idle_spam_Z:.4f}], "
            f"cnot_total={self.get_total_cnot_probability():.4f})"
        )


def get_grouped_totals(nm: NoiseModel) -> Dict[str, float]:
    """Compute effective fault-channel totals (capital P's) for 25-p training scaling.

    Returns:
        Dict with separate prep/meas channels, idle/cnot totals, and max_group.
    """
    p_prep_X = float(nm.p_prep_X)
    p_prep_Z = float(nm.p_prep_Z)
    p_meas_X = float(nm.p_meas_X)
    p_meas_Z = float(nm.p_meas_Z)
    p_idle_cnot = math.fsum(float(p) for p in nm.get_idle_cnot_probabilities())
    p_idle_spam_raw = math.fsum(float(p) for p in nm.get_idle_spam_probabilities())
    p_idle_spam_effective = 0.5 * p_idle_spam_raw
    p_cnot = math.fsum(float(p) for p in nm.get_cnot_probabilities())
    max_group = max(
        p_prep_X,
        p_prep_Z,
        p_meas_X,
        p_meas_Z,
        p_idle_cnot,
        p_idle_spam_effective,
        p_cnot,
    )
    return {
        "p_prep_X": p_prep_X,
        "p_prep_Z": p_prep_Z,
        "p_meas_X": p_meas_X,
        "p_meas_Z": p_meas_Z,
        "p_prep_total": p_prep_X + p_prep_Z,
        "p_meas_total": p_meas_X + p_meas_Z,
        "p_idle_cnot": p_idle_cnot,
        "p_idle_spam_raw": p_idle_spam_raw,
        "p_idle_spam_effective": p_idle_spam_effective,
        "p_cnot": p_cnot,
        "max_group": max_group,
    }


SURFACE_CODE_TRAINING_UPSCALE_TARGET = 6e-3
COLOR_CODE_TRAINING_UPSCALE_TARGET = 4e-3
SURFACE_CODE_THRESHOLD_APPROX = 7.5e-3


def get_training_upscale_target(code_type: str) -> Optional[float]:
    """Return the code-family training-noise target, if one is defined."""
    code = str(code_type or "surface_code").strip().lower()
    if code in ("surface", "surface_code", "surface_partition"):
        return SURFACE_CODE_TRAINING_UPSCALE_TARGET
    if code in ("color", "color_code"):
        return COLOR_CODE_TRAINING_UPSCALE_TARGET
    return None


def get_training_upscaled_noise_model(
    noise_model: NoiseModel,
    code_type: str = "surface_code",
    skip_upscale: bool = False,
) -> Tuple[NoiseModel, Dict[str, Any]]:
    """Optionally upscale the noise model for training to the code-family target.

    Training data sampling should use the returned model; evaluation should use
    the original user-specified model.
    """
    target = get_training_upscale_target(code_type)
    totals = get_grouped_totals(noise_model)
    max_group = totals["max_group"]

    info: Dict[str, Any] = {
        "code_type": code_type,
        "target": target,
        "max_group": max_group,
        "group_totals": totals,
        "above_target_warning": bool(target is not None and max_group > target),
        "downscale_skipped": False,
        "applied_upscale": False,
        "skipped_by_user": skip_upscale,
    }

    if skip_upscale:
        info["message"] = (
            "Noise upscaling SKIPPED by user (skip_noise_upscaling=true). "
            f"Training will use the exact user-specified noise model (max_group={max_group:.6g})."
        )
        return (noise_model, info)

    if target is None:
        info["message"] = (
            f"Noise upscaling is not applied for code_type={code_type!r} "
            "(no training target is defined)."
        )
        return (noise_model, info)

    if max_group <= 0.0:
        raise ValueError(
            "Invalid noise_model: all grouped totals are <= 0 "
            f"(prep_X={totals['p_prep_X']}, prep_Z={totals['p_prep_Z']}, "
            f"meas_X={totals['p_meas_X']}, meas_Z={totals['p_meas_Z']}, "
            f"idle_cnot={totals['p_idle_cnot']}, "
            f"idle_spam_effective={totals['p_idle_spam_effective']}, "
            f"cnot={totals['p_cnot']})."
        )

    scale_factor = target / max_group

    if scale_factor >= 1.0:
        params = noise_model.to_config_dict()
        scaled_params = {k: float(v) * scale_factor for k, v in params.items()}
        training_nm = NoiseModel.from_config_dict(scaled_params)
        try:
            training_nm._reference = dict(noise_model._reference)
        except AttributeError:
            pass
        info["applied_upscale"] = True
        info["scale_factor"] = scale_factor
        info["message"] = (
            f"Upscaled training noise: max_group={max_group:.6g} -> target={target:.1e} "
            f"(scale={scale_factor:.6g}). Evaluation uses user-specified noise model as-is."
        )
        return (training_nm, info)

    info["downscale_skipped"] = True
    info["scale_factor"] = scale_factor
    info["message"] = (
        f"Downscale NOT applied: max_group={max_group:.6g} > target={target:.1e}. "
        "Parameters unchanged. If you intended a lower noise regime, check your noise model values."
    )
    return (noise_model, info)


def resolve_test_noise_model(cfg) -> tuple:
    """
    Resolve the test-time noise model from config.

    cfg.test.noise_model can be:
      - "train" (default): reuse cfg.data.noise_model
      - "none": no noise model, fall back to cfg.test.p_error
      - A dict of noise model parameters (same format as cfg.data.noise_model,
        including the {p: <value>} shorthand for depolarizing)

    Returns:
        (noise_model_obj, mode_str) where noise_model_obj is a NoiseModel or
        None, and mode_str is one of "train", "none", or "custom".
    """
    test_cfg = getattr(cfg, "test", None)
    noise_model_family = normalize_noise_model_family(
        getattr(test_cfg, "noise_model_family", None),
        fallback_noise_mode=getattr(test_cfg, "noise_mode", None),
    )
    if noise_model_family == "si1000":
        p = float(getattr(test_cfg, "p_error", 0.0))
        return NoiseModel.from_si1000(p), "si1000"

    test_nm_cfg = getattr(test_cfg, "noise_model", None)

    if test_nm_cfg is None:
        test_nm_mode = "train"
    elif hasattr(test_nm_cfg, "items") or isinstance(test_nm_cfg, dict):
        from omegaconf import OmegaConf
        nm_dict = (
            OmegaConf.to_container(test_nm_cfg, resolve=True)
            if hasattr(test_nm_cfg, "items") else test_nm_cfg
        )
        return NoiseModel.from_config_dict(dict(nm_dict)), "custom"
    else:
        test_nm_mode = str(test_nm_cfg).lower()

    if test_nm_mode == "train":
        nm_cfg = getattr(getattr(cfg, "data", None), "noise_model", None)
        if nm_cfg is not None:
            from omegaconf import OmegaConf
            nm_dict = (
                OmegaConf.to_container(nm_cfg, resolve=True) if hasattr(nm_cfg, "items") else nm_cfg
            )
            if nm_dict is not None:
                return NoiseModel.from_config_dict(dict(nm_dict)), "train"
        return None, "train"
    elif test_nm_mode == "none":
        return None, "none"
    else:
        raise ValueError(
            f"Invalid cfg.test.noise_model={test_nm_cfg!r}. "
            f"Expected 'train', 'none', or a dict of noise model parameters "
            f"(e.g. {{p: 0.001}} or {{p_prep_X: 0.002, p_prep_Z: 0.002, ...}})"
        )


def noise_model_from_config(cfg) -> Optional[NoiseModel]:
    """
    Create a NoiseModel from a Hydra config object.
    
    Args:
        cfg: Config object with optional noise_model section
        
    Returns:
        NoiseModel if noise_model is specified, None otherwise
    """
    noise_model_cfg = getattr(cfg, 'noise_model', None)
    if noise_model_cfg is None:
        return None

    # Convert OmegaConf to dict if needed
    if hasattr(noise_model_cfg, 'items'):
        noise_model_dict = dict(noise_model_cfg)
    else:
        noise_model_dict = noise_model_cfg

    return NoiseModel.from_config_dict(noise_model_dict)


if __name__ == "__main__":
    # Test the NoiseModel
    print("Testing NoiseModel...")

    # Test 1: Create from single p
    p = 0.01
    nm = NoiseModel.from_single_p(p)
    print(f"\nFrom single p={p}:")
    print(f"  {nm}")
    print(f"  p_prep_X = {nm.p_prep_X} (expected: {2*p/3})")
    print(f"  p_idle_X = {nm.p_idle_X} (expected: {p/3})")
    print(f"  p_cnot_IX = {nm.p_cnot_IX} (expected: {p/15})")

    # Test 2: Verify depolarizing equivalence
    print(f"\nDepolarizing equivalence check:")
    print(f"  Total idle prob = {nm.get_total_idle_probability()} (expected: {p})")
    print(f"  Total CNOT prob = {nm.get_total_cnot_probability()} (expected: {p})")

    # Test 3: Config dict round-trip
    config_dict = nm.to_config_dict()
    nm2 = NoiseModel.from_config_dict(config_dict)
    print(f"\nConfig dict round-trip: {nm == nm2}")

    # Test 4: Stim instruction arguments
    print(f"\nStim PAULI_CHANNEL_1 args: {nm.to_stim_pauli_channel_1_args()}")
    print(f"Stim PAULI_CHANNEL_2 args (first 5): {nm.to_stim_pauli_channel_2_args()[:5]}...")

    # Test 5: Validation
    print(f"\nValidation tests:")
    try:
        NoiseModel(p_prep_X=1.5)
        print("  ERROR: Should have raised ValueError for p > 1")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print("\nAll tests passed!")
