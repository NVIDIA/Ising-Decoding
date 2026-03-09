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
    # Create with explicit parameters
    noise_model = NoiseModel(
        p_prep_X=0.005, p_prep_Z=0.005,
        p_meas_X=0.005, p_meas_Z=0.005,
        p_idle_cnot_X=0.003, p_idle_cnot_Y=0.003, p_idle_cnot_Z=0.003,
        p_idle_spam_X=0.003, p_idle_spam_Y=0.003, p_idle_spam_Z=0.003,
        p_cnot_IX=0.001, ...
    )
    
    # From config dict (requires all 25 parameters)
    noise_model = NoiseModel.from_config_dict(cfg.noise_model)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Surface-code training upscale target (below threshold ~7.5e-3). Used when sampling training data.
SURFACE_CODE_TRAINING_UPSCALE_TARGET = 6e-3
# Approximate surface code threshold for user-facing warnings.
SURFACE_CODE_THRESHOLD_APPROX = 7.5e-3


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


# Ordered list of CNOT error types (excluding II)
# Order matches Stim's PAULI_CHANNEL_2: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
CNOT_ERROR_TYPES = [
    'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ'
]

# Mapping from error type string to index (0-14)
CNOT_ERROR_INDEX = {et: i for i, et in enumerate(CNOT_ERROR_TYPES)}


@dataclass
class NoiseModel:
    """
    25-Parameter Noise Model for circuit-level noise simulation.
    
    Attributes (public semantics):
        p_prep_X: Probability that an X-basis preparation fails (i.e., prepare |+> then apply Z to flip to |->).
        p_prep_Z: Probability that a Z-basis preparation fails (i.e., prepare |0> then apply X to flip to |1>).
        p_meas_X: Probability that an X-basis measurement fails (modeled as a pre-measurement Z flip).
        p_meas_Z: Probability that a Z-basis measurement fails (modeled as a pre-measurement X flip).
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

    def to_config_dict(self) -> Dict[str, float]:
        """
        Convert to a configuration dictionary.
        
        Returns:
            Dictionary with all public parameters (25)
        """
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

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
            d: Dictionary with noise model parameters (full 25-parameter form).
               
        Returns:
            NoiseModel instance
        """
        if d is None:
            return None

        if 'p' in d or 'spam_factor' in d:
            raise ValueError(
                "Single-p noise_model configs are not supported. "
                "Specify all 25 noise_model parameters instead."
            )

        legacy_keys = {"p_idle_X", "p_idle_Y", "p_idle_Z"}
        if any(k in d for k in legacy_keys):
            raise ValueError(
                "Legacy p_idle_X/Y/Z keys are not supported. "
                "Specify p_idle_cnot_* and p_idle_spam_* explicitly."
            )

        required_keys = {k for k in asdict(cls()).keys() if not k.startswith("_")}
        missing = required_keys - set(d.keys())
        if missing:
            raise ValueError("Missing noise_model parameters: " + ", ".join(sorted(missing)))

        return cls(**d)

    @classmethod
    def from_single_p(cls, p: float, *, spam_factor: float = 2.0 / 3.0) -> "NoiseModel":
        """
        Create a NoiseModel from a single depolarizing-equivalent error rate.

        Args:
            p: Single physical error rate in [0, 1].
            spam_factor: Multiplier for prep/meas errors (default: 2/3).

        Returns:
            NoiseModel instance with 25-parameter mapping.
        """
        mapping = _single_p_mapping(float(p), spam_factor=float(spam_factor))
        return cls(**mapping)

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
        
        Useful for simulator buffer size calculations.
        
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
    """
    Compute the sum of p's per fault channel (capital P's) for the 25-p noise model.

    Returns:
        Dict with keys: p_prep, p_meas, p_idle_cnot, p_idle_spam, p_cnot, max_group.
    """
    p_prep = float(nm.p_prep_X + nm.p_prep_Z)
    p_meas = float(nm.p_meas_X + nm.p_meas_Z)
    p_idle_cnot = float(nm.get_total_idle_cnot_probability())
    p_idle_spam = float(nm.get_total_idle_spam_probability())
    p_cnot = float(nm.get_total_cnot_probability())
    max_group = max(p_prep, p_meas, p_idle_cnot, p_idle_spam, p_cnot)
    return {
        "p_prep": p_prep,
        "p_meas": p_meas,
        "p_idle_cnot": p_idle_cnot,
        "p_idle_spam": p_idle_spam,
        "p_cnot": p_cnot,
        "max_group": max_group,
    }


def get_training_upscaled_noise_model(
    noise_model: NoiseModel,
    code_type: str = "surface_code",
    skip_upscale: bool = False,
) -> Tuple[NoiseModel, Dict[str, Any]]:
    """
    For surface code only: optionally upscale the noise model for training so that
    max(P's) = SURFACE_CODE_TRAINING_UPSCALE_TARGET (6e-3). Training data sampling
    should use the returned model; evaluation should use the original user-specified model.

    - Upscaling (max_group < target): scale all 25 p's by target/max_group; info contains details.
    - Downscaling (max_group > target): do NOT change parameters; info contains a clear warning.
    - If max_group > target: info indicates the user may be above threshold / have made an error.

    For code_type != "surface_code", returns (noise_model unchanged, info with applied=False).

    Args:
        noise_model: The user-specified NoiseModel.
        code_type: Code type string (upscaling only for "surface_code").
        skip_upscale: If True, skip upscaling entirely and return the original model unchanged.
            Useful for training with exact user-specified noise parameters (e.g. benchmarking).

    Returns:
        (training_noise_model, info_dict) where info_dict has:
        - applied_upscale: bool
        - scale_factor: float (only if upscaling applied)
        - max_group: float
        - group_totals: dict (p_prep, p_meas, ...)
        - above_target_warning: bool (max_group > UPSCALE_TARGET)
        - downscale_skipped: bool (max_group > target, params not modified)
        - skipped_by_user: bool (skip_upscale was True)
    """
    target = SURFACE_CODE_TRAINING_UPSCALE_TARGET
    totals = get_grouped_totals(noise_model)
    max_group = totals["max_group"]

    info: Dict[str, Any] = {
        "max_group": max_group,
        "group_totals": totals,
        "above_target_warning": max_group > target,
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

    if code_type != "surface_code":
        info["message"
            ] = f"Noise upscaling is not applied for code_type={code_type!r} (surface_code only)."
        return (noise_model, info)

    if max_group <= 0.0:
        raise ValueError(
            "Invalid noise_model: all grouped totals are <= 0 "
            f"(prep={totals['p_prep']}, meas={totals['p_meas']}, "
            f"idle_cnot={totals['p_idle_cnot']}, idle_spam={totals['p_idle_spam']}, cnot={totals['p_cnot']})."
        )

    scale_factor = target / max_group

    if scale_factor >= 1.0:
        # Upscaling: apply scale to all 25 parameters
        params = noise_model.to_config_dict()
        scaled_params = {k: float(v) * scale_factor for k, v in params.items()}
        training_nm = NoiseModel.from_config_dict(scaled_params)
        training_nm._reference = dict(noise_model._reference)
        info["applied_upscale"] = True
        info["scale_factor"] = scale_factor
        info["message"] = (
            f"Upscaled training noise: max_group={max_group:.6g} -> target={target:.1e} "
            f"(scale={scale_factor:.6g}). Evaluation uses user-specified noise model as-is."
        )
        return (training_nm, info)

    # Downscaling: do not modify parameters
    info["downscale_skipped"] = True
    info["scale_factor"] = scale_factor
    info["message"] = (
        f"Downscale NOT applied: max_group={max_group:.6g} > target={target:.1e}. "
        "Parameters unchanged. If you intended a lower noise regime, check your noise model values."
    )
    return (noise_model, info)


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

    # Test 1: Create from explicit 25-parameter config
    p = 0.01
    mapping = _single_p_mapping(p)
    nm = NoiseModel(**mapping)
    print(f"\nFrom explicit p={p} mapping:")
    print(f"  {nm}")
    print(f"  p_prep_X = {nm.p_prep_X} (expected: {mapping['p_prep_X']})")
    print(f"  p_idle_cnot_X = {nm.p_idle_cnot_X} (expected: {mapping['p_idle_cnot_X']})")
    print(f"  p_cnot_IX = {nm.p_cnot_IX} (expected: {mapping['p_cnot_IX']})")

    # Test 2: Verify depolarizing equivalence (CNOT-layer idle + CNOT total)
    print(f"\nDepolarizing equivalence check:")
    print(f"  Total idle CNOT-layer prob = {nm.get_total_idle_cnot_probability()} (expected: {p})")
    print(f"  Total CNOT prob = {nm.get_total_cnot_probability()} (expected: {p})")

    # Test 3: Config dict round-trip
    config_dict = nm.to_config_dict()
    nm2 = NoiseModel.from_config_dict(config_dict)
    print(f"\nConfig dict round-trip: {nm == nm2}")

    # Test 4: Stim instruction arguments
    print(f"\nStim PAULI_CHANNEL_1 (CNOT-layer) args: {nm.to_stim_pauli_channel_1_args_cnot()}")
    print(f"Stim PAULI_CHANNEL_1 (SPAM-window) args: {nm.to_stim_pauli_channel_1_args_spam()}")
    print(f"Stim PAULI_CHANNEL_2 args (first 5): {nm.to_stim_pauli_channel_2_args()[:5]}...")

    # Test 5: Validation
    print(f"\nValidation tests:")
    try:
        NoiseModel(p_prep_X=1.5)
        print("  ERROR: Should have raised ValueError for p > 1")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print("\nAll tests passed!")
