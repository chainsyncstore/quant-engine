"""Shadow-vs-live drift diagnostics for rollout gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ShadowDriftStats:
    """Drift summary between shadow and live probability streams."""

    mean_abs_error: float
    directional_agreement: float
    n_samples: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.mean_abs_error <= 1.0:
            raise ValueError("mean_abs_error must be in [0, 1]")
        if not 0.0 <= self.directional_agreement <= 1.0:
            raise ValueError("directional_agreement must be in [0, 1]")
        if self.n_samples < 0:
            raise ValueError("n_samples must be >= 0")

    def within_tolerance(
        self,
        *,
        max_mae: float = 0.10,
        min_directional_agreement: float = 0.60,
    ) -> bool:
        """Return True when shadow/live drift remains inside tolerance band."""

        return (
            self.n_samples > 0
            and self.mean_abs_error <= max_mae
            and self.directional_agreement >= min_directional_agreement
        )


def compute_shadow_live_drift(
    shadow_probabilities: Sequence[float],
    live_probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> ShadowDriftStats:
    """Compute drift metrics between shadow and live model probabilities."""

    shadow = np.asarray(shadow_probabilities, dtype=float)
    live = np.asarray(live_probabilities, dtype=float)

    if shadow.shape != live.shape:
        raise ValueError("shadow_probabilities and live_probabilities must have same shape")
    if shadow.ndim != 1:
        raise ValueError("probability inputs must be 1D sequences")
    if len(shadow) == 0:
        return ShadowDriftStats(mean_abs_error=0.0, directional_agreement=0.0, n_samples=0)

    if np.any((shadow < 0.0) | (shadow > 1.0)):
        raise ValueError("shadow probabilities must be in [0, 1]")
    if np.any((live < 0.0) | (live > 1.0)):
        raise ValueError("live probabilities must be in [0, 1]")

    mae = float(np.mean(np.abs(shadow - live)))
    shadow_dir = shadow >= threshold
    live_dir = live >= threshold
    agreement = float(np.mean(shadow_dir == live_dir))

    return ShadowDriftStats(
        mean_abs_error=mae,
        directional_agreement=agreement,
        n_samples=int(len(shadow)),
    )
