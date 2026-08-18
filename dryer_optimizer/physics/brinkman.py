"""Brinkman penalization laws."""

from __future__ import annotations

import numpy as np


def resistance_from_density(
    density: np.ndarray,
    *,
    alpha_min: float,
    alpha_max: float,
    penalty: float,
) -> np.ndarray:
    """Map solid fraction ``rho`` in [0, 1] to inverse permeability.

    The user-facing fluid fraction is ``gamma = 1 - rho``.  The interpolation
    is the requested rational Brinkman law:

        alpha = alpha_min + (alpha_max-alpha_min) * q*rho/(q + 1-rho)

    Thus rho=0 is fluid and rho=1 is strongly penalized solid.
    """
    if penalty <= 0:
        raise ValueError("penalty must be positive.")
    rho = np.clip(np.asarray(density, dtype=float), 0.0, 1.0)
    denominator = penalty + 1.0 - rho
    return alpha_min + (alpha_max - alpha_min) * penalty * rho / denominator


def resistance_derivative(
    density: np.ndarray,
    *,
    alpha_min: float,
    alpha_max: float,
    penalty: float,
) -> np.ndarray:
    """Analytic derivative of :func:`resistance_from_density`."""
    if penalty <= 0:
        raise ValueError("penalty must be positive.")
    rho = np.clip(np.asarray(density, dtype=float), 0.0, 1.0)
    denominator = penalty + 1.0 - rho
    return (alpha_max - alpha_min) * penalty * (penalty + 1.0) / (denominator * denominator)


def mesh_scaled_alpha_max(viscosity: float, dx: float, dy: float, factor: float = 1000.0) -> float:
    """Choose a leakage-resistant resistance scaled to the local mesh size."""
    if viscosity <= 0 or dx <= 0 or dy <= 0 or factor <= 0:
        raise ValueError("viscosity, mesh sizes, and factor must be positive.")
    h = min(dx, dy)
    return factor * viscosity / (h * h)
