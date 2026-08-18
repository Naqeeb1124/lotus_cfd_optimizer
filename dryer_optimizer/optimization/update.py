"""Gradient-based density update for the V1 optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dryer_optimizer.config import OptimizationConfig
from dryer_optimizer.geometry.domain import DryerGeometry
from dryer_optimizer.optimization.filters import DensityFilter, smooth_projection


@dataclass
class OptimizationHistory:
    """Scalar and field diagnostics collected through an optimization run."""

    objective: list[float] = field(default_factory=list)
    cv: list[float] = field(default_factory=list)
    pressure_drop: list[float] = field(default_factory=list)
    fan_flow: list[float] = field(default_factory=list)
    fan_pressure: list[float] = field(default_factory=list)
    target_velocity: list[float] = field(default_factory=list)
    design_fraction: list[float] = field(default_factory=list)
    densities: list[np.ndarray] = field(default_factory=list)


def physical_density(
    raw_density: np.ndarray,
    geometry: DryerGeometry,
    density_filter: DensityFilter,
    *,
    threshold: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical density, filtered density, and projection derivative."""
    filtered = density_filter.apply(raw_density)
    projected, projection_derivative = smooth_projection(
        filtered,
        threshold=threshold,
        beta=beta,
    )
    projected = projected.copy()
    projected[geometry.forbidden_mask] = 0.0
    projection_derivative = projection_derivative.copy()
    projection_derivative[geometry.forbidden_mask] = 0.0
    return projected, filtered, projection_derivative


def update_density(
    raw_density: np.ndarray,
    physical_gradient: np.ndarray,
    geometry: DryerGeometry,
    density_filter: DensityFilter,
    config: OptimizationConfig,
    *,
    projection_derivative: np.ndarray | None = None,
) -> np.ndarray:
    """Back-propagate the physical gradient and apply a bounded move-limited step."""
    raw = np.asarray(raw_density, dtype=float)
    gradient = np.asarray(physical_gradient, dtype=float)
    if raw.shape != geometry.grid.p_shape or gradient.shape != raw.shape:
        raise ValueError("raw_density and physical_gradient must match the pressure grid.")
    if projection_derivative is not None:
        gradient = gradient * projection_derivative
    raw_gradient = density_filter.transpose_apply(gradient)
    raw_gradient[~geometry.design_mask] = 0.0
    max_gradient = float(np.max(np.abs(raw_gradient[geometry.design_mask])))
    if max_gradient <= 1.0e-14:
        return raw.copy()
    normalized = raw_gradient / max_gradient
    updated = raw - config.step_size * normalized
    updated = np.clip(updated, raw - config.move_limit, raw + config.move_limit)
    updated = np.clip(updated, 0.0, 1.0)
    updated[~geometry.design_mask] = 0.0
    return updated


def initial_density(geometry: DryerGeometry, config: OptimizationConfig) -> np.ndarray:
    density = np.zeros(geometry.grid.p_shape, dtype=float)
    density[geometry.design_mask] = config.initial_solid_fraction
    return density
