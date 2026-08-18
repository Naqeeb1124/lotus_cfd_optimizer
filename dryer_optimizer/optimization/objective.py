"""Flow-uniformity objective and derivatives for the fan-coupled model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dryer_optimizer.config import ObjectiveConfig
from dryer_optimizer.geometry.domain import DryerGeometry
from dryer_optimizer.geometry.trays import tray_velocity_samples
from dryer_optimizer.physics.solver import FlowState


@dataclass(frozen=True)
class ObjectiveValue:
    """Objective value and engineering diagnostics for one forward solve."""

    value: float
    cv: float
    tray_averages: np.ndarray
    target_velocity: float
    uniformity_error: float
    pressure_drop: float
    pressure_penalty: float
    volume_penalty: float
    design_fraction: float
    fan_flow: float
    fan_pressure: float


def _target_velocity(state: FlowState, geometry: DryerGeometry) -> float:
    """Convert total fan flow to the requested per-tray line velocity target.

    The fan_flow sign convention is: positive Q means air flows in the correct
    physical direction (supply→trays→return).  Using the signed value preserves
    adjoint differentiability; abs() would create a gradient discontinuity at
    Q = 0 and break the Newton warm-start.
    """
    span = max(geometry.tray_depth, geometry.grid.dx)
    through_flow = state.fan_flow  # signed; positive = correct direction
    tray_count = max(len(geometry.tray_masks), 1)
    return float(through_flow / (tray_count * state.fan_out_of_plane_width * span))


def evaluate_objective(
    state: FlowState,
    geometry: DryerGeometry,
    config: ObjectiveConfig,
) -> ObjectiveValue:
    """Evaluate SSE to the fan-flow-derived target, pressure cap, and volume cost."""
    config.validate()
    tray_averages = tray_velocity_samples(geometry, state.u)
    target = _target_velocity(state, geometry)
    errors = tray_averages - target
    uniformity_error = float(np.mean(errors * errors) / (config.velocity_reference ** 2))
    mean_velocity = float(np.mean(tray_averages))
    cv = float(np.std(tray_averages) / max(abs(mean_velocity), 1.0e-12))

    pressure_drop = max(0.0, float(state.fan_pressure))
    pressure_excess = max(0.0, pressure_drop - config.maximum_pressure_drop)
    pressure_penalty = (pressure_excess / config.pressure_drop_reference) ** 2
    design_values = state.density[geometry.design_mask]
    design_fraction = float(np.mean(design_values)) if design_values.size else 0.0
    upper_excess = max(0.0, design_fraction - config.maximum_solid_fraction)
    lower_excess = max(0.0, config.minimum_solid_fraction - design_fraction)
    volume_penalty = upper_excess * upper_excess + lower_excess * lower_excess
    value = (
        config.uniformity_weight * uniformity_error
        + config.material_cost_weight * design_fraction
        + config.volume_constraint_weight * volume_penalty
        + config.pressure_constraint_weight * pressure_penalty
    )
    return ObjectiveValue(
        value=float(value), cv=cv, tray_averages=tray_averages,
        target_velocity=float(target), uniformity_error=uniformity_error,
        pressure_drop=pressure_drop, pressure_penalty=float(pressure_penalty),
        volume_penalty=float(volume_penalty), design_fraction=design_fraction,
        fan_flow=float(state.fan_flow), fan_pressure=float(state.fan_pressure),
    )


def objective_state_gradient(
    state: FlowState,
    geometry: DryerGeometry,
    config: ObjectiveConfig,
    *,
    fan_source_direction: float,
) -> np.ndarray:
    """Return exact dJ/d(state), including target and fan-pressure dependence.

    ``fan_source_direction`` is the sign factor in Q = dir * b * dy * Σ u_face.
    It is owned by ``PhysicsConfig`` and must be supplied explicitly by the
    caller (the adjoint reads it from the solver's physics config) because
    ``ObjectiveConfig`` does not own it.  Omitting it silently defaults the
    gradient to the wrong sign whenever the configured direction differs from
    -1.0, so it is intentionally a required keyword argument.
    """
    config.validate()
    gradient = np.zeros_like(state.solution, dtype=float)
    values = tray_velocity_samples(geometry, state.u)
    n_trays = values.size
    target = _target_velocity(state, geometry)
    scale = config.uniformity_weight * 2.0 / (n_trays * config.velocity_reference ** 2)
    d_value_d_average = scale * (values - target)
    nx = geometry.grid.nx
    n_cols = geometry.tray_col_max - geometry.tray_col_min
    for derivative, row in zip(d_value_d_average, geometry.tray_cell_rows):
        for col in range(geometry.tray_col_min, geometry.tray_col_max):
            gradient[row * (nx + 1) + col] += derivative / n_cols

    # The target depends on fan flow Q = dir * b * dy * Σ u_face.
    # Chain rule: dJ/du_face = dJ/dtarget * dtarget/dQ * dQ/du_face
    #           = dJ/dtarget * (1/(N*b*span)) * (dir * b * dy)
    d_value_d_target = -float(np.sum(d_value_d_average))
    tray_count = max(len(geometry.tray_masks), 1)
    target_flow_scale = 1.0 / (tray_count * state.fan_out_of_plane_width * max(geometry.tray_depth, geometry.grid.dx))
    dQ_du = fan_source_direction * state.fan_out_of_plane_width * geometry.grid.dy
    for row in geometry.fan_face_rows:
        gradient[row * (nx + 1) + geometry.fan_face_col] += d_value_d_target * target_flow_scale * dQ_du

    # Pressure-cap penalty: J_p = w * (max(0, ΔP - ΔP_max) / ΔP_ref)²
    # Chain rule: dJ_p/du_face = dJ_p/dΔP * dΔP/dQ * dQ/du_face
    excess = max(0.0, state.fan_pressure - config.maximum_pressure_drop)
    if excess > 0.0:
        d_value_d_pressure = config.pressure_constraint_weight * 2.0 * excess / (config.pressure_drop_reference ** 2)
        d_value_d_q = d_value_d_pressure * state.fan_pressure_slope
        for row in geometry.fan_face_rows:
            gradient[row * (nx + 1) + geometry.fan_face_col] += d_value_d_q * dQ_du
    return gradient


def objective_density_gradient(
    state: FlowState,
    geometry: DryerGeometry,
    objective: ObjectiveValue,
    config: ObjectiveConfig,
) -> np.ndarray:
    """Return explicit material and volume derivatives before PDE dependence."""
    config.validate()
    gradient = np.zeros_like(state.density, dtype=float)
    count = geometry.design_cell_count
    if count == 0:
        return gradient
    per_cell = config.material_cost_weight / count
    upper_excess = max(0.0, objective.design_fraction - config.maximum_solid_fraction)
    lower_excess = max(0.0, config.minimum_solid_fraction - objective.design_fraction)
    per_cell += 2.0 * config.volume_constraint_weight * (upper_excess - lower_excess) / count
    gradient[geometry.design_mask] = per_cell
    return gradient
