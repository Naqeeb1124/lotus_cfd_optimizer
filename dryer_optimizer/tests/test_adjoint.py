"""Mandatory finite-difference verification of the fan-coupled discrete adjoint."""

import numpy as np

from dryer_optimizer.config import AppConfig
from dryer_optimizer.geometry import build_dryer_geometry
from dryer_optimizer.optimization import (
    compute_adjoint,
    evaluate_objective,
    objective_state_gradient,
)
from dryer_optimizer.physics import BrinkmanSolver


def test_adjoint_gradient_matches_finite_difference() -> None:
    config = AppConfig.default().with_overrides(
        grid={"nx": 12, "ny": 24},
        physics={"nonlinear_max_iterations": 80, "nonlinear_relaxation": 0.4},
        objective={
            "material_cost_weight": 0.0,
            "volume_constraint_weight": 0.0,
            "pressure_constraint_weight": 0.0,
        },
    )
    geometry = build_dryer_geometry(
        config.dryer,
        config.grid,
        fan_diameter=config.physics.fan_diameter,
        fan_thickness=config.physics.fan_thickness,
        fan_x_start=config.physics.fan_x_start,
    )
    solver = BrinkmanSolver(geometry, config.physics)
    density = np.zeros(geometry.grid.p_shape)
    density[geometry.design_mask] = 0.05
    state = solver.solve(density)
    objective = evaluate_objective(state, geometry, config.objective)
    adjoint = compute_adjoint(solver, state, geometry, objective, config.objective)

    cells = np.argwhere(geometry.design_mask)
    errors = []
    epsilon = 1.0e-5
    for row, col in cells[[0, len(cells) // 2, -1]]:
        plus = density.copy()
        minus = density.copy()
        plus[row, col] += epsilon
        minus[row, col] -= epsilon
        plus_value = evaluate_objective(
            solver.solve(plus), geometry, config.objective
        ).value
        minus_value = evaluate_objective(
            solver.solve(minus), geometry, config.objective
        ).value
        finite_difference = (plus_value - minus_value) / (2.0 * epsilon)
        errors.append(abs(adjoint.density_gradient[row, col] - finite_difference))

    assert max(errors) < 2.0e-4


def test_fan_source_direction_flips_fan_face_gradient_sign() -> None:
    """Pin the fan_source_direction plumbing on both target and pressure terms.

    ``fan_source_direction`` lives on ``PhysicsConfig`` and must be forwarded by
    the adjoint into ``objective_state_gradient``.  Both the target-velocity and
    the pressure-cap terms scale the fan-face gradient by dQ/du = dir*b*dy, so
    flipping the direction must flip the sign of the fan-face entries only.
    """
    config = AppConfig.default().with_overrides(
        grid={"nx": 16, "ny": 32},
        physics={"nonlinear_max_iterations": 80, "nonlinear_relaxation": 0.4},
    )
    geometry = build_dryer_geometry(
        config.dryer,
        config.grid,
        fan_diameter=config.physics.fan_diameter,
        fan_thickness=config.physics.fan_thickness,
        fan_x_start=config.physics.fan_x_start,
    )
    solver = BrinkmanSolver(geometry, config.physics)
    density = np.zeros(geometry.grid.p_shape)
    density[geometry.design_mask] = 0.05
    state = solver.solve(density)

    gradient_minus = objective_state_gradient(
        state, geometry, config.objective, fan_source_direction=-1.0
    )
    gradient_plus = objective_state_gradient(
        state, geometry, config.objective, fan_source_direction=+1.0
    )
    nx = geometry.grid.nx
    fan_faces = np.zeros(gradient_minus.shape, dtype=bool)
    for row in geometry.fan_face_rows:
        fan_faces[row * (nx + 1) + geometry.fan_face_col] = True
    assert np.allclose(gradient_plus[fan_faces], -gradient_minus[fan_faces], atol=1.0e-14)
    assert np.allclose(gradient_plus[~fan_faces], gradient_minus[~fan_faces], atol=1.0e-14)
    # The plumbing must also be exercised by the real adjoint path: the
    # state gradient that compute_adjoint actually uses must be the one built
    # with the solver's physics-config direction.
    objective = evaluate_objective(state, geometry, config.objective)
    adjoint = compute_adjoint(solver, state, geometry, objective, config.objective)
    assert np.isfinite(adjoint.density_gradient).all()
    assert float(np.max(np.abs(adjoint.density_gradient[geometry.design_mask]))) > 0.0
    expected = objective_state_gradient(
        state, geometry, config.objective,
        fan_source_direction=solver.physics.fan_source_direction,
    )
    assert np.allclose(adjoint.state_gradient, expected, atol=1.0e-14)
