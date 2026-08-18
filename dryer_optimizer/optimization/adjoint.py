"""Discrete adjoint sensitivity for the sparse Brinkman system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve

from dryer_optimizer.config import ObjectiveConfig
from dryer_optimizer.geometry.domain import DryerGeometry
from dryer_optimizer.physics.solver import BrinkmanSolver, FlowState
from dryer_optimizer.optimization.objective import (
    ObjectiveValue,
    objective_density_gradient,
    objective_state_gradient,
)


@dataclass(frozen=True)
class AdjointResult:
    """Adjoint state and total derivative with respect to physical density."""

    adjoint: np.ndarray
    state_gradient: np.ndarray
    explicit_density_gradient: np.ndarray
    density_gradient: np.ndarray


def _add_face_contribution(
    gradient: np.ndarray,
    cell_weight: float,
    face_product: float,
    row: int,
    col: int,
) -> None:
    gradient[row, col] += cell_weight * face_product


def compute_adjoint(
    solver: BrinkmanSolver,
    state: FlowState,
    geometry: DryerGeometry,
    objective: ObjectiveValue,
    config: ObjectiveConfig,
) -> AdjointResult:
    """Solve ``A.T lambda = dJ/dx`` and return ``dJ/d(rho)``.

    For ``A(rho)x=b`` the total derivative is

        dJ/drho = dJ_explicit/drho - lambda.T (dA/drho) x.

    The face resistance is the arithmetic mean of neighboring cell resistance,
    so each internal face contributes half to each adjacent design cell.

    The fan-flow sign convention for the target-velocity and pressure-cap
    state gradients is taken from the solver's ``PhysicsConfig``
    (``fan_source_direction``); the objective config does not own it.
    """
    state_gradient = objective_state_gradient(
        state,
        geometry,
        config,
        fan_source_direction=solver.physics.fan_source_direction,
    )
    adjoint = np.asarray(spsolve(state.matrix.transpose().tocsr(), state_gradient), dtype=float)
    if np.any(~np.isfinite(adjoint)):
        raise RuntimeError("Adjoint sparse solve returned non-finite values.")

    density_gradient = objective_density_gradient(state, geometry, objective, config)
    grid = geometry.grid
    nx = grid.nx
    ny = grid.ny
    u_offset = 0
    v_offset = solver.n_u

    # Only non-Dirichlet u faces contain a density-dependent resistance.
    for row in range(ny):
        for col in range(1, nx):
            contribution = -adjoint[solver.u_index(row, col)] * state.u[row, col]
            if geometry.design_mask[row, col - 1]:
                _add_face_contribution(density_gradient, 0.5 * state.d_alpha_d_density[row, col - 1], contribution, row, col - 1)
            if geometry.design_mask[row, col]:
                _add_face_contribution(density_gradient, 0.5 * state.d_alpha_d_density[row, col], contribution, row, col)

    # Bottom outlet v faces see one cell; internal v faces see two.
    for row in range(ny):
        for col in range(nx):
            if solver.boundary_conditions.v_fixed[row, col]:
                continue
            contribution = -adjoint[solver.v_index(row, col)] * state.v[row, col]
            if row == 0:
                if geometry.design_mask[0, col]:
                    _add_face_contribution(density_gradient, state.d_alpha_d_density[0, col], contribution, 0, col)
            else:
                if geometry.design_mask[row - 1, col]:
                    _add_face_contribution(density_gradient, 0.5 * state.d_alpha_d_density[row - 1, col], contribution, row - 1, col)
                if geometry.design_mask[row, col]:
                    _add_face_contribution(density_gradient, 0.5 * state.d_alpha_d_density[row, col], contribution, row, col)

    density_gradient[~geometry.design_mask] = 0.0
    return AdjointResult(
        adjoint=adjoint,
        state_gradient=state_gradient,
        explicit_density_gradient=objective_density_gradient(state, geometry, objective, config),
        density_gradient=density_gradient,
    )
