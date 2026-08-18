"""Boundary conditions for the internal-fan Y-Z dryer model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dryer_optimizer.config import PhysicsConfig
from dryer_optimizer.geometry.domain import DryerGeometry


@dataclass(frozen=True)
class BoundaryConditions:
    """Dirichlet rows for MAC faces and static-pressure outlet gauge cells."""

    u_fixed: np.ndarray
    u_values: np.ndarray
    v_fixed: np.ndarray
    v_values: np.ndarray
    p_fixed: np.ndarray
    p_values: np.ndarray

    def validate(self, geometry: DryerGeometry) -> None:
        if self.u_fixed.shape != geometry.grid.u_shape or self.u_values.shape != geometry.grid.u_shape:
            raise ValueError("u boundary arrays have the wrong shape.")
        if self.v_fixed.shape != geometry.grid.v_shape or self.v_values.shape != geometry.grid.v_shape:
            raise ValueError("v boundary arrays have the wrong shape.")
        if self.p_fixed.shape != geometry.grid.p_shape or self.p_values.shape != geometry.grid.p_shape:
            raise ValueError("p boundary arrays have the wrong shape.")
        if np.count_nonzero(self.p_fixed) < 1:
            raise ValueError("At least one static-pressure outlet cell is required.")


def make_boundary_conditions(
    geometry: DryerGeometry,
    physics: PhysicsConfig,
) -> BoundaryConditions:
    """Use pressure-driven left/right openings and no-slip top/bottom/side walls.

    The fan is internal, so no velocity is prescribed at an inlet.  The right
    return opening is held at gauge pressure.  The left opening is natural in
    the reduced section; the outlet gauge fixes the pressure null space.
    """
    del physics  # retained in the public factory signature for API stability
    grid = geometry.grid
    u_fixed = np.zeros(grid.u_shape, dtype=bool)
    u_values = np.zeros(grid.u_shape, dtype=float)
    u_fixed[0, :] = True
    u_fixed[-1, :] = True
    u_fixed[:, 0] = True
    u_fixed[:, -1] = True

    v_fixed = np.zeros(grid.v_shape, dtype=bool)
    v_values = np.zeros(grid.v_shape, dtype=float)
    # v is normal to top/bottom walls and tangential at the side boundaries.
    v_fixed[0, :] = True
    v_fixed[-1, :] = True
    v_fixed[:, 0] = True
    v_fixed[:, -1] = True

    p_fixed = np.zeros(grid.p_shape, dtype=bool)
    p_values = np.zeros(grid.p_shape, dtype=float)
    # The four corner cells have all 4 of their MAC faces fixed by the
    # intersecting no-slip walls (top/bottom tangential u, left/right tangential v).
    # We must fix them to prevent structural singularity.
    p_fixed[0, 0] = True
    p_fixed[0, -1] = True
    p_fixed[-1, 0] = True
    p_fixed[-1, -1] = True
    # Anchor the global pressure null space in a cell that is guaranteed
    # to be fluid — the center of the mechanical room above the stack.
    mech_row = min(grid.ny - 2, grid.ny - grid.ny // 8)
    p_fixed[mech_row, grid.nx // 2] = True

    conditions = BoundaryConditions(
        u_fixed=u_fixed,
        u_values=u_values,
        v_fixed=v_fixed,
        v_values=v_values,
        p_fixed=p_fixed,
        p_values=p_values,
    )
    conditions.validate(geometry)
    return conditions
