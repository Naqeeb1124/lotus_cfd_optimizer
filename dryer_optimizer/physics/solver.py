"""Sparse MAC-grid Navier–Stokes/Brinkman solver with an internal fan source.

The reduced model is a steady incompressible Y-Z side section.  The fan is a
finite-width actuator in the horizontal (airflow-depth) direction.  Its
pressure rise is obtained from the supplied 3-D performance curve using the
specified out-of-plane width::

    Q_fan = b * integral(u_fan dy)
    f_fan = DeltaP(Q_fan) / L_fan

The nonlinear convective term is solved with damped Newton iterations and
residual backtracking.
After convergence the stored sparse matrix is the analytic Jacobian of the
fully coupled discrete residual, including convection and dDeltaP/dQ.  That
matrix is what the adjoint transposes, so finite-difference verification tests
the actual fan-coupled equations rather than a frozen approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from dryer_optimizer.config import PhysicsConfig
from dryer_optimizer.geometry.domain import DryerGeometry
from dryer_optimizer.physics.boundary_conditions import BoundaryConditions, make_boundary_conditions
from dryer_optimizer.physics.brinkman import resistance_derivative, resistance_from_density
from dryer_optimizer.physics.fan import FanCurve


@dataclass
class FlowState:
    """Forward solution, nonlinear diagnostics, and final residual Jacobian."""

    density: np.ndarray
    alpha: np.ndarray
    d_alpha_d_density: np.ndarray
    solution: np.ndarray
    matrix: csr_matrix
    rhs: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    fan_flow: float = 0.0
    fan_pressure: float = 0.0
    fan_pressure_slope: float = 0.0
    fan_out_of_plane_width: float = 0.315
    left_opening_flow: float = 0.0
    right_opening_flow: float = 0.0
    nonlinear_iterations: int = 0
    nonlinear_residual: float = 0.0


class BrinkmanSolver:
    """Assemble and solve the pressure-driven fan-coupled MAC system."""

    def __init__(self, geometry: DryerGeometry, physics: PhysicsConfig):
        physics.validate()
        self.geometry = geometry
        self.physics = physics
        self.boundary_conditions = make_boundary_conditions(geometry, physics)
        self.n_u = int(np.prod(geometry.grid.u_shape))
        self.n_v = int(np.prod(geometry.grid.v_shape))
        self.n_p = int(np.prod(geometry.grid.p_shape))
        self.u_offset = 0
        self.v_offset = self.n_u
        self.p_offset = self.n_u + self.n_v
        self.mu = physics.effective_viscosity()
        self.fan_curve = FanCurve.from_pairs(physics.fan_pressure_points)
        self.alpha_max = physics.alpha_max
        self._tray_mask_cache: np.ndarray | None = None
        fan_cells = np.argwhere(geometry.fan_mask)
        if fan_cells.size == 0:
            raise ValueError("The geometry has no fan actuator cells.")
        self.fan_row_min = int(np.min(fan_cells[:, 0]))
        self.fan_row_max = int(np.max(fan_cells[:, 0])) + 1
        fan_cols = np.where(np.any(geometry.fan_mask, axis=0))[0]
        self.fan_col_min = int(fan_cols[0])
        self.fan_col_max = int(fan_cols[-1]) + 1
        # u-face columns spanning the actuator thickness.  The source is
        # distributed over these faces, while Q is measured at its mid-plane.
        self.fan_source_cols = tuple(range(max(1, self.fan_col_min + 1), min(geometry.grid.nx, self.fan_col_max + 1)))
        self.fan_source_thickness = max(len(self.fan_source_cols) * geometry.grid.dx, geometry.grid.dx)
        self.fan_face_col = int(np.clip(geometry.fan_face_col, 1, geometry.grid.nx - 1))
        self.fan_rows = tuple(range(self.fan_row_min, self.fan_row_max))

    def u_index(self, row: int, col: int) -> int:
        return self.u_offset + row * (self.geometry.grid.nx + 1) + col

    def v_index(self, row: int, col: int) -> int:
        return self.v_offset + row * self.geometry.grid.nx + col

    def p_index(self, row: int, col: int) -> int:
        return self.p_offset + row * self.geometry.grid.nx + col

    @staticmethod
    def _append(rows: list[int], cols: list[int], data: list[float], row: int, col: int, value: float) -> None:
        rows.append(int(row))
        cols.append(int(col))
        data.append(float(value))

    @staticmethod
    def _add_dict(target: dict[int, float], key: int, value: float) -> None:
        target[key] = target.get(key, 0.0) + float(value)

    def _u_alpha(self, alpha: np.ndarray, row: int, col: int) -> float:
        if col == 0:
            return float(alpha[row, 0])
        if col == self.geometry.grid.nx:
            return float(alpha[row, -1])
        return 0.5 * float(alpha[row, col - 1] + alpha[row, col])

    def _v_alpha(self, alpha: np.ndarray, row: int, col: int) -> float:
        if row == 0:
            return float(alpha[0, col])
        if row == self.geometry.grid.ny:
            return float(alpha[-1, col])
        return 0.5 * float(alpha[row - 1, col] + alpha[row, col])

    def _tray_mask(self) -> np.ndarray:
        if self._tray_mask_cache is None:
            mask = np.zeros(self.geometry.grid.p_shape, dtype=bool)
            for tray in self.geometry.tray_masks:
                mask |= tray
            self._tray_mask_cache = mask
        return self._tray_mask_cache

    def _prepare_alpha(self, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rho = np.asarray(density, dtype=float)
        grid = self.geometry.grid
        if rho.shape != grid.p_shape:
            raise ValueError(f"density has shape {rho.shape}, expected {grid.p_shape}.")
        if np.any(~np.isfinite(rho)) or np.any((rho < 0) | (rho > 1)):
            raise ValueError("density must be finite and in [0, 1].")
        alpha = resistance_from_density(
            rho, alpha_min=self.physics.alpha_min,
            alpha_max=self.alpha_max, penalty=self.physics.alpha_interpolation_q,
        )
        d_alpha = resistance_derivative(
            rho, alpha_min=self.physics.alpha_min,
            alpha_max=self.alpha_max, penalty=self.physics.alpha_interpolation_q,
        )
        # Geometry-locked solids are not design variables and must be solid
        # regardless of the zero value used in the forbidden density field.
        alpha = alpha.copy()
        alpha[self.geometry.fixed_solid] = self.alpha_max
        d_alpha = d_alpha.copy()
        d_alpha[self.geometry.fixed_solid] = 0.0
        alpha[self._tray_mask()] += self.physics.tray_resistance
        return alpha, d_alpha

    def fan_flow_and_curve(self, u: np.ndarray) -> tuple[float, float, float]:
        """Return 3-D fan flow, pressure rise, and exact pressure slope."""
        grid = self.geometry.grid
        flow = self.physics.fan_source_direction * self.physics.fan_out_of_plane_width * grid.dy * float(
            np.sum(u[list(self.fan_rows), self.fan_face_col])
        )
        pressure, slope = self.fan_curve.pressure_and_slope(flow)
        return flow, pressure, slope

    def _u_convection(self, u: np.ndarray, v: np.ndarray, row: int, col: int) -> tuple[float, dict[int, float]]:
        """Upwind u-advection and its local derivative with respect to state."""
        grid = self.geometry.grid
        dx, dy = grid.dx, grid.dy
        U = float(u[row, col])
        if U >= 0.0:
            dudx = (u[row, col] - u[row, col - 1]) / dx if col > 0 else u[row, col] / dx
            du_dx = {self.u_index(row, col): 1.0 / dx}
            if col > 0:
                du_dx[self.u_index(row, col - 1)] = -1.0 / dx
        else:
            dudx = (u[row, col + 1] - u[row, col]) / dx if col < grid.nx else -u[row, col] / dx
            du_dx = {self.u_index(row, col): -1.0 / dx}
            if col < grid.nx:
                du_dx[self.u_index(row, col + 1)] = 1.0 / dx

        vc = int(np.clip(col - 1, 0, grid.nx - 1))
        if 0 < col < grid.nx:
            v_terms = [(row, vc, 0.25), (row + 1, vc, 0.25),
                       (row, col, 0.25), (row + 1, col, 0.25)]
        else:
            v_terms = [(row, vc, 0.5), (row + 1, vc, 0.5)]
        V = sum(weight * float(v[rr, cc]) for rr, cc, weight in v_terms)
        dv = {self.v_index(rr, cc): weight for rr, cc, weight in v_terms}
        if V >= 0.0:
            dudy = (u[row, col] - u[row - 1, col]) / dy if row > 0 else u[row, col] / dy
            du_dy = {self.u_index(row, col): 1.0 / dy}
            if row > 0:
                du_dy[self.u_index(row - 1, col)] = -1.0 / dy
        else:
            dudy = (u[row + 1, col] - u[row, col]) / dy if row < grid.ny - 1 else -u[row, col] / dy
            du_dy = {self.u_index(row, col): -1.0 / dy}
            if row < grid.ny - 1:
                du_dy[self.u_index(row + 1, col)] = 1.0 / dy
        value = U * dudx + V * dudy
        derivative: dict[int, float] = {}
        for key, coefficient in du_dx.items():
            self._add_dict(derivative, key, U * coefficient)
        for key, coefficient in du_dy.items():
            self._add_dict(derivative, key, V * coefficient)
        self._add_dict(derivative, self.u_index(row, col), dudx)
        for key, coefficient in dv.items():
            self._add_dict(derivative, key, dudy * coefficient)
        return value, derivative

    def _v_convection(self, u: np.ndarray, v: np.ndarray, row: int, col: int) -> tuple[float, dict[int, float]]:
        """Upwind v-advection and its local derivative with respect to state."""
        grid = self.geometry.grid
        dx, dy = grid.dx, grid.dy
        V = float(v[row, col])
        if V >= 0.0:
            dvdy = (v[row, col] - v[row - 1, col]) / dy if row > 0 else v[row, col] / dy
            dv_dy = {self.v_index(row, col): 1.0 / dy}
            if row > 0:
                dv_dy[self.v_index(row - 1, col)] = -1.0 / dy
        else:
            dvdy = (v[row + 1, col] - v[row, col]) / dy if row < grid.ny else -v[row, col] / dy
            dv_dy = {self.v_index(row, col): -1.0 / dy}
            if row < grid.ny:
                dv_dy[self.v_index(row + 1, col)] = 1.0 / dy

        ur = int(np.clip(row - 1, 0, grid.ny - 1))
        if 0 < row < grid.ny:
            u_terms = [(ur, col, 0.25), (ur, col + 1, 0.25),
                       (row, col, 0.25), (row, col + 1, 0.25)]
        else:
            u_terms = [(ur, col, 0.5), (ur, col + 1, 0.5)]
        U = sum(weight * float(u[rr, cc]) for rr, cc, weight in u_terms)
        du = {self.u_index(rr, cc): weight for rr, cc, weight in u_terms}
        if U >= 0.0:
            dvdx = (v[row, col] - v[row, col - 1]) / dx if col > 0 else v[row, col] / dx
            dv_dx = {self.v_index(row, col): 1.0 / dx}
            if col > 0:
                dv_dx[self.v_index(row, col - 1)] = -1.0 / dx
        else:
            dvdx = (v[row, col + 1] - v[row, col]) / dx if col < grid.nx - 1 else -v[row, col] / dx
            dv_dx = {self.v_index(row, col): -1.0 / dx}
            if col < grid.nx - 1:
                dv_dx[self.v_index(row, col + 1)] = 1.0 / dx
        value = U * dvdx + V * dvdy
        derivative: dict[int, float] = {}
        for key, coefficient in dv_dx.items():
            self._add_dict(derivative, key, U * coefficient)
        for key, coefficient in dv_dy.items():
            self._add_dict(derivative, key, V * coefficient)
        self._add_dict(derivative, self.v_index(row, col), dvdy)
        for key, coefficient in du.items():
            self._add_dict(derivative, key, dvdx * coefficient)
        return value, derivative

    def assemble(
        self,
        density: np.ndarray,
        *,
        iterate: np.ndarray | None = None,
        full_jacobian: bool = False,
    ) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble an Oseen matrix or the final nonlinear residual Jacobian.

        For ``full_jacobian=False`` the advecting velocity and fan pressure are
        lagged (Picard step), making the right hand side explicit.  For the
        final Jacobian, all local convection derivatives and the dense fan
        pressure-curve coupling are included.
        """
        grid = self.geometry.grid
        alpha, d_alpha = self._prepare_alpha(density)
        x = np.zeros(self.n_u + self.n_v + self.n_p) if iterate is None else np.asarray(iterate, dtype=float)
        if x.shape != (self.n_u + self.n_v + self.n_p,):
            raise ValueError("iterate has the wrong size.")
        u = x[:self.n_u].reshape(grid.u_shape)
        v = x[self.n_u:self.n_u + self.n_v].reshape(grid.v_shape)
        fan_flow, fan_pressure, fan_slope = self.fan_flow_and_curve(u)
        fan_force = self.physics.fan_source_direction * fan_pressure / self.fan_source_thickness

        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        rhs = np.zeros(self.n_u + self.n_v + self.n_p, dtype=float)
        bc = self.boundary_conditions
        dx2 = self.mu / (grid.dx * grid.dx)
        dy2 = self.mu / (grid.dy * grid.dy)

        # Residual/Jacobian assembly for u momentum.
        for row in range(grid.ny):
            for col in range(grid.nx + 1):
                equation = self.u_index(row, col)
                if bc.u_fixed[row, col]:
                    self._append(rows, cols, values, equation, equation, 1.0)
                    rhs[equation] = bc.u_values[row, col]
                    continue
                diagonal = self._u_alpha(alpha, row, col) + (dx2 if col in (0, grid.nx) else 2.0 * dx2) + 2.0 * dy2
                if row in (0, grid.ny - 1):
                    diagonal = self._u_alpha(alpha, row, col) + (dx2 if col in (0, grid.nx) else 2.0 * dx2) + dy2
                self._append(rows, cols, values, equation, equation, diagonal)
                for cc in (col - 1, col + 1):
                    if 0 <= cc <= grid.nx:
                        self._append(rows, cols, values, equation, self.u_index(row, cc), -dx2)
                for rr in (row - 1, row + 1):
                    if 0 <= rr < grid.ny:
                        self._append(rows, cols, values, equation, self.u_index(rr, col), -dy2)
                if 0 < col < grid.nx:
                    self._append(rows, cols, values, equation, self.p_index(row, col), 1.0 / grid.dx)
                    self._append(rows, cols, values, equation, self.p_index(row, col - 1), -1.0 / grid.dx)
                elif col == 0:
                    self._append(rows, cols, values, equation, self.p_index(row, 0), 1.0 / grid.dx)
                else:
                    self._append(rows, cols, values, equation, self.p_index(row, grid.nx - 1), -1.0 / grid.dx)

                if self.physics.convection_enabled and iterate is not None:
                    _, derivative = self._u_convection(u, v, row, col)
                    if full_jacobian:
                        for key, coefficient in derivative.items():
                            self._append(rows, cols, values, equation, key, self.physics.air_density * coefficient)
                    else:
                        # Oseen: retain only transported-u coefficients; the
                        # advecting velocity is frozen during this Picard step.
                        U = float(u[row, col])
                        V = sum(weight * float(v[rr, cc]) for rr, cc, weight in (
                            [(row, int(np.clip(col - 1, 0, grid.nx - 1)), 0.25),
                             (row + 1, int(np.clip(col - 1, 0, grid.nx - 1)), 0.25),
                             (row, int(np.clip(col, 0, grid.nx - 1)), 0.25),
                             (row + 1, int(np.clip(col, 0, grid.nx - 1)), 0.25)]
                            if 0 < col < grid.nx else
                            [(row, int(np.clip(col - 1, 0, grid.nx - 1)), 0.5),
                             (row + 1, int(np.clip(col - 1, 0, grid.nx - 1)), 0.5)]
                        ))
                        if U >= 0:
                            self._append(rows, cols, values, equation, self.u_index(row, col), self.physics.air_density * U / grid.dx)
                            if col > 0:
                                self._append(rows, cols, values, equation, self.u_index(row, col - 1), -self.physics.air_density * U / grid.dx)
                        else:
                            self._append(rows, cols, values, equation, self.u_index(row, col), -self.physics.air_density * U / grid.dx)
                            if col < grid.nx:
                                self._append(rows, cols, values, equation, self.u_index(row, col + 1), self.physics.air_density * U / grid.dx)
                        if V >= 0:
                            self._append(rows, cols, values, equation, self.u_index(row, col), self.physics.air_density * V / grid.dy)
                            if row > 0:
                                self._append(rows, cols, values, equation, self.u_index(row - 1, col), -self.physics.air_density * V / grid.dy)
                        else:
                            self._append(rows, cols, values, equation, self.u_index(row, col), -self.physics.air_density * V / grid.dy)
                            if row < grid.ny - 1:
                                self._append(rows, cols, values, equation, self.u_index(row + 1, col), self.physics.air_density * V / grid.dy)

                if col in self.fan_source_cols and row in self.fan_rows:
                    rhs[equation] += fan_force
                    if full_jacobian:
                        coefficient = -fan_slope / self.fan_source_thickness
                        scale = self.physics.fan_out_of_plane_width * grid.dy
                        for fan_row in self.fan_rows:
                            self._append(rows, cols, values, equation, self.u_index(fan_row, self.fan_face_col), coefficient * scale)

        # v momentum.
        for row in range(grid.ny + 1):
            for col in range(grid.nx):
                equation = self.v_index(row, col)
                if bc.v_fixed[row, col]:
                    self._append(rows, cols, values, equation, equation, 1.0)
                    rhs[equation] = bc.v_values[row, col]
                    continue
                diagonal = self._v_alpha(alpha, row, col) + 2.0 * dx2 + (dy2 if row in (0, grid.ny) else 2.0 * dy2)
                self._append(rows, cols, values, equation, equation, diagonal)
                for cc in (col - 1, col + 1):
                    if 0 <= cc < grid.nx:
                        self._append(rows, cols, values, equation, self.v_index(row, cc), -dx2)
                for rr in (row - 1, row + 1):
                    if 0 <= rr <= grid.ny:
                        self._append(rows, cols, values, equation, self.v_index(rr, col), -dy2)
                if row > 0 and row < grid.ny:
                    self._append(rows, cols, values, equation, self.p_index(row, col), 1.0 / grid.dy)
                    self._append(rows, cols, values, equation, self.p_index(row - 1, col), -1.0 / grid.dy)
                elif row == 0:
                    self._append(rows, cols, values, equation, self.p_index(0, col), 2.0 / grid.dy)
                else:
                    self._append(rows, cols, values, equation, self.p_index(grid.ny - 1, col), -2.0 / grid.dy)
                if self.physics.convection_enabled and iterate is not None and 0 < row < grid.ny:
                    _, derivative = self._v_convection(u, v, row, col)
                    if full_jacobian:
                        for key, coefficient in derivative.items():
                            self._append(rows, cols, values, equation, key, self.physics.air_density * coefficient)
                    else:
                        # Frozen Oseen advecting U,V; use only transported v
                        # upwind gradient terms.
                        V = float(v[row, col])
                        ur = int(np.clip(row - 1, 0, grid.ny - 1))
                        U = (float(u[ur, col]) + float(u[ur, col + 1])) * 0.5
                        if 0 < row < grid.ny:
                            U = 0.5 * (float(u[ur, col]) + float(u[ur, col + 1]) + float(u[row, col]) + float(u[row, col + 1])) * 0.5
                        if U >= 0:
                            self._append(rows, cols, values, equation, self.v_index(row, col), self.physics.air_density * U / grid.dx)
                            if col > 0:
                                self._append(rows, cols, values, equation, self.v_index(row, col - 1), -self.physics.air_density * U / grid.dx)
                        else:
                            self._append(rows, cols, values, equation, self.v_index(row, col), -self.physics.air_density * U / grid.dx)
                            if col < grid.nx - 1:
                                self._append(rows, cols, values, equation, self.v_index(row, col + 1), self.physics.air_density * U / grid.dx)
                        if V >= 0:
                            self._append(rows, cols, values, equation, self.v_index(row, col), self.physics.air_density * V / grid.dy)
                            self._append(rows, cols, values, equation, self.v_index(row - 1, col), -self.physics.air_density * V / grid.dy)
                        else:
                            self._append(rows, cols, values, equation, self.v_index(row, col), -self.physics.air_density * V / grid.dy)
                            self._append(rows, cols, values, equation, self.v_index(row + 1, col), self.physics.air_density * V / grid.dy)

        # Continuity, with rightmost pressure cells enforcing static p=0.
        for row in range(grid.ny):
            for col in range(grid.nx):
                equation = self.p_index(row, col)
                if bc.p_fixed[row, col]:
                    self._append(rows, cols, values, equation, equation, 1.0)
                    rhs[equation] = bc.p_values[row, col]
                    continue
                self._append(rows, cols, values, equation, self.u_index(row, col + 1), 1.0 / grid.dx)
                self._append(rows, cols, values, equation, self.u_index(row, col), -1.0 / grid.dx)
                self._append(rows, cols, values, equation, self.v_index(row + 1, col), 1.0 / grid.dy)
                self._append(rows, cols, values, equation, self.v_index(row, col), -1.0 / grid.dy)

        matrix = csr_matrix((values, (rows, cols)), shape=(self.n_u + self.n_v + self.n_p,) * 2)
        return matrix, rhs, alpha, d_alpha

    def _residual(self, density: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        """Evaluate the nonlinear residual using the same discrete terms."""
        # Reassemble the Oseen-form residual at the current state. Because the
        # advecting velocity is evaluated from that same state, the upwind
        # product is the nonlinear residual; the final Newton Jacobian adds its
        # derivatives separately.
        A, rhs, alpha, d_alpha = self.assemble(density, iterate=x, full_jacobian=False)
        residual = A @ x - rhs
        grid = self.geometry.grid
        u = x[:self.n_u].reshape(grid.u_shape)
        v = x[self.n_u:self.n_u + self.n_v].reshape(grid.v_shape)
        fan_flow, fan_pressure, _ = self.fan_flow_and_curve(u)
        # The fan source is evaluated at the same current Q as the residual;
        # only its derivative is added in the final Newton Jacobian.
        return residual, alpha, d_alpha, fan_flow, fan_pressure, self.fan_curve.slope(fan_flow)

    def solve(self, density: np.ndarray, initial_solution: np.ndarray | None = None) -> FlowState:
        """Solve damped Newton iterations and store the full final Jacobian.

        ``initial_solution`` is useful between topology iterations: the prior
        converged flow is an excellent continuation state for a nearby density
        field and avoids restarting the fan operating-point solve from shutoff.
        """
        grid = self.geometry.grid
        alpha, d_alpha = self._prepare_alpha(density)
        size = self.n_u + self.n_v + self.n_p
        if initial_solution is None:
            x = np.zeros(size, dtype=float)
        else:
            x = np.asarray(initial_solution, dtype=float).copy()
            if x.shape != (size,) or np.any(~np.isfinite(x)):
                raise ValueError("initial_solution has the wrong shape or non-finite values.")
        # A zero initial velocity is safe for the first run; later topology
        # iterations normally use the previous converged operating point.
        converged = False
        iterations = 0
        for iterations in range(1, self.physics.nonlinear_max_iterations + 1):
            # Newton linearization of the actual residual.  In particular, the
            # fan dDeltaP/dQ low-rank coupling is included here; solving a
            # frozen-source Picard system can oscillate badly on the steep part
            # of the supplied fan curve.
            residual, _, _, _, _, _ = self._residual(density, x)
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            matrix, _, _, _ = self.assemble(density, iterate=x, full_jacobian=True)
            newton_rhs = matrix @ x - residual
            candidate = np.asarray(spsolve(matrix, newton_rhs), dtype=float)
            if np.any(~np.isfinite(candidate)):
                raise RuntimeError("Nonlinear Newton solve returned non-finite values.")
            # Backtracking on the nonlinear residual prevents an aggressive
            # Newton step from jumping across a fan-curve segment or creating
            # an upwind sign flip. Residual evaluations reuse sparse assembly
            # but avoid another linear solve.
            relaxation = self.physics.nonlinear_relaxation
            accepted = False
            updated = x.copy()
            updated_residual = residual
            updated_norm = residual_norm
            for _ in range(10):
                trial = x + relaxation * (candidate - x)
                trial_residual, _, _, _, _, _ = self._residual(density, trial)
                trial_norm = float(np.linalg.norm(trial_residual, ord=np.inf))
                if trial_norm < updated_norm:
                    updated, updated_residual, updated_norm = trial, trial_residual, trial_norm
                    accepted = True
                    break
                relaxation *= 0.5
            if not accepted:
                # Keep progress deterministic even when the piecewise fan
                # curve produces a locally flat Newton direction.
                updated = x + relaxation * (candidate - x)
                updated_residual, _, _, _, _, _ = self._residual(density, updated)
                updated_norm = float(np.linalg.norm(updated_residual, ord=np.inf))
            delta = float(np.linalg.norm(updated - x, ord=np.inf))
            x = updated
            if updated_norm < self.physics.nonlinear_tolerance and delta < self.physics.nonlinear_tolerance:
                converged = True
                break
        residual, alpha, d_alpha, fan_flow, fan_pressure, _ = self._residual(density, x)
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if not converged and residual_norm > max(self.physics.nonlinear_tolerance * 100.0, 1.0e-7):
            raise RuntimeError(
                f"Fan/Navier-Stokes solve did not converge: residual={residual_norm:.3e}, iterations={iterations}."
            )
        final_matrix, _, _, _ = self.assemble(density, iterate=x, full_jacobian=True)
        # Store rhs so matrix*x-rhs is the actual nonlinear residual at the
        # converged point. This preserves the existing residual_norm contract.
        final_rhs = final_matrix @ x - residual
        u = x[:self.n_u].reshape(grid.u_shape)
        v = x[self.n_u:self.n_u + self.n_v].reshape(grid.v_shape)
        p = x[self.n_u + self.n_v:].reshape(grid.p_shape)
        opening_scale = self.physics.fan_out_of_plane_width * grid.dy
        left_opening_flow = opening_scale * float(np.sum(u[1:-1, 0]))
        right_opening_flow = opening_scale * float(np.sum(u[1:-1, -1]))
        return FlowState(
            density=np.asarray(density, dtype=float).copy(), alpha=alpha,
            d_alpha_d_density=d_alpha, solution=x.copy(), matrix=final_matrix,
            rhs=np.asarray(final_rhs), u=u, v=v, p=p, fan_flow=fan_flow,
            fan_pressure=fan_pressure, fan_pressure_slope=self.fan_curve.slope(fan_flow),
            fan_out_of_plane_width=self.physics.fan_out_of_plane_width,
            left_opening_flow=left_opening_flow, right_opening_flow=right_opening_flow,
            nonlinear_iterations=iterations,
            nonlinear_residual=residual_norm,
        )

    def residual_norm(self, state: FlowState) -> float:
        return float(np.linalg.norm(state.matrix @ state.solution - state.rhs, ord=np.inf))
