"""Forward-flow, fan-curve, and geometry tests."""

import numpy as np

from dryer_optimizer.config import AppConfig
from dryer_optimizer.geometry import build_dryer_geometry
from dryer_optimizer.physics import BrinkmanSolver, FanCurve, make_boundary_conditions


def small_config() -> AppConfig:
    # 16x32 is the coarsest grid on which the internal fan establishes net
    # through-tray circulation (12x24 is too coarse to resolve the fan in the
    # 350 mm mechanical room and converges to a Q=0 local recirculation).
    return AppConfig.default().with_overrides(
        grid={"nx": 16, "ny": 32},
        physics={"nonlinear_max_iterations": 80, "nonlinear_relaxation": 0.4},
    )


def test_cad_constraints_and_twenty_trays() -> None:
    config = AppConfig.default()
    assert config.dryer.row_quantity == 20
    assert 1.6 <= config.dryer.chamber_height <= 1.7
    assert np.isclose(config.dryer.chamber_height, 1.63)
    geometry = build_dryer_geometry(
        config.dryer,
        config.grid,
        fan_diameter=config.physics.fan_diameter,
        fan_thickness=config.physics.fan_thickness,
        fan_x_start=config.physics.fan_x_start,
    )
    assert len(geometry.tray_masks) == 20
    assert geometry.design_cell_count > 0
    assert geometry.grid.height == config.dryer.chamber_height
    assert np.count_nonzero(geometry.fan_mask) > 0


def test_fan_curve_interpolation_and_slope() -> None:
    curve = FanCurve.from_pairs(((0.0, 1800.0), (0.25, 1650.0), (0.50, 1300.0), (0.75, 700.0), (1.0, 0.0)))
    pressure, slope = curve.pressure_and_slope(0.375)
    assert np.isclose(pressure, 1475.0)
    assert np.isclose(slope, -1400.0)
    assert np.isclose(curve.pressure(-1.0), 1800.0)
    assert np.isclose(curve.pressure(2.0), 0.0)


def test_forward_fan_brinkman_solution_is_finite_and_converged() -> None:
    config = small_config()
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

    assert np.isfinite(state.solution).all()
    assert solver.residual_norm(state) < 1.0e-7
    assert state.nonlinear_iterations <= config.physics.nonlinear_max_iterations
    # The internal fan must establish net through-tray circulation on this grid.
    assert state.fan_flow > 0.0
    # The converged operating point must lie exactly on the fan performance curve.
    assert np.isclose(
        state.fan_pressure,
        solver.fan_curve.pressure(state.fan_flow),
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    assert 0.0 <= state.fan_pressure <= config.physics.fan_pressure_points[0][1]
    # The model is sealed: side faces are no-slip walls, so opening flows are zero.
    span = config.physics.fan_out_of_plane_width * geometry.grid.dy
    left_flow = span * np.sum(state.u[1:-1, 0])
    right_flow = span * np.sum(state.u[1:-1, -1])
    assert np.isclose(left_flow, 0.0, atol=1.0e-12)
    assert np.isclose(right_flow, 0.0, atol=1.0e-12)
    # The pressure null space is anchored in the mechanical-room fluid cell and
    # the four wall corners; all anchored cells must sit at p=0 gauge.
    bc = make_boundary_conditions(geometry, config.physics)
    anchor_rows, anchor_cols = np.nonzero(bc.p_fixed)
    assert np.allclose(state.p[anchor_rows, anchor_cols], 0.0, atol=1.0e-12)
