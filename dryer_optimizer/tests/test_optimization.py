"""Optimization and manufacturing-constraint tests."""

import numpy as np

from dryer_optimizer.config import AppConfig
from dryer_optimizer.geometry import build_dryer_geometry
from dryer_optimizer.geometry.export import save_topology_npz
from dryer_optimizer.main import run_optimization
from dryer_optimizer.optimization import (
    DensityFilter,
    initial_density,
    physical_density,
    update_density,
)


def test_filter_transpose_identity() -> None:
    rng = np.random.default_rng(4)
    density = rng.random((8, 10))
    gradient = rng.random((8, 10))
    operator = DensityFilter(density.shape, radius_cells=2)
    lhs = float(np.sum(operator.apply(density) * gradient))
    rhs = float(np.sum(density * operator.transpose_apply(gradient)))
    assert np.isclose(lhs, rhs, rtol=1.0e-12, atol=1.0e-12)


def test_density_update_respects_design_mask_and_bounds() -> None:
    config = AppConfig.default().with_overrides(grid={"nx": 12, "ny": 24})
    geometry = build_dryer_geometry(config.dryer, config.grid)
    density_filter = DensityFilter(geometry.grid.p_shape, radius_cells=1)
    density = initial_density(geometry, config.optimization)
    physical, _, derivative = physical_density(
        density,
        geometry,
        density_filter,
        threshold=config.optimization.projection_threshold,
        beta=config.optimization.projection_beta,
    )
    updated = update_density(
        density,
        np.ones_like(density),
        geometry,
        density_filter,
        config.optimization,
        projection_derivative=derivative,
    )
    assert np.all((updated >= 0.0) & (updated <= 1.0))
    assert np.all(updated[geometry.forbidden_mask] == 0.0)
    assert np.max(np.abs(updated - density)) <= config.optimization.move_limit + 1.0e-12
    assert np.isfinite(physical).all()


def test_topology_npz_preserves_flow_distribution(tmp_path) -> None:
    density = np.zeros((4, 5), dtype=float)
    binary = density.astype(bool)
    velocity_u = np.ones((4, 6), dtype=float)
    velocity_v = np.ones((5, 5), dtype=float) * 2.0
    pressure = np.arange(20, dtype=float).reshape(4, 5)
    tray_averages = np.arange(20, dtype=float)
    path = save_topology_npz(
        tmp_path / "topology.npz",
        density,
        binary,
        width=0.5,
        height=1.0,
        velocity_u=velocity_u,
        velocity_v=velocity_v,
        pressure=pressure,
        tray_averages=tray_averages,
        objective=0.25,
    )
    with np.load(path) as saved:
        assert np.array_equal(saved["density"], density)
        assert np.array_equal(saved["binary"], binary)
        assert np.array_equal(saved["velocity_u"], velocity_u)
        assert np.array_equal(saved["velocity_v"], velocity_v)
        assert np.array_equal(saved["pressure"], pressure)
        assert np.array_equal(saved["tray_averages"], tray_averages)
        assert np.isclose(saved["objective"], 0.25)


def test_short_end_to_end_run() -> None:
    config = AppConfig.default().with_overrides(
        grid={"nx": 12, "ny": 24},
        optimization={"iterations": 2, "filter_radius_cells": 1},
    )
    result = run_optimization(config, save_outputs=False, show_progress=False)
    assert len(result.history.objective) == 2
    assert np.isfinite(result.density).all()
    assert result.binary_topology.shape == result.geometry.grid.p_shape
