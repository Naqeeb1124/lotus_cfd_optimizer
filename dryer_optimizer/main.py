"""Command-line entry point for the reduced dryer topology optimizer."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from dryer_optimizer.config import AppConfig, load_yaml
from dryer_optimizer.geometry import (
    build_dryer_geometry,
    save_topology_npz,
    save_topology_svg,
)
from dryer_optimizer.optimization import (
    DensityFilter,
    OptimizationHistory,
    compute_adjoint,
    evaluate_objective,
    initial_density,
    physical_density,
    update_density,
)
from dryer_optimizer.physics import BrinkmanSolver


@dataclass
class OptimizationResult:
    """Final state and diagnostics from a completed run."""

    config: AppConfig
    geometry: object
    state: object
    objective: object
    density: np.ndarray
    binary_topology: np.ndarray
    history: OptimizationHistory


def run_optimization(
    config: AppConfig | None = None,
    *,
    save_outputs: bool = True,
    save_plots: bool = True,
    show_progress: bool = True,
) -> OptimizationResult:
    """Run the forward/adjoint/update loop and optionally write deliverables."""
    app = config or AppConfig.default()
    app.validate()
    geometry = build_dryer_geometry(
        app.dryer,
        app.grid,
        fan_diameter=app.physics.fan_diameter,
        fan_thickness=app.physics.fan_thickness,
        fan_x_start=app.physics.fan_x_start,
    )
    solver = BrinkmanSolver(geometry, app.physics)
    density_filter = DensityFilter(geometry.grid.p_shape, app.optimization.filter_radius_cells)
    raw_density = initial_density(geometry, app.optimization)
    history = OptimizationHistory()
    beta = app.optimization.projection_beta
    state = None
    objective = None
    physical = None

    iterations = range(app.optimization.iterations)
    iterator = tqdm(iterations, desc="topology", disable=not show_progress)
    for iteration in iterator:
        physical, _, projection_derivative = physical_density(
            raw_density,
            geometry,
            density_filter,
            threshold=app.optimization.projection_threshold,
            beta=beta,
        )
        state = solver.solve(physical, initial_solution=state.solution if state is not None else None)
        objective = evaluate_objective(state, geometry, app.objective)
        adjoint = compute_adjoint(solver, state, geometry, objective, app.objective)

        history.objective.append(objective.value)
        history.cv.append(objective.cv)
        history.pressure_drop.append(objective.pressure_drop)
        history.fan_flow.append(objective.fan_flow)
        history.fan_pressure.append(objective.fan_pressure)
        history.target_velocity.append(objective.target_velocity)
        history.design_fraction.append(objective.design_fraction)
        history.densities.append(physical.copy())
        iterator.set_postfix(cv=f"{objective.cv:.3g}", fraction=f"{objective.design_fraction:.3g}")

        if iteration + 1 < app.optimization.iterations:
            raw_density = update_density(
                raw_density,
                adjoint.density_gradient,
                geometry,
                density_filter,
                app.optimization,
                projection_derivative=projection_derivative,
            )
            if (iteration + 1) % app.optimization.projection_ramp_every == 0:
                beta = min(app.optimization.projection_beta_max, beta * 2.0)

    assert state is not None and objective is not None and physical is not None
    binary = (physical >= app.optimization.binary_threshold) & geometry.design_mask
    result = OptimizationResult(
        config=app,
        geometry=geometry,
        state=state,
        objective=objective,
        density=physical,
        binary_topology=binary,
        history=history,
    )
    if save_outputs:
        _save_outputs(result, save_plots=save_plots)
    return result


def _save_outputs(result: OptimizationResult, *, save_plots: bool) -> None:
    output_dir = Path(result.config.optimization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_topology_npz(
        output_dir / "topology.npz",
        result.density,
        result.binary_topology,
        width=result.geometry.grid.width,
        height=result.geometry.grid.height,
        velocity_u=result.state.u,
        velocity_v=result.state.v,
        pressure=result.state.p,
        tray_averages=result.objective.tray_averages,
        objective=result.objective.value,
        fan_flow=result.objective.fan_flow,
        fan_pressure=result.objective.fan_pressure,
        target_velocity=result.objective.target_velocity,
        left_opening_flow=result.state.left_opening_flow,
        right_opening_flow=result.state.right_opening_flow,
    )
    save_topology_svg(
        output_dir / "topology.svg",
        result.binary_topology,
        width=result.geometry.grid.width,
        height=result.geometry.grid.height,
    )
    summary = {
        "objective": result.objective.value,
        "tray_cv": result.objective.cv,
        "uniformity_error": result.objective.uniformity_error,
        "target_velocity": result.objective.target_velocity,
        "fan_flow_m3_s": result.objective.fan_flow,
        "fan_pressure_pa": result.objective.fan_pressure,
        "pressure_drop": result.objective.pressure_drop,
        "pressure_penalty": result.objective.pressure_penalty,
        "volume_penalty": result.objective.volume_penalty,
        "design_fraction": result.objective.design_fraction,
        "tray_average_velocity": result.objective.tray_averages.tolist(),
        "chamber_height_m": result.config.dryer.chamber_height,
        "tray_count": result.config.dryer.row_quantity,
        "grid": [result.geometry.grid.nx, result.geometry.grid.ny],
        "fan_flow_m3_s": result.objective.fan_flow,
        "fan_pressure_pa": result.objective.fan_pressure,
        "fan_curve_points": [list(point) for point in result.config.physics.fan_pressure_points],
        "maximum_pressure_drop_pa": result.config.objective.maximum_pressure_drop,
        "target_velocity_m_s": result.objective.target_velocity,
        "left_opening_flow_m3_s": result.state.left_opening_flow,
        "right_opening_flow_m3_s": result.state.right_opening_flow,
        "fan_to_opening_flow_ratio": result.objective.fan_flow / max(abs(result.state.right_opening_flow), 1.0e-12),
        "uniformity_error": result.objective.uniformity_error,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if save_plots:
        from dryer_optimizer.visualization import plot_flow, plot_history

        plot_flow(result.state, result.geometry, result.objective, output_dir / "flow.png")
        plot_history(result.history, output_dir / "history.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a 2D tray-dryer airflow topology.")
    parser.add_argument("--config", type=Path, help="Optional YAML configuration file.")
    parser.add_argument("--iterations", type=int, help="Override optimization iteration count.")
    parser.add_argument("--nx", type=int, help="Override horizontal grid cells.")
    parser.add_argument("--ny", type=int, help="Override vertical grid cells.")
    parser.add_argument("--output-dir", type=Path, help="Override generated output directory.")
    parser.add_argument("--no-outputs", action="store_true", help="Do not write topology/diagnostic files.")
    parser.add_argument("--no-plots", action="store_true", help="Write topology files but skip Matplotlib plots.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress output.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_yaml(args.config) if args.config else AppConfig.default()
    overrides: dict[str, dict[str, object]] = {}
    if args.iterations is not None:
        overrides.setdefault("optimization", {})["iterations"] = args.iterations
    if args.output_dir is not None:
        overrides.setdefault("optimization", {})["output_dir"] = args.output_dir
    if args.nx is not None:
        overrides.setdefault("grid", {})["nx"] = args.nx
    if args.ny is not None:
        overrides.setdefault("grid", {})["ny"] = args.ny
    if overrides:
        config = config.with_overrides(**overrides)
    result = run_optimization(
        config,
        save_outputs=not args.no_outputs,
        save_plots=not args.no_plots,
        show_progress=not args.quiet,
    )
    print(f"Final objective: {result.objective.value:.6g}")
    print(f"Tray velocity CV: {result.objective.cv:.6g}")
    print(f"Baffle solid fraction: {result.objective.design_fraction:.6g}")
    print(f"Fan flow / pressure: {result.objective.fan_flow:.6g} m^3/s / {result.objective.fan_pressure:.6g} Pa")
    print(f"Opening flows L/R: {result.state.left_opening_flow:.6g} / {result.state.right_opening_flow:.6g} m^3/s")
    print(f"Target tray velocity: {result.objective.target_velocity:.6g} m/s")
    print(f"CAD-derived height: {result.config.dryer.chamber_height:.3f} m")


if __name__ == "__main__":
    main()
