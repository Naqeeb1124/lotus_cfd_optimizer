"""Configuration for the fan-coupled 2D dryer topology optimizer.

All dimensions are SI metres, kilograms, seconds, and pascals.  The corrected
Y-Z side section now includes the full 1.630 m cabinet so the fan actuator can
live in the 350 mm mechanical room above the 20-tray stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DryerConfig:
    """Dimensional layout mirrored from ``test_files/sldw_cad.py``."""

    row_quantity: int = 20
    tray_outer_width: float = 0.400
    tray_outer_depth: float = 0.300
    tray_wall_height: float = 0.0365
    tray_floor_thickness: float = 0.0015
    lip_width: float = 0.040
    shelf_air_gap: float = 0.025
    bottom_clearance: float = 0.050
    mechanical_room_height: float = 0.350
    side_clearance: float = 0.002
    plenum_top_depth: float = 0.200
    diffuser_standoff: float = 0.050
    false_wall_thickness: float = 0.002
    front_return_depth: float = 0.150
    height_min: float = 1.600
    height_max: float = 1.700

    @property
    def shelf_spacing(self) -> float:
        return self.tray_wall_height + self.shelf_air_gap

    @property
    def shelf_stack_height(self) -> float:
        return self.row_quantity * self.shelf_spacing

    @property
    def stack_top(self) -> float:
        return self.bottom_clearance + self.shelf_stack_height

    @property
    def chamber_height(self) -> float:
        return self.stack_top + self.mechanical_room_height

    @property
    def chamber_width(self) -> float:
        return self.tray_outer_width + 2.0 * self.lip_width + 2.0 * self.side_clearance

    @property
    def domain_height(self) -> float:
        """Full optimizer height, including the mechanical room."""
        return self.chamber_height

    @property
    def stack_domain_height(self) -> float:
        """Height occupied by bottom clearance plus the tray stack."""
        return self.stack_top

    @property
    def domain_width(self) -> float:
        """Y-depth from rear plenum through the front return."""
        return (
            self.plenum_top_depth
            + self.diffuser_standoff
            + self.false_wall_thickness
            + self.tray_outer_depth
            + self.front_return_depth
        )

    @property
    def tray_elevations(self) -> tuple[float, ...]:
        return tuple(self.bottom_clearance + row * self.shelf_spacing for row in range(self.row_quantity))

    def validate(self) -> None:
        if self.row_quantity != 20:
            raise ValueError("The dryer optimizer requires exactly 20 trays.")
        if not self.height_min <= self.chamber_height <= self.height_max:
            raise ValueError(
                f"CAD-derived chamber height {self.chamber_height:.4f} m is outside "
                f"the required range [{self.height_min}, {self.height_max}] m."
            )
        if min(self.tray_elevations) <= 0 or max(self.tray_elevations) >= self.chamber_height:
            raise ValueError("Tray elevations must lie inside the chamber.")


@dataclass(frozen=True)
class GridConfig:
    """Cell count for the full-height Y-Z side section."""

    nx: int = 48
    ny: int = 112

    def validate(self) -> None:
        if self.nx < 12 or self.ny < 16:
            raise ValueError("The full-height grid must have at least 12 x 16 cells.")


@dataclass(frozen=True)
class PhysicsConfig:
    """Fan-driven steady incompressible Navier–Stokes/Brinkman parameters."""

    air_density: float = 1.20
    # Effective viscosity is deliberately larger than molecular air viscosity
    # to represent unresolved turbulent mixing during laminar optimization.
    molecular_viscosity: float = 1.8e-5
    eddy_viscosity: float = 2.0e-4
    alpha_min: float = 0.0
    alpha_max: float = 1.0e5
    alpha_interpolation_q: float = 0.05
    tray_resistance: float = 0.25

    fan_diameter: float = 0.315
    fan_thickness: float = 0.120
    fan_out_of_plane_width: float = 0.315
    fan_x_start: float = 0.020
    fan_pressure_points: tuple[tuple[float, float], ...] = (
        (0.00, 1800.0),
        (0.25, 1650.0),
        (0.50, 1300.0),
        (0.75, 700.0),
        (1.00, 0.0),
    )
    fan_source_direction: float = -1.0

    nonlinear_max_iterations: int = 120
    nonlinear_tolerance: float = 1.0e-8
    nonlinear_relaxation: float = 0.30
    convection_enabled: bool = True

    def effective_viscosity(self) -> float:
        return self.molecular_viscosity + self.eddy_viscosity

    def validate(self) -> None:
        if self.air_density <= 0:
            raise ValueError("air_density must be positive.")
        if self.molecular_viscosity <= 0 or self.eddy_viscosity < 0:
            raise ValueError("Molecular viscosity must be positive and eddy viscosity non-negative.")
        if self.alpha_min < 0 or self.alpha_max <= self.alpha_min:
            raise ValueError("alpha_max must be greater than non-negative alpha_min.")
        if not 0 < self.alpha_interpolation_q <= 1:
            raise ValueError("alpha_interpolation_q must be in (0, 1].")
        if self.tray_resistance < 0:
            raise ValueError("tray_resistance must be non-negative.")
        if self.fan_diameter <= 0 or self.fan_thickness <= 0 or self.fan_out_of_plane_width <= 0:
            raise ValueError("Fan dimensions and out-of-plane width must be positive.")
        if self.fan_x_start < 0:
            raise ValueError("fan_x_start must be non-negative.")
        if self.fan_source_direction == 0:
            raise ValueError("fan_source_direction cannot be zero.")
        if self.nonlinear_max_iterations < 1 or self.nonlinear_tolerance <= 0:
            raise ValueError("Nonlinear solver settings are invalid.")
        if not 0 < self.nonlinear_relaxation <= 1:
            raise ValueError("nonlinear_relaxation must be in (0, 1].")
        flows = [point[0] for point in self.fan_pressure_points]
        pressures = [point[1] for point in self.fan_pressure_points]
        if len(flows) < 2 or any(b <= a for a, b in zip(flows, flows[1:])):
            raise ValueError("Fan flow points must be strictly increasing.")
        if any(p < 0 for p in pressures):
            raise ValueError("Fan static pressures cannot be negative.")


@dataclass(frozen=True)
class ObjectiveConfig:
    """Flow-uniformity objective and engineering constraints."""

    uniformity_weight: float = 1.0
    velocity_reference: float = 0.10
    material_cost_weight: float = 0.01
    minimum_solid_fraction: float = 0.05
    maximum_solid_fraction: float = 0.15
    volume_constraint_weight: float = 20.0
    maximum_pressure_drop: float = 300.0
    pressure_constraint_weight: float = 20.0
    pressure_drop_reference: float = 300.0

    def validate(self) -> None:
        if self.velocity_reference <= 0 or self.pressure_drop_reference <= 0:
            raise ValueError("Objective references must be positive.")
        if not 0 <= self.minimum_solid_fraction <= self.maximum_solid_fraction <= 1:
            raise ValueError("Solid-fraction bounds must satisfy 0 <= min <= max <= 1.")
        if self.maximum_pressure_drop <= 0:
            raise ValueError("maximum_pressure_drop must be positive.")
        if min(
            self.uniformity_weight,
            self.material_cost_weight,
            self.volume_constraint_weight,
            self.pressure_constraint_weight,
        ) < 0:
            raise ValueError("Objective weights cannot be negative.")


@dataclass(frozen=True)
class OptimizationConfig:
    """Density update and output settings."""

    iterations: int = 30
    initial_solid_fraction: float = 0.10
    step_size: float = 0.08
    move_limit: float = 0.04
    filter_radius_cells: int = 2
    projection_threshold: float = 0.50
    projection_beta: float = 1.0
    projection_beta_max: float = 8.0
    projection_ramp_every: int = 10
    binary_threshold: float = 0.50
    output_dir: Path = field(default_factory=lambda: Path("dryer_optimizer/data/output"))

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive.")
        if not 0 <= self.initial_solid_fraction <= 1:
            raise ValueError("initial_solid_fraction must be in [0, 1].")
        if self.step_size <= 0 or self.move_limit <= 0:
            raise ValueError("step_size and move_limit must be positive.")
        if self.filter_radius_cells < 0:
            raise ValueError("filter_radius_cells cannot be negative.")
        if not 0 < self.projection_threshold < 1:
            raise ValueError("projection_threshold must be in (0, 1).")
        if self.projection_beta <= 0 or self.projection_beta_max < self.projection_beta:
            raise ValueError("Projection beta values are invalid.")
        if self.projection_ramp_every < 1:
            raise ValueError("projection_ramp_every must be positive.")


@dataclass(frozen=True)
class AppConfig:
    dryer: DryerConfig = field(default_factory=DryerConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def validate(self) -> None:
        self.dryer.validate()
        self.grid.validate()
        self.physics.validate()
        self.objective.validate()
        self.optimization.validate()

    @classmethod
    def default(cls) -> "AppConfig":
        config = cls()
        config.validate()
        return config

    def with_overrides(self, **sections: dict[str, Any]) -> "AppConfig":
        values: dict[str, Any] = {}
        for name in ("dryer", "grid", "physics", "objective", "optimization"):
            values[name] = replace(getattr(self, name), **sections.get(name, {}))
        result = replace(self, **values)
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["optimization"]["output_dir"] = str(self.optimization.output_dir)
        return values


def load_yaml(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration YAML must contain a mapping at its root.")
    known = {"dryer", "grid", "physics", "objective", "optimization"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    if "optimization" in data and "output_dir" in data["optimization"]:
        data["optimization"]["output_dir"] = Path(data["optimization"]["output_dir"])
    return AppConfig.default().with_overrides(**data)
