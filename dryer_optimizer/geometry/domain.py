"""Structured-grid geometry for the full-height Y-Z dryer side section."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dryer_optimizer.config import DryerConfig, GridConfig


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    width: float
    height: float

    @property
    def dx(self) -> float:
        return self.width / self.nx

    @property
    def dy(self) -> float:
        return self.height / self.ny

    @property
    def cell_x(self) -> np.ndarray:
        return (np.arange(self.nx, dtype=float) + 0.5) * self.dx

    @property
    def cell_y(self) -> np.ndarray:
        return (np.arange(self.ny, dtype=float) + 0.5) * self.dy

    @property
    def u_shape(self) -> tuple[int, int]:
        return self.ny, self.nx + 1

    @property
    def v_shape(self) -> tuple[int, int]:
        return self.ny + 1, self.nx

    @property
    def p_shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    def nearest_cell_row(self, y: float) -> int:
        return int(np.clip(np.rint(y / self.dy - 0.5), 0, self.ny - 1))


@dataclass(frozen=True)
class DryerGeometry:
    grid: Grid
    fixed_solid: np.ndarray
    design_mask: np.ndarray
    forbidden_mask: np.ndarray
    tray_masks: tuple[np.ndarray, ...]
    tray_cell_rows: tuple[int, ...]
    tray_col_min: int
    tray_col_max: int
    inlet_face_mask: np.ndarray
    outlet_cell_mask: np.ndarray
    fan_mask: np.ndarray
    fan_face_col: int
    fan_face_row_min: int
    fan_face_row_max: int
    inlet_probe_row: int
    flow_direction: int = 1

    @property
    def design_cell_count(self) -> int:
        return int(np.count_nonzero(self.design_mask))

    @property
    def tray_depth(self) -> float:
        return (self.tray_col_max - self.tray_col_min) * self.grid.dx

    @property
    def fan_face_rows(self) -> range:
        return range(self.fan_face_row_min, self.fan_face_row_max)

    def validate(self) -> None:
        expected = self.grid.p_shape
        for name, value in (
            ("fixed_solid", self.fixed_solid),
            ("design_mask", self.design_mask),
            ("forbidden_mask", self.forbidden_mask),
            ("fan_mask", self.fan_mask),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}, expected {expected}.")
        if self.inlet_face_mask.shape != self.grid.v_shape:
            raise ValueError("inlet_face_mask must retain the MAC v-face shape.")
        if self.outlet_cell_mask.shape != expected:
            raise ValueError("outlet_cell_mask must match the pressure grid.")
        if len(self.tray_masks) != 20:
            raise ValueError("The geometry must contain exactly 20 tray masks.")
        if not np.array_equal(self.forbidden_mask, ~self.design_mask):
            raise ValueError("forbidden_mask must be the complement of design_mask.")
        if np.any(self.design_mask & self.fixed_solid):
            raise ValueError("Fixed solid cells cannot be design cells.")
        if np.any(self.design_mask & self.fan_mask):
            raise ValueError("The fan actuator cannot be a design cell.")
        if self.design_cell_count == 0:
            raise ValueError("The geometry has no design cells.")


def build_dryer_geometry(
    dryer: DryerConfig,
    grid_config: GridConfig,
    *,
    fan_diameter: float = 0.315,
    fan_thickness: float = 0.120,
    fan_x_start: float = 0.020,
    tray_forbidden_radius_cells: int = 1,
    boundary_clearance_cells: int = 1,
) -> DryerGeometry:
    """Create the full-height Y-Z model with an internal mechanical-room fan."""
    dryer.validate()
    grid_config.validate()
    grid = Grid(grid_config.nx, grid_config.ny, dryer.domain_width, dryer.domain_height)
    if fan_diameter <= 0 or fan_thickness <= 0 or fan_x_start < 0:
        raise ValueError("Fan geometry parameters are invalid.")

    fixed_solid = np.zeros(grid.p_shape, dtype=bool)
    # Top and bottom walls
    fixed_solid[0, :] = True
    fixed_solid[-1, :] = True
    # Left and right walls
    fixed_solid[:, 0] = True
    fixed_solid[:, -1] = True

    tray_y_min = dryer.plenum_top_depth + dryer.diffuser_standoff + dryer.false_wall_thickness
    tray_y_max = tray_y_min + dryer.tray_outer_depth
    tray_col_min = max(1, int(np.rint(tray_y_min / grid.dx)))
    tray_col_max = min(grid.nx - 1, int(np.rint(tray_y_max / grid.dx)))

    tray_floor_rows = tuple(grid.nearest_cell_row(y) for y in dryer.tray_elevations)
    tray_sample_rows = tuple(min(grid.ny - 1, row + 1) for row in tray_floor_rows)
    tray_masks: list[np.ndarray] = []
    tray_forbidden = np.zeros(grid.p_shape, dtype=bool)
    for floor_row, sample_row in zip(tray_floor_rows, tray_sample_rows):
        fixed_solid[floor_row, tray_col_min:tray_col_max] = True
        sample = np.zeros(grid.p_shape, dtype=bool)
        sample[sample_row, tray_col_min:tray_col_max] = True
        tray_masks.append(sample)
        lo = max(0, floor_row - tray_forbidden_radius_cells)
        hi = min(grid.ny, sample_row + tray_forbidden_radius_cells + 1)
        tray_forbidden[lo:hi, tray_col_min:tray_col_max] = True

    # Add the floor partition separating the mechanical room from the trays
    partition_row = grid.nearest_cell_row(dryer.stack_top)
    # The partition is solid except over the supply and return plenums
    supply_col_max = int(np.rint((dryer.plenum_top_depth) / grid.dx))
    return_col_min = int(np.rint((dryer.domain_width - dryer.front_return_depth) / grid.dx))
    fixed_solid[partition_row, supply_col_max:return_col_min] = True

    fan_center_z = dryer.stack_top + dryer.mechanical_room_height / 2.0
    fan_z_min = max(dryer.stack_top + grid.dy, fan_center_z - fan_diameter / 2.0)
    fan_z_max = min(dryer.chamber_height - grid.dy, fan_center_z + fan_diameter / 2.0)
    fan_row_min = max(1, int(np.floor(fan_z_min / grid.dy)))
    fan_row_max = min(grid.ny - 1, int(np.ceil(fan_z_max / grid.dy)))
    fan_col_min = max(1, int(np.floor(fan_x_start / grid.dx)))
    fan_col_max = min(grid.nx - 1, int(np.ceil((fan_x_start + fan_thickness) / grid.dx)))
    fan_mask = np.zeros(grid.p_shape, dtype=bool)
    fan_mask[fan_row_min:fan_row_max, fan_col_min:fan_col_max] = True
    fan_face_col = int(np.clip(np.rint((fan_x_start + fan_thickness / 2.0) / grid.dx), 1, grid.nx - 1))

    boundary_clearance = np.zeros(grid.p_shape, dtype=bool)
    boundary_clearance[:boundary_clearance_cells, :] = True
    boundary_clearance[-boundary_clearance_cells:, :] = True
    boundary_clearance[:, :boundary_clearance_cells] = True
    boundary_clearance[:, -boundary_clearance_cells:] = True

    outlet_clearance = np.zeros(grid.p_shape, dtype=bool)
    outlet_clearance[:, -max(2, boundary_clearance_cells):] = True
    design_mask = ~(fixed_solid | tray_forbidden | boundary_clearance | outlet_clearance | fan_mask)
    # Keep the physical tray chamber immutable in this reduced model; design
    # cells are available in the rear plenum, mechanical room, and front return.
    design_mask[:, tray_col_min:tray_col_max] = False
    forbidden_mask = ~design_mask

    # Retained for compatibility with old consumers. The new model does not
    # prescribe a velocity inlet; the fan is an internal pressure-rise source.
    inlet_face_mask = np.zeros(grid.v_shape, dtype=bool)
    outlet_cell_mask = np.zeros(grid.p_shape, dtype=bool)
    outlet_cell_mask[:, -1] = True

    geometry = DryerGeometry(
        grid=grid,
        fixed_solid=fixed_solid,
        design_mask=design_mask,
        forbidden_mask=forbidden_mask,
        tray_masks=tuple(tray_masks),
        tray_cell_rows=tray_sample_rows,
        tray_col_min=tray_col_min,
        tray_col_max=tray_col_max,
        inlet_face_mask=inlet_face_mask,
        outlet_cell_mask=outlet_cell_mask,
        fan_mask=fan_mask,
        fan_face_col=fan_face_col,
        fan_face_row_min=fan_row_min,
        fan_face_row_max=fan_row_max,
        inlet_probe_row=fan_row_min,
    )
    geometry.validate()
    return geometry
