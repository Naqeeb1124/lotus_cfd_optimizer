"""Tray-specific geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dryer_optimizer.config import DryerConfig
from dryer_optimizer.geometry.domain import DryerGeometry


@dataclass(frozen=True)
class TraySummary:
    """Convenient dimensional summary for reporting and downstream CAD work."""

    count: int
    elevations: np.ndarray
    spacing: float
    chamber_height: float
    chamber_width: float


def summarize_trays(config: DryerConfig) -> TraySummary:
    config.validate()
    return TraySummary(
        count=config.row_quantity,
        elevations=np.asarray(config.tray_elevations, dtype=float),
        spacing=config.shelf_spacing,
        chamber_height=config.chamber_height,
        chamber_width=config.chamber_width,
    )


def tray_velocity_samples(geometry: DryerGeometry, u: np.ndarray) -> np.ndarray:
    """Sample horizontal velocity across each tray gap."""
    if u.shape != geometry.grid.u_shape:
        raise ValueError(f"u has shape {u.shape}, expected {geometry.grid.u_shape}.")
    return np.asarray(
        [
            np.mean(u[row, geometry.tray_col_min:geometry.tray_col_max])
            for row in geometry.tray_cell_rows
        ],
        dtype=float,
    )
