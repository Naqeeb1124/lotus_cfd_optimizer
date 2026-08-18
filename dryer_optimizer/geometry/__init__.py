"""Geometry and CAD-dimension utilities."""

from .domain import DryerGeometry, Grid, build_dryer_geometry
from .export import extract_contours, save_topology_npz, save_topology_svg
from .trays import TraySummary, summarize_trays, tray_velocity_samples

__all__ = [
    "DryerGeometry",
    "Grid",
    "TraySummary",
    "build_dryer_geometry",
    "extract_contours",
    "save_topology_npz",
    "save_topology_svg",
    "summarize_trays",
    "tray_velocity_samples",
]
