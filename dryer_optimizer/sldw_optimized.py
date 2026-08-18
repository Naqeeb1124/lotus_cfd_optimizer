"""Build the 3D dryer CAD from a saved 2D optimizer topology.

Run the optimizer first, then run:

    python -m dryer_optimizer.sldw_optimized --view

The script deliberately imports build123d only when CAD construction is
requested, so the numerical optimizer and its tests remain usable headlessly.
The corrected 2D topology is a Y-Z side section: optimizer X maps to CAD Y
airflow depth and optimizer Y maps to CAD Z height. Each selected topology run
is extruded across the CAD X tray width. This is a CAD bridge for a quick CFD
check, not yet a full 3D adjoint optimization.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Support both:
#   python -m dryer_optimizer.sldw_optimized
# and:
#   python dryer_optimizer/sldw_optimized.py
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from dryer_optimizer.config import AppConfig
from dryer_optimizer.visualization import plot_optimized_distribution


HERE = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY = HERE / "data" / "output" / "topology.npz"
DEFAULT_DUMP = HERE / "sldw_dump"


@dataclass(frozen=True)
class CadDimensions:
    """Millimetre dimensions mirrored from ``test_files/sldw_cad.py``."""

    tray_count: int = 20
    tray_outer_width: float = 400.0
    tray_outer_depth: float = 300.0
    tray_wall_height: float = 36.5
    tray_floor_thickness: float = 1.5
    lip_width: float = 40.0
    shelf_air_gap: float = 25.0
    bottom_clearance: float = 50.0
    mechanical_room_height: float = 350.0
    side_clearance: float = 2.0
    false_wall_thickness: float = 2.0
    diffuser_thickness: float = 2.0
    diffuser_standoff: float = 50.0
    plenum_top_depth: float = 200.0
    plenum_bottom_depth: float = 10.0
    front_return_depth: float = 150.0
    wall_thickness: float = 2.0
    slot_height: float = 5.0
    slot_elevation: float = 38.5
    inlet_diameter: float = 150.0
    outlet_diameter: float = 100.0
    pipe_length: float = 50.0
    plug_fan_dia: float = 315.0
    plug_fan_width: float = 120.0
    lid_thickness: float = 2.0

    @property
    def shelf_spacing(self) -> float:
        return self.tray_wall_height + self.shelf_air_gap

    @property
    def shelf_stack_height(self) -> float:
        return self.tray_count * self.shelf_spacing

    @property
    def chamber_height(self) -> float:
        return self.bottom_clearance + self.shelf_stack_height + self.mechanical_room_height

    @property
    def chamber_width(self) -> float:
        return self.tray_outer_width + 2.0 * self.lip_width + 2.0 * self.side_clearance

    @property
    def chamber_depth(self) -> float:
        return (
            self.plenum_top_depth
            + self.diffuser_standoff
            + self.false_wall_thickness
            + self.tray_outer_depth
            + self.front_return_depth
        )

    @property
    def z_cabinet_bottom(self) -> float:
        return -self.chamber_height / 2.0

    @property
    def z_stack_bottom(self) -> float:
        return self.z_cabinet_bottom + self.bottom_clearance

    @property
    def z_stack_top(self) -> float:
        return self.z_stack_bottom + self.shelf_stack_height

    @property
    def z_mechanical_center(self) -> float:
        return self.z_stack_top + self.mechanical_room_height / 2.0

    @property
    def y_back_outer(self) -> float:
        return -self.chamber_depth / 2.0

    @property
    def y_diffuser(self) -> float:
        return self.y_back_outer + self.plenum_top_depth

    @property
    def y_mech_center(self) -> float:
        return 0.0

    @property
    def fan_y(self) -> float:
        return self.y_back_outer + self.plenum_top_depth + (self.plug_fan_width / 2.0) + 20.0

    @property
    def z_mech_top(self) -> float:
        return self.z_cabinet_bottom + self.chamber_height

    @property
    def y_front_outer(self) -> float:
        return self.chamber_depth / 2.0

    @property
    def y_false_wall(self) -> float:
        return self.y_diffuser + self.diffuser_standoff + self.false_wall_thickness

    @property
    def y_tray_center(self) -> float:
        return self.y_false_wall + self.tray_outer_depth / 2.0

    @property
    def optimizer_domain_width(self) -> float:
        """Full CAD Y-depth represented by the corrected 2D optimizer, in mm."""
        return self.chamber_depth

    @property
    def optimizer_domain_height(self) -> float:
        """Full cabinet height represented by the full-height optimizer, in mm."""
        return self.chamber_height

    def validate(self) -> None:
        if self.tray_count != 20:
            raise ValueError("The CAD bridge requires exactly 20 trays.")
        if not 1600.0 <= self.chamber_height <= 1700.0:
            raise ValueError(f"CAD chamber height {self.chamber_height} mm is outside 1.6–1.7 m.")


def load_topology(path: str | Path) -> dict[str, np.ndarray | float]:
    """Load and validate the optimizer NPZ deliverable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Topology file not found: {path}\n"
            "Run `python -m dryer_optimizer.main` first, or pass --run-optimizer-if-missing."
        )
    with np.load(path) as source:
        required = {
            "density",
            "binary",
            "width",
            "height",
            "velocity_u",
            "velocity_v",
            "tray_averages",
            "objective",
        }
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"Topology file is missing keys: {sorted(missing)}")
        data: dict[str, np.ndarray | float] = {
            key: np.array(source[key]) if key not in {"width", "height"} else float(source[key])
            for key in source.files
        }
    density = np.asarray(data["density"])
    binary = np.asarray(data["binary"], dtype=bool)
    if density.ndim != 2 or binary.shape != density.shape:
        raise ValueError("density and binary topology must be matching 2D arrays.")
    if not np.isfinite(density).all() or np.any((density < 0) | (density > 1)):
        raise ValueError("The topology density must be finite and in [0, 1].")
    return data


def _contiguous_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open column runs of true cells."""
    columns = np.flatnonzero(row)
    if columns.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(columns) > 1) + 1
    groups = np.split(columns, breaks)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _build_baffles(binary: np.ndarray, width_mm: float, height_mm: float, dimensions: CadDimensions):
    """Extrude corrected Y-Z topology runs across the CAD X tray width.

    The optimizer's horizontal coordinate is CAD depth (Y), not cabinet width
    (X). Runs are fused for a compact and valid STEP solid.
    """
    from build123d import Box, Compound, Location

    ny, nx = binary.shape
    dy = width_mm / nx
    dz = height_mm / ny
    # Extend across cabinet width (X) so the baffle connects to both side
    # walls, while its optimizer run occupies a Y-Z rectangle.
    baffle_width = dimensions.chamber_width
    if not np.any(binary):
        return None, 0

    solids = []
    for row in range(ny):
        z_bottom = dimensions.z_cabinet_bottom + row * dz
        for start, end in _contiguous_runs(binary[row]):
            y_center = dimensions.y_back_outer + (start + end) * dy / 2.0
            z_center = z_bottom + dz / 2.0
            solids.append(
                Location((0.0, y_center, z_center))
                * Box(baffle_width, (end - start) * dy, dz)
            )
    fused = solids[0]
    for solid in solids[1:]:
        fused = fused + solid
    return fused, int(np.count_nonzero(binary))


def _build_trays(dimensions: CadDimensions):
    """Create 20 tray solids with floors and side/front/rear walls."""
    from build123d import Box, BuildPart, Locations

    trays = []
    for row in range(dimensions.tray_count):
        z_pos = dimensions.z_stack_bottom + row * dimensions.shelf_spacing
        with BuildPart() as tray:
            with Locations((0, dimensions.y_tray_center, z_pos + dimensions.tray_floor_thickness / 2.0)):
                Box(dimensions.tray_outer_width, dimensions.tray_outer_depth, dimensions.tray_floor_thickness)
            with Locations((-(dimensions.tray_outer_width - dimensions.tray_floor_thickness) / 2.0, dimensions.y_tray_center, z_pos + dimensions.tray_wall_height / 2.0)):
                Box(dimensions.tray_floor_thickness, dimensions.tray_outer_depth, dimensions.tray_wall_height)
            with Locations(((dimensions.tray_outer_width - dimensions.tray_floor_thickness) / 2.0, dimensions.y_tray_center, z_pos + dimensions.tray_wall_height / 2.0)):
                Box(dimensions.tray_floor_thickness, dimensions.tray_outer_depth, dimensions.tray_wall_height)
            with Locations((0, dimensions.y_tray_center - (dimensions.tray_outer_depth - dimensions.tray_floor_thickness) / 2.0, z_pos + dimensions.tray_wall_height / 2.0)):
                Box(dimensions.tray_outer_width, dimensions.tray_floor_thickness, dimensions.tray_wall_height)
            with Locations((0, dimensions.y_tray_center + (dimensions.tray_outer_depth - dimensions.tray_floor_thickness) / 2.0, z_pos + dimensions.tray_wall_height / 2.0)):
                Box(dimensions.tray_outer_width, dimensions.tray_floor_thickness, dimensions.tray_wall_height)
        trays.append(tray.part)
    return trays


def _build_dryer_assembly(binary: np.ndarray, width_mm: float, height_mm: float, dimensions: CadDimensions):
    """Build the simplified sealed enclosure, trays, plenum pieces, and baffles."""
    from build123d import Align, Box, BuildPart, BuildSketch, Compound, Cylinder, Location, Locations, Mode, Plane, Rectangle, Circle

    dimensions.validate()
    with BuildPart() as enclosure:
        with Locations((0, dimensions.y_mech_center, dimensions.z_cabinet_bottom + dimensions.chamber_height / 2.0)):
            Box(
                dimensions.chamber_width + 2.0 * dimensions.wall_thickness,
                dimensions.chamber_depth + 2.0 * dimensions.wall_thickness,
                dimensions.chamber_height,
            )
        with Locations((0, dimensions.y_mech_center, dimensions.z_cabinet_bottom + dimensions.chamber_height / 2.0)):
            Box(
                dimensions.chamber_width,
                dimensions.chamber_depth,
                dimensions.chamber_height - 2.0 * dimensions.wall_thickness,
                mode=Mode.SUBTRACT,
            )
        # FRESH AIR INLET
        with BuildSketch(Plane.XZ.offset(-(dimensions.y_front_outer - dimensions.wall_thickness))):
            with Locations((0, dimensions.z_mechanical_center)):
                Circle(radius=dimensions.inlet_diameter/2.0 + dimensions.wall_thickness)
        from build123d import extrude
        extrude(amount=-dimensions.pipe_length)
        with BuildSketch(Plane.XZ.offset(-dimensions.y_front_outer)):
            with Locations((0, dimensions.z_mechanical_center)):
                Circle(radius=dimensions.inlet_diameter/2.0)
        extrude(amount=dimensions.pipe_length + 20.0, both=True, mode=Mode.SUBTRACT)

        # EXHAUST BLEED PORT
        with BuildSketch(Plane.XY.offset(dimensions.z_mech_top - dimensions.wall_thickness)):
            with Locations((0, dimensions.y_front_outer - 120.0)):
                Circle(radius=dimensions.outlet_diameter/2.0 + dimensions.wall_thickness)
        extrude(amount=dimensions.pipe_length)
        with BuildSketch(Plane.XY.offset(dimensions.z_mech_top - 10.0)):
            with Locations((0, dimensions.y_front_outer - 120.0)):
                Circle(radius=dimensions.outlet_diameter/2.0)
        extrude(amount=dimensions.pipe_length + 20.0, mode=Mode.SUBTRACT)


    z_wall_bottom = dimensions.z_cabinet_bottom + dimensions.wall_thickness
    z_wall_height = dimensions.z_stack_top - z_wall_bottom
    z_wall_center = z_wall_bottom + z_wall_height / 2.0
    with BuildPart() as false_wall:
        with Locations((0, dimensions.y_false_wall, z_wall_center)):
            Box(dimensions.chamber_width, dimensions.false_wall_thickness, z_wall_height)
        for row in range(dimensions.tray_count):
            z_pos = dimensions.z_stack_bottom + row * dimensions.shelf_spacing
            slot_z = z_pos + dimensions.tray_floor_thickness + dimensions.slot_elevation + dimensions.slot_height / 2.0
            with Locations((0, dimensions.y_false_wall, slot_z)):
                Box(dimensions.tray_outer_width, dimensions.false_wall_thickness * 3.0, dimensions.slot_height, mode=Mode.SUBTRACT)

    with BuildPart() as diffuser:
        with Locations((0, dimensions.y_diffuser, z_wall_center)):
            Box(dimensions.chamber_width, dimensions.diffuser_thickness, z_wall_height)
        with Locations((0, dimensions.y_diffuser, z_wall_center)):
            Box(dimensions.chamber_width - 40.0, dimensions.diffuser_thickness * 3.0, z_wall_height - 60.0, mode=Mode.SUBTRACT)

    # A lightweight supply wedge volume, kept separate for easy CFD zone inspection.
    with BuildPart() as supply_wedge:
        with BuildSketch(Plane.XY.offset(dimensions.z_stack_top)):
            Rectangle(dimensions.chamber_width, dimensions.plenum_top_depth, align=(Align.CENTER, Align.MAX))
        with BuildSketch(Plane.XY.offset(z_wall_bottom)):
            Rectangle(dimensions.chamber_width, dimensions.plenum_bottom_depth, align=(Align.CENTER, Align.MAX))
        from build123d import loft
        loft()
        with BuildSketch(Plane.XY.offset(dimensions.z_stack_top)):
            Rectangle(dimensions.chamber_width - 2.0 * dimensions.wall_thickness, dimensions.plenum_top_depth - dimensions.wall_thickness, align=(Align.CENTER, Align.MAX))
        with BuildSketch(Plane.XY.offset(z_wall_bottom)):
            Rectangle(dimensions.chamber_width - 2.0 * dimensions.wall_thickness, dimensions.plenum_bottom_depth - dimensions.wall_thickness, align=(Align.CENTER, Align.MAX))
        loft(mode=Mode.SUBTRACT)
    supply_wedge.part.position = (0, dimensions.y_diffuser, 0)

    # 5. AERODYNAMIC TURNING COWL & VANES
    with BuildPart() as flow_deflector:
        with BuildSketch(Plane.YZ):
            Y_back_inner = dimensions.y_back_outer + dimensions.wall_thickness
            Y_front_inner = dimensions.y_diffuser
            Z_ceiling = dimensions.z_mech_top - dimensions.wall_thickness
            Z_wedge = dimensions.z_stack_top
            
            R_max = Y_front_inner - Y_back_inner 
            Cy = Y_front_inner
            Cz = Z_ceiling - R_max
            
            with Locations((Y_back_inner, Z_ceiling)):
                Rectangle(R_max, R_max, align=(Align.MIN, Align.MAX))
            from build123d import Circle
            with Locations((Cy, Cz)):
                Circle(radius=R_max, mode=Mode.SUBTRACT)
                
            vane_t = 2.0
            for r_vane in [50.0, 100.0, 150.0]:
                with Locations((Cy, Cz)):
                    Circle(radius=r_vane + vane_t/2)
                    Circle(radius=r_vane - vane_t/2, mode=Mode.SUBTRACT)
                    
                with Locations((Cy, Cz)):
                    Rectangle(R_max * 3, R_max * 3, align=(Align.MIN, Align.CENTER), mode=Mode.SUBTRACT) 
                    Rectangle(R_max * 3, R_max * 3, align=(Align.CENTER, Align.MAX), mode=Mode.SUBTRACT)
                    
                with Locations((Cy - r_vane, Z_wedge)):
                    Rectangle(vane_t, Cz - Z_wedge, align=(Align.CENTER, Align.MIN))
                    
        extrude(amount=(dimensions.chamber_width - 2.0*dimensions.wall_thickness)/2, both=True)

    # 6. Dummy 315mm Plug Fan
    with BuildPart() as dummy_fan:
        with Locations(Location((0, dimensions.fan_y, dimensions.z_mechanical_center), (90, 0, 0))):
            Cylinder(radius=dimensions.plug_fan_dia/2.0, height=dimensions.plug_fan_width)

    # 8. Internal Floor Partition 
    with BuildPart() as partition:
        with BuildSketch(Plane.XY.offset(dimensions.z_stack_top)):
            with Locations((0, dimensions.y_mech_center)):
                Rectangle(dimensions.chamber_width, dimensions.chamber_depth)
            with Locations((0, dimensions.y_back_outer + dimensions.plenum_top_depth/2.0)):
                Rectangle(dimensions.chamber_width, dimensions.plenum_top_depth, mode=Mode.SUBTRACT)
            with Locations((0, dimensions.y_front_outer - dimensions.front_return_depth/2.0)):
                Rectangle(dimensions.chamber_width, dimensions.front_return_depth, mode=Mode.SUBTRACT)
        extrude(amount=dimensions.wall_thickness)

    # 9. Boundary Lids 
    lids = []
    with BuildPart() as in_lid:
        with BuildSketch(Plane.XZ.offset(-(dimensions.y_front_outer - dimensions.wall_thickness + dimensions.pipe_length))):
            with Locations((0, dimensions.z_mechanical_center)):
                Circle(radius=dimensions.inlet_diameter/2.0 + dimensions.wall_thickness)
        extrude(amount=-dimensions.lid_thickness)
    lids.append(in_lid.part)

    with BuildPart() as out_lid:
        with BuildSketch(Plane.XY.offset(dimensions.z_mech_top - dimensions.wall_thickness + dimensions.pipe_length)):
            with Locations((0, dimensions.y_front_outer - 120.0)):
                Circle(radius=dimensions.outlet_diameter/2.0 + dimensions.wall_thickness)
        extrude(amount=dimensions.lid_thickness)
    lids.append(out_lid.part)



    baffles, binary_cell_count = _build_baffles(binary, width_mm, height_mm, dimensions)
    children = [enclosure.part, false_wall.part, diffuser.part, supply_wedge.part, flow_deflector.part, dummy_fan.part, partition.part]
    children.extend(lids)
    children.extend(_build_trays(dimensions))
    if baffles is not None:
        children.append(baffles)
    assembly = Compound(children=children)
    return assembly, baffles, binary_cell_count


def _save_distribution_visualization(data: dict[str, np.ndarray | float], dimensions: CadDimensions, path: Path) -> Path:
    density = np.asarray(data["density"], dtype=float)
    binary = np.asarray(data["binary"], dtype=bool)
    tray_averages = np.asarray(data["tray_averages"], dtype=float)
    width_m = float(data["width"])
    height_m = float(data["height"])
    velocity_u = np.asarray(data["velocity_u"], dtype=float)
    velocity_v = np.asarray(data["velocity_v"], dtype=float)
    elevations = np.asarray(
        [
            (dimensions.bottom_clearance + row * dimensions.shelf_spacing) / 1000.0
            for row in range(dimensions.tray_count)
        ],
        dtype=float,
    )
    return plot_optimized_distribution(
        density,
        binary,
        tray_averages,
        width=width_m,
        height=height_m,
        tray_elevations=elevations,
        velocity_u=velocity_u,
        velocity_v=velocity_v,
        objective=float(data["objective"]) if "objective" in data else None,
        horizontal_label="airflow depth / CAD Y [m]",
        vertical_label="height from cabinet bottom / CAD Z [m]",
        path=path,
    )


def build_and_export(
    topology_path: str | Path = DEFAULT_TOPOLOGY,
    output_dir: str | Path = DEFAULT_DUMP,
    *,
    view: bool = False,
    density_threshold: float | None = None,
) -> dict[str, object]:
    """Load a topology, build CAD, export STEP/visualization, and return metadata."""
    data = load_topology(topology_path)
    dimensions = CadDimensions()
    width_m = float(data["width"])
    height_m = float(data["height"])
    width_mm = width_m * 1000.0
    height_mm = height_m * 1000.0
    if not np.isclose(width_mm, dimensions.optimizer_domain_width, atol=1.0):
        raise ValueError(
            f"Topology width {width_mm:.2f} mm does not match corrected CAD Y-depth "
            f"{dimensions.optimizer_domain_width:.2f} mm."
        )
    if not np.isclose(height_mm, dimensions.optimizer_domain_height, atol=1.0):
        raise ValueError(
            f"Topology height {height_mm:.2f} mm does not match corrected optimizer "
            f"height {dimensions.optimizer_domain_height:.2f} mm."
        )

    binary = np.asarray(data["binary"], dtype=bool)
    if density_threshold is not None:
        if not 0.0 <= density_threshold <= 1.0:
            raise ValueError("density_threshold must be in [0, 1].")
        binary = np.asarray(data["density"], dtype=float) >= density_threshold
        data["binary"] = binary
    assembly, baffles, binary_cell_count = _build_dryer_assembly(
        binary, width_mm, height_mm, dimensions
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    from build123d import export_step

    # Export the independent baffle solid before the larger assembly. OCC can
    # consume shared topology during a compound export in build123d 0.11.1.
    assembly_path = output / "dryer_optimized_assembly.step"
    baffle_path = None
    if baffles is not None:
        baffle_path = output / "optimized_baffles.step"
        # Rebuild an independent copy because the assembly Compound may share
        # and consume OCC topology during its own export.
        export_baffles, _ = _build_baffles(binary, width_mm, height_mm, dimensions)
        export_step(export_baffles, baffle_path)
    else:
        stale_baffles = output / "optimized_baffles.step"
        if stale_baffles.exists():
            stale_baffles.unlink()
    export_step(assembly, assembly_path)

    visualization_path = _save_distribution_visualization(data, dimensions, output / "optimized_distribution.png")
    if baffles is None:
        print("Warning: binary topology contains no baffles; exporting the CAD baseline plus flow visualization.")
    metadata = {
        "topology_path": str(Path(topology_path).resolve()),
        "assembly_step": str(assembly_path.resolve()),
        "baffles_step": str(baffle_path.resolve()) if baffle_path else None,
        "distribution_plot": str(visualization_path.resolve()),
        "tray_count": dimensions.tray_count,
        "chamber_width_mm": dimensions.chamber_width,
        "chamber_height_mm": dimensions.chamber_height,
        "optimizer_domain_width_mm": dimensions.optimizer_domain_width,
        "optimizer_domain_height_mm": dimensions.optimizer_domain_height,
        "binary_baffle_cell_count": binary_cell_count,
        "density_threshold": density_threshold,
        "tray_average_velocity": np.asarray(data["tray_averages"]).tolist(),
    }
    (output / "cad_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if view:
        try:
            from ocp_vscode import Camera, show
            show(assembly, port=3939, reset_camera=Camera.RESET)
        except ImportError as error:
            print(f"Viewer unavailable; STEP export completed: {error}")
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optimized dryer CAD from topology.npz.")
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--view", action="store_true", help="Open the result in the OCP VS Code viewer if available.")
    parser.add_argument("--run-optimizer-if-missing", action="store_true", help="Run the default optimizer when topology.npz is absent.")
    parser.add_argument("--density-threshold", type=float, help="Override the saved binary mask using density >= threshold.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.topology.exists() and args.run_optimizer_if_missing:
        from dryer_optimizer.main import run_optimization
        config = AppConfig.default().with_overrides(
            optimization={"output_dir": args.topology.parent}
        )
        run_optimization(config, save_outputs=True, save_plots=True)
        # The optimizer always writes topology.npz; honor a custom output
        # directory while resolving the generated filename explicitly.
        generated_topology = args.topology.parent / "topology.npz"
        if not generated_topology.exists():
            raise FileNotFoundError(f"Optimizer did not produce {generated_topology}")
        args.topology = generated_topology
    metadata = build_and_export(
        args.topology,
        args.output_dir,
        view=args.view,
        density_threshold=args.density_threshold,
    )
    print(f"Exported optimized assembly: {metadata['assembly_step']}")
    if metadata["baffles_step"]:
        print(f"Exported optimized baffles: {metadata['baffles_step']}")
    print(f"Distribution visualization: {metadata['distribution_plot']}")
    print(f"Tray count: {metadata['tray_count']}; height: {metadata['chamber_height_mm']:.1f} mm")


if __name__ == "__main__":
    main()
