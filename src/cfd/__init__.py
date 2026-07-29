"""Modular CFD pipeline for the autonomous design framework.

Functions:
- generate_geometry: Parametric CAD generation (Build123d).
- run_meshing: PyFluent meshing workflow (Watertight Geometry, modern API).
- run_solver: CFD solver execution (k-omega SST, Pythonic API).
- objective_function: Wrapper for optimization (returns CoV).
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import pathlib
import numpy as np
from build123d import (
    BuildPart, BuildSketch, Rectangle, Locations, Location, Mode, Align,
    Compound, Cylinder, Box, Plane, extrude, export_step, Axis, Solid
)
from src.utils.db import insert_design, insert_simulation, insert_critique
from src.utils.logger import info, warning, error

# Constants
EXPORT_DIR = pathlib.Path("E:/Projects/lotus_power/CAD_files/build123d_dump")
MESH_DIR = pathlib.Path("E:/Projects/lotus_power/meshes")
MESH_DIR.mkdir(exist_ok=True, parents=True)
EXPORT_DIR.mkdir(exist_ok=True, parents=True)

def generate_geometry(lip_w: float, shelf_spacing: float) -> Tuple[pathlib.Path, Optional[pathlib.Path], List[float]]:
    """Generate CAD geometry and return (cad_path, boi_path, tray_z_positions).
    
    All bodies (main geometry + BOI) are combined into a single STEP file.
    BOI bodies are labeled "BOI" for Fluent to pick up during meshing.
    
    Args:
        lip_w: Lip width in mm.
        shelf_spacing: Spacing between shelves in mm.
        
    Returns:
        Tuple of (combined CAD path, None, tray Z positions).
        
    Raises:
        RuntimeError: If CAD generation or export fails.
    """
    info("CAD", "Generating geometry", lip_w=lip_w, shelf_spacing=shelf_spacing)

    try:
        # Parameters (adapted from cfdready_dryer.py)
        shelf_quantity = 1
        tray_outer_w = 400.0
        tray_outer_d = 300.0
        tray_wall_h = 36.5
        t = 1.5
        inner_w = tray_outer_w - (2 * t)
        inner_d = tray_outer_d - (2 * t)
        chamber_width = tray_outer_w + (2 * lip_w)
        chamber_depth = tray_outer_d + 20.0
        chamber_height = 250.0 + ((shelf_quantity - 1) * shelf_spacing) + 380.0
        boi_h = 50.0
        boi_w = inner_w - 4.0
        boi_d = inner_d - 4.0
        start_z = -chamber_height / 2 + 250.0

        # Build tray walls
        with BuildPart() as base_tray_walls:
            with BuildSketch():
                Rectangle(tray_outer_w, tray_outer_d)
                Rectangle(inner_w, inner_d, mode=Mode.SUBTRACT)
            extrude(amount=-tray_wall_h)
            with BuildSketch(Plane.XY.offset(-tray_wall_h)):
                with Locations((-tray_outer_w / 2 - lip_w / 2, 0), (tray_outer_w / 2 + lip_w / 2, 0)):
                    Rectangle(lip_w, tray_outer_d)
            extrude(amount=t)

        # Build fan and heater
        with BuildPart() as fan_solid:
            fan_z = chamber_height / 2 - 60.0 / 2 - 40
            with Locations((0, 0, fan_z)):
                Box(300.0, 260.0, 60.0)

        with BuildPart() as heater_solid:
            heater_z = fan_z - 60.0 / 2 - 60.0 / 2 - 20
            with Locations((0, 0, heater_z)):
                Box(400.0, 280.0, 60.0)

        # Instantiate trays and BOIs
        instantiated_solid_trays = []
        instantiated_bois = []
        tray_z_positions = []

        for i in range(shelf_quantity):
            z_pos = start_z + (i * shelf_spacing)
            tray_z_positions.append(z_pos)
            loc = Location((0, 0, z_pos))
            instantiated_solid_trays.append(loc * base_tray_walls.part)

            boi_box = Solid.make_box(boi_w, boi_d, boi_h)
            boi_box = Location((-boi_w/2, -boi_d/2, -boi_h/2)) * boi_box
            boi_loc = Location((0, 0, z_pos + boi_h / 2.0))
            positioned_boi = boi_loc * boi_box
            instantiated_bois.append(positioned_boi)

        # Build main chamber fluid
        with BuildPart() as main_chamber_fluid:
            Box(chamber_width, chamber_depth, chamber_height)

        fluid_shape = main_chamber_fluid.part - fan_solid.part - heater_solid.part
        for tray in instantiated_solid_trays:
            fluid_shape = fluid_shape - tray

        # Split fluid into layers
        fluid_layers = []
        remaining_fluid = fluid_shape
        for i in range(shelf_quantity):
            z_pos = start_z + (i * shelf_spacing)
            cutting_plane = Plane(Location((0, 0, z_pos - tray_wall_h + t)))
            try:
                lower_part, remaining_fluid = remaining_fluid.split(cutting_plane)
                fluid_layers.append(lower_part)
            except Exception:
                pass
        if remaining_fluid:
            fluid_layers.append(remaining_fluid)

        # Add inlet/outlet
        with BuildPart() as inlet_fluid:
            inlet_z = chamber_height / 2 - (150.0 / 2) - 10.0
            inlet_loc = Location((chamber_width / 2, 0, inlet_z), (0, 90, 0))
            with Locations(inlet_loc):
                Cylinder(radius=150.0 / 2, height=10.0, align=(Align.CENTER, Align.CENTER, Align.MIN))

        with BuildPart() as outlet_fluid:
            outlet_z = -chamber_height / 2 + 150.0 - 35
            outlet_loc = Location((-chamber_width / 2, 0, outlet_z), (0, -90, 0))
            with Locations(outlet_loc):
                Cylinder(radius=150.0 / 2, height=10.0, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Combine all bodies into one compound and rotate
        all_main_bodies = list(fluid_layers)
        all_main_bodies.extend([inlet_fluid.part, outlet_fluid.part, fan_solid.part, heater_solid.part])
        all_main_bodies.extend(instantiated_solid_trays)
        main_domain = Compound(children=all_main_bodies).rotate(Axis.X, 180)

        # Rotate BOI bodies separately and label AFTER rotation
        # (Compound.rotate() creates new shapes, losing .label)
        boi_rotated = Compound(children=instantiated_bois).rotate(Axis.X, 180)
        for child in boi_rotated.children:
            child.label = "BOI"

        # Combine main bodies + BOI into a single STEP file
        combined = Compound(children=list(main_domain.children) + list(boi_rotated.children))
        cad_path = EXPORT_DIR / f"Combined_{lip_w}_{shelf_spacing}.STEP"
        export_step(combined, str(cad_path))

        info("CAD", "Combined geometry exported", cad_path=str(cad_path))
        return cad_path, None, tray_z_positions

    except Exception as e:
        error("CAD", "Geometry generation failed", error=str(e))
        raise RuntimeError(f"CAD generation failed: {str(e)}") from e

def run_meshing(cad_path: pathlib.Path) -> pathlib.Path:
    """Mesh the geometry using PyFluent's Watertight Geometry workflow (modern API).

    Uses the attribute-based ``session.watertight()`` workflow instead of the
    lower-level ``workflow.TaskObject`` API.

    Note: BOI bodies must be included in the CAD STEP file with "BOI" labels
    (set in generate_geometry).

    Args:
        cad_path: Path to the combined CAD STEP file (main bodies + BOI).

    Returns:
        Path to the generated .h5 mesh file.

    Raises:
        RuntimeError: If meshing fails.
    """
    info("MESH", "Starting PyFluent meshing", cad_path=str(cad_path))

    try:
        import ansys.fluent.core as pyfluent

        # Launch meshing session (GUI enabled for visibility)
        session = pyfluent.launch_fluent(
            dimension=3, precision="double", mode="meshing", ui_mode=pyfluent.UIMode.HIDDEN_GUI
        )

        # --- Modern attribute-based Watertight Geometry workflow ---
        watertight = session.watertight()

        # 1. Import combined geometry (main bodies + BOI are in a single STEP, BOI bodies labeled "BOI")
        watertight.import_geometry.file_name = str(cad_path)
        watertight.import_geometry.length_unit = "mm"
        watertight.import_geometry()

        # 2. Add Local Sizing (Body Of Influence control)
        # The BOI label is set in generate_geometry() — Fluent picks it up from the STEP file
        add_local_sizing = watertight.add_local_sizing
        add_local_sizing.boi_control_name = "tray_boi"
        add_local_sizing.boi_execution = "Body Of Influence"
        add_local_sizing.boi_face_label_list = ["BOI"]
        add_local_sizing.boi_size = 2.0
        add_local_sizing.boi_growth_rate = 1.2
        add_local_sizing.boi_zoneor_label = "label"
        add_local_sizing()

        # 4. Generate Surface Mesh
        create_surface_mesh = watertight.create_surface_mesh
        create_surface_mesh.cfd_surface_mesh_controls.min_size = 0.5
        create_surface_mesh.cfd_surface_mesh_controls.max_size = 5.0
        create_surface_mesh.cfd_surface_mesh_controls.growth_rate = 1.2
        create_surface_mesh.cfd_surface_mesh_controls.curvature_normal_angle = 15.0
        create_surface_mesh()

        # 5. Describe Geometry
        # NOTE: Since the geometry includes solid BOI bodies, we must declare
        # the geometry type as "both fluid and solid regions and/or voids"
        # rather than just "fluid".
        describe_geometry = watertight.describe_geometry
        describe_geometry.update_child_tasks(setup_type_changed=False)
        describe_geometry.setup_type = "The geometry consists of both fluid and solid regions and/or voids"
        describe_geometry.update_child_tasks(setup_type_changed=True)
        describe_geometry()

        # 6. Update Boundaries
        update_boundaries = watertight.update_boundaries
        update_boundaries.boundary_label_list = ["inlet", "outlet", "wall", "tray"]
        update_boundaries.boundary_label_type_list = ["velocity-inlet", "pressure-outlet", "wall", "wall"]
        update_boundaries()

        # 7. Update Regions
        watertight.update_regions()

        # 8. Add Boundary Layers (configure properties directly on the task)
        add_boundary_layer = watertight.add_boundary_layer
        add_boundary_layer.control_name = "boundary_layer_1"
        add_boundary_layer.number_of_layers = 3
        add_boundary_layer.growth_rate = 1.2
        add_boundary_layer.first_layer_height = 0.1
        add_boundary_layer()

        # 9. Generate Volume Mesh
        create_volume_mesh = watertight.create_volume_mesh
        create_volume_mesh.volume_fill = "poly-hexcore"
        create_volume_mesh.volume_fill_controls.hex_max_cell_length = 5.0
        create_volume_mesh.volume_fill_controls.growth_rate = 1.2
        create_volume_mesh()

        # Save mesh via TUI execute_tui (the only reliable path in meshing mode)
        mesh_path = MESH_DIR / f"{cad_path.stem}.msh.h5"
        session.execute_tui(f"/file/write-mesh {str(mesh_path)}")
        session.exit()

        info("MESH", "Mesh generated successfully", mesh_path=str(mesh_path))
        return mesh_path

    except Exception as e:
        error("MESH", "Meshing failed", error=str(e))
        raise RuntimeError(f"Meshing failed: {str(e)}") from e

def _identify_inlet_outlet(bc) -> Tuple[str, str]:
    """Identify inlet and outlet zones from wall boundary conditions.
    
    The inlet and outlet cylinders produce wall face zones named
    ``compound-0-compound:1`` and ``compound-1-compound:1`` respectively.
    
    Filtering strategy (verified against Fluent 24.1 zone naming):
    - Shadow zones: contain ``-shadow`` \u2192 skip
    - Interface zones: contain ``-compound-compound-`` \u2192 skip
    - BOI interface zones: start with ``boi-boi-`` \u2192 skip
    - Simple wall zones: kept for inlet/outlet identification
    
    Returns:
        Tuple of (inlet_zone_name, outlet_zone_name).
    
    Raises:
        RuntimeError: If zones cannot be identified.
    """
    all_wall_zones = list(bc.wall.keys()) if hasattr(bc.wall, 'keys') else []
    info("SOLVER", "All wall zones", zones=all_wall_zones)
    
    # Filter: keep only simple wall zones (not interfaces/shadows)
    surface_walls = []
    for name in all_wall_zones:
        # Skip shadow zones
        if '-shadow' in name:
            continue
        # Skip BOI interface zones (boi-boi-compound-X-compound)
        if name.startswith('boi-boi-'):
            continue
        # Skip interface zones: these contain '-compound-compound-'
        # e.g. compound-1-compound-compound-2-compound
        if '-compound-compound-' in name:
            continue
        surface_walls.append(name)
    
    info("SOLVER", "Filtered surface wall zones", zones=surface_walls)
    
    inlet_zone = None
    outlet_zone = None
    
    # Primary identification: compound-0-compound:1 = inlet, compound-1-compound:1 = outlet
    for name in surface_walls:
        if name.startswith('compound-0-compound') and ':1' in name:
            inlet_zone = name
        elif name.startswith('compound-1-compound') and ':1' in name:
            outlet_zone = name
    
    # Fallback: try without :1 suffix
    if inlet_zone is None or outlet_zone is None:
        for name in surface_walls:
            if name == 'compound-0-compound' and inlet_zone is None:
                inlet_zone = name
            elif name == 'compound-1-compound' and outlet_zone is None:
                outlet_zone = name
    
    # Last resort: pick the two smallest zones by name order
    if inlet_zone is None and len(surface_walls) >= 1:
        inlet_zone = surface_walls[0]
    if outlet_zone is None and len(surface_walls) >= 2:
        outlet_zone = surface_walls[1]
    
    if inlet_zone is None or outlet_zone is None:
        raise RuntimeError(
            f"Could not identify inlet/outlet zones. "
            f"Wall zones: {all_wall_zones}, filtered: {surface_walls}"
        )
    
    info("SOLVER", "Identified zones", inlet=inlet_zone, outlet=outlet_zone)
    return inlet_zone, outlet_zone


def run_solver(mesh_path: pathlib.Path, tray_z: List[float]) -> Tuple[float, float, List[float], List[float]]:
    """Run the CFD solver with k-omega SST turbulence model.
    
    Uses verified PyFluent 0.24.x Pythonic API:
    - bc.set_zone_type(zone_list=[...], new_type=...) for boundary type changes
    - methods.discretization_scheme.pressure/mom/k/omega for discretization
    - report_defs.surface.create() for report definitions
    - file.write(file_type='case-data') for post-processing
    
    Args:
        mesh_path: Path to the .h5 mesh file.
        tray_z: Z positions of trays.
        
    Returns:
        Tuple of (mean_cov, pressure_drop, avg_vels, std_vels).
        
    Raises:
        RuntimeError: If solver fails.
    """
    info("SOLVER", "Starting PyFluent solver", mesh_path=str(mesh_path))

    try:
        import ansys.fluent.core as pyfluent
        import pyvista as pv

        # Initialize solver session
        session = pyfluent.launch_fluent(
            dimension=3, precision="double", mode="solver",
            ui_mode=pyfluent.UIMode.HIDDEN_GUI
        )
        session.file.read(file_type="mesh", file_name=str(mesh_path))

        # --- Solver model setup (Pythonic API) ---
        setup = session.setup
        setup.models.viscous.model = "k-omega"
        setup.models.energy.enabled = True
        setup.materials.database.copy_by_name(type="fluid", name="air")

        # --- Identify inlet/outlet zones ---
        bc = setup.boundary_conditions
        inlet_zone, outlet_zone = _identify_inlet_outlet(bc)

        # --- Change zone types (verified: set_zone_type works) ---
        bc.set_zone_type(zone_list=[inlet_zone], new_type="velocity-inlet")
        bc.set_zone_type(zone_list=[outlet_zone], new_type="pressure-outlet")
        info("SOLVER", "Zone types changed", inlet=inlet_zone, outlet=outlet_zone)

        # --- Set boundary conditions ---
        vi = bc.velocity_inlet
        po = bc.pressure_outlet

        if inlet_zone in vi.keys():
            vi[inlet_zone].momentum.velocity = 5.0

        if outlet_zone in po.keys():
            po[outlet_zone].momentum.gauge_pressure = 0.0

        info("SOLVER", "Boundary conditions configured")

        # --- Solution methods (Pythonic API) ---
        # Keys verified: pressure, mom, k, omega
        solution = session.solution
        ds = solution.methods.discretization_scheme
        try:
            ds.pressure = "standard"
            ds.mom = "second-order-upwind"
            ds.k = "second-order-upwind"
            ds.omega = "second-order-upwind"
            info("SOLVER", "Discretization schemes set via Pythonic API")
        except Exception as e:
            warning("SOLVER", "Could not set discretization schemes",
                    error=str(e))

        # --- Initialize and solve ---
        solution.initialization.hybrid_initialize()
        solution.run_calculation.iterate(iter_count=300)
        info("SOLVER", "Solver iteration complete")

        # --- Post-processing: pressure drop via report definitions ---
        report_defs = solution.report_definitions
        inlet_pressure = 0.0
        outlet_pressure = 0.0

        try:
            report_defs.surface.create("inlet_avg_pressure")
            inlet_def = report_defs.surface["inlet_avg_pressure"]
            inlet_def.report_type = "area-weighted-avg"
            inlet_def.field = "pressure"
            inlet_def.surface_names = [inlet_zone]

            report_defs.surface.create("outlet_avg_pressure")
            outlet_def = report_defs.surface["outlet_avg_pressure"]
            outlet_def.report_type = "area-weighted-avg"
            outlet_def.field = "pressure"
            outlet_def.surface_names = [outlet_zone]

            report_defs.compute()

            inlet_pressure = float(inlet_def.value) if hasattr(inlet_def, 'value') else 0.0
            outlet_pressure = float(outlet_def.value) if hasattr(outlet_def, 'value') else 0.0
        except Exception as e:
            warning("SOLVER", "Report definitions failed, using zero pressure drop",
                    error=str(e))

        pressure_drop = inlet_pressure - outlet_pressure
        info("SOLVER", "Pressure drop",
             inlet_pressure=inlet_pressure, outlet_pressure=outlet_pressure,
             pressure_drop=pressure_drop)

        # --- Write case+data file for PyVista post-processing ---
        case_path = mesh_path.with_suffix(".cas.h5")
        session.file.write(file_type="case-data", file_name=str(case_path))
        session.exit()
        info("SOLVER", "Case+data written", case_path=str(case_path))

        # --- Load with PyVista for velocity field ---
        grid = pv.read(str(case_path))
        tray_velocities = []
        for z in tray_z:
            try:
                slice_data = grid.slice(normal="z", origin=(0, 0, z))
                # Try velocity-magnitude first, then any velocity field
                vel_key = None
                for key in slice_data.point_data.keys():
                    if key == "velocity-magnitude":
                        vel_key = key
                        break
                if vel_key is None:
                    for key in slice_data.point_data.keys():
                        if "velocity" in key.lower():
                            vel_key = key
                            break
                if vel_key is not None:
                    tray_velocities.append(slice_data[vel_key])
                else:
                    warning("SOLVER", "No velocity data at tray Z", z=z,
                            available_keys=list(slice_data.point_data.keys()))
                    tray_velocities.append(np.array([0.0]))
            except Exception as e:
                warning("SOLVER", "Failed to extract velocity at tray Z",
                        z=z, error=str(e))
                tray_velocities.append(np.array([0.0]))

        # Calculate CoV (coefficient of variation = std/mean)
        avg_vels = [np.mean(v) for v in tray_velocities]
        std_vels = [np.std(v) for v in tray_velocities]
        covs = [std / avg for std, avg in zip(std_vels, avg_vels) if avg > 0]
        mean_cov = np.mean(covs) if covs else float('inf')

        info("SOLVER", "Solver completed",
             mean_cov=mean_cov, pressure_drop=pressure_drop)
        return mean_cov, pressure_drop, avg_vels, std_vels

    except Exception as e:
        error("SOLVER", "Solver failed", error=str(e))
        raise RuntimeError(f"Solver failed: {str(e)}") from e

def objective_function(x: List[float]) -> float:
    """Wrapper for optimization: x = [lip_w, shelf_spacing].
    
    Args:
        x: List of design variables [lip_w, shelf_spacing].
        
    Returns:
        Scalar objective value (mean CoV).
        
    Raises:
        RuntimeError: If any stage fails.
    """
    lip_w, shelf_spacing = x

    try:
        # Generate geometry
        cad_path, boi_path, tray_z = generate_geometry(lip_w, shelf_spacing)

        # Mesh
        mesh_path = run_meshing(cad_path)

        # Solve
        mean_cov, pressure_drop, _, _ = run_solver(mesh_path, tray_z)

        # Log to DB
        design_id = insert_design(
            name=f"design_{lip_w}_{shelf_spacing}",
            params={"lip_w": lip_w, "shelf_spacing": shelf_spacing}
        )
        insert_simulation(
            design_id=design_id,
            mesh_path=str(mesh_path),
            solver_path=None,
            mean_cov=mean_cov,
            pressure_drop=pressure_drop,
            objective=mean_cov,  # Minimize CoV
            status="SUCCESS"
        )

        # Explicitly return the mean_cov
        return mean_cov

    except Exception as e:
        error("PIPELINE", "Objective function failed", error=str(e))
        raise RuntimeError(f"Objective function failed: {str(e)}") from e