"""Modular CFD pipeline for the autonomous design framework.
Functions:
- generate_geometry: Parametric CAD generation (Build123d).
- run_meshing: PyFluent meshing workflow (Watertight Geometry, TaskObject API).
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

# Assuming these are available in your local environment
from src.utils.db import insert_design, insert_simulation, insert_critique
from src.utils.logger import info, warning, error

EXPORT_DIR = pathlib.Path("E:/Projects/lotus_power/CAD_files/build123d_dump")
MESH_DIR = pathlib.Path("E:/Projects/lotus_power/meshes")
MESH_DIR.mkdir(exist_ok=True, parents=True)
EXPORT_DIR.mkdir(exist_ok=True, parents=True)

def generate_geometry(lip_w: float, shelf_spacing: float) -> Tuple[pathlib.Path, pathlib.Path, List[float]]:
    """Generate CAD geometry and return (cad_path, boi_path, tray_z_positions)."""
    info("CAD", "Generating geometry", lip_w=lip_w, shelf_spacing=shelf_spacing)
    try:
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
        
        with BuildPart() as base_tray_walls:
            with BuildSketch():
                Rectangle(tray_outer_w, tray_outer_d)
                Rectangle(inner_w, inner_d, mode=Mode.SUBTRACT)
            extrude(amount=-tray_wall_h)
            with BuildSketch(Plane.XY.offset(-tray_wall_h)):
                with Locations((-tray_outer_w / 2 - lip_w / 2, 0), (tray_outer_w / 2 + lip_w / 2, 0)):
                    Rectangle(lip_w, tray_outer_d)
            extrude(amount=t)
            
        with BuildPart() as fan_solid:
            fan_z = chamber_height / 2 - 60.0 / 2 - 40
            with Locations((0, 0, fan_z)):
                Box(300.0, 260.0, 60.0)
                
        with BuildPart() as heater_solid:
            heater_z = fan_z - 60.0 / 2 - 60.0 / 2 - 20
            with Locations((0, 0, heater_z)):
                Box(400.0, 280.0, 60.0)
                
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
            instantiated_bois.append(boi_loc * boi_box)
            
        with BuildPart() as main_chamber_fluid:
            Box(chamber_width, chamber_depth, chamber_height)
            
        fluid_shape = main_chamber_fluid.part - fan_solid.part - heater_solid.part
        for tray in instantiated_solid_trays:
            fluid_shape = fluid_shape - tray
            
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
                
        all_main_bodies = list(fluid_layers)
        all_main_bodies.extend([inlet_fluid.part, outlet_fluid.part, fan_solid.part, heater_solid.part])
        all_main_bodies.extend(instantiated_solid_trays)
        
        main_domain = Compound(children=all_main_bodies).rotate(Axis.X, 180)
        boi_domain = Compound(children=instantiated_bois).rotate(Axis.X, 180)
        
        cad_path = EXPORT_DIR / f"Body_{lip_w}_{shelf_spacing}.STEP"
        boi_path = EXPORT_DIR / f"BOI_{lip_w}_{shelf_spacing}.STEP"
        
        export_step(main_domain, str(cad_path))
        export_step(boi_domain, str(boi_path))
        
        info("CAD", "Geometry exported", cad_path=str(cad_path), boi_path=str(boi_path))
        return cad_path, boi_path, tray_z_positions
        
    except Exception as e:
        error("CAD", "Geometry generation failed", error=str(e))
        raise RuntimeError(f"CAD generation failed: {str(e)}") from e

def run_meshing(cad_path: pathlib.Path, boi_path: pathlib.Path) -> pathlib.Path:
    """Mesh the geometry using PyFluent's Watertight Geometry workflow."""
    info("MESH", "Starting PyFluent meshing", cad_path=str(cad_path), boi_path=str(boi_path))
    try:
        import ansys.fluent.core as pyfluent
        from ansys.fluent.core import Precision, UIMode #
        
        # Use Precision enum and standard launch arguments
        session = pyfluent.launch_fluent(
            version="23.2", 
            precision=Precision.SINGLE, #
            mode="meshing", 
            show_gui=True
        )
        
        workflow = session.workflow
        workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
        
        # 1. FIX: Removed problematic Refaceting dict and use .as_posix()
        import_task = workflow.TaskObject["Import Geometry"]
        import_task.Arguments.set_state({
            "FileName": cad_path.as_posix(), 
            "LengthUnit": "mm"
        })
        import_task.Execute()
        
        # 2. FIX: Fluent requires POSIX paths (forward slashes)
        session.meshing.ImportBodyOfInfluenceGeometry(FileName=boi_path.as_posix()) #
        
        local_sizing_task = workflow.TaskObject["Add Local Sizing"]
        local_sizing_task.Arguments.set_state({
            "AddChild": "yes",
            "LocalSizingName": "tray_boi",
            "SizingType": "Body Of Influence", 
            "BOISize": 2.0,
            "GrowthRate": 1.2,
            "zone_selection_list": [boi_path.stem]
        })
        local_sizing_task.AddChildAndUpdate()
        
        surface_mesh_task = workflow.TaskObject["Generate the Surface Mesh"]
        surface_mesh_task.Arguments.set_state({
            "CFDSurfaceMeshControls": {
                "MinSize": 0.5,
                "MaxSize": 5.0,
                "GrowthRate": 1.2,
                "CurvatureNormalAngle": 15.0
            }
        })
        surface_mesh_task.Execute()
        
        describe_geometry_task = workflow.TaskObject["Describe Geometry"]
        describe_geometry_task.Arguments.set_state({
            "SetupType": "The geometry consists of only fluid regions with no voids",
            "ExtractFeatures": True,
            "ExtractFeatureAngle": 40.0
        })
        describe_geometry_task.Execute()
        
        update_boundaries_task = workflow.TaskObject["Update Boundaries"]
        update_boundaries_task.Arguments.set_state({
            "BoundaryLabelList": ["inlet", "outlet", "wall", "tray"],
            "BoundaryTypeList": ["velocity-inlet", "pressure-outlet", "wall", "wall"]
        })
        update_boundaries_task.Execute()
        
        update_regions_task = workflow.TaskObject["Update Regions"]
        update_regions_task.Execute()
        
        boundary_layers_task = workflow.TaskObject["Add Boundary Layers"]
        boundary_layers_task.Arguments.set_state({
            "NumberOfLayers": 3,
            "GrowthRate": 1.2,
            "FirstLayerHeight": 0.1
        })
        boundary_layers_task.AddChildAndUpdate()
        
        volume_mesh_task = workflow.TaskObject["Generate the Volume Mesh"]
        volume_mesh_task.Arguments.set_state({
            "VolumeFill": "poly-hexcore",
            "VolumeMeshControls": {
                "MaxCellSize": 5.0,
                "GrowthRate": 1.2
            }
        })
        volume_mesh_task.Execute()
        
        mesh_path = MESH_DIR / f"{cad_path.stem}.msh.h5"
        
        # Ensure POSIX string format on write
        session.tui.file.write_mesh(mesh_path.as_posix()) 
        session.exit()
        
        info("MESH", "Mesh generated successfully", mesh_path=str(mesh_path))
        return mesh_path
        
    except Exception as e:
        error("MESH", "Meshing failed", error=str(e))
        raise RuntimeError(f"Meshing failed: {str(e)}") from e

def run_solver(mesh_path: pathlib.Path, tray_z: List[float]) -> Tuple[float, float, List[float], List[float]]:
    """Run the CFD solver with k-omega SST turbulence model."""
    info("SOLVER", "Starting PyFluent solver", mesh_path=str(mesh_path))
    try:
        import ansys.fluent.core as pyfluent
        from ansys.fluent.core import Precision, UIMode #
        import pyvista as pv
        
        session = pyfluent.launch_fluent(
            version="23.2", 
            precision=Precision.DOUBLE, #
            mode="solver", 
            show_gui=True
        )
        
        # File paths passed to fluent core must be POSIX
        session.file.read(file_type="mesh", file_name=mesh_path.as_posix())
        
        setup = session.setup
        setup.models.viscous.model = "k-omega"
        setup.models.viscous.k_omega_options.sst = True
        setup.models.energy.enabled = True
        setup.materials.database.copy_by_name(type="fluid", name="air")
        
        setup.boundary_conditions.velocity_inlet["inlet"] = {
            "momentum": {
                "velocity": {"option": "value", "value": 5.0},
                "turbulence": {"option": "intensity-and-viscosity-ratio", "intensity": 5.0, "viscosity_ratio": 10.0}
            },
            "thermal": {"temperature": 298.15}
        }
        setup.boundary_conditions.pressure_outlet["outlet"] = {
            "momentum": {"gauge_pressure": 0.0},
            "thermal": {"temperature": 298.15}
        }
        
        solution = session.solution
        solution.methods.discretization_scheme = {
            "pressure": "presto!",
            "momentum": "second-order-upwind",
            "energy": "second-order-upwind"
        }
        
        solution.initialization.hybrid_initialize()
        solution.run_calculation.iterate(iter_count=300)
        
        report = session.solution.report
        inlet_pressure = report.surface_integrals.area_weighted_average(
            surface_name="inlet",
            field="pressure"
        )["pressure"]
        outlet_pressure = report.surface_integrals.area_weighted_average(
            surface_name="outlet",
            field="pressure"
        )["pressure"]
        
        pressure_drop = inlet_pressure - outlet_pressure
        case_path = mesh_path.with_suffix(".cas.h5")
        
        session.file.write(file_type="case", file_name=case_path.as_posix())
        session.exit()
        
        grid = pv.read(str(case_path))
        tray_velocities = []
        
        for z in tray_z:
            slice = grid.slice(normal="z", origin=(0, 0, z))
            velocities = slice["velocity-magnitude"]
            tray_velocities.append(velocities)
            
        avg_vels = [np.mean(v) for v in tray_velocities]
        std_vels = [np.std(v) for v in tray_velocities]
        mean_cov = np.mean([std / avg for std, avg in zip(std_vels, avg_vels) if avg > 0])
        
        info("SOLVER", "Solver completed", mean_cov=mean_cov, pressure_drop=pressure_drop)
        return mean_cov, pressure_drop, avg_vels, std_vels
        
    except Exception as e:
        error("SOLVER", "Solver failed", error=str(e))
        raise RuntimeError(f"Solver failed: {str(e)}") from e

def objective_function(x: List[float]) -> float:
    """Wrapper for optimization: x = [lip_w, shelf_spacing]."""
    lip_w, shelf_spacing = x
    try:
        cad_path, boi_path, tray_z = generate_geometry(lip_w, shelf_spacing)
        mesh_path = run_meshing(cad_path, boi_path)
        mean_cov, pressure_drop, _, _ = run_solver(mesh_path, tray_z)
        
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
            objective=mean_cov,  
            status="SUCCESS"
        )
        return mean_cov
        
    except Exception as e:
        error("PIPELINE", "Objective function failed", error=str(e))
        raise RuntimeError(f"Objective function failed: {str(e)}") from e