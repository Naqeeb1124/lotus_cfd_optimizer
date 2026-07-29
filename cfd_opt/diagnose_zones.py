import os
import sys
from pathlib import Path

# Add environment variables and custom python path if needed
os.environ["AWP_ROOT241"] = r"E:\Ansys2024\ANSYS Inc\v241"
fluent_path = r"E:\Ansys2024\ANSYS Inc\v241\fluent\ntbin\win64\fluent.exe"

from build123d import (
    BuildPart, BuildSketch, Rectangle, Locations, Location, Mode, Align, 
    Compound, Cylinder, Box, Plane, extrude, export_step, Axis, Solid
)
import ansys.fluent.core as pyfluent

def generate_cad(shelf_quantity=2, shelf_spacing=90.0, lip_w=40.0):
    export_dir = r"E:\Projects\lotus_power\CAD files\build123d_dump"
    export_main_path = os.path.join(export_dir, "Body.STEP")
    export_boi_path = os.path.join(export_dir, "BOI.STEP")
    
    tray_outer_w = 400.0
    tray_outer_d = 300.0
    tray_wall_h = 36.5
    t = 1.5
    inner_w = tray_outer_w - (2 * t)
    inner_d = tray_outer_d - (2 * t)
    chamber_width = tray_outer_w + (2 * lip_w)
    chamber_depth = tray_outer_d + 20.0
    chamber_height = 250.0 + ((shelf_quantity - 1) * shelf_spacing) + 380.0
    inlet_diameter = 150.0
    inlet_length = 10.0
    outlet_diameter = 150.0
    outlet_length = 10.0
    heater_width = 400.0
    heater_depth = 280.0
    heater_height = 60.0
    fan_width = 300.0
    fan_depth = 260.0
    fan_height = 60.0

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
        fan_z = chamber_height / 2 - fan_height / 2 - 40
        with Locations((0, 0, fan_z)):
            Box(fan_width, fan_depth, fan_height)
            
    with BuildPart() as heater_solid:
        heater_z = fan_z - fan_height / 2 - heater_height / 2 - 20
        with Locations((0, 0, heater_z)):
            Box(heater_width, heater_depth, heater_height)
            
    instantiated_solid_trays = []
    instantiated_bois = []

    boi_h = 50.0  
    boi_w = inner_w - 4.0
    boi_d = inner_d - 4.0
    start_z = -chamber_height / 2 + 250.0
    for i in range(shelf_quantity):
        z_pos = start_z + (i * shelf_spacing)
        loc = Location((0, 0, z_pos))
        instantiated_solid_trays.append(loc * base_tray_walls.part)
        
        boi_box = Solid.make_box(boi_w, boi_d, boi_h)
        boi_box = Location((-boi_w/2, -boi_d/2, -boi_h/2)) * boi_box
        boi_loc = Location((0, 0, z_pos + boi_h / 2.0))
        moved_boi = boi_loc * boi_box
        instantiated_bois.append(moved_boi)

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
        inlet_z = chamber_height / 2 - (inlet_diameter / 2) - 10.0
        inlet_loc = Location((chamber_width / 2, 0, inlet_z), (0, 90, 0))
        with Locations(inlet_loc):
            Cylinder(radius=inlet_diameter / 2, height=inlet_length, align=(Align.CENTER, Align.CENTER, Align.MIN))
            
    with BuildPart() as outlet_fluid:
        outlet_z = -chamber_height / 2 + outlet_diameter - 35
        outlet_loc = Location((-chamber_width / 2, 0, outlet_z), (0, -90, 0))
        with Locations(outlet_loc):
            Cylinder(radius=outlet_diameter / 2, height=outlet_length, align=(Align.CENTER, Align.CENTER, Align.MIN))

    all_main_bodies = list(fluid_layers)
    all_main_bodies.extend([inlet_fluid.part, outlet_fluid.part, fan_solid.part, heater_solid.part])
    all_main_bodies.extend(instantiated_solid_trays)
    main_domain = Compound(children=all_main_bodies).rotate(Axis.X, 180)
    boi_domain = Compound(children=instantiated_bois).rotate(Axis.X, 180)

    os.makedirs(export_dir, exist_ok=True)
    export_step(main_domain, export_main_path)
    export_step(boi_domain, export_boi_path)
    print(f"STEP files exported.")
    return export_main_path, export_boi_path, shelf_quantity

def run_meshing(cad_path, boi_path, shelf_quantity):
    export_dir = r"E:\Projects\lotus_power\CAD files\build123d_dump"
    mesh_path = os.path.join(export_dir, "diagnose.msh.h5")
    
    print("Launching PyFluent Meshing...")
    meshing_session = pyfluent.launch_fluent(
        fluent_path=fluent_path,
        mode="meshing",
        precision="double",
        processor_count=4,
        show_gui=False
    )
    
    workflow = meshing_session.workflow
    workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
    
    # Import Geometry
    import_geom = workflow.TaskObject["Import Geometry"]
    import_geom.Arguments.set_state({
        "FileName": Path(cad_path).as_posix(),
        "LengthUnit": "mm",
        "UseBodyLabels": "Yes"
    })
    import_geom.Execute()
    
    # Import BOI
    import_geom.InsertNextTask(CommandName="ImportBodyOfInfluenceGeometry")
    boi_import = workflow.TaskObject["Import Body of Influence Geometry"]
    boi_import.Arguments.set_state({
        "GeometryFileName": Path(boi_path).as_posix(),
        "LengthUnit": "mm"
    })
    boi_import.Execute()
    
    # Add Local Sizing
    local_sizing = workflow.TaskObject["Add Local Sizing"]
    boi_zones = ["compound-solid-solid"] + [f"compound-solid-{i}-solid" for i in range(shelf_quantity - 1)]
    local_sizing.Arguments.set_state({
        "AddChild": "yes",
        "BOIControlName": "above_trays_refinement",
        "BOIExecution": "Body Of Influence",
        "BOIZoneorLabel": "zone",
        "BOIFaceZoneList": boi_zones,
        "BOISize": 15.0
    })
    local_sizing.AddChildAndUpdate()
    
    # Surface Mesh
    surface_mesh = workflow.TaskObject["Generate the Surface Mesh"]
    surface_mesh.Arguments.set_state({
        "CFDSurfaceMeshControls": {
            "MaxSize": 60.0,
            "MinSize": 5.0
        }
    })
    surface_mesh.Execute()
    
    # Describe Geometry
    describe_geo = workflow.TaskObject["Describe Geometry"]
    describe_geo.Arguments.set_state({
        "SetupType": "The geometry consists of both fluid and solid regions and/or voids"
    })
    describe_geo.UpdateChildTasks(SetupTypeChanged=True)
    describe_geo.Execute()
    
    # Update Boundaries & Regions
    workflow.TaskObject["Update Boundaries"].Execute()
    workflow.TaskObject["Update Regions"].Execute()
    
    # Boundary Layers
    add_bl = workflow.TaskObject["Add Boundary Layers"]
    add_bl.Arguments.set_state({"NumberOfLayers": 2})
    add_bl.AddChildAndUpdate()
    
    # Volume Mesh
    volume_mesh = workflow.TaskObject["Generate the Volume Mesh"]
    volume_mesh.Arguments.set_state({
        "VolumeFill": "poly-hexcore",
        "VolumeFillControls": {
            "HexMaxCellLength": 40.0
        }
    })
    volume_mesh.Execute()
    
    # Write Mesh
    meshing_session.tui.file.write_mesh(f'"{Path(mesh_path).as_posix()}"')
    meshing_session.exit()
    print("Meshing complete.")
    return mesh_path

def diagnose_solver(mesh_path):
    print("Launching PyFluent Solver...")
    solver = pyfluent.launch_fluent(
        fluent_path=fluent_path,
        mode="solver",
        precision="double",
        processor_count=4,
        show_gui=False
    )
    
    print("Reading mesh...")
    solver.file.read_mesh(file_name=Path(mesh_path).as_posix())
    
    print("\n--- Cell Zone Conditions (Fluids) ---")
    try:
        fluid_zones = solver.setup.cell_zone_conditions.fluid.keys()
        print("Fluid Zones:", list(fluid_zones))
    except Exception as e:
        print("Error getting fluid zones:", e)
        
    print("\n--- Cell Zone Conditions (Solids) ---")
    try:
        solid_zones = solver.setup.cell_zone_conditions.solid.keys()
        print("Solid Zones:", list(solid_zones))
    except Exception as e:
        print("Error getting solid zones:", e)
        
    print("\n--- Boundary Conditions (Velocity Inlets) ---")
    try:
        inlets = solver.setup.boundary_conditions.velocity_inlet.keys()
        print("Velocity Inlets:", list(inlets))
    except Exception as e:
        print("Error getting inlets:", e)
        
    print("\n--- Boundary Conditions (Pressure Outlets) ---")
    try:
        outlets = solver.setup.boundary_conditions.pressure_outlet.keys()
        print("Pressure Outlets:", list(outlets))
    except Exception as e:
        print("Error getting outlets:", e)
        
    print("\n--- All Boundary Zone Names ---")
    try:
        all_zones = solver.setup.boundary_conditions.wall.keys()
        print("Wall Zones:", list(all_zones))
    except Exception as e:
        print("Error getting walls:", e)

    # Let's also query via TUI just in case
    print("\nTUI Zone list:")
    tui_output = solver.tui.define.boundary_conditions.list_zones()
    print(tui_output)
    
    solver.exit()

if __name__ == "__main__":
    cad, boi, sq = generate_cad(shelf_quantity=2)
    mesh_path = run_meshing(cad, boi, sq)
    diagnose_solver(mesh_path)
