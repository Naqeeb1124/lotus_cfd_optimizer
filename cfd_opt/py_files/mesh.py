import os
import sys
from pathlib import Path
import ansys.fluent.core as pyfluent

export_dir = r"E:\Projects\lotus_power\CAD files\build123d_dump"
cad_file_path = Path(os.path.join(export_dir, "Body.STEP")).as_posix()
boi_file_path = Path(os.path.join(export_dir, "BOI.STEP")).as_posix()
mesh_file_path = Path(os.path.join(export_dir, "pollen_dryer5.msh.h5")).as_posix()

def mesh_dryer():
    print("Launching PyFluent in Meshing Mode...")
    
    try:
        meshing_session = pyfluent.launch_fluent(
            mode="meshing",
            precision="single",  # Changed to single precision for meshing
            processor_count=8    # Maxing out your 8-core CPU
        )
    except Exception as e:
        print(f"Failed to launch Fluent session: {e}")
        sys.exit(1)
        
    workflow = meshing_session.workflow
    workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
    
    # -------------------------------------------------------------------------
    # 1. IMPORT MAIN GEOMETRY
    # -------------------------------------------------------------------------
    print(f"Importing CAD Geometry from: {cad_file_path}")
    import_geom = workflow.TaskObject["Import Geometry"]
    import_geom.Arguments.set_state({
        "FileName": cad_file_path,
        "LengthUnit": "mm",
        "UseBodyLabels": "Yes"
    })
    import_geom.Execute()
    
    # -------------------------------------------------------------------------
    # 2. IMPORT BOI GEOMETRY
    # -------------------------------------------------------------------------
    print(f"Importing BOI Geometry from: {boi_file_path}")
    import_geom.InsertNextTask(CommandName="ImportBodyOfInfluenceGeometry")
    
    boi_import = workflow.TaskObject["Import Body of Influence Geometry"]
    boi_import.Arguments.set_state({
        "GeometryFileName": boi_file_path,  
        "LengthUnit": "mm"
    })
    boi_import.Execute()

    # -------------------------------------------------------------------------
    # 3. ADD LOCAL SIZING (Fixed: Targeting Zones instead of Labels)
    # -------------------------------------------------------------------------
    print("Adding volumetric local sizing control (Body of Influence)...")
    local_sizing = workflow.TaskObject["Add Local Sizing"]
    
    boi_zones = ["compound-solid-solid"] + [f"compound-solid-{i}-solid" for i in range(15)]
    
    local_sizing.Arguments.set_state({
        "AddChild": "yes",
        "BOIControlName": "above_trays_refinement",
        "BOIExecution": "Body Of Influence",
        "BOIZoneorLabel": "zone",           # CRITICAL FIX: Tell Fluent to look for Zones
        "BOIFaceZoneList": boi_zones,       # CRITICAL FIX: Pass the list to the Zone argument
        "BOISize": 10.0  
    })
    local_sizing.AddChildAndUpdate()
    
    # -------------------------------------------------------------------------
    # 4. GENERATE SURFACE MESH
    # -------------------------------------------------------------------------
    print("Generating Surface Mesh...")
    surface_mesh = workflow.TaskObject["Generate the Surface Mesh"]
    surface_mesh.Arguments.set_state({
        "CFDSurfaceMeshControls": {
            "MaxSize": 40.0,    
            "MinSize": 3.0      
        }                                                   
    })
    surface_mesh.Execute()
    
    # -------------------------------------------------------------------------
    # 5. DESCRIBE GEOMETRY
    # -------------------------------------------------------------------------
    print("Describing Geometry...")
    describe_geo = workflow.TaskObject["Describe Geometry"]
    describe_geo.Arguments.set_state({
        "SetupType": "The geometry consists of both fluid and solid regions and/or voids"
    })
    describe_geo.UpdateChildTasks(SetupTypeChanged=True)
    describe_geo.Execute()
    
    # -------------------------------------------------------------------------
    # 6. UPDATE BOUNDARIES & REGIONS
    # -------------------------------------------------------------------------
    print("Updating Boundaries and Regions...")
    workflow.TaskObject["Update Boundaries"].Execute()
    workflow.TaskObject["Update Regions"].Execute()
    
    # -------------------------------------------------------------------------
    # 7. ADD BOUNDARY LAYERS
    # -------------------------------------------------------------------------
    print("Setting up Boundary Layers...")
    add_bl = workflow.TaskObject["Add Boundary Layers"]
    add_bl.Arguments.set_state({"NumberOfLayers": 3})
    add_bl.AddChildAndUpdate()
    
    # -------------------------------------------------------------------------
    # 8. GENERATE VOLUME MESH
    # -------------------------------------------------------------------------
    print("Generating Poly-Hexcore Volume Mesh...")
    volume_mesh = workflow.TaskObject["Generate the Volume Mesh"]
    volume_mesh.Arguments.set_state({
        "VolumeFill": "poly-hexcore",
        "VolumeFillControls": {
            "HexMaxCellLength": 30.0
        }
    })
    volume_mesh.Execute()
    
    print("Checking mesh quality...")
    meshing_session.tui.mesh.check_mesh()

    # -------------------------------------------------------------------------
    # 9. EXPORT MESH FILE & EXIT
    # -------------------------------------------------------------------------
    print(f"Writing mesh to: {mesh_file_path}")
    meshing_session.tui.file.write_mesh(f'"{mesh_file_path}"')

    meshing_session.exit()
    print("Meshing completed successfully.")

if __name__ == "__main__":
    mesh_dryer()