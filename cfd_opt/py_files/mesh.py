import os
import sys
from pathlib import Path
import ansys.fluent.core as pyfluent

export_dir = r"E:\Projects\lotus_power\CAD_files\build123d_dump"
cad_file_path = Path(os.path.join(export_dir, "Body.STEP")).as_posix()
boi_file_path = Path(os.path.join(export_dir, "BOI.STEP")).as_posix()
mesh_file_path = Path(os.path.join(export_dir, "pollen_dryer5.msh.h5")).as_posix()

def mesh_dryer():
    print("Launching PyFluent in Meshing Mode...")
    try:
        meshing_session = pyfluent.launch_fluent(
            mode="meshing",
            precision="single",  
            processor_count=8    
        )
    except Exception as e:
        print(f"Failed to launch Fluent session: {e}")
        sys.exit(1)
        
    workflow = meshing_session.workflow
    workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
    
    print(f"Importing CAD Geometry from: {cad_file_path}")
    import_geom = workflow.TaskObject["Import Geometry"]
    import_geom.Arguments.set_state({
        "FileName": cad_file_path,
        "LengthUnit": "mm",
        "UseBodyLabels": "Yes"
    })
    import_geom.Execute()
    
    print(f"Importing BOI Geometry from: {boi_file_path}")
    import_geom.InsertNextTask(CommandName="ImportBodyOfInfluenceGeometry")
    boi_import = workflow.TaskObject["Import Body of Influence Geometry"]
    boi_import.Arguments.set_state({
        "GeometryFileName": boi_file_path,  
        "LengthUnit": "mm"
    })
    boi_import.Execute()
    
    print("Adding volumetric local sizing control (Body of Influence)...")
    local_sizing = workflow.TaskObject["Add Local Sizing"]
    
    # FIXED: Replaced the mismatched list comprehension with the exact allowed zones from the error log
    boi_zones = [
        "origin-compound-compound-0-compound",
        "origin-compound-compound-1-compound",
        "origin-compound-compound-2-compound",
        "origin-compound-compound-3-compound",
        "origin-compound-compound-4-compound",
        "origin-compound-compound-5-compound",
        "compound-solid-solid"
    ]
    
    local_sizing.Arguments.set_state({
        "AddChild": "yes",
        "BOIControlName": "above_trays_refinement",
        "BOIExecution": "Body Of Influence",
        "BOIZoneorLabel": "zone",           
        "BOIFaceZoneList": boi_zones,       
        "BOISize": 10.0  
    })
    local_sizing.AddChildAndUpdate()
    
    print("Generating Surface Mesh...")
    surface_mesh = workflow.TaskObject["Generate the Surface Mesh"]
    surface_mesh.Arguments.set_state({
        "CFDSurfaceMeshControls": {
            "MaxSize": 40.0,    
            "MinSize": 3.0      
        }                                                   
    })
    surface_mesh.Execute()
    
    print("Describing Geometry...")
    describe_geo = workflow.TaskObject["Describe Geometry"]
    describe_geo.Arguments.set_state({
        "SetupType": "The geometry consists of both fluid and solid regions and/or voids"
    })
    describe_geo.UpdateChildTasks(SetupTypeChanged=True)
    describe_geo.Execute()
    
    print("Updating Boundaries and Regions...")
    workflow.TaskObject["Update Boundaries"].Execute()
    workflow.TaskObject["Update Regions"].Execute()
    
    print("Setting up Boundary Layers...")
    add_bl = workflow.TaskObject["Add Boundary Layers"]
    add_bl.Arguments.set_state({"NumberOfLayers": 3})
    add_bl.AddChildAndUpdate()
    
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
    
    print(f"Writing mesh to: {mesh_file_path}")
    meshing_session.tui.file.write_mesh(f'"{mesh_file_path}"')
    meshing_session.exit()
    print("Meshing completed successfully.")

if __name__ == "__main__":
    mesh_dryer()