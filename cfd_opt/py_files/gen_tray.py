import os
from build123d import (
    BuildPart, BuildSketch, Rectangle, Locations, Location, 
    Mode, Compound, extrude, export_step, Plane
)
from ocp_vscode import show


tray_outer_w = 400.0    
tray_outer_d = 300.0    
tray_wall_h = 36.5      
t = 1.5                 
lip_w = 40.0            
inner_w = tray_outer_w - (2 * t)
inner_d = tray_outer_d - (2 * t)



with BuildSketch() as tray_floor_porous:
    Rectangle(inner_w, inner_d)
with BuildPart() as tray_solid_walls:
    with BuildSketch():
        Rectangle(tray_outer_w, tray_outer_d)
        Rectangle(inner_w, inner_d, mode=Mode.SUBTRACT)
    extrude(amount=tray_wall_h)
    with BuildSketch(Plane.XY.offset(tray_wall_h - t)):
        with Locations((-tray_outer_w / 2 - lip_w / 2, 0), 
                       (tray_outer_w / 2 + lip_w / 2, 0)):
            Rectangle(lip_w, tray_outer_d)
    extrude(amount=t)
tray_entity = Compound(children=[
    tray_floor_porous.sketch, 
    tray_solid_walls.part
])


export_dir = r"E:\Projects\lotus_power\CAD files\build123d_dump"
os.makedirs(export_dir, exist_ok=True)
export_path = os.path.join(export_dir, "Tray_Entity.STEP")


export_step(tray_entity, export_path)
print(f"Tray geometry successfully exported to: {export_path}")


show(
    tray_floor_porous, tray_solid_walls, 
    names=["tray_floor_porous", "tray_solid_walls"],
    colors=["blue", "gray"]
)