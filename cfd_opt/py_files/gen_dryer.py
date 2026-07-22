import os
from build123d import (
    BuildPart, BuildSketch, Rectangle, Locations, Location, 
    Mode, Align, Compound, Cylinder, Box, Plane, extrude, export_step, add, Axis
)
from ocp_vscode import show

export_dir = r"E:\Projects\lotus_power\CAD files\build123d_dump"
export_filename = "Body.STEP"
export_path = os.path.join(export_dir, export_filename)


shelf_quantity = 16      
shelf_spacing = 90.0

tray_outer_w = 400.0    
tray_outer_d = 300.0    
tray_wall_h = 36.5      
t = 1.5                 
lip_w = 40.0            
inner_w = tray_outer_w - (2 * t)
inner_d = tray_outer_d - (2 * t)

chamber_width = tray_outer_w + (2 * lip_w)  
chamber_depth = tray_outer_d + 20.0        



chamber_height = 250.0 + ((shelf_quantity - 1) * shelf_spacing) + 380.0

inlet_diameter = 150.0
inlet_length = 200.0
outlet_diameter = 150.0
outlet_length = 200.0

heater_width = 400.0
heater_depth = 280.0
heater_height = 60.0
fan_width = 300.0
fan_depth = 260.0
fan_height = 60.0



with BuildSketch() as base_tray_floor:
    Rectangle(inner_w, inner_d)
tray_floor_face = base_tray_floor.faces()[0]

with BuildPart() as base_tray_walls:
    with BuildSketch():
        Rectangle(tray_outer_w, tray_outer_d)
        Rectangle(inner_w, inner_d, mode=Mode.SUBTRACT)
    
    extrude(amount=-tray_wall_h) 
    
    with BuildSketch(Plane.XY.offset(-tray_wall_h)):
        with Locations((-tray_outer_w / 2 - lip_w / 2, 0), 
                       (tray_outer_w / 2 + lip_w / 2, 0)):
            Rectangle(lip_w, tray_outer_d)
    
    extrude(amount=t)

with BuildPart() as fan_zone:
    fan_z = chamber_height / 2 - fan_height / 2 - 40
    with Locations((0, 0, fan_z)):
        Box(fan_width, fan_depth, fan_height)

with BuildPart() as heater_zone:
    heater_z = fan_z - fan_height / 2 - heater_height / 2 - 20
    with Locations((0, 0, heater_z)):
        Box(heater_width, heater_depth, heater_height)

instantiated_solid_trays = []
instantiated_porous_floors = []
with BuildPart() as air_volume:
    Box(chamber_width, chamber_depth, chamber_height)
    
    inlet_z = chamber_height / 2 - (inlet_diameter / 2) - 10.0
    inlet_loc = Location((chamber_width / 2, 0, inlet_z), (0, 90, 0))
    with Locations(inlet_loc):
        Cylinder(radius=inlet_diameter / 2, height=inlet_length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    
    outlet_z = -chamber_height / 2 + outlet_diameter - 35
    outlet_loc = Location((-chamber_width / 2, 0, outlet_z), (0, -90, 0))
    with Locations(outlet_loc):
        Cylinder(radius=outlet_diameter / 2, height=outlet_length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    
    start_z = -chamber_height / 2 + 250.0  
    for i in range(shelf_quantity):
        z_pos = start_z + (i * shelf_spacing)
        loc = Location((0, 0, z_pos))
        
        moved_walls = loc * base_tray_walls.part
        add(moved_walls, mode=Mode.SUBTRACT)
        
        instantiated_solid_trays.append(moved_walls)
        instantiated_porous_floors.append(loc * tray_floor_face)
    
    add(heater_zone.part, mode=Mode.SUBTRACT)
    add(fan_zone.part, mode=Mode.SUBTRACT)

all_shapes = [air_volume.part, heater_zone.part, fan_zone.part]
all_shapes.extend(instantiated_solid_trays)
all_shapes.extend(instantiated_porous_floors)

multi_body_domain = Compound(children=all_shapes)
multi_body_domain = multi_body_domain.rotate(Axis.X, 180)

os.makedirs(export_dir, exist_ok=True)
export_step(multi_body_domain, export_path)
print(f"Multi-body geometry successfully exported to: {export_path}")

show(
    *multi_body_domain.children,
    names=["Main_Air_Volume", "Heater_Zone", "Fan_Zone"] + 
          [f"Tray_Solid_{i}" for i in range(shelf_quantity)] + 
          [f"Tray_Porous_{i}" for i in range(shelf_quantity)]
)