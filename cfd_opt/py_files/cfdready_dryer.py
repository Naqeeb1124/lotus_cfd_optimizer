import os
from build123d import (
    BuildPart,
    BuildSketch,
    Rectangle,
    Locations,
    Location,
    Mode,
    Align,
    Compound,
    Cylinder,
    Box,
    Plane,
    extrude,
    export_step,
    Axis,
    Solid
)
from ocp_vscode import show

export_dir = r"E:\Projects\lotus_power\CAD_files\build123d_dump"
export_main_path = os.path.join(export_dir, "Body.STEP")
export_boi_path = os.path.join(export_dir, "BOI.STEP")

# --- Layout Parameters ---
row_quantity = 10
column_quantity = 2
shelf_spacing = 90.0
column_spacing = 100.0  # Gap between the two columns
tray_outer_w = 400.0
tray_outer_d = 300.0
tray_wall_h = 36.5
t = 1.5
lip_w = 40.0
inner_w = tray_outer_w - (2 * t)
inner_d = tray_outer_d - (2 * t)

# Chamber width accommodates two trays + gap + lips
chamber_width = (column_quantity * tray_outer_w) + ((column_quantity - 1) * column_spacing) + (2 * lip_w)
chamber_depth = tray_outer_d + 20.0
chamber_height = 250.0 + ((row_quantity - 1) * shelf_spacing) + 380.0

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

# --- Geometry Generation ---

# 1. Base Tray (Fixed Floor Orientation)
with BuildPart() as base_tray_walls:
    # Outer side walls (Extrude from 0 down to -36.5)
    with BuildSketch():
        Rectangle(tray_outer_w, tray_outer_d)
        Rectangle(inner_w, inner_d, mode=Mode.SUBTRACT)
    extrude(amount=-tray_wall_h)
    
    # Floor base (Added at Z=0 pre-rotation, so it becomes the bottom post-rotation)
    with BuildSketch():
        Rectangle(tray_outer_w, tray_outer_d)
    extrude(amount=-t)

    # Side lips (Added at Z=-36.5 pre-rotation, so they become the top flanges post-rotation)
    with BuildSketch(Plane.XY.offset(-tray_wall_h)):
        with Locations((-tray_outer_w / 2 - lip_w / 2, 0), (tray_outer_w / 2 + lip_w / 2, 0)):
            Rectangle(lip_w, tray_outer_d)
    extrude(amount=t)

# 2. Internal Components (Fan & Heater)
with BuildPart() as fan_solid:
    fan_z = chamber_height / 2 - fan_height / 2 - 40
    with Locations((0, 0, fan_z)):
        Box(fan_width, fan_depth, fan_height)

with BuildPart() as heater_solid:
    heater_z = fan_z - fan_height / 2 - heater_height / 2 - 20
    with Locations((0, 0, heater_z)):
        Box(heater_width, heater_depth, heater_height)

# 3. Dividing Wall
wall_thickness = column_spacing - (2 * lip_w)  
wall_depth = tray_outer_d        
wall_height = ((row_quantity - 1) * shelf_spacing) + tray_wall_h + t

start_z = -chamber_height / 2 + 250.0
wall_z_pos = start_z + (((row_quantity - 1) * shelf_spacing) / 2) - (tray_wall_h / 2)

with BuildPart() as dividing_wall:
    with Locations((0, 0, wall_z_pos)):
        Box(wall_thickness, wall_depth, wall_height)

# 4. Trays & BOI Placement
instantiated_solid_trays = []
instantiated_bois = []
boi_h = 50.0 # Reverted to your original 50mm height
boi_w = inner_w - 4.0
boi_d = inner_d - 4.0

total_trays_width = (column_quantity * tray_outer_w) + ((column_quantity - 1) * column_spacing)
start_x = -total_trays_width / 2 + tray_outer_w / 2

for col in range(column_quantity):
    x_pos = start_x + col * (tray_outer_w + column_spacing)
    for row in range(row_quantity):
        z_pos = start_z + (row * shelf_spacing)
        
        # Instantiate Tray
        loc = Location((x_pos, 0, z_pos))
        instantiated_solid_trays.append(loc * base_tray_walls.part)
        
        # Instantiate BOI (Calculated to rest precisely on the pre-rotation inner floor)
        boi_box = Solid.make_box(boi_w, boi_d, boi_h)
        boi_box = Location((-boi_w / 2, -boi_d / 2, -boi_h / 2)) * boi_box
        
        # The floor inner surface is at (z_pos - t). 
        # We drop by (boi_h / 2) so it spans from the floor into the cavity.
        boi_z_center = z_pos - t - (boi_h / 2.0)
        boi_loc = Location((x_pos, 0, boi_z_center))
        instantiated_bois.append(boi_loc * boi_box)

# 5. Fluid Domain Generation
with BuildPart() as main_chamber_fluid:
    Box(chamber_width, chamber_depth, chamber_height)

# Subtract ALL solid obstructions from the fluid
fluid_shape = (
    main_chamber_fluid.part 
    - fan_solid.part 
    - heater_solid.part 
    - dividing_wall.part
)

for tray in instantiated_solid_trays:
    fluid_shape = fluid_shape - tray

# Split Fluid by Rows
fluid_layers = []
remaining_fluid = fluid_shape

for row in range(row_quantity):
    z_pos = start_z + (row * shelf_spacing)
    cutting_plane = Plane(Location((0, 0, z_pos - tray_wall_h + t)))
    try:
        lower_part, remaining_fluid = remaining_fluid.split(cutting_plane)
        fluid_layers.append(lower_part)
    except Exception:
        pass

if remaining_fluid:
    fluid_layers.append(remaining_fluid)

# Inlet / Outlet Fluid
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

# --- Assemble Everything & Export ---
all_main_bodies = list(fluid_layers)
all_main_bodies.extend([
    inlet_fluid.part, 
    outlet_fluid.part, 
    fan_solid.part, 
    heater_solid.part, 
    dividing_wall.part
])
all_main_bodies.extend(instantiated_solid_trays)

# The 180-degree rotation flips Z, correctly orienting the trays and BOIs
main_domain = Compound(children=all_main_bodies).rotate(Axis.X, 180)
boi_domain = Compound(children=instantiated_bois).rotate(Axis.X, 180)

# Render layout window
show(main_domain)

# Export assets
os.makedirs(export_dir, exist_ok=True)
export_step(main_domain, export_main_path)
export_step(boi_domain, export_boi_path)

print(f"Main geometry exported to: {export_main_path}")
print(f"BOI geometry exported to: {export_boi_path}")