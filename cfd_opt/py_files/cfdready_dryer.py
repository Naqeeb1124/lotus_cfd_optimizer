import os
from build123d import (
    BuildPart, BuildSketch, Rectangle, Locations, Location, Mode, Align, 
    Compound, Cylinder, Box, Plane, extrude, export_step, Axis, Solid
)
from ocp_vscode import show

export_dir = r"E:\Projects\lotus_power\CAD files\build123d_dump"
export_main_path = os.path.join(export_dir, "Body.STEP")
export_boi_path = os.path.join(export_dir, "BOI.STEP")

# Define design parameters
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

# ==========================================
# 1. Generate Physical Hardware Components
# ==========================================
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

# --- FIX APPLIED HERE ---
boi_h = 50.0  # Increased from 25.0 to 50.0 to fill the air gap above the tray
boi_w = inner_w - 4.0
boi_d = inner_d - 4.0
start_z = -chamber_height / 2 + 250.0

for i in range(shelf_quantity):
    z_pos = start_z + (i * shelf_spacing)
    loc = Location((0, 0, z_pos))
    instantiated_solid_trays.append(loc * base_tray_walls.part)
    
    # Create raw Solid box directly for algebra transformations
    boi_box = Solid.make_box(boi_w, boi_d, boi_h)
    # Center the box to align with build123d coordinate expectations
    boi_box = Location((-boi_w/2, -boi_d/2, -boi_h/2)) * boi_box
    
    # Changed minus (-) to plus (+) so the BOI shifts UP into the air gap
    boi_loc = Location((0, 0, z_pos + boi_h / 2.0))
    
    moved_boi = boi_loc * boi_box
    instantiated_bois.append(moved_boi)

# ==========================================
# 2. Build and Slice Fluid Domain
# ==========================================
with BuildPart() as main_chamber_fluid:
    Box(chamber_width, chamber_depth, chamber_height)

# Subtract hardware geometry from main chamber volume
fluid_shape = main_chamber_fluid.part - fan_solid.part - heater_solid.part
for tray in instantiated_solid_trays:
    fluid_shape = fluid_shape - tray

# Slice fluid body by layers safely BEFORE adding inlets/outlets
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

# Add inlet / outlet extensions to the final sliced domain list
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

# ==========================================
# 3. Assemble and Export System
# ==========================================
all_main_bodies = list(fluid_layers)
all_main_bodies.extend([inlet_fluid.part, outlet_fluid.part, fan_solid.part, heater_solid.part])
all_main_bodies.extend(instantiated_solid_trays)

main_domain = Compound(children=all_main_bodies).rotate(Axis.X, 180)
boi_domain = Compound(children=instantiated_bois).rotate(Axis.X, 180)

# Optional visualization check in VS Code via ocp_vscode
show(main_domain)

os.makedirs(export_dir, exist_ok=True)
export_step(main_domain, export_main_path)
export_step(boi_domain, export_boi_path)

print(f"Main geometry exported to: {export_main_path}")
print(f"BOI geometry exported to: {export_boi_path}")