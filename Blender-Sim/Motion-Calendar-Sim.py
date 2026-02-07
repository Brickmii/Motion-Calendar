"""
Motion Calendar Simulation - A 3D Visualization of the Six Motion Functions
============================================================================

This Blender Python script creates a 45-second animation visualizing:
1. Heat - A spinning dot with dark color blends (0-7 seconds)
2. Polarity - The dot stretches into a line with +/- ends (7-14 seconds)
3. Existence - Golden spiral emerges, clock begins (14-21 seconds)
4. Righteousness - 2D quadratic plane with X/Y axes (21-28 seconds)
5. Order - Numbers fill the quadrants (Robinson Axioms) (28-35 seconds)
6. Movement - Z-axis emerges, particle bounces in 3D spacetime (35-45 seconds)

To run: Open Blender, go to Scripting tab, paste this script, and click Run.
Then render animation: Render > Render Animation (or Ctrl+F12)

Author: Based on "Ocean From Motion" - The Motion Calendar framework
"""

import bpy
import math
import mathutils
from mathutils import Vector, Euler

# =============================================================================
# CONFIGURATION
# =============================================================================

FPS = 30
TOTAL_SECONDS = 45
TOTAL_FRAMES = FPS * TOTAL_SECONDS

# Scene timing (in seconds)
SCENE_HEAT_START = 0
SCENE_HEAT_END = 7
SCENE_POLARITY_START = 7
SCENE_POLARITY_END = 14
SCENE_EXISTENCE_START = 14
SCENE_EXISTENCE_END = 21
SCENE_RIGHTEOUSNESS_START = 21
SCENE_RIGHTEOUSNESS_END = 28
SCENE_ORDER_START = 28
SCENE_ORDER_END = 35
SCENE_MOVEMENT_START = 35
SCENE_MOVEMENT_END = 45

# =============================================================================
# MOTION CALENDAR CONSTANTS
# =============================================================================

# Golden ratio - the self-similar scaling law
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895

# Thermal quantum - the fundamental unit of heat (in Kelvin)
# Defined by the identity: K × φ² = 4
K = 4 / (PHI ** 2)  # ≈ 1.527864045000421 Kelvin

# Movement constant - total primitive directional operators
MOVEMENT_DIRECTIONS = 12  # Up, Down, Left, Right, Forward, Backward, 
                          # North, South, East, West, Above, Below

# Divisor count - number of motion functions (divisors of 12)
MOTION_FUNCTIONS = 6  # Heat, Polarity, Existence, Righteousness, Order, Movement

# The Quadratic - from Righteousness (4 quadrants)
QUADRATIC = 4

# Incomplete motion - movement directions minus one
INCOMPLETE_MOTION = MOVEMENT_DIRECTIONS - 1  # 11

# The threshold denominator: quadratic × incomplete motion
THRESHOLD_DENOMINATOR = QUADRATIC * INCOMPLETE_MOTION  # 44

# The threshold numerator: denominator + 1 (the completion that tips over)
THRESHOLD_NUMERATOR = THRESHOLD_DENOMINATOR + 1  # 45

# Emergence threshold ratio - when a new motion function is born
EMERGENCE_THRESHOLD = THRESHOLD_NUMERATOR / THRESHOLD_DENOMINATOR  # 45/44 ≈ 1.02272727...

# Maximum dimension at pure heat (Big Bang)
PLANCK_DIMENSION = 30

# Entropic bound (Ramanujan's regularized sum)
ENTROPIC_BOUND = -1/12

# =============================================================================
# DERIVED: PLANCK TEMPERATURE
# =============================================================================
# T_planck = (K × 12^30 / φ²) × (45/44)
# This is the maximum heat - the moment before Polarity emerges
# The Big Bang was heat at exactly the threshold where differentiation HAD to occur

PLANCK_TEMPERATURE = (K * (MOVEMENT_DIRECTIONS ** PLANCK_DIMENSION) / (PHI ** 2)) * EMERGENCE_THRESHOLD
# ≈ 1.416794 × 10^32 Kelvin (99.999% match to measured value)

# =============================================================================
# Print constants on load
# =============================================================================
print("=" * 60)
print("MOTION CALENDAR CONSTANTS")
print("=" * 60)
print(f"φ (Golden Ratio)      = {PHI}")
print(f"K (Thermal Quantum)   = {K} Kelvin")
print(f"K × φ²                = {K * PHI**2} (= 4)")
print(f"Movement Directions   = {MOVEMENT_DIRECTIONS}")
print(f"Motion Functions      = {MOTION_FUNCTIONS}")
print(f"Quadratic             = {QUADRATIC}")
print(f"Incomplete Motion     = {INCOMPLETE_MOTION} (12 - 1)")
print(f"Threshold Denominator = {THRESHOLD_DENOMINATOR} (4 × 11)")
print(f"Threshold Numerator   = {THRESHOLD_NUMERATOR} (44 + 1)")
print(f"Emergence Threshold   = {EMERGENCE_THRESHOLD} (45/44)")
print(f"Planck Dimension      = {PLANCK_DIMENSION}")
print(f"Entropic Bound        = {ENTROPIC_BOUND}")
print("-" * 60)
print(f"PLANCK TEMPERATURE    = {PLANCK_TEMPERATURE:.6e} Kelvin")
print(f"  Formula: (K × 12^30 / φ²) × (45/44)")
print(f"  Accepted: 1.416808 × 10^32 Kelvin")
print(f"  Accuracy: {(PLANCK_TEMPERATURE / 1.416808e32) * 100:.4f}%")
print("=" * 60)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_scene():
    """Remove all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    # Clear all meshes
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    
    # Clear curves
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)

def frame(seconds):
    """Convert seconds to frame number"""
    return int(seconds * FPS)

def setup_render_settings():
    """Configure render settings for the animation"""
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES
    scene.render.fps = FPS
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    
    # Set render engine - try EEVEE NEXT first, fall back to EEVEE
    if 'BLENDER_EEVEE_NEXT' in dir(bpy.types):
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    else:
        scene.render.engine = 'BLENDER_EEVEE'
    
    # Set background to pure black
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    
    world = bpy.context.scene.world
    world.use_nodes = True
    
    if world.node_tree:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = (0, 0, 0, 1)
            bg_node.inputs['Strength'].default_value = 0

def create_emission_material(name, color, strength=5.0):
    """Create a glowing emission material"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create emission node
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1.0)
    emission.inputs['Strength'].default_value = strength
    
    # Create output node
    output = nodes.new('ShaderNodeOutputMaterial')
    
    # Link them
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return mat

def create_animated_color_material(name):
    """Create a material with animated swirling dark colors (yellow, red, blue)"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    nodes.clear()
    
    # Texture coordinate for animation
    tex_coord = nodes.new('ShaderNodeTexCoord')
    
    # Mapping node for animation
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (2, 2, 2)
    
    # Wave texture for swirling effect
    wave1 = nodes.new('ShaderNodeTexWave')
    wave1.wave_type = 'RINGS'
    wave1.inputs['Scale'].default_value = 5.0
    wave1.inputs['Distortion'].default_value = 10.0
    
    # Color ramp for dark primary colors
    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.3, 0.1, 0.0, 1.0)  # Dark red/yellow
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.3, 1.0)  # Dark blue
    
    # Add middle element for dark yellow
    elem = color_ramp.color_ramp.elements.new(0.5)
    elem.color = (0.3, 0.2, 0.0, 1.0)  # Dark yellow
    
    # Emission for glow
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 2.0
    
    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    
    # Link nodes
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], wave1.inputs['Vector'])
    links.new(wave1.outputs['Color'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return mat, mapping

def create_text_object(text, location, scale=1.0, material=None):
    """Create a 3D text object"""
    bpy.ops.object.text_add(location=location)
    text_obj = bpy.context.active_object
    text_obj.data.body = text
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'
    text_obj.scale = (scale, scale, scale)
    
    # Extrude slightly for 3D effect
    text_obj.data.extrude = 0.05
    
    if material:
        text_obj.data.materials.append(material)
    
    return text_obj

def keyframe_visibility(obj, visible, frame_num):
    """Keyframe object visibility"""
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
    obj.keyframe_insert(data_path="hide_render", frame=frame_num)

def keyframe_location(obj, location, frame_num):
    """Keyframe object location"""
    obj.location = location
    obj.keyframe_insert(data_path="location", frame=frame_num)

def keyframe_scale(obj, scale, frame_num):
    """Keyframe object scale"""
    if isinstance(scale, (int, float)):
        obj.scale = (scale, scale, scale)
    else:
        obj.scale = scale
    obj.keyframe_insert(data_path="scale", frame=frame_num)

def keyframe_rotation(obj, rotation, frame_num):
    """Keyframe object rotation (Euler angles in radians)"""
    obj.rotation_euler = rotation
    obj.keyframe_insert(data_path="rotation_euler", frame=frame_num)

# =============================================================================
# SCENE CREATION FUNCTIONS
# =============================================================================

def create_heat_particle():
    """Create the initial spinning heat dot - representing pure Heat at the Big Bang"""
    # Create UV sphere for the dot
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 0, 0), segments=32, ring_count=16)
    dot = bpy.context.active_object
    dot.name = "HeatParticle"
    
    # Apply animated material
    mat, mapping_node = create_animated_color_material("HeatMaterial")
    dot.data.materials.append(mat)
    
    # Animate the mapping node for swirling effect
    if mapping_node:
        for sec in range(0, TOTAL_SECONDS + 1):
            f = frame(sec)
            mapping_node.inputs['Rotation'].default_value = (0, 0, sec * 0.5)
            mapping_node.inputs['Rotation'].keyframe_insert(data_path="default_value", frame=f)
    
    # Animate spinning (continuous rotation)
    for sec in range(0, TOTAL_SECONDS + 1):
        f = frame(sec)
        angle = sec * math.pi * 4  # Fast spinning
        dot.rotation_euler = (angle, angle * 0.7, angle * 0.3)
        dot.keyframe_insert(data_path="rotation_euler", frame=f)
    
    # Create Planck temperature label - appears briefly when zoomed in
    temp_mat = create_emission_material("TempLabelMaterial", (1, 0.5, 0.2), 3.0)
    temp_label = create_text_object("T = 1.416 × 10³² K", (0, 0, -0.8), 0.15, temp_mat)
    temp_label.name = "PlanckTempLabel"
    temp_label.rotation_euler = (math.pi/2, 0, 0)
    
    # Temperature label only visible during close-up (frames for 2-5 seconds)
    keyframe_visibility(temp_label, False, frame(0))
    keyframe_visibility(temp_label, True, frame(2.5))
    keyframe_visibility(temp_label, False, frame(5.5))
    
    # Also create the formula label
    formula_mat = create_emission_material("FormulaLabelMaterial", (0.8, 0.8, 1.0), 2.0)
    formula_label = create_text_object("K×12³⁰/φ² × 45/44", (0, 0, -1.1), 0.1, formula_mat)
    formula_label.name = "FormulaLabel"
    formula_label.rotation_euler = (math.pi/2, 0, 0)
    
    keyframe_visibility(formula_label, False, frame(0))
    keyframe_visibility(formula_label, True, frame(3))
    keyframe_visibility(formula_label, False, frame(5.5))
    
    return dot

def create_camera_animation(heat_particle):
    """Create and animate the camera"""
    # Create camera
    bpy.ops.object.camera_add(location=(0, -10, 0))
    camera = bpy.context.active_object
    camera.name = "MainCamera"
    bpy.context.scene.camera = camera
    
    # Point camera at origin
    camera.rotation_euler = (math.pi/2, 0, 0)
    
    # Scene 1: Heat - Start far, zoom in, zoom out
    keyframe_location(camera, (0, -10, 0), frame(0))
    keyframe_rotation(camera, (math.pi/2, 0, 0), frame(0))
    
    # Zoom in close to see the spinning colors
    keyframe_location(camera, (0, -1.5, 0), frame(2))
    
    # Stay close
    keyframe_location(camera, (0, -1.5, 0), frame(5))
    
    # Pull back
    keyframe_location(camera, (0, -10, 0), frame(7))
    
    # Scene 2: Polarity - Pan along the line
    keyframe_location(camera, (0, -10, 0), frame(8))
    
    # Move toward positive end
    keyframe_location(camera, (15, -10, 0), frame(10))
    
    # Second dot appears, swing back
    keyframe_location(camera, (0, -15, 0), frame(12))
    keyframe_location(camera, (0, -20, 2), frame(14))
    
    # Scene 3: Existence - Pull back to see golden spiral
    keyframe_location(camera, (0, -25, 5), frame(16))
    keyframe_location(camera, (0, -30, 5), frame(21))
    
    # Scene 4: Righteousness - View the quadratic plane
    keyframe_location(camera, (0, -30, 10), frame(23))
    keyframe_rotation(camera, (math.pi/3, 0, 0), frame(23))
    keyframe_location(camera, (0, -30, 10), frame(28))
    
    # Scene 5: Order - Same view for number filling
    keyframe_location(camera, (0, -30, 10), frame(35))
    
    # Scene 6: Movement - Pull back to see 3D spacetime
    keyframe_location(camera, (5, -40, 20), frame(38))
    keyframe_rotation(camera, (math.pi/4, 0, math.pi/8), frame(38))
    keyframe_location(camera, (10, -45, 25), frame(45))
    
    return camera

def create_polarity_line(heat_particle):
    """Create the polarity line stretching from the heat particle"""
    # Create a cylinder for the line
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1, location=(0, 0, 0))
    line = bpy.context.active_object
    line.name = "PolarityLine"
    line.rotation_euler = (0, math.pi/2, 0)  # Rotate to lie along X axis
    
    # Glow material
    mat = create_emission_material("PolarityMaterial", (1.0, 1.0, 1.0), 3.0)
    line.data.materials.append(mat)
    
    # Start invisible
    keyframe_visibility(line, False, frame(0))
    keyframe_scale(line, (0.01, 0.01, 0.01), frame(0))
    
    # Appear and stretch at polarity start
    keyframe_visibility(line, True, frame(SCENE_POLARITY_START))
    keyframe_scale(line, (0.01, 0.01, 0.01), frame(SCENE_POLARITY_START))
    
    # Stretch to full length
    keyframe_scale(line, (1, 1, 50), frame(SCENE_POLARITY_START + 2))
    keyframe_scale(line, (1, 1, 100), frame(SCENE_POLARITY_END))
    
    # Create + and - labels
    text_mat = create_emission_material("TextMaterial", (1.0, 1.0, 1.0), 5.0)
    
    plus_text = create_text_object("+", (5, 0, 1), 2.0, text_mat)
    plus_text.name = "PlusLabel"
    plus_text.rotation_euler = (math.pi/2, 0, 0)  # Face camera
    keyframe_visibility(plus_text, False, frame(0))
    keyframe_visibility(plus_text, True, frame(SCENE_POLARITY_START + 1))
    keyframe_location(plus_text, (5, 0, 1), frame(SCENE_POLARITY_START + 1))
    keyframe_location(plus_text, (50, 0, 1), frame(SCENE_POLARITY_END))
    
    minus_text = create_text_object("−", (-5, 0, 1), 2.0, text_mat)
    minus_text.name = "MinusLabel"
    minus_text.rotation_euler = (math.pi/2, 0, 0)  # Face camera
    keyframe_visibility(minus_text, False, frame(0))
    keyframe_visibility(minus_text, True, frame(SCENE_POLARITY_START + 1))
    keyframe_location(minus_text, (-5, 0, 1), frame(SCENE_POLARITY_START + 1))
    keyframe_location(minus_text, (-50, 0, 1), frame(SCENE_POLARITY_END))
    
    return line, plus_text, minus_text

def create_second_dot():
    """Create the existence dot at the midpoint of the polarity line.
    This dot is neutral — neither positive nor negative — and spins at the center."""
    # Place at origin: the exact midpoint of the polarity line
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0, 0, 0))
    dot2 = bpy.context.active_object
    dot2.name = "ExistenceDot"

    # Neutral color (white/silver) — distinguishable from both + and - ends
    mat = create_emission_material("ExistenceDotMaterial", (0.9, 0.9, 1.0), 5.0)
    dot2.data.materials.append(mat)

    # Start invisible
    keyframe_visibility(dot2, False, frame(0))

    # Appear at the end of polarity trailing
    keyframe_visibility(dot2, True, frame(SCENE_POLARITY_START + 4))
    keyframe_scale(dot2, (0.01, 0.01, 0.01), frame(SCENE_POLARITY_START + 4))
    keyframe_scale(dot2, (1, 1, 1), frame(SCENE_POLARITY_START + 5))

    # Anchor at center — existence stays at the midpoint, it does NOT move outward
    keyframe_location(dot2, (0, 0, 0), frame(SCENE_POLARITY_START + 4))
    keyframe_location(dot2, (0, 0, 0), frame(TOTAL_FRAMES))

    # Spinning animation — same as the heat dot
    for sec in range(SCENE_POLARITY_START + 4, TOTAL_SECONDS + 1):
        f = frame(sec)
        angle = sec * math.pi * 4
        dot2.rotation_euler = (angle, angle * 0.7, angle * 0.3)
        dot2.keyframe_insert(data_path="rotation_euler", frame=f)

    return dot2

def create_golden_spiral():
    """Create the golden spiral for existence"""
    # Create spiral using a curve
    curve_data = bpy.data.curves.new('GoldenSpiralCurve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.15
    curve_data.bevel_resolution = 4
    
    spline = curve_data.splines.new('POLY')
    
    # Generate golden spiral points
    num_points = 200
    spline.points.add(num_points - 1)
    
    for i in range(num_points):
        t = i / (num_points - 1) * 4 * math.pi  # 2 full rotations
        r = 0.3 * (PHI ** (t / (2 * math.pi)))  # Golden ratio scaling
        x = r * math.cos(t)
        y = r * math.sin(t)
        z = 0  # 2D spiral
        
        spline.points[i].co = (x, 0, y, 1)  # w=1 for POLY spline
    
    spiral_obj = bpy.data.objects.new('GoldenSpiral', curve_data)
    bpy.context.collection.objects.link(spiral_obj)
    
    # Create gradient material with primary colors
    mat = bpy.data.materials.new(name="SpiralMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Use object info for gradient
    geometry = nodes.new('ShaderNodeNewGeometry')
    
    # Separate XYZ to get position along curve
    separate = nodes.new('ShaderNodeSeparateXYZ')
    
    # Math node to create gradient value
    math_node = nodes.new('ShaderNodeMath')
    math_node.operation = 'ADD'
    
    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (1, 0, 0, 1)  # Red
    
    elem1 = color_ramp.color_ramp.elements.new(0.33)
    elem1.color = (1, 1, 0, 1)  # Yellow
    
    elem2 = color_ramp.color_ramp.elements.new(0.66)
    elem2.color = (0, 0, 1, 1)  # Blue
    
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (1, 0, 1, 1)  # Magenta
    
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 10.0
    
    output = nodes.new('ShaderNodeOutputMaterial')
    
    links.new(geometry.outputs['Position'], separate.inputs['Vector'])
    links.new(separate.outputs['X'], math_node.inputs[0])
    links.new(separate.outputs['Z'], math_node.inputs[1])
    links.new(math_node.outputs['Value'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    spiral_obj.data.materials.append(mat)
    
    # Animation - appear during existence
    keyframe_visibility(spiral_obj, False, frame(0))
    keyframe_visibility(spiral_obj, True, frame(SCENE_EXISTENCE_START))
    keyframe_scale(spiral_obj, (0.01, 0.01, 0.01), frame(SCENE_EXISTENCE_START))
    keyframe_scale(spiral_obj, (5, 5, 5), frame(SCENE_EXISTENCE_START + 3))
    
    return spiral_obj

def create_clock():
    """Create the clock that appears with existence"""
    clock_objects = []
    
    # Create clock face (torus for ring)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=2.0,
        minor_radius=0.15,
        location=(10, 0, 10)
    )
    clock_ring = bpy.context.active_object
    clock_ring.name = "ClockRing"
    clock_ring.rotation_euler = (math.pi/2, 0, 0)
    
    ring_mat = create_emission_material("ClockRingMaterial", (1, 0.8, 0.2), 4.0)
    clock_ring.data.materials.append(ring_mat)
    clock_objects.append(clock_ring)
    
    # Create clock hand
    bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=1.8, location=(10, 0, 10))
    hand = bpy.context.active_object
    hand.name = "ClockHand"
    
    # Move origin to one end
    bpy.context.view_layer.objects.active = hand
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.transform.translate(value=(0, 0, 0.9))
    bpy.ops.object.mode_set(mode='OBJECT')
    
    hand_mat = create_emission_material("HandMaterial", (1, 1, 1), 6.0)
    hand.data.materials.append(hand_mat)
    clock_objects.append(hand)
    
    # Start invisible
    for obj in clock_objects:
        keyframe_visibility(obj, False, frame(0))
        keyframe_visibility(obj, True, frame(SCENE_EXISTENCE_START + 2))
    
    # Animate hand rotation (continuous ticking)
    for sec in range(SCENE_EXISTENCE_START + 2, TOTAL_SECONDS + 1):
        f = frame(sec)
        elapsed = sec - SCENE_EXISTENCE_START - 2
        angle = elapsed * (math.pi / 5)  # One full rotation per 10 seconds
        hand.rotation_euler = (math.pi/2, angle, 0)
        hand.keyframe_insert(data_path="rotation_euler", frame=f)
    
    return clock_objects

def create_righteousness_quadrant():
    """Create the 2D quadratic plane with X and Y axes"""
    objects = []
    text_mat = create_emission_material("AxisTextMaterial", (1, 1, 1), 5.0)
    axis_mat = create_emission_material("AxisMaterial", (0.5, 0.5, 1.0), 3.0)
    
    # X axis
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=50, location=(0, 0, 0))
    x_axis = bpy.context.active_object
    x_axis.name = "XAxis"
    x_axis.rotation_euler = (0, math.pi/2, 0)
    x_axis.data.materials.append(axis_mat)
    objects.append(x_axis)
    
    # Y axis (vertical in the XZ plane for camera view)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=50, location=(0, 0, 0))
    y_axis = bpy.context.active_object
    y_axis.name = "YAxis"
    y_axis.rotation_euler = (0, 0, 0)  # Points up (Z)
    y_axis.data.materials.append(axis_mat)
    objects.append(y_axis)
    
    # X label
    x_label = create_text_object("X", (27, 0, -2), 3.0, text_mat)
    x_label.name = "XLabel"
    x_label.rotation_euler = (math.pi/2, 0, 0)
    objects.append(x_label)
    
    # Y label
    y_label = create_text_object("Y", (2, 0, 27), 3.0, text_mat)
    y_label.name = "YLabel"
    y_label.rotation_euler = (math.pi/2, 0, 0)
    objects.append(y_label)
    
    # Quadrant labels (I, II, III, IV)
    quad_mat = create_emission_material("QuadMaterial", (0.4, 0.4, 0.4), 2.0)
    quadrant_positions = [
        ("I", (12, 0, 12)),
        ("II", (-12, 0, 12)),
        ("III", (-12, 0, -12)),
        ("IV", (12, 0, -12))
    ]
    
    for label, pos in quadrant_positions:
        q_text = create_text_object(label, pos, 4.0, quad_mat)
        q_text.name = f"Quadrant{label}"
        q_text.rotation_euler = (math.pi/2, 0, 0)
        objects.append(q_text)
    
    # Start all invisible
    for obj in objects:
        keyframe_visibility(obj, False, frame(0))
        keyframe_visibility(obj, True, frame(SCENE_RIGHTEOUSNESS_START))
        keyframe_scale(obj, (0.01, 0.01, 0.01), frame(SCENE_RIGHTEOUSNESS_START))
        keyframe_scale(obj, (1, 1, 1), frame(SCENE_RIGHTEOUSNESS_START + 2))
    
    return objects

def create_order_numbers():
    """Create numbers that fill the quadrants during the Order phase"""
    numbers = []
    text_mat = create_emission_material("NumberMaterial", (1, 0.9, 0.3), 6.0)
    
    # Robinson axioms represented as sequential number plotting
    number_sequence = [
        ("0", (0, 0, 0.5)),       # Identity at origin
        ("1", (6, 0, 6)),         # First quadrant
        ("S(0)=1", (6, 0, 4)),    # Successor notation
        ("2", (-6, 0, 6)),        # Second quadrant
        ("S(1)=2", (-6, 0, 4)),
        ("3", (-6, 0, -6)),       # Third quadrant
        ("S(2)=3", (-6, 0, -8)),
        ("4", (6, 0, -6)),        # Fourth quadrant
        ("S(3)=4", (6, 0, -8)),
    ]
    
    start_frame = frame(SCENE_ORDER_START)
    frames_per_number = int((SCENE_ORDER_END - SCENE_ORDER_START) * FPS / len(number_sequence))
    
    for i, (num_text, pos) in enumerate(number_sequence):
        num_obj = create_text_object(num_text, pos, 1.5, text_mat)
        num_obj.name = f"OrderNumber_{i}"
        num_obj.rotation_euler = (math.pi/2, 0, 0)
        
        appear_frame = start_frame + i * frames_per_number
        
        keyframe_visibility(num_obj, False, frame(0))
        keyframe_visibility(num_obj, True, appear_frame)
        keyframe_scale(num_obj, (0.01, 0.01, 0.01), appear_frame)
        keyframe_scale(num_obj, (1, 1, 1), appear_frame + 10)
        
        numbers.append(num_obj)
    
    return numbers

def create_z_axis_and_spacetime(heat_particle):
    """Create the Z axis with polarity and spacetime that expands from the universal center"""
    objects = []
    
    # THE UNIVERSAL CENTER - where all motion functions emerged
    # This is the origin point (0,0,0) - found through motion, not spacetime
    UNIVERSAL_CENTER = (0, 0, 0)
    
    # Z axis extends from the UNIVERSAL CENTER in both directions (+Y and -Y)
    # This conserves the centering established by all previous motion functions
    z_mat = create_emission_material("ZAxisMaterial", (1, 0.5, 0), 4.0)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=1, location=UNIVERSAL_CENTER)
    z_axis = bpy.context.active_object
    z_axis.name = "ZAxis"
    z_axis.rotation_euler = (math.pi/2, 0, 0)  # Points in Y direction (depth)
    z_axis.data.materials.append(z_mat)
    objects.append(z_axis)
    
    # Z axis labels with POLARITY (+Z and -Z) - extending from center
    text_mat_plus = create_emission_material("ZLabelPlusMaterial", (1, 0.6, 0.2), 5.0)
    text_mat_minus = create_emission_material("ZLabelMinusMaterial", (1, 0.4, 0.1), 5.0)
    
    z_plus_label = create_text_object("+Z", (2, 3, 0), 2.0, text_mat_plus)
    z_plus_label.name = "ZPlusLabel"
    z_plus_label.rotation_euler = (math.pi/2, 0, 0)
    objects.append(z_plus_label)
    
    z_minus_label = create_text_object("-Z", (2, -3, 0), 2.0, text_mat_minus)
    z_minus_label.name = "ZMinusLabel"
    z_minus_label.rotation_euler = (math.pi/2, 0, 0)
    objects.append(z_minus_label)
    
    # Create spacetime cube edges - CENTERED ON UNIVERSAL CENTER
    # Spacetime emerges from motion, centered on the origin
    spacetime_mat = create_emission_material("SpacetimeMaterial", (0.2, 0.5, 1.0), 3.0)
    
    # Container for spacetime - centered at universal center
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=UNIVERSAL_CENTER)
    spacetime_container = bpy.context.active_object
    spacetime_container.name = "SpacetimeContainer"
    objects.append(spacetime_container)
    
    # Create cube edges for a unit cube centered at origin
    # These will scale with the container, always centered on universal center
    edge_definitions = [
        # Bottom face (Y=-1 in local space)
        ((-1, -1, -1), (1, -1, -1)),   # along X
        ((-1, -1, 1), (1, -1, 1)),     # along X
        ((-1, -1, -1), (-1, -1, 1)),   # along Z
        ((1, -1, -1), (1, -1, 1)),     # along Z
        # Top face (Y=1 in local space)
        ((-1, 1, -1), (1, 1, -1)),     # along X
        ((-1, 1, 1), (1, 1, 1)),       # along X
        ((-1, 1, -1), (-1, 1, 1)),     # along Z
        ((1, 1, -1), (1, 1, 1)),       # along Z
        # Vertical edges (along Y) - these are the Z-dimension in our view
        ((-1, -1, -1), (-1, 1, -1)),
        ((1, -1, -1), (1, 1, -1)),
        ((-1, -1, 1), (-1, 1, 1)),
        ((1, -1, 1), (1, 1, 1)),
    ]
    
    for i, (start, end) in enumerate(edge_definitions):
        mid = ((start[0]+end[0])/2, (start[1]+end[1])/2, (start[2]+end[2])/2)
        length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2 + (end[2]-start[2])**2)
        
        bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=length, location=mid)
        edge = bpy.context.active_object
        edge.name = f"SpacetimeEdge_{i}"
        
        # Calculate rotation to align with edge direction
        direction = Vector((end[0]-start[0], end[1]-start[1], end[2]-start[2])).normalized()
        up = Vector((0, 0, 1))
        if abs(direction.dot(up)) > 0.999:
            up = Vector((1, 0, 0))
        rotation = direction.to_track_quat('Z', 'Y').to_euler()
        edge.rotation_euler = rotation
        
        edge.data.materials.append(spacetime_mat)
        edge.parent = spacetime_container
        objects.append(edge)
    
    # Start everything invisible
    for obj in objects:
        keyframe_visibility(obj, False, frame(0))
    
    # === MOVEMENT PHASE ANIMATION ===
    # Z-axis appears IMMEDIATELY when movement phase begins
    # It extends from the universal center in both directions
    
    z_axis_appear_frame = frame(SCENE_MOVEMENT_START)
    
    # Z-axis appears and stretches from center outward (both +Z and -Z)
    keyframe_visibility(z_axis, True, z_axis_appear_frame)
    keyframe_scale(z_axis, (1, 1, 0.1), z_axis_appear_frame)
    
    # Z labels appear with axis
    keyframe_visibility(z_plus_label, True, z_axis_appear_frame)
    keyframe_visibility(z_minus_label, True, z_axis_appear_frame)
    
    # Labels start at center and move outward as Z extends
    keyframe_location(z_plus_label, (2, 0.5, 0), z_axis_appear_frame)
    keyframe_location(z_minus_label, (2, -0.5, 0), z_axis_appear_frame)
    
    # Spacetime appears immediately after Z-axis begins
    # It starts as a point at the universal center and expands
    spacetime_appear_frame = z_axis_appear_frame + 10
    
    keyframe_visibility(spacetime_container, True, spacetime_appear_frame)
    for edge in [obj for obj in objects if "SpacetimeEdge" in obj.name]:
        keyframe_visibility(edge, True, spacetime_appear_frame)
    
    # Spacetime starts as essentially a point
    spacetime_container.scale = (0.01, 0.01, 0.01)
    spacetime_container.keyframe_insert(data_path="scale", frame=spacetime_appear_frame)
    
    # === PARTICLE MOTION AND COORDINATED EXPANSION ===
    # The particle gains movement capability
    # Spacetime and Z-axis expand to accommodate ALL the motion
    # Everything stays centered on universal origin
    
    # Particle starts at universal center
    keyframe_location(heat_particle, UNIVERSAL_CENTER, frame(SCENE_MOVEMENT_START))
    
    # Define the motion journey - spacetime expands to accommodate each move
    # Format: (particle_position, frame, z_axis_scale, spacetime_scale)
    motion_journey = [
        # Particle begins to move - spacetime must exist to accommodate
        ((0, 0, 0), z_axis_appear_frame, 0.1, 0.01),
        # First motion - small expansion
        ((2, 3, 2), z_axis_appear_frame + 30, 5, 3),
        # Second motion - spacetime grows
        ((-3, -2, 3), z_axis_appear_frame + 60, 7, 5),
        # Third motion - further expansion
        ((4, 5, -3), z_axis_appear_frame + 90, 10, 7),
        # Fourth motion - spacetime bends to capability
        ((-4, -5, -4), z_axis_appear_frame + 120, 12, 9),
        # Fifth motion - approaching φ scaling
        ((5, 6, 5), z_axis_appear_frame + 150, 14, 11),
        # Sixth motion - golden ratio expansion
        ((-5, -6, 5), z_axis_appear_frame + 180, 14 * PHI / 2, 11 * PHI / 2),
        # Seventh motion - continuing φ growth
        ((6, 7, -6), z_axis_appear_frame + 210, 14 * PHI, 11 * PHI),
        # Final state
        ((4, -4, 4), frame(SCENE_MOVEMENT_END), 14 * PHI * 1.2, 11 * PHI * 1.2),
    ]
    
    for pos, f, z_scale, st_scale in motion_journey:
        # Particle moves
        keyframe_location(heat_particle, pos, f)
        
        # Z-axis expands (from center, both directions)
        z_axis.scale = (1, 1, z_scale)
        z_axis.keyframe_insert(data_path="scale", frame=f)
        
        # Z labels move outward with the axis
        z_plus_label.location = (2, z_scale * 0.5 + 1, 0)
        z_plus_label.keyframe_insert(data_path="location", frame=f)
        z_minus_label.location = (2, -z_scale * 0.5 - 1, 0)
        z_minus_label.keyframe_insert(data_path="location", frame=f)
        
        # Spacetime expands (centered on universal origin)
        spacetime_container.scale = (st_scale, st_scale, st_scale)
        spacetime_container.keyframe_insert(data_path="scale", frame=f)
    
    return objects

def create_speed_of_light_label():
    """Create optional 'c' label for particle spawning"""
    text_mat = create_emission_material("CMaterial", (1, 1, 0.5), 4.0)
    c_label = create_text_object("c", (15, 15, 0), 2.0, text_mat)
    c_label.name = "SpeedOfLight"
    c_label.rotation_euler = (math.pi/2, 0, 0)
    
    keyframe_visibility(c_label, False, frame(0))
    keyframe_visibility(c_label, True, frame(SCENE_MOVEMENT_START + 5))
    
    return c_label

def add_glow_effect():
    """Add bloom/glow effect if available"""
    scene = bpy.context.scene
    
    # Try EEVEE bloom (Blender 3.x - 4.x)
    try:
        if hasattr(scene, 'eevee') and hasattr(scene.eevee, 'use_bloom'):
            scene.eevee.use_bloom = True
            scene.eevee.bloom_threshold = 0.3
            scene.eevee.bloom_intensity = 0.8
            scene.eevee.bloom_radius = 6.0
            print("  Using EEVEE bloom effect")
            return
    except:
        pass
    
    # Blender 5.0+ doesn't have bloom in EEVEE the same way
    # The emission materials will still glow - just no extra bloom post-process
    print("  Bloom not available in this Blender version - using emission materials only")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to create the entire animation"""
    print("="*60)
    print("Starting Motion Calendar Simulation Creation...")
    print("="*60)
    
    # Clear the scene
    clear_scene()
    print("✓ Scene cleared")
    
    # Setup render settings
    setup_render_settings()
    print("✓ Render settings configured")
    
    # Create the heat particle (central dot)
    heat_particle = create_heat_particle()
    print("✓ Heat particle created")
    
    # Create camera animation
    camera = create_camera_animation(heat_particle)
    print("✓ Camera created and animated")
    
    # Create polarity line
    line, plus, minus = create_polarity_line(heat_particle)
    print("✓ Polarity line created")
    
    # Create second dot
    second_dot = create_second_dot()
    print("✓ Second dot created")
    
    # Golden spiral removed - doesn't add to the visualization
    
    # Clock removed - doesn't add to the visualization
    
    # Create righteousness quadrant
    quadrant_objects = create_righteousness_quadrant()
    print("✓ Righteousness quadrant created")
    
    # Create order numbers
    numbers = create_order_numbers()
    print("✓ Order numbers created")
    
    # Create Z axis and spacetime
    movement_objects = create_z_axis_and_spacetime(heat_particle)
    print("✓ Movement/spacetime created")
    
    # Create speed of light label
    c_label = create_speed_of_light_label()
    print("✓ Speed of light label created")
    
    # Add glow effect
    add_glow_effect()
    print("✓ Glow effect added")
    
    # Set current frame to start
    bpy.context.scene.frame_set(1)
    
    print("\n" + "="*60)
    print("MOTION CALENDAR SIMULATION COMPLETE!")
    print("="*60)
    print(f"Total Duration: {TOTAL_SECONDS} seconds ({TOTAL_FRAMES} frames)")
    print(f"Resolution: 1920x1080 @ {FPS} FPS")
    print("\nScene Breakdown:")
    print(f"  Heat:          {SCENE_HEAT_START}-{SCENE_HEAT_END}s")
    print(f"  Polarity:      {SCENE_POLARITY_START}-{SCENE_POLARITY_END}s")
    print(f"  Existence:     {SCENE_EXISTENCE_START}-{SCENE_EXISTENCE_END}s")
    print(f"  Righteousness: {SCENE_RIGHTEOUSNESS_START}-{SCENE_RIGHTEOUSNESS_END}s")
    print(f"  Order:         {SCENE_ORDER_START}-{SCENE_ORDER_END}s")
    print(f"  Movement:      {SCENE_MOVEMENT_START}-{SCENE_MOVEMENT_END}s")
    print("\n► To preview: Press SPACEBAR in the viewport")
    print("► To render:  Render > Render Animation (Ctrl+F12)")
    print("="*60)

# Run the script
if __name__ == "__main__":
    main()