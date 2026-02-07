"""
Motion Calendar Simulation - Particle Formation & SU Group Emergence
=====================================================================

A Blender Python script creating a 16-second (480 frame) animation visualizing:

Phase 1 (1-120):   Heat → Polarity → Existence → Wireframe Cube
Phase 2 (121-180): Existence dot vibrates → sphere morphs to icosahedron (Order)
Phase 3 (181-240): Polarity colors appear → icosahedron morphs to torus (SU(2))
Phase 4 (241-320): Heat pulses → 45/44 split → two children diverge
Phase 5 (321-480): More splits → collision event → multiple particles

To run: Open Blender, go to Scripting tab, paste this script, and click Run.
Then render animation: Render > Render Animation (or Ctrl+F12)

Author: Based on "Ocean From Motion" - The Motion Calendar framework
"""

import bpy
import bmesh
import math
import mathutils
from mathutils import Vector, Euler

# =============================================================================
# CONFIGURATION
# =============================================================================

FPS = 30
TOTAL_FRAMES = 480  # 16 seconds

# Phase frame boundaries
P1_START, P1_END = 1, 120    # Compressed existing: Heat, Polarity, Existence, Cube
P2_START, P2_END = 121, 180  # Order emergence: vibration → icosahedron
P3_START, P3_END = 181, 240  # SU group emergence: polarity colors → torus
P4_START, P4_END = 241, 320  # First 45/44 split
P5_START, P5_END = 321, 480  # Cascade splits + collision

# Sub-phase boundaries within Phase 1
P1_HEAT_START, P1_HEAT_END = 1, 30
P1_POLARITY_START, P1_POLARITY_END = 31, 60
P1_EXISTENCE_START, P1_EXISTENCE_END = 61, 90
P1_CUBE_START, P1_CUBE_END = 91, 120

# =============================================================================
# MOTION CALENDAR CONSTANTS
# =============================================================================

# Golden ratio - the self-similar scaling law
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895

# Thermal quantum - the fundamental unit of heat (in Kelvin)
K = 4 / (PHI ** 2)  # ≈ 1.527864045000421 Kelvin

# Movement constant - total primitive directional operators
MOVEMENT_DIRECTIONS = 12
MOTION_FUNCTIONS = 6
QUADRATIC = 4
INCOMPLETE_MOTION = MOVEMENT_DIRECTIONS - 1  # 11

THRESHOLD_DENOMINATOR = QUADRATIC * INCOMPLETE_MOTION  # 44
THRESHOLD_NUMERATOR = THRESHOLD_DENOMINATOR + 1  # 45
EMERGENCE_THRESHOLD = THRESHOLD_NUMERATOR / THRESHOLD_DENOMINATOR  # 45/44

PLANCK_DIMENSION = 30
ENTROPIC_BOUND = -1/12

PLANCK_TEMPERATURE = (K * (MOVEMENT_DIRECTIONS ** PLANCK_DIMENSION) / (PHI ** 2)) * EMERGENCE_THRESHOLD

# Phi-power threshold ladder for motion function emergence
THRESHOLD_HEAT = PHI ** 0       # 1.0
THRESHOLD_POLARITY = PHI ** 1   # φ
THRESHOLD_EXISTENCE = PHI ** 2  # φ²
THRESHOLD_ORDER = PHI ** 3      # φ³
THRESHOLD_MOVEMENT = PHI ** 4   # φ⁴

# Polarity color assignments per quadrant: (positive_color, negative_color)
POLARITY_COLORS = {
    'Q1': ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),     # Red / Cyan
    'Q2': ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0)),     # Green / Magenta
    'Q3': ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0)),     # Blue / Yellow
    'Q4': ((1.0, 1.0, 1.0), (0.05, 0.05, 0.05)),  # White / Black
}

# Default polarity state: +1 or -1 per quadrant
DEFAULT_POLARITY = (1, 1, -1, 1)

# =============================================================================
# Print constants on load
# =============================================================================
print("=" * 60)
print("MOTION CALENDAR CONSTANTS")
print("=" * 60)
print(f"PHI (Golden Ratio)      = {PHI}")
print(f"K (Thermal Quantum)     = {K} Kelvin")
print(f"K x PHI^2               = {K * PHI**2} (= 4)")
print(f"Movement Directions     = {MOVEMENT_DIRECTIONS}")
print(f"Motion Functions        = {MOTION_FUNCTIONS}")
print(f"Quadratic               = {QUADRATIC}")
print(f"Incomplete Motion       = {INCOMPLETE_MOTION} (12 - 1)")
print(f"Threshold Denominator   = {THRESHOLD_DENOMINATOR} (4 x 11)")
print(f"Threshold Numerator     = {THRESHOLD_NUMERATOR} (44 + 1)")
print(f"Emergence Threshold     = {EMERGENCE_THRESHOLD} (45/44)")
print(f"Planck Dimension        = {PLANCK_DIMENSION}")
print(f"Entropic Bound          = {ENTROPIC_BOUND}")
print("-" * 60)
print(f"PLANCK TEMPERATURE      = {PLANCK_TEMPERATURE:.6e} Kelvin")
print(f"  Formula: (K x 12^30 / PHI^2) x (45/44)")
print(f"  Accepted: 1.416808 x 10^32 Kelvin")
print(f"  Accuracy: {(PLANCK_TEMPERATURE / 1.416808e32) * 100:.4f}%")
print("=" * 60)

# =============================================================================
# UTILITY FUNCTIONS (kept from original)
# =============================================================================

def clear_scene():
    """Remove all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)


def setup_render_settings():
    """Configure render settings for the animation"""
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES
    scene.render.fps = FPS
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    # Blender 4.2+ renamed EEVEE to BLENDER_EEVEE_NEXT; 4.x uses BLENDER_EEVEE
    eevee_set = False
    for engine_id in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        try:
            scene.render.engine = engine_id
            eevee_set = True
            break
        except TypeError:
            continue
    if not eevee_set:
        scene.render.engine = 'CYCLES'

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
    nodes.clear()

    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1.0)
    emission.inputs['Strength'].default_value = strength

    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat


def create_animated_color_material(name):
    """Create a material with animated swirling dark colors"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex_coord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (2, 2, 2)

    wave1 = nodes.new('ShaderNodeTexWave')
    wave1.wave_type = 'RINGS'
    wave1.inputs['Scale'].default_value = 5.0
    wave1.inputs['Distortion'].default_value = 10.0

    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.3, 0.1, 0.0, 1.0)
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.3, 1.0)
    elem = color_ramp.color_ramp.elements.new(0.5)
    elem.color = (0.3, 0.2, 0.0, 1.0)

    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 2.0

    output = nodes.new('ShaderNodeOutputMaterial')

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


def add_glow_effect():
    """Add bloom/glow effect if available"""
    scene = bpy.context.scene
    try:
        if hasattr(scene, 'eevee') and hasattr(scene.eevee, 'use_bloom'):
            scene.eevee.use_bloom = True
            scene.eevee.bloom_threshold = 0.3
            scene.eevee.bloom_intensity = 0.8
            scene.eevee.bloom_radius = 6.0
            print("  Using EEVEE bloom effect")
            return
    except Exception:
        pass
    print("  Bloom not available - using emission materials only")


# =============================================================================
# MESH CREATION HELPERS
# =============================================================================

def create_wireframe_cube(location, scale_val, name_prefix="Cube"):
    """Create a wireframe cube from cylinder edges. Returns (container_empty, edge_list)."""
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=location)
    container = bpy.context.active_object
    container.name = f"{name_prefix}_Container"

    cube_mat = create_emission_material(f"{name_prefix}Mat", (0.2, 0.5, 1.0), 3.0)

    edge_definitions = [
        # Bottom face (Y=-1)
        ((-1, -1, -1), (1, -1, -1)),
        ((-1, -1, 1), (1, -1, 1)),
        ((-1, -1, -1), (-1, -1, 1)),
        ((1, -1, -1), (1, -1, 1)),
        # Top face (Y=1)
        ((-1, 1, -1), (1, 1, -1)),
        ((-1, 1, 1), (1, 1, 1)),
        ((-1, 1, -1), (-1, 1, 1)),
        ((1, 1, -1), (1, 1, 1)),
        # Vertical edges (along Y)
        ((-1, -1, -1), (-1, 1, -1)),
        ((1, -1, -1), (1, 1, -1)),
        ((-1, -1, 1), (-1, 1, 1)),
        ((1, -1, 1), (1, 1, 1)),
    ]

    edges = []
    for i, (start, end) in enumerate(edge_definitions):
        mid = ((start[0]+end[0])/2, (start[1]+end[1])/2, (start[2]+end[2])/2)
        length = math.sqrt(sum((e-s)**2 for s, e in zip(start, end)))

        bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=length, location=mid)
        edge = bpy.context.active_object
        edge.name = f"{name_prefix}_Edge_{i}"

        direction = Vector((end[0]-start[0], end[1]-start[1], end[2]-start[2])).normalized()
        up = Vector((0, 0, 1))
        if abs(direction.dot(up)) > 0.999:
            up = Vector((1, 0, 0))
        rotation = direction.to_track_quat('Z', 'Y').to_euler()
        edge.rotation_euler = rotation

        edge.data.materials.append(cube_mat)
        edge.parent = container
        edges.append(edge)

    container.scale = (scale_val, scale_val, scale_val)
    return container, edges


def create_icosahedron(location, radius, name="Icosahedron"):
    """Create a true icosahedron (20 faces) via ico_sphere with 1 subdivision."""
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=radius, location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def create_torus_particle(location, major_r, minor_r, name="TorusParticle"):
    """Create a torus mesh for SU(2) representation."""
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_r,
        minor_radius=minor_r,
        location=location
    )
    obj = bpy.context.active_object
    obj.name = name
    return obj


def create_triangular_prism(location, radius, height, name="TriPrism"):
    """Create a triangular prism via bmesh for SU(3) representation."""
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    # Bottom triangle vertices
    bottom_verts = []
    top_verts = []
    for i in range(3):
        angle = i * (2 * math.pi / 3) - math.pi / 2
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        bottom_verts.append(bm.verts.new((x, y, -height / 2)))
        top_verts.append(bm.verts.new((x, y, height / 2)))

    bm.faces.new(bottom_verts)
    bm.faces.new(top_verts)
    for i in range(3):
        j = (i + 1) % 3
        bm.faces.new([bottom_verts[i], bottom_verts[j], top_verts[j], top_verts[i]])

    bm.to_mesh(mesh)
    bm.free()

    obj.location = location
    return obj


def apply_quadrant_colors(obj, polarity_state):
    """Assign 4 emission materials to faces based on angle from center.
    polarity_state is a tuple of 4 values (+1 or -1), one per quadrant."""
    quadrant_keys = ['Q1', 'Q2', 'Q3', 'Q4']
    materials = []
    for i, qk in enumerate(quadrant_keys):
        sign = polarity_state[i]
        color = POLARITY_COLORS[qk][0] if sign > 0 else POLARITY_COLORS[qk][1]
        mat = create_emission_material(f"{obj.name}_{qk}_mat", color, 4.0)
        obj.data.materials.append(mat)
        materials.append(mat)

    # Assign faces to quadrants by angle of face center
    mesh = obj.data
    mesh.update()
    for poly in mesh.polygons:
        center = poly.center
        # Use atan2 of (z, x) relative to object center
        angle = math.atan2(center.z, center.x)
        if angle >= 0 and angle < math.pi / 2:
            poly.material_index = 0  # Q1
        elif angle >= math.pi / 2 or angle < -math.pi:
            poly.material_index = 1  # Q2
        elif angle >= -math.pi and angle < -math.pi / 2:
            poly.material_index = 2  # Q3
        else:
            poly.material_index = 3  # Q4

    mesh.update()
    return materials


def morph_shapes(old_obj, new_obj, start_frame, duration=15):
    """Cross-fade between two shapes: old shrinks out, new grows in."""
    # Old object: visible and full-size at start, shrinks to 0 and hides
    keyframe_visibility(old_obj, True, start_frame)
    keyframe_scale(old_obj, 1.0, start_frame)
    keyframe_scale(old_obj, 0.01, start_frame + duration)
    keyframe_visibility(old_obj, False, start_frame + duration + 1)

    # New object: hidden, then appears and grows from 0 to full-size
    keyframe_visibility(new_obj, False, start_frame - 1)
    keyframe_visibility(new_obj, True, start_frame)
    keyframe_scale(new_obj, 0.01, start_frame)
    keyframe_scale(new_obj, 1.0, start_frame + duration)


def create_split_flash(location, frame_num, duration=5):
    """Create a brief bright emission sphere for split/collision events."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=location, segments=16, ring_count=8)
    flash = bpy.context.active_object
    flash.name = f"Flash_{frame_num}"

    mat = create_emission_material(f"FlashMat_{frame_num}", (1.0, 1.0, 0.8), 20.0)
    flash.data.materials.append(mat)

    # Hidden by default
    keyframe_visibility(flash, False, 1)

    # Flash sequence: appear, grow, shrink, disappear
    keyframe_visibility(flash, True, frame_num)
    keyframe_scale(flash, 0.1, frame_num)
    keyframe_scale(flash, 1.5, frame_num + duration // 2)
    keyframe_scale(flash, 0.1, frame_num + duration)
    keyframe_visibility(flash, False, frame_num + duration + 1)

    return flash


def create_particle(location, heat, polarity_state, group_type, name):
    """Factory: creates a particle shape by group type, scales by heat, applies colors.
    group_type: 'sphere', 'icosahedron', 'su2' (torus), 'su3' (triangular prism)
    """
    heat_scale = 0.3 + (heat / THRESHOLD_NUMERATOR) * 0.5

    if group_type == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=heat_scale, location=location, segments=24, ring_count=12)
        obj = bpy.context.active_object
        obj.name = name
    elif group_type == 'icosahedron':
        obj = create_icosahedron(location, heat_scale, name)
    elif group_type == 'su2':
        obj = create_torus_particle(location, heat_scale * 1.2, heat_scale * 0.35, name)
    elif group_type == 'su3':
        obj = create_triangular_prism(location, heat_scale * 0.8, heat_scale * 1.5, name)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=heat_scale, location=location)
        obj = bpy.context.active_object
        obj.name = name

    apply_quadrant_colors(obj, polarity_state)
    return obj


def split_particle(parent_obj, parent_heat, parent_polarity, parent_direction,
                   split_frame, name_prefix):
    """Perform a 45/44 split: hide parent, create two children with diverging trajectories.

    Parent at heat=45 crosses the threshold. Drops to 44. Child born at 44.
    Directions diverge +/-22.5 deg from parent direction.
    Parent stays as SU(2) torus, child spawns as SU(3) triangular prism.
    Returns (child_a, child_b, child_a_data, child_b_data).
    """
    loc = parent_obj.location.copy()

    # Flash at split point
    create_split_flash(loc, split_frame, duration=5)

    # Hide parent
    keyframe_visibility(parent_obj, True, split_frame - 1)
    keyframe_visibility(parent_obj, False, split_frame)

    # Child A: inherits parent shape (SU(2) torus), heat = 44
    child_a = create_particle(
        location=loc,
        heat=THRESHOLD_DENOMINATOR,
        polarity_state=parent_polarity,
        group_type='su2',
        name=f"{name_prefix}_A"
    )

    # Child B: SU(3) triangular prism, heat = 44
    # Flip one quadrant's polarity for variety
    child_b_polarity = (
        parent_polarity[0],
        -parent_polarity[1],
        parent_polarity[2],
        parent_polarity[3],
    )
    child_b = create_particle(
        location=loc,
        heat=THRESHOLD_DENOMINATOR,
        polarity_state=child_b_polarity,
        group_type='su3',
        name=f"{name_prefix}_B"
    )

    # Both start hidden, appear at split frame
    keyframe_visibility(child_a, False, 1)
    keyframe_visibility(child_a, True, split_frame)
    keyframe_visibility(child_b, False, 1)
    keyframe_visibility(child_b, True, split_frame)

    # Start at parent location
    keyframe_location(child_a, loc, split_frame)
    keyframe_location(child_b, loc, split_frame)

    # Diverging trajectories: +/-22.5 degrees from parent direction
    diverge_angle = math.radians(22.5)
    speed = 0.08  # units per frame

    dir_a_angle = parent_direction + diverge_angle
    dir_b_angle = parent_direction - diverge_angle

    child_a_data = {
        'heat': THRESHOLD_DENOMINATOR,
        'polarity': parent_polarity,
        'direction': dir_a_angle,
        'speed': speed,
    }
    child_b_data = {
        'heat': THRESHOLD_DENOMINATOR,
        'polarity': child_b_polarity,
        'direction': dir_b_angle,
        'speed': speed,
    }

    return child_a, child_b, child_a_data, child_b_data


# =============================================================================
# PHASE FUNCTIONS
# =============================================================================

def create_phase1():
    """Phase 1 (frames 1-120): Heat → Polarity → Existence → Wireframe Cube.
    Returns (heat_particle, existence_dot, cube_container, polarity_line, plus_text, minus_text).
    """
    # --- Heat (frames 1-30) ---
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 0, 0), segments=32, ring_count=16)
    heat_particle = bpy.context.active_object
    heat_particle.name = "HeatParticle"

    mat, mapping_node = create_animated_color_material("HeatMaterial")
    heat_particle.data.materials.append(mat)

    # Animate swirling texture
    for f in range(1, TOTAL_FRAMES + 1, 5):
        t = f / FPS
        mapping_node.inputs['Rotation'].default_value = (0, 0, t * 0.5)
        mapping_node.inputs['Rotation'].keyframe_insert(data_path="default_value", frame=f)

    # Continuous spin
    for f in range(1, TOTAL_FRAMES + 1, 3):
        t = f / FPS
        angle = t * math.pi * 4
        heat_particle.rotation_euler = (angle, angle * 0.7, angle * 0.3)
        heat_particle.keyframe_insert(data_path="rotation_euler", frame=f)

    # Planck temperature label (visible during close-up frames ~8-22)
    temp_mat = create_emission_material("TempLabelMaterial", (1, 0.5, 0.2), 3.0)
    temp_label = create_text_object("T = 1.416 x 10^32 K", (0, 0, -0.8), 0.15, temp_mat)
    temp_label.name = "PlanckTempLabel"
    temp_label.rotation_euler = (math.pi/2, 0, 0)
    keyframe_visibility(temp_label, False, 1)
    keyframe_visibility(temp_label, True, 8)
    keyframe_visibility(temp_label, False, 22)

    # Formula label
    formula_mat = create_emission_material("FormulaLabelMaterial", (0.8, 0.8, 1.0), 2.0)
    formula_label = create_text_object("K x 12^30 / phi^2 x 45/44", (0, 0, -1.1), 0.1, formula_mat)
    formula_label.name = "FormulaLabel"
    formula_label.rotation_euler = (math.pi/2, 0, 0)
    keyframe_visibility(formula_label, False, 1)
    keyframe_visibility(formula_label, True, 10)
    keyframe_visibility(formula_label, False, 22)

    # --- Polarity (frames 31-60) ---
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1, location=(0, 0, 0))
    polarity_line = bpy.context.active_object
    polarity_line.name = "PolarityLine"
    polarity_line.rotation_euler = (0, math.pi/2, 0)

    line_mat = create_emission_material("PolarityMaterial", (1.0, 1.0, 1.0), 3.0)
    polarity_line.data.materials.append(line_mat)

    keyframe_visibility(polarity_line, False, 1)
    keyframe_visibility(polarity_line, True, P1_POLARITY_START)
    keyframe_scale(polarity_line, (0.01, 0.01, 0.01), P1_POLARITY_START)
    keyframe_scale(polarity_line, (1, 1, 30), P1_POLARITY_START + 10)
    keyframe_scale(polarity_line, (1, 1, 50), P1_POLARITY_END)

    text_mat = create_emission_material("TextMaterial", (1.0, 1.0, 1.0), 5.0)

    plus_text = create_text_object("+", (3, 0, 0.8), 1.5, text_mat)
    plus_text.name = "PlusLabel"
    plus_text.rotation_euler = (math.pi/2, 0, 0)
    keyframe_visibility(plus_text, False, 1)
    keyframe_visibility(plus_text, True, P1_POLARITY_START + 5)
    keyframe_location(plus_text, (3, 0, 0.8), P1_POLARITY_START + 5)
    keyframe_location(plus_text, (25, 0, 0.8), P1_POLARITY_END)

    minus_text = create_text_object("-", (-3, 0, 0.8), 1.5, text_mat)
    minus_text.name = "MinusLabel"
    minus_text.rotation_euler = (math.pi/2, 0, 0)
    keyframe_visibility(minus_text, False, 1)
    keyframe_visibility(minus_text, True, P1_POLARITY_START + 5)
    keyframe_location(minus_text, (-3, 0, 0.8), P1_POLARITY_START + 5)
    keyframe_location(minus_text, (-25, 0, 0.8), P1_POLARITY_END)

    # --- Existence (frames 61-90) ---
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0, 0, 0))
    existence_dot = bpy.context.active_object
    existence_dot.name = "ExistenceDot"

    exist_mat = create_emission_material("ExistenceDotMaterial", (0.9, 0.9, 1.0), 5.0)
    existence_dot.data.materials.append(exist_mat)

    keyframe_visibility(existence_dot, False, 1)
    keyframe_visibility(existence_dot, True, P1_EXISTENCE_START)
    keyframe_scale(existence_dot, 0.01, P1_EXISTENCE_START)
    keyframe_scale(existence_dot, 1.0, P1_EXISTENCE_START + 8)

    # Existence dot stays at origin (polarity midpoint)
    keyframe_location(existence_dot, (0, 0, 0), P1_EXISTENCE_START)
    keyframe_location(existence_dot, (0, 0, 0), TOTAL_FRAMES)

    # Spinning
    for f in range(P1_EXISTENCE_START, P1_EXISTENCE_END + 1, 3):
        t = f / FPS
        angle = t * math.pi * 4
        existence_dot.rotation_euler = (angle, angle * 0.7, angle * 0.3)
        existence_dot.keyframe_insert(data_path="rotation_euler", frame=f)

    # --- Wireframe Cube (frames 91-120) ---
    # Hide polarity labels and line as they merge into cube context
    keyframe_visibility(plus_text, False, P1_CUBE_START + 5)
    keyframe_visibility(minus_text, False, P1_CUBE_START + 5)
    keyframe_scale(polarity_line, (0.5, 0.5, 20), P1_CUBE_START)
    keyframe_visibility(polarity_line, False, P1_CUBE_START + 15)

    cube_container, cube_edges = create_wireframe_cube((0, 0, 0), 0.01, "Spacetime")

    # Hide cube initially, then expand during cube phase
    keyframe_visibility(cube_container, False, 1)
    for edge in cube_edges:
        keyframe_visibility(edge, False, 1)

    keyframe_visibility(cube_container, True, P1_CUBE_START)
    for edge in cube_edges:
        keyframe_visibility(edge, True, P1_CUBE_START)

    keyframe_scale(cube_container, 0.01, P1_CUBE_START)
    keyframe_scale(cube_container, 3.0, P1_CUBE_END)

    return heat_particle, existence_dot, cube_container, polarity_line, plus_text, minus_text


def create_phase2(existence_dot, cube_container):
    """Phase 2 (frames 121-180): Order emergence.
    Existence dot vibrates (121-140), sphere morphs to icosahedron (141-160),
    settle into structure (160-180). Returns icosahedron.
    """
    # --- Vibration (121-140) ---
    vibration_start = P2_START
    vibration_end = P2_START + 20  # frame 140

    for f in range(vibration_start, vibration_end + 1):
        t = (f - vibration_start) / 20.0
        # Increasing vibration amplitude
        amp = 0.05 + t * 0.15
        offset_x = amp * math.sin(f * 2.5)
        offset_z = amp * math.cos(f * 3.1)
        existence_dot.location = (offset_x, 0, offset_z)
        existence_dot.keyframe_insert(data_path="location", frame=f)

    # Return to center before morph
    keyframe_location(existence_dot, (0, 0, 0), vibration_end + 1)

    # --- Sphere to Icosahedron morph (141-160) ---
    morph_start = vibration_end + 1  # frame 141
    morph_duration = 15

    icosahedron = create_icosahedron((0, 0, 0), 0.3, "OrderIcosahedron")
    ico_mat = create_emission_material("IcoMaterial", (0.7, 0.85, 1.0), 5.0)
    icosahedron.data.materials.append(ico_mat)

    # Morph: existence_dot shrinks out, icosahedron grows in
    morph_shapes(existence_dot, icosahedron, morph_start, morph_duration)

    # --- Settle (160-180) ---
    settle_start = morph_start + morph_duration  # frame 156
    # Gentle rotation as it stabilizes
    for f in range(settle_start, P2_END + 1, 2):
        t = (f - settle_start) / (P2_END - settle_start)
        angle = t * math.pi * 0.5
        icosahedron.rotation_euler = (angle * 0.3, angle, angle * 0.2)
        icosahedron.keyframe_insert(data_path="rotation_euler", frame=f)

    # Cube gently expands to accommodate the structure
    keyframe_scale(cube_container, 3.0, P2_START)
    keyframe_scale(cube_container, 3.5, P2_END)

    return icosahedron


def create_phase3(icosahedron, cube_container):
    """Phase 3 (frames 181-240): SU group emergence.
    Polarity colors appear on icosahedron (181-200), icosahedron morphs to torus (200-220),
    stable colored torus particle (220-240). Returns (torus, polarity_state).
    """
    polarity_state = DEFAULT_POLARITY

    # --- Apply polarity colors to icosahedron (181-200) ---
    color_start = P3_START
    color_end = P3_START + 20  # frame 200

    # The colors "appear" - we swap materials at the color_start frame
    # First, assign quadrant colors to the icosahedron
    apply_quadrant_colors(icosahedron, polarity_state)

    # Pulse the icosahedron to indicate color activation
    keyframe_scale(icosahedron, 1.0, color_start)
    keyframe_scale(icosahedron, 1.3, color_start + 5)
    keyframe_scale(icosahedron, 1.0, color_start + 10)
    keyframe_scale(icosahedron, 1.2, color_start + 15)
    keyframe_scale(icosahedron, 1.0, color_end)

    # Gentle spin as colors activate
    for f in range(color_start, color_end + 1, 2):
        t = (f - color_start) / (color_end - color_start)
        angle = t * math.pi * 2
        icosahedron.rotation_euler = (angle * 0.2, angle, angle * 0.3)
        icosahedron.keyframe_insert(data_path="rotation_euler", frame=f)

    # --- Icosahedron to Torus morph (200-220) ---
    morph_start = color_end  # frame 200
    morph_duration = 15

    torus = create_torus_particle((0, 0, 0), 0.4, 0.12, "SU2_Torus")
    apply_quadrant_colors(torus, polarity_state)

    morph_shapes(icosahedron, torus, morph_start, morph_duration)

    # --- Stable particle (220-240) ---
    stable_start = morph_start + morph_duration  # frame 215
    # Steady rotation
    for f in range(stable_start, P3_END + 1, 2):
        t = (f - stable_start) / max(1, P3_END - stable_start)
        angle = t * math.pi
        torus.rotation_euler = (0, angle, angle * 0.3)
        torus.keyframe_insert(data_path="rotation_euler", frame=f)

    # Cube stays stable
    keyframe_scale(cube_container, 3.5, P3_START)
    keyframe_scale(cube_container, 3.5, P3_END)

    return torus, polarity_state


def create_phase4(torus, polarity_state, cube_container):
    """Phase 4 (frames 241-320): First 45/44 split.
    Heat pulse (241-260), 45/44 split at frame 270, children diverge (270-320).
    Returns (child_a, child_b, child_a_data, child_b_data).
    """
    # --- Heat pulse (241-260) ---
    pulse_start = P4_START
    pulse_end = P4_START + 20  # frame 260

    # Torus pulses (scale oscillation representing heat accumulation to 45)
    for f in range(pulse_start, pulse_end + 1, 2):
        t = (f - pulse_start) / (pulse_end - pulse_start)
        # Growing pulse amplitude - heat building
        pulse = 1.0 + 0.3 * t * math.sin(t * math.pi * 6)
        torus.scale = (pulse, pulse, pulse)
        torus.keyframe_insert(data_path="scale", frame=f)

    # Continue gentle spin during pulse
    for f in range(pulse_start, pulse_end + 1, 3):
        t = (f - pulse_start) / (pulse_end - pulse_start)
        base_angle = math.pi  # continue from phase 3 end
        angle = base_angle + t * math.pi
        torus.rotation_euler = (0, angle, angle * 0.3)
        torus.keyframe_insert(data_path="rotation_euler", frame=f)

    # Final big pulse before split
    keyframe_scale(torus, 1.4, pulse_end - 2)
    keyframe_scale(torus, 1.0, pulse_end)

    # --- 45/44 Split at frame 270 ---
    split_frame = P4_START + 30  # frame 271

    # Ensure torus is at origin before split
    keyframe_location(torus, (0, 0, 0), split_frame - 1)

    parent_direction = 0.0  # radians, along +X

    child_a, child_b, child_a_data, child_b_data = split_particle(
        parent_obj=torus,
        parent_heat=THRESHOLD_NUMERATOR,
        parent_polarity=polarity_state,
        parent_direction=parent_direction,
        split_frame=split_frame,
        name_prefix="Gen1"
    )

    # --- Children diverge (271-320) ---
    diverge_frames = P4_END - split_frame  # ~50 frames
    for f in range(split_frame, P4_END + 1, 2):
        t = (f - split_frame) / diverge_frames
        dist = t * 3.0  # how far they travel

        # Child A goes upper-right
        ax = dist * math.cos(child_a_data['direction'])
        az = dist * math.sin(child_a_data['direction'])
        keyframe_location(child_a, (ax, 0, az), f)

        # Spin child A
        child_a.rotation_euler = (t * math.pi, t * math.pi * 1.3, 0)
        child_a.keyframe_insert(data_path="rotation_euler", frame=f)

        # Child B goes lower-left
        bx = dist * math.cos(child_b_data['direction'])
        bz = dist * math.sin(child_b_data['direction'])
        keyframe_location(child_b, (bx, 0, bz), f)

        # Spin child B
        child_b.rotation_euler = (0, t * math.pi * 1.1, t * math.pi * 0.8)
        child_b.keyframe_insert(data_path="rotation_euler", frame=f)

    # Cube expands to contain diverging particles
    keyframe_scale(cube_container, 3.5, P4_START)
    keyframe_scale(cube_container, 5.0, P4_END)

    return child_a, child_b, child_a_data, child_b_data


def create_phase5(child_a, child_b, child_a_data, child_b_data, cube_container):
    """Phase 5 (frames 321-480): Cascade splits + collision.
    Child A splits at ~340, Child B splits at ~370, collision at ~440, deflection.
    Returns all_particles list.
    """
    all_particles = [child_a, child_b]

    # Get end positions of children from phase 4
    a_pos = child_a.location.copy()
    b_pos = child_b.location.copy()

    # --- Child A splits at frame 340 ---
    split_a_frame = 340

    # Move child_a to its split position and have it pulse beforehand
    for f in range(P5_START, split_a_frame, 2):
        t = (f - P5_START) / (split_a_frame - P5_START)
        # Continue moving in its direction
        dist = 3.0 + t * 1.5
        ax = dist * math.cos(child_a_data['direction'])
        az = dist * math.sin(child_a_data['direction'])
        keyframe_location(child_a, (ax, 0, az), f)

        # Pulsing as heat builds
        pulse = 1.0 + 0.2 * t * math.sin(t * math.pi * 8)
        child_a.scale = (pulse, pulse, pulse)
        child_a.keyframe_insert(data_path="scale", frame=f)

    a_split_pos = child_a.location.copy()
    keyframe_location(child_a, a_split_pos, split_a_frame - 1)

    # Perform split
    gen2_a, gen2_b, gen2_a_data, gen2_b_data = split_particle(
        parent_obj=child_a,
        parent_heat=THRESHOLD_NUMERATOR,
        parent_polarity=child_a_data['polarity'],
        parent_direction=child_a_data['direction'],
        split_frame=split_a_frame,
        name_prefix="Gen2A"
    )
    all_particles.extend([gen2_a, gen2_b])

    # Gen2 children travel outward from split point
    for f in range(split_a_frame, 370, 2):
        t = (f - split_a_frame) / 30.0
        dist = t * 2.0

        ga_x = a_split_pos.x + dist * math.cos(gen2_a_data['direction'])
        ga_z = a_split_pos.z + dist * math.sin(gen2_a_data['direction'])
        keyframe_location(gen2_a, (ga_x, 0, ga_z), f)
        gen2_a.rotation_euler = (t * math.pi, t * math.pi * 0.7, 0)
        gen2_a.keyframe_insert(data_path="rotation_euler", frame=f)

        gb_x = a_split_pos.x + dist * math.cos(gen2_b_data['direction'])
        gb_z = a_split_pos.z + dist * math.sin(gen2_b_data['direction'])
        keyframe_location(gen2_b, (gb_x, 0, gb_z), f)
        gen2_b.rotation_euler = (0, t * math.pi, t * math.pi * 0.5)
        gen2_b.keyframe_insert(data_path="rotation_euler", frame=f)

    # --- Child B splits at frame 370 ---
    split_b_frame = 370

    # Move child_b toward its split position
    for f in range(P5_START, split_b_frame, 2):
        t = (f - P5_START) / (split_b_frame - P5_START)
        dist = 3.0 + t * 1.5
        bx = dist * math.cos(child_b_data['direction'])
        bz = dist * math.sin(child_b_data['direction'])
        keyframe_location(child_b, (bx, 0, bz), f)

        pulse = 1.0 + 0.2 * t * math.sin(t * math.pi * 8)
        child_b.scale = (pulse, pulse, pulse)
        child_b.keyframe_insert(data_path="scale", frame=f)

    b_split_pos = child_b.location.copy()
    keyframe_location(child_b, b_split_pos, split_b_frame - 1)

    gen3_a, gen3_b, gen3_a_data, gen3_b_data = split_particle(
        parent_obj=child_b,
        parent_heat=THRESHOLD_NUMERATOR,
        parent_polarity=child_b_data['polarity'],
        parent_direction=child_b_data['direction'],
        split_frame=split_b_frame,
        name_prefix="Gen2B"
    )
    all_particles.extend([gen3_a, gen3_b])

    # Gen3 children travel outward
    for f in range(split_b_frame, 440, 2):
        t = (f - split_b_frame) / 70.0
        dist = t * 3.0

        g3a_x = b_split_pos.x + dist * math.cos(gen3_a_data['direction'])
        g3a_z = b_split_pos.z + dist * math.sin(gen3_a_data['direction'])
        keyframe_location(gen3_a, (g3a_x, 0, g3a_z), f)
        gen3_a.rotation_euler = (t * math.pi * 1.2, t * math.pi * 0.6, 0)
        gen3_a.keyframe_insert(data_path="rotation_euler", frame=f)

        g3b_x = b_split_pos.x + dist * math.cos(gen3_b_data['direction'])
        g3b_z = b_split_pos.z + dist * math.sin(gen3_b_data['direction'])
        keyframe_location(gen3_b, (g3b_x, 0, g3b_z), f)
        gen3_b.rotation_euler = (0, t * math.pi * 0.9, t * math.pi * 1.1)
        gen3_b.keyframe_insert(data_path="rotation_euler", frame=f)

    # Continue gen2 particles traveling through collision zone
    for f in range(370, 440, 2):
        t = (f - 340) / 100.0
        dist = t * 4.0

        ga_x = a_split_pos.x + dist * math.cos(gen2_a_data['direction'])
        ga_z = a_split_pos.z + dist * math.sin(gen2_a_data['direction'])
        keyframe_location(gen2_a, (ga_x, 0, ga_z), f)
        gen2_a.rotation_euler = (t * math.pi, t * math.pi * 0.7, 0)
        gen2_a.keyframe_insert(data_path="rotation_euler", frame=f)

        gb_x = a_split_pos.x + dist * math.cos(gen2_b_data['direction'])
        gb_z = a_split_pos.z + dist * math.sin(gen2_b_data['direction'])
        keyframe_location(gen2_b, (gb_x, 0, gb_z), f)
        gen2_b.rotation_euler = (0, t * math.pi, t * math.pi * 0.5)
        gen2_b.keyframe_insert(data_path="rotation_euler", frame=f)

    # --- Collision at frame 440 ---
    collision_frame = 440
    # Gen2_B and Gen3_A converge at a collision point
    # Calculate meeting point (midpoint of their positions at collision)
    gen2b_pos = gen2_b.location.copy()
    gen3a_pos = gen3_a.location.copy()
    collision_point = (
        (gen2b_pos.x + gen3a_pos.x) / 2,
        0,
        (gen2b_pos.z + gen3a_pos.z) / 2,
    )

    # Steer them toward collision point
    for f in range(430, collision_frame + 1, 2):
        t = (f - 430) / 10.0
        # Gen2_B moves toward collision point
        cx = gen2b_pos.x + t * (collision_point[0] - gen2b_pos.x)
        cz = gen2b_pos.z + t * (collision_point[2] - gen2b_pos.z)
        keyframe_location(gen2_b, (cx, 0, cz), f)

        # Gen3_A moves toward collision point
        dx = gen3a_pos.x + t * (collision_point[0] - gen3a_pos.x)
        dz = gen3a_pos.z + t * (collision_point[2] - gen3a_pos.z)
        keyframe_location(gen3_a, (dx, 0, dz), f)

    # Big flash at collision
    create_split_flash(collision_point, collision_frame, duration=8)

    # --- Deflection after collision (440-480) ---
    # Particles bounce off in new directions
    deflect_angle_offset = math.pi / 3  # 60 degree deflection

    # Gen2_B deflects
    gen2b_new_dir = gen2_b_data['direction'] + deflect_angle_offset
    # Gen3_A deflects
    gen3a_new_dir = gen3_a_data['direction'] - deflect_angle_offset

    col_loc = Vector(collision_point)
    for f in range(collision_frame, P5_END + 1, 2):
        t = (f - collision_frame) / (P5_END - collision_frame)
        dist = t * 3.0

        # Gen2_B bounces away
        g2b_x = col_loc.x + dist * math.cos(gen2b_new_dir)
        g2b_z = col_loc.z + dist * math.sin(gen2b_new_dir)
        keyframe_location(gen2_b, (g2b_x, 0, g2b_z), f)
        gen2_b.rotation_euler = (t * math.pi * 2, t * math.pi * 1.5, 0)
        gen2_b.keyframe_insert(data_path="rotation_euler", frame=f)

        # Gen3_A bounces away
        g3a_x = col_loc.x + dist * math.cos(gen3a_new_dir)
        g3a_z = col_loc.z + dist * math.sin(gen3a_new_dir)
        keyframe_location(gen3_a, (g3a_x, 0, g3a_z), f)
        gen3_a.rotation_euler = (0, t * math.pi * 2, t * math.pi * 1.8)
        gen3_a.keyframe_insert(data_path="rotation_euler", frame=f)

    # Continue other particles drifting during post-collision
    for f in range(collision_frame, P5_END + 1, 2):
        t = (f - collision_frame) / (P5_END - collision_frame)

        # Gen2_A keeps drifting
        dist_2a = 4.0 + t * 2.0
        ga_x = a_split_pos.x + dist_2a * math.cos(gen2_a_data['direction'])
        ga_z = a_split_pos.z + dist_2a * math.sin(gen2_a_data['direction'])
        keyframe_location(gen2_a, (ga_x, 0, ga_z), f)
        gen2_a.rotation_euler = (t * math.pi * 1.5 + math.pi, t * math.pi * 0.7, 0)
        gen2_a.keyframe_insert(data_path="rotation_euler", frame=f)

        # Gen3_B keeps drifting
        dist_3b = 3.0 + t * 2.0
        g3b_x = b_split_pos.x + dist_3b * math.cos(gen3_b_data['direction'])
        g3b_z = b_split_pos.z + dist_3b * math.sin(gen3_b_data['direction'])
        keyframe_location(gen3_b, (g3b_x, 0, g3b_z), f)
        gen3_b.rotation_euler = (0, t * math.pi * 1.3, t * math.pi * 1.1 + math.pi)
        gen3_b.keyframe_insert(data_path="rotation_euler", frame=f)

    # Cube dynamically sizes to contain all particles
    keyframe_scale(cube_container, 5.0, P5_START)
    keyframe_scale(cube_container, 6.0, split_a_frame)
    keyframe_scale(cube_container, 7.0, split_b_frame)
    keyframe_scale(cube_container, 8.0, collision_frame)
    keyframe_scale(cube_container, 10.0, P5_END)

    return all_particles


def create_camera_animation_new():
    """Create camera with Track To constraint on origin empty.
    Parametric orbit: x = r*sin(angle), y = -r*cos(angle), z = elevation.
    Zoom in for heat, pull back for polarity/existence, orbit for phases 2-5.
    """
    # Create tracking target at origin
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    target = bpy.context.active_object
    target.name = "CameraTarget"

    # Create camera
    bpy.ops.object.camera_add(location=(0, -8, 0))
    camera = bpy.context.active_object
    camera.name = "MainCamera"
    bpy.context.scene.camera = camera

    # Add Track To constraint
    track_constraint = camera.constraints.new(type='TRACK_TO')
    track_constraint.target = target
    track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    track_constraint.up_axis = 'UP_Y'

    # Phase 1: Heat - close zoom
    keyframe_location(camera, (0, -8, 1), 1)
    keyframe_location(camera, (0, -1.5, 0.3), 8)   # zoom in for temp label
    keyframe_location(camera, (0, -1.5, 0.3), 20)   # stay close
    keyframe_location(camera, (0, -8, 1), P1_HEAT_END)  # pull back

    # Phase 1: Polarity - pan along line
    keyframe_location(camera, (0, -8, 1), P1_POLARITY_START)
    keyframe_location(camera, (8, -8, 1), P1_POLARITY_START + 10)
    keyframe_location(camera, (0, -12, 2), P1_POLARITY_END)

    # Phase 1: Existence - pull back to see dot
    keyframe_location(camera, (0, -10, 3), P1_EXISTENCE_START)
    keyframe_location(camera, (0, -10, 3), P1_EXISTENCE_END)

    # Phase 1: Cube - wider view
    keyframe_location(camera, (3, -12, 4), P1_CUBE_START)
    keyframe_location(camera, (5, -12, 5), P1_CUBE_END)

    # Phase 2: Orbit while watching vibration/morph
    orbit_keyframes = [
        (P2_START, 10, math.pi * 0.1, 4),
        (P2_START + 30, 12, math.pi * 0.4, 5),
        (P2_END, 12, math.pi * 0.7, 5),
    ]
    for f, r, angle, elev in orbit_keyframes:
        x = r * math.sin(angle)
        y = -r * math.cos(angle)
        keyframe_location(camera, (x, y, elev), f)

    # Phase 3: Continue orbit
    orbit_keyframes_p3 = [
        (P3_START, 12, math.pi * 0.8, 5),
        (P3_START + 30, 13, math.pi * 1.1, 5),
        (P3_END, 14, math.pi * 1.4, 5),
    ]
    for f, r, angle, elev in orbit_keyframes_p3:
        x = r * math.sin(angle)
        y = -r * math.cos(angle)
        keyframe_location(camera, (x, y, elev), f)

    # Phase 4: Pull back for split
    orbit_keyframes_p4 = [
        (P4_START, 15, math.pi * 1.5, 6),
        (P4_START + 40, 16, math.pi * 1.7, 7),
        (P4_END, 17, math.pi * 1.9, 7),
    ]
    for f, r, angle, elev in orbit_keyframes_p4:
        x = r * math.sin(angle)
        y = -r * math.cos(angle)
        keyframe_location(camera, (x, y, elev), f)

    # Phase 5: Wide orbit to see all particles
    orbit_keyframes_p5 = [
        (P5_START, 18, math.pi * 2.0, 8),
        (360, 19, math.pi * 2.2, 9),
        (400, 20, math.pi * 2.5, 10),
        (440, 20, math.pi * 2.7, 10),  # collision moment
        (P5_END, 20, math.pi * 3.0, 10),
    ]
    for f, r, angle, elev in orbit_keyframes_p5:
        x = r * math.sin(angle)
        y = -r * math.cos(angle)
        keyframe_location(camera, (x, y, elev), f)

    return camera, target


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to create the entire animation"""
    print("=" * 60)
    print("Starting Motion Calendar Simulation Creation...")
    print("=" * 60)

    clear_scene()
    print("[OK] Scene cleared")

    setup_render_settings()
    print("[OK] Render settings configured")

    # Phase 1: Heat -> Polarity -> Existence -> Cube
    heat_particle, existence_dot, cube_container, polarity_line, plus_text, minus_text = create_phase1()
    print("[OK] Phase 1 complete: Heat, Polarity, Existence, Cube (frames 1-120)")

    # Phase 2: Order emergence (vibration -> icosahedron)
    icosahedron = create_phase2(existence_dot, cube_container)
    print("[OK] Phase 2 complete: Order emergence (frames 121-180)")

    # Phase 3: SU group emergence (colors -> torus)
    torus, polarity_state = create_phase3(icosahedron, cube_container)
    print("[OK] Phase 3 complete: SU(2) group emergence (frames 181-240)")

    # Phase 4: First 45/44 split
    child_a, child_b, child_a_data, child_b_data = create_phase4(torus, polarity_state, cube_container)
    print("[OK] Phase 4 complete: 45/44 split (frames 241-320)")

    # Phase 5: Cascade splits + collision
    all_particles = create_phase5(child_a, child_b, child_a_data, child_b_data, cube_container)
    print("[OK] Phase 5 complete: Cascade splits + collision (frames 321-480)")

    # Camera
    camera, camera_target = create_camera_animation_new()
    print("[OK] Camera animation created")

    # Glow
    add_glow_effect()
    print("[OK] Glow effect added")

    # Set current frame to start
    bpy.context.scene.frame_set(1)

    print("\n" + "=" * 60)
    print("MOTION CALENDAR SIMULATION COMPLETE!")
    print("=" * 60)
    total_seconds = TOTAL_FRAMES / FPS
    print(f"Total Duration: {total_seconds:.1f} seconds ({TOTAL_FRAMES} frames)")
    print(f"Resolution: 1920x1080 @ {FPS} FPS")
    print("\nPhase Breakdown:")
    print(f"  Phase 1 (Heat/Polarity/Existence/Cube): frames {P1_START}-{P1_END}")
    print(f"  Phase 2 (Order emergence):              frames {P2_START}-{P2_END}")
    print(f"  Phase 3 (SU(2) group emergence):        frames {P3_START}-{P3_END}")
    print(f"  Phase 4 (45/44 split):                  frames {P4_START}-{P4_END}")
    print(f"  Phase 5 (Cascade + collision):          frames {P5_START}-{P5_END}")
    print(f"\nParticles created: {len(all_particles)}")
    print("\n> To preview: Press SPACEBAR in the viewport")
    print("> To render:  Render > Render Animation (Ctrl+F12)")
    print("=" * 60)


# Run the script
if __name__ == "__main__":
    main()
