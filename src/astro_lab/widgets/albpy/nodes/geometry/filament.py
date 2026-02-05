"""
Filament Geometry Node Groups
=============================

Modern Blender 4.4 implementation for cosmic filament structures using factory functions.
"""

from typing import Any, Dict

import bpy


def create_cosmic_filament_node_group():
    """
    Create cosmic filament geometry node group using modern Blender 4.4 API.

    Returns:
        bpy.types.GeometryNodeTree: The created cosmic filament node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_CosmicFilament", "GeometryNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    interface.new_socket(
        name="Filament Count", in_out="INPUT", socket_type="NodeSocketInt"
    )
    interface.new_socket(
        name="Filament Length", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Filament Width", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Curvature", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Branch Factor", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Density Variation", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Random Seed", in_out="INPUT", socket_type="NodeSocketInt"
    )

    # Output sockets
    interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    # Set default values
    ng.interface.items_tree[1].default_value = 15  # Filament Count
    ng.interface.items_tree[2].default_value = 25.0  # Filament Length
    ng.interface.items_tree[3].default_value = 0.05  # Filament Width
    ng.interface.items_tree[4].default_value = 0.3  # Curvature
    ng.interface.items_tree[5].default_value = 0.2  # Branch Factor
    ng.interface.items_tree[6].default_value = 0.5  # Density Variation
    ng.interface.items_tree[7].default_value = 42  # Random Seed

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Create distribution volume (stretched sphere for cosmic web)
    sphere = ng.nodes.new("GeometryNodeMeshUVSphere")
    sphere.inputs["Subdivisions"].default_value = 3

    # Scale to elongated shape
    scale_distribution = ng.nodes.new("GeometryNodeTransform")
    scale_distribution.inputs["Scale"].default_value = (1.0, 1.0, 0.3)

    # Smooth shading
    set_smooth_dist = ng.nodes.new("GeometryNodeSetShadeSmooth")
    set_smooth_dist.inputs["Shade Smooth"].default_value = True

    # Distribute points for filament origins
    distribute_points = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute_points.distribute_method = "RANDOM"

    # Create individual filament curves
    curve_line = ng.nodes.new("GeometryNodeCurvePrimitiveLine")
    curve_line.mode = "POINTS"
    curve_line.inputs["Start"].default_value = (0.0, 0.0, -0.5)
    curve_line.inputs["End"].default_value = (0.0, 0.0, 0.5)

    # Add resolution to curve
    resample_curve = ng.nodes.new("GeometryNodeResampleCurve")
    resample_curve.mode = "COUNT"
    resample_curve.inputs["Count"].default_value = 32

    # Add noise for filament deformation
    noise_position = ng.nodes.new("GeometryNodeInputPosition")
    noise_texture = ng.nodes.new("ShaderNodeTexNoise")
    noise_texture.inputs["Scale"].default_value = 2.0
    noise_texture.inputs["Detail"].default_value = 3.0

    # Vector math for noise application
    noise_scale = ng.nodes.new("ShaderNodeVectorMath")
    noise_scale.operation = "MULTIPLY"
    noise_scale.inputs[1].default_value = (0.5, 0.5, 0.1)

    # Set position with noise
    set_position = ng.nodes.new("GeometryNodeSetPosition")

    # Add branching points
    branch_points = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    branch_points.distribute_method = "RANDOM"

    # Create branch geometry
    branch_curve = ng.nodes.new("GeometryNodeCurvePrimitiveLine")
    branch_curve.mode = "POINTS"
    branch_curve.inputs["Start"].default_value = (0.0, 0.0, 0.0)
    branch_curve.inputs["End"].default_value = (0.3, 0.2, 0.1)

    # Instance branches on main filament
    instance_branches = ng.nodes.new("GeometryNodeInstanceOnPoints")

    # Convert curves to mesh
    curve_to_mesh = ng.nodes.new("GeometryNodeCurveToMesh")

    # Create profile curve for filament thickness
    profile_curve = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
    profile_curve.mode = "RADIUS"

    # Join main filament and branches
    join_geometry = ng.nodes.new("GeometryNodeJoinGeometry")

    # Instance complete filament structure
    instance_filaments = ng.nodes.new("GeometryNodeInstanceOnPoints")

    # Random rotation for variety
    random_rotation = ng.nodes.new("GeometryNodeInputRandomValue")
    random_rotation.data_type = "FLOAT_VECTOR"
    random_rotation.inputs["Min"].default_value = (0.0, 0.0, 0.0)
    random_rotation.inputs["Max"].default_value = (6.28, 6.28, 6.28)

    # Rotate instances
    rotate_instances = ng.nodes.new("GeometryNodeRotateInstances")

    # Smooth shading for final result
    set_smooth_final = ng.nodes.new("GeometryNodeSetShadeSmooth")
    set_smooth_final.inputs["Shade Smooth"].default_value = True

    # Position nodes
    input_node.location = (-1400, 0)
    sphere.location = (-1200, 0)
    scale_distribution.location = (-1100, 0)
    set_smooth_dist.location = (-1000, 0)
    distribute_points.location = (-900, 0)

    # Filament creation chain
    curve_line.location = (-900, 300)
    resample_curve.location = (-800, 300)
    noise_position.location = (-1000, 500)
    noise_texture.location = (-800, 500)
    noise_scale.location = (-600, 500)
    set_position.location = (-600, 300)

    # Branching
    branch_points.location = (-500, 400)
    branch_curve.location = (-500, 600)
    instance_branches.location = (-400, 500)

    # Mesh conversion
    profile_curve.location = (-600, 100)
    curve_to_mesh.location = (-400, 300)
    join_geometry.location = (-300, 400)

    # Final instancing
    instance_filaments.location = (-200, 0)
    random_rotation.location = (-400, -200)
    rotate_instances.location = (-100, 0)
    set_smooth_final.location = (0, 0)
    output_node.location = (200, 0)

    # Connect nodes
    # Distribution setup
    ng.links.new(input_node.outputs["Filament Length"], sphere.inputs["Radius"])
    ng.links.new(sphere.outputs["Mesh"], scale_distribution.inputs["Geometry"])
    ng.links.new(
        scale_distribution.outputs["Geometry"], set_smooth_dist.inputs["Geometry"]
    )
    ng.links.new(set_smooth_dist.outputs["Geometry"], distribute_points.inputs["Mesh"])
    ng.links.new(
        input_node.outputs["Filament Count"], distribute_points.inputs["Density"]
    )
    ng.links.new(input_node.outputs["Random Seed"], distribute_points.inputs["Seed"])

    # Filament creation
    ng.links.new(input_node.outputs["Filament Length"], curve_line.inputs["End"])
    ng.links.new(curve_line.outputs["Curve"], resample_curve.inputs["Curve"])
    ng.links.new(resample_curve.outputs["Curve"], set_position.inputs["Geometry"])

    # Noise application
    ng.links.new(noise_position.outputs["Position"], noise_texture.inputs["Vector"])
    ng.links.new(input_node.outputs["Curvature"], noise_texture.inputs["Scale"])
    ng.links.new(noise_texture.outputs["Color"], noise_scale.inputs[0])
    ng.links.new(input_node.outputs["Curvature"], noise_scale.inputs[1])
    ng.links.new(noise_scale.outputs["Vector"], set_position.inputs["Offset"])

    # Branching
    ng.links.new(set_position.outputs["Geometry"], branch_points.inputs["Mesh"])
    ng.links.new(input_node.outputs["Branch Factor"], branch_points.inputs["Density"])
    ng.links.new(branch_curve.outputs["Curve"], instance_branches.inputs["Instance"])
    ng.links.new(branch_points.outputs["Points"], instance_branches.inputs["Points"])

    # Mesh conversion
    ng.links.new(input_node.outputs["Filament Width"], profile_curve.inputs["Radius"])
    ng.links.new(set_position.outputs["Geometry"], curve_to_mesh.inputs["Curve"])
    ng.links.new(profile_curve.outputs["Curve"], curve_to_mesh.inputs["Profile Curve"])
    ng.links.new(curve_to_mesh.outputs["Mesh"], join_geometry.inputs["Geometry"])
    ng.links.new(
        instance_branches.outputs["Instances"], join_geometry.inputs["Geometry"]
    )

    # Final instancing
    ng.links.new(
        join_geometry.outputs["Geometry"], instance_filaments.inputs["Instance"]
    )
    ng.links.new(
        distribute_points.outputs["Points"], instance_filaments.inputs["Points"]
    )
    ng.links.new(
        instance_filaments.outputs["Instances"], rotate_instances.inputs["Instances"]
    )

    # Random rotation
    ng.links.new(input_node.outputs["Random Seed"], random_rotation.inputs["Seed"])
    ng.links.new(random_rotation.outputs["Value"], rotate_instances.inputs["Rotation"])

    # Final output
    ng.links.new(
        rotate_instances.outputs["Instances"], set_smooth_final.inputs["Geometry"]
    )
    ng.links.new(set_smooth_final.outputs["Geometry"], output_node.inputs["Geometry"])

    return ng


def create_galaxy_bridge_node_group():
    """
    Create galaxy bridge filament for connecting galaxy structures.

    Returns:
        bpy.types.GeometryNodeTree: The created galaxy bridge node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_GalaxyBridge", "GeometryNodeTree")

    # Interface API
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Start Position", in_out="INPUT", socket_type="NodeSocketVector"
    )
    interface.new_socket(
        name="End Position", in_out="INPUT", socket_type="NodeSocketVector"
    )
    interface.new_socket(
        name="Bridge Width", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Twist Factor", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Particle Density", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    # Set defaults
    ng.interface.items_tree[0].default_value = (-10.0, 0.0, 0.0)  # Start Position
    ng.interface.items_tree[1].default_value = (10.0, 0.0, 0.0)  # End Position
    ng.interface.items_tree[2].default_value = 0.5  # Bridge Width
    ng.interface.items_tree[3].default_value = 0.2  # Twist Factor
    ng.interface.items_tree[4].default_value = 1000.0  # Particle Density

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Create curve between points
    curve_line = ng.nodes.new("GeometryNodeCurvePrimitiveLine")
    curve_line.mode = "POINTS"

    # Resample for detail
    resample_curve = ng.nodes.new("GeometryNodeResampleCurve")
    resample_curve.mode = "COUNT"
    resample_curve.inputs["Count"].default_value = 64

    # Add twist deformation
    twist_curve = ng.nodes.new("GeometryNodeCurveSplineType")

    # Create profile for bridge
    profile_curve = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
    profile_curve.mode = "RADIUS"

    # Convert to mesh
    curve_to_mesh = ng.nodes.new("GeometryNodeCurveToMesh")

    # Add particles along bridge
    distribute_particles = ng.nodes.new("GeometryNodeDistributePointsOnFaces")

    # Create particle geometry
    particle_sphere = ng.nodes.new("GeometryNodeMeshUVSphere")
    particle_sphere.inputs["Subdivisions"].default_value = 1
    particle_sphere.inputs["Radius"].default_value = 0.01

    # Instance particles
    instance_particles = ng.nodes.new("GeometryNodeInstanceOnPoints")

    # Join bridge and particles
    join_geometry = ng.nodes.new("GeometryNodeJoinGeometry")

    # Position nodes
    input_node.location = (-600, 0)
    curve_line.location = (-400, 0)
    resample_curve.location = (-200, 0)
    twist_curve.location = (-100, 0)
    profile_curve.location = (-200, 200)
    curve_to_mesh.location = (0, 0)
    distribute_particles.location = (200, 0)
    particle_sphere.location = (200, 200)
    instance_particles.location = (400, 0)
    join_geometry.location = (600, 0)
    output_node.location = (800, 0)

    # Connect nodes
    ng.links.new(input_node.outputs["Start Position"], curve_line.inputs["Start"])
    ng.links.new(input_node.outputs["End Position"], curve_line.inputs["End"])
    ng.links.new(curve_line.outputs["Curve"], resample_curve.inputs["Curve"])
    ng.links.new(resample_curve.outputs["Curve"], twist_curve.inputs["Curve"])
    ng.links.new(input_node.outputs["Bridge Width"], profile_curve.inputs["Radius"])
    ng.links.new(twist_curve.outputs["Curve"], curve_to_mesh.inputs["Curve"])
    ng.links.new(profile_curve.outputs["Curve"], curve_to_mesh.inputs["Profile Curve"])
    ng.links.new(curve_to_mesh.outputs["Mesh"], distribute_particles.inputs["Mesh"])
    ng.links.new(
        input_node.outputs["Particle Density"], distribute_particles.inputs["Density"]
    )
    ng.links.new(particle_sphere.outputs["Mesh"], instance_particles.inputs["Instance"])
    ng.links.new(
        distribute_particles.outputs["Points"], instance_particles.inputs["Points"]
    )
    ng.links.new(curve_to_mesh.outputs["Mesh"], join_geometry.inputs["Geometry"])
    ng.links.new(
        instance_particles.outputs["Instances"], join_geometry.inputs["Geometry"]
    )
    ng.links.new(join_geometry.outputs["Geometry"], output_node.inputs["Geometry"])

    return ng


# Filament presets
FILAMENT_PRESETS = {
    "cosmic_web": {
        "Filament Count": 25,
        "Filament Length": 35.0,
        "Filament Width": 0.03,
        "Curvature": 0.4,
        "Branch Factor": 0.3,
        "Density Variation": 0.6,
        "Random Seed": 42,
    },
    "galaxy_cluster": {
        "Filament Count": 15,
        "Filament Length": 20.0,
        "Filament Width": 0.08,
        "Curvature": 0.2,
        "Branch Factor": 0.5,
        "Density Variation": 0.8,
        "Random Seed": 123,
    },
    "dark_matter_web": {
        "Filament Count": 50,
        "Filament Length": 40.0,
        "Filament Width": 0.01,
        "Curvature": 0.8,
        "Branch Factor": 0.1,
        "Density Variation": 0.3,
        "Random Seed": 456,
    },
    "local_group": {
        "Filament Count": 8,
        "Filament Length": 12.0,
        "Filament Width": 0.15,
        "Curvature": 0.1,
        "Branch Factor": 0.8,
        "Density Variation": 0.9,
        "Random Seed": 789,
    },
}

# Galaxy bridge presets
GALAXY_BRIDGE_PRESETS = {
    "milky_way_andromeda": {
        "Start Position": (-2.5, 0.0, 0.0),
        "End Position": (2.5, 0.0, 0.0),
        "Bridge Width": 0.3,
        "Twist Factor": 0.1,
        "Particle Density": 2000.0,
    },
    "interacting_pair": {
        "Start Position": (-1.5, 0.0, 0.0),
        "End Position": (1.5, 0.5, 0.2),
        "Bridge Width": 0.5,
        "Twist Factor": 0.3,
        "Particle Density": 3000.0,
    },
    "merger_bridge": {
        "Start Position": (-0.8, 0.0, 0.0),
        "End Position": (0.8, 0.0, 0.0),
        "Bridge Width": 0.8,
        "Twist Factor": 0.5,
        "Particle Density": 5000.0,
    },
}


def get_filament_preset(preset_name: str) -> Dict[str, Any]:
    """Get filament preset configuration."""
    return FILAMENT_PRESETS.get(preset_name, FILAMENT_PRESETS["cosmic_web"])


def get_galaxy_bridge_preset(preset_name: str) -> Dict[str, Any]:
    """Get galaxy bridge preset configuration."""
    return GALAXY_BRIDGE_PRESETS.get(
        preset_name, GALAXY_BRIDGE_PRESETS["milky_way_andromeda"]
    )


def apply_filament_preset(obj: bpy.types.Object, preset_name: str) -> None:
    """Apply filament preset to geometry nodes modifier."""
    preset = get_filament_preset(preset_name)

    # Find geometry nodes modifier
    geom_modifier = None
    for modifier in obj.modifiers:
        if modifier.type == "NODES" and modifier.node_group:
            if "ALBPY_CosmicFilament" in modifier.node_group.name:
                geom_modifier = modifier
                break

    if geom_modifier:
        # Apply preset parameters
        for param_name, value in preset.items():
            if param_name in geom_modifier:
                geom_modifier[param_name] = value


def apply_galaxy_bridge_preset(obj: bpy.types.Object, preset_name: str) -> None:
    """Apply galaxy bridge preset to geometry nodes modifier."""
    preset = get_galaxy_bridge_preset(preset_name)

    # Find geometry nodes modifier
    geom_modifier = None
    for modifier in obj.modifiers:
        if modifier.type == "NODES" and modifier.node_group:
            if "ALBPY_GalaxyBridge" in modifier.node_group.name:
                geom_modifier = modifier
                break

    if geom_modifier:
        # Apply preset parameters
        for param_name, value in preset.items():
            if param_name in geom_modifier:
                geom_modifier[param_name] = value


def create_filament_object(name: str, preset: str = "cosmic_web") -> bpy.types.Object:
    """
    Create filament object with geometry nodes modifier.

    Args:
        name: Object name
        preset: Filament preset name

    Returns:
        bpy.types.Object: Created object with filament modifier
    """
    import bpy

    # Create empty mesh
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    obj = bpy.data.objects.new(name=name, object_data=mesh)

    # Add to scene
    bpy.context.collection.objects.link(obj)

    # Add geometry nodes modifier
    modifier = obj.modifiers.new(name="CosmicFilament", type="NODES")
    modifier.node_group = bpy.data.node_groups.get("ALBPY_CosmicFilament")

    # Apply preset
    apply_filament_preset(obj, preset)

    return obj


def create_galaxy_bridge_object(
    name: str, preset: str = "milky_way_andromeda"
) -> bpy.types.Object:
    """
    Create galaxy bridge object with geometry nodes modifier.

    Args:
        name: Object name
        preset: Galaxy bridge preset name

    Returns:
        bpy.types.Object: Created object with galaxy bridge modifier
    """
    import bpy

    # Create empty mesh
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    obj = bpy.data.objects.new(name=name, object_data=mesh)

    # Add to scene
    bpy.context.collection.objects.link(obj)

    # Add geometry nodes modifier
    modifier = obj.modifiers.new(name="GalaxyBridge", type="NODES")
    modifier.node_group = bpy.data.node_groups.get("ALBPY_GalaxyBridge")

    # Apply preset
    apply_galaxy_bridge_preset(obj, preset)

    return obj


def register():
    """Register filament geometry node groups."""
    if "ALBPY_CosmicFilament" not in bpy.data.node_groups:
        create_cosmic_filament_node_group()

    if "ALBPY_GalaxyBridge" not in bpy.data.node_groups:
        create_galaxy_bridge_node_group()


def unregister():
    """Unregister filament geometry node groups."""
    for group_name in ["ALBPY_CosmicFilament", "ALBPY_GalaxyBridge"]:
        if group_name in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
