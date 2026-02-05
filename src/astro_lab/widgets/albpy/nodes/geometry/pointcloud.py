"""
Point Cloud Geometry Node Groups
================================

Modern Blender 4.4 implementation for point cloud visualization using factory functions.
"""

from typing import Any, Dict

import bpy


def create_pointcloud_node_group():
    """
    Create point cloud geometry node group using modern Blender 4.4 API.

    Returns:
        bpy.types.GeometryNodeTree: The created point cloud node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_PointCloudVisualization", "GeometryNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    interface.new_socket(
        name="Point Count", in_out="INPUT", socket_type="NodeSocketInt"
    )
    interface.new_socket(
        name="Point Size", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Distribution Radius", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Density Falloff", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Random Seed", in_out="INPUT", socket_type="NodeSocketInt"
    )

    # Output sockets
    interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    # Set default values
    ng.interface.items_tree[1].default_value = 5000  # Point Count
    ng.interface.items_tree[2].default_value = 0.01  # Point Size
    ng.interface.items_tree[3].default_value = 12.0  # Distribution Radius
    ng.interface.items_tree[4].default_value = 2.0  # Density Falloff
    ng.interface.items_tree[5].default_value = 0  # Random Seed

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Create sphere for point distribution
    sphere = ng.nodes.new("GeometryNodeMeshUVSphere")
    sphere.inputs["Subdivisions"].default_value = 3

    # Smooth shading for distribution sphere
    set_smooth_dist = ng.nodes.new("GeometryNodeSetShadeSmooth")
    set_smooth_dist.inputs["Shade Smooth"].default_value = True

    # Distribute points on sphere surface
    distribute_points = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute_points.distribute_method = "POISSON"

    # Create point geometry (small sphere)
    point_sphere = ng.nodes.new("GeometryNodeMeshUVSphere")
    point_sphere.inputs["Subdivisions"].default_value = 2

    # Smooth shading for points
    set_smooth_point = ng.nodes.new("GeometryNodeSetShadeSmooth")
    set_smooth_point.inputs["Shade Smooth"].default_value = True

    # Instance on points
    instance_on_points = ng.nodes.new("GeometryNodeInstanceOnPoints")

    # Random size variation
    random_size = ng.nodes.new("GeometryNodeInputRandomValue")
    random_size.data_type = "FLOAT_VECTOR"
    random_size.inputs["Min"].default_value = (0.3, 0.3, 0.3)
    random_size.inputs["Max"].default_value = (1.5, 1.5, 1.5)

    # Scale instances
    scale_instances = ng.nodes.new("GeometryNodeScaleInstances")

    # Add noise for density variation
    noise = ng.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 2.0
    noise.inputs["Roughness"].default_value = 0.5

    # Color ramp for density mapping
    color_ramp = ng.nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.3
    color_ramp.color_ramp.elements[1].position = 0.7

    # Math node for density scaling
    density_scale = ng.nodes.new("ShaderNodeMath")
    density_scale.operation = "MULTIPLY"

    # Position-based density falloff
    position_node = ng.nodes.new("GeometryNodeInputPosition")

    # Vector length for distance from center
    vector_length = ng.nodes.new("ShaderNodeVectorMath")
    vector_length.operation = "LENGTH"

    # Math node for falloff calculation
    falloff_power = ng.nodes.new("ShaderNodeMath")
    falloff_power.operation = "POWER"

    falloff_invert = ng.nodes.new("ShaderNodeMath")
    falloff_invert.operation = "SUBTRACT"
    falloff_invert.inputs[0].default_value = 1.0

    # Position nodes
    input_node.location = (-1200, 0)
    sphere.location = (-1000, 0)
    set_smooth_dist.location = (-900, 0)
    distribute_points.location = (-700, 0)

    point_sphere.location = (-700, 300)
    set_smooth_point.location = (-600, 300)

    random_size.location = (-700, -200)
    scale_instances.location = (-500, 0)
    instance_on_points.location = (-300, 0)

    # Noise and density
    position_node.location = (-900, -300)
    vector_length.location = (-700, -300)
    falloff_power.location = (-500, -300)
    falloff_invert.location = (-300, -300)
    noise.location = (-500, -400)
    color_ramp.location = (-300, -400)
    density_scale.location = (-100, -300)

    output_node.location = (100, 0)

    # Connect nodes
    # Distribution sphere setup
    ng.links.new(input_node.outputs["Distribution Radius"], sphere.inputs["Radius"])
    ng.links.new(sphere.outputs["Mesh"], set_smooth_dist.inputs["Geometry"])
    ng.links.new(set_smooth_dist.outputs["Geometry"], distribute_points.inputs["Mesh"])

    # Point count and distribution
    ng.links.new(input_node.outputs["Point Count"], distribute_points.inputs["Density"])
    ng.links.new(input_node.outputs["Random Seed"], distribute_points.inputs["Seed"])

    # Point geometry
    ng.links.new(input_node.outputs["Point Size"], point_sphere.inputs["Radius"])
    ng.links.new(point_sphere.outputs["Mesh"], set_smooth_point.inputs["Geometry"])
    ng.links.new(
        set_smooth_point.outputs["Geometry"], instance_on_points.inputs["Instance"]
    )

    # Instance setup
    ng.links.new(
        distribute_points.outputs["Points"], instance_on_points.inputs["Points"]
    )
    ng.links.new(
        instance_on_points.outputs["Instances"], scale_instances.inputs["Instances"]
    )

    # Random scaling
    ng.links.new(input_node.outputs["Random Seed"], random_size.inputs["Seed"])
    ng.links.new(random_size.outputs["Value"], scale_instances.inputs["Scale"])

    # Density falloff
    ng.links.new(position_node.outputs["Position"], vector_length.inputs["Vector"])
    ng.links.new(vector_length.outputs["Value"], falloff_power.inputs[0])
    ng.links.new(input_node.outputs["Density Falloff"], falloff_power.inputs[1])
    ng.links.new(falloff_power.outputs["Value"], falloff_invert.inputs[1])

    # Noise-based density variation
    ng.links.new(position_node.outputs["Position"], noise.inputs["Vector"])
    ng.links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    ng.links.new(color_ramp.outputs["Color"], density_scale.inputs[0])
    ng.links.new(falloff_invert.outputs["Value"], density_scale.inputs[1])

    # Apply density to distribution
    ng.links.new(density_scale.outputs["Value"], distribute_points.inputs["Density"])

    # Final output
    ng.links.new(scale_instances.outputs["Instances"], output_node.inputs["Geometry"])

    return ng


def create_stellar_pointcloud_node_group():
    """
    Create stellar point cloud node group for astronomical visualizations.

    Returns:
        bpy.types.GeometryNodeTree: The created stellar point cloud node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_StellarPointCloud", "GeometryNodeTree")

    # Interface API
    interface = ng.interface

    # Input sockets
    interface.new_socket(name="Star Count", in_out="INPUT", socket_type="NodeSocketInt")
    interface.new_socket(
        name="Field Radius", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Magnitude Range", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Color Temperature", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Disk Thickness", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    # Set defaults
    ng.interface.items_tree[0].default_value = 10000  # Star Count
    ng.interface.items_tree[1].default_value = 50.0  # Field Radius
    ng.interface.items_tree[2].default_value = 5.0  # Magnitude Range
    ng.interface.items_tree[3].default_value = 5778.0  # Color Temperature
    ng.interface.items_tree[4].default_value = 0.1  # Disk Thickness

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Create disk geometry for galactic distribution
    cylinder = ng.nodes.new("GeometryNodeMeshCylinder")
    cylinder.fill_type = "NOTHING"  # Hollow cylinder

    # Scale for disk shape
    scale_disk = ng.nodes.new("GeometryNodeScaleElements")
    scale_disk.domain = "POINT"

    # Distribute points in disk
    distribute_points = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute_points.distribute_method = "RANDOM"

    # Create star geometry
    star_sphere = ng.nodes.new("GeometryNodeMeshUVSphere")
    star_sphere.inputs["Subdivisions"].default_value = 1

    # Instance stars
    instance_on_points = ng.nodes.new("GeometryNodeInstanceOnPoints")

    # Random size based on magnitude
    random_magnitude = ng.nodes.new("GeometryNodeInputRandomValue")
    random_magnitude.data_type = "FLOAT"
    random_magnitude.inputs["Min"].default_value = 0.1
    random_magnitude.inputs["Max"].default_value = 2.0

    # Scale by magnitude
    scale_by_magnitude = ng.nodes.new("ShaderNodeMath")
    scale_by_magnitude.operation = "DIVIDE"
    scale_by_magnitude.inputs[0].default_value = 1.0

    # Apply scaling
    scale_instances = ng.nodes.new("GeometryNodeScaleInstances")

    # Position nodes
    input_node.location = (-800, 0)
    cylinder.location = (-600, 0)
    scale_disk.location = (-400, 0)
    distribute_points.location = (-200, 0)
    star_sphere.location = (-200, 200)
    instance_on_points.location = (0, 0)
    random_magnitude.location = (0, -200)
    scale_by_magnitude.location = (200, -200)
    scale_instances.location = (400, 0)
    output_node.location = (600, 0)

    # Connect nodes
    ng.links.new(input_node.outputs["Field Radius"], cylinder.inputs["Radius"])
    ng.links.new(input_node.outputs["Disk Thickness"], cylinder.inputs["Depth"])
    ng.links.new(cylinder.outputs["Mesh"], scale_disk.inputs["Geometry"])
    ng.links.new(scale_disk.outputs["Geometry"], distribute_points.inputs["Mesh"])
    ng.links.new(input_node.outputs["Star Count"], distribute_points.inputs["Density"])

    ng.links.new(input_node.outputs["Magnitude Range"], star_sphere.inputs["Radius"])
    ng.links.new(star_sphere.outputs["Mesh"], instance_on_points.inputs["Instance"])
    ng.links.new(
        distribute_points.outputs["Points"], instance_on_points.inputs["Points"]
    )

    ng.links.new(random_magnitude.outputs["Value"], scale_by_magnitude.inputs[1])
    ng.links.new(scale_by_magnitude.outputs["Value"], scale_instances.inputs["Scale"])
    ng.links.new(
        instance_on_points.outputs["Instances"], scale_instances.inputs["Instances"]
    )
    ng.links.new(scale_instances.outputs["Instances"], output_node.inputs["Geometry"])

    return ng


# Point cloud presets
POINTCLOUD_PRESETS = {
    "star_field": {
        "Point Count": 10000,
        "Point Size": 0.005,
        "Distribution Radius": 20.0,
        "Density Falloff": 1.5,
        "Random Seed": 42,
    },
    "galaxy_cluster": {
        "Point Count": 2000,
        "Point Size": 0.02,
        "Distribution Radius": 8.0,
        "Density Falloff": 3.0,
        "Random Seed": 123,
    },
    "cosmic_background": {
        "Point Count": 50000,
        "Point Size": 0.002,
        "Distribution Radius": 30.0,
        "Density Falloff": 1.0,
        "Random Seed": 456,
    },
    "nebula_points": {
        "Point Count": 15000,
        "Point Size": 0.008,
        "Distribution Radius": 15.0,
        "Density Falloff": 2.5,
        "Random Seed": 789,
    },
}


# Stellar point cloud presets
STELLAR_POINTCLOUD_PRESETS = {
    "local_neighborhood": {
        "Star Count": 1000,
        "Field Radius": 10.0,
        "Magnitude Range": 0.5,
        "Color Temperature": 5778.0,
        "Disk Thickness": 0.05,
    },
    "galactic_disk": {
        "Star Count": 25000,
        "Field Radius": 50.0,
        "Magnitude Range": 2.0,
        "Color Temperature": 4500.0,
        "Disk Thickness": 0.1,
    },
    "globular_cluster": {
        "Star Count": 5000,
        "Field Radius": 5.0,
        "Magnitude Range": 1.0,
        "Color Temperature": 3500.0,
        "Disk Thickness": 1.0,  # Spherical
    },
    "open_cluster": {
        "Star Count": 500,
        "Field Radius": 3.0,
        "Magnitude Range": 0.8,
        "Color Temperature": 8000.0,
        "Disk Thickness": 0.2,
    },
}


def get_pointcloud_preset(preset_name: str) -> Dict[str, Any]:
    """Get point cloud preset configuration."""
    return POINTCLOUD_PRESETS.get(preset_name, POINTCLOUD_PRESETS["star_field"])


def get_stellar_pointcloud_preset(preset_name: str) -> Dict[str, Any]:
    """Get stellar point cloud preset configuration."""
    return STELLAR_POINTCLOUD_PRESETS.get(
        preset_name, STELLAR_POINTCLOUD_PRESETS["local_neighborhood"]
    )


def apply_pointcloud_preset(obj: bpy.types.Object, preset_name: str) -> None:
    """Apply point cloud preset to geometry nodes modifier."""
    preset = get_pointcloud_preset(preset_name)

    # Find geometry nodes modifier
    geom_modifier = None
    for modifier in obj.modifiers:
        if modifier.type == "NODES" and modifier.node_group:
            if "ALBPY_PointCloud" in modifier.node_group.name:
                geom_modifier = modifier
                break

    if geom_modifier:
        # Apply preset parameters
        for param_name, value in preset.items():
            if param_name in geom_modifier:
                geom_modifier[param_name] = value


def apply_stellar_pointcloud_preset(obj: bpy.types.Object, preset_name: str) -> None:
    """Apply stellar point cloud preset to geometry nodes modifier."""
    preset = get_stellar_pointcloud_preset(preset_name)

    # Find geometry nodes modifier
    geom_modifier = None
    for modifier in obj.modifiers:
        if modifier.type == "NODES" and modifier.node_group:
            if "ALBPY_StellarPointCloud" in modifier.node_group.name:
                geom_modifier = modifier
                break

    if geom_modifier:
        # Apply preset parameters
        for param_name, value in preset.items():
            if param_name in geom_modifier:
                geom_modifier[param_name] = value


def create_pointcloud_object(name: str, preset: str = "star_field") -> bpy.types.Object:
    """
    Create point cloud object with geometry nodes modifier.

    Args:
        name: Object name
        preset: Point cloud preset name

    Returns:
        bpy.types.Object: Created object with point cloud modifier
    """
    import bpy

    # Create empty mesh
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    obj = bpy.data.objects.new(name=name, object_data=mesh)

    # Add to scene
    bpy.context.collection.objects.link(obj)

    # Add geometry nodes modifier
    modifier = obj.modifiers.new(name="PointCloud", type="NODES")
    modifier.node_group = bpy.data.node_groups.get("ALBPY_PointCloudVisualization")

    # Apply preset
    apply_pointcloud_preset(obj, preset)

    return obj


def create_stellar_pointcloud_object(
    name: str, preset: str = "local_neighborhood"
) -> bpy.types.Object:
    """
    Create stellar point cloud object with geometry nodes modifier.

    Args:
        name: Object name
        preset: Stellar point cloud preset name

    Returns:
        bpy.types.Object: Created object with stellar point cloud modifier
    """
    import bpy

    # Create empty mesh
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    obj = bpy.data.objects.new(name=name, object_data=mesh)

    # Add to scene
    bpy.context.collection.objects.link(obj)

    # Add geometry nodes modifier
    modifier = obj.modifiers.new(name="StellarPointCloud", type="NODES")
    modifier.node_group = bpy.data.node_groups.get("ALBPY_StellarPointCloud")

    # Apply preset
    apply_stellar_pointcloud_preset(obj, preset)

    return obj


def register():
    """Register point cloud geometry node groups."""
    if "ALBPY_PointCloudVisualization" not in bpy.data.node_groups:
        create_pointcloud_node_group()

    if "ALBPY_StellarPointCloud" not in bpy.data.node_groups:
        create_stellar_pointcloud_node_group()


def unregister():
    """Unregister point cloud geometry node groups."""
    for group_name in ["ALBPY_PointCloudVisualization", "ALBPY_StellarPointCloud"]:
        if group_name in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
