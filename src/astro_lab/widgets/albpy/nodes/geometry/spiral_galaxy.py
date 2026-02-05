"""
Spiral Galaxy Geometry Node Group
=================================

Modern Blender 4.4 implementation using factory functions and interface API.
"""

from typing import Any, Dict

import bpy


def create_spiral_galaxy_node_group():
    """
    Erstellt eine Spiral Galaxy Geometry Node Group mit moderner Blender 4.4 API.

    Returns:
        bpy.types.GeometryNodeTree: Die erstellte Node Group
    """
    # Node-Group erstellen
    ng = bpy.data.node_groups.new("ALBPY_SpiralGalaxy", "GeometryNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input-Sockets
    interface.new_socket(name="Star Count", in_out="INPUT", socket_type="NodeSocketInt")
    interface.new_socket(
        name="Galaxy Radius", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Number of Arms", in_out="INPUT", socket_type="NodeSocketInt"
    )
    interface.new_socket(
        name="Arm Tightness", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Central Bulge", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Disk Thickness", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output-Sockets
    interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    # Default-Werte für realistische Galaxie
    ng.interface.items_tree[0].default_value = 25000  # Star Count
    ng.interface.items_tree[1].default_value = 50.0  # Galaxy Radius (kpc)
    ng.interface.items_tree[2].default_value = 2  # Number of Arms
    ng.interface.items_tree[3].default_value = 0.3  # Arm Tightness
    ng.interface.items_tree[4].default_value = 0.2  # Central Bulge
    ng.interface.items_tree[5].default_value = 0.1  # Disk Thickness

    # Node-Erstellung
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Spiral-Disk erstellen
    cylinder = ng.nodes.new("GeometryNodeMeshCylinder")
    cylinder.fill_type = "NGON"

    # Smooth shading für Disk
    set_smooth_disk = ng.nodes.new("GeometryNodeSetShadeSmooth")
    set_smooth_disk.inputs["Shade Smooth"].default_value = True

    # Punkte auf Disk verteilen
    distribute = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute.distribute_method = "RANDOM"

    # Spiral-Transformation für Arme
    position_node = ng.nodes.new("GeometryNodeInputPosition")
    separate_xyz = ng.nodes.new("ShaderNodeSeparateXYZ")
    combine_xyz = ng.nodes.new("ShaderNodeCombineXYZ")

    # Mathematische Nodes für Spiral-Berechnung
    math_multiply = ng.nodes.new("ShaderNodeMath")
    math_multiply.operation = "MULTIPLY"

    math_add = ng.nodes.new("ShaderNodeMath")
    math_add.operation = "ADD"

    # Set Position für Spiral-Effekt
    set_position = ng.nodes.new("GeometryNodeSetPosition")

    # Sterne instanziieren
    ico_sphere = ng.nodes.new("GeometryNodeMeshIcoSphere")
    ico_sphere.inputs["Subdivisions"].default_value = 1
    ico_sphere.inputs["Radius"].default_value = 0.02

    # Smooth shading für Sterne
    set_smooth_star = ng.nodes.new("GeometryNodeSetShadeSmooth")
    set_smooth_star.inputs["Shade Smooth"].default_value = True

    instance = ng.nodes.new("GeometryNodeInstanceOnPoints")

    # Random Scale für realistische Sterne
    random_value = ng.nodes.new("FunctionNodeRandomValue")
    random_value.data_type = "FLOAT"
    random_value.inputs["Min"].default_value = 0.5
    random_value.inputs["Max"].default_value = 2.0

    # Positionierung der Nodes
    input_node.location = (-800, 0)
    cylinder.location = (-600, 0)
    set_smooth_disk.location = (-450, 0)
    distribute.location = (-300, 0)

    position_node.location = (-150, 200)
    separate_xyz.location = (0, 200)
    math_multiply.location = (150, 200)
    math_add.location = (300, 200)
    combine_xyz.location = (450, 200)
    set_position.location = (600, 0)

    ico_sphere.location = (-150, -200)
    set_smooth_star.location = (0, -200)
    random_value.location = (150, -200)
    instance.location = (750, 0)
    output_node.location = (900, 0)

    # Verbindungen - Basis-Geometrie
    ng.links.new(input_node.outputs["Galaxy Radius"], cylinder.inputs["Radius"])
    ng.links.new(input_node.outputs["Disk Thickness"], cylinder.inputs["Depth"])
    ng.links.new(cylinder.outputs["Mesh"], set_smooth_disk.inputs["Geometry"])
    ng.links.new(set_smooth_disk.outputs["Geometry"], distribute.inputs["Mesh"])
    ng.links.new(input_node.outputs["Star Count"], distribute.inputs["Density"])

    # Spiral-Transformation
    ng.links.new(distribute.outputs["Points"], set_position.inputs["Geometry"])
    ng.links.new(position_node.outputs["Position"], separate_xyz.inputs["Vector"])
    ng.links.new(separate_xyz.outputs["X"], math_multiply.inputs[0])
    ng.links.new(input_node.outputs["Arm Tightness"], math_multiply.inputs[1])
    ng.links.new(math_multiply.outputs["Value"], math_add.inputs[0])
    ng.links.new(separate_xyz.outputs["Y"], math_add.inputs[1])
    ng.links.new(separate_xyz.outputs["X"], combine_xyz.inputs["X"])
    ng.links.new(math_add.outputs["Value"], combine_xyz.inputs["Y"])
    ng.links.new(separate_xyz.outputs["Z"], combine_xyz.inputs["Z"])
    ng.links.new(combine_xyz.outputs["Vector"], set_position.inputs["Position"])

    # Sterne-Instanziierung
    ng.links.new(ico_sphere.outputs["Mesh"], set_smooth_star.inputs["Geometry"])
    ng.links.new(set_smooth_star.outputs["Geometry"], instance.inputs["Instance"])
    ng.links.new(set_position.outputs["Geometry"], instance.inputs["Points"])
    ng.links.new(random_value.outputs["Value"], instance.inputs["Scale"])

    # Final Output
    ng.links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])

    return ng


# Spiral galaxy presets
SPIRAL_GALAXY_PRESETS = {
    "milky_way": {
        "Star Count": 25000,
        "Galaxy Radius": 50.0,
        "Number of Arms": 4,
        "Arm Tightness": 0.3,
        "Central Bulge": 0.2,
        "Disk Thickness": 0.1,
    },
    "andromeda": {
        "Star Count": 40000,
        "Galaxy Radius": 60.0,
        "Number of Arms": 2,
        "Arm Tightness": 0.4,
        "Central Bulge": 0.3,
        "Disk Thickness": 0.15,
    },
    "grand_design": {
        "Star Count": 30000,
        "Galaxy Radius": 45.0,
        "Number of Arms": 2,
        "Arm Tightness": 0.5,
        "Central Bulge": 0.15,
        "Disk Thickness": 0.08,
    },
    "flocculent": {
        "Star Count": 20000,
        "Galaxy Radius": 35.0,
        "Number of Arms": 6,
        "Arm Tightness": 0.2,
        "Central Bulge": 0.25,
        "Disk Thickness": 0.12,
    },
}


def get_spiral_galaxy_preset(preset_name: str) -> Dict[str, Any]:
    """Get preset for spiral galaxy configuration."""
    return SPIRAL_GALAXY_PRESETS.get(preset_name, SPIRAL_GALAXY_PRESETS["milky_way"])


def apply_spiral_galaxy_preset(obj: bpy.types.Object, preset_name: str) -> None:
    """Apply spiral galaxy preset to object with geometry nodes modifier."""
    preset = get_spiral_galaxy_preset(preset_name)

    # Find geometry nodes modifier
    for modifier in obj.modifiers:
        if modifier.type == "NODES" and modifier.node_group:
            if "ALBPY_SpiralGalaxy" in modifier.node_group.name:
                # Apply preset parameters
                for param_name, value in preset.items():
                    input_name = f"Input_{param_name.replace(' ', '_')}"
                    if input_name in modifier:
                        modifier[input_name] = value
                break


def register():
    """Register spiral galaxy node group using factory function."""
    if "ALBPY_SpiralGalaxy" not in bpy.data.node_groups:
        create_spiral_galaxy_node_group()


def unregister():
    """Unregister spiral galaxy node group."""
    if "ALBPY_SpiralGalaxy" in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups["ALBPY_SpiralGalaxy"])
