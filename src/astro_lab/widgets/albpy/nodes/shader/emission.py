"""
Emission Shader Node Groups
===========================

Modern Blender 4.4 implementation for emission shaders with astronomical applications.
"""

from typing import Any, Dict

import bpy


def create_emission_node_group():
    """
    Create emission shader node group using modern Blender 4.4 API.

    Returns:
        bpy.types.ShaderNodeTree: The created emission shader node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_EmissionShader", "ShaderNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input sockets
    interface.new_socket(name="Color", in_out="INPUT", socket_type="NodeSocketColor")
    interface.new_socket(name="Strength", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Temperature", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Use Blackbody", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Set default values
    ng.interface.items_tree[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Color
    ng.interface.items_tree[1].default_value = 5.0  # Strength
    ng.interface.items_tree[2].default_value = 5778.0  # Temperature
    ng.interface.items_tree[3].default_value = 0.0  # Use Blackbody

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Emission shader
    emission = ng.nodes.new("ShaderNodeEmission")

    # Blackbody node for temperature-based emission
    blackbody = ng.nodes.new("ShaderNodeBlackbody")

    # Mix node for choosing between color and blackbody
    mix_color = ng.nodes.new("ShaderNodeMix")
    mix_color.data_type = "RGBA"
    mix_color.blend_type = "MIX"

    # Position nodes
    input_node.location = (-400, 0)
    blackbody.location = (-200, 100)
    mix_color.location = (0, 0)
    emission.location = (200, 0)
    output_node.location = (400, 0)

    # Connect nodes
    ng.links.new(input_node.outputs["Temperature"], blackbody.inputs["Temperature"])
    ng.links.new(input_node.outputs["Color"], mix_color.inputs["Color1"])
    ng.links.new(blackbody.outputs["Color"], mix_color.inputs["Color2"])
    ng.links.new(input_node.outputs["Use Blackbody"], mix_color.inputs["Fac"])
    ng.links.new(mix_color.outputs["Result"], emission.inputs["Color"])
    ng.links.new(input_node.outputs["Strength"], emission.inputs["Strength"])
    ng.links.new(emission.outputs["Emission"], output_node.inputs["Shader"])

    return ng


def create_advanced_emission_node_group():
    """
    Create advanced emission shader with multiple emission types.

    Returns:
        bpy.types.ShaderNodeTree: The created advanced emission node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_AdvancedEmission", "ShaderNodeTree")

    # Interface API
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Base Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(
        name="Emission Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(name="Strength", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Temperature", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Glow Radius", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Fresnel Mix", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Set defaults
    ng.interface.items_tree[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Base Color
    ng.interface.items_tree[1].default_value = (1.0, 0.5, 0.2, 1.0)  # Emission Color
    ng.interface.items_tree[2].default_value = 3.0  # Strength
    ng.interface.items_tree[3].default_value = 6500.0  # Temperature
    ng.interface.items_tree[4].default_value = 0.1  # Glow Radius
    ng.interface.items_tree[5].default_value = 0.2  # Fresnel Mix

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Core emission
    emission = ng.nodes.new("ShaderNodeEmission")

    # Blackbody for temperature
    blackbody = ng.nodes.new("ShaderNodeBlackbody")

    # Fresnel for edge emission
    fresnel = ng.nodes.new("ShaderNodeFresnel")

    # Mix shaders
    mix_shader = ng.nodes.new("ShaderNodeMix")
    mix_shader.data_type = "RGBA"

    # Color ramp for glow effects
    color_ramp = ng.nodes.new("ShaderNodeValToRGB")

    # Position nodes
    input_node.location = (-600, 0)
    blackbody.location = (-400, 100)
    fresnel.location = (-400, -100)
    color_ramp.location = (-200, -100)
    mix_shader.location = (0, 0)
    emission.location = (200, 0)
    output_node.location = (400, 0)

    # Connect nodes
    ng.links.new(input_node.outputs["Temperature"], blackbody.inputs["Temperature"])
    ng.links.new(input_node.outputs["Glow Radius"], fresnel.inputs["IOR"])
    ng.links.new(fresnel.outputs["Fac"], color_ramp.inputs["Fac"])
    ng.links.new(input_node.outputs["Base Color"], mix_shader.inputs["Color1"])
    ng.links.new(blackbody.outputs["Color"], mix_shader.inputs["Color2"])
    ng.links.new(input_node.outputs["Fresnel Mix"], mix_shader.inputs["Fac"])
    ng.links.new(mix_shader.outputs["Result"], emission.inputs["Color"])
    ng.links.new(input_node.outputs["Strength"], emission.inputs["Strength"])
    ng.links.new(emission.outputs["Emission"], output_node.inputs["Shader"])

    return ng


# Emission presets for different astronomical objects
EMISSION_PRESETS = {
    "star_core": {
        "Color": (1.0, 1.0, 1.0, 1.0),
        "Strength": 10.0,
        "Temperature": 5778.0,
        "Use Blackbody": 1.0,
    },
    "nebula_emission": {
        "Color": (1.0, 0.3, 0.5, 1.0),
        "Strength": 2.0,
        "Temperature": 8000.0,
        "Use Blackbody": 0.3,
    },
    "galaxy_core": {
        "Color": (1.0, 0.8, 0.6, 1.0),
        "Strength": 5.0,
        "Temperature": 4000.0,
        "Use Blackbody": 0.7,
    },
    "supernova": {
        "Color": (0.8, 0.9, 1.0, 1.0),
        "Strength": 50.0,
        "Temperature": 15000.0,
        "Use Blackbody": 1.0,
    },
    "pulsar": {
        "Color": (0.6, 0.8, 1.0, 1.0),
        "Strength": 20.0,
        "Temperature": 100000.0,
        "Use Blackbody": 1.0,
    },
}


def get_emission_preset(preset_name: str) -> Dict[str, Any]:
    """Get emission preset configuration."""
    return EMISSION_PRESETS.get(preset_name, EMISSION_PRESETS["star_core"])


def apply_emission_preset(material: bpy.types.Material, preset_name: str) -> None:
    """Apply emission preset to material."""
    preset = get_emission_preset(preset_name)

    if not material.use_nodes:
        material.use_nodes = True

    # Find emission node group
    emission_node = None
    for node in material.node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_Emission" in node.node_tree.name:
                emission_node = node
                break

    if emission_node:
        # Apply preset parameters
        for param_name, value in preset.items():
            if param_name in emission_node.inputs:
                emission_node.inputs[param_name].default_value = value


def create_emission_material(
    name: str, preset: str = "star_core"
) -> bpy.types.Material:
    """
    Create emission material with node group.

    Args:
        name: Material name
        preset: Emission preset name

    Returns:
        bpy.types.Material: Created material
    """
    # Create material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    # Clear default nodes
    mat.node_tree.nodes.clear()

    # Add emission node group
    emission_group = mat.node_tree.nodes.new("ShaderNodeGroup")
    emission_group.node_tree = bpy.data.node_groups.get("ALBPY_EmissionShader")

    # Add material output
    output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")

    # Position nodes
    emission_group.location = (0, 0)
    output.location = (300, 0)

    # Connect nodes
    mat.node_tree.links.new(emission_group.outputs["Shader"], output.inputs["Surface"])

    # Apply preset
    apply_emission_preset(mat, preset)

    return mat


def register():
    """Register emission shader node groups."""
    if "ALBPY_EmissionShader" not in bpy.data.node_groups:
        create_emission_node_group()

    if "ALBPY_AdvancedEmission" not in bpy.data.node_groups:
        create_advanced_emission_node_group()


def unregister():
    """Unregister emission shader node groups."""
    for group_name in ["ALBPY_EmissionShader", "ALBPY_AdvancedEmission"]:
        if group_name in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
