"""
Doppler Effect Shader Node Groups
=================================

Modern Blender 4.4 implementation for Doppler effect visualization in astronomical contexts.
"""

from typing import Any, Dict

import bpy


def create_doppler_effect_node_group():
    """
    Create Doppler effect shader node group using modern Blender 4.4 API.

    Returns:
        bpy.types.ShaderNodeTree: The created Doppler effect shader node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_DopplerEffect", "ShaderNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Base Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(name="Velocity", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Rest Wavelength", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Emission Strength", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Doppler Intensity", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Set default values
    ng.interface.items_tree[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Base Color
    ng.interface.items_tree[1].default_value = 0.0  # Velocity (km/s)
    ng.interface.items_tree[2].default_value = 656.3  # Rest Wavelength (Hα line)
    ng.interface.items_tree[3].default_value = 5.0  # Emission Strength
    ng.interface.items_tree[4].default_value = 1.0  # Doppler Intensity

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Emission shader
    emission = ng.nodes.new("ShaderNodeEmission")

    # Math nodes for Doppler calculation
    # Velocity normalization (divide by speed of light)
    velocity_norm = ng.nodes.new("ShaderNodeMath")
    velocity_norm.operation = "DIVIDE"
    velocity_norm.inputs[1].default_value = 299792.458  # Speed of light in km/s

    # Doppler factor: sqrt((1 - v/c) / (1 + v/c))
    one_minus_v = ng.nodes.new("ShaderNodeMath")
    one_minus_v.operation = "SUBTRACT"
    one_minus_v.inputs[0].default_value = 1.0

    one_plus_v = ng.nodes.new("ShaderNodeMath")
    one_plus_v.operation = "ADD"
    one_plus_v.inputs[0].default_value = 1.0

    divide_factors = ng.nodes.new("ShaderNodeMath")
    divide_factors.operation = "DIVIDE"

    sqrt_doppler = ng.nodes.new("ShaderNodeMath")
    sqrt_doppler.operation = "POWER"
    sqrt_doppler.inputs[1].default_value = 0.5

    # Wavelength shift
    wavelength_shift = ng.nodes.new("ShaderNodeMath")
    wavelength_shift.operation = "MULTIPLY"

    # Convert wavelength to color (simplified)
    wavelength_to_hue = ng.nodes.new("ShaderNodeMath")
    wavelength_to_hue.operation = "DIVIDE"
    wavelength_to_hue.inputs[1].default_value = 700.0  # Normalize to visible range

    # HSV to RGB conversion
    hsv_node = ng.nodes.new("ShaderNodeHueSaturation")
    hsv_node.inputs["Saturation"].default_value = 1.0
    hsv_node.inputs["Value"].default_value = 1.0

    # Color ramp for redshift/blueshift
    color_ramp = ng.nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.3, 0.3, 1.0, 1.0)  # Blue (approaching)
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (1.0, 0.3, 0.3, 1.0)  # Red (receding)

    # Mix between base color and Doppler color
    mix_color = ng.nodes.new("ShaderNodeMix")
    mix_color.data_type = "RGBA"
    mix_color.blend_type = "MIX"

    # Position nodes
    input_node.location = (-800, 0)
    velocity_norm.location = (-600, 200)
    one_minus_v.location = (-600, 100)
    one_plus_v.location = (-600, 0)
    divide_factors.location = (-400, 100)
    sqrt_doppler.location = (-400, 0)
    wavelength_shift.location = (-400, -100)
    wavelength_to_hue.location = (-200, -100)
    color_ramp.location = (-200, 100)
    hsv_node.location = (-200, 200)
    mix_color.location = (0, 0)
    emission.location = (200, 0)
    output_node.location = (400, 0)

    # Connect nodes
    # Doppler calculations
    ng.links.new(input_node.outputs["Velocity"], velocity_norm.inputs[0])
    ng.links.new(velocity_norm.outputs["Value"], one_minus_v.inputs[1])
    ng.links.new(velocity_norm.outputs["Value"], one_plus_v.inputs[1])
    ng.links.new(one_minus_v.outputs["Value"], divide_factors.inputs[0])
    ng.links.new(one_plus_v.outputs["Value"], divide_factors.inputs[1])
    ng.links.new(divide_factors.outputs["Value"], sqrt_doppler.inputs[0])

    # Wavelength calculation
    ng.links.new(input_node.outputs["Rest Wavelength"], wavelength_shift.inputs[0])
    ng.links.new(sqrt_doppler.outputs["Value"], wavelength_shift.inputs[1])
    ng.links.new(wavelength_shift.outputs["Value"], wavelength_to_hue.inputs[0])

    # Color mapping
    ng.links.new(wavelength_to_hue.outputs["Value"], color_ramp.inputs["Fac"])
    ng.links.new(wavelength_to_hue.outputs["Value"], hsv_node.inputs["Hue"])

    # Color mixing
    ng.links.new(input_node.outputs["Base Color"], mix_color.inputs["Color1"])
    ng.links.new(color_ramp.outputs["Color"], mix_color.inputs["Color2"])
    ng.links.new(input_node.outputs["Doppler Intensity"], mix_color.inputs["Fac"])

    # Final emission
    ng.links.new(mix_color.outputs["Result"], emission.inputs["Color"])
    ng.links.new(input_node.outputs["Emission Strength"], emission.inputs["Strength"])
    ng.links.new(emission.outputs["Emission"], output_node.inputs["Shader"])

    return ng


def create_redshift_visualization_node_group():
    """
    Create redshift visualization shader for cosmological applications.

    Returns:
        bpy.types.ShaderNodeTree: The created redshift visualization node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_RedshiftVisualization", "ShaderNodeTree")

    # Interface API
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Base Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(name="Redshift", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Emission Strength", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Distance Dimming", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Set defaults
    ng.interface.items_tree[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Base Color
    ng.interface.items_tree[1].default_value = 0.0  # Redshift (z)
    ng.interface.items_tree[2].default_value = 3.0  # Emission Strength
    ng.interface.items_tree[3].default_value = 1.0  # Distance Dimming

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Emission shader
    emission = ng.nodes.new("ShaderNodeEmission")

    # Redshift color mapping
    redshift_ramp = ng.nodes.new("ShaderNodeValToRGB")
    redshift_ramp.color_ramp.elements[0].position = 0.0
    redshift_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)  # z=0 (white)
    redshift_ramp.color_ramp.elements[1].position = 1.0
    redshift_ramp.color_ramp.elements[1].color = (1.0, 0.2, 0.2, 1.0)  # High z (red)

    # Distance dimming calculation
    distance_factor = ng.nodes.new("ShaderNodeMath")
    distance_factor.operation = "DIVIDE"
    distance_factor.inputs[0].default_value = 1.0

    # Mix base color with redshift color
    mix_color = ng.nodes.new("ShaderNodeMix")
    mix_color.data_type = "RGBA"
    mix_color.blend_type = "MIX"

    # Strength modification by distance
    strength_mod = ng.nodes.new("ShaderNodeMath")
    strength_mod.operation = "MULTIPLY"

    # Position nodes
    input_node.location = (-400, 0)
    redshift_ramp.location = (-200, 100)
    distance_factor.location = (-200, -100)
    mix_color.location = (0, 0)
    strength_mod.location = (0, -100)
    emission.location = (200, 0)
    output_node.location = (400, 0)

    # Connect nodes
    ng.links.new(input_node.outputs["Redshift"], redshift_ramp.inputs["Fac"])
    ng.links.new(input_node.outputs["Distance Dimming"], distance_factor.inputs[1])

    ng.links.new(input_node.outputs["Base Color"], mix_color.inputs["Color1"])
    ng.links.new(redshift_ramp.outputs["Color"], mix_color.inputs["Color2"])
    ng.links.new(input_node.outputs["Redshift"], mix_color.inputs["Fac"])

    ng.links.new(input_node.outputs["Emission Strength"], strength_mod.inputs[0])
    ng.links.new(distance_factor.outputs["Value"], strength_mod.inputs[1])

    ng.links.new(mix_color.outputs["Result"], emission.inputs["Color"])
    ng.links.new(strength_mod.outputs["Value"], emission.inputs["Strength"])
    ng.links.new(emission.outputs["Emission"], output_node.inputs["Shader"])

    return ng


# Doppler effect presets
DOPPLER_PRESETS = {
    "approaching_star": {
        "Base Color": (1.0, 1.0, 1.0, 1.0),
        "Velocity": -50.0,  # Negative = approaching (km/s)
        "Rest Wavelength": 656.3,  # Hα line
        "Emission Strength": 8.0,
        "Doppler Intensity": 0.7,
    },
    "receding_galaxy": {
        "Base Color": (0.9, 0.9, 1.0, 1.0),
        "Velocity": 2000.0,  # Positive = receding (km/s)
        "Rest Wavelength": 656.3,
        "Emission Strength": 5.0,
        "Doppler Intensity": 0.8,
    },
    "binary_star_blue": {
        "Base Color": (0.8, 0.9, 1.0, 1.0),
        "Velocity": -100.0,  # Approaching component
        "Rest Wavelength": 486.1,  # Hβ line
        "Emission Strength": 12.0,
        "Doppler Intensity": 0.6,
    },
    "binary_star_red": {
        "Base Color": (1.0, 0.9, 0.8, 1.0),
        "Velocity": 100.0,  # Receding component
        "Rest Wavelength": 486.1,  # Hβ line
        "Emission Strength": 12.0,
        "Doppler Intensity": 0.6,
    },
    "quasar": {
        "Base Color": (0.8, 0.8, 1.0, 1.0),
        "Velocity": 50000.0,  # High redshift
        "Rest Wavelength": 1215.7,  # Lyman α
        "Emission Strength": 20.0,
        "Doppler Intensity": 0.9,
    },
}


def get_doppler_preset(preset_name: str) -> Dict[str, Any]:
    """Get Doppler effect preset configuration."""
    return DOPPLER_PRESETS.get(preset_name, DOPPLER_PRESETS["approaching_star"])


def apply_doppler_preset(material: bpy.types.Material, preset_name: str) -> None:
    """Apply Doppler effect preset to material."""
    preset = get_doppler_preset(preset_name)

    if not material.use_nodes:
        material.use_nodes = True

    # Find Doppler node group
    doppler_node = None
    for node in material.node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_Doppler" in node.node_tree.name:
                doppler_node = node
                break

    if doppler_node:
        # Apply preset parameters
        for param_name, value in preset.items():
            if param_name in doppler_node.inputs:
                doppler_node.inputs[param_name].default_value = value


def create_doppler_material(
    name: str, preset: str = "approaching_star"
) -> bpy.types.Material:
    """
    Create Doppler effect material with node group.

    Args:
        name: Material name
        preset: Doppler effect preset name

    Returns:
        bpy.types.Material: Created material
    """
    # Create material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    # Clear default nodes
    mat.node_tree.nodes.clear()

    # Add Doppler node group
    doppler_group = mat.node_tree.nodes.new("ShaderNodeGroup")
    doppler_group.node_tree = bpy.data.node_groups.get("ALBPY_DopplerEffect")

    # Add material output
    output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")

    # Position nodes
    doppler_group.location = (0, 0)
    output.location = (300, 0)

    # Connect nodes
    mat.node_tree.links.new(doppler_group.outputs["Shader"], output.inputs["Surface"])

    # Apply preset
    apply_doppler_preset(mat, preset)

    return mat


def register():
    """Register Doppler effect shader node groups."""
    if "ALBPY_DopplerEffect" not in bpy.data.node_groups:
        create_doppler_effect_node_group()

    if "ALBPY_RedshiftVisualization" not in bpy.data.node_groups:
        create_redshift_visualization_node_group()


def unregister():
    """Unregister Doppler effect shader node groups."""
    for group_name in ["ALBPY_DopplerEffect", "ALBPY_RedshiftVisualization"]:
        if group_name in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
