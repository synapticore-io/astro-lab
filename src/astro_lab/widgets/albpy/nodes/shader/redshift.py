"""
Redshift Shader Node Groups
===========================

Modern Blender 4.4 implementation for cosmological redshift visualization.
"""

from typing import Any, Dict

import bpy


def create_redshift_node_group():
    """
    Create redshift shader node group using modern Blender 4.4 API.

    Returns:
        bpy.types.ShaderNodeTree: The created redshift shader node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_RedshiftShader", "ShaderNodeTree")

    # Interface API (Blender 4.4)
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
        name="Distance Factor", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Cosmological Dimming", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Set default values
    ng.interface.items_tree[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Base Color
    ng.interface.items_tree[1].default_value = 0.0  # Redshift (z)
    ng.interface.items_tree[2].default_value = 5.0  # Emission Strength
    ng.interface.items_tree[3].default_value = 1.0  # Distance Factor
    ng.interface.items_tree[4].default_value = 1.0  # Cosmological Dimming

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Emission shader
    emission = ng.nodes.new("ShaderNodeEmission")

    # Color ramp for redshift mapping
    redshift_ramp = ng.nodes.new("ShaderNodeValToRGB")
    redshift_ramp.color_ramp.elements[0].position = 0.0
    redshift_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)  # z=0 (white)
    redshift_ramp.color_ramp.elements[1].position = 1.0
    redshift_ramp.color_ramp.elements[1].color = (1.0, 0.3, 0.1, 1.0)  # High z (red)

    # Add intermediate color stops for realistic redshift colors
    redshift_ramp.color_ramp.elements.new(0.2)
    redshift_ramp.color_ramp.elements[1].position = 0.2
    redshift_ramp.color_ramp.elements[1].color = (
        1.0,
        0.9,
        0.7,
        1.0,
    )  # z~0.5 (yellow-white)

    redshift_ramp.color_ramp.elements.new(0.5)
    redshift_ramp.color_ramp.elements[2].position = 0.5
    redshift_ramp.color_ramp.elements[2].color = (1.0, 0.7, 0.4, 1.0)  # z~1.5 (orange)

    redshift_ramp.color_ramp.elements.new(0.8)
    redshift_ramp.color_ramp.elements[3].position = 0.8
    redshift_ramp.color_ramp.elements[3].color = (
        1.0,
        0.5,
        0.2,
        1.0,
    )  # z~4 (red-orange)

    # Math node for redshift normalization (z/10 for practical range)
    redshift_normalize = ng.nodes.new("ShaderNodeMath")
    redshift_normalize.operation = "DIVIDE"
    redshift_normalize.inputs[1].default_value = 10.0

    # Mix node for combining base color with redshift color
    mix_color = ng.nodes.new("ShaderNodeMix")
    mix_color.data_type = "RGBA"
    mix_color.blend_type = "MIX"

    # Distance dimming calculation
    distance_dimming = ng.nodes.new("ShaderNodeMath")
    distance_dimming.operation = "DIVIDE"
    distance_dimming.inputs[0].default_value = 1.0

    # Cosmological dimming (1+z)^4 effect
    z_plus_one = ng.nodes.new("ShaderNodeMath")
    z_plus_one.operation = "ADD"
    z_plus_one.inputs[1].default_value = 1.0

    cosmo_dimming = ng.nodes.new("ShaderNodeMath")
    cosmo_dimming.operation = "POWER"
    cosmo_dimming.inputs[1].default_value = 4.0

    cosmo_factor = ng.nodes.new("ShaderNodeMath")
    cosmo_factor.operation = "DIVIDE"
    cosmo_factor.inputs[0].default_value = 1.0

    # Final strength calculation
    final_strength = ng.nodes.new("ShaderNodeMath")
    final_strength.operation = "MULTIPLY"

    strength_dimming = ng.nodes.new("ShaderNodeMath")
    strength_dimming.operation = "MULTIPLY"

    # Position nodes
    input_node.location = (-800, 0)
    redshift_normalize.location = (-600, 200)
    redshift_ramp.location = (-400, 200)
    mix_color.location = (-200, 0)

    # Distance and cosmological dimming
    z_plus_one.location = (-600, -200)
    cosmo_dimming.location = (-400, -200)
    cosmo_factor.location = (-200, -200)
    distance_dimming.location = (-200, -300)

    # Strength calculations
    strength_dimming.location = (0, -200)
    final_strength.location = (200, -100)

    emission.location = (400, 0)
    output_node.location = (600, 0)

    # Connect nodes
    # Redshift color calculation
    ng.links.new(input_node.outputs["Redshift"], redshift_normalize.inputs[0])
    ng.links.new(redshift_normalize.outputs["Value"], redshift_ramp.inputs["Fac"])
    ng.links.new(input_node.outputs["Base Color"], mix_color.inputs["Color1"])
    ng.links.new(redshift_ramp.outputs["Color"], mix_color.inputs["Color2"])
    ng.links.new(redshift_normalize.outputs["Value"], mix_color.inputs["Fac"])

    # Cosmological dimming
    ng.links.new(input_node.outputs["Redshift"], z_plus_one.inputs[0])
    ng.links.new(z_plus_one.outputs["Value"], cosmo_dimming.inputs[0])
    ng.links.new(cosmo_dimming.outputs["Value"], cosmo_factor.inputs[1])
    ng.links.new(input_node.outputs["Cosmological Dimming"], cosmo_factor.inputs[0])

    # Distance dimming
    ng.links.new(input_node.outputs["Distance Factor"], distance_dimming.inputs[1])

    # Combine dimming effects
    ng.links.new(cosmo_factor.outputs["Value"], strength_dimming.inputs[0])
    ng.links.new(distance_dimming.outputs["Value"], strength_dimming.inputs[1])
    ng.links.new(input_node.outputs["Emission Strength"], final_strength.inputs[0])
    ng.links.new(strength_dimming.outputs["Value"], final_strength.inputs[1])

    # Final emission
    ng.links.new(mix_color.outputs["Result"], emission.inputs["Color"])
    ng.links.new(final_strength.outputs["Value"], emission.inputs["Strength"])
    ng.links.new(emission.outputs["Emission"], output_node.inputs["Shader"])

    return ng


def create_lyman_alpha_forest_node_group():
    """
    Create Lyman-alpha forest shader for high-redshift quasar visualization.

    Returns:
        bpy.types.ShaderNodeTree: The created Lyman-alpha forest shader node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_LymanAlphaForest", "ShaderNodeTree")

    # Interface API
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Base Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(name="Redshift", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Absorption Lines", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Line Strength", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Emission Strength", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Set defaults
    ng.interface.items_tree[0].default_value = (0.8, 0.9, 1.0, 1.0)  # Base Color
    ng.interface.items_tree[1].default_value = 2.0  # Redshift
    ng.interface.items_tree[2].default_value = 0.5  # Absorption Lines
    ng.interface.items_tree[3].default_value = 0.3  # Line Strength
    ng.interface.items_tree[4].default_value = 8.0  # Emission Strength

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Emission shader
    emission = ng.nodes.new("ShaderNodeEmission")

    # Noise for absorption line simulation
    absorption_noise = ng.nodes.new("ShaderNodeTexNoise")
    absorption_noise.inputs["Scale"].default_value = 20.0
    absorption_noise.inputs["Detail"].default_value = 5.0

    # Color ramp for absorption lines
    absorption_ramp = ng.nodes.new("ShaderNodeValToRGB")
    absorption_ramp.color_ramp.elements[0].position = 0.0
    absorption_ramp.color_ramp.elements[0].color = (
        0.1,
        0.1,
        0.1,
        1.0,
    )  # Dark absorption
    absorption_ramp.color_ramp.elements[1].position = 1.0
    absorption_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # No absorption

    # Mix absorption with base color
    mix_absorption = ng.nodes.new("ShaderNodeMix")
    mix_absorption.data_type = "RGBA"
    mix_absorption.blend_type = "MULTIPLY"

    # Redshift-dependent scaling
    redshift_scale = ng.nodes.new("ShaderNodeMath")
    redshift_scale.operation = "MULTIPLY"
    redshift_scale.inputs[1].default_value = 0.5

    # Position nodes
    input_node.location = (-600, 0)
    absorption_noise.location = (-400, 200)
    absorption_ramp.location = (-200, 200)
    redshift_scale.location = (-400, -100)
    mix_absorption.location = (0, 0)
    emission.location = (200, 0)
    output_node.location = (400, 0)

    # Connect nodes
    ng.links.new(input_node.outputs["Redshift"], redshift_scale.inputs[0])
    ng.links.new(redshift_scale.outputs["Value"], absorption_noise.inputs["Scale"])
    ng.links.new(absorption_noise.outputs["Fac"], absorption_ramp.inputs["Fac"])
    ng.links.new(input_node.outputs["Base Color"], mix_absorption.inputs["Color1"])
    ng.links.new(absorption_ramp.outputs["Color"], mix_absorption.inputs["Color2"])
    ng.links.new(input_node.outputs["Line Strength"], mix_absorption.inputs["Fac"])
    ng.links.new(mix_absorption.outputs["Result"], emission.inputs["Color"])
    ng.links.new(input_node.outputs["Emission Strength"], emission.inputs["Strength"])
    ng.links.new(emission.outputs["Emission"], output_node.inputs["Shader"])

    return ng


# Redshift presets for different cosmological epochs
REDSHIFT_PRESETS = {
    "local_universe": {
        "Base Color": (1.0, 1.0, 1.0, 1.0),
        "Redshift": 0.0,
        "Emission Strength": 5.0,
        "Distance Factor": 1.0,
        "Cosmological Dimming": 1.0,
    },
    "nearby_galaxy": {
        "Base Color": (0.9, 0.9, 1.0, 1.0),
        "Redshift": 0.01,
        "Emission Strength": 4.0,
        "Distance Factor": 1.5,
        "Cosmological Dimming": 1.0,
    },
    "distant_galaxy": {
        "Base Color": (1.0, 0.8, 0.6, 1.0),
        "Redshift": 1.0,
        "Emission Strength": 8.0,
        "Distance Factor": 10.0,
        "Cosmological Dimming": 0.8,
    },
    "high_redshift": {
        "Base Color": (1.0, 0.6, 0.3, 1.0),
        "Redshift": 3.0,
        "Emission Strength": 12.0,
        "Distance Factor": 50.0,
        "Cosmological Dimming": 0.5,
    },
    "early_universe": {
        "Base Color": (1.0, 0.4, 0.2, 1.0),
        "Redshift": 6.0,
        "Emission Strength": 20.0,
        "Distance Factor": 100.0,
        "Cosmological Dimming": 0.3,
    },
    "reionization_era": {
        "Base Color": (1.0, 0.3, 0.1, 1.0),
        "Redshift": 10.0,
        "Emission Strength": 30.0,
        "Distance Factor": 200.0,
        "Cosmological Dimming": 0.2,
    },
}


def get_redshift_preset(preset_name: str) -> Dict[str, Any]:
    """Get redshift preset configuration."""
    return REDSHIFT_PRESETS.get(preset_name, REDSHIFT_PRESETS["local_universe"])


def apply_redshift_preset(material: bpy.types.Material, preset_name: str) -> None:
    """Apply redshift preset to material."""
    preset = get_redshift_preset(preset_name)

    if not material.use_nodes:
        material.use_nodes = True

    # Find redshift node group
    redshift_node = None
    for node in material.node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_Redshift" in node.node_tree.name:
                redshift_node = node
                break

    if redshift_node:
        # Apply preset parameters
        for param_name, value in preset.items():
            if param_name in redshift_node.inputs:
                redshift_node.inputs[param_name].default_value = value


def create_redshift_material(
    name: str, preset: str = "local_universe"
) -> bpy.types.Material:
    """
    Create redshift material with node group.

    Args:
        name: Material name
        preset: Redshift preset name

    Returns:
        bpy.types.Material: Created material
    """
    # Create material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    # Clear default nodes
    mat.node_tree.nodes.clear()

    # Add redshift node group
    redshift_group = mat.node_tree.nodes.new("ShaderNodeGroup")
    redshift_group.node_tree = bpy.data.node_groups.get("ALBPY_RedshiftShader")

    # Add material output
    output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")

    # Position nodes
    redshift_group.location = (0, 0)
    output.location = (300, 0)

    # Connect nodes
    mat.node_tree.links.new(redshift_group.outputs["Shader"], output.inputs["Surface"])

    # Apply preset
    apply_redshift_preset(mat, preset)

    return mat


def register():
    """Register redshift shader node groups."""
    if "ALBPY_RedshiftShader" not in bpy.data.node_groups:
        create_redshift_node_group()

    if "ALBPY_LymanAlphaForest" not in bpy.data.node_groups:
        create_lyman_alpha_forest_node_group()


def unregister():
    """Unregister redshift shader node groups."""
    for group_name in ["ALBPY_RedshiftShader", "ALBPY_LymanAlphaForest"]:
        if group_name in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
