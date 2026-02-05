"""
Absorption Shader Node Groups
=============================

Modern Blender 4.4 implementation for absorption shaders with astronomical applications.
"""

from typing import Any, Dict

import bpy


def create_absorption_node_group():
    """
    Create absorption shader node group using modern Blender 4.4 API.

    Returns:
        bpy.types.ShaderNodeTree: The created absorption shader node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_AbsorptionShader", "ShaderNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Base Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(
        name="Absorption Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(
        name="Absorption Strength", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Roughness", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(name="Metallic", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Transmission", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Set default values
    ng.interface.items_tree[0].default_value = (0.8, 0.8, 0.8, 1.0)  # Base Color
    ng.interface.items_tree[1].default_value = (0.1, 0.1, 0.3, 1.0)  # Absorption Color
    ng.interface.items_tree[2].default_value = 0.5  # Absorption Strength
    ng.interface.items_tree[3].default_value = 0.5  # Roughness
    ng.interface.items_tree[4].default_value = 0.0  # Metallic
    ng.interface.items_tree[5].default_value = 0.5  # Transmission

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Principled BSDF for base material
    principled = ng.nodes.new("ShaderNodeBsdfPrincipled")

    # Transparent BSDF for transmission
    transparent = ng.nodes.new("ShaderNodeBsdfTransparent")

    # Volume absorption
    volume_absorption = ng.nodes.new("ShaderNodeVolumeAbsorption")

    # Mix shader for combining surface and transmission
    mix_shader = ng.nodes.new("ShaderNodeMixShader")

    # Math node for absorption mixing
    absorption_mix = ng.nodes.new("ShaderNodeMath")
    absorption_mix.operation = "MULTIPLY"

    # Color mix for absorption effect
    color_mix = ng.nodes.new("ShaderNodeMix")
    color_mix.data_type = "RGBA"
    color_mix.blend_type = "MULTIPLY"

    # Position nodes
    input_node.location = (-600, 0)
    absorption_mix.location = (-400, 200)
    color_mix.location = (-400, 0)
    principled.location = (-200, 100)
    transparent.location = (-200, -100)
    volume_absorption.location = (-200, -300)
    mix_shader.location = (0, 0)
    output_node.location = (200, 0)

    # Connect nodes
    # Base material setup
    ng.links.new(input_node.outputs["Base Color"], color_mix.inputs["Color1"])
    ng.links.new(input_node.outputs["Absorption Color"], color_mix.inputs["Color2"])
    ng.links.new(input_node.outputs["Absorption Strength"], color_mix.inputs["Fac"])
    ng.links.new(color_mix.outputs["Result"], principled.inputs["Base Color"])
    ng.links.new(input_node.outputs["Roughness"], principled.inputs["Roughness"])
    ng.links.new(input_node.outputs["Metallic"], principled.inputs["Metallic"])

    # Transmission setup
    ng.links.new(input_node.outputs["Absorption Color"], transparent.inputs["Color"])

    # Volume absorption
    ng.links.new(
        input_node.outputs["Absorption Color"], volume_absorption.inputs["Color"]
    )
    ng.links.new(
        input_node.outputs["Absorption Strength"], volume_absorption.inputs["Density"]
    )

    # Mix surface and transmission
    ng.links.new(principled.outputs["BSDF"], mix_shader.inputs[1])
    ng.links.new(transparent.outputs["BSDF"], mix_shader.inputs[2])
    ng.links.new(input_node.outputs["Transmission"], mix_shader.inputs["Fac"])

    # Output
    ng.links.new(mix_shader.outputs["Shader"], output_node.inputs["Shader"])

    return ng


def create_atmospheric_absorption_node_group():
    """
    Create atmospheric absorption shader for planetary atmospheres.

    Returns:
        bpy.types.ShaderNodeTree: The created atmospheric absorption node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_AtmosphericAbsorption", "ShaderNodeTree")

    # Interface API
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Atmosphere Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(name="Density", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Scattering", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Absorption", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Anisotropy", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(name="Volume", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Set defaults
    ng.interface.items_tree[0].default_value = (0.4, 0.7, 1.0, 1.0)  # Atmosphere Color
    ng.interface.items_tree[1].default_value = 0.1  # Density
    ng.interface.items_tree[2].default_value = 0.8  # Scattering
    ng.interface.items_tree[3].default_value = 0.1  # Absorption
    ng.interface.items_tree[4].default_value = 0.0  # Anisotropy

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Volume scatter
    volume_scatter = ng.nodes.new("ShaderNodeVolumeScatter")

    # Volume absorption
    volume_absorption = ng.nodes.new("ShaderNodeVolumeAbsorption")

    # Add shader for combining volume effects
    add_shader = ng.nodes.new("ShaderNodeAddShader")

    # Math nodes for density scaling
    scatter_density = ng.nodes.new("ShaderNodeMath")
    scatter_density.operation = "MULTIPLY"

    absorption_density = ng.nodes.new("ShaderNodeMath")
    absorption_density.operation = "MULTIPLY"

    # Position nodes
    input_node.location = (-400, 0)
    scatter_density.location = (-200, 100)
    absorption_density.location = (-200, -100)
    volume_scatter.location = (0, 100)
    volume_absorption.location = (0, -100)
    add_shader.location = (200, 0)
    output_node.location = (400, 0)

    # Connect nodes
    ng.links.new(input_node.outputs["Density"], scatter_density.inputs[0])
    ng.links.new(input_node.outputs["Scattering"], scatter_density.inputs[1])
    ng.links.new(input_node.outputs["Density"], absorption_density.inputs[0])
    ng.links.new(input_node.outputs["Absorption"], absorption_density.inputs[1])

    ng.links.new(input_node.outputs["Atmosphere Color"], volume_scatter.inputs["Color"])
    ng.links.new(scatter_density.outputs["Value"], volume_scatter.inputs["Density"])
    ng.links.new(input_node.outputs["Anisotropy"], volume_scatter.inputs["Anisotropy"])

    ng.links.new(
        input_node.outputs["Atmosphere Color"], volume_absorption.inputs["Color"]
    )
    ng.links.new(
        absorption_density.outputs["Value"], volume_absorption.inputs["Density"]
    )

    ng.links.new(volume_scatter.outputs["Volume"], add_shader.inputs[0])
    ng.links.new(volume_absorption.outputs["Volume"], add_shader.inputs[1])
    ng.links.new(add_shader.outputs["Shader"], output_node.inputs["Volume"])

    return ng


# Absorption presets for different astronomical objects
ABSORPTION_PRESETS = {
    "dust_lane": {
        "Base Color": (0.3, 0.2, 0.1, 1.0),
        "Absorption Color": (0.1, 0.05, 0.02, 1.0),
        "Absorption Strength": 0.8,
        "Roughness": 0.9,
        "Metallic": 0.0,
        "Transmission": 0.1,
    },
    "nebula_dark": {
        "Base Color": (0.2, 0.1, 0.3, 1.0),
        "Absorption Color": (0.05, 0.02, 0.1, 1.0),
        "Absorption Strength": 0.6,
        "Roughness": 0.8,
        "Metallic": 0.0,
        "Transmission": 0.3,
    },
    "planetary_atmosphere": {
        "Base Color": (0.4, 0.7, 1.0, 1.0),
        "Absorption Color": (0.1, 0.2, 0.4, 1.0),
        "Absorption Strength": 0.3,
        "Roughness": 0.0,
        "Metallic": 0.0,
        "Transmission": 0.8,
    },
    "stellar_atmosphere": {
        "Base Color": (1.0, 0.8, 0.6, 1.0),
        "Absorption Color": (0.3, 0.1, 0.05, 1.0),
        "Absorption Strength": 0.4,
        "Roughness": 0.2,
        "Metallic": 0.0,
        "Transmission": 0.6,
    },
}


def get_absorption_preset(preset_name: str) -> Dict[str, Any]:
    """Get absorption preset configuration."""
    return ABSORPTION_PRESETS.get(preset_name, ABSORPTION_PRESETS["dust_lane"])


def apply_absorption_preset(material: bpy.types.Material, preset_name: str) -> None:
    """Apply absorption preset to material."""
    preset = get_absorption_preset(preset_name)

    if not material.use_nodes:
        material.use_nodes = True

    # Find absorption node group
    absorption_node = None
    for node in material.node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_Absorption" in node.node_tree.name:
                absorption_node = node
                break

    if absorption_node:
        # Apply preset parameters
        for param_name, value in preset.items():
            if param_name in absorption_node.inputs:
                absorption_node.inputs[param_name].default_value = value


def create_absorption_material(
    name: str, preset: str = "dust_lane"
) -> bpy.types.Material:
    """
    Create absorption material with node group.

    Args:
        name: Material name
        preset: Absorption preset name

    Returns:
        bpy.types.Material: Created material
    """
    # Create material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    # Clear default nodes
    mat.node_tree.nodes.clear()

    # Add absorption node group
    absorption_group = mat.node_tree.nodes.new("ShaderNodeGroup")
    absorption_group.node_tree = bpy.data.node_groups.get("ALBPY_AbsorptionShader")

    # Add material output
    output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")

    # Position nodes
    absorption_group.location = (0, 0)
    output.location = (300, 0)

    # Connect nodes
    mat.node_tree.links.new(
        absorption_group.outputs["Shader"], output.inputs["Surface"]
    )

    # Apply preset
    apply_absorption_preset(mat, preset)

    return mat


def register():
    """Register absorption shader node groups."""
    if "ALBPY_AbsorptionShader" not in bpy.data.node_groups:
        create_absorption_node_group()

    if "ALBPY_AtmosphericAbsorption" not in bpy.data.node_groups:
        create_atmospheric_absorption_node_group()


def unregister():
    """Unregister absorption shader node groups."""
    for group_name in ["ALBPY_AbsorptionShader", "ALBPY_AtmosphericAbsorption"]:
        if group_name in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
