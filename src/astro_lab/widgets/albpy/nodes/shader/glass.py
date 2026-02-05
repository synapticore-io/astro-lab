"""
Glass Shader Node Groups
========================

Modern Blender 4.4 implementation for transparent glass materials for astronomical applications.
"""

from typing import Any, Dict

import bpy


def create_glass_node_group():
    """
    Create glass shader node group using modern Blender 4.4 API.

    Returns:
        bpy.types.ShaderNodeTree: The created glass shader node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_GlassShader", "ShaderNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Base Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(
        name="Transmission", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(name="IOR", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Roughness", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Absorption Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(
        name="Absorption Density", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Noise Scale", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Set default values
    ng.interface.items_tree[0].default_value = (0.9, 0.95, 1.0, 1.0)  # Base Color
    ng.interface.items_tree[1].default_value = 0.95  # Transmission
    ng.interface.items_tree[2].default_value = 1.45  # IOR
    ng.interface.items_tree[3].default_value = 0.0  # Roughness
    ng.interface.items_tree[4].default_value = (0.8, 0.9, 1.0, 1.0)  # Absorption Color
    ng.interface.items_tree[5].default_value = 0.1  # Absorption Density
    ng.interface.items_tree[6].default_value = 10.0  # Noise Scale

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Principled BSDF for glass
    principled = ng.nodes.new("ShaderNodeBsdfPrincipled")

    # Noise for surface variation
    noise = ng.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Detail"].default_value = 2.0
    noise.inputs["Roughness"].default_value = 0.5

    # Color ramp for noise variation
    color_ramp = ng.nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.4
    color_ramp.color_ramp.elements[1].position = 0.6

    # Mix color with noise
    mix_color = ng.nodes.new("ShaderNodeMix")
    mix_color.data_type = "RGBA"
    mix_color.blend_type = "MIX"

    # Math node for roughness variation
    roughness_variation = ng.nodes.new("ShaderNodeMath")
    roughness_variation.operation = "MULTIPLY"
    roughness_variation.inputs[1].default_value = 0.1

    # Add roughness variation
    roughness_add = ng.nodes.new("ShaderNodeMath")
    roughness_add.operation = "ADD"

    # Volume absorption
    volume_absorption = ng.nodes.new("ShaderNodeVolumeAbsorption")

    # Position nodes
    input_node.location = (-600, 0)
    noise.location = (-400, 200)
    color_ramp.location = (-200, 200)
    mix_color.location = (0, 100)
    roughness_variation.location = (-200, -100)
    roughness_add.location = (0, -100)
    volume_absorption.location = (0, -300)
    principled.location = (200, 0)
    output_node.location = (400, 0)

    # Connect nodes
    # Surface color with noise variation
    ng.links.new(input_node.outputs["Noise Scale"], noise.inputs["Scale"])
    ng.links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    ng.links.new(input_node.outputs["Base Color"], mix_color.inputs["Color1"])
    ng.links.new(color_ramp.outputs["Color"], mix_color.inputs["Color2"])
    ng.links.new(color_ramp.outputs["Alpha"], mix_color.inputs["Fac"])
    ng.links.new(mix_color.outputs["Result"], principled.inputs["Base Color"])

    # Glass properties
    ng.links.new(input_node.outputs["Transmission"], principled.inputs["Transmission"])
    ng.links.new(input_node.outputs["IOR"], principled.inputs["IOR"])

    # Roughness with variation
    ng.links.new(noise.outputs["Fac"], roughness_variation.inputs[0])
    ng.links.new(input_node.outputs["Roughness"], roughness_add.inputs[0])
    ng.links.new(roughness_variation.outputs["Value"], roughness_add.inputs[1])
    ng.links.new(roughness_add.outputs["Value"], principled.inputs["Roughness"])

    # Volume absorption
    ng.links.new(
        input_node.outputs["Absorption Color"], volume_absorption.inputs["Color"]
    )
    ng.links.new(
        input_node.outputs["Absorption Density"], volume_absorption.inputs["Density"]
    )

    # Output
    ng.links.new(principled.outputs["BSDF"], output_node.inputs["Shader"])

    return ng


def create_crystal_node_group():
    """
    Create crystal shader node group for crystalline materials.

    Returns:
        bpy.types.ShaderNodeTree: The created crystal shader node group
    """
    # Create node group
    ng = bpy.data.node_groups.new("ALBPY_CrystalShader", "ShaderNodeTree")

    # Interface API
    interface = ng.interface

    # Input sockets
    interface.new_socket(
        name="Base Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(
        name="Crystal Faces", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Refraction", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Dispersion", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Internal Glow", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Set defaults
    ng.interface.items_tree[0].default_value = (0.9, 0.95, 1.0, 1.0)  # Base Color
    ng.interface.items_tree[1].default_value = 8.0  # Crystal Faces
    ng.interface.items_tree[2].default_value = 1.8  # Refraction
    ng.interface.items_tree[3].default_value = 0.05  # Dispersion
    ng.interface.items_tree[4].default_value = 0.1  # Internal Glow

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Glass BSDF
    glass_bsdf = ng.nodes.new("ShaderNodeBsdfGlass")

    # Emission for internal glow
    emission = ng.nodes.new("ShaderNodeEmission")

    # Add shader
    add_shader = ng.nodes.new("ShaderNodeAddShader")

    # Voronoi for crystal faces
    voronoi = ng.nodes.new("ShaderNodeTexVoronoi")
    voronoi.feature = "DISTANCE_TO_EDGE"

    # Color ramp for faceting
    facet_ramp = ng.nodes.new("ShaderNodeValToRGB")
    facet_ramp.color_ramp.elements[0].position = 0.2
    facet_ramp.color_ramp.elements[1].position = 0.8

    # Fresnel for edge highlights
    fresnel = ng.nodes.new("ShaderNodeFresnel")

    # Position nodes
    input_node.location = (-600, 0)
    voronoi.location = (-400, 200)
    facet_ramp.location = (-200, 200)
    fresnel.location = (-400, -200)
    glass_bsdf.location = (-200, 0)
    emission.location = (-200, -100)
    add_shader.location = (0, 0)
    output_node.location = (200, 0)

    # Connect nodes
    ng.links.new(input_node.outputs["Crystal Faces"], voronoi.inputs["Scale"])
    ng.links.new(voronoi.outputs["Distance"], facet_ramp.inputs["Fac"])
    ng.links.new(facet_ramp.outputs["Color"], glass_bsdf.inputs["Color"])
    ng.links.new(input_node.outputs["Refraction"], glass_bsdf.inputs["IOR"])
    ng.links.new(input_node.outputs["Refraction"], fresnel.inputs["IOR"])

    ng.links.new(input_node.outputs["Base Color"], emission.inputs["Color"])
    ng.links.new(input_node.outputs["Internal Glow"], emission.inputs["Strength"])

    ng.links.new(glass_bsdf.outputs["BSDF"], add_shader.inputs[0])
    ng.links.new(emission.outputs["Emission"], add_shader.inputs[1])
    ng.links.new(add_shader.outputs["Shader"], output_node.inputs["Shader"])

    return ng


# Glass presets for different materials
GLASS_PRESETS = {
    "optical_glass": {
        "Base Color": (0.98, 0.98, 1.0, 1.0),
        "Transmission": 0.98,
        "IOR": 1.52,
        "Roughness": 0.0,
        "Absorption Color": (0.9, 0.95, 1.0, 1.0),
        "Absorption Density": 0.05,
        "Noise Scale": 50.0,
    },
    "crystalline_lens": {
        "Base Color": (0.95, 0.98, 1.0, 1.0),
        "Transmission": 0.95,
        "IOR": 1.76,
        "Roughness": 0.02,
        "Absorption Color": (0.8, 0.9, 1.0, 1.0),
        "Absorption Density": 0.1,
        "Noise Scale": 20.0,
    },
    "space_glass": {
        "Base Color": (0.9, 0.95, 1.0, 1.0),
        "Transmission": 0.92,
        "IOR": 1.45,
        "Roughness": 0.05,
        "Absorption Color": (0.7, 0.8, 0.9, 1.0),
        "Absorption Density": 0.15,
        "Noise Scale": 15.0,
    },
    "telescope_mirror": {
        "Base Color": (0.95, 0.95, 0.98, 1.0),
        "Transmission": 0.0,
        "IOR": 1.52,
        "Roughness": 0.001,
        "Absorption Color": (0.9, 0.9, 0.95, 1.0),
        "Absorption Density": 0.0,
        "Noise Scale": 100.0,
    },
    "detector_cover": {
        "Base Color": (0.85, 0.9, 0.95, 1.0),
        "Transmission": 0.85,
        "IOR": 1.6,
        "Roughness": 0.1,
        "Absorption Color": (0.6, 0.7, 0.8, 1.0),
        "Absorption Density": 0.2,
        "Noise Scale": 5.0,
    },
}


def get_glass_preset(preset_name: str) -> Dict[str, Any]:
    """Get glass preset configuration."""
    return GLASS_PRESETS.get(preset_name, GLASS_PRESETS["optical_glass"])


def apply_glass_preset(material: bpy.types.Material, preset_name: str) -> None:
    """Apply glass preset to material."""
    preset = get_glass_preset(preset_name)

    if not material.use_nodes:
        material.use_nodes = True

    # Find glass node group
    glass_node = None
    for node in material.node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_Glass" in node.node_tree.name:
                glass_node = node
                break

    if glass_node:
        # Apply preset parameters
        for param_name, value in preset.items():
            if param_name in glass_node.inputs:
                glass_node.inputs[param_name].default_value = value


def create_glass_material(
    name: str, preset: str = "optical_glass"
) -> bpy.types.Material:
    """
    Create glass material with node group.

    Args:
        name: Material name
        preset: Glass preset name

    Returns:
        bpy.types.Material: Created material
    """
    # Create material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    # Clear default nodes
    mat.node_tree.nodes.clear()

    # Add glass node group
    glass_group = mat.node_tree.nodes.new("ShaderNodeGroup")
    glass_group.node_tree = bpy.data.node_groups.get("ALBPY_GlassShader")

    # Add material output
    output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")

    # Position nodes
    glass_group.location = (0, 0)
    output.location = (300, 0)

    # Connect nodes
    mat.node_tree.links.new(glass_group.outputs["Shader"], output.inputs["Surface"])

    # Apply preset
    apply_glass_preset(mat, preset)

    # Set material properties for glass
    mat.use_screen_refraction = True
    mat.refraction_depth = 0.1

    return mat


def register():
    """Register glass shader node groups."""
    if "ALBPY_GlassShader" not in bpy.data.node_groups:
        create_glass_node_group()

    if "ALBPY_CrystalShader" not in bpy.data.node_groups:
        create_crystal_node_group()


def unregister():
    """Unregister glass shader node groups."""
    for group_name in ["ALBPY_GlassShader", "ALBPY_CrystalShader"]:
        if group_name in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
