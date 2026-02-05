"""
Stellar Shader Node Groups
==========================

Modern Blender 4.4 implementation for stellar materials and blackbody radiation.
"""

from typing import Any, Dict

import bpy


def create_stellar_blackbody_node_group():
    """
    Erstellt eine Stellar Blackbody Shader Node Group mit moderner Blender 4.4 API.

    Returns:
        bpy.types.ShaderNodeTree: Die erstellte Shader Node Group
    """
    # Node-Group erstellen
    ng = bpy.data.node_groups.new("ALBPY_StellarBlackbody", "ShaderNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input-Sockets
    interface.new_socket(
        name="Temperature", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Luminosity", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Stellar Class", in_out="INPUT", socket_type="NodeSocketString"
    )
    interface.new_socket(
        name="Emission Strength", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Corona Intensity", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output-Sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Default-Werte für G-Klasse Stern (Sonne)
    ng.interface.items_tree[0].default_value = 5778.0  # Temperature (K)
    ng.interface.items_tree[1].default_value = 1.0  # Luminosity (Solar)
    ng.interface.items_tree[2].default_value = "G"  # Stellar Class
    ng.interface.items_tree[3].default_value = 10.0  # Emission Strength
    ng.interface.items_tree[4].default_value = 0.1  # Corona Intensity

    # Nodes erstellen
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Blackbody-Radiation für Sternfarbe
    blackbody = ng.nodes.new("ShaderNodeBlackbody")

    # Emission Shader für Stern
    emission = ng.nodes.new("ShaderNodeEmission")

    # ColorRamp für Temperatur-basierte Anpassung
    color_ramp = ng.nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (1.0, 0.3, 0.1, 1.0)  # Kühle Sterne (rot)
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (
        0.7,
        0.8,
        1.0,
        1.0,
    )  # Heiße Sterne (blau)

    # Math Node für Temperatur-Normalisierung
    temp_normalize = ng.nodes.new("ShaderNodeMath")
    temp_normalize.operation = "DIVIDE"
    temp_normalize.inputs[
        1
    ].default_value = 40000.0  # Max Temperatur für Normalisierung

    # Mix Node für Farb-Interpolation
    mix_color = ng.nodes.new("ShaderNodeMix")
    mix_color.data_type = "RGBA"
    mix_color.blend_type = "MIX"

    # Math Node für Luminositäts-Skalierung
    luminosity_scale = ng.nodes.new("ShaderNodeMath")
    luminosity_scale.operation = "MULTIPLY"

    # Corona-Effekt (zusätzliche Emission)
    corona_emission = ng.nodes.new("ShaderNodeEmission")

    # Add Shader für Corona + Hauptemission
    add_shader = ng.nodes.new("ShaderNodeAddShader")

    # Positionierung
    input_node.location = (-600, 0)
    blackbody.location = (-400, 100)
    temp_normalize.location = (-400, -100)
    color_ramp.location = (-200, -100)
    mix_color.location = (0, 0)
    luminosity_scale.location = (0, 200)
    emission.location = (200, 100)
    corona_emission.location = (200, -100)
    add_shader.location = (400, 0)
    output_node.location = (600, 0)

    # Verbindungen
    # Temperatur-basierte Farbe
    ng.links.new(input_node.outputs["Temperature"], blackbody.inputs["Temperature"])
    ng.links.new(input_node.outputs["Temperature"], temp_normalize.inputs[0])
    ng.links.new(temp_normalize.outputs["Value"], color_ramp.inputs["Fac"])

    # Farb-Mixing zwischen Blackbody und ColorRamp
    ng.links.new(blackbody.outputs["Color"], mix_color.inputs["Color1"])
    ng.links.new(color_ramp.outputs["Color"], mix_color.inputs["Color2"])
    ng.links.new(temp_normalize.outputs["Value"], mix_color.inputs["Fac"])

    # Luminositäts-Skalierung
    ng.links.new(input_node.outputs["Emission Strength"], luminosity_scale.inputs[0])
    ng.links.new(input_node.outputs["Luminosity"], luminosity_scale.inputs[1])

    # Hauptemission
    ng.links.new(mix_color.outputs["Result"], emission.inputs["Color"])
    ng.links.new(luminosity_scale.outputs["Value"], emission.inputs["Strength"])

    # Corona-Effekt
    ng.links.new(mix_color.outputs["Result"], corona_emission.inputs["Color"])
    ng.links.new(
        input_node.outputs["Corona Intensity"], corona_emission.inputs["Strength"]
    )

    # Shader-Kombination
    ng.links.new(emission.outputs["Emission"], add_shader.inputs[0])
    ng.links.new(corona_emission.outputs["Emission"], add_shader.inputs[1])
    ng.links.new(add_shader.outputs["Shader"], output_node.inputs["Shader"])

    return ng


def create_stellar_classification_node_group():
    """
    Erstellt eine vereinfachte Stellar Classification Node Group.

    Returns:
        bpy.types.ShaderNodeTree: Die erstellte Node Group
    """
    # Node-Group erstellen
    ng = bpy.data.node_groups.new("ALBPY_StellarClassification", "ShaderNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input-Sockets
    interface.new_socket(
        name="Spectral Class", in_out="INPUT", socket_type="NodeSocketString"
    )
    interface.new_socket(
        name="Custom Temperature", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Custom Strength", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output-Sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Default-Werte
    ng.interface.items_tree[0].default_value = "G"
    ng.interface.items_tree[1].default_value = 5778.0
    ng.interface.items_tree[2].default_value = 2.0

    # Einfache Implementierung mit Blackbody + Emission
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    blackbody = ng.nodes.new("ShaderNodeBlackbody")
    emission = ng.nodes.new("ShaderNodeEmission")

    # Positionierung
    input_node.location = (-300, 0)
    blackbody.location = (-100, 0)
    emission.location = (100, 0)
    output_node.location = (300, 0)

    # Verbindungen
    ng.links.new(
        input_node.outputs["Custom Temperature"], blackbody.inputs["Temperature"]
    )
    ng.links.new(blackbody.outputs["Color"], emission.inputs["Color"])
    ng.links.new(input_node.outputs["Custom Strength"], emission.inputs["Strength"])
    ng.links.new(emission.outputs["Emission"], output_node.inputs["Shader"])

    return ng


# Stellar classification presets
STELLAR_PRESETS = {
    "O": {
        "Temperature": 30000.0,
        "Luminosity": 100000.0,
        "Stellar Class": "O",
        "Emission Strength": 50.0,
        "Corona Intensity": 5.0,
    },
    "B": {
        "Temperature": 20000.0,
        "Luminosity": 1000.0,
        "Stellar Class": "B",
        "Emission Strength": 25.0,
        "Corona Intensity": 2.0,
    },
    "A": {
        "Temperature": 8500.0,
        "Luminosity": 25.0,
        "Stellar Class": "A",
        "Emission Strength": 15.0,
        "Corona Intensity": 1.0,
    },
    "F": {
        "Temperature": 6500.0,
        "Luminosity": 5.0,
        "Stellar Class": "F",
        "Emission Strength": 8.0,
        "Corona Intensity": 0.5,
    },
    "G": {
        "Temperature": 5778.0,
        "Luminosity": 1.0,
        "Stellar Class": "G",
        "Emission Strength": 5.0,
        "Corona Intensity": 0.3,
    },
    "K": {
        "Temperature": 4000.0,
        "Luminosity": 0.5,
        "Stellar Class": "K",
        "Emission Strength": 3.0,
        "Corona Intensity": 0.1,
    },
    "M": {
        "Temperature": 3000.0,
        "Luminosity": 0.01,
        "Stellar Class": "M",
        "Emission Strength": 1.0,
        "Corona Intensity": 0.05,
    },
}


def get_stellar_preset(spectral_class: str) -> Dict[str, Any]:
    """Get preset for a given spectral class."""
    return STELLAR_PRESETS.get(spectral_class.upper(), STELLAR_PRESETS["G"])


def apply_stellar_preset(material: bpy.types.Material, spectral_class: str) -> None:
    """Apply stellar preset to material using node groups."""
    preset = get_stellar_preset(spectral_class)

    if not material.use_nodes:
        material.use_nodes = True

    # Find the stellar node group
    stellar_node = None
    for node in material.node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_Stellar" in node.node_tree.name:
                stellar_node = node
                break

    if stellar_node:
        # Apply preset parameters
        for param_name, value in preset.items():
            if param_name in stellar_node.inputs:
                stellar_node.inputs[param_name].default_value = value


def create_stellar_material(name: str, spectral_class: str = "G") -> bpy.types.Material:
    """
    Create a complete stellar material with node group.

    Args:
        name: Material name
        spectral_class: Stellar spectral class (O, B, A, F, G, K, M)

    Returns:
        bpy.types.Material: Created material
    """
    # Create new material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    # Clear default nodes
    mat.node_tree.nodes.clear()

    # Add stellar blackbody node group
    stellar_group = mat.node_tree.nodes.new("ShaderNodeGroup")
    stellar_group.node_tree = bpy.data.node_groups.get("ALBPY_StellarBlackbody")

    # Add material output
    output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")

    # Position nodes
    stellar_group.location = (0, 0)
    output.location = (300, 0)

    # Connect nodes
    mat.node_tree.links.new(stellar_group.outputs["Shader"], output.inputs["Surface"])

    # Apply spectral class preset
    apply_stellar_preset(mat, spectral_class)

    return mat


def register():
    """Register stellar shader node groups using factory functions."""
    if "ALBPY_StellarBlackbody" not in bpy.data.node_groups:
        create_stellar_blackbody_node_group()

    if "ALBPY_StellarClassification" not in bpy.data.node_groups:
        create_stellar_classification_node_group()


def unregister():
    """Unregister stellar shader node groups."""
    for group_name in ["ALBPY_StellarBlackbody", "ALBPY_StellarClassification"]:
        if group_name in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
