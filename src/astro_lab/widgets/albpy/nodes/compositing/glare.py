"""
Glare Compositing Node Groups
=============================

Modern Blender 4.4 implementation for stellar glare and diffraction effects.
"""

from typing import Any, Dict

import bpy


def create_stellar_glare_node_group():
    """
    Erstellt eine Stellar Glare Compositing Node Group mit moderner Blender 4.4 API.

    Returns:
        bpy.types.CompositorNodeTree: Die erstellte Compositing Node Group
    """
    # Node-Group erstellen
    ng = bpy.data.node_groups.new("ALBPY_StellarGlare", "CompositorNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input-Sockets
    interface.new_socket(name="Image", in_out="INPUT", socket_type="NodeSocketColor")
    interface.new_socket(
        name="Glare Threshold", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Glare Mix", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(name="Streaks", in_out="INPUT", socket_type="NodeSocketInt")
    interface.new_socket(
        name="Streak Length", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(name="Fade", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Angle Offset", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output-Sockets
    interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    # Default-Werte für realistische Sterne
    ng.interface.items_tree[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Image
    ng.interface.items_tree[1].default_value = 1.0  # Glare Threshold
    ng.interface.items_tree[2].default_value = 0.8  # Glare Mix
    ng.interface.items_tree[3].default_value = 4  # Streaks (4 für typische Diffraktion)
    ng.interface.items_tree[4].default_value = 2.0  # Streak Length
    ng.interface.items_tree[5].default_value = 0.9  # Fade
    ng.interface.items_tree[6].default_value = 0.0  # Angle Offset

    # Nodes erstellen
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Glare Node (Blender built-in)
    glare = ng.nodes.new("CompositorNodeGlare")
    glare.glare_type = "STREAKS"
    glare.quality = "HIGH"

    # Mix Node für Glare Blending
    mix_glare = ng.nodes.new("CompositorNodeMixRGB")
    mix_glare.blend_type = "ADD"

    # Positionierung
    input_node.location = (-400, 0)
    glare.location = (-200, 0)
    mix_glare.location = (0, 0)
    output_node.location = (200, 0)

    # Verbindungen
    ng.links.new(input_node.outputs["Image"], glare.inputs["Image"])
    ng.links.new(input_node.outputs["Glare Threshold"], glare.inputs["Threshold"])
    ng.links.new(input_node.outputs["Streaks"], glare.inputs["Streaks"])
    ng.links.new(input_node.outputs["Streak Length"], glare.inputs["Streak Length"])
    ng.links.new(input_node.outputs["Fade"], glare.inputs["Fade"])
    ng.links.new(input_node.outputs["Angle Offset"], glare.inputs["Angle Offset"])

    # Mix original mit Glare
    ng.links.new(input_node.outputs["Image"], mix_glare.inputs["Image1"])
    ng.links.new(glare.outputs["Image"], mix_glare.inputs["Image2"])
    ng.links.new(input_node.outputs["Glare Mix"], mix_glare.inputs["Fac"])

    ng.links.new(mix_glare.outputs["Image"], output_node.inputs["Image"])

    return ng


def create_airy_disk_node_group():
    """
    Erstellt eine Airy Disk Compositing Node Group für realistische Beugungsscheibchen.

    Returns:
        bpy.types.CompositorNodeTree: Die erstellte Node Group
    """
    # Node-Group erstellen
    ng = bpy.data.node_groups.new("ALBPY_AiryDisk", "CompositorNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input-Sockets
    interface.new_socket(name="Image", in_out="INPUT", socket_type="NodeSocketColor")
    interface.new_socket(
        name="Star Threshold", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Telescope Diameter", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Wavelength", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Airy Intensity", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output-Sockets
    interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    # Default-Werte
    ng.interface.items_tree[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Image
    ng.interface.items_tree[1].default_value = 0.8  # Star Threshold
    ng.interface.items_tree[2].default_value = 200.0  # Telescope Diameter (mm)
    ng.interface.items_tree[3].default_value = 550.0  # Wavelength (nm)
    ng.interface.items_tree[4].default_value = 0.3  # Airy Intensity

    # Nodes erstellen
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Blur für Airy Disk Simulation (vereinfacht)
    blur = ng.nodes.new("CompositorNodeBlur")
    blur.filter_type = "GAUSS"

    # Math Node für Blur-Size Berechnung basierend auf Telescope
    telescope_calc = ng.nodes.new("CompositorNodeMath")
    telescope_calc.operation = "DIVIDE"
    telescope_calc.inputs[0].default_value = 1000.0  # Basis-Wert

    # Bright/Contrast für Star Enhancement
    bright_contrast = ng.nodes.new("CompositorNodeBrightContrast")

    # Mix für Airy Effect
    mix_airy = ng.nodes.new("CompositorNodeMixRGB")
    mix_airy.blend_type = "ADD"

    # Positionierung
    input_node.location = (-600, 0)
    telescope_calc.location = (-400, -200)
    bright_contrast.location = (-400, 0)
    blur.location = (-200, 0)
    mix_airy.location = (0, 0)
    output_node.location = (200, 0)

    # Verbindungen
    ng.links.new(input_node.outputs["Image"], bright_contrast.inputs["Image"])
    ng.links.new(input_node.outputs["Star Threshold"], bright_contrast.inputs["Bright"])

    ng.links.new(input_node.outputs["Telescope Diameter"], telescope_calc.inputs[1])
    ng.links.new(telescope_calc.outputs["Value"], blur.inputs["Size"])

    ng.links.new(bright_contrast.outputs["Image"], blur.inputs["Image"])

    ng.links.new(input_node.outputs["Image"], mix_airy.inputs["Image1"])
    ng.links.new(blur.outputs["Image"], mix_airy.inputs["Image2"])
    ng.links.new(input_node.outputs["Airy Intensity"], mix_airy.inputs["Fac"])

    ng.links.new(mix_airy.outputs["Image"], output_node.inputs["Image"])

    return ng


def create_atmospheric_turbulence_node_group():
    """
    Erstellt eine Atmospheric Turbulence Node Group für Seeing-Effekte.

    Returns:
        bpy.types.CompositorNodeTree: Die erstellte Node Group
    """
    # Node-Group erstellen
    ng = bpy.data.node_groups.new("ALBPY_AtmosphericTurbulence", "CompositorNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input-Sockets
    interface.new_socket(name="Image", in_out="INPUT", socket_type="NodeSocketColor")
    interface.new_socket(name="Seeing", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Turbulence Strength", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Scintillation", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output-Sockets
    interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    # Default-Werte für typische Seeing-Bedingungen
    ng.interface.items_tree[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Image
    ng.interface.items_tree[1].default_value = 1.5  # Seeing (arcsec)
    ng.interface.items_tree[2].default_value = 0.2  # Turbulence Strength
    ng.interface.items_tree[3].default_value = 0.1  # Scintillation

    # Nodes erstellen
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Blur für Seeing
    seeing_blur = ng.nodes.new("CompositorNodeBlur")
    seeing_blur.filter_type = "GAUSS"

    # Distort für Turbulence
    distort = ng.nodes.new("CompositorNodeDisplace")

    # Noise für Displacement
    noise = ng.nodes.new("CompositorNodeTexture")
    noise.texture = bpy.data.textures.new("TurbulenceNoise", "CLOUDS")

    # Scale für Noise
    scale = ng.nodes.new("CompositorNodeScale")
    scale.space = "RELATIVE"

    # Mix für finalen Effekt
    mix_final = ng.nodes.new("CompositorNodeMixRGB")
    mix_final.blend_type = "MIX"

    # Positionierung
    input_node.location = (-600, 0)
    noise.location = (-400, -200)
    scale.location = (-200, -200)
    seeing_blur.location = (-400, 0)
    distort.location = (-200, 0)
    mix_final.location = (0, 0)
    output_node.location = (200, 0)

    # Verbindungen
    ng.links.new(input_node.outputs["Image"], seeing_blur.inputs["Image"])
    ng.links.new(input_node.outputs["Seeing"], seeing_blur.inputs["Size"])

    # Turbulence displacement
    ng.links.new(noise.outputs["Color"], scale.inputs["Image"])
    ng.links.new(input_node.outputs["Turbulence Strength"], scale.inputs["X"])
    ng.links.new(input_node.outputs["Turbulence Strength"], scale.inputs["Y"])

    ng.links.new(seeing_blur.outputs["Image"], distort.inputs["Image"])
    ng.links.new(scale.outputs["Image"], distort.inputs["Vector"])

    ng.links.new(seeing_blur.outputs["Image"], mix_final.inputs["Image1"])
    ng.links.new(distort.outputs["Image"], mix_final.inputs["Image2"])
    ng.links.new(input_node.outputs["Scintillation"], mix_final.inputs["Fac"])

    ng.links.new(mix_final.outputs["Image"], output_node.inputs["Image"])

    return ng


# Glare Presets für verschiedene Teleskop-Typen
GLARE_PRESETS = {
    "refractor": {
        "Glare Threshold": 0.8,
        "Glare Mix": 0.6,
        "Streaks": 4,
        "Streak Length": 1.5,
        "Fade": 0.95,
        "Angle Offset": 0.0,
    },
    "reflector": {
        "Glare Threshold": 0.9,
        "Glare Mix": 0.8,
        "Streaks": 4,
        "Streak Length": 2.0,
        "Fade": 0.90,
        "Angle Offset": 45.0,
    },
    "schmidt_cassegrain": {
        "Glare Threshold": 0.85,
        "Glare Mix": 0.7,
        "Streaks": 4,
        "Streak Length": 1.8,
        "Fade": 0.92,
        "Angle Offset": 0.0,
    },
    "hubble": {
        "Glare Threshold": 0.95,
        "Glare Mix": 0.9,
        "Streaks": 4,
        "Streak Length": 3.0,
        "Fade": 0.85,
        "Angle Offset": 45.0,
    },
    "webb": {
        "Glare Threshold": 0.98,
        "Glare Mix": 0.95,
        "Streaks": 6,
        "Streak Length": 4.0,
        "Fade": 0.80,
        "Angle Offset": 0.0,
    },
}


def get_glare_preset(telescope_type: str) -> Dict[str, Any]:
    """Get glare preset for telescope type."""
    return GLARE_PRESETS.get(telescope_type, GLARE_PRESETS["refractor"])


def apply_glare_preset(
    compositor: bpy.types.CompositorNodeTree, telescope_type: str
) -> None:
    """Apply glare preset to compositor."""
    preset = get_glare_preset(telescope_type)

    # Find stellar glare node group
    for node in compositor.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_StellarGlare" in node.node_tree.name:
                # Apply preset parameters
                for param_name, value in preset.items():
                    if param_name in node.inputs:
                        node.inputs[param_name].default_value = value
                break


def register():
    """Register glare compositing node groups using factory functions."""
    node_groups_to_create = [
        create_stellar_glare_node_group,
        create_airy_disk_node_group,
        create_atmospheric_turbulence_node_group,
    ]

    for create_func in node_groups_to_create:
        try:
            create_func()
        except Exception as e:
            print(f"Failed to create glare node group: {e}")


def unregister():
    """Unregister glare compositing node groups."""
    glare_node_groups = [
        "ALBPY_StellarGlare",
        "ALBPY_AiryDisk",
        "ALBPY_AtmosphericTurbulence",
    ]

    for group_name in glare_node_groups:
        if group_name in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
