"""
Galaxy Shader Node Groups
=========================

Modern Blender 4.4 implementation for galaxy materials and morphology visualization.
"""

from typing import Any, Dict

import bpy


def create_galaxy_disk_node_group():
    """
    Erstellt eine Galaxy Disk Shader Node Group mit moderner Blender 4.4 API.

    Returns:
        bpy.types.ShaderNodeTree: Die erstellte Shader Node Group
    """
    # Node-Group erstellen
    ng = bpy.data.node_groups.new("ALBPY_GalaxyDisk", "ShaderNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input-Sockets
    interface.new_socket(
        name="Color Index", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Star Formation Rate", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Metallicity", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(name="Age", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Dust Extinction", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Spiral Arm Contrast", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output-Sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Default-Werte für Sb-Typ Galaxie
    ng.interface.items_tree[0].default_value = 0.6  # Color Index (g-r)
    ng.interface.items_tree[1].default_value = 3.0  # Star Formation Rate (Msun/yr)
    ng.interface.items_tree[2].default_value = 1.0  # Metallicity (solar)
    ng.interface.items_tree[3].default_value = 8.0  # Age (Gyr)
    ng.interface.items_tree[4].default_value = 0.3  # Dust Extinction
    ng.interface.items_tree[5].default_value = 0.7  # Spiral Arm Contrast

    # Nodes erstellen
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Color ramp für Farb-Temperatur Mapping
    color_ramp_temp = ng.nodes.new("ShaderNodeValToRGB")
    color_ramp_temp.color_ramp.elements[0].position = 0.0
    color_ramp_temp.color_ramp.elements[0].color = (0.3, 0.4, 1.0, 1.0)  # Blau (jung)
    color_ramp_temp.color_ramp.elements[1].position = 1.0
    color_ramp_temp.color_ramp.elements[1].color = (1.0, 0.7, 0.4, 1.0)  # Rot (alt)

    # Emission für Sterne
    emission_young = ng.nodes.new("ShaderNodeEmission")
    emission_old = ng.nodes.new("ShaderNodeEmission")

    # Mix Node für Stern-Populationen
    mix_populations = ng.nodes.new("ShaderNodeMix")
    mix_populations.data_type = "RGBA"
    mix_populations.blend_type = "MIX"

    # Math Nodes für SFR-basierte Mischung
    sfr_normalize = ng.nodes.new("ShaderNodeMath")
    sfr_normalize.operation = "DIVIDE"
    sfr_normalize.inputs[1].default_value = 10.0  # Normalisierung

    # Dust extinction
    dust_mix = ng.nodes.new("ShaderNodeMix")
    dust_mix.data_type = "RGBA"
    dust_mix.blend_type = "MULTIPLY"

    # Dust color (reddening)
    dust_color = ng.nodes.new("ShaderNodeRGB")
    dust_color.outputs[0].default_value = (0.8, 0.6, 0.4, 1.0)

    # Final emission
    final_emission = ng.nodes.new("ShaderNodeEmission")

    # Positionierung
    input_node.location = (-800, 0)
    color_ramp_temp.location = (-600, 200)
    emission_young.location = (-400, 300)
    emission_old.location = (-400, 100)
    sfr_normalize.location = (-600, -100)
    mix_populations.location = (-200, 200)
    dust_color.location = (-200, -200)
    dust_mix.location = (0, 0)
    final_emission.location = (200, 0)
    output_node.location = (400, 0)

    # Verbindungen
    # Farb-Mapping basierend auf Age/Color Index
    ng.links.new(input_node.outputs["Color Index"], color_ramp_temp.inputs["Fac"])

    # Junge Sterne (blau, hohe SFR)
    emission_young.inputs["Color"].default_value = (0.7, 0.8, 1.0, 1.0)
    ng.links.new(
        input_node.outputs["Star Formation Rate"], emission_young.inputs["Strength"]
    )

    # Alte Sterne (rot/gelb)
    ng.links.new(color_ramp_temp.outputs["Color"], emission_old.inputs["Color"])
    ng.links.new(input_node.outputs["Age"], emission_old.inputs["Strength"])

    # Population mixing basierend auf SFR
    ng.links.new(input_node.outputs["Star Formation Rate"], sfr_normalize.inputs[0])
    ng.links.new(sfr_normalize.outputs["Value"], mix_populations.inputs["Fac"])
    ng.links.new(emission_young.outputs["Emission"], mix_populations.inputs["Color1"])
    ng.links.new(emission_old.outputs["Emission"], mix_populations.inputs["Color2"])

    # Dust extinction anwenden
    ng.links.new(mix_populations.outputs["Result"], dust_mix.inputs["Color1"])
    ng.links.new(dust_color.outputs["Color"], dust_mix.inputs["Color2"])
    ng.links.new(input_node.outputs["Dust Extinction"], dust_mix.inputs["Fac"])

    # Final output
    ng.links.new(dust_mix.outputs["Result"], output_node.inputs["Shader"])

    return ng


def create_galaxy_bulge_node_group():
    """
    Erstellt eine Galaxy Bulge Shader Node Group.

    Returns:
        bpy.types.ShaderNodeTree: Die erstellte Node Group
    """
    # Node-Group erstellen
    ng = bpy.data.node_groups.new("ALBPY_GalaxyBulge", "ShaderNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input-Sockets
    interface.new_socket(
        name="Bulge Mass", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Metallicity", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(name="Age", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Central Velocity Dispersion",
        in_out="INPUT",
        socket_type="NodeSocketFloat",
    )

    # Output-Sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Default-Werte für typischen Bulge
    ng.interface.items_tree[0].default_value = 1e10  # Bulge Mass (Msun)
    ng.interface.items_tree[1].default_value = 1.5  # Metallicity (super-solar)
    ng.interface.items_tree[2].default_value = 12.0  # Age (Gyr, alt)
    ng.interface.items_tree[3].default_value = 200.0  # Velocity Dispersion (km/s)

    # Nodes erstellen
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Bulge ist typischerweise rot/gelb (alte Sterne)
    bulge_color = ng.nodes.new("ShaderNodeRGB")
    bulge_color.outputs[0].default_value = (1.0, 0.8, 0.6, 1.0)

    # Emission basierend auf Masse
    mass_normalize = ng.nodes.new("ShaderNodeMath")
    mass_normalize.operation = "DIVIDE"
    mass_normalize.inputs[1].default_value = 1e11  # Normalisierung

    bulge_emission = ng.nodes.new("ShaderNodeEmission")

    # Positionierung
    input_node.location = (-400, 0)
    bulge_color.location = (-200, 100)
    mass_normalize.location = (-200, -100)
    bulge_emission.location = (0, 0)
    output_node.location = (200, 0)

    # Verbindungen
    ng.links.new(bulge_color.outputs["Color"], bulge_emission.inputs["Color"])
    ng.links.new(input_node.outputs["Bulge Mass"], mass_normalize.inputs[0])
    ng.links.new(mass_normalize.outputs["Value"], bulge_emission.inputs["Strength"])
    ng.links.new(bulge_emission.outputs["Emission"], output_node.inputs["Shader"])

    return ng


def create_galaxy_halo_node_group():
    """
    Erstellt eine Galaxy Halo Shader Node Group für diffuses Licht.

    Returns:
        bpy.types.ShaderNodeTree: Die erstellte Node Group
    """
    # Node-Group erstellen
    ng = bpy.data.node_groups.new("ALBPY_GalaxyHalo", "ShaderNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input-Sockets
    interface.new_socket(
        name="Halo Mass", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Halo Color", in_out="INPUT", socket_type="NodeSocketColor"
    )
    interface.new_socket(
        name="Halo Intensity", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Halo Size", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output-Sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Default-Werte
    ng.interface.items_tree[0].default_value = 1e12  # Halo Mass (Msun)
    ng.interface.items_tree[1].default_value = (0.9, 0.9, 1.0, 1.0)  # Leicht bläulich
    ng.interface.items_tree[2].default_value = 0.1  # Schwach
    ng.interface.items_tree[3].default_value = 100.0  # Größe (kpc)

    # Nodes erstellen
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Sehr schwache Emission für Halo
    halo_emission = ng.nodes.new("ShaderNodeEmission")

    # Transparenz für diffusen Effekt
    transparent = ng.nodes.new("ShaderNodeBsdfTransparent")
    mix_shader = ng.nodes.new("ShaderNodeMixShader")

    # Positionierung
    input_node.location = (-400, 0)
    halo_emission.location = (-200, 100)
    transparent.location = (-200, -100)
    mix_shader.location = (0, 0)
    output_node.location = (200, 0)

    # Verbindungen
    ng.links.new(input_node.outputs["Halo Color"], halo_emission.inputs["Color"])
    ng.links.new(input_node.outputs["Halo Intensity"], halo_emission.inputs["Strength"])
    ng.links.new(input_node.outputs["Halo Intensity"], mix_shader.inputs["Fac"])
    ng.links.new(halo_emission.outputs["Emission"], mix_shader.inputs[2])
    ng.links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
    ng.links.new(mix_shader.outputs["Shader"], output_node.inputs["Shader"])

    return ng


def create_galaxy_composite_node_group():
    """
    Erstellt eine Composite Galaxy Shader Node Group die alle Komponenten kombiniert.

    Returns:
        bpy.types.ShaderNodeTree: Die erstellte Node Group
    """
    # Node-Group erstellen
    ng = bpy.data.node_groups.new("ALBPY_GalaxyComposite", "ShaderNodeTree")

    # Interface API (Blender 4.4)
    interface = ng.interface

    # Input-Sockets
    interface.new_socket(
        name="Galaxy Type", in_out="INPUT", socket_type="NodeSocketString"
    )
    interface.new_socket(
        name="Stellar Mass", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Bulge to Disk Ratio", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Star Formation Rate", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(
        name="Metallicity", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    interface.new_socket(name="Age", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket(
        name="Dust Extinction", in_out="INPUT", socket_type="NodeSocketFloat"
    )

    # Output-Sockets
    interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    # Default-Werte für Sb Galaxie
    ng.interface.items_tree[0].default_value = "Sb"
    ng.interface.items_tree[1].default_value = 5e10  # Stellar Mass
    ng.interface.items_tree[2].default_value = 0.3  # B/D Ratio
    ng.interface.items_tree[3].default_value = 3.0  # SFR
    ng.interface.items_tree[4].default_value = 1.0  # Metallicity
    ng.interface.items_tree[5].default_value = 8.0  # Age
    ng.interface.items_tree[6].default_value = 0.3  # Dust

    # Nodes erstellen
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")

    # Sub-Gruppen für Komponenten
    disk_group = ng.nodes.new("ShaderNodeGroup")
    disk_group.node_tree = bpy.data.node_groups.get("ALBPY_GalaxyDisk")

    bulge_group = ng.nodes.new("ShaderNodeGroup")
    bulge_group.node_tree = bpy.data.node_groups.get("ALBPY_GalaxyBulge")

    halo_group = ng.nodes.new("ShaderNodeGroup")
    halo_group.node_tree = bpy.data.node_groups.get("ALBPY_GalaxyHalo")

    # Mix Shader für Komponenten
    mix_disk_bulge = ng.nodes.new("ShaderNodeMixShader")
    mix_final = ng.nodes.new("ShaderNodeAddShader")

    # Positionierung
    input_node.location = (-600, 0)
    disk_group.location = (-300, 200)
    bulge_group.location = (-300, 0)
    halo_group.location = (-300, -200)
    mix_disk_bulge.location = (0, 100)
    mix_final.location = (200, 0)
    output_node.location = (400, 0)

    # Verbindungen - Disk
    ng.links.new(
        input_node.outputs["Star Formation Rate"],
        disk_group.inputs["Star Formation Rate"],
    )
    ng.links.new(input_node.outputs["Metallicity"], disk_group.inputs["Metallicity"])
    ng.links.new(input_node.outputs["Age"], disk_group.inputs["Age"])
    ng.links.new(
        input_node.outputs["Dust Extinction"], disk_group.inputs["Dust Extinction"]
    )

    # Verbindungen - Bulge
    ng.links.new(input_node.outputs["Stellar Mass"], bulge_group.inputs["Bulge Mass"])
    ng.links.new(input_node.outputs["Metallicity"], bulge_group.inputs["Metallicity"])
    ng.links.new(input_node.outputs["Age"], bulge_group.inputs["Age"])

    # Verbindungen - Halo
    ng.links.new(input_node.outputs["Stellar Mass"], halo_group.inputs["Halo Mass"])

    # Shader mixing
    ng.links.new(
        input_node.outputs["Bulge to Disk Ratio"], mix_disk_bulge.inputs["Fac"]
    )
    ng.links.new(disk_group.outputs["Shader"], mix_disk_bulge.inputs[1])
    ng.links.new(bulge_group.outputs["Shader"], mix_disk_bulge.inputs[2])

    ng.links.new(mix_disk_bulge.outputs["Shader"], mix_final.inputs[0])
    ng.links.new(halo_group.outputs["Shader"], mix_final.inputs[1])

    ng.links.new(mix_final.outputs["Shader"], output_node.inputs["Shader"])

    return ng


# Galaxy Type Presets
GALAXY_SHADER_PRESETS = {
    "E0": {
        "Galaxy Type": "E0",
        "Stellar Mass": 1e11,
        "Bulge to Disk Ratio": 100.0,  # Pure bulge
        "Star Formation Rate": 0.1,
        "Metallicity": 1.5,
        "Age": 12.0,
        "Dust Extinction": 0.1,
    },
    "Sa": {
        "Galaxy Type": "Sa",
        "Stellar Mass": 8e10,
        "Bulge to Disk Ratio": 0.8,
        "Star Formation Rate": 2.0,
        "Metallicity": 1.2,
        "Age": 10.0,
        "Dust Extinction": 0.2,
    },
    "Sb": {
        "Galaxy Type": "Sb",
        "Stellar Mass": 5e10,
        "Bulge to Disk Ratio": 0.3,
        "Star Formation Rate": 3.0,
        "Metallicity": 1.0,
        "Age": 8.0,
        "Dust Extinction": 0.3,
    },
    "Sc": {
        "Galaxy Type": "Sc",
        "Stellar Mass": 3e10,
        "Bulge to Disk Ratio": 0.1,
        "Star Formation Rate": 5.0,
        "Metallicity": 0.8,
        "Age": 6.0,
        "Dust Extinction": 0.4,
    },
    "Irr": {
        "Galaxy Type": "Irr",
        "Stellar Mass": 1e9,
        "Bulge to Disk Ratio": 0.02,
        "Star Formation Rate": 8.0,
        "Metallicity": 0.5,
        "Age": 4.0,
        "Dust Extinction": 0.2,
    },
}


def get_galaxy_shader_preset(galaxy_type: str) -> Dict[str, Any]:
    """Get shader preset for galaxy type."""
    return GALAXY_SHADER_PRESETS.get(galaxy_type, GALAXY_SHADER_PRESETS["Sb"])


def apply_galaxy_shader_preset(material: bpy.types.Material, galaxy_type: str) -> None:
    """Apply galaxy shader preset to material."""
    preset = get_galaxy_shader_preset(galaxy_type)

    if not material.use_nodes:
        material.use_nodes = True

    # Find the galaxy composite node group
    for node in material.node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_GalaxyComposite" in node.node_tree.name:
                # Apply preset parameters
                for param_name, value in preset.items():
                    if param_name in node.inputs:
                        node.inputs[param_name].default_value = value
                break


def create_galaxy_material(name: str, galaxy_type: str = "Sb") -> bpy.types.Material:
    """
    Create complete galaxy material with all components.

    Args:
        name: Material name
        galaxy_type: Galaxy morphological type

    Returns:
        bpy.types.Material: Created material
    """
    # Create new material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    # Clear default nodes
    mat.node_tree.nodes.clear()

    # Add galaxy composite node group
    galaxy_group = mat.node_tree.nodes.new("ShaderNodeGroup")
    galaxy_group.node_tree = bpy.data.node_groups.get("ALBPY_GalaxyComposite")

    # Add material output
    output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")

    # Position nodes
    galaxy_group.location = (0, 0)
    output.location = (300, 0)

    # Connect nodes
    mat.node_tree.links.new(galaxy_group.outputs["Shader"], output.inputs["Surface"])

    # Apply galaxy type preset
    apply_galaxy_shader_preset(mat, galaxy_type)

    return mat


def register():
    """Register galaxy shader node groups using factory functions."""
    node_groups_to_create = [
        create_galaxy_disk_node_group,
        create_galaxy_bulge_node_group,
        create_galaxy_halo_node_group,
        create_galaxy_composite_node_group,
    ]

    for create_func in node_groups_to_create:
        try:
            create_func()
        except Exception as e:
            print(f"Failed to create galaxy node group: {e}")


def unregister():
    """Unregister galaxy shader node groups."""
    galaxy_node_groups = [
        "ALBPY_GalaxyDisk",
        "ALBPY_GalaxyBulge",
        "ALBPY_GalaxyHalo",
        "ALBPY_GalaxyComposite",
    ]

    for group_name in galaxy_node_groups:
        if group_name in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
