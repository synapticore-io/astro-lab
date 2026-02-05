"""
Shader Node Groups for AlbPy
============================

Modern Blender 4.4 shader node groups for astronomical materials.
"""

import logging
from typing import Any, Dict

import bpy

# Import individual shader modules
from . import (  # Additional modules can be imported here as they are refactored
    absorption,
    doppler,
    emission,
    galaxy,
    glass,
    redshift,
    star,
)

logger = logging.getLogger(__name__)


def register():
    """Register all shader node groups using factory functions."""
    logger.info("🎨 Registering AlbPy Shader Node Groups...")

    try:
        # Register core astronomical shader nodes
        star.register()
        galaxy.register()
        emission.register()
        absorption.register()
        doppler.register()
        redshift.register()
        glass.register()

        # Additional shader nodes (implement as needed)
        # nebula.register()
        # planet.register()
        # atmosphere.register()
        # holographic.register()
        # iridescent.register()
        # metallic.register()
        # energy_field.register()
        # force_field.register()
        # subsurface.register()

        logger.info("✅ Shader Node Groups registered successfully")

    except Exception as e:
        logger.error(f"❌ Failed to register Shader Node Groups: {e}")
        raise


def unregister():
    """Unregister all shader node groups."""
    logger.info("🧹 Unregistering AlbPy Shader Node Groups...")

    try:
        # Unregister shader modules (reverse order)
        glass.unregister()
        redshift.unregister()
        doppler.unregister()
        absorption.unregister()
        emission.unregister()
        galaxy.unregister()
        star.unregister()

        # Additional modules (when implemented)
        # subsurface.unregister()
        # force_field.unregister()
        # energy_field.unregister()
        # metallic.unregister()
        # iridescent.unregister()
        # holographic.unregister()
        # atmosphere.unregister()
        # planet.unregister()
        # nebula.unregister()

        logger.info("✅ Shader Node Groups unregistered successfully")

    except Exception as e:
        logger.error(f"❌ Failed to unregister Shader Node Groups: {e}")


def get_available_shader_nodes() -> Dict[str, Any]:
    """
    Get list of available shader node groups.

    Returns:
        Dict mapping node group names to their metadata
    """
    available_nodes = {}

    # Check which ALBPY shader node groups exist
    albpy_shader_nodes = [
        "ALBPY_StellarBlackbody",
        "ALBPY_StellarClassification",
        "ALBPY_GalaxyDisk",
        "ALBPY_GalaxyBulge",
        "ALBPY_GalaxyHalo",
        "ALBPY_GalaxyComposite",
        "ALBPY_EmissionShader",
        "ALBPY_AdvancedEmission",
        "ALBPY_AbsorptionShader",
        "ALBPY_AtmosphericAbsorption",
        "ALBPY_DopplerEffect",
        "ALBPY_RedshiftVisualization",
        "ALBPY_RedshiftShader",
        "ALBPY_LymanAlphaForest",
        "ALBPY_GlassShader",
        "ALBPY_CrystalShader",
        # Add more as implemented
        # "ALBPY_NebulaEmission",
        # "ALBPY_PlanetSurface",
        # "ALBPY_AtmosphereScattering",
        # "ALBPY_HolographicShader",
        # "ALBPY_IridescentShader",
        # "ALBPY_MetallicShader",
        # "ALBPY_EnergyField",
        # "ALBPY_ForceField",
        # "ALBPY_SubsurfaceShader",
    ]

    for node_name in albpy_shader_nodes:
        if node_name in bpy.data.node_groups:
            node_group = bpy.data.node_groups[node_name]
            available_nodes[node_name] = {
                "name": node_group.name,
                "type": node_group.type,
                "inputs": [
                    inp.name
                    for inp in node_group.interface.items_tree
                    if inp.in_out == "INPUT"
                ],
                "outputs": [
                    out.name
                    for out in node_group.interface.items_tree
                    if out.in_out == "OUTPUT"
                ],
                "description": getattr(
                    node_group, "description", "No description available"
                ),
            }

    return available_nodes


def create_astronomical_material(
    name: str, shader_type: str = "stellar", **kwargs
) -> bpy.types.Material:
    """
    Create astronomical material with appropriate shader node group.

    Args:
        name: Material name
        shader_type: Type of astronomical shader
        **kwargs: Additional parameters for the shader

    Returns:
        Created material with node group
    """
    # Create new material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    # Clear default nodes
    mat.node_tree.nodes.clear()

    # Add appropriate node group based on type
    if shader_type == "stellar":
        return star.create_stellar_material(name, kwargs.get("spectral_class", "G"))
    elif shader_type == "galaxy":
        return galaxy.create_galaxy_material(name, kwargs.get("galaxy_type", "Sb"))
    elif shader_type == "emission":
        return emission.create_emission_material(
            name, kwargs.get("preset", "star_core")
        )
    elif shader_type == "absorption":
        return absorption.create_absorption_material(
            name, kwargs.get("preset", "dust_lane")
        )
    elif shader_type == "doppler":
        return doppler.create_doppler_material(
            name, kwargs.get("preset", "approaching_star")
        )
    elif shader_type == "redshift":
        return redshift.create_redshift_material(
            name, kwargs.get("preset", "local_universe")
        )
    elif shader_type == "glass":
        return glass.create_glass_material(name, kwargs.get("preset", "optical_glass"))
    # elif shader_type == "nebula":
    #     return nebula.create_nebula_material(name, **kwargs)
    # elif shader_type == "planet":
    #     return planet.create_planet_material(name, **kwargs)
    # elif shader_type == "atmosphere":
    #     return atmosphere.create_atmosphere_material(name, **kwargs)
    else:
        logger.warning(f"Unknown shader type: {shader_type}")
        return mat


def get_stellar_presets() -> Dict[str, Dict[str, Any]]:
    """Get all available stellar presets."""
    return star.STELLAR_PRESETS


def get_galaxy_presets() -> Dict[str, Dict[str, Any]]:
    """Get all available galaxy presets."""
    return galaxy.GALAXY_SHADER_PRESETS


def get_emission_presets() -> Dict[str, Dict[str, Any]]:
    """Get all available emission presets."""
    return emission.EMISSION_PRESETS


def get_absorption_presets() -> Dict[str, Dict[str, Any]]:
    """Get all available absorption presets."""
    return absorption.ABSORPTION_PRESETS


def get_doppler_presets() -> Dict[str, Dict[str, Any]]:
    """Get all available Doppler effect presets."""
    return doppler.DOPPLER_PRESETS


def get_redshift_presets() -> Dict[str, Dict[str, Any]]:
    """Get all available redshift presets."""
    return redshift.REDSHIFT_PRESETS


def get_glass_presets() -> Dict[str, Dict[str, Any]]:
    """Get all available glass presets."""
    return glass.GLASS_PRESETS


def get_all_presets() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Get all shader presets organized by type."""
    return {
        "stellar": get_stellar_presets(),
        "galaxy": get_galaxy_presets(),
        "emission": get_emission_presets(),
        "absorption": get_absorption_presets(),
        "doppler": get_doppler_presets(),
        "redshift": get_redshift_presets(),
        "glass": get_glass_presets(),
        # Add more as implemented
        # "nebula": get_nebula_presets(),
        # "planet": get_planet_presets(),
        # "atmosphere": get_atmosphere_presets(),
    }


def apply_material_preset(
    material: bpy.types.Material, preset_name: str, shader_type: str = "stellar"
) -> None:
    """
    Apply preset to existing material.

    Args:
        material: Blender material
        preset_name: Name of the preset
        shader_type: Type of shader ("stellar", "galaxy", etc.)
    """
    if shader_type == "stellar":
        star.apply_stellar_preset(material, preset_name)
    elif shader_type == "galaxy":
        galaxy.apply_galaxy_shader_preset(material, preset_name)
    elif shader_type == "emission":
        emission.apply_emission_preset(material, preset_name)
    elif shader_type == "absorption":
        absorption.apply_absorption_preset(material, preset_name)
    elif shader_type == "doppler":
        doppler.apply_doppler_preset(material, preset_name)
    elif shader_type == "redshift":
        redshift.apply_redshift_preset(material, preset_name)
    elif shader_type == "glass":
        glass.apply_glass_preset(material, preset_name)
    else:
        logger.warning(f"Presets for {shader_type} not yet implemented")


def create_material_library() -> Dict[str, bpy.types.Material]:
    """
    Create a library of all available astronomical materials.

    Returns:
        Dict mapping material names to created materials
    """
    materials = {}

    # Create stellar materials
    stellar_presets = get_stellar_presets()
    for spectral_class in stellar_presets.keys():
        mat_name = f"Stellar_{spectral_class}_Class"
        materials[mat_name] = create_astronomical_material(
            mat_name, "stellar", spectral_class=spectral_class
        )

    # Create galaxy materials
    galaxy_presets = get_galaxy_presets()
    for galaxy_type in galaxy_presets.keys():
        mat_name = f"Galaxy_{galaxy_type}_Type"
        materials[mat_name] = create_astronomical_material(
            mat_name, "galaxy", galaxy_type=galaxy_type
        )

    # Create emission materials
    emission_presets = get_emission_presets()
    for preset_name in emission_presets.keys():
        mat_name = f"Emission_{preset_name}"
        materials[mat_name] = create_astronomical_material(
            mat_name, "emission", preset=preset_name
        )

    # Create absorption materials
    absorption_presets = get_absorption_presets()
    for preset_name in absorption_presets.keys():
        mat_name = f"Absorption_{preset_name}"
        materials[mat_name] = create_astronomical_material(
            mat_name, "absorption", preset=preset_name
        )

    # Create Doppler materials
    doppler_presets = get_doppler_presets()
    for preset_name in doppler_presets.keys():
        mat_name = f"Doppler_{preset_name}"
        materials[mat_name] = create_astronomical_material(
            mat_name, "doppler", preset=preset_name
        )

    # Create redshift materials
    redshift_presets = get_redshift_presets()
    for preset_name in redshift_presets.keys():
        mat_name = f"Redshift_{preset_name}"
        materials[mat_name] = create_astronomical_material(
            mat_name, "redshift", preset=preset_name
        )

    # Create glass materials
    glass_presets = get_glass_presets()
    for preset_name in glass_presets.keys():
        mat_name = f"Glass_{preset_name}"
        materials[mat_name] = create_astronomical_material(
            mat_name, "glass", preset=preset_name
        )

    logger.info(f"Created material library with {len(materials)} materials")
    return materials


def cleanup_albpy_materials():
    """Remove all AlbPy materials from the scene."""
    materials_to_remove = []
    for mat in bpy.data.materials:
        if mat.name.startswith(
            (
                "Stellar_",
                "Galaxy_",
                "Emission_",
                "Absorption_",
                "Doppler_",
                "Redshift_",
                "Glass_",
                "Nebula_",
                "Planet_",
                "Atmosphere_",
                "ALBPY_",
            )
        ):
            materials_to_remove.append(mat)

    for mat in materials_to_remove:
        bpy.data.materials.remove(mat)

    logger.info(f"Removed {len(materials_to_remove)} AlbPy materials")


def get_material_statistics() -> Dict[str, Any]:
    """Get statistics about AlbPy materials in the scene."""
    stats = {
        "total_materials": len(bpy.data.materials),
        "albpy_materials": 0,
        "by_type": {},
        "node_groups_used": set(),
    }

    # Count AlbPy materials by type
    for mat in bpy.data.materials:
        if mat.name.startswith(
            (
                "Stellar_",
                "Galaxy_",
                "Emission_",
                "Absorption_",
                "Doppler_",
                "Redshift_",
                "Glass_",
                "ALBPY_",
            )
        ):
            stats["albpy_materials"] += 1

            # Determine type
            for prefix in [
                "Stellar",
                "Galaxy",
                "Emission",
                "Absorption",
                "Doppler",
                "Redshift",
                "Glass",
            ]:
                if mat.name.startswith(f"{prefix}_"):
                    stats["by_type"][prefix] = stats["by_type"].get(prefix, 0) + 1
                    break

            # Check for node groups
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == "GROUP" and node.node_tree:
                        if "ALBPY_" in node.node_tree.name:
                            stats["node_groups_used"].add(node.node_tree.name)

    stats["node_groups_used"] = list(stats["node_groups_used"])
    return stats


__all__ = [
    "register",
    "unregister",
    "get_available_shader_nodes",
    "create_astronomical_material",
    "get_stellar_presets",
    "get_galaxy_presets",
    "get_emission_presets",
    "get_absorption_presets",
    "get_doppler_presets",
    "get_redshift_presets",
    "get_glass_presets",
    "get_all_presets",
    "apply_material_preset",
    "create_material_library",
    "cleanup_albpy_materials",
    "get_material_statistics",
]
