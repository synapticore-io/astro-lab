"""
Compositing Node Groups for AlbPy
=================================

Modern Blender 4.4 compositing node groups for astronomical post-processing.
"""

import logging
from typing import Any, Dict

import bpy

# Import individual compositing modules
from . import (  # Additional modules can be imported here; lens_flare,; color_grading,; star_glow,; multi_panel,
    glare,
)

logger = logging.getLogger(__name__)


def register():
    """Register all compositing node groups using factory functions."""
    logger.info("🎬 Registering AlbPy Compositing Node Groups...")

    try:
        # Register core astronomical compositing nodes
        glare.register()

        # Additional compositing nodes (implement as needed)
        # lens_flare.register()
        # color_grading.register()
        # star_glow.register()
        # multi_panel.register()

        logger.info("✅ Compositing Node Groups registered successfully")

    except Exception as e:
        logger.error(f"❌ Failed to register Compositing Node Groups: {e}")
        raise


def unregister():
    """Unregister all compositing node groups."""
    logger.info("🧹 Unregistering AlbPy Compositing Node Groups...")

    try:
        # Unregister compositing modules
        # Additional modules first
        # multi_panel.unregister()
        # star_glow.unregister()
        # color_grading.unregister()
        # lens_flare.unregister()

        # Core modules
        glare.unregister()

        logger.info("✅ Compositing Node Groups unregistered successfully")

    except Exception as e:
        logger.error(f"❌ Failed to unregister Compositing Node Groups: {e}")


def get_available_compositing_nodes() -> Dict[str, Any]:
    """
    Get list of available compositing node groups.

    Returns:
        Dict mapping node group names to their metadata
    """
    available_nodes = {}

    # Check which ALBPY compositing node groups exist
    albpy_compositing_nodes = [
        "ALBPY_StellarGlare",
        "ALBPY_AiryDisk",
        "ALBPY_AtmosphericTurbulence",
        # Add more as implemented
        # "ALBPY_LensFlare",
        # "ALBPY_ColorGrading",
        # "ALBPY_StarGlow",
        # "ALBPY_MultiPanel",
    ]

    for node_name in albpy_compositing_nodes:
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


def setup_astronomical_compositor(
    scene_name: str = "AstroScene",
) -> bpy.types.CompositorNodeTree:
    """
    Setup compositor with astronomical post-processing nodes.

    Args:
        scene_name: Name of the scene to setup compositing for

    Returns:
        The compositor node tree
    """
    # Get or create scene
    if scene_name in bpy.data.scenes:
        scene = bpy.data.scenes[scene_name]
    else:
        scene = bpy.data.scenes.new(scene_name)

    # Enable compositor
    scene.use_nodes = True
    compositor = scene.node_tree

    # Clear existing nodes
    compositor.nodes.clear()

    # Create basic astronomical compositing pipeline

    # Input nodes
    render_layers = compositor.nodes.new("CompositorNodeRLayers")

    # Stellar glare
    stellar_glare = compositor.nodes.new("CompositorNodeGroup")
    stellar_glare.node_tree = bpy.data.node_groups.get("ALBPY_StellarGlare")
    stellar_glare.name = "Stellar Glare"

    # Atmospheric effects
    atmospheric = compositor.nodes.new("CompositorNodeGroup")
    atmospheric.node_tree = bpy.data.node_groups.get("ALBPY_AtmosphericTurbulence")
    atmospheric.name = "Atmospheric Turbulence"

    # Airy disk
    airy_disk = compositor.nodes.new("CompositorNodeGroup")
    airy_disk.node_tree = bpy.data.node_groups.get("ALBPY_AiryDisk")
    airy_disk.name = "Airy Disk"

    # Color correction
    color_balance = compositor.nodes.new("CompositorNodeColorBalance")
    color_balance.correction_method = "LIFT_GAMMA_GAIN"

    # Final output
    composite = compositor.nodes.new("CompositorNodeComposite")

    # Positioning
    render_layers.location = (-800, 0)
    stellar_glare.location = (-600, 0)
    atmospheric.location = (-400, 0)
    airy_disk.location = (-200, 0)
    color_balance.location = (0, 0)
    composite.location = (200, 0)

    # Connections
    compositor.links.new(render_layers.outputs["Image"], stellar_glare.inputs["Image"])
    compositor.links.new(stellar_glare.outputs["Image"], atmospheric.inputs["Image"])
    compositor.links.new(atmospheric.outputs["Image"], airy_disk.inputs["Image"])
    compositor.links.new(airy_disk.outputs["Image"], color_balance.inputs["Image"])
    compositor.links.new(color_balance.outputs["Image"], composite.inputs["Image"])

    logger.info(f"✅ Setup astronomical compositor for scene: {scene_name}")
    return compositor


def apply_telescope_profile(
    compositor: bpy.types.CompositorNodeTree, telescope_type: str = "refractor"
) -> None:
    """
    Apply telescope-specific settings to compositor.

    Args:
        compositor: Compositor node tree
        telescope_type: Type of telescope (refractor, reflector, etc.)
    """
    # Apply glare settings
    glare.apply_glare_preset(compositor, telescope_type)

    # Telescope-specific atmospheric settings
    atmospheric_settings = {
        "refractor": {"Seeing": 1.2, "Turbulence Strength": 0.15},
        "reflector": {"Seeing": 1.5, "Turbulence Strength": 0.20},
        "schmidt_cassegrain": {"Seeing": 1.3, "Turbulence Strength": 0.18},
        "hubble": {"Seeing": 0.05, "Turbulence Strength": 0.0},  # Space telescope
        "webb": {"Seeing": 0.03, "Turbulence Strength": 0.0},  # Space telescope
    }

    settings = atmospheric_settings.get(
        telescope_type, atmospheric_settings["refractor"]
    )

    # Apply settings to atmospheric turbulence node
    for node in compositor.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_AtmosphericTurbulence" in node.node_tree.name:
                for param_name, value in settings.items():
                    if param_name in node.inputs:
                        node.inputs[param_name].default_value = value
                break

    logger.info(f"✅ Applied {telescope_type} profile to compositor")


def create_astrophotography_compositor() -> bpy.types.CompositorNodeTree:
    """
    Create specialized compositor for astrophotography simulation.

    Returns:
        Compositor node tree configured for astrophotography
    """
    # Create new scene for astrophotography
    astro_scene = bpy.data.scenes.new("Astrophotography")
    astro_scene.use_nodes = True
    compositor = astro_scene.node_tree

    # Setup astronomical compositor
    setup_astronomical_compositor("Astrophotography")

    # Add astrophotography-specific nodes

    # Lens effects
    lens_distortion = compositor.nodes.new("CompositorNodeLensdist")
    lens_distortion.inputs["Distort"].default_value = -0.02  # Slight barrel distortion

    # Vignetting
    vignette = compositor.nodes.new("CompositorNodeEllipseMask")
    vignette.mask_type = "SUBTRACT"

    # Film grain/noise
    noise = compositor.nodes.new("CompositorNodeTexture")
    if "FilmGrain" not in bpy.data.textures:
        grain_texture = bpy.data.textures.new("FilmGrain", "NOISE")
        grain_texture.noise_scale = 0.1
    noise.texture = bpy.data.textures["FilmGrain"]

    # Mix noise
    mix_noise = compositor.nodes.new("CompositorNodeMixRGB")
    mix_noise.blend_type = "OVERLAY"
    mix_noise.inputs["Fac"].default_value = 0.05

    # Insert into pipeline before final composite
    composite_node = None
    for node in compositor.nodes:
        if node.type == "COMPOSITE":
            composite_node = node
            break

    if composite_node:
        # Find the last node before composite
        last_node = None
        for link in compositor.links:
            if link.to_node == composite_node:
                last_node = link.from_node
                compositor.links.remove(link)
                break

        if last_node:
            # Insert new nodes into pipeline
            lens_distortion.location = (
                last_node.location[0] + 200,
                last_node.location[1],
            )
            mix_noise.location = (
                lens_distortion.location[0] + 200,
                lens_distortion.location[1],
            )
            composite_node.location = (
                mix_noise.location[0] + 200,
                mix_noise.location[1],
            )

            # Connect new pipeline
            compositor.links.new(
                last_node.outputs["Image"], lens_distortion.inputs["Image"]
            )
            compositor.links.new(
                lens_distortion.outputs["Image"], mix_noise.inputs["Image1"]
            )
            compositor.links.new(noise.outputs["Color"], mix_noise.inputs["Image2"])
            compositor.links.new(
                mix_noise.outputs["Image"], composite_node.inputs["Image"]
            )

    logger.info("✅ Created astrophotography compositor")
    return compositor


def get_compositing_presets() -> Dict[str, Dict[str, Any]]:
    """Get all available compositing presets."""
    return {
        "telescope_types": glare.GLARE_PRESETS,
        "atmospheric_conditions": {
            "excellent": {"Seeing": 0.8, "Turbulence Strength": 0.05},
            "good": {"Seeing": 1.2, "Turbulence Strength": 0.10},
            "average": {"Seeing": 1.8, "Turbulence Strength": 0.20},
            "poor": {"Seeing": 3.0, "Turbulence Strength": 0.40},
        },
    }


__all__ = [
    "register",
    "unregister",
    "get_available_compositing_nodes",
    "setup_astronomical_compositor",
    "apply_telescope_profile",
    "create_astrophotography_compositor",
    "get_compositing_presets",
]
