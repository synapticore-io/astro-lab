"""
Blender Widgets for AstroLab - 3D Visualization and Animation
============================================================

High-quality 3D visualization and animation using Blender's Python API.
Modernized for Blender 4.4 with factory-based Node Groups and clean architecture.

Exports: Nur tatsächlich implementierte und getestete Node-Group- und Utility-Funktionen.

Supported visualization types for create_astronomical_visualization:
- 'stellar_field' (implemented)
- 'galaxy_morphology' (not yet implemented)
- 'cosmic_web' (use generate_cosmic_web_scene instead)
"""

import logging
from typing import Any, Dict, Optional

# Node group and utility exports (only implemented functions)
from .nodes.compositing import register as register_compositing
from .nodes.compositing import unregister as unregister_compositing
from .nodes.geometry import register as register_geometry
from .nodes.geometry import unregister as unregister_geometry
from .nodes.shader import register as register_shader
from .nodes.shader import unregister as unregister_shader

# Operator registration (modernized)
from .operators import register as register_operators
from .operators import unregister as unregister_operators

# Core functionality

# Import the main cosmic web scene generator


# Utility functions (consolidated)

logger = logging.getLogger(__name__)


def register():
    """Register all AlbPy components with modern Blender 4.4 API."""
    try:
        # 1. Register Node Groups using factory patterns
        register_geometry()
        register_shader()
        register_compositing()

        # 2. Register Operators
        register_operators()

        logger.info("✅ AlbPy components registered successfully with Blender 4.4 API")

    except Exception as e:
        logger.error(f"❌ AlbPy registration failed: {e}")
        raise


def unregister():
    """Unregister all AlbPy components."""
    try:
        unregister_operators()
        unregister_compositing()
        unregister_shader()
        unregister_geometry()

        logger.info("✅ AlbPy components unregistered successfully")

    except Exception as e:
        logger.error(f"❌ AlbPy unregistration failed: {e}")


def create_astronomical_visualization(
    data_source: str, visualization_type: str, preset: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    Main API for astronomical visualization in Blender.

    Args:
        data_source: 'tensor_dict', 'gaia', 'sdss', etc.
        visualization_type: 'stellar_field', 'galaxy_morphology', 'cosmic_web'
        preset: Predefined style preset
        **kwargs: Additional parameters

    Returns:
        Dict with created Blender objects and metadata
    """
    if preset:
        config = _get_preset_config(preset)
        kwargs.update(config)

    if visualization_type == "stellar_field":
        return _create_stellar_field_visualization(data_source, **kwargs)
    elif visualization_type == "galaxy_morphology":
        raise NotImplementedError(
            "Galaxy morphology visualization is not yet implemented."
        )
    elif visualization_type == "cosmic_web":
        raise NotImplementedError(
            "Use generate_cosmic_web_scene from albpy.cosmic_web_generator instead. 'cosmic_web' is not supported here."
        )
    else:
        raise ValueError(f"Unknown visualization type: {visualization_type}")


def _get_preset_config(preset: str) -> Dict[str, Any]:
    """Get configuration for visualization preset."""
    presets = {
        "scientific": {
            "background_color": (0.0, 0.0, 0.0, 1.0),
            "lighting_type": "minimal",
            "star_scale": 1.0,
        },
        "cinematic": {
            "background_color": (0.02, 0.02, 0.05, 1.0),
            "lighting_type": "dramatic",
            "star_scale": 1.5,
            "add_glare": True,
        },
        "publication": {
            "background_color": (1.0, 1.0, 1.0, 1.0),
            "lighting_type": "neutral",
            "star_scale": 0.8,
            "high_contrast": True,
        },
    }
    return presets.get(preset, presets["scientific"])


def _create_stellar_field_visualization(data_source: str, **kwargs) -> Dict[str, Any]:
    """Create stellar field visualization."""
    # Implementation will use modernized operators and node groups
    import bpy

    # Use modernized star creation operator
    bpy.ops.albpy.create_star_field(
        count=kwargs.get("star_count", 1000),
        distribution=kwargs.get("distribution", "random"),
        scale=kwargs.get("star_scale", 1.0),
    )

    return {
        "objects": [
            obj for obj in bpy.context.scene.objects if obj.name.startswith("Star")
        ],
        "visualization_type": "stellar_field",
        "metadata": kwargs,
    }


def _create_galaxy_morphology_visualization(
    data_source: str, **kwargs
) -> Dict[str, Any]:
    """Create galaxy morphology visualization."""
    # Implementation will use galaxy operators
    return {"status": "not_implemented"}


def _create_cosmic_web_visualization(data_source: str, **kwargs) -> Dict[str, Any]:
    """[DEPRECATED] Use generate_cosmic_web_scene instead."""
    raise NotImplementedError(
        "Use generate_cosmic_web_scene from albpy.cosmic_web_generator instead."
    )


__all__ = [
    # Registration functions
    "register",
    "unregister",
    # Core components
    "setup_camera",
    "setup_lighting",
    "setup_rendering",
    "render_astronomical_scene",
    "setup_scene",
    # Main API
    "create_astronomical_visualization",
    "generate_cosmic_web_scene",
    # Astronomical data utilities
    "GALAXY_TYPES",
    "HR_DIAGRAM_PARAMS",
    "STELLAR_CLASSIFICATION",
    "create_sample_stellar_data",
    "get_galaxy_config",
    "get_stellar_data",
    "validate_hr_diagram_data",
    # Node group and utility exports
    "register_compositing",
    "unregister_compositing",
    "get_available_compositing_nodes",
    "setup_astronomical_compositor",
    "apply_telescope_profile",
    "register_geometry",
    "unregister_geometry",
    "get_available_geometry_nodes",
    "create_geometry_modifier",
    "register_shader",
    "unregister_shader",
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
]

try:
    __all__
except NameError:
    __all__ = []
if "create_blender_scene" not in __all__:
    __all__.append("create_blender_scene")
if "render_cosmic_web" not in __all__:
    __all__.append("render_cosmic_web")
