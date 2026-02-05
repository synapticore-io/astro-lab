"""
Compositing Presets for AlbPy
============================

Provides preset configurations for compositing node groups.
"""

from typing import Any, Dict


class CompositingPresets:
    """Preset configurations for compositing node groups."""

    @staticmethod
    def cinematic_preset() -> Dict[str, Any]:
        """Cinematic post-processing preset."""
        return {
            "lens_flare": {
                "node_group": "ALBPY_NG_LensFlare",
                "parameters": {
                    "Flare Type": "stellar",
                    "Intensity": 0.8,
                    "Distortion": 0.02,
                    "Glow Mix": 0.3,
                },
            },
            "vignette": {
                "node_group": "ALBPY_NG_Vignette",
                "parameters": {
                    "Intensity": 0.4,
                    "Radius": 0.7,
                    "Softness": 0.25,
                },
            },
            "color_grading": {
                "node_group": "ALBPY_NG_ColorGrading",
                "parameters": {
                    "Style": "cinematic",
                    "Contrast": 1.2,
                    "Saturation": 0.9,
                    "Gain": 1.1,
                    "Lift": 0.05,
                    "Gamma": 1.0,
                },
            },
            "star_glow": {
                "node_group": "ALBPY_NG_StarGlow",
                "parameters": {
                    "Glow Intensity": 1.2,
                    "Glow Size": 11,
                    "Cross Pattern": 0.6,
                    "Angle Offset": 0.0,
                },
            },
            "artistic_filters": {
                "node_group": "ALBPY_NG_ArtisticFilters",
                "parameters": {
                    "Filter Type": "cinematic",
                    "Film Grain Intensity": 0.15,
                    "Chromatic Aberration": 0.01,
                    "Noise Scale": 80.0,
                    "Noise Detail": 3.0,
                },
            },
        }

    @staticmethod
    def dramatic_preset() -> Dict[str, Any]:
        """Dramatic post-processing preset."""
        return {
            "lens_flare": {
                "node_group": "ALBPY_NG_LensFlare",
                "parameters": {
                    "Flare Type": "nebula",
                    "Intensity": 1.2,
                    "Distortion": 0.03,
                    "Glow Mix": 0.4,
                },
            },
            "vignette": {
                "node_group": "ALBPY_NG_Vignette",
                "parameters": {
                    "Intensity": 0.6,
                    "Radius": 0.6,
                    "Softness": 0.1,
                },
            },
            "color_grading": {
                "node_group": "ALBPY_NG_ColorGrading",
                "parameters": {
                    "Style": "dramatic",
                    "Contrast": 1.5,
                    "Saturation": 1.4,
                    "Gain": 1.3,
                    "Lift": 0.15,
                    "Gamma": 1.0,
                },
            },
            "star_glow": {
                "node_group": "ALBPY_NG_StarGlow",
                "parameters": {
                    "Glow Intensity": 1.5,
                    "Glow Size": 13,
                    "Cross Pattern": 0.7,
                    "Angle Offset": 0.0,
                },
            },
            "artistic_filters": {
                "node_group": "ALBPY_NG_ArtisticFilters",
                "parameters": {
                    "Filter Type": "vintage",
                    "Film Grain Intensity": 0.2,
                    "Chromatic Aberration": 0.03,
                    "Noise Scale": 120.0,
                    "Noise Detail": 1.5,
                },
            },
        }

    @staticmethod
    def dreamy_preset() -> Dict[str, Any]:
        """Dreamy post-processing preset."""
        return {
            "lens_flare": {
                "node_group": "ALBPY_NG_LensFlare",
                "parameters": {
                    "Flare Type": "galactic",
                    "Intensity": 0.6,
                    "Distortion": 0.015,
                    "Glow Mix": 0.25,
                },
            },
            "vignette": {
                "node_group": "ALBPY_NG_Vignette",
                "parameters": {
                    "Intensity": 0.2,
                    "Radius": 0.8,
                    "Softness": 0.3,
                },
            },
            "color_grading": {
                "node_group": "ALBPY_NG_ColorGrading",
                "parameters": {
                    "Style": "dreamy",
                    "Contrast": 0.8,
                    "Saturation": 1.2,
                    "Gain": 1.0,
                    "Lift": 0.05,
                    "Gamma": 1.0,
                },
            },
            "star_glow": {
                "node_group": "ALBPY_NG_StarGlow",
                "parameters": {
                    "Glow Intensity": 0.8,
                    "Glow Size": 7,
                    "Cross Pattern": 0.3,
                    "Angle Offset": 0.0,
                },
            },
            "artistic_filters": {
                "node_group": "ALBPY_NG_ArtisticFilters",
                "parameters": {
                    "Filter Type": "film_grain",
                    "Film Grain Intensity": 0.1,
                    "Chromatic Aberration": 0.0,
                    "Noise Scale": 100.0,
                    "Noise Detail": 2.0,
                },
            },
        }

    @staticmethod
    def scientific_preset() -> Dict[str, Any]:
        """Scientific post-processing preset."""
        return {
            "lens_flare": {
                "node_group": "ALBPY_NG_LensFlare",
                "parameters": {
                    "Flare Type": "stellar",
                    "Intensity": 0.0,
                    "Distortion": 0.0,
                    "Glow Mix": 0.0,
                },
            },
            "vignette": {
                "node_group": "ALBPY_NG_Vignette",
                "parameters": {
                    "Intensity": 0.0,
                    "Radius": 0.8,
                    "Softness": 0.2,
                },
            },
            "color_grading": {
                "node_group": "ALBPY_NG_ColorGrading",
                "parameters": {
                    "Style": "scientific",
                    "Contrast": 1.1,
                    "Saturation": 1.0,
                    "Gain": 1.0,
                    "Lift": 0.0,
                    "Gamma": 1.0,
                },
            },
            "star_glow": {
                "node_group": "ALBPY_NG_StarGlow",
                "parameters": {
                    "Glow Intensity": 0.0,
                    "Glow Size": 9,
                    "Cross Pattern": 0.0,
                    "Angle Offset": 0.0,
                },
            },
            "artistic_filters": {
                "node_group": "ALBPY_NG_ArtisticFilters",
                "parameters": {
                    "Filter Type": "clean",
                    "Film Grain Intensity": 0.0,
                    "Chromatic Aberration": 0.0,
                    "Noise Scale": 100.0,
                    "Noise Detail": 2.0,
                },
            },
        }

    @staticmethod
    def multi_panel_presets() -> Dict[str, Any]:
        """Multi-panel layout presets."""
        return {
            "2x2_grid": {
                "node_group": "ALBPY_NG_MultiPanel",
                "parameters": {
                    "Rows": 2,
                    "Columns": 2,
                    "Panel Spacing": 0.1,
                    "Border Width": 0.02,
                    "Border Color": (0.3, 0.3, 0.3, 1.0),
                },
            },
            "horizontal_2": {
                "node_group": "ALBPY_NG_MultiPanel",
                "parameters": {
                    "Rows": 1,
                    "Columns": 2,
                    "Panel Spacing": 0.15,
                    "Border Width": 0.03,
                    "Border Color": (0.4, 0.4, 0.4, 1.0),
                },
            },
            "vertical_2": {
                "node_group": "ALBPY_NG_MultiPanel",
                "parameters": {
                    "Rows": 2,
                    "Columns": 1,
                    "Panel Spacing": 0.15,
                    "Border Width": 0.03,
                    "Border Color": (0.4, 0.4, 0.4, 1.0),
                },
            },
            "3x3_grid": {
                "node_group": "ALBPY_NG_MultiPanel",
                "parameters": {
                    "Rows": 3,
                    "Columns": 3,
                    "Panel Spacing": 0.08,
                    "Border Width": 0.015,
                    "Border Color": (0.25, 0.25, 0.25, 1.0),
                },
            },
            "astronomical": {
                "node_group": "ALBPY_NG_MultiPanel",
                "parameters": {
                    "Rows": 2,
                    "Columns": 2,
                    "Panel Spacing": 0.12,
                    "Border Width": 0.025,
                    "Border Color": (0.2, 0.3, 0.5, 1.0),
                },
            },
        }


def register():
    """Register compositing presets."""
    # Presets are just configuration data, no actual registration needed


def unregister():
    """Unregister compositing presets."""
    # Presets are just configuration data, no actual unregistration needed
