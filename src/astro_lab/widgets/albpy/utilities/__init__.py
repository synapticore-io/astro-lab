"""
AlbPy Utilities
===============

Comprehensive utility functions for astronomical calculations and data processing.
Modernized for integration with AstroLab TensorDict system.
"""

# Astronomical data utilities
from .astronomical_data import (
    GALAXY_TYPES,
    HR_DIAGRAM_PARAMS,
    STELLAR_CLASSIFICATION,
    create_sample_galaxy_data,
    create_sample_hr_diagram_data,
    create_sample_nebula_data,
    create_sample_stellar_data,
    get_galaxy_config,
    get_stellar_data,
    validate_hr_diagram_data,
)

# Galaxy calculation utilities
from .galaxy_utilities import (
    calculate_bulge_to_disk_ratio,
    calculate_galaxy_age_from_color,
    calculate_galaxy_angular_size,
    calculate_galaxy_color_index,
    calculate_galaxy_density_profile,
    calculate_galaxy_distance_modulus,
    calculate_galaxy_luminosity_distance,
    calculate_galaxy_mass_from_luminosity,
    calculate_galaxy_metallicity,
    calculate_galaxy_rotation_curve,
    calculate_galaxy_size_from_mass,
    calculate_galaxy_surface_brightness,
    calculate_star_formation_rate,
    get_galaxy_morphology_params,
    get_galaxy_properties,
)

# Nebula calculation utilities
from .nebula_utilities import (
    calculate_emission_line_ratios,
    calculate_nebula_angular_size,
    calculate_nebula_color_index,
    calculate_nebula_density,
    calculate_nebula_distance_modulus,
    calculate_nebula_expansion_velocity,
    calculate_nebula_ionization_parameter,
    calculate_nebula_luminosity,
    calculate_nebula_mass,
    calculate_nebula_metallicity,
    calculate_nebula_size,
    calculate_nebula_temperature,
    get_nebula_properties,
)

# Physics calculation utilities
from .physics_utilities import (
    calculate_dark_matter_profile,
    calculate_escape_velocity,
    calculate_gravitational_force,
    calculate_gravitational_potential,
    calculate_hill_sphere,
    calculate_hohmann_transfer,
    calculate_orbital_elements,
    calculate_orbital_period,
    calculate_orbital_position,
    calculate_roche_limit,
    calculate_tidal_force,
    calculate_virial_velocity,
    create_poliastro_orbit,
    propagate_orbit,
)

# Post-processing utilities
from .post_processing_utilities import (
    apply_astrophotography_processing,
    calculate_bloom_intensity,
    calculate_chromatic_aberration,
    calculate_color_temperature_to_rgb,
    calculate_contrast_curve,
    calculate_depth_of_field,
    calculate_exposure_value,
    calculate_film_grain_intensity,
    calculate_lens_distortion,
    calculate_motion_blur_angle,
    calculate_saturation_multiplier,
    calculate_signal_to_noise_ratio,
    calculate_vignette_intensity,
    get_cinematic_preset,
    get_dramatic_preset,
    get_dreamy_preset,
    get_scientific_preset,
)

__all__ = [
    # Astronomical data
    "GALAXY_TYPES",
    "HR_DIAGRAM_PARAMS",
    "STELLAR_CLASSIFICATION",
    "create_sample_stellar_data",
    "get_galaxy_config",
    "get_stellar_data",
    "validate_hr_diagram_data",
    "create_sample_galaxy_data",
    "create_sample_nebula_data",
    "create_sample_hr_diagram_data",
    # Galaxy utilities
    "calculate_galaxy_age_from_color",
    "calculate_galaxy_angular_size",
    "calculate_galaxy_color_index",
    "calculate_galaxy_density_profile",
    "calculate_galaxy_distance_modulus",
    "calculate_galaxy_luminosity_distance",
    "calculate_galaxy_mass_from_luminosity",
    "calculate_galaxy_metallicity",
    "calculate_galaxy_rotation_curve",
    "calculate_galaxy_size_from_mass",
    "calculate_star_formation_rate",
    "get_galaxy_properties",
    "get_galaxy_morphology_params",
    "calculate_galaxy_surface_brightness",
    "calculate_bulge_to_disk_ratio",
    # Nebula utilities
    "calculate_nebula_angular_size",
    "calculate_nebula_color_index",
    "calculate_nebula_density",
    "calculate_nebula_distance_modulus",
    "calculate_nebula_expansion_velocity",
    "calculate_nebula_luminosity",
    "calculate_nebula_mass",
    "calculate_nebula_metallicity",
    "calculate_nebula_size",
    "calculate_nebula_temperature",
    "get_nebula_properties",
    "calculate_nebula_ionization_parameter",
    "calculate_emission_line_ratios",
    # Physics utilities
    "calculate_escape_velocity",
    "calculate_gravitational_force",
    "calculate_hill_sphere",
    "calculate_orbital_elements",
    "calculate_orbital_period",
    "calculate_roche_limit",
    "calculate_tidal_force",
    "create_poliastro_orbit",
    "propagate_orbit",
    "calculate_orbital_position",
    "calculate_hohmann_transfer",
    "calculate_gravitational_potential",
    "calculate_virial_velocity",
    "calculate_dark_matter_profile",
    # Post-processing utilities
    "calculate_bloom_intensity",
    "calculate_chromatic_aberration",
    "calculate_color_temperature_to_rgb",
    "calculate_contrast_curve",
    "calculate_depth_of_field",
    "calculate_exposure_value",
    "calculate_film_grain_intensity",
    "calculate_lens_distortion",
    "calculate_motion_blur_angle",
    "calculate_saturation_multiplier",
    "calculate_vignette_intensity",
    "get_cinematic_preset",
    "get_dramatic_preset",
    "get_dreamy_preset",
    "get_scientific_preset",
    "apply_astrophotography_processing",
    "calculate_signal_to_noise_ratio",
]
