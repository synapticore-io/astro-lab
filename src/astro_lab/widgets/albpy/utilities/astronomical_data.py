"""
Astronomical Data Utilities
===========================

Comprehensive astronomical data definitions, sample data generators, and validation functions.
Integration with AstroLab TensorDict system for modern data handling.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Stellar Classification System (Morgan-Keenan)
STELLAR_CLASSIFICATION = {
    "O": {
        "temperature_range": (30000, 50000),
        "mass_range": (15, 90),
        "luminosity_range": (30000, 1000000),
        "color": (0.6, 0.7, 1.0),
        "lifetime_myr": 10,
        "main_sequence_fraction": 0.00003,
        "examples": ["Alnitak", "Mintaka", "Alnilam"],
    },
    "B": {
        "temperature_range": (10000, 30000),
        "mass_range": (2.1, 16),
        "luminosity_range": (25, 30000),
        "color": (0.7, 0.8, 1.0),
        "lifetime_myr": 400,
        "main_sequence_fraction": 0.0013,
        "examples": ["Rigel", "Spica", "Regulus"],
    },
    "A": {
        "temperature_range": (7500, 10000),
        "mass_range": (1.4, 2.1),
        "luminosity_range": (5, 25),
        "color": (0.9, 0.9, 1.0),
        "lifetime_myr": 2000,
        "main_sequence_fraction": 0.006,
        "examples": ["Sirius", "Vega", "Altair"],
    },
    "F": {
        "temperature_range": (6000, 7500),
        "mass_range": (1.04, 1.4),
        "luminosity_range": (1.5, 5),
        "color": (1.0, 1.0, 0.9),
        "lifetime_myr": 7000,
        "main_sequence_fraction": 0.03,
        "examples": ["Procyon", "Canopus", "Polaris"],
    },
    "G": {
        "temperature_range": (5200, 6000),
        "mass_range": (0.8, 1.04),
        "luminosity_range": (0.6, 1.5),
        "color": (1.0, 1.0, 0.8),
        "lifetime_myr": 12000,
        "main_sequence_fraction": 0.076,
        "examples": ["Sun", "Alpha Centauri A", "Capella"],
    },
    "K": {
        "temperature_range": (3700, 5200),
        "mass_range": (0.45, 0.8),
        "luminosity_range": (0.08, 0.6),
        "color": (1.0, 0.8, 0.6),
        "lifetime_myr": 50000,
        "main_sequence_fraction": 0.121,
        "examples": ["Arcturus", "Aldebaran", "Alpha Centauri B"],
    },
    "M": {
        "temperature_range": (2400, 3700),
        "mass_range": (0.08, 0.45),
        "luminosity_range": (0.0001, 0.08),
        "color": (1.0, 0.6, 0.4),
        "lifetime_myr": 100000,
        "main_sequence_fraction": 0.763,
        "examples": ["Proxima Centauri", "Barnard's Star", "Wolf 359"],
    },
}

# Galaxy Classification (Hubble Sequence)
GALAXY_TYPES = {
    "E0": {
        "description": "Elliptical, spherical",
        "ellipticity": 0.0,
        "bulge_fraction": 1.0,
        "disk_fraction": 0.0,
        "star_formation_rate": 0.1,
        "typical_mass": 1e11,
        "color_index": 0.9,
        "examples": ["M87", "NGC 4472"],
    },
    "E3": {
        "description": "Elliptical, moderate flattening",
        "ellipticity": 0.3,
        "bulge_fraction": 1.0,
        "disk_fraction": 0.0,
        "star_formation_rate": 0.2,
        "typical_mass": 8e10,
        "color_index": 0.85,
        "examples": ["NGC 4374", "NGC 4649"],
    },
    "E7": {
        "description": "Elliptical, highly flattened",
        "ellipticity": 0.7,
        "bulge_fraction": 1.0,
        "disk_fraction": 0.0,
        "star_formation_rate": 0.3,
        "typical_mass": 5e10,
        "color_index": 0.8,
        "examples": ["NGC 3115", "NGC 5866"],
    },
    "S0": {
        "description": "Lenticular, no spiral arms",
        "ellipticity": 0.4,
        "bulge_fraction": 0.6,
        "disk_fraction": 0.4,
        "star_formation_rate": 0.5,
        "typical_mass": 7e10,
        "color_index": 0.75,
        "examples": ["NGC 5866", "Cartwheel Galaxy"],
    },
    "Sa": {
        "description": "Spiral, tightly wound arms",
        "ellipticity": 0.2,
        "bulge_fraction": 0.4,
        "disk_fraction": 0.6,
        "star_formation_rate": 2.0,
        "typical_mass": 1e11,
        "color_index": 0.6,
        "examples": ["M81", "NGC 4594"],
    },
    "Sb": {
        "description": "Spiral, moderately wound arms",
        "ellipticity": 0.3,
        "bulge_fraction": 0.3,
        "disk_fraction": 0.7,
        "star_formation_rate": 3.0,
        "typical_mass": 8e10,
        "color_index": 0.5,
        "examples": ["M31 (Andromeda)", "M101"],
    },
    "Sc": {
        "description": "Spiral, loosely wound arms",
        "ellipticity": 0.4,
        "bulge_fraction": 0.2,
        "disk_fraction": 0.8,
        "star_formation_rate": 5.0,
        "typical_mass": 5e10,
        "color_index": 0.4,
        "examples": ["Milky Way", "M33"],
    },
    "SBa": {
        "description": "Barred spiral, tight arms",
        "ellipticity": 0.2,
        "bulge_fraction": 0.35,
        "disk_fraction": 0.65,
        "star_formation_rate": 2.5,
        "typical_mass": 9e10,
        "color_index": 0.55,
        "examples": ["NGC 1365", "NGC 7479"],
    },
    "SBb": {
        "description": "Barred spiral, moderate arms",
        "ellipticity": 0.3,
        "bulge_fraction": 0.25,
        "disk_fraction": 0.75,
        "star_formation_rate": 4.0,
        "typical_mass": 7e10,
        "color_index": 0.45,
        "examples": ["NGC 1300", "M95"],
    },
    "SBc": {
        "description": "Barred spiral, loose arms",
        "ellipticity": 0.4,
        "bulge_fraction": 0.15,
        "disk_fraction": 0.85,
        "star_formation_rate": 6.0,
        "typical_mass": 4e10,
        "color_index": 0.35,
        "examples": ["NGC 175", "M91"],
    },
    "Irr": {
        "description": "Irregular galaxy",
        "ellipticity": 0.6,
        "bulge_fraction": 0.1,
        "disk_fraction": 0.9,
        "star_formation_rate": 8.0,
        "typical_mass": 1e9,
        "color_index": 0.2,
        "examples": ["Large Magellanic Cloud", "Small Magellanic Cloud", "M82"],
    },
}

# HR Diagram Parameters
HR_DIAGRAM_PARAMS = {
    "main_sequence": {
        "mass_range": (0.08, 120),
        "temperature_range": (2400, 50000),
        "luminosity_range": (1e-4, 1e6),
        "lifetime_range": (1e6, 1e11),  # years
        "color_range": (-0.4, 2.0),  # B-V color index
    },
    "red_giants": {
        "mass_range": (0.5, 8),
        "temperature_range": (3000, 5000),
        "luminosity_range": (10, 1000),
        "lifetime_range": (1e6, 1e9),
        "color_range": (1.0, 2.0),
    },
    "white_dwarfs": {
        "mass_range": (0.17, 1.4),
        "temperature_range": (4000, 150000),
        "luminosity_range": (1e-4, 100),
        "lifetime_range": (1e9, 1e15),
        "color_range": (-0.5, 1.5),
    },
    "supergiants": {
        "mass_range": (8, 120),
        "temperature_range": (3000, 50000),
        "luminosity_range": (1000, 1000000),
        "lifetime_range": (1e6, 5e7),
        "color_range": (0.0, 2.0),
    },
}

# Nebula Types
NEBULA_TYPES = {
    "H_II": {
        "description": "Emission nebula (star-forming region)",
        "temperature_range": (8000, 12000),
        "density_range": (1e2, 1e6),  # particles/cm³
        "size_range": (1, 100),  # parsecs
        "ionization_parameter": (-3, -1),
        "dominant_lines": ["H-alpha", "H-beta", "[O III]", "[N II]"],
        "examples": ["Orion Nebula", "Eagle Nebula", "Rosette Nebula"],
    },
    "planetary": {
        "description": "Planetary nebula (evolved star envelope)",
        "temperature_range": (50000, 200000),
        "density_range": (1e3, 1e7),
        "size_range": (0.1, 5),
        "ionization_parameter": (-2, 0),
        "dominant_lines": ["[O III]", "H-alpha", "[N II]", "He II"],
        "examples": ["Ring Nebula", "Helix Nebula", "Cat's Eye Nebula"],
    },
    "supernova_remnant": {
        "description": "Supernova explosion remnant",
        "temperature_range": (1e6, 1e8),
        "density_range": (0.1, 1e3),
        "size_range": (10, 100),
        "expansion_velocity": (1000, 10000),  # km/s
        "dominant_lines": ["X-ray continuum", "[S II]", "[O III]"],
        "examples": ["Crab Nebula", "Veil Nebula", "Cassiopeia A"],
    },
    "dark": {
        "description": "Dark nebula (dust cloud)",
        "temperature_range": (10, 50),
        "density_range": (1e2, 1e5),
        "size_range": (0.1, 50),
        "extinction_Av": (1, 50),  # magnitudes
        "examples": ["Horsehead Nebula", "Coal Sack", "Barnard 68"],
    },
    "reflection": {
        "description": "Reflection nebula (dust reflecting starlight)",
        "temperature_range": (20, 100),
        "density_range": (1e1, 1e4),
        "size_range": (0.5, 20),
        "albedo": (0.1, 0.6),
        "examples": ["Pleiades", "Witch Head Nebula", "NGC 7023"],
    },
}


def create_sample_stellar_data(
    n_stars: int = 1000, spectral_classes: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Create sample stellar data with realistic distributions.

    Args:
        n_stars: Number of stars to generate
        spectral_classes: List of spectral classes to include (default: all)

    Returns:
        Dict containing stellar parameters as tensors
    """
    if spectral_classes is None:
        spectral_classes = list(STELLAR_CLASSIFICATION.keys())

    # Generate spectral classes based on realistic frequencies
    class_weights = [
        STELLAR_CLASSIFICATION[sc]["main_sequence_fraction"] for sc in spectral_classes
    ]
    class_weights = np.array(class_weights) / np.sum(class_weights)

    stellar_classes = np.random.choice(spectral_classes, size=n_stars, p=class_weights)

    # Initialize arrays
    temperatures = np.zeros(n_stars)
    masses = np.zeros(n_stars)
    luminosities = np.zeros(n_stars)
    colors_r = np.zeros(n_stars)
    colors_g = np.zeros(n_stars)
    colors_b = np.zeros(n_stars)
    lifetimes = np.zeros(n_stars)

    # Generate properties for each star
    for i, sc in enumerate(stellar_classes):
        params = STELLAR_CLASSIFICATION[sc]

        # Temperature (log-normal distribution within range)
        temp_min, temp_max = params["temperature_range"]
        temperatures[i] = np.random.uniform(temp_min, temp_max)

        # Mass (log-normal distribution)
        mass_min, mass_max = params["mass_range"]
        masses[i] = np.random.lognormal(np.log(np.sqrt(mass_min * mass_max)), 0.3)
        masses[i] = np.clip(masses[i], mass_min, mass_max)

        # Luminosity (mass-luminosity relation with scatter)
        if masses[i] < 0.43:
            lum_base = 0.23 * (masses[i] ** 2.3)
        elif masses[i] < 2:
            lum_base = masses[i] ** 4
        elif masses[i] < 20:
            lum_base = 1.5 * (masses[i] ** 3.5)
        else:
            lum_base = 32000 * masses[i]

        luminosities[i] = lum_base * np.random.lognormal(0, 0.2)

        # Color
        colors_r[i], colors_g[i], colors_b[i] = params["color"]

        # Add scatter to colors
        color_scatter = 0.1
        colors_r[i] += np.random.normal(0, color_scatter)
        colors_g[i] += np.random.normal(0, color_scatter)
        colors_b[i] += np.random.normal(0, color_scatter)

        # Lifetime
        lifetimes[i] = params["lifetime_myr"] * np.random.lognormal(0, 0.3)

    # Convert to tensors
    data = {
        "spectral_class": stellar_classes,
        "temperature": torch.tensor(temperatures, dtype=torch.float32),
        "mass": torch.tensor(masses, dtype=torch.float32),
        "luminosity": torch.tensor(luminosities, dtype=torch.float32),
        "color_r": torch.tensor(colors_r, dtype=torch.float32),
        "color_g": torch.tensor(colors_g, dtype=torch.float32),
        "color_b": torch.tensor(colors_b, dtype=torch.float32),
        "lifetime_myr": torch.tensor(lifetimes, dtype=torch.float32),
    }

    return data


def create_sample_galaxy_data(
    n_galaxies: int = 500, galaxy_types: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Create sample galaxy data with realistic morphological distributions.

    Args:
        n_galaxies: Number of galaxies to generate
        galaxy_types: List of galaxy types to include

    Returns:
        Dict containing galaxy parameters as tensors
    """
    if galaxy_types is None:
        galaxy_types = list(GALAXY_TYPES.keys())

    # Realistic galaxy type frequencies (roughly based on observations)
    type_weights = {
        "E0": 0.05,
        "E3": 0.08,
        "E7": 0.07,
        "S0": 0.15,
        "Sa": 0.10,
        "Sb": 0.15,
        "Sc": 0.20,
        "SBa": 0.05,
        "SBb": 0.08,
        "SBc": 0.12,
        "Irr": 0.15,
    }

    available_weights = [type_weights.get(gt, 0.1) for gt in galaxy_types]
    available_weights = np.array(available_weights) / np.sum(available_weights)

    galaxy_classes = np.random.choice(
        galaxy_types, size=n_galaxies, p=available_weights
    )

    # Initialize arrays
    masses = np.zeros(n_galaxies)
    star_formation_rates = np.zeros(n_galaxies)
    color_indices = np.zeros(n_galaxies)
    bulge_fractions = np.zeros(n_galaxies)
    disk_fractions = np.zeros(n_galaxies)
    ellipticities = np.zeros(n_galaxies)

    # Generate properties for each galaxy
    for i, gt in enumerate(galaxy_classes):
        params = GALAXY_TYPES[gt]

        # Mass (log-normal distribution around typical mass)
        typical_mass = params["typical_mass"]
        masses[i] = np.random.lognormal(np.log(typical_mass), 0.5)

        # Star formation rate (with mass dependence)
        base_sfr = params["star_formation_rate"]
        mass_factor = (masses[i] / typical_mass) ** 0.7
        star_formation_rates[i] = base_sfr * mass_factor * np.random.lognormal(0, 0.3)

        # Color index (with scatter)
        color_indices[i] = params["color_index"] + np.random.normal(0, 0.1)

        # Morphological parameters
        bulge_fractions[i] = params["bulge_fraction"] + np.random.normal(0, 0.1)
        disk_fractions[i] = params["disk_fraction"] + np.random.normal(0, 0.1)
        ellipticities[i] = params["ellipticity"] + np.random.normal(0, 0.1)

        # Ensure physical constraints
        bulge_fractions[i] = np.clip(bulge_fractions[i], 0, 1)
        disk_fractions[i] = np.clip(disk_fractions[i], 0, 1)
        ellipticities[i] = np.clip(ellipticities[i], 0, 0.9)

        # Normalize bulge + disk fractions
        total_frac = bulge_fractions[i] + disk_fractions[i]
        if total_frac > 1:
            bulge_fractions[i] /= total_frac
            disk_fractions[i] /= total_frac

    # Convert to tensors
    data = {
        "galaxy_type": galaxy_classes,
        "mass": torch.tensor(masses, dtype=torch.float32),
        "star_formation_rate": torch.tensor(star_formation_rates, dtype=torch.float32),
        "color_index": torch.tensor(color_indices, dtype=torch.float32),
        "bulge_fraction": torch.tensor(bulge_fractions, dtype=torch.float32),
        "disk_fraction": torch.tensor(disk_fractions, dtype=torch.float32),
        "ellipticity": torch.tensor(ellipticities, dtype=torch.float32),
    }

    return data


def create_sample_nebula_data(
    n_nebulae: int = 200, nebula_types: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Create sample nebula data with realistic physical properties.

    Args:
        n_nebulae: Number of nebulae to generate
        nebula_types: List of nebula types to include

    Returns:
        Dict containing nebula parameters as tensors
    """
    if nebula_types is None:
        nebula_types = list(NEBULA_TYPES.keys())

    # Realistic nebula type frequencies
    type_weights = {
        "H_II": 0.4,
        "planetary": 0.2,
        "supernova_remnant": 0.1,
        "dark": 0.2,
        "reflection": 0.1,
    }

    available_weights = [type_weights.get(nt, 0.1) for nt in nebula_types]
    available_weights = np.array(available_weights) / np.sum(available_weights)

    nebula_classes = np.random.choice(nebula_types, size=n_nebulae, p=available_weights)

    # Initialize arrays
    temperatures = np.zeros(n_nebulae)
    densities = np.zeros(n_nebulae)
    sizes = np.zeros(n_nebulae)

    # Generate properties for each nebula
    for i, nt in enumerate(nebula_classes):
        params = NEBULA_TYPES[nt]

        # Temperature (log-uniform in range)
        temp_min, temp_max = params["temperature_range"]
        temperatures[i] = np.random.uniform(np.log10(temp_min), np.log10(temp_max))
        temperatures[i] = 10 ** temperatures[i]

        # Density (log-uniform in range)
        dens_min, dens_max = params["density_range"]
        densities[i] = np.random.uniform(np.log10(dens_min), np.log10(dens_max))
        densities[i] = 10 ** densities[i]

        # Size (log-uniform in range)
        size_min, size_max = params["size_range"]
        sizes[i] = np.random.uniform(np.log10(size_min), np.log10(size_max))
        sizes[i] = 10 ** sizes[i]

    # Convert to tensors
    data = {
        "nebula_type": nebula_classes,
        "temperature": torch.tensor(temperatures, dtype=torch.float32),
        "density": torch.tensor(densities, dtype=torch.float32),
        "size_pc": torch.tensor(sizes, dtype=torch.float32),
    }

    return data


def create_sample_hr_diagram_data(n_stars: int = 2000) -> Dict[str, torch.Tensor]:
    """
    Create sample HR diagram data with all stellar evolution phases.

    Args:
        n_stars: Number of stars to generate

    Returns:
        Dict containing HR diagram data
    """
    # Distribution of stellar evolution phases
    phase_weights = {
        "main_sequence": 0.85,
        "red_giants": 0.08,
        "white_dwarfs": 0.05,
        "supergiants": 0.02,
    }

    phases = np.random.choice(
        list(phase_weights.keys()), size=n_stars, p=list(phase_weights.values())
    )

    # Initialize arrays
    temperatures = np.zeros(n_stars)
    luminosities = np.zeros(n_stars)
    masses = np.zeros(n_stars)
    colors = np.zeros(n_stars)

    # Generate properties for each stellar phase
    for i, phase in enumerate(phases):
        params = HR_DIAGRAM_PARAMS[phase]

        # Temperature (log-uniform in range)
        temp_min, temp_max = params["temperature_range"]
        temperatures[i] = np.random.uniform(temp_min, temp_max)

        # Luminosity (log-uniform in range)
        lum_min, lum_max = params["luminosity_range"]
        luminosities[i] = np.random.uniform(np.log10(lum_min), np.log10(lum_max))
        luminosities[i] = 10 ** luminosities[i]

        # Mass (log-uniform in range)
        mass_min, mass_max = params["mass_range"]
        masses[i] = np.random.uniform(mass_min, mass_max)

        # Color index (B-V)
        color_min, color_max = params["color_range"]
        colors[i] = np.random.uniform(color_min, color_max)

    # Convert to tensors
    data = {
        "evolution_phase": phases,
        "temperature": torch.tensor(temperatures, dtype=torch.float32),
        "luminosity": torch.tensor(luminosities, dtype=torch.float32),
        "mass": torch.tensor(masses, dtype=torch.float32),
        "color_bv": torch.tensor(colors, dtype=torch.float32),
        "log_temperature": torch.tensor(np.log10(temperatures), dtype=torch.float32),
        "log_luminosity": torch.tensor(np.log10(luminosities), dtype=torch.float32),
    }

    return data


def get_stellar_data(spectral_class: str) -> Dict[str, Any]:
    """Get stellar data for a specific spectral class."""
    return STELLAR_CLASSIFICATION.get(spectral_class, STELLAR_CLASSIFICATION["G"])


def get_galaxy_config(galaxy_type: str) -> Dict[str, Any]:
    """Get galaxy configuration for a specific type."""
    return GALAXY_TYPES.get(galaxy_type, GALAXY_TYPES["Sb"])


def validate_hr_diagram_data(data: Dict[str, torch.Tensor]) -> bool:
    """
    Validate HR diagram data for physical consistency.

    Args:
        data: HR diagram data dict

    Returns:
        True if data is valid
    """
    required_keys = ["temperature", "luminosity"]

    # Check required keys
    for key in required_keys:
        if key not in data:
            logger.error(f"Missing required key: {key}")
            return False

    # Check data ranges
    temp = data["temperature"]
    lum = data["luminosity"]

    if torch.any(temp < 1000) or torch.any(temp > 100000):
        logger.warning("Temperature values outside expected range (1000-100000 K)")

    if torch.any(lum < 1e-6) or torch.any(lum > 1e7):
        logger.warning("Luminosity values outside expected range (1e-6 - 1e7 solar)")

    # Check for NaN or infinite values
    if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
        logger.error("Invalid temperature values (NaN or Inf)")
        return False

    if torch.any(torch.isnan(lum)) or torch.any(torch.isinf(lum)):
        logger.error("Invalid luminosity values (NaN or Inf)")
        return False

    logger.info("HR diagram data validation passed")
    return True
