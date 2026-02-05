"""
Post-Processing Calculation Utilities
====================================

Comprehensive post-processing calculations for astronomical image rendering.
Includes camera effects, atmospheric simulation, and scientific processing.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def calculate_exposure_value(
    iso: float, aperture: float, shutter_speed: float
) -> float:
    """
    Calculate exposure value (EV) from camera settings.

    Args:
        iso: ISO sensitivity
        aperture: F-number (f-stop)
        shutter_speed: Shutter speed in seconds

    Returns:
        Exposure value
    """
    ev = np.log2((aperture**2) / shutter_speed) + np.log2(iso / 100)
    return ev


def calculate_depth_of_field(
    focal_length_mm: float,
    aperture: float,
    subject_distance_m: float,
    circle_of_confusion_mm: float = 0.03,
) -> Dict[str, float]:
    """
    Calculate depth of field parameters.

    Args:
        focal_length_mm: Focal length in millimeters
        aperture: F-number
        subject_distance_m: Distance to subject in meters
        circle_of_confusion_mm: Circle of confusion in mm

    Returns:
        Dict containing DOF parameters
    """
    f = focal_length_mm / 1000  # Convert to meters
    d = subject_distance_m
    c = circle_of_confusion_mm / 1000  # Convert to meters

    # Hyperfocal distance
    H = f**2 / (aperture * c) + f

    # Near and far distances
    if d > H:
        # Subject beyond hyperfocal distance
        near = H * d / (H + d - f)
        far = float("inf")
    else:
        # Subject within hyperfocal distance
        near = H * d / (H + d - f)
        far = H * d / (H - d + f)

    # Total depth of field
    total_dof = far - near if far != float("inf") else float("inf")

    return {
        "hyperfocal_distance_m": H,
        "near_distance_m": near,
        "far_distance_m": far,
        "total_depth_of_field_m": total_dof,
        "focal_length_m": f,
        "aperture": aperture,
        "subject_distance_m": d,
    }


def calculate_bloom_intensity(
    luminance: float, threshold: float = 1.0, intensity: float = 1.0
) -> float:
    """
    Calculate bloom effect intensity for bright objects.

    Args:
        luminance: Object luminance (relative)
        threshold: Bloom threshold
        intensity: Bloom intensity multiplier

    Returns:
        Bloom intensity factor
    """
    if luminance > threshold:
        bloom_factor = intensity * (luminance - threshold) / threshold
        return min(bloom_factor, 10.0)  # Cap bloom intensity
    return 0.0


def calculate_lens_distortion(
    x: float, y: float, distortion_coefficients: List[float]
) -> Tuple[float, float]:
    """
    Calculate lens distortion correction.

    Args:
        x: Normalized x coordinate (-1 to 1)
        y: Normalized y coordinate (-1 to 1)
        distortion_coefficients: [k1, k2, k3, p1, p2] distortion coefficients

    Returns:
        Corrected (x, y) coordinates
    """
    if len(distortion_coefficients) < 5:
        distortion_coefficients.extend([0.0] * (5 - len(distortion_coefficients)))

    k1, k2, k3, p1, p2 = distortion_coefficients

    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2**3

    # Radial distortion
    radial_factor = 1 + k1 * r2 + k2 * r4 + k3 * r6

    # Tangential distortion
    dx_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    dy_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    # Apply corrections
    x_corrected = x * radial_factor + dx_tangential
    y_corrected = y * radial_factor + dy_tangential

    return x_corrected, y_corrected


def calculate_chromatic_aberration(
    wavelength_nm: float,
    reference_wavelength_nm: float = 550.0,
    aberration_strength: float = 0.01,
) -> float:
    """
    Calculate chromatic aberration offset.

    Args:
        wavelength_nm: Light wavelength in nanometers
        reference_wavelength_nm: Reference wavelength (typically green)
        aberration_strength: Aberration strength factor

    Returns:
        Radial offset factor
    """
    # Simple dispersion model
    dispersion = (reference_wavelength_nm - wavelength_nm) / reference_wavelength_nm
    aberration_offset = aberration_strength * dispersion

    return aberration_offset


def calculate_vignette_intensity(
    x: float,
    y: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    vignette_strength: float = 0.3,
) -> float:
    """
    Calculate vignetting intensity.

    Args:
        x: Normalized x coordinate
        y: Normalized y coordinate
        center_x: Vignette center x
        center_y: Vignette center y
        vignette_strength: Vignetting strength

    Returns:
        Vignetting factor (0-1)
    """
    # Distance from center
    dx = x - center_x
    dy = y - center_y
    distance = np.sqrt(dx**2 + dy**2)

    # Vignetting falloff (cosine-like)
    vignette_factor = 1.0 - vignette_strength * distance**2

    return max(0.0, min(1.0, vignette_factor))


def calculate_motion_blur_angle(
    velocity_x: float, velocity_y: float, exposure_time: float, pixel_scale: float
) -> Tuple[float, float]:
    """
    Calculate motion blur parameters.

    Args:
        velocity_x: X velocity in arcsec/s
        velocity_y: Y velocity in arcsec/s
        exposure_time: Exposure time in seconds
        pixel_scale: Pixel scale in arcsec/pixel

    Returns:
        Motion blur (angle_degrees, length_pixels)
    """
    # Total displacement during exposure
    displacement_x = velocity_x * exposure_time
    displacement_y = velocity_y * exposure_time

    # Blur length in pixels
    blur_length_pixels = np.sqrt(displacement_x**2 + displacement_y**2) / pixel_scale

    # Blur angle
    blur_angle_rad = np.arctan2(displacement_y, displacement_x)
    blur_angle_deg = np.degrees(blur_angle_rad)

    return blur_angle_deg, blur_length_pixels


def calculate_color_temperature_to_rgb(
    temperature_k: float,
) -> Tuple[float, float, float]:
    """
    Convert color temperature to RGB values.

    Args:
        temperature_k: Color temperature in Kelvin

    Returns:
        RGB tuple (0-1 range)
    """
    # Clamp temperature to reasonable range
    temp = np.clip(temperature_k, 1000, 40000)
    temp = temp / 100.0

    # Calculate red component
    if temp <= 66:
        red = 255
    else:
        red = temp - 60
        red = 329.698727446 * (red**-0.1332047592)
        red = np.clip(red, 0, 255)

    # Calculate green component
    if temp <= 66:
        green = temp
        green = 99.4708025861 * np.log(green) - 161.1195681661
        green = np.clip(green, 0, 255)
    else:
        green = temp - 60
        green = 288.1221695283 * (green**-0.0755148492)
        green = np.clip(green, 0, 255)

    # Calculate blue component
    if temp >= 66:
        blue = 255
    elif temp <= 19:
        blue = 0
    else:
        blue = temp - 10
        blue = 138.5177312231 * np.log(blue) - 305.0447927307
        blue = np.clip(blue, 0, 255)

    # Normalize to 0-1 range
    return (red / 255.0, green / 255.0, blue / 255.0)


def calculate_saturation_multiplier(
    luminance: float, saturation_curve: str = "natural"
) -> float:
    """
    Calculate saturation adjustment based on luminance.

    Args:
        luminance: Pixel luminance (0-1)
        saturation_curve: Saturation curve type

    Returns:
        Saturation multiplier
    """
    if saturation_curve == "natural":
        # Natural saturation - higher for mid-tones
        saturation = 1.0 + 0.3 * np.sin(np.pi * luminance)
    elif saturation_curve == "vibrant":
        # Enhanced saturation
        saturation = 1.0 + 0.5 * (1 - abs(2 * luminance - 1))
    elif saturation_curve == "muted":
        # Reduced saturation
        saturation = 0.7 + 0.3 * luminance
    else:
        saturation = 1.0

    return max(0.0, saturation)


def calculate_contrast_curve(luminance: float, contrast_type: str = "film") -> float:
    """
    Apply contrast curve to luminance.

    Args:
        luminance: Input luminance (0-1)
        contrast_type: Type of contrast curve

    Returns:
        Output luminance after contrast adjustment
    """
    if contrast_type == "film":
        # Film-like S-curve
        return 1 / (1 + np.exp(-10 * (luminance - 0.5)))
    elif contrast_type == "digital":
        # Digital camera curve
        gamma = 2.2
        return luminance ** (1 / gamma)
    elif contrast_type == "linear":
        # Linear response
        return luminance
    elif contrast_type == "log":
        # Logarithmic curve for high dynamic range
        return np.log10(1 + 9 * luminance) if luminance > 0 else 0
    else:
        return luminance


def calculate_film_grain_intensity(
    iso: float, luminance: float, grain_type: str = "digital"
) -> float:
    """
    Calculate film grain/noise intensity.

    Args:
        iso: ISO sensitivity
        luminance: Pixel luminance (0-1)
        grain_type: Type of grain/noise

    Returns:
        Grain intensity
    """
    if grain_type == "digital":
        # Digital noise model
        base_noise = np.sqrt(iso / 100) * 0.01
        # Noise is higher in shadows
        luminance_factor = 1 / (luminance + 0.1)
        noise_intensity = base_noise * luminance_factor
    elif grain_type == "film":
        # Film grain model
        base_grain = np.sqrt(iso / 100) * 0.005
        # Grain is more uniform across luminance
        noise_intensity = base_grain * (1 + 0.5 * (1 - luminance))
    else:
        noise_intensity = 0.0

    return min(noise_intensity, 0.1)  # Cap noise intensity


def apply_astrophotography_processing(
    image_data: np.ndarray, processing_params: Dict[str, Any]
) -> np.ndarray:
    """
    Apply astrophotography-specific processing.

    Args:
        image_data: Input image array
        processing_params: Processing parameters

    Returns:
        Processed image array
    """
    processed = image_data.copy()

    # Dark frame subtraction
    if "dark_frame" in processing_params:
        dark = processing_params["dark_frame"]
        processed = np.maximum(processed - dark, 0)

    # Flat field correction
    if "flat_frame" in processing_params:
        flat = processing_params["flat_frame"]
        flat_normalized = flat / np.mean(flat)
        processed = processed / flat_normalized

    # Bias subtraction
    if "bias_level" in processing_params:
        bias = processing_params["bias_level"]
        processed = np.maximum(processed - bias, 0)

    # Gradient removal (simple)
    if processing_params.get("remove_gradient", False):
        # Fit 2D polynomial to background
        from scipy import ndimage

        background = ndimage.gaussian_filter(processed, sigma=50)
        processed = processed - background + np.median(background)

    # Star enhancement
    if processing_params.get("enhance_stars", False):
        # Simple unsharp masking
        from scipy import ndimage

        blurred = ndimage.gaussian_filter(processed, sigma=2)
        mask = processed - blurred
        enhanced = processed + 0.5 * mask
        processed = np.maximum(enhanced, processed)

    return processed


def calculate_signal_to_noise_ratio(
    signal: float,
    read_noise: float,
    dark_current: float,
    exposure_time: float,
    quantum_efficiency: float = 0.9,
) -> float:
    """
    Calculate signal-to-noise ratio for astronomical imaging.

    Args:
        signal: Signal in electrons
        read_noise: Read noise in electrons
        dark_current: Dark current in electrons/s
        exposure_time: Exposure time in seconds
        quantum_efficiency: Quantum efficiency of detector

    Returns:
        Signal-to-noise ratio
    """
    # Total signal
    total_signal = signal * quantum_efficiency

    # Noise sources
    photon_noise = np.sqrt(total_signal)  # Shot noise
    dark_noise = np.sqrt(dark_current * exposure_time)
    total_noise = np.sqrt(photon_noise**2 + read_noise**2 + dark_noise**2)

    # SNR
    snr = total_signal / total_noise if total_noise > 0 else 0

    return snr


def get_scientific_preset() -> Dict[str, Any]:
    """Get preset for scientific visualization."""
    return {
        "contrast_type": "linear",
        "saturation_curve": "muted",
        "color_temperature": 6500,
        "vignette_strength": 0.0,
        "bloom_threshold": 10.0,
        "bloom_intensity": 0.1,
        "grain_type": None,
        "gamma": 1.0,
        "background_color": (0.0, 0.0, 0.0),
        "use_linear_workflow": True,
    }


def get_cinematic_preset() -> Dict[str, Any]:
    """Get preset for cinematic visualization."""
    return {
        "contrast_type": "film",
        "saturation_curve": "vibrant",
        "color_temperature": 5600,
        "vignette_strength": 0.3,
        "bloom_threshold": 1.0,
        "bloom_intensity": 2.0,
        "grain_type": "film",
        "gamma": 2.2,
        "background_color": (0.02, 0.02, 0.05),
        "use_linear_workflow": False,
    }


def get_dramatic_preset() -> Dict[str, Any]:
    """Get preset for dramatic visualization."""
    return {
        "contrast_type": "film",
        "saturation_curve": "vibrant",
        "color_temperature": 4800,
        "vignette_strength": 0.5,
        "bloom_threshold": 0.5,
        "bloom_intensity": 3.0,
        "grain_type": "digital",
        "gamma": 1.8,
        "background_color": (0.05, 0.02, 0.02),
        "use_linear_workflow": False,
    }


def get_dreamy_preset() -> Dict[str, Any]:
    """Get preset for dreamy/soft visualization."""
    return {
        "contrast_type": "digital",
        "saturation_curve": "natural",
        "color_temperature": 7200,
        "vignette_strength": 0.2,
        "bloom_threshold": 0.3,
        "bloom_intensity": 4.0,
        "grain_type": None,
        "gamma": 2.4,
        "background_color": (0.02, 0.03, 0.05),
        "use_linear_workflow": False,
        "soft_glow": True,
    }
