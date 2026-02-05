"""
Physics Calculation Utilities
=============================

Comprehensive physics calculations for astronomical objects and systems.
Includes orbital mechanics, gravitational physics, and dark matter modeling.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
G = 6.674e-11  # m³/kg/s² (gravitational constant)
SOLAR_MASS = 1.989e30  # kg
EARTH_MASS = 5.972e24  # kg
JUPITER_MASS = 1.898e27  # kg
AU = 1.496e11  # meters (astronomical unit)
PC_TO_M = 3.086e16  # meters per parsec
C_LIGHT = 2.998e8  # m/s
YEAR_TO_S = 3.156e7  # seconds per year


def calculate_gravitational_force(mass1: float, mass2: float, distance: float) -> float:
    """
    Calculate gravitational force between two masses.

    Args:
        mass1: Mass of first object in kg
        mass2: Mass of second object in kg
        distance: Distance between objects in meters

    Returns:
        Gravitational force in Newtons
    """
    return G * mass1 * mass2 / (distance**2)


def calculate_escape_velocity(mass: float, radius: float) -> float:
    """
    Calculate escape velocity from an object.

    Args:
        mass: Object mass in kg
        radius: Object radius in meters

    Returns:
        Escape velocity in m/s
    """
    return np.sqrt(2 * G * mass / radius)


def calculate_orbital_period(semi_major_axis: float, central_mass: float) -> float:
    """
    Calculate orbital period using Kepler's third law.

    Args:
        semi_major_axis: Semi-major axis in meters
        central_mass: Central mass in kg

    Returns:
        Orbital period in seconds
    """
    return 2 * np.pi * np.sqrt(semi_major_axis**3 / (G * central_mass))


def calculate_orbital_velocity(semi_major_axis: float, central_mass: float) -> float:
    """
    Calculate circular orbital velocity.

    Args:
        semi_major_axis: Orbital radius in meters
        central_mass: Central mass in kg

    Returns:
        Orbital velocity in m/s
    """
    return np.sqrt(G * central_mass / semi_major_axis)


def calculate_orbital_elements(
    position: np.ndarray, velocity: np.ndarray, central_mass: float
) -> Dict[str, float]:
    """
    Calculate orbital elements from position and velocity vectors.

    Args:
        position: Position vector [x, y, z] in meters
        velocity: Velocity vector [vx, vy, vz] in m/s
        central_mass: Central mass in kg

    Returns:
        Dict containing orbital elements
    """
    r = np.linalg.norm(position)
    v = np.linalg.norm(velocity)

    # Specific orbital energy
    energy = v**2 / 2 - G * central_mass / r

    # Semi-major axis
    a = -G * central_mass / (2 * energy)

    # Angular momentum vector
    h_vec = np.cross(position, velocity)
    h = np.linalg.norm(h_vec)

    # Eccentricity vector
    mu = G * central_mass
    e_vec = np.cross(velocity, h_vec) / mu - position / r
    e = np.linalg.norm(e_vec)

    # Inclination
    i = np.arccos(h_vec[2] / h)

    # Longitude of ascending node
    n_vec = np.cross([0, 0, 1], h_vec)
    if np.linalg.norm(n_vec) > 0:
        n_vec = n_vec / np.linalg.norm(n_vec)
        omega = np.arccos(n_vec[0])
        if n_vec[1] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0

    # Argument of periapsis
    if e > 0:
        if np.linalg.norm(n_vec) > 0:
            w = np.arccos(np.dot(n_vec, e_vec) / e)
            if e_vec[2] < 0:
                w = 2 * np.pi - w
        else:
            w = np.arccos(e_vec[0] / e)
            if e_vec[1] < 0:
                w = 2 * np.pi - w
    else:
        w = 0

    # True anomaly
    if e > 0:
        nu = np.arccos(np.dot(e_vec, position) / (e * r))
        if np.dot(position, velocity) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0

    return {
        "semi_major_axis": a,
        "eccentricity": e,
        "inclination": np.degrees(i),
        "longitude_ascending_node": np.degrees(omega),
        "argument_of_periapsis": np.degrees(w),
        "true_anomaly": np.degrees(nu),
        "period": calculate_orbital_period(a, central_mass),
        "energy": energy,
        "angular_momentum": h,
    }


def calculate_orbital_position(
    semi_major_axis: float,
    eccentricity: float,
    true_anomaly: float,
    inclination: float = 0,
    longitude_ascending_node: float = 0,
    argument_of_periapsis: float = 0,
) -> np.ndarray:
    """
    Calculate position from orbital elements.

    Args:
        semi_major_axis: Semi-major axis in meters
        eccentricity: Orbital eccentricity
        true_anomaly: True anomaly in degrees
        inclination: Inclination in degrees
        longitude_ascending_node: Longitude of ascending node in degrees
        argument_of_periapsis: Argument of periapsis in degrees

    Returns:
        Position vector [x, y, z] in meters
    """
    # Convert angles to radians
    nu = np.radians(true_anomaly)
    i = np.radians(inclination)
    omega = np.radians(longitude_ascending_node)
    w = np.radians(argument_of_periapsis)

    # Distance from focus
    r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(nu))

    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = 0

    # Rotation matrices
    # Argument of periapsis rotation
    R_w = np.array([[np.cos(w), -np.sin(w), 0], [np.sin(w), np.cos(w), 0], [0, 0, 1]])

    # Inclination rotation
    R_i = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])

    # Longitude of ascending node rotation
    R_omega = np.array(
        [
            [np.cos(omega), -np.sin(omega), 0],
            [np.sin(omega), np.cos(omega), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation
    R = R_omega @ R_i @ R_w

    # Transform to inertial frame
    position_orb = np.array([x_orb, y_orb, z_orb])
    position = R @ position_orb

    return position


def calculate_hill_sphere(
    primary_mass: float, secondary_mass: float, orbital_distance: float
) -> float:
    """
    Calculate Hill sphere radius for gravitational influence.

    Args:
        primary_mass: Primary mass in kg
        secondary_mass: Secondary mass in kg
        orbital_distance: Orbital separation in meters

    Returns:
        Hill sphere radius in meters
    """
    mass_ratio = secondary_mass / primary_mass
    hill_radius = orbital_distance * (mass_ratio / 3) ** (1 / 3)

    return hill_radius


def calculate_roche_limit(
    primary_mass: float, primary_radius: float, secondary_density: float
) -> float:
    """
    Calculate Roche limit for tidal disruption.

    Args:
        primary_mass: Primary mass in kg
        primary_radius: Primary radius in meters
        secondary_density: Secondary density in kg/m³

    Returns:
        Roche limit distance in meters
    """
    primary_density = primary_mass / ((4 / 3) * np.pi * primary_radius**3)

    # Rigid body Roche limit
    roche_limit = (
        2.44 * primary_radius * (primary_density / secondary_density) ** (1 / 3)
    )

    return roche_limit


def calculate_tidal_force(
    primary_mass: float, distance: float, object_size: float
) -> float:
    """
    Calculate tidal force on extended object.

    Args:
        primary_mass: Primary mass in kg
        distance: Distance to primary in meters
        object_size: Size of object experiencing tidal force in meters

    Returns:
        Tidal acceleration in m/s²
    """
    # Tidal acceleration = 2 * G * M * r / d³
    tidal_accel = 2 * G * primary_mass * object_size / (distance**3)

    return tidal_accel


def calculate_hohmann_transfer(
    r1: float, r2: float, central_mass: float
) -> Dict[str, float]:
    """
    Calculate Hohmann transfer orbit parameters.

    Args:
        r1: Initial orbital radius in meters
        r2: Final orbital radius in meters
        central_mass: Central mass in kg

    Returns:
        Dict containing transfer orbit parameters
    """
    # Transfer orbit semi-major axis
    a_transfer = (r1 + r2) / 2

    # Velocities
    v1_circular = np.sqrt(G * central_mass / r1)
    v2_circular = np.sqrt(G * central_mass / r2)

    v1_transfer = np.sqrt(G * central_mass * (2 / r1 - 1 / a_transfer))
    v2_transfer = np.sqrt(G * central_mass * (2 / r2 - 1 / a_transfer))

    # Delta-v requirements
    delta_v1 = abs(v1_transfer - v1_circular)
    delta_v2 = abs(v2_circular - v2_transfer)
    delta_v_total = delta_v1 + delta_v2

    # Transfer time
    transfer_period = 2 * np.pi * np.sqrt(a_transfer**3 / (G * central_mass))
    transfer_time = transfer_period / 2

    return {
        "transfer_semi_major_axis": a_transfer,
        "delta_v1": delta_v1,
        "delta_v2": delta_v2,
        "total_delta_v": delta_v_total,
        "transfer_time": transfer_time,
        "transfer_period": transfer_period,
    }


def calculate_gravitational_potential(mass: float, radius: float) -> float:
    """
    Calculate gravitational potential at surface.

    Args:
        mass: Object mass in kg
        radius: Object radius in meters

    Returns:
        Gravitational potential in J/kg
    """
    return -G * mass / radius


def calculate_virial_velocity(mass: float, radius: float) -> float:
    """
    Calculate virial velocity for gravitational system.

    Args:
        mass: System mass in kg
        radius: System radius in meters

    Returns:
        Virial velocity in m/s
    """
    return np.sqrt(G * mass / radius)


def calculate_dark_matter_profile(
    radius_kpc: np.ndarray,
    mass_200: float,
    concentration: float = 10.0,
    profile_type: str = "NFW",
) -> np.ndarray:
    """
    Calculate dark matter halo density profile.

    Args:
        radius_kpc: Radii in kiloparsecs
        mass_200: Virial mass in solar masses
        concentration: Halo concentration parameter
        profile_type: "NFW", "Einasto", or "Burkert"

    Returns:
        Density profile in Msun/pc³
    """
    # Convert to appropriate units
    r = radius_kpc * 1000  # Convert to pc
    M_200 = mass_200  # Already in solar masses

    # Critical density of universe (approximate)
    rho_crit = 2.78e11 * 0.3  # Msun/Mpc³ * Omega_m
    rho_crit_pc = rho_crit / 1e18  # Convert to Msun/pc³

    # Virial radius
    r_200 = (3 * M_200 / (4 * np.pi * 200 * rho_crit_pc)) ** (1 / 3) / 1000  # kpc
    r_200_pc = r_200 * 1000  # pc

    # Scale radius
    r_s = r_200_pc / concentration

    if profile_type == "NFW":
        # Navarro-Frenk-White profile
        x = r / r_s

        # Characteristic density
        delta_c = (
            200
            / 3
            * concentration**3
            / (np.log(1 + concentration) - concentration / (1 + concentration))
        )
        rho_s = delta_c * rho_crit_pc

        # NFW profile
        density = rho_s / (x * (1 + x) ** 2)

    elif profile_type == "Einasto":
        # Einasto profile
        alpha = 0.17  # Typical value
        r_e = r_s  # Use scale radius as effective radius

        # Normalization (approximate)
        rho_e = M_200 / (4 * np.pi * r_e**3 * alpha * np.exp(2 / alpha))

        # Einasto profile
        density = rho_e * np.exp(-2 / alpha * ((r / r_e) ** alpha - 1))

    elif profile_type == "Burkert":
        # Burkert profile (cored)
        r_b = r_s
        rho_b = M_200 / (4 * np.pi * r_b**3)  # Approximate normalization

        # Burkert profile
        density = rho_b / ((1 + r / r_b) * (1 + (r / r_b) ** 2))

    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

    return density


def propagate_orbit(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    central_mass: float,
    time_steps: np.ndarray,
    method: str = "rk4",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate orbital motion using numerical integration.

    Args:
        initial_position: Initial position vector [x, y, z] in meters
        initial_velocity: Initial velocity vector [vx, vy, vz] in m/s
        central_mass: Central mass in kg
        time_steps: Array of time points in seconds
        method: Integration method ("euler", "rk4")

    Returns:
        Tuple of (positions, velocities) arrays
    """
    n_steps = len(time_steps)
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))

    # Initial conditions
    positions[0] = initial_position
    velocities[0] = initial_velocity

    for i in range(1, n_steps):
        dt = time_steps[i] - time_steps[i - 1]

        if method == "euler":
            # Simple Euler integration
            r = np.linalg.norm(positions[i - 1])
            acceleration = -G * central_mass * positions[i - 1] / r**3

            velocities[i] = velocities[i - 1] + acceleration * dt
            positions[i] = positions[i - 1] + velocities[i - 1] * dt

        elif method == "rk4":
            # Runge-Kutta 4th order
            def derivatives(pos, vel):
                r = np.linalg.norm(pos)
                acc = -G * central_mass * pos / r**3
                return vel, acc

            # RK4 integration
            pos = positions[i - 1]
            vel = velocities[i - 1]

            k1_pos, k1_vel = derivatives(pos, vel)
            k2_pos, k2_vel = derivatives(pos + k1_pos * dt / 2, vel + k1_vel * dt / 2)
            k3_pos, k3_vel = derivatives(pos + k2_pos * dt / 2, vel + k2_vel * dt / 2)
            k4_pos, k4_vel = derivatives(pos + k3_pos * dt, vel + k3_vel * dt)

            positions[i] = pos + dt / 6 * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
            velocities[i] = vel + dt / 6 * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)

    return positions, velocities


def create_poliastro_orbit(
    semi_major_axis_au: float,
    eccentricity: float,
    inclination_deg: float = 0,
    central_body: str = "Sun",
) -> Dict[str, Any]:
    """
    Create orbit parameters compatible with poliastro library.

    Args:
        semi_major_axis_au: Semi-major axis in AU
        eccentricity: Orbital eccentricity
        inclination_deg: Inclination in degrees
        central_body: Central body name

    Returns:
        Dict containing orbit parameters
    """
    # Convert to SI units
    a_m = semi_major_axis_au * AU

    # Central body masses
    body_masses = {
        "Sun": 1.989e30,
        "Earth": 5.972e24,
        "Jupiter": 1.898e27,
        "Mars": 6.39e23,
    }

    central_mass = body_masses.get(central_body, 1.989e30)

    # Calculate orbital period
    period = calculate_orbital_period(a_m, central_mass)

    # Calculate mean motion
    mean_motion = 2 * np.pi / period

    return {
        "semi_major_axis_au": semi_major_axis_au,
        "semi_major_axis_m": a_m,
        "eccentricity": eccentricity,
        "inclination_deg": inclination_deg,
        "inclination_rad": np.radians(inclination_deg),
        "central_mass_kg": central_mass,
        "period_s": period,
        "period_years": period / YEAR_TO_S,
        "mean_motion_rad_s": mean_motion,
        "central_body": central_body,
    }
