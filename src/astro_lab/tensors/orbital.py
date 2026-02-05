"""
Orbital TensorDict for orbital mechanics calculations.

TensorDict for orbital elements and orbital mechanics calculations
with proper Keplerian mechanics and orbital propagation using real astronomical data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .base import AstroTensorDict
from .mixins import ValidationMixin


class OrbitTensorDict(AstroTensorDict, ValidationMixin):
    """
    TensorDict for orbital elements and mechanics.

    Handles real orbital data for asteroids, comets, and exoplanets
    with proper Keplerian orbital mechanics.

    Structure:
    {
        "elements": Tensor[N, 6],  # a, e, i, Omega, omega, M
        "epoch": Tensor[N],        # Epoch of orbital elements (JD)
        "meta": {
            "frame": str,          # Reference frame (e.g., "ecliptic", "equatorial")
            "units": Dict[str, str],
            "central_body": str,   # Central body (e.g., "Sun", "Earth")
        }
    }
    """

    def __init__(
        self,
        elements: torch.Tensor,
        epoch: Optional[torch.Tensor] = None,
        frame: str = "ecliptic",
        central_body: str = "Sun",
        **kwargs,
    ):
        """
        Initialize OrbitTensorDict with real orbital elements.

        Args:
            elements: [N, 6] Tensor with orbital elements [a, e, i, Omega, omega, M]
                     where:
                     - a: semi-major axis (AU)
                     - e: eccentricity (dimensionless)
                     - i: inclination (degrees)
                     - Omega: longitude of ascending node (degrees)
                     - omega: argument of periapsis (degrees)
                     - M: mean anomaly at epoch (degrees)
            epoch: [N] Epoch of orbital elements (Julian Date, optional)
            frame: Reference frame for the orbital elements
            central_body: Central body for the orbits
        """
        if elements.shape[-1] != 6:
            raise ValueError(
                f"Orbital elements must have shape [..., 6], got {elements.shape}"
            )

        n_objects = elements.shape[0]

        if epoch is None:
            # Default to J2000.0 epoch
            epoch = torch.full((n_objects,), 2451545.0)  # J2000.0 in Julian Days

        data = {
            "elements": elements,
            "epoch": epoch,
            "meta": {
                "frame": frame,
                "central_body": central_body,
                "units": {
                    "a": "au",
                    "e": "dimensionless",
                    "i": "degrees",
                    "Omega": "degrees",
                    "omega": "degrees",
                    "M": "degrees",
                    "epoch": "julian_days",
                },
            },
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    def extract_features(
        self, feature_types: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract orbital features from real orbital data.

        Args:
            feature_types: Types of features to extract
                         ('orbital', 'dynamics', 'classification', 'temporal')
            **kwargs: Additional extraction parameters

        Returns:
            Dictionary of extracted orbital features
        """
        # Get base features
        features = super().extract_features(feature_types, **kwargs)

        # Add orbital-specific computed features
        if feature_types is None or "orbital" in feature_types:
            # Basic orbital elements
            features["semi_major_axis"] = self.semi_major_axis
            features["eccentricity"] = self.eccentricity
            features["inclination"] = self.inclination
            features["longitude_ascending_node"] = self.longitude_of_ascending_node
            features["argument_periapsis"] = self.argument_of_periapsis
            features["mean_anomaly"] = self.mean_anomaly

        if feature_types is None or "dynamics" in feature_types:
            # Dynamical features based on real orbital mechanics
            features["orbital_period"] = self.compute_period()
            features["aphelion_distance"] = self.semi_major_axis * (
                1 + self.eccentricity
            )
            features["perihelion_distance"] = self.semi_major_axis * (
                1 - self.eccentricity
            )

            # Specific orbital energy (negative for bound orbits)
            features["orbital_energy"] = -1.0 / (2 * self.semi_major_axis)

            # Specific angular momentum
            features["angular_momentum"] = torch.sqrt(
                self.semi_major_axis * (1 - self.eccentricity**2)
            )

        if feature_types is None or "classification" in feature_types:
            # Orbital family classification based on real criteria
            a = self.semi_major_axis
            e = self.eccentricity
            i = self.inclination

            # Asteroid belt classification (real boundaries)
            features["is_main_belt"] = ((a >= 2.06) & (a <= 3.27) & (e < 0.3)).float()
            features["is_near_earth"] = (a < 1.3).float()
            features["is_jupiter_trojan"] = ((a >= 4.6) & (a <= 5.5)).float()
            features["is_kuiper_belt"] = (a >= 30.0).float()

            # Eccentricity classification
            features["is_circular"] = (e < 0.1).float()
            features["is_elliptical"] = ((e >= 0.1) & (e < 0.9)).float()
            features["is_highly_eccentric"] = (e >= 0.9).float()
            features["is_parabolic"] = ((e >= 0.99) & (e < 1.01)).float()
            features["is_hyperbolic"] = (e > 1.01).float()

            # Inclination classification
            features["is_low_inclination"] = (i < 10.0).float()
            features["is_moderate_inclination"] = ((i >= 10.0) & (i < 30.0)).float()
            features["is_high_inclination"] = (i >= 30.0).float()

        if feature_types is None or "temporal" in feature_types:
            # Temporal features based on epoch
            features["epoch_jd"] = self["epoch"]

            # Years since J2000.0
            features["years_since_j2000"] = (self["epoch"] - 2451545.0) / 365.25

        return features

    @property
    def semi_major_axis(self) -> torch.Tensor:
        """Semi-major axis in AU."""
        return self["elements"][..., 0]

    @property
    def eccentricity(self) -> torch.Tensor:
        """Eccentricity (dimensionless)."""
        return self["elements"][..., 1]

    @property
    def inclination(self) -> torch.Tensor:
        """Inclination in degrees."""
        return self["elements"][..., 2]

    @property
    def longitude_of_ascending_node(self) -> torch.Tensor:
        """Longitude of ascending node in degrees."""
        return self["elements"][..., 3]

    @property
    def argument_of_periapsis(self) -> torch.Tensor:
        """Argument of periapsis in degrees."""
        return self["elements"][..., 4]

    @property
    def mean_anomaly(self) -> torch.Tensor:
        """Mean anomaly in degrees."""
        return self["elements"][..., 5]

    def compute_period(self) -> torch.Tensor:
        """
        Calculate orbital period using Kepler's Third Law.

        Returns:
            Orbital period in years
        """
        # Kepler's Third Law: P² = a³ (when a is in AU and P is in years)
        a_au = self.semi_major_axis
        return torch.sqrt(a_au**3)

    def compute_mean_motion(self) -> torch.Tensor:
        """
        Calculate mean motion (degrees per day).

        Returns:
            Mean motion in degrees/day
        """
        period_days = self.compute_period() * 365.25
        return 360.0 / period_days

    def solve_kepler_equation(self, mean_anomaly_deg: torch.Tensor) -> torch.Tensor:
        """
        Solve Kepler's equation for eccentric anomaly using Newton-Raphson method.

        Args:
            mean_anomaly_deg: Mean anomaly in degrees

        Returns:
            Eccentric anomaly in degrees
        """
        M = torch.deg2rad(mean_anomaly_deg)
        e = self.eccentricity

        # Initial guess
        E = M.clone()

        # Newton-Raphson iteration
        for _ in range(10):  # Usually converges in 3-5 iterations
            f = E - e * torch.sin(E) - M
            fp = 1 - e * torch.cos(E)
            E = E - f / fp

            # Check convergence
            if torch.max(torch.abs(f)) < 1e-12:
                break

        return torch.rad2deg(E)

    def to_cartesian(self, time_jd: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert orbital elements to Cartesian coordinates using proper orbital mechanics.

        Args:
            time_jd: Time in Julian Days for position calculation (optional)

        Returns:
            [N, 6] Tensor with [x, y, z, vx, vy, vz] in appropriate units
        """
        if time_jd is None:
            time_jd = self["epoch"]

        # Calculate mean anomaly at the given time
        n = self.compute_mean_motion()  # degrees/day
        dt = time_jd - self["epoch"]  # days since epoch
        M = self.mean_anomaly + n * dt
        M = torch.fmod(M, 360.0)  # Normalize to [0, 360)

        # Solve Kepler's equation for eccentric anomaly
        E = self.solve_kepler_equation(M)
        E_rad = torch.deg2rad(E)

        # Calculate true anomaly
        e = self.eccentricity
        f = 2 * torch.atan2(
            torch.sqrt(1 + e) * torch.sin(E_rad / 2),
            torch.sqrt(1 - e) * torch.cos(E_rad / 2),
        )

        # Calculate distance
        a = self.semi_major_axis
        r = a * (1 - e * torch.cos(E_rad))

        # Position in orbital plane
        x_orb = r * torch.cos(f)
        y_orb = r * torch.sin(f)
        z_orb = torch.zeros_like(x_orb)

        # Velocity in orbital plane (using proper orbital mechanics)
        mu = 1.0  # Standard gravitational parameter (in AU³/year²)
        h = torch.sqrt(mu * a * (1 - e**2))  # Specific angular momentum

        vx_orb = -(mu / h) * torch.sin(f)
        vy_orb = (mu / h) * (e + torch.cos(f))
        vz_orb = torch.zeros_like(vx_orb)

        # Convert angles to radians for rotation matrices
        i_rad = torch.deg2rad(self.inclination)
        Omega_rad = torch.deg2rad(self.longitude_of_ascending_node)
        omega_rad = torch.deg2rad(self.argument_of_periapsis)

        # Rotation matrices for coordinate transformation
        cos_Omega = torch.cos(Omega_rad)
        sin_Omega = torch.sin(Omega_rad)
        cos_i = torch.cos(i_rad)
        sin_i = torch.sin(i_rad)
        cos_omega = torch.cos(omega_rad)
        sin_omega = torch.sin(omega_rad)

        # Transformation matrix elements
        P11 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i
        P12 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i
        P21 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i
        P22 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i
        P31 = sin_omega * sin_i
        P32 = cos_omega * sin_i

        # Transform to heliocentric coordinates
        x = P11 * x_orb + P12 * y_orb
        y = P21 * x_orb + P22 * y_orb
        z = P31 * x_orb + P32 * y_orb

        vx = P11 * vx_orb + P12 * vy_orb
        vy = P21 * vx_orb + P22 * vy_orb
        vz = P31 * vx_orb + P32 * vy_orb

        return torch.stack([x, y, z, vx, vy, vz], dim=-1)

    def propagate(self, delta_time_days: torch.Tensor) -> "OrbitTensorDict":
        """
        Propagate the orbits using Keplerian orbital mechanics.

        Args:
            delta_time_days: Time difference in days

        Returns:
            New OrbitTensorDict with propagated elements
        """
        # Calculate new mean anomaly using mean motion
        n = self.compute_mean_motion()  # degrees/day
        new_M = self.mean_anomaly + n * delta_time_days
        new_M = torch.fmod(new_M, 360.0)  # Normalize to [0, 360)

        # Copy orbital elements and update mean anomaly
        new_elements = self["elements"].clone()
        new_elements[..., 5] = new_M

        return OrbitTensorDict(
            elements=new_elements,
            epoch=self["epoch"] + delta_time_days,
            frame=self["meta"]["frame"],
            central_body=self["meta"]["central_body"],
        )

    def get_orbital_position_at_phase(
        self, orbital_phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Get orbital position for given orbital phase (0-1).

        Args:
            orbital_phase: Orbital phase from 0 to 1

        Returns:
            [N, 3] Cartesian coordinates
        """
        # Convert phase to mean anomaly
        M_new = orbital_phase * 360.0  # degrees

        # Use current orbital elements but with new mean anomaly
        temp_elements = self["elements"].clone()
        temp_elements[..., 5] = M_new

        temp_orbit = OrbitTensorDict(
            elements=temp_elements,
            epoch=self["epoch"],
            frame=self["meta"]["frame"],
            central_body=self["meta"]["central_body"],
        )

        cartesian = temp_orbit.to_cartesian()
        return cartesian[..., :3]  # Just position, not velocity


def from_kepler_elements(
    semi_major_axes: torch.Tensor,
    eccentricities: torch.Tensor,
    inclinations: torch.Tensor,
    longitudes_asc_node: Optional[torch.Tensor] = None,
    arguments_periapsis: Optional[torch.Tensor] = None,
    mean_anomalies: Optional[torch.Tensor] = None,
    **kwargs,
) -> OrbitTensorDict:
    """
    Create OrbitTensorDict from individual orbital elements.

    Args:
        semi_major_axes: [N] Semi-major axes in AU
        eccentricities: [N] Eccentricities (dimensionless)
        inclinations: [N] Inclinations in degrees
        longitudes_asc_node: [N] Longitudes of ascending node in degrees (optional)
        arguments_periapsis: [N] Arguments of periapsis in degrees (optional)
        mean_anomalies: [N] Mean anomalies in degrees (optional)

    Returns:
        OrbitTensorDict with complete orbital elements
    """
    n_objects = semi_major_axes.shape[0]

    # Set defaults for optional elements
    if longitudes_asc_node is None:
        longitudes_asc_node = torch.zeros(n_objects)
    if arguments_periapsis is None:
        arguments_periapsis = torch.zeros(n_objects)
    if mean_anomalies is None:
        mean_anomalies = torch.zeros(n_objects)

    # Create complete orbital elements tensor
    elements = torch.stack(
        [
            semi_major_axes,
            eccentricities,
            inclinations,
            longitudes_asc_node,
            arguments_periapsis,
            mean_anomalies,
        ],
        dim=-1,
    )

    return OrbitTensorDict(elements, **kwargs)


def from_asteroid_database(asteroid_data: Dict[str, torch.Tensor]) -> OrbitTensorDict:
    """
    Create OrbitTensorDict from real asteroid database format.

    Args:
        asteroid_data: Dictionary with asteroid orbital elements

    Returns:
        OrbitTensorDict with real asteroid orbits
    """
    return from_kepler_elements(
        semi_major_axes=asteroid_data["a"],
        eccentricities=asteroid_data["e"],
        inclinations=asteroid_data["i"],
        longitudes_asc_node=asteroid_data.get("Omega", None),
        arguments_periapsis=asteroid_data.get("omega", None),
        mean_anomalies=asteroid_data.get("M", None),
        frame="ecliptic",
        central_body="Sun",
    )
