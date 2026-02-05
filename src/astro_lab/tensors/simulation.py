"""
Simulation TensorDict for N-Body simulations and cosmological data.

TensorDict for N-Body simulations and cosmological data with proper
gravitational dynamics and energy conservation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from .base import AstroTensorDict
from .mixins import ValidationMixin
from .spatial import SpatialTensorDict


class SimulationTensorDict(AstroTensorDict, ValidationMixin):
    """
    TensorDict for N-Body simulations and cosmological data.

    Structure:
    {
        "spatial": SpatialTensorDict,  # Particle positions (3D coordinates)
        "velocities": Tensor[N, 3],    # Particle velocities
        "masses": Tensor[N],           # Particle masses
        "potential": Tensor[N],        # Gravitational potential
        "forces": Tensor[N, 3],        # Forces (optional)
        "meta": {
            "simulation_type": str,
            "time_step": float,
            "current_time": float,
            "units": Dict[str, str],
            "cosmology": Dict[str, float],
        }
    }
    """

    def __init__(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        masses: torch.Tensor,
        simulation_type: str = "nbody",
        time_step: float = 0.01,
        current_time: float = 0.0,
        cosmology: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Initialize SimulationTensorDict.

        Args:
            positions: [N, 3] Particle positions
            velocities: [N, 3] Particle velocities
            masses: [N] Particle masses
            simulation_type: Type of simulation
            time_step: Time step
            current_time: Current time
            cosmology: Cosmological parameters
            **kwargs: Additional arguments
        """
        if positions.shape != velocities.shape:
            raise ValueError("Positions and velocities must have same shape")

        if positions.shape[0] != masses.shape[0]:
            raise ValueError("Number of positions must match number of masses")

        n_objects = positions.shape[0]

        # Default cosmology (Planck 2018)
        if cosmology is None:
            cosmology = {
                "H0": 67.4,  # Hubble constant
                "Omega_m": 0.315,  # Matter density
                "Omega_L": 0.685,  # Dark energy density
                "Omega_b": 0.049,  # Baryon density
            }

        # Create spatial component
        spatial = SpatialTensorDict(
            coordinates=positions,
            coordinate_system="cartesian",
            unit="kpc",
        )

        data = {
            "spatial": spatial,
            "velocities": velocities,
            "masses": masses,
            "potential": torch.zeros(n_objects),
            "forces": torch.zeros_like(positions),
            "meta": {
                "simulation_type": simulation_type,
                "time_step": time_step,
                "current_time": current_time,
                "units": {
                    "length": "kpc",
                    "velocity": "km/s",
                    "mass": "solar_mass",
                    "time": "Myr",
                },
                "cosmology": cosmology,
            },
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    def extract_features(
        self, feature_types: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract simulation features from the TensorDict.

        Args:
            feature_types: Types of features to extract ('simulation', 'dynamics', 'energetics')
            **kwargs: Additional extraction parameters

        Returns:
            Dictionary of extracted simulation features
        """
        # Get base features
        features = super().extract_features(feature_types, **kwargs)

        # Add simulation-specific computed features
        if feature_types is None or "simulation" in feature_types:
            # Basic simulation properties
            features["current_time"] = torch.tensor(
                self["meta"]["current_time"], dtype=torch.float32
            )
            features["time_step"] = torch.tensor(
                self["meta"]["time_step"], dtype=torch.float32
            )
            features["total_mass"] = torch.sum(self.masses)
            features["n_particles"] = torch.tensor(float(self.n_objects))

        if feature_types is None or "dynamics" in feature_types:
            # Dynamical features
            velocities = self.velocities
            features["velocity_magnitude"] = torch.norm(velocities, dim=-1)
            features["mean_velocity"] = torch.mean(torch.norm(velocities, dim=-1))
            features["velocity_dispersion"] = torch.std(torch.norm(velocities, dim=-1))

            # Center of mass
            com = self.compute_center_of_mass()
            features["center_of_mass_x"] = com[0]
            features["center_of_mass_y"] = com[1]
            features["center_of_mass_z"] = com[2]

            # System size
            com_distances = torch.norm(self.positions - com, dim=-1)
            features["virial_radius"] = torch.std(com_distances)
            features["max_radius"] = torch.max(com_distances)

        if feature_types is None or "energetics" in feature_types:
            # Energy features
            kinetic_energies = self.compute_kinetic_energy()
            features["kinetic_energy"] = torch.sum(kinetic_energies)
            features["mean_kinetic_energy"] = torch.mean(kinetic_energies)

            # Potential energy (if computed)
            if torch.any(self["potential"] != 0):
                features["potential_energy"] = torch.sum(self["potential"])
                features["total_energy"] = (
                    features["kinetic_energy"] + features["potential_energy"]
                )
                features["virial_ratio"] = (
                    2
                    * features["kinetic_energy"]
                    / torch.abs(features["potential_energy"])
                )
            else:
                # Compute potential energy
                potential = self.compute_potential_energy()
                features["potential_energy"] = torch.sum(potential)
                features["total_energy"] = (
                    features["kinetic_energy"] + features["potential_energy"]
                )
                features["virial_ratio"] = (
                    2
                    * features["kinetic_energy"]
                    / torch.abs(features["potential_energy"])
                )

        return features

    @property
    def positions(self) -> torch.Tensor:
        """Particle positions."""
        return self["spatial"]["coordinates"]

    @property
    def velocities(self) -> torch.Tensor:
        """Particle velocities."""
        return self["velocities"]

    @property
    def masses(self) -> torch.Tensor:
        """Particle masses."""
        return self["masses"]

    @property
    def spatial(self) -> SpatialTensorDict:
        """Spatial component."""
        return self["spatial"]

    def compute_gravitational_forces(self, softening: float = 0.1) -> torch.Tensor:
        """
        Computes gravitational forces between all particles.

        Args:
            softening: Softening parameter to avoid singularities

        Returns:
            [N, 3] Force tensor
        """
        positions = self.positions
        masses = self.masses
        n_particles = positions.shape[0]

        # Pairwise distances
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 3]
        distances = torch.norm(r_ij, dim=-1)  # [N, N]

        # Apply softening
        distances = torch.sqrt(distances**2 + softening**2)

        # Gravitational forces: F = G * m1 * m2 / r^2 * r_hat
        G = 4.301e-6  # Gravitational constant in kpc * (km/s)^2 / solar_mass

        # Avoid self-interaction
        mask = torch.eye(n_particles, dtype=torch.bool, device=positions.device)
        distances = distances.masked_fill(mask, float("inf"))

        # Force magnitudes
        force_magnitudes = G * masses.unsqueeze(1) * masses.unsqueeze(0) / distances**2
        force_magnitudes = force_magnitudes.masked_fill(mask, 0)

        # Force directions
        force_directions = r_ij / distances.unsqueeze(-1)
        force_directions = force_directions.masked_fill(mask.unsqueeze(-1), 0)

        # Total forces
        forces = -torch.sum(force_magnitudes.unsqueeze(-1) * force_directions, dim=1)

        self["forces"] = forces
        return forces

    def compute_potential_energy(self, softening: float = 0.1) -> torch.Tensor:
        """
        Computes gravitational potential for each particle.

        Args:
            softening: Softening parameter

        Returns:
            [N] Potential tensor
        """
        positions = self.positions
        masses = self.masses
        n_particles = positions.shape[0]

        # Pairwise distances
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.norm(r_ij, dim=-1)

        # Apply softening
        distances = torch.sqrt(distances**2 + softening**2)

        # Avoid self-interaction
        mask = torch.eye(n_particles, dtype=torch.bool, device=positions.device)
        distances = distances.masked_fill(mask, float("inf"))

        G = 4.301e-6  # Gravitational constant

        # Potential: Phi = -G * sum(m_j / r_ij)
        potential = -G * torch.sum(masses.unsqueeze(0) / distances, dim=1)

        self["potential"] = potential
        return potential

    def leapfrog_step(self) -> "SimulationTensorDict":
        """
        Performs a Leapfrog integration step.

        Returns:
            New SimulationTensorDict with updated state
        """
        dt = self["meta", "time_step"]

        # Compute current forces
        forces = self.compute_gravitational_forces()
        accelerations = forces / self.masses.unsqueeze(-1)

        # Leapfrog integration
        # v(t + dt/2) = v(t) + a(t) * dt/2
        half_step_velocities = self.velocities + accelerations * dt / 2

        # x(t + dt) = x(t) + v(t + dt/2) * dt
        new_positions = self.positions + half_step_velocities * dt

        # Create new state for force computation
        temp_sim = SimulationTensorDict(
            positions=new_positions,
            velocities=half_step_velocities,
            masses=self.masses,
            simulation_type=self["meta", "simulation_type"],
            time_step=dt,
            current_time=self["meta", "current_time"],
            cosmology=self["meta", "cosmology"],
        )

        # Compute new forces
        new_forces = temp_sim.compute_gravitational_forces()
        new_accelerations = new_forces / self.masses.unsqueeze(-1)

        # v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        new_velocities = half_step_velocities + new_accelerations * dt / 2

        return SimulationTensorDict(
            positions=new_positions,
            velocities=new_velocities,
            masses=self.masses,
            simulation_type=self["meta", "simulation_type"],
            time_step=dt,
            current_time=self["meta", "current_time"] + dt,
            cosmology=self["meta", "cosmology"],
        )

    def compute_kinetic_energy(self) -> torch.Tensor:
        """Computes kinetic energy."""
        velocities = self.velocities
        masses = self.masses

        # E_kin = 1/2 * m * v^2
        kinetic_energy = 0.5 * masses * torch.sum(velocities**2, dim=-1)
        return kinetic_energy

    def compute_total_energy(self) -> torch.Tensor:
        """Computes total energy of the system."""
        kinetic = torch.sum(self.compute_kinetic_energy())
        potential = torch.sum(self.compute_potential_energy())
        return kinetic + potential

    def compute_center_of_mass(self) -> torch.Tensor:
        """Computes center of mass of the system."""
        total_mass = torch.sum(self.masses)
        com = torch.sum(self.positions * self.masses.unsqueeze(-1), dim=0) / total_mass
        return com

    def compute_angular_momentum(self) -> torch.Tensor:
        """Computes total angular momentum of the system."""
        com = self.compute_center_of_mass()
        relative_positions = self.positions - com

        # L = sum(m_i * r_i x v_i)
        angular_momentum = torch.sum(
            self.masses.unsqueeze(-1)
            * torch.cross(relative_positions, self.velocities),
            dim=0,
        )
        return angular_momentum

    def run_simulation(
        self, n_steps: int, save_interval: int = 1
    ) -> List["SimulationTensorDict"]:
        """
        Runs N-Body simulation over multiple time steps.

        Args:
            n_steps: Number of time steps
            save_interval: Save interval

        Returns:
            List of SimulationTensorDict snapshots
        """
        snapshots = []
        current_sim = self

        for step in range(n_steps):
            if step % save_interval == 0:
                snapshots.append(current_sim)

            current_sim = current_sim.leapfrog_step()

            # Optional: Energy conservation check
            if step % 100 == 0:
                energy = current_sim.compute_total_energy()
                print(f"Step {step}: Total Energy = {energy:.6f}")

        # Last state addition
        snapshots.append(current_sim)

        return snapshots

    @classmethod
    def create_nbody_simulation(
        cls, n_particles: int = 100, system_type: str = "cluster", **kwargs
    ) -> "SimulationTensorDict":
        """
        Creates N-body simulation.

        Args:
            n_particles: Number of particles
            system_type: Type of system ("cluster", "galaxy", "solar_system")

        Returns:
            SimulationTensorDict
        """
        if system_type == "cluster":
            # Globular cluster
            positions = torch.randn(n_particles, 3) * 10  # kpc
            velocities = torch.randn(n_particles, 3) * 10  # km/s
            masses = torch.ones(n_particles)  # Solar masses

        elif system_type == "galaxy":
            # Disk galaxy
            r = torch.exp(torch.rand(n_particles)) * 5  # kpc
            theta = torch.rand(n_particles) * 2 * math.pi
            z = torch.normal(0, 0.5, (n_particles,))

            positions = torch.stack(
                [r * torch.cos(theta), r * torch.sin(theta), z], dim=-1
            )

            # Rotation velocity
            v_rot = torch.sqrt(200 * r / (r + 1))  # km/s
            velocities = torch.stack(
                [
                    -v_rot * torch.sin(theta),
                    v_rot * torch.cos(theta),
                    torch.zeros(n_particles),
                ],
                dim=-1,
            )

            masses = torch.ones(n_particles)

        elif system_type == "solar_system":
            # Simplified solar system
            if n_particles != 9:
                n_particles = 9  # Sun + 8 planets

            # Planet distances in AU
            distances = torch.tensor([0, 0.39, 0.72, 1.0, 1.52, 5.2, 9.5, 19.2, 30.1])
            masses = torch.tensor(
                [1.0, 0.055, 0.815, 1.0, 0.107, 317.8, 95.2, 14.5, 17.1]
            )  # Earth masses

            positions = torch.zeros(n_particles, 3)
            velocities = torch.zeros(n_particles, 3)

            for i in range(1, n_particles):
                positions[i, 0] = distances[i] * 1.496e8  # Convert to km
                velocities[i, 1] = torch.sqrt(
                    1.327e20 / (distances[i] * 1.496e8)
                )  # Orbital velocity

        else:
            raise ValueError(f"Unknown system type: {system_type}")

        return cls(
            positions=positions,
            velocities=velocities,
            masses=masses,
            simulation_type=system_type,
            **kwargs,
        )
