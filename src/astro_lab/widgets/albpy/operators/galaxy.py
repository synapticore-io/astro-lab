"""
Galaxy Creation Operators
=========================

Modern Blender 4.4 operators for creating galaxy objects with realistic properties.
"""

from typing import Set

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator


class ALBPY_OT_CreateGalaxy(Operator):
    """Create a realistic galaxy object with morphological properties."""

    bl_idname = "albpy.create_galaxy"
    bl_label = "Create Galaxy"
    bl_description = (
        "Create a galaxy object with realistic morphological and photometric properties"
    )
    bl_options = {"REGISTER", "UNDO"}

    # Galaxy Properties
    galaxy_type: EnumProperty(
        name="Galaxy Type",
        description="Hubble morphological classification",
        items=[
            ("E0", "E0 - Elliptical (spherical)", "Spherical elliptical galaxy"),
            ("E3", "E3 - Elliptical (moderate)", "Moderately flattened elliptical"),
            ("E7", "E7 - Elliptical (flattened)", "Highly flattened elliptical"),
            ("S0", "S0 - Lenticular", "Lenticular galaxy (no spiral arms)"),
            ("Sa", "Sa - Spiral (tight arms)", "Spiral with tightly wound arms"),
            ("Sb", "Sb - Spiral (moderate arms)", "Spiral with moderately wound arms"),
            ("Sc", "Sc - Spiral (loose arms)", "Spiral with loosely wound arms"),
            ("SBa", "SBa - Barred Spiral (tight)", "Barred spiral with tight arms"),
            (
                "SBb",
                "SBb - Barred Spiral (moderate)",
                "Barred spiral with moderate arms",
            ),
            ("SBc", "SBc - Barred Spiral (loose)", "Barred spiral with loose arms"),
            ("Irr", "Irregular", "Irregular galaxy"),
        ],
        default="Sb",
    )

    stellar_mass: FloatProperty(
        name="Stellar Mass (10^10 Msun)",
        description="Stellar mass in units of 10^10 solar masses",
        default=5.0,
        min=0.1,
        max=100.0,
        step=0.5,
    )

    effective_radius: FloatProperty(
        name="Effective Radius (kpc)",
        description="Half-light radius in kiloparsecs",
        default=3.0,
        min=0.5,
        max=50.0,
        step=0.5,
    )

    # Observational Properties
    distance: FloatProperty(
        name="Distance (Mpc)",
        description="Distance to galaxy in megaparsecs",
        default=10.0,
        min=1.0,
        max=1000.0,
        step=1.0,
    )

    inclination: FloatProperty(
        name="Inclination (degrees)",
        description="Inclination angle (0=face-on, 90=edge-on)",
        default=45.0,
        min=0.0,
        max=90.0,
        step=5.0,
    )

    position_angle: FloatProperty(
        name="Position Angle (degrees)",
        description="Position angle of major axis",
        default=0.0,
        min=0.0,
        max=180.0,
        step=5.0,
    )

    # Visual Properties
    add_material: BoolProperty(
        name="Add Galaxy Material",
        description="Create and apply galaxy material with realistic colors",
        default=True,
    )

    add_bulge_disk: BoolProperty(
        name="Separate Bulge/Disk",
        description="Create separate bulge and disk components",
        default=True,
    )

    resolution: EnumProperty(
        name="Resolution",
        description="Mesh resolution for galaxy components",
        items=[
            ("LOW", "Low", "Low resolution (fast)"),
            ("MEDIUM", "Medium", "Medium resolution (balanced)"),
            ("HIGH", "High", "High resolution (detailed)"),
        ],
        default="MEDIUM",
    )

    def execute(self, context: bpy.types.Context) -> Set[str]:
        """Execute galaxy creation."""

        try:
            # Get galaxy properties from utilities
            from ..utilities.galaxy_utilities import (
                get_galaxy_morphology_params,
                get_galaxy_properties,
            )

            # Calculate realistic properties
            galaxy_props = get_galaxy_properties(
                self.galaxy_type,
                self.stellar_mass * 1e10,  # Convert to solar masses
                redshift=0.0,  # Local universe
            )

            morph_params = get_galaxy_morphology_params(self.galaxy_type)

            # Create galaxy components
            galaxy_objects = []

            if self.add_bulge_disk and not morph_params["is_irregular"]:
                # Create separate bulge and disk
                if morph_params["bulge_fraction"] > 0.05:  # Significant bulge
                    bulge_obj = self._create_galaxy_bulge(
                        context, galaxy_props, morph_params
                    )
                    galaxy_objects.append(bulge_obj)

                if morph_params["disk_fraction"] > 0.05:  # Significant disk
                    disk_obj = self._create_galaxy_disk(
                        context, galaxy_props, morph_params
                    )
                    galaxy_objects.append(disk_obj)
            else:
                # Create single galaxy object
                galaxy_obj = self._create_single_galaxy(
                    context, galaxy_props, morph_params
                )
                galaxy_objects.append(galaxy_obj)

            # Apply transformations
            for obj in galaxy_objects:
                self._apply_galaxy_transform(obj)

            # Apply materials
            if self.add_material:
                for obj in galaxy_objects:
                    self._apply_galaxy_material(obj, galaxy_props, morph_params)

            # Select created objects
            bpy.ops.object.select_all(action="DESELECT")
            for obj in galaxy_objects:
                obj.select_set(True)

            if galaxy_objects:
                context.view_layer.objects.active = galaxy_objects[0]

            self.report(
                {"INFO"},
                f"Created {self.galaxy_type} galaxy with {len(galaxy_objects)} components",
            )
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to create galaxy: {str(e)}")
            return {"CANCELLED"}

    def _create_galaxy_bulge(
        self, context: bpy.types.Context, galaxy_props: dict, morph_params: dict
    ) -> bpy.types.Object:
        """Create galaxy bulge component."""

        # Create bulge mesh
        mesh = bpy.data.meshes.new(f"Bulge_{self.galaxy_type}")
        bm = bmesh.new()

        # Create ellipsoid for bulge
        subdivisions = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}[self.resolution]

        bmesh.ops.create_uvsphere(
            bm,
            u_segments=16 * (subdivisions + 1),
            v_segments=8 * (subdivisions + 1),
            radius=self.effective_radius * 0.5,  # Bulge is smaller
        )

        # Apply bulge ellipticity
        bulge_ellipticity = min(
            morph_params["ellipticity"], 0.7
        )  # Bulges are less flat

        for vert in bm.verts:
            vert.co.z *= 1 - bulge_ellipticity

        # Apply smooth shading
        for face in bm.faces:
            face.smooth = True

        bm.to_mesh(mesh)
        bm.free()

        # Create object
        obj = bpy.data.objects.new(f"Galaxy_Bulge_{self.galaxy_type}", mesh)
        context.collection.objects.link(obj)

        return obj

    def _create_galaxy_disk(
        self, context: bpy.types.Context, galaxy_props: dict, morph_params: dict
    ) -> bpy.types.Object:
        """Create galaxy disk component."""

        # Create disk mesh
        mesh = bpy.data.meshes.new(f"Disk_{self.galaxy_type}")
        bm = bmesh.new()

        # Create cylinder for disk
        subdivisions = {"LOW": 16, "MEDIUM": 32, "HIGH": 64}[self.resolution]

        bmesh.ops.create_cylinder(
            bm,
            vertices=subdivisions,
            radius=self.effective_radius,
            depth=self.effective_radius * 0.1,  # Thin disk
            cap_ends=True,
        )

        # Apply disk ellipticity
        disk_ellipticity = morph_params["ellipticity"]

        for vert in bm.verts:
            # Create elliptical disk
            distance = (vert.co.x**2 + vert.co.y**2) ** 0.5
            if distance > 0:
                angle = np.arctan2(vert.co.y, vert.co.x)
                vert.co.x = distance * np.cos(angle)
                vert.co.y = distance * np.sin(angle) * (1 - disk_ellipticity)

        # Add spiral structure if spiral galaxy
        if morph_params["has_spiral_arms"]:
            self._add_spiral_structure(bm, morph_params)

        # Apply smooth shading
        for face in bm.faces:
            face.smooth = True

        bm.to_mesh(mesh)
        bm.free()

        # Create object
        obj = bpy.data.objects.new(f"Galaxy_Disk_{self.galaxy_type}", mesh)
        context.collection.objects.link(obj)

        return obj

    def _create_single_galaxy(
        self, context: bpy.types.Context, galaxy_props: dict, morph_params: dict
    ) -> bpy.types.Object:
        """Create single galaxy object."""

        # Create galaxy mesh
        mesh = bpy.data.meshes.new(f"Galaxy_{self.galaxy_type}")
        bm = bmesh.new()

        if morph_params["is_irregular"]:
            # Irregular galaxy - use deformed sphere
            bmesh.ops.create_icosphere(bm, subdivisions=2, radius=self.effective_radius)

            # Add random deformation
            for vert in bm.verts:
                noise = np.random.normal(0, 0.3)
                vert.co *= 1 + noise

        elif morph_params["is_early_type"]:
            # Elliptical galaxy
            bmesh.ops.create_uvsphere(
                bm, u_segments=32, v_segments=16, radius=self.effective_radius
            )

            # Apply ellipticity
            for vert in bm.verts:
                vert.co.z *= 1 - morph_params["ellipticity"]

        else:
            # Spiral galaxy - disk with bulge
            bmesh.ops.create_cylinder(
                bm,
                vertices=64,
                radius=self.effective_radius,
                depth=self.effective_radius * 0.2,
                cap_ends=True,
            )

        # Apply smooth shading
        for face in bm.faces:
            face.smooth = True

        bm.to_mesh(mesh)
        bm.free()

        # Create object
        obj = bpy.data.objects.new(f"Galaxy_{self.galaxy_type}", mesh)
        context.collection.objects.link(obj)

        return obj

    def _add_spiral_structure(self, bm: bmesh.types.BMesh, morph_params: dict) -> None:
        """Add spiral arm structure to disk mesh."""

        # Simple spiral arm displacement
        spiral_strength = 0.1
        arm_count = 2 if "grand_design" in self.galaxy_type.lower() else 4

        for vert in bm.verts:
            radius = (vert.co.x**2 + vert.co.y**2) ** 0.5
            if radius > 0:
                angle = np.arctan2(vert.co.y, vert.co.x)

                # Spiral pattern
                spiral_phase = arm_count * angle - 2 * radius / self.effective_radius
                spiral_amplitude = spiral_strength * np.sin(spiral_phase)

                # Apply displacement
                vert.co.z += spiral_amplitude * self.effective_radius * 0.1

    def _apply_galaxy_transform(self, obj: bpy.types.Object) -> None:
        """Apply distance, inclination, and position angle."""

        # Set location at cursor
        obj.location = bpy.context.scene.cursor.location

        # Apply inclination (rotation around X-axis)
        obj.rotation_euler[0] = np.radians(self.inclination)

        # Apply position angle (rotation around Z-axis)
        obj.rotation_euler[2] = np.radians(self.position_angle)

        # Scale based on distance (angular size effect)
        angular_scale = self.effective_radius / self.distance  # Rough approximation
        scale_factor = angular_scale * 10  # Arbitrary scaling for visibility
        obj.scale = (scale_factor, scale_factor, scale_factor)

    def _apply_galaxy_material(
        self, obj: bpy.types.Object, galaxy_props: dict, morph_params: dict
    ) -> None:
        """Apply galaxy material."""

        from ..nodes.shader.galaxy import create_galaxy_material

        # Create galaxy material
        mat_name = f"Galaxy_{self.galaxy_type}_{obj.name}"
        material = create_galaxy_material(mat_name, self.galaxy_type)

        # Apply to object
        if not obj.data.materials:
            obj.data.materials.append(material)
        else:
            obj.data.materials[0] = material


class ALBPY_OT_CreateGalaxyCluster(Operator):
    """Create a cluster of galaxies with realistic distribution."""

    bl_idname = "albpy.create_galaxy_cluster"
    bl_label = "Create Galaxy Cluster"
    bl_description = "Create a cluster of galaxies with realistic spatial and morphological distribution"
    bl_options = {"REGISTER", "UNDO"}

    galaxy_count: IntProperty(
        name="Galaxy Count",
        description="Number of galaxies in cluster",
        default=20,
        min=5,
        max=200,
    )

    cluster_radius: FloatProperty(
        name="Cluster Radius (Mpc)",
        description="Radius of galaxy cluster in megaparsecs",
        default=5.0,
        min=1.0,
        max=50.0,
    )

    central_galaxy: BoolProperty(
        name="Central Giant Galaxy",
        description="Include central dominant galaxy (cD galaxy)",
        default=True,
    )

    mixed_types: BoolProperty(
        name="Mixed Galaxy Types",
        description="Include mix of elliptical and spiral galaxies",
        default=True,
    )

    redshift: FloatProperty(
        name="Cluster Redshift",
        description="Redshift of galaxy cluster",
        default=0.05,
        min=0.0,
        max=2.0,
    )

    def execute(self, context: bpy.types.Context) -> Set[str]:
        """Execute galaxy cluster creation."""

        try:
            cluster_galaxies = []

            # Galaxy type distribution for clusters (more ellipticals)
            if self.mixed_types:
                galaxy_types = ["E0", "E3", "E7", "S0", "Sa", "Sb", "Sc"]
                type_weights = [0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02]  # E-dominated
            else:
                galaxy_types = ["E3"]
                type_weights = [1.0]

            # Create central galaxy if requested
            if self.central_galaxy:
                # Central cD galaxy
                central_galaxy = self._create_cluster_galaxy(
                    context,
                    "E0",
                    position=(0, 0, 0),
                    stellar_mass=50.0,
                    effective_radius=20.0,
                )
                cluster_galaxies.append(central_galaxy)
                start_index = 1
            else:
                start_index = 0

            # Create cluster member galaxies
            for i in range(start_index, self.galaxy_count):
                # Random galaxy type
                galaxy_type = np.random.choice(galaxy_types, p=type_weights)

                # Random position in cluster (3D Gaussian)
                r = np.random.exponential(
                    self.cluster_radius / 3
                )  # Exponential radial profile
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)

                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)

                # Mass-dependent size (larger galaxies are rarer)
                stellar_mass = np.random.lognormal(
                    np.log(5.0), 0.7
                )  # Log-normal mass function
                stellar_mass = np.clip(stellar_mass, 0.5, 30.0)

                effective_radius = 2.0 + stellar_mass * 0.3  # Mass-size relation

                # Create galaxy
                galaxy = self._create_cluster_galaxy(
                    context,
                    galaxy_type,
                    position=(x, y, z),
                    stellar_mass=stellar_mass,
                    effective_radius=effective_radius,
                )
                cluster_galaxies.append(galaxy)

            # Group galaxies in collection
            cluster_collection = bpy.data.collections.new(
                f"Galaxy_Cluster_z{self.redshift:.2f}"
            )
            context.scene.collection.children.link(cluster_collection)

            for galaxy in cluster_galaxies:
                context.collection.objects.unlink(galaxy)
                cluster_collection.objects.link(galaxy)

            # Select all cluster galaxies
            bpy.ops.object.select_all(action="DESELECT")
            for galaxy in cluster_galaxies:
                galaxy.select_set(True)

            if cluster_galaxies:
                context.view_layer.objects.active = cluster_galaxies[0]

            self.report(
                {"INFO"},
                f"Created galaxy cluster with {len(cluster_galaxies)} galaxies",
            )
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to create galaxy cluster: {str(e)}")
            return {"CANCELLED"}

    def _create_cluster_galaxy(
        self,
        context: bpy.types.Context,
        galaxy_type: str,
        position: tuple,
        stellar_mass: float,
        effective_radius: float,
    ) -> bpy.types.Object:
        """Create individual galaxy for cluster."""

        # Use galaxy creation operator
        bpy.ops.albpy.create_galaxy(
            galaxy_type=galaxy_type,
            stellar_mass=stellar_mass,
            effective_radius=effective_radius,
            distance=100.0,  # Cluster distance
            inclination=np.random.uniform(0, 90),
            position_angle=np.random.uniform(0, 180),
            add_material=True,
            add_bulge_disk=False,  # Single object for cluster
            resolution="LOW",  # Faster for many galaxies
        )

        # Get created galaxy
        galaxy = context.active_object

        # Set position
        galaxy.location = position

        # Random orientation
        galaxy.rotation_euler = (
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, 2 * np.pi),
        )

        return galaxy


def register():
    """Register galaxy operators."""
    bpy.utils.register_class(ALBPY_OT_CreateGalaxy)
    bpy.utils.register_class(ALBPY_OT_CreateGalaxyCluster)


def unregister():
    """Unregister galaxy operators."""
    bpy.utils.unregister_class(ALBPY_OT_CreateGalaxy)
    bpy.utils.unregister_class(ALBPY_OT_CreateGalaxyCluster)
