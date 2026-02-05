"""
Star Creation Operators
=======================

Modern Blender 4.4 operators for creating stellar objects with realistic properties.
"""

from typing import Set

import bmesh
import bpy
import mathutils
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from bpy.types import Operator


class ALBPY_OT_CreateStar(Operator):
    """Create a realistic star object with stellar classification and materials."""

    bl_idname = "albpy.create_star"
    bl_label = "Create Star"
    bl_description = (
        "Create a star object with realistic stellar properties and materials"
    )
    bl_options = {"REGISTER", "UNDO"}

    # Stellar Properties
    spectral_class: EnumProperty(
        name="Spectral Class",
        description="Stellar spectral classification",
        items=[
            ("O", "O-Class", "Blue supergiant (30,000+ K)"),
            ("B", "B-Class", "Blue-white main sequence (10,000-30,000 K)"),
            ("A", "A-Class", "White main sequence (7,500-10,000 K)"),
            ("F", "F-Class", "Yellow-white main sequence (6,000-7,500 K)"),
            ("G", "G-Class", "Yellow main sequence (5,200-6,000 K) - Sun-like"),
            ("K", "K-Class", "Orange main sequence (3,700-5,200 K)"),
            ("M", "M-Class", "Red dwarf (2,400-3,700 K)"),
        ],
        default="G",
    )

    temperature: FloatProperty(
        name="Temperature (K)",
        description="Stellar surface temperature in Kelvin",
        default=5778.0,
        min=2000.0,
        max=50000.0,
        step=100,
    )

    radius: FloatProperty(
        name="Radius (Solar Radii)",
        description="Stellar radius in solar radii",
        default=1.0,
        min=0.1,
        max=1000.0,
        step=0.1,
    )

    luminosity: FloatProperty(
        name="Luminosity (Solar)",
        description="Stellar luminosity in solar luminosities",
        default=1.0,
        min=0.0001,
        max=1000000.0,
        step=0.1,
    )

    # Visual Properties
    emission_strength: FloatProperty(
        name="Emission Strength",
        description="Material emission strength multiplier",
        default=10.0,
        min=0.1,
        max=100.0,
        step=0.5,
    )

    corona_intensity: FloatProperty(
        name="Corona Intensity",
        description="Stellar corona glow intensity",
        default=0.3,
        min=0.0,
        max=5.0,
        step=0.1,
    )

    # Additional Options
    add_halo: BoolProperty(
        name="Add Light Halo",
        description="Add point light for stellar illumination",
        default=True,
    )

    add_material: BoolProperty(
        name="Add Stellar Material",
        description="Create and apply stellar material with node groups",
        default=True,
    )

    subdivisions: EnumProperty(
        name="Subdivisions",
        description="Mesh subdivision level",
        items=[
            ("0", "Low (ICO-0)", "Low detail (12 faces)"),
            ("1", "Medium (ICO-1)", "Medium detail (42 faces)"),
            ("2", "High (ICO-2)", "High detail (162 faces)"),
            ("3", "Very High (ICO-3)", "Very high detail (642 faces)"),
        ],
        default="2",
    )

    def execute(self, context: bpy.types.Context) -> Set[str]:
        """Execute star creation with modern practices."""

        try:
            # Update properties based on spectral class
            self._update_stellar_properties()

            # Create star mesh
            star_obj = self._create_star_mesh(context)

            # Apply stellar material if requested
            if self.add_material:
                self._apply_stellar_material(star_obj)

            # Add lighting if requested
            if self.add_halo:
                self._create_stellar_lighting(context, star_obj)

            # Set as active object
            context.view_layer.objects.active = star_obj
            star_obj.select_set(True)

            self.report(
                {"INFO"}, f"Created {self.spectral_class}-class star: {star_obj.name}"
            )
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to create star: {str(e)}")
            return {"CANCELLED"}

    def _update_stellar_properties(self) -> None:
        """Update stellar properties based on spectral class."""
        from ..nodes.shader.star import STELLAR_PRESETS

        if self.spectral_class in STELLAR_PRESETS:
            preset = STELLAR_PRESETS[self.spectral_class]

            # Update properties if not manually modified
            if not hasattr(self, "_manual_temp"):
                self.temperature = preset["Temperature"]
            if not hasattr(self, "_manual_lum"):
                self.luminosity = preset["Luminosity"]
            if not hasattr(self, "_manual_em"):
                self.emission_strength = preset["Emission Strength"]
            if not hasattr(self, "_manual_corona"):
                self.corona_intensity = preset["Corona Intensity"]

    def _create_star_mesh(self, context: bpy.types.Context) -> bpy.types.Object:
        """Create the star mesh geometry."""

        # Create mesh using bmesh for better control
        mesh = bpy.data.meshes.new(f"Star_{self.spectral_class}")
        bm = bmesh.new()

        # Create icosphere
        bmesh.ops.create_icosphere(
            bm, subdivisions=int(self.subdivisions), radius=self.radius
        )

        # Apply smooth shading
        for face in bm.faces:
            face.smooth = True

        # Update mesh
        bm.to_mesh(mesh)
        bm.free()

        # Create object
        star_obj = bpy.data.objects.new(
            f"Star_{self.spectral_class}_{self.temperature:.0f}K", mesh
        )
        context.collection.objects.link(star_obj)

        # Set location at cursor
        star_obj.location = context.scene.cursor.location

        return star_obj

    def _apply_stellar_material(self, star_obj: bpy.types.Object) -> None:
        """Apply stellar material with node groups."""
        from ..nodes.shader.star import create_stellar_material

        # Create stellar material
        mat_name = f"Stellar_{self.spectral_class}_{self.temperature:.0f}K"

        # Check if material already exists
        existing_mat = bpy.data.materials.get(mat_name)
        if existing_mat:
            star_obj.data.materials.append(existing_mat)
        else:
            # Create new stellar material
            stellar_mat = create_stellar_material(mat_name, self.spectral_class)
            star_obj.data.materials.append(stellar_mat)

            # Apply custom properties
            if stellar_mat.node_tree:
                for node in stellar_mat.node_tree.nodes:
                    if node.type == "GROUP" and "ALBPY_Stellar" in str(node.node_tree):
                        node.inputs["Temperature"].default_value = self.temperature
                        node.inputs["Luminosity"].default_value = self.luminosity
                        node.inputs[
                            "Emission Strength"
                        ].default_value = self.emission_strength
                        node.inputs[
                            "Corona Intensity"
                        ].default_value = self.corona_intensity

    def _create_stellar_lighting(
        self, context: bpy.types.Context, star_obj: bpy.types.Object
    ) -> None:
        """Create point light for stellar illumination."""

        # Create point light
        light_data = bpy.data.lights.new(f"StarLight_{star_obj.name}", "POINT")
        light_obj = bpy.data.objects.new(f"Light_{star_obj.name}", light_data)
        context.collection.objects.link(light_obj)

        # Position at star location
        light_obj.location = star_obj.location

        # Calculate realistic light properties
        # Energy scales with luminosity and emission strength
        base_energy = self.luminosity * self.emission_strength * 1000
        light_data.energy = base_energy

        # Color based on temperature (simplified blackbody)
        if self.temperature > 10000:
            light_data.color = (0.7, 0.8, 1.0)  # Blue-white
        elif self.temperature > 7000:
            light_data.color = (0.9, 0.9, 1.0)  # White
        elif self.temperature > 5000:
            light_data.color = (1.0, 1.0, 0.8)  # Yellow-white
        elif self.temperature > 3500:
            light_data.color = (1.0, 0.8, 0.6)  # Orange
        else:
            light_data.color = (1.0, 0.6, 0.4)  # Red

        # Parent light to star
        light_obj.parent = star_obj
        light_obj.parent_type = "OBJECT"

    def draw(self, context: bpy.types.Context) -> None:
        """Draw operator UI."""
        layout = self.layout

        # Stellar Classification
        col = layout.column()
        col.prop(self, "spectral_class")

        # Stellar Properties
        box = layout.box()
        box.label(text="Stellar Properties")
        row = box.row()
        row.prop(self, "temperature")
        row.prop(self, "luminosity")
        row = box.row()
        row.prop(self, "radius")

        # Visual Properties
        box = layout.box()
        box.label(text="Visual Properties")
        row = box.row()
        row.prop(self, "emission_strength")
        row.prop(self, "corona_intensity")

        # Options
        box = layout.box()
        box.label(text="Options")
        row = box.row()
        row.prop(self, "add_material")
        row.prop(self, "add_halo")
        box.prop(self, "subdivisions")


class ALBPY_OT_CreateStarField(Operator):
    """Create a field of multiple stars with distribution patterns."""

    bl_idname = "albpy.create_star_field"
    bl_label = "Create Star Field"
    bl_description = "Create multiple stars with realistic distribution"
    bl_options = {"REGISTER", "UNDO"}

    count: FloatProperty(
        name="Star Count",
        description="Number of stars to create",
        default=100,
        min=1,
        max=10000,
        step=1,
    )

    distribution: EnumProperty(
        name="Distribution",
        description="Spatial distribution pattern",
        items=[
            ("RANDOM", "Random", "Random distribution in sphere"),
            ("DISK", "Galactic Disk", "Disk-like distribution"),
            ("CLUSTER", "Star Cluster", "Clustered distribution"),
        ],
        default="RANDOM",
    )

    radius: FloatProperty(
        name="Field Radius",
        description="Radius of star field",
        default=50.0,
        min=1.0,
        max=1000.0,
    )

    mixed_classes: BoolProperty(
        name="Mixed Spectral Classes",
        description="Create stars with mixed spectral classes",
        default=True,
    )

    def execute(self, context: bpy.types.Context) -> Set[str]:
        """Execute star field creation."""
        import random

        # Spectral class probabilities (realistic)
        spectral_weights = {
            "M": 0.76,  # Red dwarfs are most common
            "K": 0.12,
            "G": 0.08,
            "F": 0.03,
            "A": 0.006,
            "B": 0.0013,
            "O": 0.00003,
        }

        created_stars = []

        for i in range(int(self.count)):
            # Choose spectral class
            if self.mixed_classes:
                spectral_class = random.choices(
                    list(spectral_weights.keys()),
                    weights=list(spectral_weights.values()),
                )[0]
            else:
                spectral_class = "G"

            # Generate position based on distribution
            if self.distribution == "RANDOM":
                # Uniform random in sphere
                phi = random.uniform(0, 2 * 3.14159)
                costheta = random.uniform(-1, 1)
                u = random.uniform(0, 1)

                theta = 3.14159 * 0.5 - mathutils.math.acos(costheta)
                r = self.radius * (u ** (1 / 3))

                x = r * mathutils.math.cos(theta) * mathutils.math.cos(phi)
                y = r * mathutils.math.cos(theta) * mathutils.math.sin(phi)
                z = r * mathutils.math.sin(theta)

            elif self.distribution == "DISK":
                # Disk distribution
                r = random.uniform(0, self.radius)
                phi = random.uniform(0, 2 * 3.14159)
                z = random.gauss(0, self.radius * 0.1)  # Thin disk

                x = r * mathutils.math.cos(phi)
                y = r * mathutils.math.sin(phi)

            else:  # CLUSTER
                # Gaussian cluster
                x = random.gauss(0, self.radius * 0.3)
                y = random.gauss(0, self.radius * 0.3)
                z = random.gauss(0, self.radius * 0.3)

            # Create star using existing operator
            bpy.ops.albpy.create_star(
                spectral_class=spectral_class,
                add_halo=False,  # Too many lights would be slow
                subdivisions="1",  # Lower detail for fields
            )

            # Position the star
            if context.active_object:
                context.active_object.location = (x, y, z)
                created_stars.append(context.active_object)

        self.report({"INFO"}, f"Created star field with {len(created_stars)} stars")
        return {"FINISHED"}


def register():
    """Register star operators."""
    bpy.utils.register_class(ALBPY_OT_CreateStar)
    bpy.utils.register_class(ALBPY_OT_CreateStarField)


def unregister():
    """Unregister star operators."""
    bpy.utils.unregister_class(ALBPY_OT_CreateStar)
    bpy.utils.unregister_class(ALBPY_OT_CreateStarField)
