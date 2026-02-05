import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator

# Import the astronomical data utilities
from ..utilities.astronomical_data import (
    get_sample_data_for_visualization,
    load_survey_data,
)


class ALBPY_OT_CreateRadarChart(Operator):
    bl_idname = "albpy.create_radar_chart"
    bl_label = "Create Astronomical Radar Chart"
    bl_description = "Create a radar chart with real astronomical data from surveys"

    # Properties for data selection
    survey: EnumProperty(
        name="Survey",
        description="Select astronomical survey",
        items=[
            ("gaia", "Gaia DR3", "Gaia stellar data"),
            ("sdss", "SDSS DR17", "SDSS galaxy data"),
            ("nsa", "NSA", "NASA Sloan Atlas"),
            ("tng50", "TNG50", "TNG50 simulation"),
            ("exoplanet", "Exoplanet", "Exoplanet data"),
        ],
        default="gaia",
    )

    data_type: EnumProperty(
        name="Data Type",
        description="Type of data to visualize",
        items=[
            ("magnitudes", "Magnitudes", "Photometric magnitudes"),
            ("colors", "Colors", "Color indices"),
            ("properties", "Properties", "Physical properties"),
            ("custom", "Custom", "Custom selected features"),
        ],
        default="magnitudes",
    )

    max_points: IntProperty(
        name="Max Points",
        description="Maximum number of data points to sample",
        default=1000,
        min=100,
        max=10000,
    )

    scale: FloatProperty(
        name="Scale",
        description="Scale factor for the chart",
        default=5.0,
        min=1.0,
        max=20.0,
    )

    def execute(self, context):
        # Load real astronomical data using the utilities
        data, title = load_survey_data(
            survey=self.survey, data_type=self.data_type, max_points=self.max_points
        )

        if data is None:
            # Fallback to sample data
            sample_data = get_sample_data_for_visualization(
                self.survey, self.max_points
            )
            data = sample_data["features"]
            title = sample_data["title"]

            self.report({"WARNING"}, f"Using sample data for {self.survey}")

        # Create radar chart
        self._create_radar_chart(data, title, self.scale)

        self.report({"INFO"}, f"Created radar chart: {title}")
        return {"FINISHED"}

    def _create_radar_chart(self, data, title, scale):
        labels = list(data.keys())
        values = list(data.values())
        n_vars = len(labels)

        if n_vars < 3:
            return

        # Normalize values to 0-1 range for visualization
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized_values = [0.5] * len(values)

        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)

        # Create background grid
        self._create_radar_grid(angles, labels, scale)

        # Create data polygon
        data_coords = []
        for i, (angle, value) in enumerate(zip(angles, normalized_values)):
            x = value * scale * np.cos(angle)
            y = value * scale * np.sin(angle)
            data_coords.append([x, y, 0])

        # Close polygon
        data_coords.append(data_coords[0])

        # Create data line with astronomical color
        self._create_curve_line(data_coords, f"{title}_data", [0.2, 0.6, 1.0], 0.02)

        # Add title
        self._create_text_object(title, [0, scale + 1, 0], 0.5)

        # Add value labels
        for i, (angle, value, original_value) in enumerate(
            zip(angles, normalized_values, values)
        ):
            label_pos = [
                value * scale * 1.2 * np.cos(angle),
                value * scale * 1.2 * np.sin(angle),
                0,
            ]
            self._create_text_object(f"{original_value:.2f}", label_pos, 0.2)

    def _create_radar_grid(self, angles, labels, scale):
        # Create concentric circles
        for radius in [0.2, 0.4, 0.6, 0.8, 1.0]:
            circle_points = []
            for angle in np.linspace(0, 2 * np.pi, 64):
                x = radius * scale * np.cos(angle)
                y = radius * scale * np.sin(angle)
                circle_points.append([x, y, 0])

            self._create_curve_line(
                circle_points, f"grid_circle_{radius}", [0.5, 0.5, 0.5], 0.005
            )

        # Create radial lines and labels
        for i, angle in enumerate(angles):
            # Radial line
            line_points = [[0, 0, 0], [scale * np.cos(angle), scale * np.sin(angle), 0]]
            self._create_curve_line(
                line_points, f"grid_line_{i}", [0.5, 0.5, 0.5], 0.005
            )

            # Label
            label_pos = [scale * 1.1 * np.cos(angle), scale * 1.1 * np.sin(angle), 0]
            self._create_text_object(labels[i], label_pos, 0.3)

    def _create_curve_line(self, points, name, color, line_width):
        # Create curve data
        curve_data = bpy.data.curves.new(name, type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.resolution_u = 2

        # Create spline
        spline = curve_data.splines.new("POLY")
        spline.points.add(len(points) - 1)

        for i, point in enumerate(points):
            spline.points[i].co = (*point, 1)

        # Create object
        curve_obj = bpy.data.objects.new(name, curve_data)
        bpy.context.scene.collection.objects.link(curve_obj)

        # Create material using shader node group
        material = self._create_emission_material(f"{name}_mat", color)
        if material:
            curve_obj.data.materials.append(material)

        # Set line width
        curve_data.bevel_depth = line_width

    def _create_text_object(self, text, position, size):
        # Create text data
        text_data = bpy.data.curves.new("Text", type="FONT")
        text_data.body = text
        text_data.size = size

        # Create text object
        text_obj = bpy.data.objects.new("Text", text_data)
        text_obj.location = position
        bpy.context.scene.collection.objects.link(text_obj)

        # Create material using shader node group
        material = self._create_emission_material("TextMaterial", [1.0, 1.0, 1.0])
        if material:
            text_obj.data.materials.append(material)

    def _create_emission_material(self, name, color, strength=2.0):
        # Use existing emission shader node group
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Use emission shader node group
        emission_group = nodes.new("ShaderNodeGroup")
        emission_group.node_tree = bpy.data.node_groups.get("ALBPY_NG_Emission")

        # Create output node
        output = nodes.new("ShaderNodeOutputMaterial")

        # Link nodes
        links.new(emission_group.outputs["Shader"], output.inputs["Surface"])

        # Set parameters
        emission_group.inputs["Color"].default_value = (*color, 1.0)
        emission_group.inputs["Strength"].default_value = strength

        return material


def register():
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreateRadarChart)

    bpy.utils.register_class(ALBPY_OT_CreateRadarChart)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreateRadarChart)
