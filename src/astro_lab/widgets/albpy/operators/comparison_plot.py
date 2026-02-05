import bpy
import numpy as np
from bpy.types import Operator


class ALBPY_OT_CreateComparisonPlot(Operator):
    bl_idname = "albpy.create_comparison_plot"
    bl_label = "Create Comparison Plot"
    bl_description = (
        "Create a comparison histogram plot with direct geometry and shader node groups"
    )

    def execute(self, context):
        # Example data
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(1, 1, 100)
        labels = ["Sample 1", "Sample 2"]
        title = "Comparison Plot"

        self._create_comparison_plot(data1, data2, labels, title)

        self.report({"INFO"}, f"Created comparison plot: {title}")
        return {"FINISHED"}

    def _create_comparison_plot(self, data1, data2, labels, title):
        colors = [[0.2, 0.6, 1.0], [1.0, 0.4, 0.2]]

        # Create histograms
        self._create_histogram(data1, f"{title}_hist1", colors[0], offset_x=-2.0)
        self._create_histogram(data2, f"{title}_hist2", colors[1], offset_x=2.0)

        # Add title
        self._create_text_object(title, [0, 4, 0], 0.6)

    def _create_histogram(self, data, name, color, bins=20, offset_x=0.0):
        # Calculate histogram
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize heights
        max_height = 2.0
        normalized_heights = hist / hist.max() * max_height if hist.max() > 0 else hist

        # Create bars
        bar_width = (bin_edges[1] - bin_edges[0]) * 0.8
        for i, (center, height) in enumerate(zip(bin_centers, normalized_heights)):
            if height > 0:
                bar_points = [
                    [offset_x + center - bar_width / 2, 0, 0],
                    [offset_x + center + bar_width / 2, 0, 0],
                    [offset_x + center + bar_width / 2, height, 0],
                    [offset_x + center - bar_width / 2, height, 0],
                    [offset_x + center - bar_width / 2, 0, 0],
                ]

                self._create_curve_line(
                    bar_points, f"{name}_bar_{i}", color, line_width=0.02
                )

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


def register():
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreateComparisonPlot)

    bpy.utils.register_class(ALBPY_OT_CreateComparisonPlot)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreateComparisonPlot)
