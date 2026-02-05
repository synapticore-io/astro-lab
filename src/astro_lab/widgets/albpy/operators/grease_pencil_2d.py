import bpy
import numpy as np
import polars as pl
from bpy.types import Operator

from ..grease_pencil_2d import GreasePencil2DPlotter


class ALBPY_OT_CreateRadarChart(Operator):
    bl_idname = "albpy.create_radar_chart"
    bl_label = "Create Radar Chart"
    bl_description = "Create a radar chart using Grease Pencil 2D Plotter"

    def execute(self, context):
        plotter = GreasePencil2DPlotter()
        # Example data
        data = {"A": 0.8, "B": 0.6, "C": 0.9, "D": 0.7, "E": 0.5}
        plotter.create_radar_chart(data, title="Example Radar Chart")
        self.report({"INFO"}, "Created radar chart.")
        return {"FINISHED"}


class ALBPY_OT_CreateMultiPanelPlot(Operator):
    bl_idname = "albpy.create_multi_panel_plot"
    bl_label = "Create Multi-Panel Plot"
    bl_description = "Create a multi-panel plot using Grease Pencil 2D Plotter"

    def execute(self, context):
        plotter = GreasePencil2DPlotter()
        # Example data: 4 panels with random data
        dfs = [
            pl.DataFrame({"x": np.random.rand(20), "y": np.random.rand(20)})
            for _ in range(4)
        ]
        titles = [f"Panel {i + 1}" for i in range(4)]
        types = ["scatter"] * 4
        plotter.create_multi_panel_plot(dfs, titles, types, layout=(2, 2))
        self.report({"INFO"}, "Created multi-panel plot.")
        return {"FINISHED"}


class ALBPY_OT_CreateComparisonPlot(Operator):
    bl_idname = "albpy.create_comparison_plot"
    bl_label = "Create Comparison Plot"
    bl_description = "Create a comparison histogram plot using Grease Pencil 2D Plotter"

    def execute(self, context):
        plotter = GreasePencil2DPlotter()
        # Example data
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(1, 1, 100)
        labels = ["Sample 1", "Sample 2"]
        plotter.create_comparison_plot(data1, data2, labels, title="Comparison Plot")
        self.report({"INFO"}, "Created comparison plot.")
        return {"FINISHED"}


def register():
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreateRadarChart)
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreateMultiPanelPlot)
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreateComparisonPlot)

    bpy.utils.register_class(ALBPY_OT_CreateRadarChart)
    bpy.utils.register_class(ALBPY_OT_CreateMultiPanelPlot)
    bpy.utils.register_class(ALBPY_OT_CreateComparisonPlot)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreateRadarChart)
    bpy.utils.unregister_class(ALBPY_OT_CreateMultiPanelPlot)
    bpy.utils.unregister_class(ALBPY_OT_CreateComparisonPlot)
