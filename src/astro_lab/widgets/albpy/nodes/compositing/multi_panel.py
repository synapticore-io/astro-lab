"""
Multi-Panel Compositing Node Group
==================================

Creates a multi-panel layout for astronomical data visualization
with support for different shaders and effects per panel.
"""

import bpy


class AlbpyMultiPanelCompositingGroup(bpy.types.CompositorNodeGroup):
    bl_idname = "ALBPY_NG_MultiPanel"
    bl_label = "Albpy Multi-Panel Layout"

    @classmethod
    def poll(cls, ntree):  # type: ignore
        return ntree.bl_idname == "CompositorNodeTree"

    def init(self, context):
        # Clear existing nodes
        self.nodes.clear()
        self.inputs.clear()
        self.outputs.clear()

        # Create inputs for panels (default 4 panels)
        for i in range(4):
            self.inputs.new("NodeSocketColor", f"Panel {i + 1}")

        # Create background input
        self.inputs.new("NodeSocketColor", "Background")

        # Create layout control inputs
        self.inputs.new("NodeSocketInt", "Rows")
        self.inputs.new("NodeSocketInt", "Columns")
        self.inputs.new("NodeSocketFloat", "Panel Spacing")
        self.inputs.new("NodeSocketFloat", "Border Width")
        self.inputs.new("NodeSocketColor", "Border Color")

        # Create outputs
        self.outputs.new("NodeSocketColor", "Multi-Panel Output")

        # Set default values
        self.inputs["Rows"].default_value = 2
        self.inputs["Columns"].default_value = 2
        self.inputs["Panel Spacing"].default_value = 0.1
        self.inputs["Border Width"].default_value = 0.02
        self.inputs["Border Color"].default_value = (0.3, 0.3, 0.3, 1.0)
        self.inputs["Background"].default_value = (0.1, 0.1, 0.1, 1.0)

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create mix nodes for combining panels
        mix_nodes = []
        for i in range(4):
            mix_node = self.nodes.new("CompositorNodeMixRGB")
            mix_node.blend_type = "ALPHA_OVER"
            mix_node.location = (200 * (i % 2), -200 * (i // 2))
            mix_nodes.append(mix_node)

        # Create scale nodes for panel positioning
        scale_nodes = []
        for i in range(4):
            scale_node = self.nodes.new("CompositorNodeScale")
            scale_node.location = (100 * (i % 2), -100 * (i // 2))
            scale_nodes.append(scale_node)

        # Create translate nodes for panel positioning
        translate_nodes = []
        for i in range(4):
            translate_node = self.nodes.new("CompositorNodeTranslate")
            translate_node.location = (150 * (i % 2), -150 * (i // 2))
            translate_nodes.append(translate_node)

        # Position main nodes
        input_node.location = (-400, 0)
        output_node.location = (600, 0)

        # Connect panel inputs to scale nodes
        for i in range(4):
            self.links.new(
                input_node.outputs[f"Panel {i + 1}"], scale_nodes[i].inputs["Image"]
            )

        # Connect scale nodes to translate nodes
        for i in range(4):
            self.links.new(
                scale_nodes[i].outputs["Image"], translate_nodes[i].inputs["Image"]
            )

        # Connect translate nodes to mix nodes
        for i in range(4):
            self.links.new(
                translate_nodes[i].outputs["Image"],
                mix_nodes[i].inputs[1],  # Color 1
            )

        # Connect background to first mix node
        self.links.new(
            input_node.outputs["Background"],
            mix_nodes[0].inputs[2],  # Color 2
        )

        # Chain mix nodes together
        for i in range(3):
            self.links.new(
                mix_nodes[i].outputs["Image"],
                mix_nodes[i + 1].inputs[2],  # Color 2
            )

        # Connect final mix node to output
        self.links.new(
            mix_nodes[3].outputs["Image"], output_node.inputs["Multi-Panel Output"]
        )


# Multi-panel presets
MULTI_PANEL_PRESETS = {
    "2x2_grid": {
        "Rows": 2,
        "Columns": 2,
        "Panel Spacing": 0.1,
        "Border Width": 0.02,
        "Border Color": (0.3, 0.3, 0.3, 1.0),
    },
    "horizontal_2": {
        "Rows": 1,
        "Columns": 2,
        "Panel Spacing": 0.15,
        "Border Width": 0.03,
        "Border Color": (0.4, 0.4, 0.4, 1.0),
    },
    "vertical_2": {
        "Rows": 2,
        "Columns": 1,
        "Panel Spacing": 0.15,
        "Border Width": 0.03,
        "Border Color": (0.4, 0.4, 0.4, 1.0),
    },
    "3x3_grid": {
        "Rows": 3,
        "Columns": 3,
        "Panel Spacing": 0.08,
        "Border Width": 0.015,
        "Border Color": (0.25, 0.25, 0.25, 1.0),
    },
    "astronomical": {
        "Rows": 2,
        "Columns": 2,
        "Panel Spacing": 0.12,
        "Border Width": 0.025,
        "Border Color": (0.2, 0.3, 0.5, 1.0),
    },
}


def get_multi_panel_preset(preset_name: str) -> dict:
    """Get preset for multi-panel configuration."""
    return MULTI_PANEL_PRESETS.get(preset_name, MULTI_PANEL_PRESETS["2x2_grid"])


def apply_multi_panel_preset(
    node_group: bpy.types.CompositorNodeGroup, preset_name: str
) -> None:
    """Apply multi-panel preset to node group."""
    preset = get_multi_panel_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def create_multi_panel_layout(
    rows: int = 2,
    columns: int = 2,
    panel_spacing: float = 0.1,
    border_width: float = 0.02,
    border_color: tuple = (0.3, 0.3, 0.3, 1.0),
) -> bpy.types.CompositorNodeGroup:
    """Create a multi-panel layout node group with specified parameters."""

    # Create the node group
    node_group = bpy.data.node_groups.new(
        type="CompositorNodeTree", name="ALBPY_NG_MultiPanel"
    )

    # Set parameters
    node_group.inputs["Rows"].default_value = rows
    node_group.inputs["Columns"].default_value = columns
    node_group.inputs["Panel Spacing"].default_value = panel_spacing
    node_group.inputs["Border Width"].default_value = border_width
    node_group.inputs["Border Color"].default_value = border_color

    return node_group


def register():
    # REMOVED: bpy.utils.register_class(AlbpyMultiPanelCompositingGroup)

    bpy.utils.register_class(AlbpyMultiPanelCompositingGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyMultiPanelCompositingGroup)
