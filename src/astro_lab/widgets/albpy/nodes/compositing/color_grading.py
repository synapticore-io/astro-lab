import bpy


class AlbpyColorGradingCompositingGroup(bpy.types.CompositorNodeGroup):
    bl_idname = "ALBPY_NG_ColorGrading"
    bl_label = "Albpy Color Grading Compositing Group"

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == "CompositorNodeTree"

    def init(self, context):
        # Clear existing nodes
        self.nodes.clear()
        self.inputs.clear()
        self.outputs.clear()

        # Create inputs
        self.inputs.new("NodeSocketColor", "Image")
        self.inputs.new("NodeSocketString", "Style")
        self.inputs.new("NodeSocketFloat", "Contrast")
        self.inputs.new("NodeSocketFloat", "Saturation")
        self.inputs.new("NodeSocketFloat", "Gain")
        self.inputs.new("NodeSocketFloat", "Lift")
        self.inputs.new("NodeSocketFloat", "Gamma")

        # Create outputs
        self.outputs.new("NodeSocketColor", "Image")

        # Set default values
        self.inputs["Style"].default_value = "cinematic"
        self.inputs["Contrast"].default_value = 1.0
        self.inputs["Saturation"].default_value = 1.0
        self.inputs["Gain"].default_value = 1.0
        self.inputs["Lift"].default_value = 0.0
        self.inputs["Gamma"].default_value = 1.0

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Color correction
        color_correction = self.nodes.new("CompositorNodeColorCorrection")

        # Position nodes
        input_node.location = (-200, 0)
        color_correction.location = (0, 0)
        output_node.location = (200, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Image"], color_correction.inputs["Image"])
        self.links.new(color_correction.outputs["Image"], output_node.inputs["Image"])

        # Connect parameters
        self.links.new(
            input_node.outputs["Contrast"], color_correction.inputs["Master Contrast"]
        )
        self.links.new(
            input_node.outputs["Saturation"],
            color_correction.inputs["Master Saturation"],
        )
        self.links.new(
            input_node.outputs["Gain"], color_correction.inputs["Master Gain"]
        )
        self.links.new(
            input_node.outputs["Lift"], color_correction.inputs["Master Lift"]
        )
        self.links.new(
            input_node.outputs["Gamma"], color_correction.inputs["Master Gamma"]
        )


# Color grading presets
COLOR_GRADING_PRESETS = {
    "cinematic": {
        "Style": "cinematic",
        "Contrast": 1.2,
        "Saturation": 0.9,
        "Gain": 1.1,
        "Lift": 0.05,
        "Gamma": 1.0,
    },
    "warm": {
        "Style": "warm",
        "Contrast": 1.0,
        "Saturation": 1.3,
        "Gain": 1.2,
        "Lift": 0.1,
        "Gamma": 0.9,
    },
    "cool": {
        "Style": "cool",
        "Contrast": 1.0,
        "Saturation": 1.1,
        "Gain": 0.9,
        "Lift": -0.05,
        "Gamma": 1.1,
    },
    "dramatic": {
        "Style": "dramatic",
        "Contrast": 1.5,
        "Saturation": 1.4,
        "Gain": 1.3,
        "Lift": 0.15,
        "Gamma": 1.0,
    },
    "dreamy": {
        "Style": "dreamy",
        "Contrast": 0.8,
        "Saturation": 1.2,
        "Gain": 1.0,
        "Lift": 0.05,
        "Gamma": 1.0,
    },
    "scientific": {
        "Style": "scientific",
        "Contrast": 1.1,
        "Saturation": 1.0,
        "Gain": 1.0,
        "Lift": 0.0,
        "Gamma": 1.0,
    },
}


def get_color_grading_preset(preset_name: str) -> dict:
    """Get preset for color grading configuration."""
    return COLOR_GRADING_PRESETS.get(preset_name, COLOR_GRADING_PRESETS["cinematic"])


def apply_color_grading_preset(
    node_group: bpy.types.CompositorNodeGroup, preset_name: str
) -> None:
    """Apply color grading preset to node group."""
    preset = get_color_grading_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyColorGradingCompositingGroup)

    bpy.utils.register_class(AlbpyColorGradingCompositingGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyColorGradingCompositingGroup)
