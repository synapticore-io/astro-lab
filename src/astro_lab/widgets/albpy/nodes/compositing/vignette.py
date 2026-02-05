import bpy


class AlbpyVignetteCompositingGroup(bpy.types.CompositorNodeGroup):
    bl_idname = "ALBPY_NG_Vignette"
    bl_label = "Albpy Vignette Compositing Group"

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
        self.inputs.new("NodeSocketFloat", "Intensity")
        self.inputs.new("NodeSocketFloat", "Radius")
        self.inputs.new("NodeSocketFloat", "Softness")

        # Create outputs
        self.outputs.new("NodeSocketColor", "Image")

        # Set default values
        self.inputs["Intensity"].default_value = 0.3
        self.inputs["Radius"].default_value = 0.8
        self.inputs["Softness"].default_value = 0.2

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Vignette using radial gradient
        radial = self.nodes.new("CompositorNodeEllipseMask")

        # Invert mask for vignette
        invert = self.nodes.new("CompositorNodeInvert")

        # Mix with original image
        mix = self.nodes.new("CompositorNodeMixRGB")
        mix.blend_type = "MULTIPLY"

        # Position nodes
        input_node.location = (-600, 0)
        radial.location = (-400, 200)
        invert.location = (-200, 200)
        mix.location = (0, 0)
        output_node.location = (200, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Image"], mix.inputs["Image1"])
        self.links.new(radial.outputs["Mask"], invert.inputs["Color"])
        self.links.new(invert.outputs["Color"], mix.inputs["Image2"])
        self.links.new(mix.outputs["Image"], output_node.inputs["Image"])

        # Connect parameters
        self.links.new(input_node.outputs["Radius"], radial.inputs["Width"])
        self.links.new(input_node.outputs["Radius"], radial.inputs["Height"])
        self.links.new(input_node.outputs["Softness"], radial.inputs["Mask"])
        self.links.new(input_node.outputs["Intensity"], mix.inputs["Fac"])


# Vignette presets
VIGNETTE_PRESETS = {
    "subtle": {
        "Intensity": 0.2,
        "Radius": 0.8,
        "Softness": 0.3,
    },
    "medium": {
        "Intensity": 0.4,
        "Radius": 0.7,
        "Softness": 0.2,
    },
    "strong": {
        "Intensity": 0.6,
        "Radius": 0.6,
        "Softness": 0.1,
    },
    "cinematic": {
        "Intensity": 0.5,
        "Radius": 0.65,
        "Softness": 0.25,
    },
}


def get_vignette_preset(preset_name: str) -> dict:
    """Get preset for vignette configuration."""
    return VIGNETTE_PRESETS.get(preset_name, VIGNETTE_PRESETS["medium"])


def apply_vignette_preset(
    node_group: bpy.types.CompositorNodeGroup, preset_name: str
) -> None:
    """Apply vignette preset to node group."""
    preset = get_vignette_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyVignetteCompositingGroup)

    bpy.utils.register_class(AlbpyVignetteCompositingGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyVignetteCompositingGroup)
