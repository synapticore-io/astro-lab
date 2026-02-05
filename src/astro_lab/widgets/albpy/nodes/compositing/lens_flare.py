import bpy


class AlbpyLensFlareCompositingGroup(bpy.types.CompositorNodeGroup):
    bl_idname = "ALBPY_NG_LensFlare"
    bl_label = "Albpy Lens Flare Compositing Group"

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
        self.inputs.new("NodeSocketString", "Flare Type")
        self.inputs.new("NodeSocketFloat", "Intensity")
        self.inputs.new("NodeSocketFloat", "Distortion")
        self.inputs.new("NodeSocketFloat", "Glow Mix")

        # Create outputs
        self.outputs.new("NodeSocketColor", "Image")

        # Set default values
        self.inputs["Flare Type"].default_value = "stellar"
        self.inputs["Intensity"].default_value = 1.0
        self.inputs["Distortion"].default_value = 0.02
        self.inputs["Glow Mix"].default_value = 0.3

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Lens distortion for flare
        lens_distortion = self.nodes.new("CompositorNodeLensDist")

        # Glow effect
        glow = self.nodes.new("CompositorNodeGlare")
        glow.glare_type = "FOG_GLOW"
        glow.quality = "HIGH"
        glow.size = 9

        # Color correction for flare type
        color_correction = self.nodes.new("CompositorNodeColorCorrection")

        # Position nodes
        input_node.location = (-600, 0)
        lens_distortion.location = (-400, 0)
        glow.location = (-200, 0)
        color_correction.location = (0, 0)
        output_node.location = (200, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Image"], lens_distortion.inputs["Image"])
        self.links.new(lens_distortion.outputs["Image"], glow.inputs["Image"])
        self.links.new(glow.outputs["Image"], color_correction.inputs["Image"])
        self.links.new(color_correction.outputs["Image"], output_node.inputs["Image"])

        # Connect parameters
        self.links.new(
            input_node.outputs["Distortion"], lens_distortion.inputs["Distort"]
        )
        self.links.new(input_node.outputs["Glow Mix"], glow.inputs["Mix"])


# Lens flare presets
LENS_FLARE_PRESETS = {
    "stellar": {
        "Flare Type": "stellar",
        "Intensity": 1.0,
        "Distortion": 0.02,
        "Glow Mix": 0.3,
    },
    "nebula": {
        "Flare Type": "nebula",
        "Intensity": 1.2,
        "Distortion": 0.03,
        "Glow Mix": 0.4,
    },
    "galactic": {
        "Flare Type": "galactic",
        "Intensity": 0.8,
        "Distortion": 0.015,
        "Glow Mix": 0.25,
    },
}


def get_lens_flare_preset(preset_name: str) -> dict:
    """Get preset for lens flare configuration."""
    return LENS_FLARE_PRESETS.get(preset_name, LENS_FLARE_PRESETS["stellar"])


def apply_lens_flare_preset(
    node_group: bpy.types.CompositorNodeGroup, preset_name: str
) -> None:
    """Apply lens flare preset to node group."""
    preset = get_lens_flare_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyLensFlareCompositingGroup)

    bpy.utils.register_class(AlbpyLensFlareCompositingGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyLensFlareCompositingGroup)
