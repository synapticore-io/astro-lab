import bpy


class AlbpyStarGlowCompositingGroup(bpy.types.CompositorNodeGroup):
    bl_idname = "ALBPY_NG_StarGlow"
    bl_label = "Albpy Star Glow Compositing Group"

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
        self.inputs.new("NodeSocketFloat", "Glow Intensity")
        self.inputs.new("NodeSocketFloat", "Glow Size")
        self.inputs.new("NodeSocketFloat", "Cross Pattern")
        self.inputs.new("NodeSocketFloat", "Angle Offset")

        # Create outputs
        self.outputs.new("NodeSocketColor", "Image")

        # Set default values
        self.inputs["Glow Intensity"].default_value = 1.0
        self.inputs["Glow Size"].default_value = 9.0
        self.inputs["Cross Pattern"].default_value = 0.5
        self.inputs["Angle Offset"].default_value = 0.0

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # First glow effect
        glow1 = self.nodes.new("CompositorNodeGlare")
        glow1.glare_type = "STREAKS"
        glow1.quality = "HIGH"

        # Second glow for cross pattern
        glow2 = self.nodes.new("CompositorNodeGlare")
        glow2.glare_type = "STREAKS"
        glow2.quality = "HIGH"

        # Mix glows
        mix = self.nodes.new("CompositorNodeMixRGB")
        mix.blend_type = "ADD"

        # Position nodes
        input_node.location = (-800, 0)
        glow1.location = (-600, 100)
        glow2.location = (-600, -100)
        mix.location = (-400, 0)
        output_node.location = (-200, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Image"], glow1.inputs["Image"])
        self.links.new(input_node.outputs["Image"], glow2.inputs["Image"])
        self.links.new(glow1.outputs["Image"], mix.inputs["Image1"])
        self.links.new(glow2.outputs["Image"], mix.inputs["Image2"])
        self.links.new(mix.outputs["Image"], output_node.inputs["Image"])

        # Connect parameters
        self.links.new(input_node.outputs["Glow Intensity"], glow1.inputs["Mix"])
        self.links.new(input_node.outputs["Glow Size"], glow1.inputs["Size"])
        self.links.new(input_node.outputs["Cross Pattern"], glow2.inputs["Mix"])
        self.links.new(input_node.outputs["Glow Size"], glow2.inputs["Size"])
        self.links.new(input_node.outputs["Angle Offset"], glow1.inputs["Angle Offset"])
        self.links.new(input_node.outputs["Angle Offset"], glow2.inputs["Angle Offset"])


# Star glow presets
STAR_GLOW_PRESETS = {
    "subtle": {
        "Glow Intensity": 0.6,
        "Glow Size": 7,
        "Cross Pattern": 0.3,
        "Angle Offset": 0.0,
    },
    "medium": {
        "Glow Intensity": 1.0,
        "Glow Size": 9,
        "Cross Pattern": 0.5,
        "Angle Offset": 0.0,
    },
    "strong": {
        "Glow Intensity": 1.5,
        "Glow Size": 13,
        "Cross Pattern": 0.7,
        "Angle Offset": 0.0,
    },
    "cross": {
        "Glow Intensity": 1.2,
        "Glow Size": 11,
        "Cross Pattern": 1.0,
        "Angle Offset": 1.5708,  # 90 degrees
    },
    "cinematic": {
        "Glow Intensity": 1.2,
        "Glow Size": 11,
        "Cross Pattern": 0.6,
        "Angle Offset": 0.0,
    },
}


def get_star_glow_preset(preset_name: str) -> dict:
    """Get preset for star glow configuration."""
    return STAR_GLOW_PRESETS.get(preset_name, STAR_GLOW_PRESETS["medium"])


def apply_star_glow_preset(
    node_group: bpy.types.CompositorNodeGroup, preset_name: str
) -> None:
    """Apply star glow preset to node group."""
    preset = get_star_glow_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyStarGlowCompositingGroup)

    bpy.utils.register_class(AlbpyStarGlowCompositingGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyStarGlowCompositingGroup)
