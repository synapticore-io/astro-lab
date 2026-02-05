import bpy


class AlbpyArtisticFiltersCompositingGroup(bpy.types.CompositorNodeGroup):
    bl_idname = "ALBPY_NG_ArtisticFilters"
    bl_label = "Albpy Artistic Filters Compositing Group"

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
        self.inputs.new("NodeSocketString", "Filter Type")
        self.inputs.new("NodeSocketFloat", "Film Grain Intensity")
        self.inputs.new("NodeSocketFloat", "Chromatic Aberration")
        self.inputs.new("NodeSocketFloat", "Noise Scale")
        self.inputs.new("NodeSocketFloat", "Noise Detail")

        # Create outputs
        self.outputs.new("NodeSocketColor", "Image")

        # Set default values
        self.inputs["Filter Type"].default_value = "film_grain"
        self.inputs["Film Grain Intensity"].default_value = 0.1
        self.inputs["Chromatic Aberration"].default_value = 0.02
        self.inputs["Noise Scale"].default_value = 100.0
        self.inputs["Noise Detail"].default_value = 2.0

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Film grain effect
        noise = self.nodes.new("CompositorNodeTexNoise")
        grain_mix = self.nodes.new("CompositorNodeMixRGB")
        grain_mix.blend_type = "OVERLAY"

        # Chromatic aberration effect
        separate = self.nodes.new("CompositorNodeSepRGBA")
        lens_red = self.nodes.new("CompositorNodeLensDist")
        lens_blue = self.nodes.new("CompositorNodeLensDist")
        combine = self.nodes.new("CompositorNodeCombRGBA")

        # Final mix
        final_mix = self.nodes.new("CompositorNodeMixRGB")
        final_mix.blend_type = "OVERLAY"

        # Position nodes
        input_node.location = (-1000, 0)
        noise.location = (-800, 200)
        grain_mix.location = (-600, 100)
        separate.location = (-800, -200)
        lens_red.location = (-600, -100)
        lens_blue.location = (-600, -300)
        combine.location = (-400, -200)
        final_mix.location = (-200, 0)
        output_node.location = (0, 0)

        # Connect film grain
        self.links.new(input_node.outputs["Image"], grain_mix.inputs["Image1"])
        self.links.new(noise.outputs["Color"], grain_mix.inputs["Image2"])

        # Connect chromatic aberration
        self.links.new(input_node.outputs["Image"], separate.inputs["Image"])
        self.links.new(separate.outputs["R"], lens_red.inputs["Image"])
        self.links.new(separate.outputs["B"], lens_blue.inputs["Image"])
        self.links.new(separate.outputs["G"], combine.inputs["G"])
        self.links.new(lens_red.outputs["Image"], combine.inputs["R"])
        self.links.new(lens_blue.outputs["Image"], combine.inputs["B"])

        # Connect final mix
        self.links.new(grain_mix.outputs["Image"], final_mix.inputs["Image1"])
        self.links.new(combine.outputs["Image"], final_mix.inputs["Image2"])
        self.links.new(final_mix.outputs["Image"], output_node.inputs["Image"])

        # Connect parameters
        self.links.new(input_node.outputs["Noise Scale"], noise.inputs["Scale"])
        self.links.new(input_node.outputs["Noise Detail"], noise.inputs["Detail"])
        self.links.new(
            input_node.outputs["Film Grain Intensity"], grain_mix.inputs["Fac"]
        )
        self.links.new(
            input_node.outputs["Chromatic Aberration"], lens_red.inputs["Distort"]
        )
        self.links.new(
            input_node.outputs["Chromatic Aberration"], lens_blue.inputs["Distort"]
        )


# Artistic filter presets
ARTISTIC_FILTER_PRESETS = {
    "film_grain": {
        "Filter Type": "film_grain",
        "Film Grain Intensity": 0.1,
        "Chromatic Aberration": 0.0,
        "Noise Scale": 100.0,
        "Noise Detail": 2.0,
    },
    "chromatic_aberration": {
        "Filter Type": "chromatic_aberration",
        "Film Grain Intensity": 0.0,
        "Chromatic Aberration": 0.02,
        "Noise Scale": 100.0,
        "Noise Detail": 2.0,
    },
    "cinematic": {
        "Filter Type": "cinematic",
        "Film Grain Intensity": 0.15,
        "Chromatic Aberration": 0.01,
        "Noise Scale": 80.0,
        "Noise Detail": 3.0,
    },
    "vintage": {
        "Filter Type": "vintage",
        "Film Grain Intensity": 0.2,
        "Chromatic Aberration": 0.03,
        "Noise Scale": 120.0,
        "Noise Detail": 1.5,
    },
    "clean": {
        "Filter Type": "clean",
        "Film Grain Intensity": 0.0,
        "Chromatic Aberration": 0.0,
        "Noise Scale": 100.0,
        "Noise Detail": 2.0,
    },
}


def get_artistic_filter_preset(preset_name: str) -> dict:
    """Get preset for artistic filter configuration."""
    return ARTISTIC_FILTER_PRESETS.get(
        preset_name, ARTISTIC_FILTER_PRESETS["film_grain"]
    )


def apply_artistic_filter_preset(
    node_group: bpy.types.CompositorNodeGroup, preset_name: str
) -> None:
    """Apply artistic filter preset to node group."""
    preset = get_artistic_filter_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyArtisticFiltersCompositingGroup)

    bpy.utils.register_class(AlbpyArtisticFiltersCompositingGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyArtisticFiltersCompositingGroup)
