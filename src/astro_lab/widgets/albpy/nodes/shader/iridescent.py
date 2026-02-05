import bpy


class AlbpyIridescentShaderGroup(bpy.types.ShaderNodeGroup):
    bl_idname = "ALBPY_NG_Iridescent"
    bl_label = "Albpy Iridescent Shader Group"

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == "ShaderNodeTree"

    def init(self, context):
        # Clear existing nodes
        self.nodes.clear()
        self.inputs.clear()
        self.outputs.clear()

        # Create inputs
        self.inputs.new("NodeSocketColor", "Base Color")
        self.inputs.new("NodeSocketFloat", "Iridescence Strength")
        self.inputs.new("NodeSocketFloat", "Iridescence Shift")
        self.inputs.new("NodeSocketFloat", "Metallic")
        self.inputs.new("NodeSocketFloat", "Roughness")

        # Create outputs
        self.outputs.new("NodeSocketShader", "Shader")

        # Set default values
        self.inputs["Base Color"].default_value = (0.8, 0.2, 0.8, 1.0)
        self.inputs["Iridescence Strength"].default_value = 1.0
        self.inputs["Iridescence Shift"].default_value = 0.0
        self.inputs["Metallic"].default_value = 0.8
        self.inputs["Roughness"].default_value = 0.1

        # Create nodes
        output = self.nodes.new("NodeGroupOutput")
        bsdf = self.nodes.new("ShaderNodeBsdfPrincipled")
        fresnel = self.nodes.new("ShaderNodeFresnel")
        mix_rgb = self.nodes.new("ShaderNodeMixRGB")

        # Position nodes
        fresnel.location = (-400, 200)
        mix_rgb.location = (-200, 0)
        bsdf.location = (0, 0)
        output.location = (200, 0)

        # Connect nodes
        self.links.new(self.inputs["Base Color"], mix_rgb.inputs["Color1"])
        self.links.new(fresnel.outputs["Fac"], mix_rgb.inputs["Fac"])
        self.links.new(mix_rgb.outputs["Color"], bsdf.inputs["Base Color"])
        self.links.new(self.inputs["Iridescence Strength"], bsdf.inputs["Iridescence"])
        self.links.new(self.inputs["Iridescence Shift"], bsdf.inputs["Iridescence IOR"])
        self.links.new(self.inputs["Metallic"], bsdf.inputs["Metallic"])
        self.links.new(self.inputs["Roughness"], bsdf.inputs["Roughness"])
        self.links.new(bsdf.outputs["BSDF"], output.inputs["Shader"])

        # Set additional properties
        bsdf.inputs["Specular"].default_value = 1.0
        fresnel.inputs["IOR"].default_value = 1.5
        mix_rgb.blend_type = "ADD"
        mix_rgb.inputs["Color2"].default_value = (0.3, 0.3, 0.3, 1.0)


def register():
    # REMOVED: bpy.utils.register_class(AlbpyIridescentShaderGroup)

    bpy.utils.register_class(AlbpyIridescentShaderGroup)


# Iridescent material presets
IRIDESCENT_PRESETS = {
    "luxury_teal": {
        "Base Color": (0.0, 0.8, 0.6, 1.0),
        "Iridescence Strength": 0.8,
        "Iridescence Shift": 0.2,
        "Metallic": 0.8,
        "Roughness": 0.1,
    },
    "cosmic_purple": {
        "Base Color": (0.8, 0.2, 0.8, 1.0),
        "Iridescence Strength": 1.0,
        "Iridescence Shift": 0.0,
        "Metallic": 0.9,
        "Roughness": 0.05,
    },
    "aurora_green": {
        "Base Color": (0.2, 0.9, 0.4, 1.0),
        "Iridescence Strength": 0.6,
        "Iridescence Shift": 0.3,
        "Metallic": 0.7,
        "Roughness": 0.15,
    },
    "solar_gold": {
        "Base Color": (1.0, 0.8, 0.2, 1.0),
        "Iridescence Strength": 0.9,
        "Iridescence Shift": 0.1,
        "Metallic": 0.95,
        "Roughness": 0.08,
    },
}


def get_iridescent_preset(preset_name: str) -> dict:
    """Get preset for iridescent material."""
    return IRIDESCENT_PRESETS.get(preset_name, IRIDESCENT_PRESETS["luxury_teal"])


def apply_iridescent_preset(material: bpy.types.Material, preset_name: str) -> None:
    """Apply iridescent preset to material."""
    preset = get_iridescent_preset(preset_name)

    if not material.use_nodes:
        material.use_nodes = True

    # Find the iridescent node group
    for node in material.node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_NG_Iridescent" in node.node_tree.name:
                # Apply preset parameters
                for param_name, value in preset.items():
                    if param_name in node.inputs:
                        node.inputs[param_name].default_value = value
                break


def unregister():
    bpy.utils.unregister_class(AlbpyIridescentShaderGroup)
