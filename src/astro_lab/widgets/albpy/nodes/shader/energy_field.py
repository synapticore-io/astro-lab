import bpy


class AlbpyEnergyFieldShaderGroup(bpy.types.ShaderNodeGroup):
    bl_idname = "ALBPY_NG_EnergyField"
    bl_label = "Albpy Energy Field Shader Group"

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == "ShaderNodeTree"

    def init(self, context):
        # Clear existing nodes
        self.nodes.clear()
        self.inputs.clear()
        self.outputs.clear()

        # Create inputs
        self.inputs.new("NodeSocketColor", "Color")
        self.inputs.new("NodeSocketFloat", "Energy Strength")
        self.inputs.new("NodeSocketFloat", "Pulse Speed")
        self.inputs.new("NodeSocketFloat", "Noise Scale")

        # Create outputs
        self.outputs.new("NodeSocketShader", "Shader")

        # Set default values
        self.inputs["Color"].default_value = (0.2, 0.8, 1.0, 1.0)
        self.inputs["Energy Strength"].default_value = 2.0
        self.inputs["Pulse Speed"].default_value = 1.0
        self.inputs["Noise Scale"].default_value = 5.0

        # Create nodes
        output = self.nodes.new("NodeGroupOutput")
        emission = self.nodes.new("ShaderNodeEmission")
        sine = self.nodes.new("ShaderNodeMath")
        scale = self.nodes.new("ShaderNodeMath")
        offset = self.nodes.new("ShaderNodeMath")
        noise = self.nodes.new("ShaderNodeTexNoise")
        mix = self.nodes.new("ShaderNodeMixRGB")

        # Position nodes
        sine.location = (-600, 0)
        scale.location = (-400, 0)
        offset.location = (-200, 0)
        noise.location = (-600, -200)
        mix.location = (0, 0)
        emission.location = (200, 0)
        output.location = (400, 0)

        # Connect nodes
        self.links.new(sine.outputs[0], scale.inputs[0])
        self.links.new(scale.outputs[0], offset.inputs[0])
        self.links.new(noise.outputs["Color"], mix.inputs["Color2"])
        self.links.new(offset.outputs[0], mix.inputs["Color1"])
        self.links.new(mix.outputs["Color"], emission.inputs["Color"])
        self.links.new(emission.outputs["Emission"], output.inputs["Shader"])

        # Set properties
        sine.operation = "SINE"
        sine.inputs[1].default_value = 1.0
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = 0.5
        offset.operation = "ADD"
        offset.inputs[1].default_value = 0.5
        noise.inputs["Detail"].default_value = 8.0
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 0.3
        emission.inputs["Strength"].default_value = 2.0

        # Connect inputs
        self.links.new(self.inputs["Color"], emission.inputs["Color"])
        self.links.new(self.inputs["Energy Strength"], emission.inputs["Strength"])
        self.links.new(self.inputs["Pulse Speed"], sine.inputs[1])
        self.links.new(self.inputs["Noise Scale"], noise.inputs["Scale"])


def register():
    # REMOVED: bpy.utils.register_class(AlbpyEnergyFieldShaderGroup)

    bpy.utils.register_class(AlbpyEnergyFieldShaderGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyEnergyFieldShaderGroup)
