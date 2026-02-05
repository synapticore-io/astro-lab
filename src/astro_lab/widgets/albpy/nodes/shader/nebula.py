import bpy


class AlbpyNebulaShaderGroup(bpy.types.ShaderNodeGroup):
    bl_idname = "ALBPY_NG_Nebula"
    bl_label = "Albpy Nebula Shader Group"

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == "ShaderNodeTree"

    def init(self, context):
        # Clear existing nodes
        self.nodes.clear()
        self.inputs.clear()
        self.outputs.clear()

        # Create inputs
        self.inputs.new("NodeSocketColor", "Primary Color")
        self.inputs.new("NodeSocketColor", "Secondary Color")
        self.inputs.new("NodeSocketFloat", "Emission Strength")
        self.inputs.new("NodeSocketFloat", "Density")
        self.inputs.new("NodeSocketFloat", "Density Variation")
        self.inputs.new("NodeSocketString", "Emission Lines")
        self.inputs.new("NodeSocketFloat", "Noise Scale 1")
        self.inputs.new("NodeSocketFloat", "Noise Scale 2")

        # Create outputs
        self.outputs.new("NodeSocketShader", "Shader")
        self.outputs.new("NodeSocketShader", "Volume")

        # Set default values
        self.inputs["Primary Color"].default_value = (0.8, 0.2, 0.2, 1.0)
        self.inputs["Secondary Color"].default_value = (0.2, 0.8, 0.3, 1.0)
        self.inputs["Emission Strength"].default_value = 2.0
        self.inputs["Density"].default_value = 0.5
        self.inputs["Density Variation"].default_value = 0.5
        self.inputs["Emission Lines"].default_value = "H_alpha,O_III"
        self.inputs["Noise Scale 1"].default_value = 1.0
        self.inputs["Noise Scale 2"].default_value = 5.0

        # Create nodes
        emission = self.nodes.new("ShaderNodeVolumeEmission")
        output = self.nodes.new("NodeGroupOutput")

        # Density variation with multiple noise scales
        noise1 = self.nodes.new("ShaderNodeTexNoise")
        noise1.inputs["Detail"].default_value = 8.0

        noise2 = self.nodes.new("ShaderNodeTexNoise")
        noise2.inputs["Detail"].default_value = 4.0

        # Combine noises for complex structure
        multiply = self.nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"

        # Color ramp for density falloff
        ramp = self.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.1
        ramp.color_ramp.elements[1].position = 0.9

        # Emission strength variation
        strength_variation = self.nodes.new("ShaderNodeMath")
        strength_variation.operation = "MULTIPLY"
        strength_variation.inputs[1].default_value = 2.0

        # Color mixing for emission lines
        color_mix = self.nodes.new("ShaderNodeMixRGB")
        color_mix.blend_type = "ADD"

        # Position nodes
        noise1.location = (-800, 0)
        noise2.location = (-800, -200)
        multiply.location = (-600, -100)
        ramp.location = (-400, 0)
        strength_variation.location = (-400, -300)
        color_mix.location = (-200, 0)
        emission.location = (0, 0)
        output.location = (200, 0)

        # Connect nodes
        self.links.new(self.inputs["Noise Scale 1"], noise1.inputs["Scale"])
        self.links.new(self.inputs["Noise Scale 2"], noise2.inputs["Scale"])
        self.links.new(noise1.outputs["Fac"], multiply.inputs[0])
        self.links.new(noise2.outputs["Fac"], multiply.inputs[1])
        self.links.new(multiply.outputs["Value"], ramp.inputs["Fac"])
        self.links.new(ramp.outputs["Color"], emission.inputs["Density"])
        self.links.new(multiply.outputs["Value"], strength_variation.inputs[0])
        self.links.new(strength_variation.outputs["Value"], emission.inputs["Strength"])
        self.links.new(self.inputs["Primary Color"], color_mix.inputs["Color1"])
        self.links.new(self.inputs["Secondary Color"], color_mix.inputs["Color2"])
        self.links.new(self.inputs["Density Variation"], color_mix.inputs["Fac"])
        self.links.new(color_mix.outputs["Color"], emission.inputs["Color"])
        self.links.new(emission.outputs["Volume"], output.inputs["Volume"])


def register():
    # REMOVED: bpy.utils.register_class(AlbpyNebulaShaderGroup)

    bpy.utils.register_class(AlbpyNebulaShaderGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyNebulaShaderGroup)
