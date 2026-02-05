import bpy


class AlbpyHolographicShaderGroup(bpy.types.ShaderNodeGroup):
    bl_idname = "ALBPY_NG_Holographic"
    bl_label = "Albpy Holographic Shader Group"

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
        self.inputs.new("NodeSocketFloat", "Hologram Strength")
        self.inputs.new("NodeSocketFloat", "Scan Speed")
        self.inputs.new("NodeSocketFloat", "Transparency")

        # Create outputs
        self.outputs.new("NodeSocketShader", "Shader")

        # Set default values
        self.inputs["Base Color"].default_value = (0.2, 0.8, 1.0, 1.0)
        self.inputs["Hologram Strength"].default_value = 1.0
        self.inputs["Scan Speed"].default_value = 1.0
        self.inputs["Transparency"].default_value = 0.7

        # Create nodes
        output = self.nodes.new("NodeGroupOutput")
        emission = self.nodes.new("ShaderNodeEmission")
        transparent = self.nodes.new("ShaderNodeBsdfTransparent")
        mix_shader = self.nodes.new("ShaderNodeMixShader")

        # Create scanning line effect
        tex_coord = self.nodes.new("ShaderNodeTexCoord")
        wave = self.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "SAW"
        ramp = self.nodes.new("ShaderNodeValToRGB")
        noise = self.nodes.new("ShaderNodeTexNoise")
        mix_noise = self.nodes.new("ShaderNodeMixRGB")

        # Position nodes
        tex_coord.location = (-800, 0)
        wave.location = (-600, 0)
        ramp.location = (-400, 0)
        noise.location = (-600, -200)
        mix_noise.location = (-200, 0)
        emission.location = (0, 100)
        transparent.location = (0, -100)
        mix_shader.location = (200, 0)
        output.location = (400, 0)

        # Connect nodes
        self.links.new(tex_coord.outputs["Generated"], wave.inputs["Vector"])
        self.links.new(wave.outputs["Color"], ramp.inputs["Fac"])
        self.links.new(noise.outputs["Color"], mix_noise.inputs["Color2"])
        self.links.new(ramp.outputs["Color"], mix_noise.inputs["Color1"])
        self.links.new(mix_noise.outputs["Color"], emission.inputs["Color"])
        self.links.new(emission.outputs["Emission"], mix_shader.inputs[1])
        self.links.new(transparent.outputs["BSDF"], mix_shader.inputs[2])
        self.links.new(mix_shader.outputs["Shader"], output.inputs["Shader"])

        # Set properties
        wave.inputs["Scale"].default_value = 50.0
        wave.inputs["Distortion"].default_value = 0.5
        ramp.color_ramp.elements[0].position = 0.4
        ramp.color_ramp.elements[1].position = 0.6
        noise.inputs["Scale"].default_value = 100.0
        noise.inputs["Detail"].default_value = 10.0
        mix_noise.blend_type = "MULTIPLY"
        mix_noise.inputs["Fac"].default_value = 0.3
        emission.inputs["Strength"].default_value = 2.0
        mix_shader.inputs["Fac"].default_value = 0.7

        # Connect inputs
        self.links.new(self.inputs["Base Color"], emission.inputs["Color"])
        self.links.new(self.inputs["Hologram Strength"], emission.inputs["Strength"])
        self.links.new(self.inputs["Scan Speed"], wave.inputs["Scale"])
        self.links.new(self.inputs["Transparency"], mix_shader.inputs["Fac"])


def register():
    # REMOVED: bpy.utils.register_class(AlbpyHolographicShaderGroup)

    bpy.utils.register_class(AlbpyHolographicShaderGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyHolographicShaderGroup)
