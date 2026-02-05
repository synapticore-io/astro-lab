import bpy


class AlbpyPlanetShaderGroup(bpy.types.ShaderNodeGroup):
    bl_idname = "ALBPY_NG_Planet"
    bl_label = "Albpy Planetary Surface Shader Group"

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
        self.inputs.new("NodeSocketFloat", "Roughness")
        self.inputs.new("NodeSocketFloat", "Metallic")
        self.inputs.new("NodeSocketFloat", "Specular")
        self.inputs.new("NodeSocketFloat", "Transmission")
        self.inputs.new("NodeSocketString", "Planet Type")
        self.inputs.new("NodeSocketFloat", "Surface Detail")
        self.inputs.new("NodeSocketFloat", "Band Scale")

        # Create outputs
        self.outputs.new("NodeSocketShader", "Shader")

        # Set default values
        self.inputs["Base Color"].default_value = (0.4, 0.3, 0.2, 1.0)
        self.inputs["Roughness"].default_value = 0.8
        self.inputs["Metallic"].default_value = 0.1
        self.inputs["Specular"].default_value = 0.3
        self.inputs["Transmission"].default_value = 0.0
        self.inputs["Planet Type"].default_value = "terrestrial"
        self.inputs["Surface Detail"].default_value = 5.0
        self.inputs["Band Scale"].default_value = 2.0

        # Create nodes
        principled = self.nodes.new("ShaderNodeBsdfPrincipled")
        output = self.nodes.new("NodeGroupOutput")

        # Surface texture variation
        noise = self.nodes.new("ShaderNodeTexNoise")
        color_mix = self.nodes.new("ShaderNodeMixRGB")
        color_mix.blend_type = "MULTIPLY"

        # Gas giant bands
        coords = self.nodes.new("ShaderNodeTexCoord")
        mapping = self.nodes.new("ShaderNodeMapping")
        wave = self.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        band_ramp = self.nodes.new("ShaderNodeValToRGB")
        band_mix = self.nodes.new("ShaderNodeMixRGB")

        # Position nodes
        coords.location = (-800, -400)
        mapping.location = (-600, -400)
        wave.location = (-400, -400)
        band_ramp.location = (-200, -400)
        band_mix.location = (-200, -200)
        noise.location = (-400, 0)
        color_mix.location = (-200, 0)
        principled.location = (0, 0)
        output.location = (200, 0)

        # Connect basic nodes
        self.links.new(self.inputs["Base Color"], color_mix.inputs["Color1"])
        self.links.new(noise.outputs["Color"], color_mix.inputs["Color2"])
        self.links.new(color_mix.outputs["Color"], band_mix.inputs["Color1"])
        self.links.new(band_mix.outputs["Color"], principled.inputs["Base Color"])
        self.links.new(self.inputs["Roughness"], principled.inputs["Roughness"])
        self.links.new(self.inputs["Metallic"], principled.inputs["Metallic"])
        self.links.new(self.inputs["Specular"], principled.inputs["Specular"])
        self.links.new(self.inputs["Transmission"], principled.inputs["Transmission"])
        self.links.new(principled.outputs["BSDF"], output.inputs["Shader"])

        # Connect gas giant bands
        self.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
        self.links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
        self.links.new(wave.outputs["Color"], band_ramp.inputs["Fac"])
        self.links.new(band_ramp.outputs["Color"], band_mix.inputs["Color2"])

        # Set band ramp colors
        band_ramp.color_ramp.elements[0].color = (0.8, 0.6, 0.4, 1.0)
        band_ramp.color_ramp.elements[1].color = (0.4, 0.3, 0.2, 1.0)


def register():
    # REMOVED: bpy.utils.register_class(AlbpyPlanetShaderGroup)

    bpy.utils.register_class(AlbpyPlanetShaderGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyPlanetShaderGroup)
