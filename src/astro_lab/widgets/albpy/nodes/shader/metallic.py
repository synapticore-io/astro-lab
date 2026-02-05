import bpy


class AlbpyMetallicShaderGroup(bpy.types.ShaderNodeGroup):
    bl_idname = "ALBPY_NG_Metallic"
    bl_label = "Albpy Metallic Shader Group"

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
        self.inputs.new("NodeSocketFloat", "Metallic")
        self.inputs.new("NodeSocketFloat", "Roughness")
        self.inputs.new("NodeSocketFloat", "Anisotropic")
        self.inputs.new("NodeSocketFloat", "Anisotropic Rotation")

        # Create outputs
        self.outputs.new("NodeSocketShader", "Shader")

        # Set default values
        self.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        self.inputs["Metallic"].default_value = 1.0
        self.inputs["Roughness"].default_value = 0.2
        self.inputs["Anisotropic"].default_value = 0.0
        self.inputs["Anisotropic Rotation"].default_value = 0.0

        # Create nodes
        output = self.nodes.new("NodeGroupOutput")
        bsdf = self.nodes.new("ShaderNodeBsdfPrincipled")

        # Position nodes
        bsdf.location = (0, 0)
        output.location = (200, 0)

        # Connect nodes
        self.links.new(bsdf.outputs["BSDF"], output.inputs["Shader"])

        # Set properties
        bsdf.inputs["Specular"].default_value = 0.5

        # Connect inputs
        self.links.new(self.inputs["Color"], bsdf.inputs["Base Color"])
        self.links.new(self.inputs["Metallic"], bsdf.inputs["Metallic"])
        self.links.new(self.inputs["Roughness"], bsdf.inputs["Roughness"])
        self.links.new(self.inputs["Anisotropic"], bsdf.inputs["Anisotropic"])
        self.links.new(
            self.inputs["Anisotropic Rotation"], bsdf.inputs["Anisotropic Rotation"]
        )


def register():
    # REMOVED: bpy.utils.register_class(AlbpyMetallicShaderGroup)

    bpy.utils.register_class(AlbpyMetallicShaderGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyMetallicShaderGroup)
