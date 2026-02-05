import bpy


class AlbpySubsurfaceShaderGroup(bpy.types.ShaderNodeGroup):
    bl_idname = "ALBPY_NG_Subsurface"
    bl_label = "Albpy Subsurface Shader Group"

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == "ShaderNodeTree"

    def init(self, context):
        self.inputs.new("NodeSocketColor", "Color")
        self.inputs.new("NodeSocketFloat", "Subsurface")
        self.inputs.new("NodeSocketVector", "Subsurface Radius")
        principled = self.nodes.new("ShaderNodeBsdfPrincipled")
        output = self.nodes.new("NodeGroupOutput")
        self.outputs.new("NodeSocketShader", "Shader")
        self.links.new(self.inputs["Color"], principled.inputs["Base Color"])
        self.links.new(self.inputs["Subsurface"], principled.inputs["Subsurface"])
        self.links.new(
            self.inputs["Subsurface Radius"], principled.inputs["Subsurface Radius"]
        )
        self.links.new(principled.outputs["BSDF"], output.inputs["Shader"])


def register():
    # REMOVED: bpy.utils.register_class(AlbpySubsurfaceShaderGroup)

    bpy.utils.register_class(AlbpySubsurfaceShaderGroup)


def unregister():
    bpy.utils.unregister_class(AlbpySubsurfaceShaderGroup)
