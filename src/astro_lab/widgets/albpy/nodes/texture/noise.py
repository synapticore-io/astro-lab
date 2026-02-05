import bpy


class AlbpyNoiseTextureGroup(bpy.types.TextureNodeGroup):
    bl_idname = "ALBPY_NG_NoiseTexture"
    bl_label = "Albpy Noise Texture Group"

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == "TextureNodeTree"

    def init(self, context):
        noise = self.nodes.new("TextureNodeNoise")
        output = self.nodes.new("NodeGroupOutput")
        self.links.new(noise.outputs["Color"], output.inputs[0])


def register():
    # REMOVED: bpy.utils.register_class(AlbpyNoiseTextureGroup)

    bpy.utils.register_class(AlbpyNoiseTextureGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyNoiseTextureGroup)
