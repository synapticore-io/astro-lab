import bpy
from bpy.types import Operator


class ALBPY_OT_CreatePoint(Operator):
    bl_idname = "albpy.create_point"
    bl_label = "Create Point"
    bl_description = "Create a single point object (icosphere)"

    size: bpy.props.FloatProperty(name="Size", default=0.1, min=0.01, max=10.0)

    def execute(self, context):
        bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0), subdivisions=1)
        point = bpy.context.active_object
        point.name = "Point"
        point.scale = (self.size, self.size, self.size)
        # TODO: Assign emission material via Node Group/Presets if available
        self.report({"INFO"}, f"Created point: {point.name}")
        return {"FINISHED"}


def register():
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreatePoint)

    bpy.utils.register_class(ALBPY_OT_CreatePoint)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreatePoint)
