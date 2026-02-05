import bpy
from bpy.types import Operator


class ALBPY_OT_CreatePlanet(Operator):
    bl_idname = "albpy.create_planet"
    bl_label = "Create Planet"
    bl_description = "Create a planet object using Node Groups or utilities"

    planet_type: bpy.props.EnumProperty(
        name="Planet Type",
        items=[
            ("terrestrial", "Terrestrial", "Rocky planet"),
            ("gas_giant", "Gas Giant", "Gas giant planet"),
            ("ice_giant", "Ice Giant", "Ice giant planet"),
        ],
        default="terrestrial",
    )
    radius: bpy.props.FloatProperty(name="Radius", default=1.0, min=0.1, max=100.0)

    def execute(self, context):
        bpy.ops.mesh.primitive_uv_sphere_add(
            location=(0, 0, 0), radius=self.radius, segments=64, ring_count=32
        )
        planet = bpy.context.active_object
        planet.name = f"Planet_{self.planet_type}"
        # TODO: Assign material via Node Group/Presets if available
        self.report({"INFO"}, f"Created planet: {planet.name}")
        return {"FINISHED"}


def register():
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreatePlanet)

    bpy.utils.register_class(ALBPY_OT_CreatePlanet)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreatePlanet)
