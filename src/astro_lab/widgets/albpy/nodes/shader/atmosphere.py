import math

import bpy


class AlbpyAtmosphereShaderGroup(bpy.types.ShaderNodeGroup):
    bl_idname = "ALBPY_NG_Atmosphere"
    bl_label = "Albpy Atmospheric Scattering Shader Group"

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == "ShaderNodeTree"

    def init(self, context):
        # Clear existing nodes
        self.nodes.clear()
        self.inputs.clear()
        self.outputs.clear()

        # Create inputs
        self.inputs.new("NodeSocketColor", "Atmosphere Color")
        self.inputs.new("NodeSocketFloat", "Density")
        self.inputs.new("NodeSocketFloat", "Scale Height")
        self.inputs.new("NodeSocketFloat", "Planet Radius")
        self.inputs.new("NodeSocketString", "Atmosphere Type")

        # Create outputs
        self.outputs.new("NodeSocketShader", "Volume")

        # Set default values
        self.inputs["Atmosphere Color"].default_value = (0.3, 0.6, 1.0, 1.0)
        self.inputs["Density"].default_value = 0.1
        self.inputs["Scale Height"].default_value = 8.5
        self.inputs["Planet Radius"].default_value = 1.0
        self.inputs["Atmosphere Type"].default_value = "earth"

        # Create nodes
        scatter = self.nodes.new("ShaderNodeVolumeScatter")
        output = self.nodes.new("NodeGroupOutput")

        # Altitude-dependent density calculation
        geometry = self.nodes.new("ShaderNodeNewGeometry")
        vector_length = self.nodes.new("ShaderNodeVectorMath")
        vector_length.operation = "LENGTH"

        # Exponential density falloff
        subtract = self.nodes.new("ShaderNodeMath")
        subtract.operation = "SUBTRACT"

        divide = self.nodes.new("ShaderNodeMath")
        divide.operation = "DIVIDE"

        power = self.nodes.new("ShaderNodeMath")
        power.operation = "POWER"
        power.inputs[0].default_value = math.e

        multiply_negative = self.nodes.new("ShaderNodeMath")
        multiply_negative.operation = "MULTIPLY"
        multiply_negative.inputs[1].default_value = -1.0

        multiply_density = self.nodes.new("ShaderNodeMath")
        multiply_density.operation = "MULTIPLY"

        # Position nodes
        geometry.location = (-800, -200)
        vector_length.location = (-600, -200)
        subtract.location = (-400, -200)
        divide.location = (-200, -200)
        multiply_negative.location = (0, -300)
        power.location = (0, -200)
        multiply_density.location = (200, -100)
        scatter.location = (400, 0)
        output.location = (600, 0)

        # Connect nodes
        self.links.new(geometry.outputs["Position"], vector_length.inputs[0])
        self.links.new(vector_length.outputs["Value"], subtract.inputs[1])
        self.links.new(self.inputs["Planet Radius"], subtract.inputs[0])
        self.links.new(subtract.outputs["Value"], divide.inputs[0])
        self.links.new(self.inputs["Scale Height"], divide.inputs[1])
        self.links.new(divide.outputs["Value"], multiply_negative.inputs[0])
        self.links.new(multiply_negative.outputs["Value"], power.inputs[1])
        self.links.new(power.outputs["Value"], multiply_density.inputs[0])
        self.links.new(self.inputs["Density"], multiply_density.inputs[1])
        self.links.new(multiply_density.outputs["Value"], scatter.inputs["Density"])
        self.links.new(self.inputs["Atmosphere Color"], scatter.inputs["Color"])
        self.links.new(scatter.outputs["Volume"], output.inputs["Volume"])


# Atmospheric presets
ATMOSPHERE_PRESETS = {
    "earth": {
        "Atmosphere Color": (0.3, 0.6, 1.0, 1.0),
        "Density": 0.1,
        "Scale Height": 8.5,
        "Planet Radius": 1.0,
    },
    "mars": {
        "Atmosphere Color": (0.8, 0.5, 0.3, 1.0),
        "Density": 0.01,
        "Scale Height": 11.0,
        "Planet Radius": 0.53,
    },
    "venus": {
        "Atmosphere Color": (0.9, 0.8, 0.6, 1.0),
        "Density": 0.9,
        "Scale Height": 15.0,
        "Planet Radius": 0.95,
    },
    "titan": {
        "Atmosphere Color": (0.7, 0.6, 0.4, 1.0),
        "Density": 0.15,
        "Scale Height": 20.0,
        "Planet Radius": 0.4,
    },
}


def get_atmosphere_preset(atmosphere_type: str) -> dict:
    """Get preset for atmospheric configuration."""
    return ATMOSPHERE_PRESETS.get(atmosphere_type, ATMOSPHERE_PRESETS["earth"])


def apply_atmosphere_preset(material: bpy.types.Material, preset_name: str) -> None:
    """Apply atmospheric preset to material."""
    preset = get_atmosphere_preset(preset_name)

    # Find node group
    for node in material.node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            if "ALBPY_NG_Atmosphere" in node.node_tree.name:
                # Apply preset parameters
                for param_name, value in preset.items():
                    if param_name in node.inputs:
                        node.inputs[param_name].default_value = value
                break


def register():
    # REMOVED: bpy.utils.register_class(AlbpyAtmosphereShaderGroup)

    bpy.utils.register_class(AlbpyAtmosphereShaderGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyAtmosphereShaderGroup)
