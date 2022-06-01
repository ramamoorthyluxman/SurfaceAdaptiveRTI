from bpy.types import PropertyGroup
from bpy.props import BoolProperty


class AcquisitionSettings(PropertyGroup):

    focus_limits_auto : BoolProperty(   
        name="Automatic focus positions", 
        description="Auto setting of camera position limts.",
        default=True,
    )
    