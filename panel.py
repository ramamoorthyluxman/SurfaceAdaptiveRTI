import bpy

from bpy.types import Panel

class RTI_PT_Panel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_label = "Surface adaptiv RTI"
    bl_category = "RTI Util"

    def draw(self, context):
        scene = context.scene
        
        layout = self.layout
        row = layout.row()
        col = row.column()
        col.operator("object.apply_all_mods", text="Apply all")
        layout.prop(mytool, "focus_limits_auto")
        


