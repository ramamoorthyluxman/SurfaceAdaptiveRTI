import bpy
from bpy.types import Operator


class RTI_OT_Apply_ALL_Op(Operator):
    bl_idname = "object.apply_all_mods"
    bl_label = "Apply all"
    bl_description = "asdasdad"

    @classmethod
    def poll(cls, context):
        obj = context.object
        if obj is not None:
            if obj.mode == "OBJECT":
                return True
        return False
    
    
    def execute(self, context):
        active_obj = context.view_layer.objects.active

        for mod in active_obj.modifiers:
            bpy.ops.object.modifier_apply(modifier = mod.name)

        return ('FINISHED')




