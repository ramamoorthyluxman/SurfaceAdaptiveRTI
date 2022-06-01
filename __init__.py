# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "SurfaceAdaptiveRTI",
    "author" : "Ramamoorthy Luxman",
    "description" : "Virtual surface adaptive RTI (NBLP)",
    "blender" : (3, 1, 2),
    "version" : (0, 0, 1),
    "location" : "3D View > Tools > SurfaceAdaptiveRTI",
    "warning" : "",
    "category" : "3D View"
}

import os
import bpy
import numpy as np
import math
from mathutils import Vector
from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )
from bpy.types import (Panel,
                       Operator,
                       PropertyGroup,
                       )

#################################### Helpers ####################################
def Cartesian2Polar3D(x, y, z):
    """
    Takes X, Y, and Z coordinates as input and converts them to a polar
    coordinate system

    Source: https://stackoverflow.com/questions/10868135/cartesian-to-polar-3d-coordinates

    """

    r = math.sqrt(x*x + y*y + z*z)

    longitude = math.acos(x / math.sqrt(x*x + y*y)) * (-1 if y < 0 else 1)

    latitude = math.acos(z / r)

    return r, longitude, latitude


def Polar2Cartesian3D(r, longitude, latitude):
    """
    Takes, r, longitude, and latitude coordinates in a polar coordinate
    system and converts them to a 3D cartesian coordinate system

    Source: https://stackoverflow.com/questions/10868135/cartesian-to-polar-3d-coordinates
    """

    x = r * math.sin(latitude) * math.cos(longitude)
    y = r * math.sin(latitude) * math.sin(longitude)
    z = r * math.cos(latitude)

    return x, y, z

# def generate_homogenous_theta(n):



#################################### PropertyGroups ####################################

class light(bpy.types.PropertyGroup):
    light = bpy.props.PointerProperty(name="Light object", 
                                      type = bpy.types.Object,
                                      description = "A light source")
                                      
                                      
class camera(bpy.types.PropertyGroup):
    camera = bpy.props.PointerProperty(name="Camera object", 
                                       type = bpy.types.Object,
                                       description = "Camera")


class surface(bpy.types.PropertyGroup):
    surface = bpy.props.PointerProperty(name="Surface object", 
                                       type = bpy.types.Object,
                                       description = "Surface")


class lightSettings(PropertyGroup):

    nblp: BoolProperty(   
        name="NBLP", 
        description="NBLP algorithm generated Light Positions",
        default=True,
    )

    lp_file_path : StringProperty(
        name="LP file", 
        subtype="FILE_PATH",
        description="File path for light positions file (.lp)",
        default="",
        maxlen=1024
    )

    dome_radius : FloatProperty(
        name="RTI dome radius",
        description="Radius of RTI dome [m]",
        default=1.0
    )

    light_list = []

class surfaceSettings(PropertyGroup):

    def update(self, context):
        mat = context.scene.objects[context.scene.surface_panel.surface[0]].active_material
        node_tree = mat.node_tree
        nodes = node_tree.nodes
        bsdf = nodes.get("Principled BSDF") 
        bsdf.inputs[6].default_value=self.metallic 
        bsdf.inputs[7].default_value=self.specularity
        bsdf.inputs[9].default_value=self.roughness

    mesh_file_path : StringProperty(
        name="Surface mesh file", 
        subtype="FILE_PATH",
        description="File path for the surface mesh file",
        default="",
        maxlen=1024
    ) 
    
    texture_image_path : StringProperty(
        name="Surface texture image file",
        subtype="FILE_PATH",
        description="File path for the surface texture image file",
        default="",
        maxlen=1024

    )

    metallic : FloatProperty(
        name="Metallic",
        description="Surface metallic",
        default=0.0,
        min=0.0,
        max=1.0,
        update=update
    )

    roughness : FloatProperty(
        name="Roughness",
        description="Surface roughness",
        default=0.0,
        update=update,
        min=0.0,
        max=1.0,
    )

    specularity : FloatProperty(
        name="Specularity",
        description="Specularity",
        default=0.5,
        update=update,
        min=0.0,
        max=1.0,
    )

    surface = []
    surface_bbox = []


class cameraSettings(PropertyGroup):

    def update(self, context):        
        # data = bpy.data.objects['Camera'].data
        data = bpy.data.objects[context.scene.camera_panel.camera[0]].data
        data.ortho_scale = self.ortho_scale
        image_aspect_ratio = data.sensor_width/ data.sensor_height        

    def update_camera_height(self,context):
        cameras_obj = [cam for cam in bpy.data.objects if cam.type == 'CAMERA']
        if len(cameras_obj) != 1:
            self.report({'ERROR'}, "Camera doesn't exist in scene or there is more than 1 camera.")
            return {'FINISHED'}
        
        cameras_obj[0].location[2] = self.camera_height

    
    camera_height : FloatProperty(
        name="Camera height",
        description="Camera position height",
        default=1.03,
        min=0,
        max=3.0,        
        update=update_camera_height
    )
    
    ortho_scale : FloatProperty(
        name="Orthographic scale", 
        description="Orthographic projection scale",
        default=0.24,
        min=0.000,
        max=3,        
        update=update
    )

    subject : PointerProperty(
        name="Focus on", 
        description="Subject to focus",
        type=bpy.types.Object,
        
    )
    

    resolution_factor : FloatProperty(
        name="Resolution",
        description="Resolution scale factor",
        min=0,
        max=5,
        default=1.0,
        update=update
    )


    camera = []
    
    
class acquisitionSettings(PropertyGroup):    
    output_path : StringProperty(
        name="Output path", 
        subtype="FILE_PATH",
        description="File path for saving the rti acquisition",
        default="",
        maxlen=1024
    )
    
    csvOutputLines = []


#################################### Operators ####################################
class reset_scene(Operator):
    bl_label = "Reset all"
    bl_idname = "rti.reset_scene"
    
    def execute(self, context):
        scene = context.scene
        for current_light in bpy.data.lights:
            bpy.data.lights.remove(current_light)
        for current_object in bpy.data.objects:
            bpy.data.objects.remove(current_object)

        return {"FINISHED"} 


class createLights(Operator):
    bl_label = "Create lights"
    bl_idname = "rti.create_lights"
    
    def execute(self, context):
        scene = context.scene
        mytool = scene.light_panel
        mytool.light_list.clear()

        light_obj = [light for light in bpy.data.objects if light.type == 'LIGHT']
        for light in bpy.data.lights:
            light.animation_data_clear()
            bpy.data.lights.remove(light)

        # Deselect any currently selected objects
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except:
            pass

        

        if not os.path.isfile(mytool.lp_file_path) and not mytool.nblp:
            self.report({"ERROR"})

        light_sources = bpy.data.objects.new(name = "light_sources", object_data = None)
        scene.collection.objects.link(light_sources)        

        # Read in .lp data
        try:
            file = open(mytool.lp_file_path)
        except RuntimeError as ex:
            error_report = "\n".join(ex.args)
            print("Caught error:", error_report)
            return {'ERROR'}

        rows = file.readlines()
        file.close()

        # Parse for number of lights
        numLights = int(rows[0].split()[0])

        for idx in range(1, numLights + 1):
            cols = rows[idx].split()
            x = float(cols[1])
            y = float(cols[2])
            z = float(cols[3])

            r, long, lat = Cartesian2Polar3D(x,y,z)
            r = mytool.dome_radius
            x, y, z = Polar2Cartesian3D(r,long,lat)
            lightData = bpy.data.lights.new(name="RTI_light"+str(idx), type="SUN")

            current_light = bpy.data.objects.new(name="Light_{0}".format(idx), object_data=lightData)
            
            current_light.location = (x, y, z)

            scene.collection.objects.link(current_light)   
            
            current_light.rotation_mode = 'QUATERNION'
            current_light.rotation_quaternion = Vector((x,y,z)).to_track_quat('Z','Y')

            current_light.parent = light_sources

            mytool.light_list.append(current_light.name)

        return {"FINISHED"}   
    
class createCamera(Operator):
    bl_idname = "rti.create_camera"
    bl_label = "Create camera"    
    def execute(self, context):
        scene = context.scene
        cameras_obj = [cam for cam in bpy.data.objects if cam.type == 'CAMERA']
        scene.camera_panel.camera.clear()
        if len(cameras_obj) != 0:
            # self.report({'ERROR'}, "Camera already exist in scene.")
            # return {'FINISHED'}
            for cam in bpy.data.cameras:
                cam.animation_data_clear()
                bpy.data.cameras.remove(cam)

        camera_data = bpy.data.cameras.new("Camera")
        camera_data.dof.use_dof = False
        camera_data.type="ORTHO"
        camera_object = bpy.data.objects.new("Camera", camera_data)
        
        # Link camera to current scene
        scene.collection.objects.link(camera_object)
        
        # Move camera to default location at top of dome
        camera_object.location = (0,0,scene.light_panel.dome_radius)
        scene.camera_panel.camera.append(camera_object.name)

        return {'FINISHED'}     


class importSurface(Operator):
    bl_idname = "rti.import_surface"
    bl_label = "Import surface"    
    def execute(self, context):
        scene = context.scene
        surfacetool = scene.surface_panel
        objects = [obj for obj in bpy.data.objects if obj.type != 'CAMERA']

        if len(objects) != 0:
            self.report({'ERROR'}, "Surface already exist in scene. Delete the old surface to add new surface")
            return {'FINISHED'}

        if scene.surface_panel.mesh_file_path.endswith(".OBJ"):
            bpy.ops.import_scene.obj(filepath=scene.surface_panel.mesh_file_path)
            ob = bpy.context.selected_objects[0]
            ob.rotation_euler[0] = 0.0523599
            surfacetool.surface.append(ob.name)
            
        return {'FINISHED'} 

class addTexture(Operator):
    bl_idname = "rti.add_texture"
    bl_label = "Add texture"    
    def execute(self, context):
        scene = context.scene
        mat = scene.objects[scene.surface_panel.surface[0]].active_material
        nodes = mat.node_tree.nodes
        nodes.clear()
        node_principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_principled.location = 0,0
        node_tex = nodes.new('ShaderNodeTexImage')
        node_tex.image = bpy.data.images.load(scene.surface_panel.texture_image_path)
        node_tex.location = -400,0

        node_output = nodes.new(type='ShaderNodeOutputMaterial')   
        node_output.location = 400,0

        links = mat.node_tree.links
        link = links.new(node_tex.outputs["Color"], node_principled.inputs["Base Color"])
        link = links.new(node_principled.outputs["BSDF"], node_output.inputs["Surface"])

        node_tex = nodes.new('ShaderNodeTexImage')
        node_tex.location = -400,0
        img = bpy.data.images.get(scene.surface_panel.texture_image_path)
        if img:
            node_tex.image = img        

        return {'FINISHED'} 


class SetAnimation(Operator):
    bl_idname = "rti.set_animation"
    bl_label = "Create animation"

    def execute(self, context):
        scene = context.scene        

        # bpy.
        scene.timeline_markers.clear()
        scene.animation_data_clear()    

        if len(bpy.data.worlds)>0:
            bpy.data.worlds.remove(bpy.data.worlds["World"], do_unlink=True)

        data = bpy.data.objects[context.scene.camera_panel.camera[0]].data
        if context.scene.camera_panel.subject is not None:
            data.dof.focus_object = context.scene.camera_panel.subject

        
        numLights = len(scene.light_panel.light_list)

        for i in range(0, numLights):
            current_light = bpy.data.objects[scene.light_panel.light_list[i]]
            current_light.data.energy = 0  
            current_light.data.keyframe_insert(data_path="energy", frame=1)
            
        
        for i in range(0, numLights):
            current_light = bpy.data.objects[scene.light_panel.light_list[i]] 
            current_light.data.energy = 2
            current_light.data.keyframe_insert(data_path="energy", frame=i+1)
            current_light.data.energy = 0
            current_light.data.keyframe_insert(data_path="energy", frame=i+2)


        return {'FINISHED'}

class acquire(Operator):
    bl_idname = "rti.acquire"
    bl_label = "Acquire"    
    bl_use_preview = True
    
    def execute(self, context):
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.preview_samples = 1024
        bpy.context.scene.cycles.samples = 4000
        bpy.context.scene.cycles.use_preview_denoising = True
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.render.resolution_x = int(1920*context.scene.camera_panel.resolution_factor) 
        bpy.context.scene.render.resolution_y = int(1080*context.scene.camera_panel.resolution_factor)
        # bpy.context.scene.render.use_border = True
        # bpy.context.scene.render.use_crop_to_border = True
        bpy.context.scene.frame_start = 1
        bpy.context.scene.render.filepath = context.scene.acquisition_panel.output_path
        bpy.context.scene.render.image_settings.color_mode = 'BW'
        bpy.context.scene.render.image_settings.color_depth = '8'
        bpy.context.scene.render.use_overwrite = True
        context.scene.camera = context.scene.objects[context.scene.camera_panel.camera[0]] 
        bpy.ops.render.play_rendered_anim() 
        if not context.scene.light_panel.nblp:
            bpy.context.scene.frame_end = len(context.scene.light_panel.light_list)
            bpy.ops.rti.set_animation()
            bpy.ops.render.render(animation=True, use_viewport = True, write_still=True)
            

        return {'FINISHED'}

#################################### Panels #######################################
class rti_panel(Panel):

    bl_label = "Surface Adaptive RTI"
    bl_idname = "VIEW3D_PT_surface_adaptive_rti"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Surface Adaptive RTI"
        
    def draw(self, context):
        layout = self.layout
        scene = context.scene

class light_panel(Panel):
    bl_label = "Light positions"
    bl_parent_id = "VIEW3D_PT_surface_adaptive_rti"
    bl_idname = "VIEW3D_PT_light_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Surface Adaptive RTI"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        lighttool = scene.light_panel
        layout.label(text="Light positions")
        layout.prop(lighttool, "nblp")
        layout.prop(lighttool, "dome_radius")

        if not lighttool.nblp:
            layout.prop(lighttool,"lp_file_path")
            row = layout.row(align = True)
            row.operator("rti.create_lights")
        else:
            for current_light in bpy.data.lights:
                bpy.data.lights.remove(current_light) 

class surface_panel(Panel):
    bl_label = "Surface"
    bl_parent_id = "VIEW3D_PT_surface_adaptive_rti"
    bl_idname = "VIEW3D_PT_surface_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Surface Adaptive RTI"
    def draw(self, context):
        layout = self.layout
        scene = context.scene        
        surfacetool = scene.surface_panel
        layout.label(text="Surface")
        layout.prop(surfacetool, "mesh_file_path")
        row = layout.row(align = True)
        row.operator("rti.import_surface")
        layout.label(text="Texture")
        layout.prop(surfacetool, "texture_image_path")
        row = layout.row(align = True)
        row.operator("rti.add_texture")
        layout.prop(surfacetool, "metallic", slider=True)
        layout.prop(surfacetool, "roughness", slider=True)
        layout.prop(surfacetool, "specularity", slider=True)
        
class camera_panel(Panel):
    bl_label = "Camera"
    bl_parent_id = "VIEW3D_PT_surface_adaptive_rti"
    bl_idname = "VIEW3D_PT_camera_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Surface Adaptive RTI"
    def draw(self, context):
        layout = self.layout
        scene = context.scene        
        cameratool = scene.camera_panel
        layout.label(text="Camera")
        layout.prop(cameratool, "camera_height", slider=True)
        layout.prop(cameratool, "ortho_scale", slider=True)
        layout.prop(cameratool, "subject")
        layout.prop(cameratool, "resolution_factor")
        row = layout.row(align = True)
        row.operator("rti.create_camera")

class acquisition_panel(Panel):
    bl_label = "Acquisition"
    bl_parent_id = "VIEW3D_PT_surface_adaptive_rti"
    bl_idname = "VIEW3D_PT_acquisition_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Surface Adaptive RTI"
    def draw(self, context):
        layout = self.layout
        scene = context.scene        
        layout.label(text="Acquisition")
        layout.prop(scene.acquisition_panel, "output_path")
        # layout.operator("rti.set_animation")
        layout.operator("rti.acquire")
        row = layout.row(align = True)
        row.operator("rti.reset_scene")
    
#################################### Register classes #######################################
classes = (reset_scene, 
            light, 
            camera, 
            surface,
            lightSettings, 
            cameraSettings, 
            surfaceSettings, 
            acquisitionSettings,
            createLights, 
            createCamera, 
            importSurface,
            addTexture,
            SetAnimation,
            acquire,
            rti_panel,
            surface_panel,
            light_panel,             
            camera_panel,
            acquisition_panel)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.light_panel = PointerProperty(type=lightSettings)
    bpy.types.Scene.surface_panel = PointerProperty(type=surfaceSettings)
    bpy.types.Scene.camera_panel = PointerProperty(type=cameraSettings)
    bpy.types.Scene.acquisition_panel = PointerProperty(type=acquisitionSettings)
    
    

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.light_panel

if __name__ == "__main__":
    register()