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

from email.policy import default
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
        mat = context.object.active_material
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


class cameraSettings(PropertyGroup):

    def update(self, context):        
        # data = bpy.data.objects['Camera'].data
        data = bpy.data.objects[context.scene.camera_panel.camera[0]].data
        data.lens = self.focus
        data.dof.aperture_fstop = self.aperture
        data.sensor_width = self.sensor_width    
        image_aspect_ratio = data.sensor_width/ data.sensor_height
        bpy.context.scene.render.resolution_x = int(abs(math.sqrt(self.resolution * image_aspect_ratio)))
        bpy.context.scene.render.resolution_y = int(self.resolution/ bpy.context.scene.render.resolution_x)       

    def update_camera_height(self,context):
        cameras_obj = [cam for cam in bpy.data.objects if cam.type == 'CAMERA']
        if len(cameras_obj) != 1:
            self.report({'ERROR'}, "Camera doesn't exist in scene or there is more than 1 camera.")
            return {'FINISHED'}
        
        cameras_obj[0].location[2] = self.camera_height

    
    camera_height : FloatProperty(
        name="Camera height",
        description="Camera position height",
        default=0.5,
        min=0,
        max=3.0,        
        update=update_camera_height
    )
    
    focus : FloatProperty(
        name="Focal length", 
        description="Focus distance for non-static camera placement",
        default=50.0,
        min=0.000,
        max=200.00,
        step=0.5,
        update=update
    )

    subject : PointerProperty(
        name="Focus on", 
        description="Subject to focus",
        type=bpy.types.Object,
        
    )
    
    aperture : FloatProperty(
        name="Aperture size (f/#)",
        description="Aperture size, measured in f-stops",
        min=1,
        max=64,
        step=0.2,
        update=update
    )

    resolution : FloatProperty(
        name="Resolution",
        description="Sensor resolution",
        min=16000,
        max=100000000,
        default=25000,
        update=update
    )

    sensor_width : FloatProperty(
        name="Sensor width (mm)",
        description="Sensor width",
        default=36.0,
        update = update
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
        # bpy.data.worlöds["World"].node_tree.nodes["RGB"].outputs[0].default_value = (0, 0, 0, 1)

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
            self.report({'ERROR'}, "Camera already exist in scene.")
            return {'FINISHED'}


        camera_data = bpy.data.cameras.new("Camera")
        camera_data.dof.use_dof = True
        camera_data.lens = scene.camera_panel.focus
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

class SetRender(Operator):
    bl_idname = "rti.set_render"
    bl_label = "Set render settings"

    def execute(self, context):
        scene = context.scene
        
        # Make sure compositing and nodes are enabled so that we can generate depth and normal images with render passes
        if not scene.render.use_compositing:
            scene.render.use_compositing = True
        if not scene.use_nodes:
            scene.use_nodes = True

        # Remove all existing nodes to create new ones
        try:
            for node in scene.node_tree.nodes:
                scene.node_tree.nodes.remove(node)
        except:
            pass

        outputPath = scene.acquisition_panel.output_path
        # fileName = "Image"
        # fileName = scene.file_tool.output_file_name

        # Error handling
        if outputPath == "":
            self.report({'ERROR'}, "Output file path not set.")
            return {'CANCELLED'}
        # if fileName == "":
        #     self.report({'ERROR'}, "Output file name not set.")
        #     return {'CANCELLED'}

        # Get total numbers of frames
        numLights = len(scene.light_panel.light_list)


        # Set filepath as well as format for iterated filenames
        scene.render.filepath = "{0}/Renders/Image-{1}".format(outputPath,"#")
        
        # Make sure Cycles is set as render engine
        scene.render.engine = 'CYCLES'

        # Image output settings
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"

        # Set color management to linear (?)
        scene.display_settings.display_device = 'None'

        # Disable overwriting of output images by default
        scene.render.use_overwrite = False

        # Set render passes
        current_render_layer = scene.view_layers['ViewLayer']
        # current_render_layer = scene.view_layers.active
        current_render_layer.use_pass_combined = True
        current_render_layer.use_pass_z = True
        current_render_layer.use_pass_normal = True
        current_render_layer.use_pass_shadow = True
        current_render_layer.use_pass_diffuse_direct = True
        current_render_layer.use_pass_diffuse_indirect = True
        current_render_layer.use_pass_diffuse_color = True
        current_render_layer.use_pass_glossy_direct = True
        current_render_layer.use_pass_glossy_indirect = True
        current_render_layer.use_pass_glossy_color = True

        # Create nodes for Render Layers, map range, normalization, and output files
        ## NOTE: Positioning of nodes isn't considered as it's not important for background processes.
        render_layers_node = scene.node_tree.nodes.new(type="CompositorNodeRLayers")
        # normalize_node = scene.node_tree.nodes.new(type="CompositorNodeNormalize")
        output_node_z = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        output_node_normal = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")

        # if not scene.sff_tool.focus_limits_auto:
        #     # Set map range node settings
        #     map_range_node = scene.node_tree.nodes.new(type="CompositorNodeMapRange")
        #     map_range_node.use_clamp = True
        #     map_range_node.inputs[2].default_value = scene.camera_panel.camera_height
        #     # map_range_node.inputs[1].default_value = (scene.sff_tool.camera_height - np.max(scene.sff_tool.zPosList))
        #     map_range_node.inputs[1].default_value = (scene.camera_panel.camera_height - scene.sff_tool.max_z_pos)

        #     # Link nodes together
        #     scene.node_tree.links.new(render_layers_node.outputs['Depth'], map_range_node.inputs['Value'])
        #     scene.node_tree.links.new(map_range_node.outputs['Value'], output_node_z.inputs['Image'])

        # elif scene.sff_tool.focus_limits_auto:
        normalize_node = scene.node_tree.nodes.new(type="CompositorNodeNormalize")

            # Link nodes together
        scene.node_tree.links.new(render_layers_node.outputs['Depth'], normalize_node.inputs['Value'])
        scene.node_tree.links.new(normalize_node.outputs['Value'], output_node_z.inputs['Image'])

        # Set normal node output
        scene.node_tree.links.new(render_layers_node.outputs['Normal'], output_node_normal.inputs['Image'])

        # Set output filepaths
        output_node_z.base_path = scene.file_tool.output_path + "/Depth/"
        output_node_normal.base_path = scene.file_tool.output_path + "/Normal/"

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
        layout.prop(cameratool, "focus", slider=True)
        layout.prop(cameratool, "aperture", slider=True)
        layout.prop(cameratool, "subject")
        layout.prop(cameratool, "resolution")
        layout.prop(cameratool, "sensor_width")
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
        layout.operator("rti.set_animation")
        layout.operator("rti.set_render")
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
            SetRender,
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