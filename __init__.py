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
from random import sample
import bpy
import numpy as np
import math
import time
import cv2
import numpy as np
from mathutils import Vector
import matplotlib.pyplot as plot
import yaml
from itertools import islice
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


#################################### Global vars { ################################
updated_lps = []
#################################### Global vars } ################################

#################################### Helpers { ####################################

# def cart2sph(x, y, z):
def cart2sph(*args):
    x = args[0]
    y = args[1]
    z = 1.0
    if len(args)>2:
        z = args[2]
    print(x,",",y,",",z)
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, az, el


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


# Reads the standard .lp file to extract the light positions. 
# This function calculates just the directions - thetas and phis. The radius of the dome in the actual lp file is ignored and considers the radius passed to this function only to recreate new light positions. 
def read_lp_file(file_path, dome_radius):
    light_positions = []
    # Read in .lp data

    if not os.path.exists(file_path):
        return light_positions
    
    try:
        file = open(file_path)
    except RuntimeError as ex:
        error_report = "\n".join(ex.args)
        print("Caught error:", error_report)
        return {'ERROR'}

    rows = file.readlines()
    file.close()

    numLights = int(rows[0].split()[0])

    for idx in range(1, numLights + 1):
        cols = rows[idx].split()
        x = float(cols[1])
        y = float(cols[2])
        z = float(cols[3])
        r, long, lat = cart2sph(x,y,z)
        r = dome_radius
        x, y, z = sph2cart(long,lat,r)
        light_positions.append((float(x),float(y),float(z)))
    
    return light_positions   

############################ Generate n evenly spaced hemispherical points { ##################################

def generate_n_evenly_spaced_hemispherical_points(samples = 45):
    samples = 2*samples
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    cartesian_points = []
    polar_points = []
    for i in range(int(samples/2), samples):
        z = -1 + (i / float(samples - 1)) * 2  # z goes from 0 to 1
        radius = math.sqrt(1 - z * z)  # radius at z
        theta = phi * i  # golden angle increment
        y = math.cos(theta) * radius
        x = math.sin(theta) * radius
        r, az, el = cart2sph(x,y,z)
        # convert the angles to be in the range 0 to 2 pi
        az = (az + np.pi) % (2 * np.pi) - np.pi
        el = (el + np.pi) % (2 * np.pi) - np.pi
        polar_points.append((float(az),float(el)))
        cartesian_points.append((x, y, z))            
        
    # plot.figure().add_subplot(111, projection='3d').scatter([p[0] for p in cartesian_points], [p[1] for p in cartesian_points], [p[2] for p in cartesian_points]);    
    # plot.show()

    return polar_points, cartesian_points


############################ Generate n evenly spaced hemispherical points } ##################################

def generate_homogenous_points_along_theta(samples = 45, dome_radius=1.0, phi=55.0):
        light_positions_cartesian = []
        light_positions_polar = []
        dome_radius = dome_radius
        phi = math.radians(phi)
        if samples%2 != 0:
            samples = samples+1
        for i in range(0,samples):
            theta = i*(math.radians(360.0)/samples)
            x, y, z = sph2cart(theta, phi, dome_radius)
            light_positions_cartesian.append((x,y,z))
            light_positions_polar.append((theta, phi))
        

        return  light_positions_polar, light_positions_cartesian

#################################### Helpers } ####################################

#################################### NBLP algorithms { ####################################

#################################### NBLP Basic { ####################################

class Nblp:
    iterations = []
    class iteration:
        def __init__(self, lps_polar, lps_cartesian, iteration_nb):
            self.lps_polar = lps_polar
            self.nb_images = len(lps_polar)
            self.lps_cartesian = lps_cartesian
            self.filenames_subtext = "nblp_iteration_"+str(iteration_nb)+"_"
            self.iteration_nb = iteration_nb

        def plot_lps(self):
            fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
            for i in range(0,self.nb_images):
                theta = self.lps_polar[i][0]
                radius = math.degrees(self.lps_polar[i][1])
                ax.plot(theta, radius,"o")
                ax.set_rmax(2)
                ax.set_rticks([90, 60, 30, 0]) 
                ax.set_rlabel_position(-22.5)  
                ax.grid(True)
                ax.set_title("Light positions projected to 2D plane")
            return plot

        def rename_files(self, path):
            for i in range(0, self.nb_images):
                theta=str(math.floor(100*math.degrees(self.lps_polar[i][0]))/100)
                phi=str(math.floor(100*math.degrees(self.lps_polar[i][1]))/100)
                old_name = path+"nblp_iteration_"+str(self.iteration_nb)+"_"+str(i+1)+".png"
                new_name = path+"nblp_iteration_"+str(self.iteration_nb)+"_"+str(i+1)+"_theta_" + theta + "_phi_" + phi + ".png"
                os.rename(old_name, new_name)                
            
    def __init__(self):
        iteration_nb = len(self.iterations)
        lps_polar, lps_cartesian = self.generate_homogenous_points_along_theta(n=45, dome_radius=1, phi=math.radians(45.0), iteration_nb=iteration_nb)        
        step = self.iteration(lps_polar, lps_cartesian, iteration_nb)
        self.iterations.append(step)

    def dense_acquisition(self):   
        iteration_nb = len(self.iterations)     
        lps_polar, lps_cartesian = self.generate_homogenous_points_along_theta(n=45, dome_radius=1, phi=math.radians(45.0), iteration_nb=iteration_nb)
        step = self.iteration(lps_polar, lps_cartesian, iteration_nb)
        self.iterations.append(step)

    def generate_homogenous_points_along_theta(self, n, dome_radius, phi, iteration_nb):
        light_positions_cartesian = []
        light_positions_polar = []
        self.dome_radius = dome_radius
        if n%2 != 0:
            n = n+1
        for i in range(0,n):
            theta = i*(math.radians(360.0)/n)
            x, y, z = sph2cart(theta, phi, dome_radius)
            light_positions_cartesian.append((x,y,z))
            light_positions_polar.append((theta, phi))
        

        return  light_positions_polar, light_positions_cartesian

    def calculate_entropies(self,iteration_nb, file_path):
        print("Calculating entropies")
        img_path = file_path+"\\..\\"+self.iterations[iteration_nb].filenames_subtext+str(1)+".png"
        img_sum = cv2.imread(img_path)
        for i in range(0, self.iterations[iteration_nb].nb_images):
            img_path = file_path+"\\..\\"+self.iterations[iteration_nb].filenames_subtext+str(i+1)+".png"
            print(img_path)
            img = cv2.imread(img_path)
            img_sum = cv2.addWeighted(img_sum,0.5,img,0.5,0)
        

        for i in range(0, self.iterations[iteration_nb].nb_images):
            img_path = file_path+"\\..\\"+self.iterations[iteration_nb].filenames_subtext+str(i+1)+".png"
            img_diff = cv2.absdiff(img_sum,cv2.imread(img_path))
            normalized_img_diff = np.zeros(img_diff.shape)
            # min_val = img_diff[..., 0].min()
            # max_val = img_diff[..., 0].max()
            min_val = img_diff.min()
            max_val = img_diff.max()
            normalized_img_diff = img_diff * (255/(max_val-min_val))
            # cv2.normalize(img_diff, normalized_img_diff, min_val, max_val, cv2.NORM_MINMAX)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                cv2.imwrite(file_path+self.iterations[iteration_nb].filenames_subtext+str(i)+".png", img_diff)
                cv2.imwrite(file_path+self.iterations[iteration_nb].filenames_subtext+str(i)+"_normalized.png", normalized_img_diff)

    def generate_lp_file(self, iteration_nb, file_path):
        data = str(self.iterations[iteration_nb].nb_images)
        for i in range(0, self.iterations[iteration_nb].nb_images):
            step =self.iterations[iteration_nb] 
            data = data+"\n"+step.filenames_subtext+str(i+1)+".png\t"+str(step.lps_cartesian[i][0])+"\t"+str(step.lps_cartesian[i][1])+"\t"+str(step.lps_cartesian[i][2])
        with open(file_path, 'w') as f:
            f.write(data)

    def write_log(self, path):
        with open(path, 'w') as file:
            iterations = []
            for i in range(0, len(self.iterations)):
                iteration = {{'iteration nb': i},
                             {'lps_polar': self.iterations[i].lps_polar},
                             {'nb_images': self.iterations[i].nb_images},
                             {'lps_cartesian':self.iterations[i].lps_cartesian},
                             {'filenames_subtext': self.iterations[i].filenames_subtext},
                             {'iteration_nb':self.iterations[i].iteration_nb}}

                iterations.append(iteration)
            yaml.dump(iterations, file)    

#################################### NBLP Basic } ####################################

#################################### NBLP Basic approach 2 = Relighting loss gradient descent approach { ####################################

class Nblp_2:
    iterations = []
    class iteration:
        def __init__(self, lps_polar, lps_cartesian, iteration_nb, context):
            self.lps_polar = lps_polar
            self.nb_images = len(lps_polar)
            self.lps_cartesian = lps_cartesian
            self.filenames_subtext = "nblp_iteration_"+str(iteration_nb)+"_"
            self.iteration_nb = iteration_nb
            self.context = context
            self.generate_file_names()
            # self.loss = 1.0
            

        def plot_lps(self):
            fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
            for i in range(0,self.nb_images):
                theta = self.lps_polar[i][0]
                radius = math.degrees(self.lps_polar[i][1])
                ax.plot(theta, radius,"o")
                ax.set_rmax(2)
                ax.set_rticks([90, 60, 30, 0]) 
                ax.set_rlabel_position(-22.5)  
                ax.grid(True)
                ax.set_title("Light positions projected to 2D plane")
            return plot

        def rename_files(self, path):
            for i in range(0, self.nb_images):
                theta=str(math.floor(100*math.degrees(self.lps_polar[i][0]))/100)
                phi=str(math.floor(100*math.degrees(self.lps_polar[i][1]))/100)
                old_name = path+"nblp_iteration_"+str(self.iteration_nb)+"_"+str(i+1)+".png"
                new_name = path+self.file_names[i]                
                os.rename(old_name, new_name)

        def generate_lp_file(self, file_path):
            data = str(self.nb_images)
            for i in range(0, self.nb_images):
                data = data+"\n"+self.file_names[i]+"\t"+str(self.lps_cartesian[i][0])+"\t"+str(self.lps_cartesian[i][1])+"\t"+str(self.lps_cartesian[i][2])
            with open(file_path, 'w') as f:
                f.write(data)

        def calculate_entropies(self,iteration_nb, file_path):
            print("Calculating entropies")
            img_path = file_path+"\\..\\"+self.file_names[0]
            img_sum = cv2.imread(img_path)
            for i in range(0, self.nb_images):
                img_path = file_path+"\\..\\"+self.file_names[i]
                print(img_path)
                img = cv2.imread(img_path)
                img_sum = cv2.addWeighted(img_sum,0.5,img,0.5,0)            

            for i in range(0, self.nb_images):
                img_path = file_path+"\\..\\"+self.file_names[i]
                img_diff = cv2.absdiff(img_sum,cv2.imread(img_path))
                normalized_img_diff = np.zeros(img_diff.shape)
                # min_val = img_diff[..., 0].min()
                # max_val = img_diff[..., 0].max()
                min_val = img_diff.min()
                max_val = img_diff.max()
                normalized_img_diff = img_diff * (255/(max_val-min_val))
                # cv2.normalize(img_diff, normalized_img_diff, min_val, max_val, cv2.NORM_MINMAX)
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                    cv2.imwrite(file_path+self.file_names[i], img_diff)
                    cv2.imwrite(file_path+self.file_names[i].split(".")[0]+"_normalized.png", normalized_img_diff)

        def generate_file_names(self):
            self.file_names = []
            for i in range(0, self.nb_images):
                theta=str(math.floor(100*math.degrees(self.lps_polar[i][0]))/100)
                phi=str(math.floor(100*math.degrees(self.lps_polar[i][1]))/100)
                self.file_names.append("nblp_iteration_"+str(self.iteration_nb)+"_"+str(i+1)+"_theta_" + theta + "_phi_" + phi + ".png")

        def execute_acq(self):
            global updated_lps
            updated_lps = self.lps_cartesian
            bpy.context.scene.render.filepath = self.context.scene.acquisition_panel.output_path+self.filenames_subtext+"#.png"
            bpy.ops.rti.create_lights()
            bpy.context.scene.frame_end = len(self.context.scene.light_panel.light_list)
            bpy.ops.rti.set_animation()
            bpy.ops.render.render(animation=True, use_viewport = True, write_still=True)
            bpy.ops.render.play_rendered_anim()         
            self.plot_lps().savefig(self.context.scene.acquisition_panel.output_path+"iteration_"+str(self.iteration_nb)+"_"+str(self.nb_images)+".png")
            self.rename_files(self.context.scene.acquisition_panel.output_path)
            self.generate_lp_file(file_path=self.context.scene.acquisition_panel.output_path+"iteration_"+str(self.iteration_nb)+".lp")
            self.calculate_entropies(self.iteration_nb,self.context.scene.acquisition_panel.output_path+"\\entropies\\") 

            
    def __init__(self, context):
        path = context.scene.acquisition_panel.output_path
        self.context = context
        if not os.path.exists(path +"log.yaml"):
            lps_polar, lps_cartesian = generate_n_evenly_spaced_hemispherical_points(samples=1000)
            # lps_polar, lps_cartesian = generate_homogenous_points_along_theta(samples=30)
            step = self.iteration(lps_polar=lps_polar, lps_cartesian=lps_cartesian, iteration_nb=0, context=context)
            step.execute_acq()
            self.iterations.append(step)
        else:
            print(path + "log.yaml")
            data = None
            with open(path + "log.yaml") as f:
                data = yaml.load(f.read(), yaml.loader.Loader)
                
            for i in range(0, len(data)):
                step = self.iteration(data[i]['lps_polar'], data[i]['lps_cartesian'], len(data)-1, context=context)
                self.iterations.append(step)

            if os.path.exists(path + "next_iteration.lp"):
                lps_cartesian = read_lp_file(path + "next_iteration.lp", 1.0)
                print(len(lps_cartesian))
                lps_polar = []
                for i in range(0, len(lps_cartesian)):
                    r, az, el = cart2sph(lps_cartesian[i][0], lps_cartesian[i][1], lps_cartesian[i][2])
                    lps_polar.append((float(az),float(el)))
                step = self.iteration(lps_polar=lps_polar, lps_cartesian=lps_cartesian, iteration_nb=len(data), context=context)
                step.execute_acq()
                self.iterations.append(step)

        # for i in range(0,len(self.iterations)):

        #     data = str(self.nb_images)
        #     for i in range(0, self.nb_images):
        #         data = data+"\n"+self.file_names[i]+"\t"+str(self.lps_cartesian[i][0])+"\t"+str(self.lps_cartesian[i][1])+"\t"+str(self.lps_cartesian[i][2])
        #     with open(file_path, 'w') as f:
        #         f.write(data)

    def dense_acquisition(self):   
        iteration_nb = len(self.iterations)     
        lps_polar, lps_cartesian = self.generate_homogenous_points_along_theta(n=100, dome_radius=1, phi=math.radians(50.0), iteration_nb=iteration_nb)
        step = self.iteration(lps_polar, lps_cartesian, iteration_nb)
        self.iterations.append(step)

    
    
    

    def write_log(self, path):
        open(path, "w").close()
        with open(path, 'w') as file:
            iterations = []
            for i in range(0, len(self.iterations)):
                iteration = {'lps_polar': list(self.iterations[i].lps_polar),
                             'nb_images': self.iterations[i].nb_images,
                             'lps_cartesian': list(self.iterations[i].lps_cartesian),
                             'filenames_subtext': self.iterations[i].filenames_subtext,
                             'iteration_nb':self.iterations[i].iteration_nb,
                             'file_names': self.iterations[i].file_names}

                iterations.append(iteration)
            yaml.dump(iterations, file) 




                
#################################### NBLP Basic approach 2 = Relighting loss } ####################################

#################################### NBLP algorithms } ####################################

#################################### PropertyGroups { ####################################
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

#################################### PropertyGroups } ####################################

#################################### Menu { ####################################
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
        maxlen=1024, 
    )

    dome_radius : FloatProperty(
        name="RTI dome radius",
        description="Radius of RTI dome [m]",
        default=1.0
    )

    light_strength : FloatProperty(
        name="Light strength",
        description="Strength of the light source",
        default=0.5
    )

    light_positions = []
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
        cameras_obj= [cam for cam in bpy.data.objects if cam.type == 'CAMERA']
        # data = cameras_obj[0].data
        data = bpy.data.objects[context.scene.camera_panel.camera[0]].data
        data.lens = self.focal_length
        bpy.context.scene.render.resolution_y = int(self.aspect_ratio*bpy.context.scene.render.resolution_x)
        data.display_size = self.view_port_size
        data.sensor_width = self.sensor_size

    def update_camera_height(self,context):
        cameras_obj = [cam for cam in bpy.data.objects if cam.type == 'CAMERA']
        if len(cameras_obj) != 1:
            self.report({'ERROR'}, "Camera doesn't exist in scene or there is more than 1 camera.")
            return {'FINISHED'}
        
        cameras_obj[0].location[2] = self.camera_height

    
    camera_height : FloatProperty(
        name="Camera height",
        description="Camera position height",
        default=1,
        min=0,
        max=4.0,        
        update=update_camera_height
    )
    
    aspect_ratio : FloatProperty(
        name="Aspect ratio", 
        description="Aspect ratio of the sensor",
        default=1,
        min=0.000,
        max=2,        
        update=update
    )

    focal_length : FloatProperty(
        name="Focal length", 
        description="Focal length",
        default=199.6,
        min=0.000,
        max=1000,  
        step=1,   
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

    view_port_size : FloatProperty(
        name="View port size",
        description="View port size",
        min=0,
        max=100,
        default=0.04,
        update=update
    )

    sensor_size : FloatProperty(
        name="Sensor size",
        description="Sensor size",
        min=0,
        max=1000,
        default=8,
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
#################################### Menu } ####################################

#################################### Operators { ####################################

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
        global updated_lps
        scene = context.scene
        mytool = scene.light_panel
        mytool.light_list.clear()
        
        for current_light in bpy.data.lights:
            current_light.animation_data_clear()
            bpy.data.lights.remove(current_light)
                      
        mytool.light_positions = updated_lps

        if not os.path.isfile(mytool.lp_file_path) and not mytool.nblp:
            self.report({"ERROR"})

        light_sources = bpy.context.scene.objects.get("light_sources")
        if not light_sources:
            light_sources = bpy.data.objects.new(name = "light_sources", object_data = None)
            scene.collection.objects.link(light_sources)        

        for idx in range(0, len(mytool.light_positions)):
            lightData = bpy.data.lights.new(name="RTI_light"+str(idx), type="SUN")

            current_light = bpy.data.objects.new(name="Light_{0}".format(idx), object_data=lightData)
            
            (x,y,z) = mytool.light_positions[idx]
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
        camera_data.dof.use_dof = True
        camera_data.type="PERSP"
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
        objects = [obj for obj in bpy.data.objects if obj.type != 'CAMERA' and obj.type !="SUN"]

        if len(objects) != 0:
            self.report({'ERROR'}, "Surface already exist in scene. Delete the old surface to add new surface")
            return {'FINISHED'}

        if scene.surface_panel.mesh_file_path.endswith(".OBJ") or scene.surface_panel.mesh_file_path.endswith(".obj"):
            bpy.ops.import_scene.obj(filepath=scene.surface_panel.mesh_file_path)
            ob = bpy.context.selected_objects[0]
            # ob.rotation_euler[0] = 0.0523599
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

        print("Setting new animation with",numLights)

        for i in range(0, numLights):
            current_light = bpy.data.objects[scene.light_panel.light_list[i]]
            current_light.data.energy = 0.0  

        
        for k in range(0, numLights):
            for i in range(0, numLights):
                current_light = bpy.data.objects[scene.light_panel.light_list[i]]
                if i != k:                
                    current_light.data.energy = 0.0  
                    current_light.data.keyframe_insert(data_path="energy", frame=k+1)
                else:
                    current_light.data.energy = scene.light_panel.light_strength  
                    current_light.data.keyframe_insert(data_path="energy", frame=k+1)
        
        
        return {'FINISHED'}

class acquire(Operator):
    bl_idname = "rti.acquire"
    bl_label = "Acquire"    
    bl_use_preview = True
    
    def execute(self, context):
        global updated_lps
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.preview_samples = 100
        bpy.context.scene.cycles.samples = 1000
        bpy.context.scene.cycles.use_preview_denoising = True
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.render.resolution_x = int(1920*context.scene.camera_panel.resolution_factor) 
        bpy.context.scene.render.resolution_y = int(context.scene.camera_panel.aspect_ratio*bpy.context.scene.render.resolution_x)
        bpy.context.scene.frame_start = 1
        bpy.context.scene.render.filepath = context.scene.acquisition_panel.output_path
        bpy.context.scene.render.image_settings.color_mode = 'BW'
        bpy.context.scene.render.image_settings.color_depth = '8'
        bpy.context.scene.render.use_overwrite = True
        context.scene.camera = context.scene.objects[context.scene.camera_panel.camera[0]] 
        
        if not context.scene.light_panel.nblp:
            bpy.context.scene.frame_end = len(context.scene.light_panel.light_list)
            bpy.ops.rti.set_animation()
            bpy.ops.render.render(animation=True, use_viewport = True, write_still=True)            
            bpy.ops.render.play_rendered_anim() 
            
        else:
            print("Executing NBLP")
            nblp = Nblp_2(context=context)
            
            # updated_lps = nblp.iterations[0].lps_cartesian 
            # nblp.generate_lp_file(iteration_nb=0, file_path=context.scene.acquisition_panel.output_path+"iteration_"+str(len(nblp.iterations)-1)+".lp")
            # bpy.context.scene.render.filepath = context.scene.acquisition_panel.output_path+nblp.iterations[0].filenames_subtext+"#.png"
            # bpy.ops.rti.create_lights()
            # bpy.context.scene.frame_end = len(context.scene.light_panel.light_list)
            # bpy.ops.rti.set_animation()
            # bpy.ops.render.render(animation=True, use_viewport = True, write_still=True)
            # bpy.ops.render.play_rendered_anim() 
            # nblp.iterations[0].plot_lps().savefig(context.scene.acquisition_panel.output_path+"30.png")
            # nblp.iterations[0].rename_files(context.scene.acquisition_panel.output_path)
            # nblp.calculate_entropies(len(nblp.iterations)-1,context.scene.acquisition_panel.output_path+"\\entropies\\")

            # nblp.dense_acquisition()
            # updated_lps = nblp.iterations[len(nblp.iterations)-1].lps_cartesian
            # nblp.generate_lp_file(iteration_nb=1, file_path=context.scene.acquisition_panel.output_path+"iteration_"+str(len(nblp.iterations)-1)+".lp")
            # bpy.context.scene.render.filepath = context.scene.acquisition_panel.output_path+"dense_acquisition\\"+nblp.iterations[len(nblp.iterations)-1].filenames_subtext+"#.png"
            # bpy.ops.rti.create_lights()
            # bpy.context.scene.frame_end = len(context.scene.light_panel.light_list)
            # bpy.ops.rti.set_animation()
            # bpy.ops.render.render(animation=True, use_viewport = True, write_still=True)
            # bpy.ops.render.play_rendered_anim()  
            # nblp.iterations[len(nblp.iterations)-1].plot_lps().savefig(context.scene.acquisition_panel.output_path+"dense_acquisition\\30.png")
            # nblp.iterations[len(nblp.iterations)-1].rename_files(context.scene.acquisition_panel.output_path+"dense_acquisition\\")  

            nblp.write_log(context.scene.acquisition_panel.output_path+"log.yaml")        


        return {'FINISHED'}

#################################### Operators } ####################################
#################################### Panels { #######################################
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
        global updated_lps
        layout = self.layout
        scene = context.scene
        lighttool = scene.light_panel
        layout.label(text="Light positions")
        layout.prop(lighttool, "nblp")
        layout.prop(lighttool, "light_strength")
        layout.prop(lighttool, "dome_radius")

        if not lighttool.nblp:
            layout.prop(lighttool,"lp_file_path")
            row = layout.row(align = True)  
            updated_lps = read_lp_file(lighttool.lp_file_path, lighttool.dome_radius)
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
        layout.prop(cameratool, "aspect_ratio", slider=True)
        layout.prop(cameratool, "focal_length", slider=True)
        layout.prop(cameratool, "subject")
        layout.prop(cameratool, "resolution_factor")
        layout.prop(cameratool, "view_port_size", slider=True)
        layout.prop(cameratool, "sensor_size", slider=True)
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

#################################### Panels } #######################################
    
#################################### Register classes { #######################################
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

#################################### Register classes } #######################################