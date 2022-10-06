from cmath import asin
from logging import critical, root
from pickletools import read_long1
from posixpath import abspath
from socket import inet_ntoa
from tracemalloc import start
import cv2
import random 
import numpy as np
import os
import glob 
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
from numpy import inner, savetxt
# conda install -c conda-forge dataclasses
from dataclasses import dataclass
from numpy.lib.stride_tricks import as_strided
import plotly.graph_objects as go
import plotly
from scipy.signal import argrelextrema
import copy
# conda install pandas
import plotly.express as px
from sympy import interpolate
import matlab.engine

##########################################################################################
# Params
Output_figures_loc = os.path.abspath(r"D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\coin\LDR_Homogeneous_20220926_205820_coin\dmd\dmd\figures")
Min_lp_distance_threshold = 0.008
Search_radius_for_calc_point_densities = 0.02
Point_density_threshold = 3
New_lp_file_name = "acquisition.lp"
Matlab_functions_dir = "C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\nblp\\SurfaceAdaptiveRTI\\matlab_functions\\"
Data_interpolation_sample_size = 200
Gradient_ascent_step_size = 0
Gradient_ascent_found_minimum = 99999
Gradient_ascent_found_maximum = 0
Theta_signal_validation_gradient_threshold = 10
Desired_validation_theta_acq_size = 20

##########################################################################################
class light_position():
    def __init__(self, **kwargs):
        allowed_args = ('x', 'y', 'z', 'az_radians', 'el_radians', 'az_degrees', 'el_degrees', 'radius', None)
        if set(kwargs.keys()).issubset(allowed_args) and len(kwargs.items())>2:
            self.__dict__.update(kwargs)
            if 'x' in kwargs.keys():
                self.cart2sph()
            if 'az_degrees' in kwargs.keys():
                self.az_radians = math.radians(self.az_degrees)
                self.el_radians = math.radians(self.el_degrees)
                self.sph2cart()
            if 'az_radians' in kwargs.keys():
                self.az_degrees = math.degrees(self.az_radians)
                self.el_degrees = math.degrees(self.el_radians)
                self.sph2cart()
        if kwargs==None:
            default_values = {'x':None, 'y':None, 'z':None, 'az_radians':None, 'el_radians':None, 'az_degrees':None, 'el_degrees':None, 'radius':None} 
            self.__dict__.update(default_values)

        rejected_keys = set(kwargs.keys()) - set(allowed_args)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor:{}".format(rejected_keys))

    def cart2sph(self):
        hxy = np.hypot(self.x, self.y)
        self.radius = np.hypot(hxy, self.z)
        self.el_radians = np.arctan2(self.z, hxy)
        self.az_radians = np.arctan2(self.y, self.x)
        self.az_degrees = math.degrees(self.az_radians)
        self.el_degrees = math.degrees(self.el_radians)

    def sph2cart(self):
        rcos_theta = self.radius * np.cos(self.el_radians)
        self.x = rcos_theta * np.cos(self.az_radians)
        self.y = rcos_theta * np.sin(self.az_radians)
        self.z = self.radius * np.sin(self.el_radians)
##########################################################################################
class light_positions():
    def __init__(self, **kwargs):
        allowed_args = ('lps', 'lp_file_path', None)
        if set(kwargs.keys()).issubset(allowed_args):
            self.__dict__.update(kwargs)
            if 'lp_file_path' in kwargs.keys():
                self.read_lp_file()

        if kwargs==None:
            default_values = {'lps':None, 'lp_file_path':None, 'img_file_names':None} 
            self.__dict__.update(default_values)

        rejected_keys = set(kwargs.keys()) - set(allowed_args)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor:{}".format(rejected_keys))
        
    def read_lp_file(self):
        self.lps = []
        self.img_file_names = []
        if not os.path.exists(self.lp_file_path):
            raise ValueError("Lp file path does not exist:{}".format(self.lp_file_path))
        try:
            file = open(self.lp_file_path)
        except RuntimeError as ex:
            error_report = "\n".join(ex.args)
            print("Caught error:", error_report)
            return {'ERROR'}
        rows = file.readlines()
        file.close()
        numLights = int(rows[0].split()[0])
        for idx in range(1, numLights + 1):
            cols = rows[idx].split()
            self.img_file_names.append(str(cols[0]))
            self.lps.append(light_position(x=float(cols[1]),y=float(cols[2]),z=float(cols[3])))

    def write_lp_file(self):
        data = str(len(self.lps))
        for i in range(0, len(self.lps)):
            data = data+"\n"+self.img_file_names[i]+" "+str(self.lps[i].x)+" "+str(self.lps[i].y)+" "+str(self.lps[i].z)
        with open(self.lp_file_path, 'w') as f:
            f.write(data)

    def plot_lps_3d(self):
        go_fig = go.Figure(data=[go.Scatter3d(x=[lp.x for lp in self.lps], y=[lp.y for lp in self.lps], z=[lp.z for lp in self.lps], mode="markers")])
        go_fig.update_layout(title='Light positions, '+str(len(self.lps)), autosize=True)
        file_name = os.path.basename(self.lp_file_path).split(".lp")[0]
        go_fig.write_html(Output_figures_loc+"\\"+file_name+"_3d.html")

    def plot_lps_projected(self):
        go_fig = go.Figure(data=[go.Scatterpolar(r=[lp.el_degrees for lp in self.lps], theta=[lp.az_degrees for lp in self.lps], mode = 'markers')])
        go_fig.update_layout(title='Light positions, '+str(len(self.lps)), autosize=True)
        file_name = os.path.basename(self.lp_file_path).split(".lp")[0]        
        go_fig.write_html(Output_figures_loc+"\\"+file_name+"_projected.html")

    class unqiue_light_positions:
        def __init__(self,lps):
            self.data = [[lp.x,lp.y,lp.z] for lp in lps ]            
        
        def get_unqiue_lps(self):
            result = []
            for element in self.data:
                if all(self.condition(element,other) for other in result):
                    result.append(element)
            return result

        def condition(self,xs,ys):    
            return sum((x-y)*(x-y) for x,y in zip(xs,ys)) > Min_lp_distance_threshold*Min_lp_distance_threshold

    class point_densities:
        def __init__(self,lps):
            self.data = [[lp.x,lp.y,lp.z] for lp in lps ]
            
        
        def get_densities(self):
            densities = []
            for element in self.data:
                within_search_radius = np.array([self.condition(element,other) for other in self.data])
                densities.append(len(within_search_radius[within_search_radius==True])-1)
            return densities

        def condition(self,xs,ys):    
            return sum((x-y)*(x-y) for x,y in zip(xs,ys)) < Search_radius_for_calc_point_densities * Search_radius_for_calc_point_densities

##########################################################################################
class image:
    def __init__(self,**kwargs):
        allowed_args = ('img_mat', 'img_vect', 'img_path', 'h', 'w', None)
        if set(kwargs.keys()).issubset(allowed_args):
            self.__dict__.update(kwargs)
            if 'img_path' in kwargs.keys():
                self.img_mat = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            if 'img_path' in kwargs.keys() or 'img_mat' in kwargs.keys():
                self.h, self.w = self.img_mat.shape
                self.img_vect = np.reshape(self.img_mat, (1, self.h * self.w))[0]
            if 'img_vect' in kwargs.keys() and 'h' in kwargs.keys() and 'w' in kwargs.keys():
                self.img_mat = np.reshape(self.img_vect, (self.h,self.w))
        if kwargs==None:
            default_values = {'img_mat':None, 'img_vect':None, 'img_path':None,'h':None, 'w':None } 
            self.__dict__.update(default_values)
                
        rejected_keys = set(kwargs.keys()) - set(allowed_args)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor:{}".format(rejected_keys))

    def get_roi(self,**kwargs):
        allowed_args = ('rect', 'w_from_center', 'h_from_center', None)
        if set(kwargs.keys()).issubset(allowed_args):
            if 'rect' in kwargs.keys():
                self.roi_rect = kwargs.get("rect")                
            if 'w_from_center' in kwargs.keys() and 'h_from_center' in kwargs.keys():
                self.roi_rect = {'x':int(self.w/2) - kwargs.get("w_from_center"), 
                                'w':kwargs.get("w_from_center"), 
                                'y':int(self.h/2) - kwargs.get("h_from_center"), 
                                'h':kwargs.get("h_from_center")}                
            img = self.img_mat[self.roi_rect['y']:self.roi_rect['y']+self.roi_rect['h'], + self.roi_rect['x']:self.roi_rect['x']+self.roi_rect['w']]
            self.roi_img = image(img_mat=img)
            return self.roi_img

    def resize_img(self, img, **kwargs):
        allowed_args = ('ratio', 'desired_w')
        if set(kwargs.keys()).issubset(allowed_args):
            if 'ratio' in kwargs.keys():
                ratio = int(1/kwargs.get('ratio'))
                self.resized_image = image(img_mat=img[::int(ratio), ::int(ratio)])
                return self.resized_image
            elif 'desired_w' in kwargs.keys():
                slice_step = img.shape[1]/kwargs.get('desired_w')
                self.resized_image = image(img_mat=img[::int(slice_step), ::int(slice_step)])
                return self.resized_image
            else:
                raise ValueError("Invalid arguments in constructor:{}".format(kwargs))    
        else:
            raise ValueError("Invalid arguments in constructor:{}".format(kwargs))

##########################################################################################
class acquisition:
    def __init__(self,**kwargs):
        allowed_args = ('data', None)
        if set(kwargs.keys()).issubset(allowed_args):
            self.__dict__.update(kwargs)
            if 'data' in kwargs.keys():
                self.lps = light_positions(lp_file_path=self.data['dir']+"\\"+self.data['lp_file_name'])
                if not os.path.exists(self.data['dir']+"\\nblp\\"):
                    os.mkdir(self.data['dir']+"\\nblp\\")
                    os.mkdir(self.data['dir']+"\\nblp\\figures")
        elif kwargs==None:
            default_values = {'lps':None, 'data':None} 
            self.__dict__.update(default_values)
        else:
            raise ValueError("Invalid arguments in constructor:{}".format(kwargs))

    def get_images(self):
        return ([image(img_path=self.data['dir']+"\\"+img_file_name) for img_file_name in self.lps.img_file_names])
    
    def get_roi_images(self,**kwargs):
        if 'roi' in kwargs.keys():
            org_images = [image(img_path=self.data['dir']+"\\"+img_file_name) for img_file_name in self.lps.img_file_names]
            return [img.get_roi(**kwargs['roi']) for img in org_images]

    def get_resized_images(self, **kwargs):
        if 'resize' in kwargs.keys():
            org_images = [image(img_path=self.data['dir']+"\\"+img_file_name) for img_file_name in self.lps.img_file_names]
            return [img.resize_img(img.img_mat,**kwargs['resize']) for img in org_images]

    def copy_acquisition(self, new_data, **kwargs):        
        if not os.path.exists(new_data['dir']):
            os.mkdir(new_data['dir'])
            os.mkdir(new_data['dir']+"\\nblp")
            os.mkdir(new_data['dir']+"\\nblp\\figures")
        allowed_args = ('roi', 'resize', None)
        if set(kwargs.keys()).issubset(allowed_args):
            org_images = self.get_images()
            if 'roi' in kwargs.keys() and 'resize' not in kwargs.keys():
                roi_images = [img.get_roi(**kwargs['roi']) for img in org_images]
                for i in range(0,len(roi_images)):
                    cv2.imwrite(new_data['dir']+"\\"+self.lps.img_file_names[i], roi_images[i].img_mat)
            else: 
                resized_images = [img.resize_img(img.img_mat,**kwargs['resize']) for img in org_images]
                if 'roi' in kwargs.keys():
                    roi_images = [img.get_roi(**kwargs['roi']) for img in resized_images]                    
                    for i in range(0,len(roi_images)):
                        cv2.imwrite(new_data['dir']+"\\"+self.lps.img_file_names[i], roi_images[i].img_mat)
                else:
                    for i in range(0,len(resized_images)):
                        cv2.imwrite(new_data['dir']+"\\"+self.lps.img_file_names[i], resized_images[i].img_mat)
            if len(kwargs.keys()) <1:
                for i in range(0,len(org_images)):
                    cv2.imwrite(new_data['dir']+"\\"+self.lps.img_file_names[i], np.array(org_images[i].img_mat))
            os.system("copy "+self.data['dir']+"\\"+self.data['lp_file_name']+" "+ new_data['dir']+"\\"+new_data['lp_file_name'])
        else:
            raise ValueError("Invalid arguments in constructor:{}".format(kwargs))

##########################################################################################
class relighting:
    def __init__(self,acquisition, matlab_engine):
        self.acquisition = acquisition
        self.eng = matlab_engine
        self.azimuths = [lp.az_radians for lp in self.acquisition.lps.lps]
        self.elevations = [lp.el_radians for lp in self.acquisition.lps.lps]
        self.acq_images = acquisition.get_images()
        self.acq_images_mat = np.transpose(np.array([img.img_vect[0] for img in self.acq_images]))

    def save_relighted_images(self, relighted_images_mat):
        save_path = self.acquisition.data['dir']+"\\nblp\\relighted_img_"
        file_names = self.acquisition.lps.img_file_names  
        [cv2.imwrite(save_path+file_names[i], np.reshape(relighted_images_mat[:, i], (self.acq_images[i].img_mat.shape))) for i in range(0,len(self.azimuths))]

    def matlab2python_data(self,relighted_images_mat_matlab):
        relighted_images_mat = np.transpose(np.array(relighted_images_mat_matlab._data).reshape(relighted_images_mat_matlab.size, order='F'))
        relighted_images_mat[relighted_images_mat<0] = 0
        relighted_images_mat[relighted_images_mat>1] = 1   
        relighted_images_mat = 255*relighted_images_mat
        self.save_relighted_images(relighted_images_mat=relighted_images_mat)
        return (relighted_images_mat)
        

    def get_dmd_relighted_images(self):
        self.eng.get_dmd_coeffs(self.acquisition.data['dir'])
        relighted_images_mat_matlab = self.eng.get_dmd_interpolated_img(matlab.double(self.azimuths), matlab.double(self.elevations))
        return self.acq_images_mat, self.matlab2python_data(relighted_images_mat_matlab)

    def get_ptm_relighted_images(self):
        self.eng.get_ptm_coeffs(self.acquisition.data['dir'])        
        relighted_images_mat_matlab = self.eng.get_ptm_interpolated_img(matlab.double(self.azimuths), matlab.double(self.elevations))        
        return self.acq_images_mat, self.matlab2python_data(relighted_images_mat_matlab)


##########################################################################################
class loss_function:
    def __init__(self,acquisition,**kwargs):
        self.acquisition = acquisition

    def get_model_fit_loss(self, model):
        matlab_engine = matlab.engine.start_matlab()
        for subdir, dirs, files in os.walk(Matlab_functions_dir):
            matlab_engine.addpath(subdir)
        org_images_mat = None
        relighted_images_mat = None
        if model=='dmd':
            org_images_mat, relighted_images_mat = relighting(self.acquisition,matlab_engine).get_dmd_relighted_images()
        else:
            org_images_mat, relighted_images_mat = relighting(self.acquisition,matlab_engine).get_ptm_relighted_images()
        error_matrix = np.abs(np.subtract(org_images_mat, relighted_images_mat))
        normalized_error_matrix = error_matrix/error_matrix.sum(axis=1)[:, np.newaxis]
        loss_function = np.sum(normalized_error_matrix, axis=0)/error_matrix.shape[0]
        absolute_loss_vector = np.sum(error_matrix, axis=0)/error_matrix.shape[0]
        self.save_model_fit_loss_plot(loss_function, "model_fitting_loss_function")
        self.save_model_fit_loss_plot(absolute_loss_vector, "model_fitting_absolute_loss")
        return {'loss_function': loss_function, 'absolute_loss_vector': absolute_loss_vector}

    def get_theta_gradient_loss(self):
        acq_images = self.acquisition.get_images()
        acq_images_mat = np.transpose(np.array([img.img_vect[0] for img in acq_images]))
        #append acq_images_mat with itself thrice to make it look circular array
        gradient_matrix = np.gradient(np.tile(acq_images_mat,3), axis=1)[:,acq_images_mat.shape[1]:2*acq_images_mat.shape[1]]
        weight_matrix= (gradient_matrix*gradient_matrix.shape[0])/np.sum(gradient_matrix,axis=0)
        weighted_gradient_matrix = np.multiply(gradient_matrix, weight_matrix)
        loss_function = np.sum(weighted_gradient_matrix, axis=0)/gradient_matrix.shape[0]        
        self.save_theta_gradient_loss_plot(loss_function=loss_function,title="theta_gradient_plot")
        return {'loss_function':loss_function}

    def interpolate_data(self, loss_function):
        x = np.linspace(-1,1,Data_interpolation_sample_size)
        y = np.linspace(-1,1,Data_interpolation_sample_size)
        x_grid, y_grid = np.meshgrid(x, y)
        xs = [lp.x for lp in self.acquisition.lps.lps]
        ys = [lp.y for lp in self.acquisition.lps.lps]
        z_grid = griddata((xs,ys), loss_function, (x_grid, y_grid), method='cubic')
        return x_grid, y_grid, z_grid

    def save_model_fit_loss_plot(self, loss_function, title):
        x_grid, y_grid, z_grid = self.interpolate_data(loss_function)
        xs = [lp.x for lp in self.acquisition.lps.lps]
        ys = [lp.y for lp in self.acquisition.lps.lps]
        go_fig = go.Figure(data=[go.Surface(z=z_grid, y=y_grid,x=x_grid, opacity=0.5), go.Scatter3d(x=xs, y=ys, z=loss_function, mode="markers")])
        mean_loss = np.nanmean(loss_function)        
        go_fig.update_layout(title=title+', mean loss =  '+str(mean_loss), autosize=True, xaxis_title ='lu', yaxis_title = 'lv')
        go_fig.write_html(self.acquisition.data['dir']+"\\nblp\\figures\\"+title+".html")

    def save_theta_gradient_loss_plot(self, loss_function, title):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        azimuths = [lp.az_degrees for lp in self.acquisition.lps.lps]
        mean_elevations = np.mean([lp.el_degrees for lp in self.acquisition.lps.lps])
        ax.plot(azimuths, loss_function,'-', label="Phi =  "+ str(mean_elevations))
        ax.set_rlabel_position(-22.5)  
        ax.grid(True)
        ax.legend()
        ax.set_title(title)
        plt.savefig(self.acquisition.data['dir']+"\\nblp\\figures\\"+title+".png")

##########################################################################################
class validation:
    def __init__(self, **kwargs):
        default_values = {'nblp_acq':None, 'dense_acq':None} 
        self.__dict__.update(default_values)
        allowed_args = ('nblp_acq', 'dense_acq', None)
        if set(kwargs.keys()).issubset(allowed_args):
            self.__dict__.update(kwargs)                   
        
        rejected_keys = set(kwargs.keys()) - set(allowed_args)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor:{}".format(rejected_keys))

class dense_acquisition_measurements:
    def __init__(self, **kwargs):
        default_values = {'dense_acq':None, 'pixels_list_1d':None, 'pixels_mask_for_analysis':None,'pixels_indices_2d':None} 
        self.__dict__.update(default_values)
        allowed_args = ('dense_acq', 'pixels_list_1d', 'pixels_mask_for_analysis', 'pixels_indices_2d', None)
        if set(kwargs.keys()).issubset(allowed_args):
            self.__dict__.update(kwargs) 

        self.h,self.w = self.dense_acq.get_images()[0].img_mat.shape
        self.xs = [lp.x for lp in self.dense_acq.lps.lps]
        self.ys = [lp.y for lp in self.dense_acq.lps.lps]
        self.azimuths = [lp.az_degrees for lp in self.dense_acq.lps.lps]
        self.elevations = [lp.el_degrees for lp in self.dense_acq.lps.lps]
        self.save_metrics_path = self.dense_acq.data['dir']+"\\metrics"    
        if not os.path.exists(self.save_metrics_path):
            os.mkdir(self.save_metrics_path)
        if 'pixels_list_1d' in kwargs.keys():
            self.pixels_list_1d = kwargs['pixels_list_1d']
            self.pixels_indices_2d = [(int(self.pixels_list_1d[i]/self.w),self.pixels_list_1d[i]%self.w) for i in range(0,len(self.pixels_list_1d))]
        if 'pixels_mask_for_analysis' in kwargs.keys():
            print('mask code to be done')
        if 'pixels_indices_2d' in kwargs.keys():
            self.pixels_list_1d = [self.w*p[0]+int(self.w%p[1]) for p in kwargs['pixels_indices_2d']]
        
        rejected_keys = set(kwargs.keys()) - set(allowed_args)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor:{}".format(rejected_keys))

    def get_dense_acquisition_measurements(self):
        self.pixels_intensities_matrix = [list(map(img.img_vect.__getitem__, self.pixels_list_1d)) for img in self.dense_acq.get_images()]
        self.find_best_light_positions_in_theta_space()
        # self.find_best_light_positions()

    def plot_pixel_intensities_2d(self, signal, title):
        go_fig = go.Figure(data=[go.Scatterpolar(r=[lp.el_degrees for lp in self.dense_acq.lps.lps], theta=[lp.az_degrees for lp in self.dense_acq.lps.lps], mode = 'markers', marker=dict(color=signal, showscale=True) )])
        go_fig.update_layout(title=title, autosize=True)
        go_fig.write_html(self.save_metrics_path+"\\"+title+"_projected.html")

    def radial_polar_plot_2d(self, thetas, phis, signal, title):
        go_fig = go.Figure(data=[go.Scatterpolar(r=phis, theta=thetas, mode = 'markers', marker=dict(color=signal, showscale=True) )])
        go_fig.update_layout(title=title, autosize=True)
        go_fig.write_html(self.save_metrics_path+"\\"+title+"_projected.html")

    def plot_pixel_intensities_3d(self, grid_data, signal, title):        
        x_grid, y_grid, z_grid = grid_data['x_grid'], grid_data['y_grid'], grid_data['z_grid']
        go_fig = go.Figure(data=[go.Surface(z=z_grid, y=y_grid,x=x_grid, opacity=0.5), go.Scatter3d(x=[lp.x for lp in self.dense_acq.lps.lps], y=[lp.y for lp in self.dense_acq.lps.lps], z=signal, mode="markers",marker=dict(color=signal, size=3))])
        go_fig.update_layout(title=title, autosize=True, xaxis_title ='lu', yaxis_title = 'lv')
        go_fig.write_html(self.save_metrics_path+"\\"+title+".html")

    def interpolate_intensities(self, pixel_intensities):
        x = np.linspace(-1,1,max(Data_interpolation_sample_size,len(self.dense_acq.lps.lps)))
        y = np.linspace(-1,1,max(Data_interpolation_sample_size,len(self.dense_acq.lps.lps)))
        x_grid, y_grid = np.meshgrid(x, y)        
        z_grid = griddata((self.xs,self.ys), pixel_intensities, (x_grid, y_grid), method='cubic')
        return x_grid, y_grid, z_grid

    def find_best_light_positions_in_theta_space(self):
        self.theta_idx_step_size = len(self.xs)/Desired_validation_theta_acq_size
        for i in range(0, len(self.pixels_list_1d)):
            i_pixel_intensities = [intensity[i] for intensity in self.pixels_intensities_matrix]
            self.plot_pixel_intensities_2d(signal=i_pixel_intensities, title="Pixl_intensities_2D_("+str(self.pixels_indices_2d[i])+")")
            x_grid, y_grid, z_grid = self.interpolate_intensities(i_pixel_intensities)
            self.plot_pixel_intensities_3d(grid_data={'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': z_grid}, signal=i_pixel_intensities,  title="Pixl_intensities_3D_("+str(self.pixels_indices_2d[i])+")" )
            nblp_points = []
            current_theta_idx = 0
            nblp_points.append(current_theta_idx)
            while current_theta_idx<len(self.azimuths)-1:
                print("e: ", current_theta_idx)
                print("f: ", len(self.azimuths)-1)
                current_theta_idx = self.next_theta(current_theta_idx=current_theta_idx, signal=i_pixel_intensities)
                nblp_points.append(current_theta_idx)
            signal = [i_pixel_intensities[nblp_points[i]] for i in range(0,len(nblp_points))]
            phis = [self.elevations[nblp_points[i]] for i in range(0,len(nblp_points))]
            thetas = [self.azimuths[nblp_points[i]] for i in range(0,len(nblp_points))]
            self.radial_polar_plot_2d(thetas=thetas,phis=phis,signal=signal,title="theoretical_nblps_("+str(self.pixels_indices_2d[i])+")_nb_pts-"+ str(len(nblp_points)))

    def next_theta(self,current_theta_idx, signal):
        next_theta_idx = int(current_theta_idx+self.theta_idx_step_size)
        if next_theta_idx>len(signal)-1:
            next_theta_idx = len(signal)-1
        while(abs(signal[next_theta_idx]-signal[current_theta_idx])>Theta_signal_validation_gradient_threshold and current_theta_idx!=next_theta_idx-1):
            print("a: ",(abs(signal[next_theta_idx]-signal[current_theta_idx])))
            print("b: ",Theta_signal_validation_gradient_threshold )
            print("c: ", current_theta_idx)
            print("d: ", next_theta_idx)
            next_theta_idx = next_theta_idx-1
        return next_theta_idx

    def find_best_light_positions_in_u_v_space(self):
        for i in range(0, len(self.pixels_list_1d)):
            i_pixel_intensities = [intensity[i] for intensity in self.pixels_intensities_matrix]
            self.plot_pixel_intensities_2d(signal=i_pixel_intensities, title="Pixl_intensities_2D_("+str(self.pixels_indices_2d[i])+")")
            x_grid, y_grid, z_grid = self.interpolate_intensities(i_pixel_intensities)
            self.plot_pixel_intensities_3d(grid_data={'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': z_grid}, signal=i_pixel_intensities,  title="Pixl_intensities_3D_("+str(self.pixels_indices_2d[i])+")" )
            g, gx, gy, signal_gx, signal_gy, signal_g = self.compute_gradients_distribution_in_uv_space(z_grid=z_grid, pixel_intensities=i_pixel_intensities)
            gx_grid_data = {'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': gx}
            gy_grid_data = {'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': gy}
            g_grid_data = {'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': g}
            self.plot_pixel_intensities_2d(signal=signal_gx, title="gradient_x_2D_("+str(self.pixels_indices_2d[i])+")")
            self.plot_pixel_intensities_2d(signal=signal_gy, title="gradient_y_2D_("+str(self.pixels_indices_2d[i])+")")
            self.plot_pixel_intensities_2d(signal=signal_g, title="gradient_2D_("+str(self.pixels_indices_2d[i])+")")
            self.plot_pixel_intensities_3d(grid_data=gx_grid_data, signal=signal_gx, title="gradient_3D_("+str(self.pixels_indices_2d[i])+")")
            self.plot_pixel_intensities_3d(grid_data=gy_grid_data, signal=signal_gy, title="gradient_3Dx_("+str(self.pixels_indices_2d[i])+")")
            self.plot_pixel_intensities_3d(grid_data=g_grid_data, signal=signal_g, title="gradient_3Dy_("+str(self.pixels_indices_2d[i])+")")

    def compute_gradients_distribution_in_uv_space(self, z_grid, pixel_intensities):   
        gx, gy = np.gradient(z_grid)
        g = gx+gy
        # assuming the linspace limits are -1 to 1
        marker_indices = [(int((len(self.xs)*(1+self.xs[i]))/2), int((len(self.ys)*(1+self.ys[i]))/2)) for i in range(0,len(self.xs))]
        signal_gx = [gx[marker_indices[i][1]][marker_indices[i][0]] for i in range(0,len(self.xs))]
        signal_gy = [gy[marker_indices[i][1]][marker_indices[i][0]] for i in range(0,len(self.xs))]
        signal_g = [g[marker_indices[i][1]][marker_indices[i][0]] for i in range(0,len(self.xs))]
        return g, gx, gy, signal_gx, signal_gy, signal_g

        



data = {'dir':os.path.abspath(r'E:\acquisitions\nblp_v2\rust_coarse_dense'), 'lp_file_name': "iteration_0.lp"}
data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\coin\LDR_20221005_144547_dense_ring'), 'lp_file_name': "acquisition.lp"}
k = acquisition(data=data)

# pixels = [(10,15),(100,200), (400,500)]
pixels = [(1504,2056)]

# pixels = [100,1,700, 1800]
l = dense_acquisition_measurements(dense_acq = k,pixels_indices_2d= pixels)
s = l.get_dense_acquisition_measurements()





#     def dense_acq_intensities_distribution(self):
#         images = self.dense_acq.get_images()
#         mean_intensities_distribution = [np.mean(images[i].img_vect) for i in range(0,len(images))]
#         self.plot_mean_intensities_distribution(data=mean_intensities_distribution, title="mean_intensities_distribution_dense_acq")
#         return mean_intensities_distribution

#     def plot_mean_intensities_distribution(self, data, title):
#         fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#         azimuths = [lp.az_degrees for lp in self.dense_acq.lps.lps]
#         mean_elevations = np.mean([lp.el_degrees for lp in self.dense_acq.lps.lps])
#         ax.plot(azimuths, data,'-', label="Phi =  "+ str(mean_elevations))
#         ax.set_rlabel_position(-22.5)  
#         ax.grid(True)
#         ax.legend()
#         ax.set_title(title)
#         plt.savefig(self.dense_acq.data['dir']+"\\nblp\\figures\\"+title+".png")

# ##########################################################################################

# class optimize:
#     def __init__(self, **kwargs):
#         allowed_args = ('u', 'v', 'loss_function', None)
#         if set(kwargs.keys()).issubset(allowed_args):
#             self.__dict__.update(kwargs)
            
#         rejected_keys = set(kwargs.keys()) - set(allowed_args)
#         if rejected_keys:
#             raise ValueError("Invalid arguments in constructor:{}".format(rejected_keys))
#         u = self.u
#         v = self.v        
#         self.az = np.arcsin(math.sqrt((u*u) + (v*v)))
#         self.el = np.arctan(v/u)
        
        
#     def get_nblp(self):
#         lp = light_position(az_radians=self.az, el_radians=self.el, radius=1.0)
#         a,b,c = lp.x,lp.y,lp.z
#         nblp = light_position(x=-c, y=-b, z=a)
#         return nblp



# data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\rust_coarse\theta_gradient\sparse'), 'lp_file_name': "iteration_0.lp"}
# k = acquisition(data=data)
# l = loss_function(k)
# l.get_theta_gradient_loss()

# data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\rust_coarse\theta_gradient\dense'), 'lp_file_name': "iteration_0.lp"}
# k = acquisition(data=data)
# l = evaluation(dense_acq=k)
# s = l.dense_acq_intensities_distribution()

# nblp = optimize(u=-0.63671,v=-0.77064).get_nblp()
# print(nblp.x, nblp.y, nblp.z, nblp.az_degrees, nblp.el_degrees)

# data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\rust_coarse\dmd\iteration_1'), 'lp_file_name': "acquisition.lp"}
# k = acquisition(data=data)
# l = loss_function(k)
# l.get_model_fit_loss('dmd')




##########################################################################################


# data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\rust_coarse\dmd'), 'lp_file_name': "acquisition.lp"}
# k = acquisition(data=data)
# l = loss_function(k)
# l.get_model_fit_loss(eng, 'dmd')

# data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\rust_coarse\theta_gradient\sparse'), 'lp_file_name': "iteration_0.lp"}
# k = acquisition(data=data)
# l = loss_function(k)
# l.get_theta_gradient_loss()

# data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\rust_coarse\theta_gradient\dense'), 'lp_file_name': "iteration_0.lp"}
# k = acquisition(data=data)
# l = evaluation(dense_acq=k)
# s = l.dense_acq_intensities_distribution()





# data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\coin\LDR_Homogeneous_20220926_205820_coin\dmd'), 'lp_file_name': "iteration0.lp"}
# k = acquisition(data=data)
# print(len(k.get_images()))
# print(k.get_images()[0].img_mat.shape)
# roi = {'rect':{'x':10,'y':10,'w':300,'h':200}}
# print(k.get_roi_images(roi=roi)[0].img_mat.shape)
# resize = {'ratio':0.5}
# print(k.get_resized_images(resize=resize)[0].img_mat.shape)
# new_data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\coin\LDR_Homogeneous_20220926_205820_coin\dmd\rois'), 'lp_file_name': "iteration0.lp"}
# roi_rect = {'rect':{'x':10,'y':10,'w':300,'h':200}}
# # k.copy_acquisition(new_data)
# # k.copy_acquisition(new_data, roi=roi_rect)
# resize = {'ratio':0.5}
# roi = {'w_from_center':50,'h_from_center':300}
# k.copy_acquisition(new_data, resize=resize, roi=roi_rect)

# k.copy_acquisition(new_data, roi=roi)

# a = {'x':10,'y':10,'z':10}
     
# print(a.items())
# k = light_position(**a)


# img = image(img_path = os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\coin\LDR_Homogeneous_20220926_205820_coin\dmd\Annular_Light.png'))
# rect = {'x':100, 'y':100, 'h':100,'w':100}
# cv2.imshow("jhkjh", img.get_roi(w_from_center=100, h_from_center=100))
# cv2.imshow("asdasd", img.resize_img(img.img_mat, desired_w=100))
# cv2.waitKey(0)


# lps = []
# for i in range(0,5):
#     lps.append(lp(x=i,y=i+0.5,z=i+1))

# lps.append(lp(x=50,y=5,z=i+1))
# lps.append(lp(x=1,y=2,z=3))

# k = light_positions()
# k.lps = lps

# densities = k.point_densities(k.lps).get_densities()

# print(densities)
    
# lps = light_positions(lp_file_path=os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\coin\LDR_Homogeneous_20220926_205820_coin\dmd\iteration0.lp'))
# lps.plot_lps_3d()
# lps.plot_lps_projected()
        