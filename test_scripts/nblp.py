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
from numpy import inner, linspace, savetxt
# conda install -c conda-forge dataclasses
from dataclasses import dataclass
from numpy.lib.stride_tricks import as_strided
import plotly.graph_objects as go
import plotly
from scipy.signal import argrelextrema
import copy
# conda install pandas
import plotly.express as px
from sympy import interpolate, sign
# import matlab.engine
from scipy.fft import fft, fftfreq
# conda install numexpr
import numexpr as ne
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import base64
import io 



##########################################################################################
# Params
Output_figures_loc = os.path.abspath(r"D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\coin\LDR_Homogeneous_20220926_205820_coin\dmd\dmd\figures")
Min_lp_distance_threshold = 0.015
Search_radius_for_calc_point_densities = 0.008
Point_density_threshold = 3
New_lp_file_name = "acquisition.lp"
Matlab_functions_dir = "C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\nblp\\SurfaceAdaptiveRTI\\matlab_functions\\"
Data_interpolation_sample_size = 200
Gradient_ascent_step_size = 0
Gradient_ascent_found_minimum = 99999
Gradient_ascent_found_maximum = 0
Desired_validation_theta_acq_size = 3
GRAD_CRITERION = 50
DIFF_CRITERION = 120
SAMPLE_FREQ_CRITERION = 180

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

    class unique_light_positions:
        def __init__(self, lps):
            self.data = [[lp.x,lp.y,lp.z] for lp in lps ]          
            self.unique_element_indices = []            
        def get_unique_lps(self):
            result = []
            data_filtered = copy.deepcopy(self.data)
            for idx, element in enumerate(self.data):
                if element in data_filtered:
                    result.append(light_position(x=element[0], y=element[1], z=element[2]))
                    self.unique_element_indices.append(idx)
                    elements_to_remove = [self.condition(element,other) for other in data_filtered]
                    data_filtered = [d for (d, remove) in zip(data_filtered, elements_to_remove) if not remove]                
            return result
        def condition(self,xs,ys):    
            return sum((x-y)*(x-y) for x,y in zip(xs,ys)) < Min_lp_distance_threshold*Min_lp_distance_threshold 

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
                self.roi_rect = {'x':int(self.w/2) - int(kwargs.get("w_from_center")/2), 
                                'w':kwargs.get("w_from_center"), 
                                'y':int(self.h/2) - int(kwargs.get("h_from_center")/2), 
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
    
    def get_acquisition_matrix(self):
        return (np.array([img.img_vect for img in self.get_images()]).T)

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
        # matlab_engine = matlab.engine.start_matlab()
        matlab_engine = None
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

##########################################################################################
class dense_ring_acquisition_measurements:
    def __init__(self, **kwargs):
        default_values = {'dense_acq':None, 'pixels_list_1d':None, 'pixels_mask_for_analysis':None,'pixels_indices_2d':None} 
        self.__dict__.update(default_values)
        allowed_args = ('dense_acq', 'pixels_list_1d', 'pixels_mask_for_analysis', 'pixels_indices_2d', None)
        if set(kwargs.keys()).issubset(allowed_args):
            self.__dict__.update(kwargs) 

        if dense_acq is not None:
            self.h,self.w = self.dense_acq.get_images()[0].img_mat.shape
            self.pixels_intensities_matrix = self.dense_acq.get_acquisition_matrix()
            
            self.xs = [lp.x for lp in self.dense_acq.lps.lps]
            self.ys = [lp.y for lp in self.dense_acq.lps.lps]
            self.azimuths = [lp.az_degrees for lp in self.dense_acq.lps.lps]
            self.elevations = [lp.el_degrees for lp in self.dense_acq.lps.lps]
            self.save_metrics_path = self.dense_acq.data['dir']+"\\metrics"    
            if not os.path.exists(self.save_metrics_path):
                os.mkdir(self.save_metrics_path)
            self.metrics_log = open(self.save_metrics_path+"\\metric_logs.txt","w")
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

    def get_dense_ring_acquisition_measurements(self):
        start_t = time.time()
        print("Loading pixels")
        if self.pixels_list_1d is not None:
            self.pixels_intensities_matrix = np.array([list(map(img.img_vect.__getitem__, self.pixels_list_1d)) for img in self.dense_acq.get_images()]).T
        print("Finished loading the pixel intensities to pixels intensiteis matrix in ", time.time()-start_t, "s")
        self.find_best_light_positions_in_theta_space()        

    def plot_pixel_intensities_2d(self, lps, signal, title):
        go_fig = go.Figure(data=[go.Scatterpolar(r=[lp.el_degrees for lp in lps], theta=[lp.az_degrees for lp in lps], mode = 'markers', marker=dict(color=signal, showscale=True) )])
        go_fig.update_layout(title=title, autosize=True)
        return go_fig
        
    def plot_distribution(self, histograms, group_labels ):
        x = [list(hist[1]) for hist in histograms]
        y = [list(hist[0]) for hist in histograms]
        data = []
        for i in range(0,len(x)):
            x[i] = (np.array(x[i])[1:] + np.array(x[i])[:-1]) / 2
            pd_data = {
                "gradients": x[i],
                "counts": y[i]
            }
            data.append(pd.DataFrame(data=pd_data).assign(group = group_labels[i]))
        df = pd.concat(data)
        return px.histogram(data_frame=df, x="gradients", y="counts", marginal="violin", color='group')        
        
    def radial_polar_plot_2d(self, lps, signal, title):
        go_fig = go.Figure(data=[go.Scatterpolar(r=signal, theta=[lp.az_degrees for lp in lps], mode = 'lines'), go.Scatterpolar(r=signal, theta=[lp.az_degrees for lp in lps], mode = 'markers', marker=dict(color=signal, showscale=True))])
        go_fig.update_layout(title=title, autosize=True)
        return go_fig

    def plot_pixel_intensities_3d(self, grid_data, signal, title):        
        x_grid, y_grid, z_grid = grid_data['x_grid'], grid_data['y_grid'], grid_data['z_grid']
        go_fig = go.Figure(data=[go.Surface(z=z_grid, y=y_grid,x=x_grid, opacity=0.5), go.Scatter3d(x=[lp.x for lp in self.dense_acq.lps.lps], y=[lp.y for lp in self.dense_acq.lps.lps], z=signal, mode="markers",marker=dict(color=signal, size=3))])
        go_fig.update_layout(title=title, autosize=True, xaxis_title ='lu', yaxis_title = 'lv')
        return go_fig

    def interpolate_intensities(self, pixel_intensities):
        x = np.linspace(-1,1,max(Data_interpolation_sample_size,len(self.dense_acq.lps.lps)))
        y = np.linspace(-1,1,max(Data_interpolation_sample_size,len(self.dense_acq.lps.lps)))
        x_grid, y_grid = np.meshgrid(x, y)        
        z_grid = griddata((self.xs,self.ys), pixel_intensities, (x_grid, y_grid), method='cubic')
        return x_grid, y_grid, z_grid

    def plot_overlaid_heat_map(self, data, overlay_img):
        data = ((data - np.min(data)) / (np.max(data) - np.min(data)))*255
        dst = cv2.addWeighted(data.astype(np.uint8), 0.9, overlay_img, 0.1, 0.0)
        return px.imshow(dst)

    def analyze_dense_acq(self):
        self.theta_idx_step_size = int(len(self.xs)/Desired_validation_theta_acq_size)
        self.metrics_log.write("Step size: "+ str(self.theta_idx_step_size)+"\n")
        acq_gradients = np.gradient(self.pixels_intensities_matrix, axis=1)
        acq_max_gradients = np.amax(np.abs(acq_gradients), axis=1)
        self.plot_overlaid_heat_map(acq_max_gradients.reshape(self.h, self.w), self.dense_acq.get_images()[0].img_mat).write_html(self.save_metrics_path+"\\Specularity_map.html")
        acq_gradients = acq_gradients.flatten()
        self.max_gradient = max(abs(acq_gradients))
        self.metrics_log.write("Max gradient: "+str(self.max_gradient)+"\n")
        self.median_gradient = np.median(abs(acq_gradients))
        self.metrics_log.write("Median gradient: "+str(self.median_gradient)+"\n")
        max_gradient_idx = np.argwhere(abs(acq_gradients)==self.max_gradient)[0][0]
        self.max_gradient_pxl_idx = int(max_gradient_idx/len(self.azimuths))
        self.metrics_log.write("Max gradient pxl idx: "+str(self.max_gradient_pxl_idx)+"\n")
        self.radial_polar_plot_2d(lps=self.dense_acq.lps.lps,signal=acq_gradients[max_gradient_idx - (max_gradient_idx%len(self.azimuths)) : max_gradient_idx - (max_gradient_idx%len(self.azimuths))+len(self.azimuths)], title="Gradients of pixel showing max gradient").write_html(self.save_metrics_path+"\\Max_pxl_gradient.html")
        self.radial_polar_plot_2d(lps=self.dense_acq.lps.lps, signal=self.pixels_intensities_matrix[self.max_gradient_pxl_idx,:], title="Max gradient pxl intensities").write_html(self.save_metrics_path+"\\Max_gradient_pxl_intensity.html")
        self.dense_acq_gradients_histogram = np.histogram(acq_gradients)
        del acq_gradients
        self.plot_pixel_intensities_2d(lps=self.dense_acq.lps.lps, signal=np.mean(self.pixels_intensities_matrix,axis=0), title="Mean_pixl_intensities_2D").write_html(self.save_metrics_path+"\\Pxl_intensities_2D_projected.html")
        self.radial_polar_plot_2d(lps=self.dense_acq.lps.lps, signal = np.mean(self.pixels_intensities_matrix,axis=0), title="Mean_Pixl_intensities_radial_2D").write_html(self.save_metrics_path+"\\Pxl_intensities_radial_2D.html")
        x_grid, y_grid, z_grid = self.interpolate_intensities(np.mean(self.pixels_intensities_matrix,axis=0))
        self.plot_pixel_intensities_3d(grid_data={'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': z_grid}, signal=np.mean(self.pixels_intensities_matrix,axis=0),  title="Mean_pixl_intensities_3D" ).write_html(self.save_metrics_path+"\\Pxl_intensities_3d.html")
        self.metrics_log.close()
        self.metrics_log = open(self.save_metrics_path+"\\metric_logs.txt","a")

    def analyze_blps_acq(self):
        self.plot_pixel_intensities_2d(lps=self.blps.lps,signal=self.elevations, title="Theoretical_nblps_nb_pts-"+ str(len(self.blps.lps))).write_html(self.save_metrics_path+"\\Theoretical_blps.html")
        lp_filter = self.blps.unique_light_positions(self.blps.lps)
        blps_filtered = light_positions(lps=lp_filter.get_unique_lps())
        blps_filtered.img_file_names = [self.blps.img_file_names[i] for i in lp_filter.unique_element_indices]
        critical_pxls_filtered = [self.trigger_pxls[i] for i in lp_filter.unique_element_indices]
        nblp_criterions_filtered = [self.nblp_criterions[i] for i in lp_filter.unique_element_indices]
        self.plot_pixel_intensities_2d(lps=blps_filtered.lps,signal=nblp_criterions_filtered, title="Theoretical_nblps_filtered_nb_pts-"+ str(len(blps_filtered.lps))).write_html(self.save_metrics_path+"\\Theoretical_blps_filtered.html")
        blps_filtered.lp_file_path = self.dense_acq.data['dir']+"\\blps.lp"        
        blps_filtered.write_lp_file()
        blps_filtered_acq = acquisition(data={'dir': self.dense_acq.data['dir'],  'lp_file_name': "blps.lp"})
        blps_filtered_pixels_intensities_matrix = blps_filtered_acq.get_acquisition_matrix()
        if self.pixels_list_1d is not None:
            blps_filtered_pixels_intensities_matrix = np.array([list(map(img.img_vect.__getitem__, self.pixels_list_1d)) for img in blps_filtered_acq.get_images()]).T 
        blps_filtered_gradients_matrix = np.gradient(blps_filtered_pixels_intensities_matrix, axis=1)
        self.radial_polar_plot_2d(lps=blps_filtered_acq.lps.lps,signal=blps_filtered_gradients_matrix[self.max_gradient_pxl_idx, :], title="Gradients of pixel showing max gradient in BLP acquisition").write_html(self.save_metrics_path+"\\Max_pxl_gradient_blp_acq.html")
        self.radial_polar_plot_2d(lps=blps_filtered_acq.lps.lps, signal=blps_filtered_pixels_intensities_matrix[self.max_gradient_pxl_idx,:], title="Max gradient pxl intensities in BLP acquisition").write_html(self.save_metrics_path+"\\Max_gradient_pxl_intensity_blp_acq.html")
        blps_filtered_gradients = blps_filtered_gradients_matrix.flatten()
        self.metrics_log.write("Max blps gradient: "+str(max(blps_filtered_gradients))+"\n")
        blps_filtered_gradients_histogram = np.histogram(blps_filtered_gradients)
        self.plot_distribution(histograms=[self.dense_acq_gradients_histogram, blps_filtered_gradients_histogram], group_labels=["Dense acq gradients", "Best light position acq gradients"]).write_html(self.save_metrics_path+"\\Gradients_distribution_plot.html")
        pxl_criticality_img = np.zeros(self.h *self.w)
        pxl_criticality_array = self.pxls_criticality_matrix.sum(axis=1).flatten()
        if self.pixels_list_1d is not None:
            for i in range(0, len(self.pixels_list_1d)):
                pxl_criticality_img[self.pixels_list_1d[i]] = pxl_criticality_array[i]            
        else:
            pxl_criticality_img = self.pxls_criticality_matrix.sum(axis=1).flatten()
        pxl_criticality_img = pxl_criticality_img.reshape(self.h,self.w)
        self.plot_overlaid_heat_map(data=pxl_criticality_img, overlay_img=blps_filtered_acq.get_images()[0].img_mat).write_html(self.save_metrics_path+"\\Pxl_criticalities_heat_map.html")
        del blps_filtered_gradients
        

    def find_best_light_positions_in_theta_space(self):
        start_t = time.time()
        self.sorted_theta_indices = np.argsort(self.azimuths)
        print("Analyzing the acquisition")
        self.analyze_dense_acq()
        self.acq_matrix = copy.deepcopy(self.pixels_intensities_matrix)[:,self.sorted_theta_indices]
        del self.pixels_intensities_matrix
        print("Finished analyzing in: ", time.time()-start_t) 
        nblps_indices = []
        self.trigger_pxls = []
        self.nblp_criterions = []
        self.pxls_criticality_matrix = np.zeros(self.acq_matrix.shape,dtype=bool)
        print(self.pxls_criticality_matrix.shape)
        nblps_indices.append(self.sorted_theta_indices[0])
        current_theta_idx = 1
        while current_theta_idx<len(self.azimuths)-1:
            current_theta_idx, pxl, criterion = self.next_theta(current_theta_idx=current_theta_idx)               
            nblps_indices.append(self.sorted_theta_indices[current_theta_idx])
            self.trigger_pxls.append(pxl)
            self.nblp_criterions.append(criterion)
        del self.acq_matrix
        self.blps = light_positions(lps=[self.dense_acq.lps.lps[i] for i in nblps_indices])
        self.blps.img_file_names = [self.dense_acq.lps.img_file_names[i] for i in nblps_indices]
        self.analyze_blps_acq()
        self.metrics_log.close()

    def abs_uint_difference(self,a,b):
        return ne.evaluate('abs(a-b)')
    
    def next_theta(self, current_theta_idx):
        self.theta_signal_validation_difference_threshold = min(45,int(2*self.max_gradient))
        self.theta_signal_validation_gradient_threshold = 0.75*self.max_gradient
        print('current theta idx: ', current_theta_idx)
        next_theta_idx = min(current_theta_idx+self.theta_idx_step_size,len(self.azimuths)-1)
        trigger_pxl = None
        criterion = SAMPLE_FREQ_CRITERION
        if current_theta_idx+1 < next_theta_idx:
            gradient_nblp = next_theta_idx
            differences_nblp = next_theta_idx    
            gradient_pxl = None
            differences_pxl = None
            gradients = abs(np.gradient(self.acq_matrix[:,current_theta_idx+1:next_theta_idx+1], axis=1))
            gradients_indices = np.argwhere(gradients.flatten()>self.theta_signal_validation_gradient_threshold).flatten()
            gradient_nblps = gradients_indices%(next_theta_idx-current_theta_idx)
            if len(gradient_nblps)>0:
                min_loc = np.argwhere(gradient_nblps==min(gradient_nblps)).flatten()[0]
                gradient_nblp = current_theta_idx+gradient_nblps[min_loc]+1
                gradient_pxls = (gradients_indices/(next_theta_idx-current_theta_idx)).astype(int)
                gradient_pxl = gradient_pxls[min_loc]
                gradient_critical_pxls_indices = np.argwhere(gradients[:,gradient_nblps[min_loc]]>self.theta_signal_validation_gradient_threshold).flatten()                
            del gradients
            differences = self.abs_uint_difference(self.acq_matrix[:,current_theta_idx+1:next_theta_idx+1],self.acq_matrix[:,current_theta_idx,None]).astype(int)
            differences_indices = np.argwhere(differences.flatten()>self.theta_signal_validation_difference_threshold).flatten()
            differences_nblps = differences_indices%(next_theta_idx-current_theta_idx)            
            if len(differences_nblps)>0:
                min_loc = np.argwhere(differences_nblps==min(differences_nblps)).flatten()[0]
                differences_nblp = current_theta_idx+differences_nblps[min_loc]+1
                differences_pxls = (differences_indices/(next_theta_idx-current_theta_idx)).astype(int)
                differences_pxl = differences_pxls[min_loc]
                difference_critical_pxls_indices = np.argwhere(differences[:,differences_nblps[min_loc]]>self.theta_signal_validation_difference_threshold).flatten()
            del differences
            if gradient_nblp<differences_nblp:                
                trigger_pxl = gradient_pxl
                next_theta_idx = gradient_nblp
                criterion = GRAD_CRITERION
                gradient_critical_pxls_indices = gradient_critical_pxls_indices*len(self.azimuths) + next_theta_idx
                np.put(self.pxls_criticality_matrix, gradient_critical_pxls_indices, True)                
            if differences_nblp<gradient_nblp:
                trigger_pxl = differences_pxl
                criterion = DIFF_CRITERION
                next_theta_idx = differences_nblp
                difference_critical_pxls_indices = difference_critical_pxls_indices*len(self.azimuths) + next_theta_idx
                np.put(self.pxls_criticality_matrix, difference_critical_pxls_indices, True)                                
        print('Next theta idx: ', next_theta_idx)
        return next_theta_idx, trigger_pxl, criterion
##########################################################################################
class dense_acq_measurement:

    def find_best_light_positions_in_u_v_space(self):
        for i in range(0, len(self.pixels_list_1d)):
            i_pixel_intensities = [intensity[i] for intensity in self.pixels_intensities_matrix]
            self.plot_pixel_intensities_2d(signal=i_pixel_intensities, title="Pixl_intensities_2D_("+str(self.pixels_indices_2d[i])+")").write_html(self.save_metrics_path+"\\Pixl_intensities_2D_projected.html")
            x_grid, y_grid, z_grid = self.interpolate_intensities(i_pixel_intensities)
            self.plot_pixel_intensities_3d(grid_data={'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': z_grid}, signal=i_pixel_intensities,  title="Pixl_intensities_3D_("+str(self.pixels_indices_2d[i])+")" ).write_html(self.save_metrics_path+"\\Pxl_intensities_3D.html")
            g, gx, gy, signal_gx, signal_gy, signal_g = self.compute_gradients_distribution_in_uv_space(z_grid=z_grid, pixel_intensities=i_pixel_intensities)
            gx_grid_data = {'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': gx}
            gy_grid_data = {'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': gy}
            g_grid_data = {'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': g}
            self.plot_pixel_intensities_2d(signal=signal_gx, title="gradient_x_2D_("+str(self.pixels_indices_2d[i])+")").write_html(self.save_metrics_path+"\\Gradient_x_2D_projected.html")
            self.plot_pixel_intensities_2d(signal=signal_gy, title="gradient_y_2D_("+str(self.pixels_indices_2d[i])+")").write_html(self.save_metrics_path+"\\Gradient_y_2D_projected.html")
            self.plot_pixel_intensities_2d(signal=signal_g, title="gradient_2D_("+str(self.pixels_indices_2d[i])+")").write_html(self.save_metrics_path+"\\Gradient_2D_projected.html")
            self.plot_pixel_intensities_3d(grid_data=gx_grid_data, signal=signal_gx, title="gradient_3D_("+str(self.pixels_indices_2d[i])+")").write_html(self.save_metrics_path+"\\Gradient_3d.html")
            self.plot_pixel_intensities_3d(grid_data=gy_grid_data, signal=signal_gy, title="gradient_3Dx_("+str(self.pixels_indices_2d[i])+")").write_html(self.save_metrics_path+"\\Gradient_3d_x.html")
            self.plot_pixel_intensities_3d(grid_data=g_grid_data, signal=signal_g, title="gradient_3Dy_("+str(self.pixels_indices_2d[i])+")").write_html(self.save_metrics_path+"\\Gradient_3d_y.html")

    def compute_gradients_distribution_in_uv_space(self, z_grid, pixel_intensities):   
        gx, gy = np.gradient(z_grid)
        g = gx+gy
        # assuming the linspace limits are -1 to 1
        marker_indices = [(int((len(self.xs)*(1+self.xs[i]))/2), int((len(self.ys)*(1+self.ys[i]))/2)) for i in range(0,len(self.xs))]
        signal_gx = [gx[marker_indices[i][1]][marker_indices[i][0]] for i in range(0,len(self.xs))]
        signal_gy = [gy[marker_indices[i][1]][marker_indices[i][0]] for i in range(0,len(self.xs))]
        signal_g = [g[marker_indices[i][1]][marker_indices[i][0]] for i in range(0,len(self.xs))]
        return g, gx, gy, signal_gx, signal_gy, signal_g

##########################################################################################
class ring_nblp:
    def __init__(self, **kwargs):
        default_values = {'acq':None, 'initial_nb': 10} 
        self.__dict__.update(default_values)
        allowed_args = ('dense_acq', 'initial_nb', None)
        if set(kwargs.keys()).issubset(allowed_args):
            self.__dict__.update(kwargs) 

        self.h,self.w = self.dense_acq.get_images()[0].img_mat.shape



# data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\rust_coarse\theta_gradient\dense'), 'lp_file_name': "iteration_0.lp"}
# data = {'dir':os.path.abspath(r'D:\imvia_phd\data\nblp_v2\nblp_2_acquisitions\coin\LDR_20221005_144547_dense_ring'), 'lp_file_name': "acquisition.lp"}
# data = {'dir': os.path.abspath(r'E:\acquisitions\nblp_v2\coin\LDR_20221005_144547_dense_ring\roi_acq'),  'lp_file_name': "acquisition.lp"}
# k = acquisition(data=data)

# roi = {'w_from_center':500, 'h_from_center':500}
# new_acq = {'dir': os.path.abspath(r'E:\acquisitions\nblp_v2\brushed_metal\ring_dense_acquisition\phi_45_500'),  'lp_file_name': "iteration_0.lp"}
# new_acq = {'dir': os.path.abspath(r'E:\acquisitions\nblp_v2\rust_coarse\theta_gradient\dense'),  'lp_file_name': "blps.lp"}
new_acq = {'dir': os.path.abspath(r'C:\Users\Ramamoorthy_Luxman\Desktop\roi_acq'),  'lp_file_name': "acquisition.lp"}
# k.copy_acquisition(new_data=new_acq,resize = {'ratio':0.5})
# k.copy_acquisition(new_acq, roi=roi)
# k = new_acq

k = acquisition(data=new_acq)

# # pixels = [(10,15),(100,200), (400,500)]
# pixels = [(800,500)]
# pixels = [(100,100),(100,200)]

# pixels = [100,1,700, 1800]
# l = dense_acquisition_measurements(dense_acq = k,pixels_indices_2d= pixels)
# l = dense_acquisition_measurements(dense_acq=k, pixels_list_1d = [118236])
l = dense_ring_acquisition_measurements(dense_acq = k)
s = l.get_dense_acquisition_measurements()
# lps = k.lps.lps[::int(np.ceil( len(k.lps.lps) / Desired_validation_theta_acq_size ))]
# lps = k.lps.lps
# lps.append(lps[0])
# signal = l.pixels_intensities_matrix[l.pixels_list_1d,:][0][::int(np.ceil( len(k.lps.lps) / Desired_validation_theta_acq_size ))]
# signal = l.pixels_intensities_matrix[l.pixels_list_1d,:][0]
# signal = np.append(signal, signal[0])
# print(signal)
# print(len(signal))
# print(len(lps))
# l.radial_polar_plot_2d(lps=lps, signal = signal, title="(500,500)_Pixl_intensities_radial_2D")   
# thetas = [lp.az_degrees for lp in lps]
# timestep = 360/len(thetas)
# spectrum = np.fft.fft(signal)
# n = signal.size
# freq = np.fft.fftfreq(n, d=timestep)
# print(len(freq))
# # freq = np.fft.fftfreq(len(spectrum))
# spectrum_cut = spectrum.copy()
# spectrum_cut[(freq>0)] = 0
# # spectrum_cut[(freq>-0.002)] = 0
# # spectrum_cut[(freq<-0.005)] = 0
# cut_signal = np.fft.ifft(spectrum_cut)


# plt.subplot(221)
# plt.plot(thetas,signal)
# plt.subplot(222)
# plt.plot(freq,spectrum)
# plt.subplot(223)
# plt.plot(freq,spectrum_cut)
# plt.subplot(224)
# plt.plot(thetas,cut_signal)

# # SAMPLE_RATE = 
# # yf = fft(signal)
# # xf = fftfreq(N, 1 / SAMPLE_RATE)
# l = dense_acquisition_measurements(dense_acq = k)
# s = l.get_dense_acquisition_measurements()

# Fs = len(signal)/360
# T = 1/Fs
# L = len(signal)
# t = thetas
# Y = fft(signal)
# P2 = np.abs(Y/L);
# P1 = P2[1:int(L/2)+1];
# P1[2:len(P1)-1] = 2*P1[2:len(P1)-1]

# f = Fs*linspace(0,int(L/2), num=len(P1))/L;
# fig = plt.figure()
# plt.subplot(121)
# plt.plot(thetas,signal)
# plt.subplot(122)
# plt.plot(f,P1) 
# # title("Single-Sided Amplitude Spectrum of X(t)")
# # xlabel("f (Hz)")
# # ylabel("|P1(f)|")
# plt.show()





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
        