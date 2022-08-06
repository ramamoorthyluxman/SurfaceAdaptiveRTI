import matplotlib.pyplot as plt
from asyncore import read
import numpy as np
from scipy.signal import resample
import cv2
import glob
import math
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import cm 
import matplotlib.pyplot as plot
from scipy.interpolate import interp1d

#############################################################################################################


def cart2sph(x, y, z):
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

def read_lp_file(file_path):
    light_positions = []
    image_files = []
    # Read in .lp data
    try:
        file = open(file_path)
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
        light_positions.append((x,y,z))
        image_files.append(cols[0])

    return light_positions, image_files


def normalize_list(list, target_min, target_max):
    minimum = min(list)
    maximum = max(list)
    for i in range(0, len(list)):
        list[i] = (target_max - target_min)*(list[i]-minimum)/(maximum-minimum)
    return list


def plot_intensities(thetas_mat, intensities_mat, phis, pixel_loc):
    fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
    for k in range(0, len(thetas_mat)):
        intensities = intensities_mat[k]
        thetas = thetas_mat[k]
        
        for i in range(0,len(thetas)):
            thetas[i] = math.atan2(math.sin(thetas[i]), math.cos(thetas[i]))
        
        thetas.append(thetas[0])
        intensities.append(intensities[0])
        
        ax.plot(thetas, intensities,'-', label="Phi =  "+ str(math.degrees(phis[k])))
        ax.set_rticks([270,240,210,180,150,120,90,60,30,0]) 
        ax.set_rlabel_position(-22.5)  
        ax.grid(True)
        ax.legend()
        ax.set_title("Pixel " + str(pixel_loc) + " intensities distribution over az and el")
    return plot


intensities_mat = []
thetas_mat = []
phis = []


rootdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\brushed_metal\\simple_brushed_metal_2\\"


img_file = rootdir+"phi_5_collimated\\nblp_iteration_0_1_theta_0.0_phi_5.0.png"
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
h,w = img.shape

pixel_loc = (int(h/4), int(3*w/4))
# pixel_loc = (int(h/4), int(w/4))
# pixel_loc = (int(h/2), int(w/2))
# pixel_loc = (int(3*h/4), int(w/4))
# pixel_loc = (int(3*h/4), int(3*w/4))


# pixel_loc = (500, 610)
# pixel_loc = (560, 610)
# pixel_loc = (550, 610)
# pixel_loc = (560, 620)
# pixel_loc = (550, 620)
# pixel_loc = (540, 620)
# pixel_loc = (490, 620)
# pixel_loc = (450, 620)
# pixel_loc = (440, 620)


for subdir, dirs, files in os.walk(rootdir):  
    dataset_dir =  subdir+"\\"
    lp_file = dataset_dir + "iteration_0.lp"
    if not os.path.exists(lp_file):
        continue
    light_positions, image_files = read_lp_file(lp_file)

    intensities = []
    thetas = []

    for i in range(0, len(light_positions)):
        filename = image_files[i].split('.')[0]    
        img_file = glob.glob(dataset_dir+filename+"_*.png")[0]
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)    
        intensities.append(img[pixel_loc[0]][pixel_loc[1]])    
        r, longitude, latitude = cart2sph(light_positions[i][0], light_positions[i][1], light_positions[i][2])
        thetas.append(math.floor((longitude)*100)/100)

    thetas_mat.append(thetas)
    intensities_mat.append(intensities)

    r, longitude, latitude = cart2sph(light_positions[i][0], light_positions[i][1], light_positions[i][2])
    phis.append(math.floor((latitude)*100)/100)

    plot_intensities(thetas_mat, intensities_mat, phis, pixel_loc).savefig(rootdir+"phi_plot"+str(pixel_loc)+".png")
