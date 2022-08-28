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
import plotly.graph_objects as go
import time
import matlab.engine
import array
import scipy.io
eng = matlab.engine.start_matlab()



eng.addpath("C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\nblp\\SurfaceAdaptiveRTI\\matlab_functions\\pkg_py")
eng.addpath("C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\nblp\\SurfaceAdaptiveRTI\\matlab_functions\\pkg_fcns")
eng.addpath("C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\nblp\\SurfaceAdaptiveRTI\\matlab_functions\\pkg_DMD")
eng.addpath("C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\nblp\\SurfaceAdaptiveRTI\\matlab_functions\\pkg_DMD\\DMD_basis")


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


rootdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_2_acquisitions\\brushed_metal\\simple_brushed_metal\\"

def get_dmd_coeffs(light_positions, image_files):    
    filename = image_files[0].split('.')[0]    
    img_file = img_file = glob.glob(rootdir+filename+"_*.png")[0]
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    h,w = img.shape
    lps = []
    # images = []
    images = np.zeros((len(light_positions),h*w), np.int16)
    x = []
    y = []
    z = []
    print("Loading the images")
    start_time = time.time()
    for i in range(0, len(light_positions)):
        filename = image_files[i].split('.')[0]    
        img_file = glob.glob(rootdir+filename+"_*.png")[0]
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        # normalize_factor = 255 * np.ones(np.array(img).shape)
        # normalised_image = img/255
        # images.append(img)
        images[i,:] = np.reshape(np.array(img),(1,h*w))
        x.append(light_positions[i][0])
        y.append(light_positions[i][1])
        z.append(light_positions[i][2])
    print("Finished loading the images in ", (time.time() - start_time), "seconds")


    # images_mat = np.array(images)
    # images_mat = np.reshape(images_mat, (len(x), h*w)) 
    print("Converting the loaded images to matlab array")
    start_time = time.time()
    images_mat = images
    matlab_img_mat = matlab.uint8(images_mat.tolist())
    print("Finished converting the images in ", (time.time() - start_time), "seconds")
    # print(np.array(matlab_img_mat).shape)

    print("Reshaping and converting the lps to matlab array")       
    start_time = time.time()
    nb_modes = 45
    lps.append(x)
    lps.append(y)
    lps.append(z)
    lps = np.array(lps)
    lps = np.reshape(lps, [len(x), 3])
    lps = matlab.double(lps.tolist())
    print("Finished converting the lps np array to matlab array in ", (time.time() - start_time), "seconds")
    

    print("Passing the inputs to the Matlab function lib to get the coeffs")
    start_time = time.time()
    modal_basis, normal_elt, Up, dmd_coeffs = eng.get_dmd_coeffs(matlab_img_mat, lps, nb_modes, nargout=4)
    print("Received the coeffs from the executed matlab lib func in ",(time.time() - start_time), "seconds")

    return modal_basis, normal_elt, Up, dmd_coeffs, lps

def get_relighted_img(modal_basis, normal_elt, Up, dmd_coeffs, lp, h, w):
    nb_modes = 45
    relighted_img = eng.get_dmd_interpolated_img(lp, modal_basis,  nb_modes, normal_elt, Up, dmd_coeffs, h, w)
    return relighted_img




tempdir = rootdir+"temp"
if not os.path.exists(tempdir):
    os.mkdir(tempdir)
lp_file = rootdir + "iteration_0.lp"
light_positions, image_files = read_lp_file(lp_file)

print("Fitting and getting DMD coeffs")
start_time = time.time()
modal_basis, normal_elt, Up, dmd_coeffs, lps_matlab_array = get_dmd_coeffs(light_positions=light_positions, image_files=image_files)
print("Finished fitting and calculated the coeffs in ", (time.time() - start_time), "seconds")

filename = image_files[0].split('.')[0] 
img_file = glob.glob(rootdir+filename+"_*.png")[0]
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
h,w = img.shape
print("Acquisition image size: ", h, w)

error_matrix = []

for i in range(0, len(light_positions)):
    print("Calculating the error of, ",i,"th direction")
    start_time = time.time()
    filename = image_files[i].split('.')[0] 
    img_file = glob.glob(rootdir+filename+"_*.png")[0]
    org_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    print("Relighting the", i,"th image")
    print(light_positions)
    print(light_positions[i])
    relighted_img = get_relighted_img(modal_basis=modal_basis, normal_elt=normal_elt, Up=Up, dmd_coeffs=dmd_coeffs, lp=lps_matlab_array[i], h=h, w=w)    
    error = relighted_img-org_img
    error = error.reshape(h*w)
    error_matrix.append(error)
    print("Finished calculating the error matrix in ", (time.time() - start_time), "seconds")

file_path = tempdir+'\\data.mat'
scipy.io.savemat(file_path, {'error_matrix': np.array(error_matrix)})



