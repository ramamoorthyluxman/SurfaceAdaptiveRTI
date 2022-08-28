from logging import root
from tracemalloc import start
import cv2
import random 
import numpy as np
import os
import glob 
import matlab.engine
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import griddata

import math



print("Initiating the matlab engine.")
start_t = time.time()
eng = matlab.engine.start_matlab()
print("Done initiating the matlab engine. Time taken: ", time.time() - start_t)

print("Adding all matlab function paths.")
start_t = time.time()
matlab_functions_dir = "C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\nblp\\SurfaceAdaptiveRTI\\matlab_functions\\"
for subdir, dirs, files in os.walk(matlab_functions_dir):
    eng.addpath(subdir)
print("Done adding all the matlab function paths. Time taken: ", time.time() - start_t)


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

rootdir = "E:\\acquisitions\\LDR_20210227_200254_homo\\"

print("Calculating the DMD coeffs.")
start_t = time.time()
acquisition = eng.get_dmd_coeffs(rootdir)
print("Done calculating the DMD coeffs. Time taken: ", time.time() - start_t)

tempdir = rootdir+"temp"
if not os.path.exists(tempdir):
    os.mkdir(tempdir)
lp_file = rootdir + "acquisition.lp"
print("lp file: ", lp_file)
light_positions = []
image_files = []
if os.path.exists(lp_file):    
    light_positions, image_files = read_lp_file(lp_file)


# filename = image_files[0].split('.')[0] 
# img_file = glob.glob(rootdir+filename+"_*.png")[0]

# img_file = glob.glob(rootdir+filename+".png")[0]
img_file = rootdir+image_files[0]
print(img_file)
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
h,w = img.shape
print("Acquisition image size: ", h, w)

dataset_dir =  rootdir+"\\"
lp_file = dataset_dir + "iteration_0.lp"

images = []
xs = []
ys = []
zs = []
thetas = []
phis = []
# error_matrix = np.zeros((h*w, len(light_positions)))

error_matrix = np.zeros((h*w, 7))

# for i in range(0, len(light_positions)):    
for i in range(0, 7):
    # filename = image_files[i].split('.')[0]    
    # img_file = glob.glob(dataset_dir+filename+"_*.png")[0]
    img_file = rootdir + image_files[i]
    acquired_img = np.array(cv2.imread(img_file, cv2.IMREAD_GRAYSCALE))
    acquired_img =  np.reshape(acquired_img, (1, h*w))
    x, y, z =   light_positions[i][0], light_positions[i][1], light_positions[i][2]
    xs.append(x)
    ys.append(y)
    zs.append(z)
    r, az, el = cart2sph(x, y, z)
    thetas.append(az)
    phis.append(el)
    print("Relighting ", i, "th  light position." )
    start_t = time.time()
    relighted_img = np.array((eng.get_dmd_interpolated_img(float(az), float(el))))
    print("Done relighting the ", i, "th  light position. Time taken: ", time.time()-start_t, "s" )
    relighted_img = np.reshape(relighted_img, (1, h*w))
    cv2.imwrite(tempdir+"relighted_img_"+ image_files[i]+".png",relighted_img)  

    print("Updating the error matrix")
    start_t = time.time()
    error_matrix[:,i] = np.absolute(np.subtract(relighted_img, acquired_img))
    print("Finished calculating the error matrices. Time taken: ", time.time()-start_t, "s")

print("Normalization of error matrix and calculating the error vector.")
start_t = time.time()
sum_of_rows = error_matrix.sum(axis=1)
normalized_error_matrix = error_matrix / sum_of_rows[:, np.newaxis]
error_vector = np.sum(normalized_error_matrix, axis=0)
error_vector_sum = error_vector.sum()
normalized_error_vector = error_vector/error_vector_sum
normalized_error_vector = (normalized_error_vector - np.min(normalized_error_vector))/np.ptp(normalized_error_vector)
print(normalized_error_vector)

print("Surface Plotting in polar co-ordinates")
x, y, z = sph2cart(thetas, phis, normalized_error_vector)

# target grid to interpolate to
xi = yi = np.linspace(-1.0,1.0, 100)
xi,yi = np.meshgrid(xi,yi)

# set mask
# mask = (xi < 1.0) & (xi > -1.0) & (yi < 1.0) & (yi > -1.0)


zi = griddata((x,y),z,(xi,yi),method='cubic')
print(zi)
# mask out the field
# zi[mask] = np.nan
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(xi, yi, zi, cmap=plt.cm.YlGnBu_r)

# Tweak the limits and add latex math labels.
ax.set_zlim(0, 1)
ax.set_xlabel(r'$l_u$')
ax.set_ylabel(r'$l_v$')
ax.set_zlabel(r'$epsilon$')

plt.show()

print("Finished normalization of error matrix and calculation of the error vector. Time taken: ", time.time()-start_t, "s")