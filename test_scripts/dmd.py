from tracemalloc import start
import cv2
import random 
import numpy as np
import os
import glob 
import matlab.engine
import time

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

rootdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_2_acquisitions\\brushed_metal\\simple_brushed_metal\\"

print("Calculating the DMD coeffs.")
start_t = time.time()
acquisition = eng.get_dmd_coeffs(rootdir)
print("Done calculating the DMD coeffs. Time taken: ", time.time() - start_t)

dataset_dir =  rootdir+"\\"
lp_file = dataset_dir + "iteration_0.lp"

images = []
xs = []
ys = []
zs = []
thetas = []
phis = []
if os.path.exists(lp_file):    
    light_positions, image_files = read_lp_file(lp_file)    
    for i in range(0, len(light_positions)):    
        filename = image_files[i].split('.')[0]    
        img_file = glob.glob(dataset_dir+filename+"_*.png")[0]
        acquired_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)    
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
        # cv2.imwrite("relighted_img.png",relighted_img)
        print("Done relighting the ", i, "th  light position. Time taken: ", time.time()-start_t )

