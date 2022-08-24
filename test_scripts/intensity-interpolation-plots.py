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


def plot_intensities(thetas, intensities, dense_thetas, dense_intensities, pixel_loc):
    intensities = normalize_list(intensities, 0, 255)
    dense_intensities = normalize_list(dense_intensities, 0, 255)
    fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
    for i in range(0,len(thetas)):
        thetas[i] = math.atan2(math.sin(thetas[i]), math.cos(thetas[i]))
        dense_thetas[i] = math.atan2(math.sin(dense_thetas[i]), math.cos(dense_thetas[i]))
    # f2 = interp1d(thetas, intensities, fill_value="extrapolate")
    f2 = interp1d(thetas, intensities, kind='cubic', fill_value="extrapolate")
    new_thetas = np.linspace(0,2*math.pi,num=100, endpoint=True)
    thetas.append(thetas[0])
    intensities.append(intensities[0])
    dense_thetas.append(thetas[0])
    dense_intensities.append(dense_intensities[0])
    
    # ax.plot(thetas, intensities, 'ro', new_thetas, f2(new_thetas), 'bx-', label="Sparse acquisition, "+ str(len(thetas)-1) + " points")
    ax.plot(thetas, intensities,'bx-', label="Sparse acquisition, "+ str(len(thetas)-1) + " points")
    ax.plot(dense_thetas, dense_intensities,'r-', label="Dense acquisition, "+ str(len(dense_thetas)-1) + " points")
    ax.set_rticks([270,240,210,180,150,120,90,60,30,0]) 
    ax.set_rlabel_position(-22.5)  
    ax.grid(True)
    ax.legend()
    ax.set_title("Pixel " + str(pixel_loc) + " intensities (normalized) distribution over az @ el="+str(75.0))
    return plot

dataset_dir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\brushed_metal\\simple_brushed_metal_2\\phi_75_collimated\\" 
lp_file = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\brushed_metal\\simple_brushed_metal_2\\phi_75_collimated\\iteration_0.lp"
light_positions, image_files = read_lp_file(lp_file)

intensities = []
thetas = []
phis = []

img0 = image_files[0].split('.')[0]    
img_file = glob.glob(dataset_dir+img0+"_*.png")[0]
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
h,w = img.shape
display_img = img.copy()


pixel_loc = (int(h/4), int(3*w/4))
cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)
pixel_loc = (int(h/4), int(w/4))
cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)
pixel_loc = (int(h/2), int(w/2))
cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)
pixel_loc = (int(3*h/4), int(w/4))
cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)
pixel_loc = (int(3*h/4), int(3*w/4))
cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)

# pixel_loc = (500, 610)
# cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
# cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)

# pixel_loc = (560, 610)
# cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
# cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)

# pixel_loc = (550, 610)
# cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
# cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)

# pixel_loc = (560, 620)
# cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
# cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)

# pixel_loc = (550, 620)
# cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
# cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)

# pixel_loc = (540, 620)
# cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
# cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)

# pixel_loc = (490, 620)
# cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
# cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)

# pixel_loc = (450, 620)
# cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
# cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)

# pixel_loc = (440, 620)
# cv2.circle(display_img, pixel_loc, 5, (255, 255, 255), -1)
# cv2.putText(img=display_img, text=str(pixel_loc), org=(pixel_loc[0]-50, pixel_loc[1]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)


cv2.imwrite("D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\brushed_metal\\simple_brushed_metal_2\\phi_75_collimated\\pixel_loc.png",display_img)

pixel_loc = (int(h/4), int(3*w/4))
pixel_loc = (int(h/4), int(w/4))
pixel_loc = (int(h/2), int(w/2))
pixel_loc = (int(3*h/4), int(w/4))
pixel_loc = (int(3*h/4), int(3*w/4))

# pixel_loc = (500, 610)
# pixel_loc = (560, 610)
# pixel_loc = (550, 610)
# pixel_loc = (560, 620)
# pixel_loc = (550, 620)
# pixel_loc = (540, 620)
# pixel_loc = (490, 620)
# pixel_loc = (450, 620)
# pixel_loc = (440, 620)

for i in range(0, len(light_positions)):
    filename = image_files[i].split('.')[0]    
    img_file = glob.glob(dataset_dir+filename+"_*.png")[0]
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)    
    intensities.append(img[pixel_loc[0]][pixel_loc[1]])    
    r, longitude, latitude = cart2sph(light_positions[i][0], light_positions[i][1], light_positions[i][2])
    thetas.append(math.floor((longitude)*100)/100)
    phis.append(math.floor((latitude)*100)/100)



dense_dataset_dir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\brushed_metal\\simple_brushed_metal_2\\phi_75_collimated\\dense_acquisition\\"
dense_lp_file = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\brushed_metal\\simple_brushed_metal_2\\phi_75_collimated\\iteration_1.lp"
dense_light_positions, dense_image_files = read_lp_file(dense_lp_file)
dense_intensities = []
dense_thetas = []
for i in range(0, len(dense_light_positions)):
    filename = dense_image_files[i].split('.')[0] 
    img_file = glob.glob(dense_dataset_dir+filename+"_*.png")[0]
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    
    r, longitude, latitude = cart2sph(dense_light_positions[i][0], dense_light_positions[i][1], dense_light_positions[i][2])
    dense_thetas.append(math.floor((longitude)*100)/100)
    dense_intensities.append(img[pixel_loc[0]][pixel_loc[1]])


plot_intensities(thetas, intensities, dense_thetas, dense_intensities, pixel_loc).savefig(dataset_dir+"intensity_plot_"+str(pixel_loc)+".png")
