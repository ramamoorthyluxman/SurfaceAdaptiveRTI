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



rootdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\green_rust_metal\\phi_5_collimated\\"

def plot_graze_diff(img1, img2, file_names):
    # fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
    for k in range(0, len(img1)):
        diff_img = abs(img1[k] - img2[k])
        mean_img = np.float32(diff_img)
        mean_img = (img1[k]+img2[k])/2
        graze_img = np.float32(diff_img)
        graze_img = cv2.divide(np.float32(diff_img), np.float32(mean_img), dtype=cv2.CV_32F)
        
        normalizedImg = np.zeros(graze_img.shape, np.uint8)
        normalizedImg = cv2.normalize(graze_img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        # normalizedImg = cv2.bitwise_not(normalizedImg)


        cv2.imwrite(rootdir+"temp\\"+file_names[k]+".png", normalizedImg)
        graze_img = cv2.resize(graze_img, (100,100))
        # xx, yy = np.mgrid[0:graze_img.shape[0], 0:graze_img.shape[1]]
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(xx, yy, graze_img ,rstride=1, cstride=1, cmap="autumn_r", linewidth=0)
        # ax.set_title("Images " + file_names[k])
        # plt.savefig(rootdir+file_names[k]+".pdf")


        z_data=np.array(normalizedImg)

        fig = go.Figure(data=[go.Surface(z=z_data)])

        fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                        width=500, height=500,
                        margin=dict(l=65, r=50, b=65, t=90))

        # fig.show()
        fig.write_html(rootdir+file_names[k]+".html")


img_file_names = []
first_images = []
second_images = []

lp_file = rootdir + "iteration_0.lp"
light_positions, image_files = read_lp_file(lp_file)

for i in range(0, int(len(light_positions)/2)):
    filename = image_files[i].split('.')[0] 
    img_file = glob.glob(rootdir+filename+"_*.png")[0]
    print(img_file)
    img1 = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)    

    filename2 = image_files[i+int(len(light_positions)/2)].split('.')[0]   
    img_file2 = glob.glob(rootdir+filename2+"_*.png")[0]
    print(img_file2)
    img2 = cv2.imread(img_file2, cv2.IMREAD_GRAYSCALE) 

    first_images.append(img1)
    second_images.append(img2)

    img_file_names.append(filename+"_"+filename2)
    
plot_graze_diff(first_images, second_images, img_file_names)
