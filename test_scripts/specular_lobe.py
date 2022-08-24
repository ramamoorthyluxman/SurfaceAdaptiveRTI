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
from scipy.interpolate import interp1d, griddata
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import numpy as np
import cdd as pcdd
import matplotlib.pyplot as plt
import random
from plotly.offline import plot

#############################################################################################################
rootdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\brushed_metal\\simple_brushed_metal_2\\"

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


# def plot_intensities(thetas, phis, intensities, pixel):
#     normalized_intensities = normalize_list(intensities,0, 1)
#     X = []
#     Y = []
#     Z = []
#     for i in range(0, len(thetas)):
#         xx,yy,zz = sph2cart(thetas[i], phis[i], normalized_intensities[i])
#         X.append(round(xx,2))
#         Y.append(round(yy,2))
#         Z.append(round(zz,2))
#         # print(round(xx,2), round(yy,2), round(zz,2))
#     print(min(X), max(X), min(Y), max(Y))
#     xi = np.arange(min(X), max(X), 0.1)
#     yi = np.arange(min(Y), max(Y), 0.1)
#     # xi,yi = np.meshgrid(xi,yi)
#     grid_x, grid_y = np.meshgrid(xi,yi)
#     zi = griddata((X,Y), Z, (grid_x, grid_y) , method='cubic')
    
#     fig = go.Figure(data=[go.Surface(z = zi, y=yi, x=xi)])

#     fig.update_layout(title=str(pixel), autosize=False,
#                     width=500, height=500,
#                     margin=dict(l=65, r=50, b=65, t=90))

#     # fig.show()
#     fig.write_html(rootdir+str(pixel)+".html")

def plot_intensities(thetas, phis, intensities, pixel):

    normalized_intensities = normalize_list(intensities,0, 1)
    data = []
    for i in range(0, len(thetas)):
        xx,yy,zz = sph2cart(thetas[i], phis[i], normalized_intensities[i]) 
        xx = round(xx,2)              
        yy = round(yy,2)
        zz = round(zz,2)
        vector = go.Scatter3d( x = [0,xx],
                                y = [0,yy],
                                z = [0,zz],
                                line = dict( color = "blue", width = 6)
                     )
        data.append(vector)
        cone = go.Cone(x=[xx], y=[yy], z=[zz], u=[0.3*xx], v=[0.3*yy], w=[0.3*zz], anchor="tip", colorscale='Blues')
        layout = go.Layout(margin = dict( l = 0,
                                  r = 0,
                                  b = 0,
                                  t = 0)
                          )
        data.append(cone)
    fig = go.Figure(data=data,layout=layout)
    # plot(fig,filename="vector.html",auto_open=False,image='png',image_height=800,image_width=1500)
    fig.write_html(rootdir+str(pixel)+".html")





img_file = rootdir+"phi_5_collimated\\nblp_iteration_0_1_theta_0.0_phi_5.0.png"
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
h,w = img.shape
for samples in range(0, 3):
    pixel_loc = (random.randint(0, h), random.randint(0, w))

    intensities = []
    thetas = []
    phis = []

    for subdir, dirs, files in os.walk(rootdir):  
        dataset_dir =  subdir+"\\"
        lp_file = dataset_dir + "iteration_0.lp"
        if not os.path.exists(lp_file):
            continue
        light_positions, image_files = read_lp_file(lp_file)    
        for i in range(0, len(light_positions)):
            filename = image_files[i].split('.')[0]    
            img_file = glob.glob(dataset_dir+filename+"_*.png")[0]
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)    
            intensities.append(img[pixel_loc[0]][pixel_loc[1]])    
            r, longitude, latitude = cart2sph(light_positions[i][0], light_positions[i][1], light_positions[i][2])
            thetas.append(round(longitude, 5))
            phis.append(round(latitude, 5))
            # thetas.append(math.floor((longitude)*100)/100)
            # phis.append(math.floor((latitude)*100)/100)

        
    plot_intensities(thetas, phis, intensities, pixel_loc)


