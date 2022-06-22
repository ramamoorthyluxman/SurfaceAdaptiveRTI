import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
import cv2
import glob
import math
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib
from matplotlib import cm 

#############################################################################################################

def Cartesian2Polar3D(x, y, z):
    """ 
    Takes X, Y, and Z coordinates as input and converts them to a polar
    coordinate system

    Source: https://stackoverflow.com/questions/10868135/cartesian-to-polar-3d-coordinates

    """

    r = math.sqrt(x*x + y*y + z*z)

    longitude = math.acos(x / math.sqrt(x*x + y*y)) * (-1 if y < 0 else 1)

    latitude = math.asin(z / r)

    return r, longitude, latitude


def Polar2Cartesian3D(r, longitude, latitude):
    """
    Takes, r, longitude, and latitude coordinates in a polar coordinate
    system and converts them to a 3D cartesian coordinate system

    Source: https://stackoverflow.com/questions/10868135/cartesian-to-polar-3d-coordinates
    """

    x = r * math.sin(latitude) * math.cos(longitude)
    y = r * math.sin(latitude) * math.sin(longitude)
    z = r * math.cos(latitude)

    return x, y, z

def read_lp_file(file_path, dome_radius):
    light_positions = []
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

        # r, long, lat = Cartesian2Polar3D(x,y,z)
        # r = mytool.dome_radius
        # r = dome_radius
        # x, y, z = Polar2Cartesian3D(r,long,lat)
        light_positions.append((x,y,z))
    
    return light_positions    

###############################################################################################################


# # Generate random example data
# theta_calculated = [10,20,50,61,80,21,100]
# # theta_measured = [10,20,50,61,80,21,100]
# theta_measured = np.linspace(np.deg2rad(60), np.deg2rad(150), 100)
# # r_calculated = resample(np.random.uniform(2.5, 3.5, 10), len(theta_calculated))
# r_measured = resample(np.random.uniform(3.5, 5.5, 10), len(theta_measured))
# r_calculated = [1.0,1.2,1.5,1.0,2.3,1.0,1.6]
# # r_measured = [3.4,3.2,2.5,3.0,3.3,2.0,2.6]

# # Plot curves
# plt.polar(theta_calculated, r_calculated, color="red", label="calculated")
# plt.polar(theta_measured, r_measured, color="blue", label="measured")

# # Add legend
# plt.legend(loc="center")

# # Adjust ticks to data, taking different step sizes into account
# plt.xticks([
#     *np.arange(min(theta_measured), max(theta_measured) + np.deg2rad(1), np.deg2rad(30)),
#     *np.arange(min(theta_calculated), max(theta_calculated) + np.deg2rad(1), np.deg2rad(15)),
# ])
# plt.yticks(np.arange(2, 6 + 1))

# plt.show()

########################################## Global alpha images ########################################################

# imdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\stone\\theta_85\\"

# ext = ['png', 'jpg', 'gif']    # Add image formats here

# files = []
# [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

# img = cv2.imread(files[0])
# s = np.zeros(img.shape)

# for file in files:
#     img = cv2.imread(file)
#     s = s + img

# n = len(files)
# print(n)

# alpha_image = []
# save_path = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\stone\\theta_85\\alpha_images\\global\\"
# for i in range(0,n):
#     img = cv2.imread(files[i])
#     alpha_image = abs(n*img - s)/(math.pow(n,2) - n)
#     normalized_alpha_img = np.zeros(img.shape)
#     min_val = alpha_image[..., 0].min()
#     max_val = alpha_image[..., 0].max()
#     normalized_alpha_img = alpha_image * (255/(max_val-min_val))
#     file_name = os.path.basename(files[i])
#     cv2.imwrite(save_path+file_name, alpha_image)
#     cv2.imwrite(save_path+"normalized_"+file_name, normalized_alpha_img)

##################################################################################################################
######################################global alpha plots##########################################################

# imdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\stone\\theta_85\\alpha_images\\global\\"
# ext = ['png', 'jpg', 'gif']    # Add image formats here

# files = []
# [files.extend(glob.glob(imdir + 'normalized_*.' + e)) for e in ext]


# light_positions = read_lp_file("D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\stone\\theta_85\\iteration_0.lp", 1.0)
# light_poses = []

# for i in range(0,len(files)):
#     light_poses.append(Cartesian2Polar3D(light_positions[i][0], light_positions[i][1], light_positions[i][2]))


# for i in range(0, len(files)):
#     file = files[i]
#     img = cv2.imread(file, cv2.IMREAD_GRAYSCALE) 
#     downscale_size = 1.0
#     width = int(img.shape[0]*downscale_size)
#     height = int(img.shape[1]*downscale_size)
#     img = cv2.resize(img, [width, height],interpolation = cv2.INTER_AREA)
#     xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]    
#     # ax = axs[i, (i%4)]
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')    
#     # Labels.
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('alpha_value')

#     ax.azim = 45
#     ax.dist = 15
#     ax.elev = 45
#     ax.set_title('Theta: ' + str(math.degrees(light_poses[i][1])) + ', Phi: ' + str(math.degrees(light_poses[i][2])))
#     surf = ax.plot_surface(xx, yy, img, cmap=matplotlib.cm.coolwarm,
#                        linewidth=0, antialiased=False)
    

#     # surf = ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap=plt.cm.gray,
#     #             linewidth=0)
#     # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=5)
    
# plt.show()
    # plt.savefig("D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\stone\\theta_85\\alpha_images\\"+str(i))

##################################################################################################################
######################################local alpha plots##########################################################


# window_size_factor = 100
# imdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\stone\\theta_85\\alpha_images\\global\\"
# save_path = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\stone\\theta_85\\alpha_images\\local\\"
# ext = ['png', 'jpg', 'gif']    # Add image formats here

# files = []
# [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

# image = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
# windows_rows = int(image.shape[1]/window_size_factor)
# windows_cols = int(image.shape[0]/window_size_factor)

# local_maxs = []
# local_mins = []

# for p in range(0, window_size_factor):
#     local_mins_row = []
#     local_maxs_row = []
#     for q in range(0, window_size_factor):
#         local_min = 100000
#         local_max = 0
#         for i in range(0, len(files)):
#             img = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
#             w = img[(p*windows_cols):(p*windows_cols)+windows_cols, (q*windows_rows):(q*windows_rows)+windows_rows]
#             min_val = 0           
#             if w[np.nonzero(w)].size != 0:
#                 min_val = np.min(w[np.nonzero(w)])
#             max_val = w.max()
#             if min_val < local_min:
#                 local_min = min_val
#             if max_val > local_max:
#                 local_max = max_val
#         local_mins_row.append(local_min)
#         local_maxs_row.append(local_max)
#     local_mins.append(local_mins_row)
#     local_maxs.append(local_maxs_row)

# for i in range(0, len(files)):
#     img = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
#     normalized_img = np.zeros(img.shape)            
#     for p in range(0, window_size_factor):
#         for q in range(0, window_size_factor):
#             w_img = np.array(img[(p*windows_cols):(p*windows_cols)+windows_cols, (q*windows_rows):(q*windows_rows)+windows_rows])
#             w_img = w_img-local_mins[p][q]
#             w_img[w_img<0] = 0
#             w_img[np.isnan(w_img)] = 0
#             if (local_maxs[p][q]-local_mins[p][q]) != 0:
#                 normalized_img[(p*windows_cols):(p*windows_cols)+windows_cols, (q*windows_rows):(q*windows_rows)+windows_rows] = (255/(local_maxs[p][q]-local_mins[p][q])) * w_img
#             else:
#                 normalized_img[(p*windows_cols):(p*windows_cols)+windows_cols, (q*windows_rows):(q*windows_rows)+windows_rows] = 0*w_img
#             normalized_img[np.isnan(normalized_img)] = 0
#             print("*******")
#             print(local_maxs[p][q])
#             print(local_mins[p][q])
#             # print((255/(local_maxs[p][q]-local_mins[p][q])))
#             # # print(img[(p*windows_cols):((p*windows_cols)+windows_cols), (q*windows_rows):((q*windows_rows)+windows_rows)].max())
#             # # print(img[(p*windows_cols):((p*windows_cols)+windows_cols), (q*windows_rows):((q*windows_rows)+windows_rows)].min())
#             print(img.max())
#     file_name = os.path.basename(files[i])
#     cv2.imwrite(save_path+"normalized_"+file_name, normalized_img)
#     # print(normalized_img.max())
#     # print(normalized_img.min())

######################################local alpha contour plots##########################################################

imdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\stone\\theta_85\\alpha_images\\global\\"
ext = ['png', 'jpg', 'gif']    # Add image formats here

files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]


light_positions = read_lp_file("D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\stone\\theta_85\\iteration_0.lp", 1.0)
light_poses = []

r = []
theta = []


for i in range(0,len(light_positions)):
    light_poses = (Cartesian2Polar3D(light_positions[i][0], light_positions[i][1], light_positions[i][2]))

    file = files[i]
    # fig = plt.figure()
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE) 
    downscale_size = 1.0
    width = int(img.shape[0]*downscale_size)
    height = int(img.shape[1]*downscale_size)

    r.append(img.mean())
    theta.append(light_poses[0])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r)
    ax.set_rmax(2)
    ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("A line plot on a polar axis", va='bottom')
    plt.show()



    

















########################################################################################################################