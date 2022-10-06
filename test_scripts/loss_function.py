from cmath import asin
from logging import critical, root
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
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
from numpy import savetxt
# conda install -c conda-forge dataclasses
from dataclasses import dataclass
from numpy.lib.stride_tricks import as_strided
import plotly.graph_objects as go
import plotly
from scipy.signal import argrelextrema
import copy

# gradient ascent 
# https://jackmckew.dev/3d-gradient-ascent-in-python.html


fitter = "dmd"

crop_roi_size_ht = 0.75
crop_roi_size_wd = 0.5

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


def downscale_acquisition(path, new_path, lp_file_name):
    lp_file = path + lp_file_name
    ratio = 1
    if os.path.exists(lp_file):    
        light_positions, image_files = read_lp_file(lp_file)
        for i in range(0, len(light_positions)):
            img_file = rootdir + image_files[i]
            acq_img = np.array(cv2.imread(img_file, cv2.IMREAD_GRAYSCALE))
            ht,wd = acq_img.shape
            ht_start = int((ht/2) - (crop_roi_size_ht*ht/2))
            ht_end = int((ht/2) + (crop_roi_size_ht*ht/2))
            wd_start = int((wd/2) - (crop_roi_size_wd*wd/2))
            wd_end = int((wd/2) + (crop_roi_size_wd*wd/2))
            ht_start = int(ht/2)-2
            ht_end = int(ht/2)+2
            wd_start = int(wd/2)-2
            wd_end = int(wd/2)+2
            print(ht_start, ht_end, wd_start, wd_end)
            acquired_img = acq_img[ht_start:ht_end,wd_start:wd_end]
            print(acquired_img.shape)
            if acquired_img.shape[1]>10000:
                ratio = 500/acquired_img.shape[0]
                ratio = int(1/ratio)
            donwscaled_img = acquired_img[::int(ratio), ::int(ratio)]                
            cv2.imwrite(new_path+image_files[i], donwscaled_img)              
        source = path+lp_file_name
        destination = new_path + "acquisition.lp"
        os.system('copy '+ source + " "+ destination)
    
    return new_path      
    
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

def multiDimenDist(point1,point2):
   #find the difference between the two points, its really the same as below
   deltaVals = [point2[dimension]-point1[dimension] for dimension in range(len(point1))]
   runningSquared = 0
   #because the pythagarom theorm works for any dimension we can just use that
   for coOrd in deltaVals:
       runningSquared += coOrd**2
   return runningSquared**(1/2)
def findVec(point1,point2,unitSphere = False):
  #setting unitSphere to True will make the vector scaled down to a sphere with a radius one, instead of it's orginal length
  finalVector = [0 for coOrd in point1]
  for dimension, coOrd in enumerate(point1):
      #finding total differnce for that co-ordinate(x,y,z...)
      deltaCoOrd = point2[dimension]-coOrd
      #adding total difference
      finalVector[dimension] = deltaCoOrd
  if unitSphere:
      totalDist = multiDimenDist(point1,point2)
      unitVector =[]
      for dimen in finalVector:
          unitVector.append( dimen/totalDist)
      return unitVector
  else:
      return finalVector


def generate_3d_plot(step_histories, z_data):
    data = []
    print("step histories len: ",len(step_histories))
    for i in range(0, len(step_histories)):
        # Initialise empty lists for markers
        step_history = step_histories[i]
        step_markers_x = []
        step_markers_y = []
        step_markers_z = []
        step_markers_u = []
        step_markers_v = []
        step_markers_w = []
        
        for index, step in enumerate(step_history):
            step_markers_x.append(step.x_index)
            step_markers_y.append(step.y_index)
            step_markers_z.append(step.value)
            
            # If we haven't reached the final step, calculate the vector between the current step and the next step
            if index < len(steps)-1:
                vec1 = [step.x_index,step.y_index,step.value]
                vec2 = [steps[index+1].x_index,steps[index+1].y_index,steps[index+1].value]

                result_vector = findVec(vec1,vec2)
                step_markers_u.append(result_vector[0])
                step_markers_v.append(result_vector[1])
                step_markers_w.append(result_vector[2])
            else:
                step_markers_u.append(0.1)
                step_markers_v.append(0.1)
                step_markers_w.append(0.1)

        # data.append(go.Cone(x=step_markers_x,y=step_markers_y,z=step_markers_z,u=step_markers_u,v=step_markers_v,w=step_markers_w,sizemode="absolute",sizeref=2,anchor='tail'))
        data.append(go.Cone(x=step_markers_x,y=step_markers_y,z=step_markers_z,u=step_markers_u,v=step_markers_v,w=step_markers_w,sizemode="scaled",sizeref=0.05,anchor='tail'))
        data.append(go.Scatter3d(x=step_markers_x, y=step_markers_y, z=step_markers_z, mode='lines', line=dict(color='red',width=2)))

    data.append(go.Surface(z=z_data,opacity=0.5))
    
    # Include cones at each marker to show direction of step, scatter3d is to show the red line between points and surface for the terrain
    fig = go.Figure(data=data)


    # Z axis is limited to the extent of the terrain array
    fig.update_layout(
        title='Gradient ascent Steps',
        scene = dict(zaxis = dict(range=[np.nanmin(z_data),np.nanmax(z_data)],),),)
    return fig
    
# rootdir = "E:\\acquisitions\\LDR_Homogeneous_20210114_192233_150\\" 
rootdir = "E:\\acquisitions\\LDR_Homogeneous_20220309_155612_Paper\\" 
rootdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_2_acquisitions\\coin\\LDR_Homogeneous_20220926_205820_coin\\dmd\\"
lp_file_name = "iteration0_1.lp"


temp_dir = None
if fitter == "dmd":
    temp_dir = rootdir+"dmd"
elif fitter == "ptm":
    temp_dir = rootdir+"ptm"
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)
    os.mkdir(temp_dir+"\\figures")
    

print("Downscaling the images")
start_t = time.time()
rootdir = downscale_acquisition(rootdir,  rootdir + fitter + "\\", lp_file_name=lp_file_name)
print("Done downscaling the images: ", time.time() - start_t)

print("Calculating the Fitter coeffs.")
start_t = time.time()
acquisition = None
if fitter == "dmd":
    acquisition = eng.get_dmd_coeffs(rootdir)
elif fitter == "ptm":
    acquisition = eng.get_ptm_coeffs(rootdir)

print("Done calculating the Fitter coeffs. Time taken: ", time.time() - start_t)



lp_file = rootdir + "acquisition.lp"
figures_dir = rootdir+"figures\\"
slight_positions = []
image_files = []
if os.path.exists(lp_file):    
    print("LP file exists")
    light_positions, image_files = read_lp_file(lp_file)
    


img_file = rootdir+image_files[0]
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
h,w = img.shape
print("Acquisition image size: ", h, w)

images = []
xs = []
ys = []
zs = []
thetas = []
phis = []
error_matrix = np.zeros((h*w, len(light_positions)))


for i in range(0, len(light_positions)):    
    img_file = rootdir + image_files[i]
    acquired_img = np.array(cv2.imread(img_file, cv2.IMREAD_GRAYSCALE))
    
    x, y, z =   light_positions[i][0], light_positions[i][1], light_positions[i][2]
    xs.append(x)
    ys.append(y)
    zs.append(z)
    r, az, el = cart2sph(x, y, z)
    thetas.append(float(az))
    phis.append(float(el))
    
print("Relighting the surfaces " )
start_t = time.time()
relighted_image_mat = None
if fitter=="dmd":
    relighted_img_mat = eng.get_dmd_interpolated_img(matlab.double(thetas), matlab.double(phis))
elif fitter=="ptm":
    relighted_img_mat = eng.get_ptm_interpolated_img(matlab.double(thetas), matlab.double(phis))
relighted_img = np.array(relighted_img_mat._data).reshape(relighted_img_mat.size, order='F')
relighted_img[relighted_img<0] = 0
relighted_img[relighted_img>1] = 1   
relighted_img = 255*relighted_img    
print(relighted_img.shape)
print("Done relighting the surface with ", len(light_positions), " light positions. Time taken: ", time.time()-start_t, "s" )

print("Creating the the error matrix")
start_t = time.time()
for i in range(0, len(light_positions)):
    relighted_image = np.transpose(np.reshape(relighted_img[i, :], (w,h)))
    cv2.imwrite(temp_dir+"\\relighted_img_"+ image_files[i]+".png",relighted_image)  
    relighted_image = np.reshape(relighted_image, (1, h*w))
    acquired_img =  np.reshape(acquired_img, (1, h*w))    
    error_matrix[:,i] = np.abs(np.subtract(relighted_image, acquired_img))
print("Finished updating the error matrices. Time taken: ", time.time()-start_t, "s")

print("Normalization of error matrix and calculating the error vector.")
start_t = time.time()
sum_of_rows = error_matrix.sum(axis=1)
sum_of_rows[sum_of_rows == 0] = 0.1
normalized_error_matrix = error_matrix / sum_of_rows[:, np.newaxis]
error_vector = np.sum(normalized_error_matrix, axis=0)
normalized_error_vector = (error_vector/(h*w))
absolute_error_vector = (np.sum(error_matrix, axis=0))/(h*w)
print("Done normalizing the error vector. Time taken: ", time.time()-start_t, "s")

print("Creating the plots and finding the critical points - global maximum, minimum and local manimas and saddle poitns")
start_t = time.time()
# x = thetas
# y = phis

z = normalized_error_vector
z_absolute = absolute_error_vector
# x1 = np.linspace(min(x), max(x), 50)
# y1 = np.linspace(min(y), max(y), 50)
x1 = np.linspace(-1,1,200)
y1 = np.linspace(-1,1,200)
x2, y2 = np.meshgrid(x1, y1)

z2 = griddata((xs,ys), z, (x2, y2), method='cubic')
z2_absolute = griddata((xs,ys), z_absolute, (x2, y2), method='cubic')
z2[z2<0] = 0
z2_absolute[z2<0] = 0
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.coolwarm,
#     linewidth=0, vmin=0, vmax = np.amax(z2, where=~np.isnan(z2), initial=-1), antialiased=False)
# ax.set_zlim(0, np.amax(z2, where=~np.isnan(z2), initial=-1))
# ax.contour(x2, y2, z2, zdir='z', offset=0, cmap=cm.coolwarm)
# ax.contour(x2, y2, z2, zdir='x', offset=-min(x), cmap=cm.coolwarm)
# ax.contour(x2, y2, z2, zdir='y', offset=-min(y), cmap=cm.coolwarm)
# ax.set_xlabel(r'$l_u$')
# ax.set_ylabel(r'$l_v$')
# ax.set_zlabel(r'$\epsilon$')
# for i in range(0,len(xs)):
#     ax.scatter(xs[i], ys[i], z[i], marker='o')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.title('Loss function')
# plt.savefig(figures_dir+'loss_function.png')

# fig1 = plt.figure()
# ax1 = fig1.gca(projection='3d')
# for i in range(0,len(xs)):
#     ax1.scatter(xs[i], ys[i], z2_absolute[i], marker='o')
# surf2 = ax1.plot_surface(x2, y2, z2_absolute, rstride=1, cstride=1, cmap=cm.coolwarm,
#     linewidth=0, vmin=0, vmax = np.amax(z2_absolute, where=~np.isnan(z2_absolute), initial=-1), antialiased=False)
# ax1.set_zlim(0, np.amax(z2_absolute, where=~np.isnan(z2_absolute), initial=-1))
# ax1.set_xlabel(r'$l_u$')
# ax1.set_ylabel(r'$l_v$')
# ax1.set_zlabel(r'$\epsilon$')
# fig1.colorbar(surf2, shrink=0.5, aspect=5)
# plt.title('Absolute loss')
# plt.savefig(figures_dir+'absolute_loss_function.png')

go_fig_loss_function = go.Figure(data=[go.Surface(z=z2, y=y2,x=x2, opacity=0.5), go.Scatter3d(x=xs, y=ys, z=z, mode="markers")])
mean_loss = np.nanmean(z2)
# go_fig_loss_function.update_layout(title='Loss function, mean absolute loss =  '+str(mean_loss), autosize=False,
#                 width=500, height=500,
#                 margin=dict(l=65, r=50, b=65, t=90,xaxis = dict(title='lu'), xaxis = dict(title='lv')))
go_fig_loss_function.update_layout(title='Loss function, mean loss =  '+str(mean_loss), autosize=True, xaxis_title ='lu', yaxis_title = 'lv')

# fig.show()
go_fig_loss_function.write_html(figures_dir+"loss_function.html")


go_fig_absolute_loss = go.Figure(data=[go.Surface(z=z2_absolute, y=y2,x=x2, opacity=0.5), go.Scatter3d(x=xs, y=ys, z=z_absolute, mode="markers")])
mean_absolute_loss = np.nanmean(z2_absolute)
go_fig_absolute_loss.update_layout(title='Loss function, mean absolute loss =  '+str(mean_absolute_loss), xaxis_title ='lu', yaxis_title = 'lv')

# fig.show()
go_fig_absolute_loss.write_html(figures_dir+"absolute_loss.html")

fig2 = plt.figure()
# plt.contour(x2, y2, z2, levels = np.logspace(-2,3,100))
plt.contourf(x2, y2, z2, 20)
plt.contour(x2, y2, z2, levels = np.logspace(-2,3,100), colors = 'k', linewidths = 1, linestyles = 'solid')
# fig4 = plt.figure()
gradx, grady = np.gradient(z2)

# savetxt('z2.csv', z2, delimiter=',')
plt.quiver(x2, y2, gradx , grady)


local_maxima_indices_x =  argrelextrema(z2, np.greater, axis=1)
local_maxima_indices_y = argrelextrema(z2, np.greater, axis=0)

# print("e")
# print(local_maxima_indices_x)
# print(local_maxima_indices_y)

plt.xlabel('$\\phi$'); plt.ylabel("$\\theta$")
plt.title("Contour plot of loss function for different values of $\\theta$ and $\\phi$s");

plt.savefig(figures_dir+'loss_function_contour_plot.png')

fig3 = plt.figure()
plt.imshow(z2,origin='lower',cmap='terrain')


critical_points = []

indices = np.where(z2 == np.nanmax(z2))
(max_z_x_location, max_z_y_location) = (indices[0][0],indices[1][0])
critical_points.append((max_z_x_location, max_z_y_location, np.nanmax(z2)))
print("maximum location: ", (max_z_x_location, max_z_y_location), "max value: ", z2[max_z_x_location][max_z_y_location], ", ", np.nanmax(z2))
plt.plot(max_z_x_location,max_z_y_location,marker='x')

# Find minimum value index in numpy array
indices = np.where(z2 == np.nanmin(z2))
min_z_x_location, min_z_y_location = (indices[0][0],indices[1][0])
plt.plot(min_z_x_location,min_z_y_location,marker='D', markersize=20)
print("minmum location: ", (min_z_x_location, min_z_y_location), "min value: ", z2[min_z_x_location][min_z_y_location])

global_minimum = np.nanmin(z2)


for i in range(0, len(local_maxima_indices_y[0])):
    # plt.plot(local_maxima_indices_y[0][i], local_maxima_indices_y[1][i], marker="o", markersize=15)
    x_index = local_maxima_indices_y[0][i]
    y_index = local_maxima_indices_y[1][i]
    critical_points.append((x_index, y_index ,z2[x_index][y_index] ))

for i in range(0, len(local_maxima_indices_x[0])):
    # plt.plot(local_maxima_indices_x[0][i], local_maxima_indices_x[1][i], marker="o", markersize=15)
    x_index = local_maxima_indices_x[0][i]
    y_index = local_maxima_indices_x[1][i]
    critical_points.append((x_index, y_index ,z2[x_index][y_index] ))

print("len of critical points: ", len(critical_points))

def sort_key(list_arr):
    return list_arr[2]

critical_points.sort(key=sort_key, reverse=True)

critical_points = critical_points[0:min(len(critical_points), 15)]

for i in range(0, len(critical_points)):
    plt.plot(critical_points[i][0], critical_points[i][1], marker="o", markersize=15)

print("Finished plotting and spotting critical points. Time taken: ", time.time()-start_t, "s")

plt.savefig(figures_dir+'critical_points.png')

def sliding_window(arr, window_size):
    """ Construct a sliding window view of the array"""
    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)

def cell_neighbours(arr, i, j, d):
    """Return d-th neighbors of cell (i, j)"""
    w = sliding_window(arr, 2*d+1)

    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d, 0, w.shape[1]-1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1,j0:j1].ravel()

class ascent_step:
    def __init__(self, value, x_index, y_index):
        self.value= value
        self.x_index= x_index
        self.y_index= y_index


def gradient_ascent_3d(array,x_start,y_start,steps=50,step_size=1, filter_elements = [], plot=False):

    temp_array = copy.deepcopy(array)
    for i in range(0,len(filter_elements)):
        temp_array[temp_array==filter_elements[i][2]] = 0.0

    success = True
    # Initial point to start gradient ascent at
    step = ascent_step(temp_array[y_start][x_start],x_start,y_start)
    
    # Store each step taken in gradient ascent in a list
    step_history = []
    step_history.append(step)
    
    # Plot 2D representation of temp_array with startng point as a red marker
    if plot:
        plt.imshow(temp_array,origin='lower',cmap='terrain')
        plt.plot(x_start,y_start,'ro')
    current_x = x_start
    current_y = y_start
    next_step = None
    # Loop through specified number of steps of gradient ascent to take
    for i in range(steps):
        prev_x = current_x
        prev_y = current_y
        
        # Extract temp_array of neighbouring cells around current step location with size nominated
        neighbours=cell_neighbours(temp_array,current_y,current_x,step_size)
        if len(neighbours) == 0:
            print("Failed: neighbours len is 0")
            success = False
            break
        # Locate minimum in temp_array (steepest slope from current point)
        next_step = np.nanmax(neighbours)
        if np.isnan(next_step):
            print("Failed: Next step is nan")
            success = False
            break
        indices = np.where(temp_array == next_step)

        # Update current point to now be the next point after stepping
        current_x, current_y = (indices[1][0],indices[0][0])
        step = ascent_step(temp_array[current_y][current_x],current_x,current_y)
        
        step_history.append(step)
        
        # Plot each step taken as a black line to the current point nominated by a red marker
        if plot:
            plt.plot([prev_x,current_x],[prev_y,current_y],'k-')
            plt.plot(current_x,current_y,'ro')
            
        # If step is to the same location as previously, this infers convergence and end loop
        if prev_y == current_y and prev_x == current_x:
            # print(f"Converged in {i} steps")
            break
    return next_step,step_history, success


step_size = 0
found_minimum = 99999
found_maximum = 0

print("Performing the gradient ascent from the identified ciritical points to the maxima.")
start_t = time.time()

critical_points_steps = []

step_histories = []
already_searched_maximas = []
# for i in range(0,len(critical_points)):
#     start_x = min_z_x_location
#     start_y = min_z_y_location

#     end_x = critical_points[i][0]
#     end_y = critical_points[i][1]

#     critical_value = critical_points[i][2]

#     success = False

#     while found_maximum != critical_value:
#         step_size += 1
#         found_maximum, steps, success = gradient_ascent_3d(z2, start_y, start_x, steps=50, step_size=step_size, filter_elements = already_searched_maximas, plot=False)
#         if not success:
#             break

#     if success:
#         found_maximum,steps, success = gradient_ascent_3d(z2,start_y,start_x, steps=50, step_size=step_size,filter_elements = already_searched_maximas, plot=True)
#         step_histories.append(steps)

#         for j in range(0, len(steps)):
#             if steps[j].value == global_minimum:
#                 continue
#             else:
#                 critical_points_steps.append((steps[j].y_index, steps[j].x_index, steps[j].value))

#     already_searched_maximas = copy.deepcopy(critical_points[0:i])
# print("len of step histories: ", len(step_histories))

# go_fig = generate_3d_plot(step_histories, z2)
# go_fig.write_html(rootdir+"figures\\gradient_ascent.html")

start_x = min_z_x_location
start_y = min_z_y_location

global_maximum = np.nanmax(z2)
success = False
while found_maximum != global_maximum:
    step_size += 1
    found_maximum,steps, success = gradient_ascent_3d(z2,start_y,start_x, steps=50, step_size=step_size,filter_elements = already_searched_maximas, plot=True)
    if not success:
        break

if success:
        found_maximum,steps, success = gradient_ascent_3d(z2,start_y,start_x, steps=50, step_size=step_size,filter_elements = already_searched_maximas, plot=True)
        step_histories.append(steps)
        for j in range(0, len(steps)):
            if steps[j].value == global_minimum:
                continue
            else:
                critical_points_steps.append((steps[j].y_index, steps[j].x_index, steps[j].value))


go_fig = generate_3d_plot(step_histories, z2)
go_fig.write_html(rootdir+"figures\\gradient_ascent.html")


for i in range(0, len(critical_points_steps)):
    critical_points.append(critical_points_steps[i])


fig4 = plt.figure()
plt.imshow(z2,origin='lower',cmap='terrain')

next_lps = []

x_criticals = []
y_criticals = []
repeating_indices = []
new_point_indices = []
for i in range(0,len(critical_points)):
    x_criticals.append(x2[critical_points[i][0]][critical_points[i][1]])
    y_criticals.append(y2[critical_points[i][0]][critical_points[i][1]])

for i in range(0, len(critical_points)):
    if critical_points[i][2] == global_minimum:
        continue
    else:
        x_diff = abs(np.array(xs)-x_criticals[i])
        y_diff = abs(np.array(ys)-y_criticals[i])
        new_point = True
        for j in range(0,len(xs)):
            if abs(x_diff[j])<0.005 and abs(y_diff[j])<0.005:
                new_point = False
                break

        if new_point:
            new_point_indices.append(i)

        x_diff = abs(x_criticals-x_criticals[i])
        y_diff = abs(y_criticals-y_criticals[i])

        for j in range(i,len(critical_points)):
            if abs(x_diff[j])<0.008 and abs(y_diff[j])<0.008 and j != i and j not in repeating_indices:
                repeating_indices.append(j)


        # if new_point:
        
            #     u = y_criticals[i]
            #     v = x_criticals[i]
            #     az = np.arcsin(math.sqrt((u*u) + (v*v)))
            #     el = np.arctan(v/u)
            #     a,b,c = sph2cart(az,el,1.0)
            #     r,theta,phi = cart2sph(-c,-b,a)
            #     next_lps.append((theta,phi))
            #     plt.plot(critical_points[i][0], critical_points[i][1], marker="o", markersize=15)
print("new point indices: ", new_point_indices)
print("repeating indices: ", repeating_indices)

for i in range(0, len(critical_points)):
    if i in new_point_indices and i not in repeating_indices:
        u = y_criticals[i]
        v = x_criticals[i]
        az = np.arcsin(math.sqrt((u*u) + (v*v)))
        el = np.arctan(v/u)
        a,b,c = sph2cart(az,el,1.0)
        r,theta,phi = cart2sph(-c,-b,a)
        if math.tan(phi)<3.73205080757:
            next_lps.append((theta,phi))
            plt.plot(critical_points[i][0], critical_points[i][1], marker="o", markersize=15)



next_lps = np.array(next_lps)
next_lps = list(set(tuple(p) for p in next_lps))

data = str(len(next_lps))

for i in range(0, len(next_lps)):
    x,y,z = sph2cart(next_lps[i][0], next_lps[i][1], 1)
    data = data+"\n"+str(i)+".png\t"+str(x)+"\t"+str(y)+"\t"+str(z)
    with open(rootdir+"next_iteration.lp", 'w') as f:
        f.write(data)

plt.title('Next best light positions')
print("Finished performing the gradient ascent in: ", time.time()-start_t)

plt.savefig(figures_dir+'result.png')

plt.show()