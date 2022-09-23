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


# gradient descent 
# https://jackmckew.dev/3d-gradient-descent-in-python.html


fitter = "dmd"


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
            acquired_img = np.array(cv2.imread(img_file, cv2.IMREAD_GRAYSCALE))
            if acquired_img.shape[0]>2000:
                ratio = 400/acquired_img.shape[0]
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

# rootdir = "E:\\acquisitions\\LDR_Homogeneous_20210114_192233_150\\" 
rootdir = "E:\\acquisitions\\LDR_Homogeneous_20220309_155612_Paper\\" 
rootdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_2_acquisitions\\rust_coarse\\dmd\\"
lp_file_name = "iteration_0_1_2_3_4_5_6.lp"


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
dmd_figures_dir = rootdir+"figures\\"
ptm_figures_dir = rootdir+"figures\\"
light_positions = []
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
    error_matrix[:,i] = np.absolute(np.subtract(relighted_image, acquired_img))
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
x1 = np.linspace(-1,1,50)
y1 = np.linspace(-1,1,50)
x2, y2 = np.meshgrid(x1, y1)

z2 = griddata((xs,ys), z, (x2, y2), method='cubic')
z2_absolute = griddata((xs,ys), z_absolute, (x2, y2), method='cubic')
z2[z2<0] = 0
z2_absolute[z2<0] = 0
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, vmin=0, vmax = np.amax(z2, where=~np.isnan(z2), initial=-1), antialiased=False)
ax.set_zlim(0, np.amax(z2, where=~np.isnan(z2), initial=-1))
# ax.contour(x2, y2, z2, zdir='z', offset=0, cmap=cm.coolwarm)
# ax.contour(x2, y2, z2, zdir='x', offset=-min(x), cmap=cm.coolwarm)
# ax.contour(x2, y2, z2, zdir='y', offset=-min(y), cmap=cm.coolwarm)
ax.set_xlabel(r'$l_u$')
ax.set_ylabel(r'$l_v$')
ax.set_zlabel(r'$\epsilon$')
for i in range(0,len(xs)):
    ax.scatter(xs[i], ys[i], zs[i], marker='o')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Loss function')
plt.savefig(dmd_figures_dir+'loss_function.png')

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
surf2 = ax1.plot_surface(x2, y2, z2_absolute, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, vmin=0, vmax = np.amax(z2_absolute, where=~np.isnan(z2_absolute), initial=-1), antialiased=False)
ax1.set_zlim(0, np.amax(z2_absolute, where=~np.isnan(z2_absolute), initial=-1))
ax1.set_xlabel(r'$l_u$')
ax1.set_ylabel(r'$l_v$')
ax1.set_zlabel(r'$\epsilon$')
fig1.colorbar(surf2, shrink=0.5, aspect=5)
plt.title('Absolute loss')
plt.savefig(dmd_figures_dir+'absolute_loss_function.png')

fig2 = plt.figure()
# plt.contour(x2, y2, z2, levels = np.logspace(-2,3,100))
plt.contourf(x2, y2, z2, 20)
plt.contour(x2, y2, z2, levels = np.logspace(-2,3,100), colors = 'k', linewidths = 1, linestyles = 'solid')
# fig4 = plt.figure()
gradx, grady = np.gradient(z2)

# savetxt('z2.csv', z2, delimiter=',')
plt.quiver(x2, y2, gradx , grady)


grad_x_zero_indices = np.where(np.abs(gradx)==0.0)
grad_y_zero_indices = np.where(np.abs(grady)==0.0)

grad_x_zero_indices = np.array([grad_x_zero_indices[0],grad_x_zero_indices[1]])
grad_y_zero_indices = np.array([grad_y_zero_indices[0],grad_y_zero_indices[1]])

grad_gradxx, grad_gradxy = np.gradient(gradx)
grad_gradyy, grad_gradyx = np.gradient(grady)

grad_grad_x_above_zero_indices = np.where(grad_gradxx<0.0)
grad_grad_y_above_zero_indices = np.where(grad_gradyy<0.0)

grad_grad_x_above_zero_indices = np.array([grad_grad_x_above_zero_indices[0], grad_grad_x_above_zero_indices[1]])
grad_grad_y_above_zero_indices = np.array([grad_grad_y_above_zero_indices[0], grad_grad_y_above_zero_indices[1]])

local_maxima_indices_x = list(set(zip(*grad_x_zero_indices)) & set(zip(*grad_grad_x_above_zero_indices)))
local_maxima_indices_y= list(set(zip(*grad_y_zero_indices)) & set(zip(*grad_grad_y_above_zero_indices)))

grad_grad_x_zero_indices = np.where(np.abs(grad_gradxx)==0.00)
grad_grad_y_zero_indices = np.where(np.abs(grad_gradyy)==0.00)

grad_grad_x_zero_indices = np.array([grad_grad_x_zero_indices[0], grad_grad_x_zero_indices[1]])
grad_grad_y_zero_indices = np.array([grad_grad_y_zero_indices[0], grad_grad_y_zero_indices[1]])

saddle_points_x = list(set(zip(*grad_x_zero_indices)) & set(zip(*grad_grad_x_zero_indices)))
saddle_points_y = list(set(zip(*grad_y_zero_indices)) & set(zip(*grad_grad_y_zero_indices)))



plt.xlabel('$\\phi$'); plt.ylabel("$\\theta$")
plt.title("Contour plot of loss function for different values of $\\theta$ and $\\phi$s");

plt.savefig(dmd_figures_dir+'loss_function_contour_plot.png')

fig3 = plt.figure()
plt.imshow(z2,origin='lower',cmap='terrain')


critical_points = []

indices = np.where(z2 == np.nanmax(z2))
(max_z_x_location, max_z_y_location) = (indices[0][0],indices[1][0])
critical_points.append((max_z_x_location, max_z_y_location))
print("maximum location: ", (max_z_x_location, max_z_y_location), "max value: ", z2[max_z_x_location][max_z_y_location])
plt.plot(max_z_x_location,max_z_y_location,marker='x')

# Find minimum value index in numpy array
indices = np.where(z2 == np.nanmin(z2))
min_z_x_location, min_z_y_location = (indices[0][0],indices[1][0])
plt.plot(min_z_x_location,min_z_y_location,marker='D', markersize=20)
print("minmum location: ", (min_z_x_location, min_z_y_location), "max value: ", z2[min_z_x_location][min_z_y_location])

global_minimum = z2[min_z_x_location, min_z_y_location]



for i in range(0, len(saddle_points_y)):
    plt.plot(saddle_points_y[i][0], saddle_points_y[i][1], marker="v", markersize=15)
    # critical_points.append((saddle_points_y[i][0], saddle_points_y[i][1]))

for i in range(0, len(local_maxima_indices_y)):
    plt.plot(local_maxima_indices_y[i][0], local_maxima_indices_y[i][1], marker="o", markersize=15)
    critical_points.append((local_maxima_indices_y[i][0], local_maxima_indices_y[i][1]))

for i in range(0, len(saddle_points_x)):
    plt.plot(saddle_points_x[i][0], saddle_points_x[i][1], marker="v", markersize=15)
    # critical_points.append((saddle_points_x[i][0], saddle_points_x[i][1]))

for i in range(0, len(local_maxima_indices_x)):
    plt.plot(local_maxima_indices_x[i][0], local_maxima_indices_x[i][1], marker="o", markersize=15)
    critical_points.append((local_maxima_indices_x[i][0], local_maxima_indices_x[i][1]))


print("Finished plotting and spotting critical points. Time taken: ", time.time()-start_t, "s")

plt.savefig(dmd_figures_dir+'critical_points.png')

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

class descent_step:
    def __init__(self, value, x_index, y_index):
        self.value= value
        self.x_index= x_index
        self.y_index= y_index

def gradient_ascent_3d(array,x_start,y_start,steps=50,step_size=1,plot=False):
    success = True
    # Initial point to start gradient descent at
    step = descent_step(array[y_start][x_start],x_start,y_start)
    
    # Store each step taken in gradient descent in a list
    step_history = []
    step_history.append(step)
    
    # Plot 2D representation of array with startng point as a red marker
    if plot:
        plt.imshow(array,origin='lower',cmap='terrain')
        plt.plot(x_start,y_start,'ro')
    current_x = x_start
    current_y = y_start

    # Loop through specified number of steps of gradient descent to take
    for i in range(steps):
        prev_x = current_x
        prev_y = current_y
        
        # Extract array of neighbouring cells around current step location with size nominated
        neighbours=cell_neighbours(array,current_y,current_x,step_size)
        
        # Locate minimum in array (steepest slope from current point)
        next_step = np.nanmax(neighbours)
        if np.isnan(next_step):
            success = False
            break
        indices = np.where(array == next_step)

        
        
        # Update current point to now be the next point after stepping
        current_x, current_y = (indices[1][0],indices[0][0])
        step = descent_step(array[current_y][current_x],current_x,current_y)
        
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

print("critical points: ", critical_points)

for i in range(0,len(critical_points)):
    start_x = min_z_x_location
    start_y = min_z_y_location
    
    end_x = critical_points[i][0]
    end_y = critical_points[i][1]

    critical_value = z2[end_x][end_y]

    print(critical_value)

    sucess = None

    while found_maximum != critical_value:
        step_size += 1
        found_maximum, steps, success = gradient_ascent_3d(z2, start_x, start_y, step_size=step_size, plot=False)
        if not success:
            break

    if success:
        found_maximum,steps, success = gradient_ascent_3d(z2,start_x,start_y,steps=100, step_size=step_size,plot=True)
    
        for j in range(0, len(steps)):
            if steps[j].value == global_minimum:
                continue
            else:
                critical_points_steps.append((steps[j].x_index, steps[j].y_index))


# start_x = min_z_x_location
# start_y = min_z_y_location

# global_maximum = np.nanmax(z2)

# while found_maximum != global_maximum:
#     step_size += 1
#     found_maximum, steps = gradient_ascent_3d(z2, start_x, start_y, step_size=step_size, plot=False)

# found_maximum, steps = gradient_ascent_3d(z2, start_x, start_y, step_size=step_size, plot=True)


for i in range(0, len(critical_points_steps)):
    critical_points.append(critical_points_steps[i])


fig4 = plt.figure()
plt.imshow(z2,origin='lower',cmap='terrain')

next_lps = []

for i in range(0, len(critical_points)):
    if z2[critical_points[i][0]][critical_points[i][1]] == global_minimum:
        continue
    else:
        x = x2[critical_points[i][0]][critical_points[i][1]]
        y = y2[critical_points[i][0]][critical_points[i][1]]
        x_diff = abs(np.array(xs)-x)
        y_diff = abs(np.array(ys)-y)
        new_point = True
        for j in range(0,len(xs)):
            if abs(x_diff[j])<0.01 and abs(y_diff[j])<0.01:
                new_point = False
                break
        if new_point:
            u = y
            v = x
            az = np.arcsin(math.sqrt((u*u) + (v*v)))
            el = np.arctan(v/u)
            a,b,c = sph2cart(az,el,1.0)
            r,theta,phi = cart2sph(-c,-b,a)
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


print("Finished performing the gradient ascent in: ", time.time()-start_t)

plt.savefig(dmd_figures_dir+'result.png')

plt.show()