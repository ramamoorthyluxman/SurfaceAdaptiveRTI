import math
import matplotlib.pyplot as plot
import numpy as np

# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
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


# def fibonacci_sphere(samples=15):

#     points = []
#     phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

#     X = []
#     Y = []
#     Z = []
#     phis = []
#     thetas = []
#     i=0
#     k=0
#     for i in range(int(samples/2), samples):
#         z = -1 + (i / float(samples - 1)) * 2  # y goes from 1 to -1
#         radius = math.sqrt(1 - z * z)  # radius at y
#         theta = phi * i  # golden angle increment
#         y = math.cos(theta) * radius
#         x = math.sin(theta) * radius

#         r, az, el = cart2sph(x,y,z)
#         # convert the angles to be in the range 0 to 2 pi
#         az = (az + np.pi) % (2 * np.pi) - np.pi
#         el = (el + np.pi) % (2 * np.pi) - np.pi
        
#         print(az,el)
#         print(i)
#         phis.append(el)
#         thetas.append(az)
#         points.append((x, y, z))            
#         X.append(x)
#         Y.append(y)
#         Z.append(z)

#     plot.figure().add_subplot(111, projection='3d').scatter(X, Y, Z);    
#     plot.show()

#     return points

# fibonacci_sphere(50)

def generate_n_evenly_spaced_hemispherical_points(samples = 50):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    cartesian_points = []
    polar_points = []
    for i in range(int(samples/2), samples):
        z = -1 + (i / float(samples - 1)) * 2  # z goes from 0 to 1
        radius = math.sqrt(1 - z * z)  # radius at z
        theta = phi * i  # golden angle increment
        y = math.cos(theta) * radius
        x = math.sin(theta) * radius
        r, az, el = cart2sph(x,y,z)
        # convert the angles to be in the range 0 to 2 pi
        az = (az + np.pi) % (2 * np.pi) - np.pi
        el = (el + np.pi) % (2 * np.pi) - np.pi
        polar_points.append((az,el))
        cartesian_points.append((x, y, z))            
        
    plot.figure().add_subplot(111, projection='3d').scatter([p[0] for p in cartesian_points], [p[1] for p in cartesian_points], [p[2] for p in cartesian_points]);    
    plot.show()

    return cartesian_points, polar_points


generate_n_evenly_spaced_hemispherical_points(50)



# ############################ Generate a dense cloud of evenly spaced spherical points { ##################################

# # # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

# dense_cloud_num_pts = 15000  
# dense_cloud_indices =np.arange(0, dense_cloud_num_pts, dtype=float) + 0.5

# dense_cloud_phis = np.arccos(1 - 2*dense_cloud_indices/dense_cloud_num_pts)
# dense_cloud_thetas = np.pi * (1 + 5**0.5) * dense_cloud_indices

# # convert the angles to be in the range 0 to 2 pi
# dense_cloud_phis = np.arctan2(np.sin(dense_cloud_phis), np.cos(dense_cloud_phis))
# dense_cloud_thetas = np.arctan2(np.sin(dense_cloud_thetas), np.cos(dense_cloud_thetas))

# for i in range(0,len(dense_cloud_phis)):
#     if dense_cloud_phis[i]<0:
#         dense_cloud_phis[i]=abs(dense_cloud_phis[i]) + 2*(np.pi - abs(dense_cloud_phis[i]))
#     if dense_cloud_thetas[i]<0:
#         dense_cloud_thetas[i]=abs(dense_cloud_thetas[i]) + 2*(np.pi - abs(dense_cloud_thetas[i]))

# # Uncomment the lines below to visualise the generated cloud. 
# # dense_cloud_x, dense_cloud_y, dense_cloud_z = np.cos(dense_cloud_theta) * np.sin(dense_cloud_phi), np.sin(dense_cloud_theta) * np.sin(dense_cloud_phi), np.cos(dense_cloud_phi);
# # plot.figure().add_subplot(111, projection='3d').scatter(dense_cloud_x, dense_cloud_y, dense_cloud_z);
# # plot.show()

# ############################ Generate a dense cloud of evenly spaced spherical points } ##################################

# ############################ Function to generate evenly spaced points between a given range of polar co-ordinates { ##################################
# ## Angles are passed in degrees. 
# def generate_evenly_spaced_points(min_theta, min_phi, max_theta, max_phi, nb):
#     dense_cloud_points = list(zip(dense_cloud_thetas, dense_cloud_phis))
#     print("inputs", min_theta, min_phi, max_theta, max_phi, nb)
#     print(len(dense_cloud_points))
#     filtered_points = [x for x in dense_cloud_points if (x[0]>=np.radians(min_theta) and x[0]<=np.radians(max_theta) and x[1]>=np.radians(min_phi) and x[1]<=np.radians(max_phi))]
#     filtered_points = filtered_points[0:(len(filtered_points)-(len(filtered_points)%nb))]
#     lps_polar = filtered_points[::int(len(filtered_points)/nb)]
#     lps_cartesian = []
#     for i in range(0,len(lps_polar)):
#         x,y,z = sph2cart(lps_polar[i][0], lps_polar[i][1], 1.0)
#         lps_cartesian.append([x,y,z])
#     return lps_polar, lps_cartesian

# ############################ Function to generate evenly spaced points between a given range of polar co-ordinates } ##################################