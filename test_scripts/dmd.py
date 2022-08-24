import cv2
import random 
import numpy as np
import os
import glob 
import matlab.engine
eng = matlab.engine.start_matlab()

eng.addpath("C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\codes\\matlab\\yuly\\pkg_py")
eng.addpath("C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\codes\\matlab\\yuly\\pkg_fcns")
eng.addpath("C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\codes\\matlab\\yuly\\pkg_DMD")
eng.addpath("C:\\Users\\Ramamoorthy_Luxman\\OneDrive - Université de Bourgogne\\imvia\\work\\codes\\matlab\\yuly\\pkg_DMD\\DMD_basis")

rootdir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_2_acquisitions\\brushed_metal\\simple_brushed_metal\\"

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

images = []
img_file = rootdir+"nblp_iteration_0_1_theta_54.61_phi_0.57.png"
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
h,w = img.shape
x = []
y = []
z = []
lps = []
images = []
    
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
        # normalize_factor = 255 * np.ones(np.array(img).shape)
        # normalised_image = img/255
        images.append(img)
        x.append(light_positions[i][0])
        y.append(light_positions[i][1])
        z.append(light_positions[i][2])
        print("*****")

images_mat = np.array(images)
images_mat = np.reshape(images_mat, (len(x), h*w)) 
matlab_img_mat = matlab.uint8(images_mat.tolist())
print(np.array(matlab_img_mat).shape)
print("*****")

            
nb_modes = 45
lps.append(x)
lps.append(y)
lps.append(z)
lps = np.array(lps)
lps = np.reshape(lps, [len(x), 3])
lps = matlab.double(lps.tolist())
print("############")
modal_basis, normal_elt, Up, dmd_coeffs = eng.get_dmd_coeffs(matlab_img_mat, lps, nb_modes, nargout=4)

relighted_img = eng.get_interpolated_img(lps[10], modal_basis,  nb_modes, normal_elt, Up, dmd_coeffs, h, w)

test_dir = lps[2]



print("???????")