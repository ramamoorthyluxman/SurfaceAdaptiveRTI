import numpy as np
import glob
import cv2

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

dataset_dir = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\brushed_metal\\circular_brushed_metal\\phi_45_collimated\\" 
lp_file = "D:\\imvia_phd\\data\\nblp_v2\\nblp_acquisition\\brushed_metal\\circular_brushed_metal\\phi_45_collimated\\iteration_0.lp"
light_positions, image_files = read_lp_file(lp_file)

img0 = image_files[0].split('.')[0]    
img_file = glob.glob(dataset_dir+img0+"_*.png")[0]
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
h,w = img.shape
display_img = img.copy()

cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 

coefficient_of_variation_img = np.zeros(img.shape,dtype=np.uint8)

for x in range(0,w):
    for y in range(0,h):
        intensities = []
        for i in range(0, len(light_positions)):
            filename = image_files[i].split('.')[0]    
            img_file = glob.glob(dataset_dir+filename+"_*.png")[0]
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)    
            intensities.append(img[y,x])    
        coefficient_of_variation_img[y,x] = cv(intensities)
cv2.imshow("output", coefficient_of_variation_img)
cv2.waitKey(1)
    
