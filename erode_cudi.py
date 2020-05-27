import pickle
import tools.data_manager as data_manager
import tools.image_utils as img
import cv2
import tools.segmentation_kmeans as segmentator
import tools.methods_circle_detection as methods
import numpy as np
from scipy import ndimage as ndi


import matplotlib.pyplot as plt

reload_images = True
if reload_images:
    # load, crop and normalize our images and store them
    # into a dictionary {patient_label:[array_pre, array_post]}
    print("load data from .png files")
    dict_data = data_manager.read_pictures()
    with open("dict_data.bin", "wb") as bin_file:
        pickle.dump(dict_data, bin_file)
else:
    print("load data from dictionary")
    with open("dict_data.bin", "rb") as bin_file:
        dict_data = pickle.load(bin_file)
      
calculate_binaries = True
if calculate_binaries:
    list_all_preprocessed_binaries = []
    for index_patient, patient in enumerate(dict_data):
        # pick and convert image
        image = dict_data[patient][1]
        image = image.astype("uint8")
        # blur image
        image_blurred = cv2.medianBlur(image, 29)
        # segment image using k-means segmentation
        image_segmented = segmentator.run_kmean_on_single_image(image_blurred, k=10,
                                                                precision=10000, max_iterations=1000)
        # find lower threshold for binarizing images
        """ the idea i had here was that all the electrodes always occupy the same area on each picture.
            this function basically returns the pixel value, at which we need to threshold in our binary
            function, so that all pixels that have a higher intensity will collectively make up at least 
            "fraction_of_image_threshold" percent of the picture - electrodes seem to take up about 5-10% of each
             image"""
        lower_threshold = img.intelligent_get_threshold(image_segmented,
                                                        fraction_of_image_threshold=0.08)
        # binarize image
        image_binary = methods.binarize_image(image_segmented, 
                                              lower_threshold=lower_threshold, upper_threshold=255)
        list_all_preprocessed_binaries.append(image_binary)


# crop the correct border by checking if left most or right most collum has a value in it.
# if it does => remove some from this side.
lst_cropped_binary = []
replacement_columns = np.zeros((723,250),dtype=int)
for i in list_all_preprocessed_binaries:
    if sum(i[:,0]) != 0: #if spiral starts left side remove some and add empty space
        new_binary = i[:,250:]
        new_binary = np.append(replacement_columns,new_binary,axis=1)
        lst_cropped_binary.append(new_binary.astype("uint8"))
        plt.imshow(new_binary)
        plt.show()
    if sum(i[:,0]) == 0:
        new_binary = i[:,:(1129-250)]
        new_binary = np.append(new_binary,replacement_columns,axis=1)
        lst_cropped_binary.append(new_binary.astype("uint8"))
        plt.imshow(new_binary)
        plt.show()

###################################################################
def area_of_contour(binary_image):
    area = cv2.contourArea(binary_image[0])
    return area

def find_contours(image):
       _, contours,_ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       return contours

def area_of_each_contour(contours):
    contour_counter = 0
    super_lst =[] #super_lst = [[AREA,IMG],[AREA,IMG]...]
    for i in contours:
        black_img = np.zeros((723,1129))
        black_img = cv2.fillPoly(black_img, pts =[i], color = (255)).astype("uint8")
        single_contour = find_contours(black_img)
        area = area_of_contour(single_contour)
        super_lst.append([area,black_img])
        contour_counter += 1
    return super_lst

def erode_until_split(image_of_largest_area_in_contour_dict):
    image = image_of_largest_area_in_contour_dict
    contours = find_contours(image)
    number_of_blops = 0
    counter = 0
    
    while number_of_blops<2:
        counter += 1
        image = ndi.binary_erosion(image,iterations=1).astype("uint8")
        contours = find_contours(image)
        number_of_blops = len(contours)
        if counter == 20:
            break
    #produce images from the two blops
    another_lst=[]
    for i in contours:
        black_img = np.zeros((723,1129))
        black_img = cv2.fillPoly(black_img, pts =[i], color = (255)).astype("uint8")
        single_contour = find_contours(black_img)
        area = area_of_contour(single_contour)
        another_lst.append([area,black_img])
    #calculate areas 
    return another_lst


def individual_erosion(binary_image):
    
    contours_lst = find_contours(binary_image)
    lst =[]
    lst = area_of_each_contour(contours_lst)
    lst =sorted(lst, key=lambda x: x[0],reverse=True)
    largest_area = lst[0]
    while largest_area[0]>2500:
        #remove the largest contour from super_lst as it gets split
        del lst[0]
        another_lst = erode_until_split(largest_area[1])
        
        #merge another_lst into lst
        index = 0
        for j in another_lst:
            if j[0]==0:
                del another_lst[index]
            else:
                lst.append(j)
            index += 1
        print(lst)
        lst = sorted(lst, key=lambda x: x[0],reverse=True)
        largest_area=lst[0]
    #creates the final binary image      
    final_map = np.zeros((723,1129))
    for i in range(len(lst)):
        final_map += lst[i][1]
        
    return final_map,lst


for i in lst_cropped_binary:
    tryhard,lst = individual_erosion(i)
    f, axarr = plt.subplots(2,1)
    axarr[0].imshow(i)
    axarr[1].imshow(tryhard)
