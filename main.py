import cv2
import tools.data_manager as data_manager
import pickle
import tools.image_utils as img
from matplotlib import pyplot as plt
import numpy as np

""" first we need to decide if we want to recompute our data with different
parameters """

recompute_data = False

""" in this chapter we handle the preprocessing of our images, loading,
cropping and normalizing """
reload_images = True
if reload_images or recompute_data:
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

""" in this chapter i transform the dictionary """
recompute_image_edges = False
if recompute_image_edges or recompute_data:
    # create edge maps of images
    print("create edge maps from images")
    dict_data_edges = img.data_to_edges(dict_data, sigma_pre=40, sigma_post=40)
    with open("dict_data_edges.bin", "wb") as bin_file:
        pickle.dump(dict_data_edges, bin_file)

else:
    print("load edge maps from dictionary")
    with open("dict_data_edges.bin", "rb") as bin_file:
        dict_data_edges = pickle.load(bin_file)

""" Create a segmented image """
recompute_segmented_images = False
if recompute_segmented_images or recompute_data:
    # create segmented images using K-means clustering
    print("recompute Segmented images")
    dict_data_segmented = img.segment_img_data(dict_data, 3, 3)
    with open("dict_data_segmented.bin", "wb") as bin_file:
        pickle.dump(dict_data_segmented, bin_file)
else:
    print("load segmented images from pickle")
    with open("dict_data_segmented.bin", "rb") as bin_file:
        dict_data_segmented = pickle.load(bin_file)


""" this approach trys to find the electrodes using their intensities and contours.
The images get first binarized where each image is thresholded by looking at their 
intensity distribution. After that each binary image is  cropped at the "tail" of 
the cochlea, then the contours are extracted and get eroded depending of their area.
bigger areas get eroded first until the biggest area is too small for further erosion.
Then the center of mass of each contour is calculated and represents a single electrode"""

print("processing images")
lst_binary_preprocessed=img.calculate_binaries(dict_data)
lst_cropped_binaries = img.crop_binaries(lst_binary_preprocessed,270)
lst_individual_erosion = [img.individual_erosion(i) for i in lst_cropped_binaries]
dict_of_electrode_centers = img.get_center_of_electrodes(lst_individual_erosion)
plot_electrode_centers = True
if plot_electrode_centers:
    counter = 1
    for patient in dict_data:
        coords = dict_of_electrode_centers[counter]
        counter += 1
        final_map = np.zeros((723,1129))
        for coordinate in coords:
            x = coordinate[0]
            y = coordinate[1]
            cv2.circle(dict_data[patient][1],(x,y),7,(0,0,255),-1)
        plt.imshow(dict_data[patient][1])
        plt.show()
            
#save some plots at some points during pipeline for the report
# import scipy.misc
# scipy.misc.imsave('afterWatermark.jpg', dict_data["18"][1]) 
# scipy.misc.imsave("afterBinarization.jpg",lst_binary_preprocessed[8])
# scipy.misc.imsave("afterCropping.jpg",lst_cropped_binaries[8])
# scipy.misc.imsave("afterErosion.jpg",lst_individual_erosion[8])
# scipy.misc.imsave('afterFindingElectrodes.jpg', dict_data["18"][1]) 

#cropp more away from the lst_cropped_binaries and after that try to fit circle with houghtransform
# to get the center of the cochlea.
test_lst=img.crop_binaries(lst_binary_preprocessed, 430)
dict_binaries = {}
for patient in enumerate(dict_data):
    dict_binaries[patient[1]]=test_lst[patient[0]]
    
hough_circle_detection = True
if hough_circle_detection:
    spiral_centres = {}
    
    for patient in dict_binaries:
        
        image = dict_binaries[patient].astype("uint8")
        image = cv2.GaussianBlur(image, (5,5), 10)
        try:
            circles_image = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 3, 100000, 
                                                       param1=50, param2=30, minRadius=80, 
                                                       maxRadius=180)
            spiral_centres[patient] = circles_image
            print(patient, "in try", "number of circles found:", len(circles_image[0]))
        except:
            print(patient, "returned None, ergo no circles found")
            pass

    for i in spiral_centres:  # use arrowkeys to go through the images
        img.circles_show(dict_binaries[i], spiral_centres[i])

# merge both dictionaries to have one with both spiral centres and electrode centres for each patient
lst_electrodes = []
for i in dict_of_electrode_centers:
    lst_electrodes.append(dict_of_electrode_centers[i])
dict_results = {}
for i in enumerate(spiral_centres):
    dict_results[i[1]]=[spiral_centres[i[1]],lst_electrodes[i[0]]]
#to new dictionary to work is dict_results={PATIENTID:[SPIRALCENTER,ELECTRODES],...} 
print(dict_results)
save_results = True
if save_results:
    with open("dict_results.bin", "wb") as bin_file:
        pickle.dump(dict_results, bin_file)


