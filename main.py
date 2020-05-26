import cv2

import tools.data_manager as data_manager
import pickle
import tools.image_utils as img
import tools.prototyping_cuba as prototyper
import tools.methods_circle_detection as methods
from matplotlib import pyplot as plt
import tools.segmentation_kmeans as segmentator

""" first we need to decide if we want to recompute our data with different
parameters """

recompute_data = False

""" in this chapter we handle the preprocessing of our images, loading,
cropping and normalizing """
reload_images = False
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
    dict_data_segmented = segmentator.segment_img_data(dict_data, 3, 3)
    with open("dict_data_segmented.bin", "wb") as bin_file:
        pickle.dump(dict_data_segmented, bin_file)
else:
    print("load segmented images from pickle")
    with open("dict_data_segmented.bin", "rb") as bin_file:
        dict_data_segmented = pickle.load(bin_file)

"""for patient in dict_data_segmented:
    img.plot_image_list(dict_data_segmented[patient])"""

""" here we attempt to find circles in our image, once with image and once with it's edge map """
hough_circle_detection = False
if hough_circle_detection or recompute_data:
    # dict_data_cropped = methods.crop_images(dict_data, y0=150, y1=550, x0=600,x1=800)
    dict_of_centres = {}
    for patient in dict_data:
        print(patient)
        image = dict_data[patient][1]
        try:
            circles_image = methods.circles_find(image)
            dict_of_centres[patient] = circles_image
            print(patient, "in try", "number of circles found:", len(circles_image[0]))
        except:
            print(patient, "returned None, ergo no circles found")
            pass

    for i in dict_of_centres:  # use arrowkeys to go through the images
        methods.circles_show(dict_data[i][1], dict_of_centres[i])

""" this approach trys to find the electrodes using their intensities and contours.
maybe we could implement how far the points should be a part maximal and minimal to 
get rid of the wrong center points. some pictures would benefit from an image
erosion such as img 5. but i dont know how to choose these images beforehand"""
find_centers = True
if find_centers or recompute_data:
    result_dict = {}
    from scipy import ndimage as ndi

    list_all_preprocessed = []

    for index_patient, patient in enumerate(dict_data_segmented):

        # pick and convert image
        image = dict_data[patient][1]
        image = image.astype("uint8")

        # blur image
        image_blurred = cv2.medianBlur(image, 29)

        # find local max
        image_max = ndi.maximum_filter(image_blurred, size=10)  # size gives the shape that is taken from the
        # to the filter function.

        # segment image using k-means segmentation
        image_segmented = segmentator.run_kmean_on_single_image(image_blurred, k=10,
                                                                precision=10000, max_iterations=1000)

        # find lower threshold for binarizing images
        """ the idea i had here was that all the electrodes always occupy the same area on each picture.
            this function basically returns the pixel value, at which we need to threshold in our binary
            function, so that all pixels that have a higher intensity will collectively make up at least 
            "fraction_of_image_threshold" percent of the picture - electrodes seem to take up about 5-10% of each
             image"""
        lower_threshold = img.intelligent_get_threshold(image_segmented, fraction_of_image_threshold=0.08)

        # binarize image
        image_binary = methods.binarize_image(image_segmented, lower_threshold=lower_threshold, upper_threshold=255)
        list_all_preprocessed.append(image_binary)  # add to a list to plot it later

        # now find the contours to calculate their centres
        image_binary = cv2.convertScaleAbs(image_binary)  # need to convert to special format...

        # actually find contours
        contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # calculating the centres of each contour
        lst_of_centres = []

        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            coords = [cX, cY]
            lst_of_centres.append(coords)
        print("number of centres found", len(lst_of_centres))
        for i in lst_of_centres:
            x = i[0]
            y = i[1]
            result = cv2.circle(image_binary, (x, y), 5, (0, 0, 255), -1)
        result_dict[patient] = result, lst_of_centres
    img.plot_preprocessed_image(list_all_preprocessed)


for i in result_dict:
    plt.imshow(result_dict[i][0])
    plt.show()