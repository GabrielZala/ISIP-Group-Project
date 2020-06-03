import numpy as np
import cv2 # conda install -c menpo opencv
from scipy import ndimage as ndi


def intelligent_get_threshold(image, fraction_of_image_threshold=0.06, print_info=False):
    dict_histogram = {}
    for row in image:  # iterate through rows
        for value in row:  # iterate though values in rows
            if value in dict_histogram.keys():  # check if value was already found in histogram dictionary
                dict_histogram[value] += 1  # if so, increment number of pixels with this value
            else:
                dict_histogram[value] = 1  # if not, add value to dict with number of pixels = 1

    total_pixels = len(image.flatten())
    list_keys_sorted = sorted(list(dict_histogram.keys()))

    if print_info:
        print("total pixels", total_pixels)
        print("histogram has", len(dict_histogram.keys()), "bins")
        for value in list_keys_sorted:
            npixels = dict_histogram[value]
            print("intensity:", value, "makes up", round(npixels / total_pixels, 2), "percent of the image")

    list_keys_sorted_inverse = sorted(list_keys_sorted, reverse=True)
    total_fraction_of_pixels = 0
    for index_bin, bin_key in enumerate(list_keys_sorted_inverse):
        percentage_of_pixels_in_bin = dict_histogram[bin_key] / total_pixels
        total_fraction_of_pixels += percentage_of_pixels_in_bin
        if total_fraction_of_pixels >= fraction_of_image_threshold:
            lower_threshold_for_binarizing = list_keys_sorted_inverse[index_bin]
            if print_info:
                print("bin for last image", lower_threshold_for_binarizing)
            
            if lower_threshold_for_binarizing == list_keys_sorted_inverse[0]:
                return lower_threshold_for_binarizing - 1   # this is to prevent image 2 from vanishing
            return lower_threshold_for_binarizing




def run_kmean_on_single_image(image_array, k, precision=10, max_iterations=0.1):

    image_array = np.uint8(image_array)

    # blur image beforehand
    image_array = cv2.medianBlur(image_array, 11)

    # make the image flat
    image_flattened = image_array.flatten()
    # image_flattened = image_array.reshape((-1, 3))

    # convert to np.float32
    image_flat_converted = np.float32(image_flattened)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, precision, max_iterations)

    # run k-means clustering
    ret, label, center = cv2.kmeans(image_flat_converted, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)

    res = center[label.flatten()]  # label is actually already flat, but do it anyways for consistency
    res2 = res.reshape(image_array.shape)
    
    return res2




#########################cudi's stuff####################
# binarize image
def binarize_image(image, lower_threshold, upper_threshold):
    mask = (image > lower_threshold) & (image < upper_threshold)  # sets all values to 0 that are not within
    image = image * mask  # within thresholds
    image[image > 0] = 255  # set all values to 255 that are above 0 (black in the image)
    return image

def calculate_binaries(dict_data):
    """creates binary images out of the grayscale images"""
    list_all_preprocessed_binaries = []
    for index_patient, patient in enumerate(dict_data):
        # pick and convert image
        image = dict_data[patient][1]
        image = image.astype("uint8")
        # blur image
        image_blurred = cv2.medianBlur(image, 29)
        # segment image using k-means segmentation
        image_segmented = run_kmean_on_single_image(image_blurred, k=10,
                                                                precision=10000, max_iterations=1000)
        # find lower threshold for binarizing images
        """ the idea i had here was that all the electrodes always occupy the same area on each picture.
            this function basically returns the pixel value, at which we need to threshold in our binary
            function, so that all pixels that have a higher intensity will collectively make up at least 
            "fraction_of_image_threshold" percent of the picture - electrodes seem to take up about 5-10% of each
             image"""
        lower_threshold = intelligent_get_threshold(image_segmented,
                                                        fraction_of_image_threshold=0.08)
        # binarize image
        image_binary = binarize_image(image_segmented, 
                                              lower_threshold=lower_threshold, upper_threshold=255)
        list_all_preprocessed_binaries.append(image_binary)
    return list_all_preprocessed_binaries

def crop_binaries(list_of_binary_images,how_much_to_remove):
    """this function takes the post binary images and crops them where the spiral
    hits the edge of the image in order to reduce the area of possible electrodes
    returns a list of images all with the previous size of post OP images"""
    lst_cropped_binary = []
    replacement_columns = np.zeros((723,how_much_to_remove),dtype=int)
    for i in list_of_binary_images:
        if sum(i[:,0]) != 0: #if spiral starts left side remove some and add empty space
            new_binary = i[:,how_much_to_remove:]
            new_binary = np.append(replacement_columns,new_binary,axis=1)
            lst_cropped_binary.append(new_binary.astype("uint8"))
        if sum(i[:,0]) == 0:
            new_binary = i[:,:(1129-how_much_to_remove)]
            new_binary = np.append(new_binary,replacement_columns,axis=1)
            lst_cropped_binary.append(new_binary.astype("uint8"))
    return lst_cropped_binary

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
    image = cv2.medianBlur(image, 11)
    image = ndi.binary_erosion(image,iterations=1).astype("uint8")
    contours = find_contours(image)
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
    smallest_area = lst[-1]
    while largest_area[0]>(smallest_area[0]):
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

        lst = sorted(lst, key=lambda x: x[0],reverse=True)
        largest_area=lst[0]
    #creates the final binary image      
    final_map = np.zeros((723,1129))
    for i in range(len(lst)):
        final_map += lst[i][1]
        
    return final_map


def get_center_of_electrodes(lst_of_images):
    result_dict = {}
    counter = 1
    for image in lst_of_images:
        contours = find_contours(image.astype("uint8"))
        lst_of_centers = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            coords = [cX, cY]
            lst_of_centers.append(coords)
        result_dict[counter] = lst_of_centers  
        counter += 1
    return result_dict



def hough_circle(image):
    """
    Hough Circle Transform:
        first number in function: dp=1, resolution factor between image and detection
        second number : minDist = minimum number of pixels between circle centers
        param1: number used for canny edge detection
        param2: threshold for circle detection (small values=more circles will be detected)
        min/maxRadius: only detects circles that are between those values
    """

    # run circle detection algo
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 2, 0, param1=30, param2=100, minRadius=100, maxRadius=200)
    """print(type(circles))  # just some data on what's found
    print(circles)
    print(np.shape(circles))"""

    circles = np.uint16(np.around(circles))  # what the fuck is this
    return circles

def circles_show(image, circles):

    cimg = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_GRAY2BGR)

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_angle(center, electrodes):
  """
  Description: calculates the angel between a center and two points
  Input: center: tuple of x and y
         electrodes: list of two tuples
  Return: phi: angle in degrees as float
  """
  
  ### calculate (euclidean) distances of the triangle
  a = np.linalg.norm(electrodes[0]-electrodes[1])
  b = np.linalg.norm(center-electrodes[0])
  c = np.linalg.norm(center-electrodes[1])
  
  ### apply cosine rule
  phi = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2*b*c)))
  
  return phi
