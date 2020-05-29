from matplotlib import pyplot as plt
from skimage import feature, color
import numpy as np
import copy
from scipy.ndimage import morphology as mp
import cv2 # conda install -c menpo opencv
from scipy import ndimage as ndi
def edge_map(image, sigma):
    # Returns the edge map of a given image.
    #
    # Inputs:
    #   img: image of shape (n, m, 3) or (n, m)
    #
    # Outputs:
    #   edges: the edge map of image

    img_grayscale = color.rgb2gray(image)
    img_edges = feature.canny(img_grayscale, sigma)

    return img_edges


def show(image_array):
    plt.imshow(image_array)
    plt.show()


def plot_image_list(list_images):
    # get how many images you want to plot horizontally
    nCols = len(list_images)
    nRows = 1

    # commence plotting
    for index_image, image in enumerate(list_images, 1):  # start at 1 because of shitty pyplot indexing
        plt.subplot(nRows, nCols, index_image)  # open subplot space to plot to
        plt.imshow(image)  # plot the image into subplot space

    plt.show()  # show plotted images


def plot_preprocessed_image(list_images):
    # get how many images you want to plot horizontally
    nCols = 4
    nRows = 3

    # commence plotting
    for index_image, image in enumerate(list_images, 1):  # start at 1 because of shitty pyplot indexing
        plt.subplot(nRows, nCols, index_image)  # open subplot space to plot to
        plt.imshow(image)  # plot the image into subplot space

    plt.show()  # show plotted images


def test_sigmas(data, testing_range=range(10, 15, 1), pre_or_post=0, patient="03"):
    sample_image = data[patient][pre_or_post]

    list_image_edge_maps = []

    for sigma in testing_range:
        edge_map_current_sigma = edge_map(sample_image, sigma)
        list_image_edge_maps.append(edge_map_current_sigma)

    plot_image_list(list_image_edge_maps)


def data_to_edges(dict_data, sigma_pre, sigma_post):
    dict_data_edges = {}

    for patient in dict_data:
        # read images
        image_pre = dict_data[patient][0]
        image_post = dict_data[patient][1]

        # create their respective edge maps
        edge_map_pre = edge_map(image_pre, sigma_pre)
        edge_map_post = edge_map(image_post, sigma_post)

        dict_data_edges[patient] = [edge_map_pre, edge_map_post]  # add to edge dictionary

    return dict_data_edges


def create_material_masks(dict_data):  # old idea, k-means is just kinda better

    # set thresholds for bone/fluid/auas
    t_bone = 150
    t_gas = 40

    for patient in dict_data:

        image_pre = dict_data[patient][0]  # [600:640, 540:600]
        image_post = dict_data[patient][1]

        image_pre_bone = copy.copy(image_pre)
        image_pre_fluid = copy.copy(image_pre)
        image_pre_gas = copy.copy(image_pre)

        for i_rows, row in enumerate(image_pre):
            for i_cols, value in enumerate(row):
                if value >= t_bone:
                    image_pre_bone[i_rows, i_cols] = 1
                    image_pre_fluid[i_rows, i_cols] = 0
                    image_pre_gas[i_rows, i_cols] = 0

                elif value <= t_gas:
                    image_pre_bone[i_rows, i_cols] = 0
                    image_pre_fluid[i_rows, i_cols] = 0
                    image_pre_gas[i_rows, i_cols] = 1

                else:
                    image_pre_bone[i_rows, i_cols] = 0
                    image_pre_fluid[i_rows, i_cols] = 1
                    image_pre_gas[i_rows, i_cols] = 0

        plot_image_list([image_pre_bone, image_pre_fluid, image_pre_gas])

    return None


def equalize_histogram(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


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
            return lower_threshold_for_binarizing


def normalize_image(image):  # from https://en.wikipedia.org/wiki/Normalization_(image_processing)

    max_old = np.max(image)
    max_new = 255

    min_old = np.min(image)
    min_new = 0

    image = ((image - np.min(image)) / ((max_new - min_new) / (max_old - min_old))) + min_new

    return image


def distance_transform_binary(image_binary):  # , sampling_factor
    image_distance_transform = cv2.distanceTransform(image_binary, 2, 3)
    # image_distance_transform = mp.distance_transform_edt(image_binary, sampling=sampling_factor, return_distances=True, return_indices=False)
    return image_distance_transform


def watersheddy(image_bin):
    import sys
    import cv2
    import numpy
    from scipy.ndimage import label

    def segment_on_dt(a, image):
        border = cv2.dilate(image, None, iterations=5)
        border = border - cv2.erode(border, None)

        dt = cv2.distanceTransform(image, 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
        show(dt)
        _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        # Completing the markers now.
        lbl[border == 255] = 255

        lbl = lbl.astype(numpy.int32)

        cv2.watershed(a, lbl)

        lbl[lbl == -1] = 0
        lbl = lbl.astype(numpy.uint8)
        return 255 - lbl

    image_bin = image_bin
    img_bin = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN,
                               numpy.ones((3, 3), dtype=int))

    result = segment_on_dt(image_bin, img_bin)
    cv2.imwrite(sys.argv[2], result)

    result[result != 255] = 0
    result = cv2.dilate(result, None)
    image_bin[result == 255] = (0, 0, 255)
    show(image_bin)


def watershed(image, image_distance_transform):
    import sys
    import cv2
    import numpy
    from scipy.ndimage import label

    border = cv2.dilate(image_distance_transform, None, iterations=5)
    border = border - cv2.erode(border, None)

    plotlist = [image_distance_transform]
    image_distance_transform = ((image_distance_transform - image_distance_transform.min()) / (
                image_distance_transform.max() - image_distance_transform.min()) * 255).astype(numpy.uint8)
    plotlist.append(image_distance_transform)

    image_distance_transform = cv2.threshold(image_distance_transform, 180, 255, cv2.THRESH_BINARY)[1]

    plotlist.append(image_distance_transform)

    # plot_image_list(plotlist)

    lbl, ncc = label(image_distance_transform)
    lbl = lbl * (255 / (ncc + 1))

    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(numpy.int32)



    image_shape = np.shape(image)

    image_converted = np.empty(shape=(image_shape[0], image_shape[1], 3))
    for channel in range(3):
        image_converted[:, :, channel] = image

    print(lbl.dtype)
    print(image.dtype)

    print(np.shape(image_converted))
    cv2.watershed(image, lbl)
    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl

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


def segment_img_data(dict_data, k_pre=3, k_post=3):

    Ks = [k_pre, k_post]

    dict_data_segmented = {}

    for patient in dict_data:

        segmented_images_current_patient = []

        for i in range(2):  # for pre and post image, hence indices [0, 1]

            # get pre image & K-parameter
            image = dict_data[patient][i]
            K = Ks[i]

            segmented_image = run_kmean_on_single_image(image, K)  # generate segmented image

            segmented_images_current_patient.append(segmented_image)   # append segmented image to list

        dict_data_segmented[patient] = segmented_images_current_patient  # add patient images to data dictionary

    return dict_data_segmented

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
    print(lst[0])
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
        print("number of centres found", len(lst_of_centers))
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

def get_center(array_of_center_coords):
    #need to find a way to remove outliers...
    center_coords = array_of_center_coords.reshape((len(array_of_center_coords[0]),len(array_of_center_coords[0][0])))
    center_x = int(np.sum(center_coords[:,0])/len(center_coords[:,0]))
    center_y = int(np.sum(center_coords[:,1])/len(center_coords[:,1]))
    spiral_center = np.array([[[center_x,center_y,10]]])
    return spiral_center

def crop_images(input_dict,y0=0,y1=723,x0=0,x1=1129):
    dictionary = input_dict
    for i in dictionary:
        for j in range(2):
            image = dictionary[i][j]
            image = image[y0:y1,x0:x1]
            dictionary[i][j] = image
    return dictionary

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def circle_cropping(image, center_entry, radius):
    center = center_entry[0][0][0][0:2]
    mask=create_circular_mask(723, 1129, center, radius)
    masked_img = image.copy()
    masked_img[~mask] = 0
    return masked_img
