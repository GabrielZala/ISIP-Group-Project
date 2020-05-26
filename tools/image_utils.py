from matplotlib import pyplot as plt
from skimage import feature, color
import numpy as np
import copy
from scipy.ndimage import morphology as mp
import cv2


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

    # set thresholds for bone/fluid/gas
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
